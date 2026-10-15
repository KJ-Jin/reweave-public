"""Offline motif canonicalization of exported reference trajectories (spec section 17.B)."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from ..handles import Arena, Mono, Sum, cost, deserialize, poly, reachable, to_expr
from ..search import peak_rss_mb
from .. import poly as p
from .reference import STEM, read_trajectories

MOTIFS = 'trajectory_motifs.jsonl'


class Canonizer:
    """Memoize canon by exact polynomial value for one offline analysis process."""

    def __init__(self) -> None:
        self.cache: dict[tuple,tuple] = {}
        self.calls = 0
        self.hits = 0

    def key(self, q: p.Poly) -> tuple:
        """Return the canonical renaming-invariant key of an exact polynomial."""
        assert not p.TIMED_EXHAUSTIVE_ACTIVE, 'offline canonicalization runs outside timed search'
        self.calls += 1
        value = tuple(sorted(q.items()))
        if value in self.cache:
            self.hits += 1
            return self.cache[value]
        self.cache[value] = p.canon(q)
        return self.cache[value]


def motif_key(canonical: tuple) -> str:
    """Hash a canonical polynomial into its stable motif identifier."""
    return hashlib.sha256(repr(canonical).encode()).hexdigest()


def state_motifs(arena: Arena, root: int, canonizer: Canonizer) -> dict[str,dict]:
    """Deduplicate one state's reachable non-Mono handles by canonical polynomial."""
    found: dict[str,dict] = {}
    for h in sorted(reachable(arena,[root])):
        node = arena.handles[h]
        if isinstance(node,Mono):
            continue
        canonical = canonizer.key(poly(arena,h))
        key = motif_key(canonical)
        kind = 'S' if isinstance(node,Sum) else 'P'
        if key in found:
            entry = found[key]
            entry['kinds'] = sorted(set(entry['kinds']+[kind]))
            entry['c_sf'] = min(entry['c_sf'],cost(arena,h))
            continue
        found[key] = dict(canonical_key=key,canonical_terms=[list(pair) for pair in canonical],
                          kinds=[kind],c_sf=cost(arena,h),handle=h)
    return found


def trajectory_motifs(record: dict, canonizer: Canonizer) -> dict:
    """Rebuild one exported trajectory and collect its per-state reachable non-Mono motifs."""
    arena = Arena()
    ids = deserialize(arena,record['nodes'])
    states = [step['state'] for step in record['path']]+[record['path'][-1]['next_state']] if record['path'] else []
    generators = [dict((m,c) for m,c in terms) for terms in record['generator_nodes']]
    target = dict((m,c) for m,c in record['target'])
    motifs: dict[str,dict] = {}
    handles = 0
    for step,index in enumerate(states):
        root = ids[index]
        for key,entry in state_motifs(arena,root,canonizer).items():
            handles += 1
            if key not in motifs:
                value = poly(arena,entry['handle'])
                motifs[key] = dict({k:v for k,v in entry.items() if k != 'handle'},first_seen_step=step,hits=0,
                                   generator_node=any(value == q for q in generators),
                                   is_target=value == target,representative=to_expr(arena,entry['handle']))
            motifs[key]['hits'] += 1
            motifs[key]['kinds'] = sorted(set(motifs[key]['kinds']+entry['kinds']))
            motifs[key]['c_sf'] = min(motifs[key]['c_sf'],entry['c_sf'])
    body = {key:record[key] for key in ('task_id','cell_id','set','family','params','slot_mix','n_vars',
                                        'occurrences','distinct','start_cost','c_gen','shortest_depth')}
    return dict(body,path_c_sf=[step['state_c_sf'] for step in record['path']]+
                ([record['path'][-1]['next_c_sf']] if record['path'] else []),
                states=len(states),non_mono_handles=handles,
                motifs=[motifs[key] for key in sorted(motifs)])


def extract(records: list[dict]) -> tuple[list[dict],dict]:
    """Canonicalize every exported trajectory offline and measure the analysis workload."""
    began = time.perf_counter()
    canonizer = Canonizer()
    blocks = [trajectory_motifs(record,canonizer) for record in records]
    return blocks,dict(trajectories=len(blocks),states=sum(b['states'] for b in blocks),
                       non_mono_handles=sum(b['non_mono_handles'] for b in blocks),
                       unique_polynomials_canonicalized=len(canonizer.cache),canon_calls=canonizer.calls,
                       canon_cache_hits=canonizer.hits,
                       canon_cache_hit_rate=canonizer.hits/canonizer.calls if canonizer.calls else None,
                       distinct_motifs=len({m['canonical_key'] for b in blocks for m in b['motifs']}),
                       wall_seconds=time.perf_counter()-began,peak_rss_mb=peak_rss_mb())


def write_blocks(blocks: list[dict], path: Path) -> Path:
    """Write motif blocks as deterministic JSON lines."""
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as stream:
        for block in blocks:
            stream.write(json.dumps(block,sort_keys=True)+'\n')
    return path


def read_blocks(path: Path) -> list[dict]:
    """Read every motif block in file order."""
    if not path.is_file():
        raise ValueError(f'motif blocks missing: {path}')
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def preflight(records: list[dict], cell_id: int, output: Path) -> dict:
    """Measure the complete post-search motif analysis of one representative cell (section 17.D)."""
    selected = [record for record in records if record['cell_id'] == cell_id]
    if not selected:
        raise ValueError(f'cell {cell_id} exported no accepted reference trajectory')
    blocks,stats = extract(selected)
    report = dict(cell_id=cell_id,set=selected[0]['set'],family=selected[0]['family'],
                  params=selected[0]['params'],slot_mix=selected[0]['slot_mix'],**stats)
    report['seconds_per_trajectory'] = stats['wall_seconds']/stats['trajectories']
    output.mkdir(parents=True,exist_ok=True)
    (output/'motif_resource_preflight.json').write_text(json.dumps(report,indent=1,sort_keys=True)+'\n')
    return report


def preflight_cell(records: list[dict]) -> int:
    """Pick the deterministic representative cell: the most trajectory states, ties by lowest cell id."""
    per: dict[int,int] = {}
    for record in records:
        per[record['cell_id']] = per.get(record['cell_id'],0)+len(record['path'])+1
    return min(per,key=lambda cell:(-per[cell],cell))


def main() -> None:
    """Run one offline motif stage: the single-cell preflight or the full extraction."""
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=('preflight','extract'))
    parser.add_argument('--trajectories',type=Path,default=Path(f'out/pilot/{STEM}.jsonl'))
    parser.add_argument('--output',type=Path,default=Path('out/exploratory'))
    args = parser.parse_args()
    records = read_trajectories(args.trajectories)
    began = time.perf_counter()
    if args.stage == 'preflight':
        print(json.dumps(preflight(records,preflight_cell(records),args.output),indent=1,sort_keys=True))
    else:
        blocks,stats = extract(records)
        write_blocks(blocks,args.output/MOTIFS)
        (args.output/'motif_extraction_stats.json').write_text(json.dumps(stats,indent=1,sort_keys=True)+'\n')
        print(json.dumps(stats,indent=1,sort_keys=True))
    print(f'{args.stage} done in {time.perf_counter()-began:.0f} s',flush=True)


if __name__ == '__main__':
    main()
