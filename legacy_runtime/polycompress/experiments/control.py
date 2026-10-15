"""Independent diagnostic random-DAG calibration set and its motif control (section 17.E/17.F)."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from ..config import GenCfg, RandomOODCfg
from ..handles import Arena, Mono, cost, mk_mono, mk_prod, mk_sum, poly, to_expr
from .. import generator as g
from .. import poly as p
from .motifs import Canonizer, motif_key

NAMESPACE = 'motif_control_random_ood'
MANIFEST = 'exploratory_only.json'


def formal_canons(cfg: dict) -> tuple[set,dict]:
    """Regenerate every formal benchmark set and return its canonical targets and per-set counts."""
    sets,statuses = g.generate_splits(cfg['gen'],cfg['calc'],cfg['pilot'],cfg['compute'],
                                      cfg['random_ood'],cfg['scale'])
    taken,counts = set(),{}
    for name in sorted(sets):
        tasks = g.set_tasks(name,sets[name])
        counts[name] = len(tasks)
        taken |= {p.canon(task.target) for task in tasks}
    return taken,counts


def control_set(cfg: GenCfg, ood: RandomOODCfg, taken: set) -> list:
    """Accept diagnostic hidden-DAG proposals in id order under the independent tagged namespace."""
    result,seen = [],set(taken)
    for j in range(100*ood.n_tasks):
        instance = g._hidden_task(NAMESPACE,NAMESPACE,j,(3,4,5,6),(ood.min_gates,ood.max_gates),cfg)
        if instance is None or g.size_rejection(instance[1],cfg):
            continue
        shape = p.canon(instance[1].target)
        if shape in seen:
            continue
        seen.add(shape)
        instance[1].meta['id'] = len(result)
        result.append(instance)
        if len(result) == ood.n_tasks:
            return result
    raise ValueError(f'{NAMESPACE} accepted only {len(result)} of {ood.n_tasks} tasks')


def hidden_handles(arena: Arena, nodes: list, n_vars: int) -> list[int]:
    """Rebuild a hidden DAG's gates as handles in gate order and return one handle per gate."""
    values = [mk_mono(arena,1 << (4*i)) for i in range(n_vars)]
    for op,a,b in nodes:
        pair = Counter([values[a],values[b]])
        values.append(mk_sum(arena,pair) if op == 'ADD' else mk_prod(arena,pair))
    return values[n_vars:]


def hidden_block(task, canonizer: Canonizer) -> dict:
    """Collect one hidden DAG's non-leaf non-Mono nodes, deduplicated by canonical polynomial."""
    arena = Arena()
    gates = hidden_handles(arena,task.meta['hidden_gates'],task.meta['hidden_leaves'])
    target = poly(arena,gates[-1])
    motifs: dict[str,dict] = {}
    for h in gates:
        if isinstance(arena.handles[h],Mono):
            continue
        canonical = canonizer.key(poly(arena,h))
        key = motif_key(canonical)
        if key in motifs:
            motifs[key]['hits'] += 1
            motifs[key]['c_sf'] = min(motifs[key]['c_sf'],cost(arena,h))
            continue
        motifs[key] = dict(canonical_key=key,canonical_terms=[list(pair) for pair in canonical],
                           kinds=['S' if arena.handles[h].__class__.__name__ == 'Sum' else 'P'],
                           c_sf=cost(arena,h),first_seen_step=1,hits=1,generator_node=True,
                           is_target=poly(arena,h) == target,representative=to_expr(arena,h))
    return dict(task_id=task.meta['id'],cell_id=-1,set=NAMESPACE,family=NAMESPACE,params=[],slot_mix=[],
                n_vars=task.n_vars,occurrences=task.meta['occurrences'],distinct=task.meta['distinct'],
                start_cost=task.meta['start_cost'],c_gen=task.c_gen,shortest_depth=task.c_gen,
                path_c_sf=[task.meta['start_cost'],task.c_gen],states=1,
                non_mono_handles=sum(1 for h in gates if not isinstance(arena.handles[h],Mono)),
                motifs=[motifs[key] for key in sorted(motifs)])


def manifest(instances: list, counts: dict, canonizer: Canonizer) -> dict:
    """Record every diagnostic-only task with the identity needed for the Gate-A exclusion audit."""
    rows = []
    for arena,task,start in instances:
        canonical = canonizer.key(task.target)
        rows.append(dict(diagnostic_task_id=task.meta['id'],proposal=task.meta['proposal'],
                         generator_namespace=f"{NAMESPACE}|{task.meta['proposal']}",
                         target_sha256=hashlib.sha256(repr(sorted(task.target.items())).encode()).hexdigest(),
                         canonical_key_sha256=motif_key(canonical),
                         canonical_terms=[list(pair) for pair in canonical],
                         hidden_dag_sha256=task.meta['hidden_dag_sha256'],hidden_leaves=task.meta['hidden_leaves'],
                         n_vars=task.n_vars,c_gen=task.c_gen,start_cost=task.meta['start_cost'],
                         occurrences=task.meta['occurrences'],distinct=task.meta['distinct'],
                         target=sorted(task.target.items())))
    return dict(namespace=NAMESPACE,mechanism='section 10.9 hidden compact-DAG proposal and acceptance logic',
                rng_rule='seed("PolyCompress|<gen_seed_test>|motif_control_random_ood|<j>")',
                status='exploratory-only; never a formal benchmark instance, never used for training, '
                       'validation, test, splits, comparator selection, thresholds or later experiments',
                formal_set_sizes=counts,n_tasks=len(rows),tasks=rows)


def exploratory_exclusion_audit(sets: dict[str,list], path: Path) -> dict:
    """Assert every exploratory-only target is canon-disjoint from every formal benchmark set."""
    if not path.is_file():
        return dict(present=False,manifest=str(path),checked=False,n_tasks=0)
    record = json.loads(path.read_text())
    formal: dict[tuple,str] = {}
    for name in sorted(sets):
        for task in g.set_tasks(name,sets[name]):
            formal.setdefault(p.canon(task.target),name)
    collisions = []
    for row in record['tasks']:
        canonical = p.canon(dict((m,c) for m,c in row['target']))
        assert motif_key(canonical) == row['canonical_key_sha256'], 'exploratory canonical key reproduces'
        if canonical in formal:
            collisions.append([row['diagnostic_task_id'],formal[canonical]])
    assert not collisions, f'exploratory-only tasks collide with formal sets: {collisions}'
    return dict(present=True,manifest=str(path),namespace=record['namespace'],n_tasks=len(record['tasks']),
                checked=True,canon_disjoint=True,formal_canons=len(formal),
                manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                note='exclusion audit only; exploratory tasks are never Gate-A benchmark instances or criteria')


def main() -> None:
    """Generate the diagnostic random-DAG calibration set, its manifest and its motif blocks."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('out/exploratory'))
    args = parser.parse_args()
    from ..protocol import _configs
    cfg,began = _configs(),time.perf_counter()
    taken,counts = formal_canons(cfg)
    print(f'formal canonical targets: {len(taken)} over {sum(counts.values())} tasks',flush=True)
    instances = control_set(cfg['gen'],cfg['random_ood'],taken)
    canonizer = Canonizer()
    args.output.mkdir(parents=True,exist_ok=True)
    record = manifest(instances,counts,canonizer)
    (args.output/MANIFEST).write_text(json.dumps(record,indent=1,sort_keys=True)+'\n')
    blocks = [hidden_block(task,canonizer) for arena,task,start in instances]
    with (args.output/'control_motifs.jsonl').open('w') as stream:
        for block in blocks:
            stream.write(json.dumps(block,sort_keys=True)+'\n')
    print(json.dumps(dict(tasks=len(instances),motif_blocks=len(blocks),
                          distinct_motifs=len({m['canonical_key'] for b in blocks for m in b['motifs']}),
                          canon_calls=canonizer.calls,canon_cache_hits=canonizer.hits,
                          seconds=time.perf_counter()-began),indent=1))


if __name__ == '__main__':
    main()
