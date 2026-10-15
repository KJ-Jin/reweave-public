"""Gate-A freeze candidate, structural dry-run, ceiling summary, census guard, and the mechanical report."""

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from .. import generator as g
from ..protocol import _configs, defaults


def _census(rows: list[dict], cells: int) -> None:
    grid = _configs()['pilot'].r_max_grid
    keys = [(str(row['cell_id']),str(row['r_max'])) for row in rows]
    expected = {(str(cell),str(r)) for cell in range(cells) for r in grid}
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError(f'incomplete or duplicated pilot census: {len(rows)} rows for {cells} cells x {grid}')


def write_candidate(rows: list[dict], output: Path, candidate: Path, cells: int) -> str:
    """Copy the declared defaults untouched after verifying the complete pilot census."""
    _census(rows,cells)
    output.mkdir(parents=True,exist_ok=True)
    data = (json.dumps(defaults(),sort_keys=True,indent=2)+'\n').encode()
    candidate.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def candidate_from_results(directory: str | Path, candidate: str | Path, cells: int) -> str:
    """Write the candidate from merged shard results."""
    path = Path(directory)/'results.csv'
    if not path.is_file():
        raise ValueError(f'merged results missing: {path}')
    with path.open(newline='') as stream:
        return write_candidate(list(csv.DictReader(stream)),Path(directory),Path(candidate),cells)


def _num(value: str) -> float | None:
    return None if value in ('','None') else {'True':1.0,'False':0.0}.get(value,None) if value in ('True','False') else float(value)


def _table(rows: list[dict], columns: list[str]) -> str:
    lines = ['| '+' | '.join(columns)+' |','|'+'---|'*len(columns)]
    lines += ['| '+' | '.join(str(row[c]) for c in columns)+' |' for row in rows]
    return '\n'.join(lines)


def _aggregate(rows: list[dict], keys: tuple[str,...], columns: tuple[str,...]) -> list[dict]:
    groups: dict[tuple,list[dict]] = defaultdict(list)
    for row in rows:
        if int(row['instances']):
            groups[tuple(row[k] for k in keys)].append(row)
    out = []
    for key in sorted(groups,key=lambda k:tuple((int(v) if v.isdecimal() else v) for v in k)):
        rs = groups[key]
        entry = dict(zip(keys,key),cells=len(rs),instances=sum(int(r['instances']) for r in rs))
        for column in columns:
            values = [_num(r[column]) for r in rs if _num(r[column]) is not None]
            entry[column] = f'{sum(values)/len(values):.3f}' if values else '—'
        out.append(entry)
    return out


AGG = ('occurrences','candidates_at_start','truncated','reference_eligible','exhaustive_covered','bound_recall','gen_node_hit',
       'teacher_reached','teacher_cost','teacher_seconds','enumerate_seconds')
STATUSES = ('accepted','rejected','uncovered','ineligible')


def _statuses(rows: list[dict], keys: tuple[str,...]) -> list[dict]:
    groups: dict[tuple,list[dict]] = defaultdict(list)
    for row in rows:
        if int(row['instances']):
            groups[tuple(row[k] for k in keys)].append(row)
    out = []
    for key in sorted(groups,key=lambda k:tuple((int(v) if v.isdecimal() else v) for v in k)):
        counts = defaultdict(int)
        for row in groups[key]:
            for status,n in json.loads(row['reference_status']).items():
                counts[status] += n
        entry = dict(zip(keys,key),instances=sum(int(r['instances']) for r in groups[key]))
        entry.update(eligible=sum(counts[s] for s in STATUSES if s != 'ineligible'),
                     attempted=sum(counts[s] for s in STATUSES if s != 'ineligible'),**{s:counts[s] for s in STATUSES})
        out.append(entry)
    return out


NOTES = ('Rows labelled `train`, `test_disc`, `test_comp` and `test_e4` are the stratified slot-mix calibration census of '
         'each set construction (§10.6), not the benchmark splits: `train` is the train-construction census. '
         '`reference_eligible` is the fraction of instances with `occurrences <= exhaustive_max_occurrences`; only eligible '
         'instances are attempted by the exhaustive reference, and `accepted` / `rejected` / `uncovered` describe attempted '
         'searches only (`uncovered` = time cap). `ineligible` instances were never attempted. `exhaustive_covered` and '
         '`bound_recall` are means over cells of per-cell means restricted to attempted (respectively accepted) references; '
         'they are not pooled action-level rates. Teacher columns come from the ImmediateGain + CostRanker round-0 teacher on '
         'the same census.')


REQUIRED = ('results.csv','splits.json','truncation_diagnostic.csv','rmax_diagnostic.csv',
            'shape_pool_dry_run.json','canonical_diversity.json','ig_ceiling_summary.json',
            'serialization_smoke.json','e7_manifest.json','e7_manifest.sha256')


def require_inputs(output: Path, candidate: Path | None, names: tuple[str,...]) -> None:
    """Refuse a Gate-A assembly whose declared inputs are not all present."""
    missing = [name for name in names if not (output/name).is_file()]
    if candidate is not None and not candidate.is_file():
        missing.append(str(candidate))
    if missing:
        raise ValueError(f'Gate-A inputs missing under {output}: {missing}')


def write_report(output: Path, candidate: Path) -> Path:
    """Assemble the Gate-A report mechanically from the artifacts, plus assessment.md when present."""
    require_inputs(output,candidate,REQUIRED)
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    rows = list(csv.DictReader((output/'results.csv').open()))
    splits = json.loads((output/'splits.json').read_text())
    trunc = list(csv.DictReader((output/'truncation_diagnostic.csv').open()))
    rmax = list(csv.DictReader((output/'rmax_diagnostic.csv').open()))
    dry = json.loads((output/'shape_pool_dry_run.json').read_text())
    diversity = json.loads((output/'canonical_diversity.json').read_text())
    ceiling = json.loads((output/'ig_ceiling_summary.json').read_text())
    smoke = json.loads((output/'serialization_smoke.json').read_text())
    parts = [f'# Phase 1 calibration freeze candidate\n\nCandidate SHA-256: `{digest}`\n\n'
             f'Gate A action (append to `APPROVALS.md`): `CALIBRATION {digest} APPROVE` or `REJECT <reason>`.\n']
    parts.append('## Split audit\n\n'+_table([dict(set=k,instances=v,canonical_shapes=splits['distinct_canons'].get(k,'—'))
                                             for k,v in splits['sets'].items()],['set','instances','canonical_shapes']))
    parts.append(f"Cross-set canonical collisions: {splits['canon_collisions'] or 'none'}. E7 manifest hashes: `{splits.get('e7_hashes',splits['e7_manifest_sha256'])}`; "
                 f"round trip: {splits.get('e7_round_trip','not recorded')}. E7b grid statuses: {splits['e7b']['statuses']}. "
                 f"Pending: {splits['pending'] or 'none'}.")
    parts.append('## Calibration census notes\n\n'+NOTES)
    status_columns = ['instances','eligible','attempted']+list(STATUSES)
    parts.append('## Exhaustive reference eligibility and status by construction set and r_max\n\n'
                 +_table(_statuses(rows,('set','r_max')),['set','r_max']+status_columns))
    parts.append('## Exhaustive reference eligibility and status by construction set and family (r_max = 3)\n\n'
                 +_table(_statuses([r for r in rows if r['r_max'] == '3'],('set','family')),['set','family']+status_columns))
    parts.append('## Calibration census by construction set and r_max\n\n'+_table(_aggregate(rows,('set','r_max'),AGG),['set','r_max','cells','instances']+list(AGG)))
    parts.append('## Calibration census by construction set, family, params and r_max\n\n'+_table(_aggregate(rows,('set','family','params','r_max'),AGG),
                                                                               ['set','family','params','r_max','cells','instances']+list(AGG)))
    parts.append('## Truncation diagnostic\n\n'+_table(trunc,list(trunc[0])))
    agg = _aggregate([dict(r,instances='1') for r in rmax],('r_max',),('covered','reached_C_gen','best_cost','best_depth','states','seconds'))
    parts.append('## Bounded r_max reachability (all exhaustive-eligible instances)\n\n'+_table(agg,list(agg[0])))
    parts.append('## Pre-generation structural dry-run (`shape_pool_dry_run.json`)\n\n'
                 +_table([dict(cell=cell,**row) for cell,row in dry['composition'].items()],
                         ['cell','grid','admissible_constructions','unique_canons','owned_shapes'])
                 +'\n\n'+_table([dict(outer_family=f,**row) for f,row in dry['discovery'].items()],
                                 ['outer_family','admissible_shapes','excluded_by_test_comp','owned_shapes'])
                 +'\n\n'+_table([dict(row,params=json.dumps(row['params']),**row.pop('owned')) for row in dry['ordinary']],
                                 ['family','params','available','reserved','val','test_id','train']))
    parts.append('## Canonical-shape diversity census (`canonical_diversity.json`)\n\n'
                 +_table([dict(split=name,scope='aggregate',**body['aggregate'])
                          for name,body in diversity.items()]
                         +[dict(split=name,scope=f"{row['family']}{row['params']}",**{k:v for k,v in row.items() if k not in ('family','params')})
                           for name,body in diversity.items() for row in body['strata']],
                         ['split','scope','n_instances','n_distinct_canon','n_skeleton_signatures','instances_per_canon',
                          'instances_per_canon_distribution']))
    parts.append(f"## Primary-inference strata (canonical-shape counts frozen before training)\n\n"
                 +_table(ceiling['primary_strata'],['stratum','n_instances','n_distinct_canon']))
    parts.append('## ImmediateGain ceiling and headroom (`ig_ceiling_summary.json`)\n\n'
                 +_table(ceiling['by_family'],['stratum','instances','SR_IG','headroom'])
                 +f"\n\nFrozen non-ceiling cell set `headroom_cells` (`SR_IG < 0.9`) at the frozen `r_max`: "
                 f"{len(ceiling['headroom_cells'])} of {ceiling['n_cells']} measured cells.")
    eligible = splits['oracle_eligible_random_ood']
    parts.append(f"## Random-OOD oracle eligibility (frozen before Phase 2)\n\n"
                 f"Rule `{eligible['rule']}`: {eligible['n_eligible']} of {eligible['n_full']} tasks eligible.\n\n"
                 +_table([dict(scope=scope,statistic=name,distribution=json.dumps(dist))
                          for scope,body in eligible['distributions'].items() for name,dist in body.items()],
                         ['scope','statistic','distribution']))
    parts.append('## Frozen qualitative display ids\n\n'+json.dumps(splits['display_task_ids']))
    parts.append(f"## Training-record serialization projection\n\nRound-0 dataset {smoke['projected_round_bytes']/2**20:.1f} MiB; "
                 f"full five-seed pipeline {smoke['projected_pipeline_bytes']/2**20:.1f} MiB over "
                 f"{smoke['dataset_equivalents']} dataset-equivalents; worker peak RSS {smoke['worker_peak_rss_mb']:.0f} MiB.")
    assessment = output/'assessment.md'
    parts.append('## Assessment\n\n'+(assessment.read_text() if assessment.is_file() else '(no assessment.md present)'))
    parts.append('## Full pilot table (one row per cell and r_max)\n\n'+_table(rows,list(rows[0])))
    path = output/'freeze_report.md'
    path.write_text('\n\n'.join(parts)+'\n')
    return path


def shape_pool_dry_run(cfg: dict, output: Path) -> dict:
    """Enumerate only admissible constructions and canonical shapes; emit no concrete instance."""
    comp,disc,reserved = g.reserve_shapes(cfg['gen'])
    cells = g.shape_table(cfg['gen'],reserved)
    report = dict(reserved_canons=len(reserved),
                  composition={cell:dict(grid=len(g.composition_grid(*cell.split('∘'),cfg['gen'])),
                                         admissible_constructions=pool['constructions'],
                                         unique_canons=pool['unique_canons'],owned_shapes=len(pool['shapes']))
                               for cell,pool in comp.items()},
                  discovery={family:dict(admissible_shapes=len(cell['shapes']),
                                         excluded_by_test_comp=len(cell['excluded']),
                                         owned_shapes=len(cell['owned'])) for family,cell in disc.items()},
                  ordinary=[dict(family=cell['family'],params=list(cell['params']),available=len(cell['shapes']),
                                 reserved=len(cell['reserved']),
                                 owned={split:len(g.owned_groups(cell,split)) for split in ('val','test_id','train')})
                            for cell in cells])
    output.mkdir(parents=True,exist_ok=True)
    (output/'shape_pool_dry_run.json').write_text(json.dumps(report,indent=1)+'\n')
    return report


def ig_ceiling_summary(rows: list[dict], diversity: dict, output: Path) -> dict:
    """Freeze the ImmediateGain reached-C_gen rate, its headroom and the non-ceiling cell set."""
    frozen = str(_configs()['calc'].r_max)
    cells,families = [],defaultdict(list)
    for row in rows:
        if row['r_max'] != frozen or not int(row['instances']):
            continue
        rate,weight = _num(row['teacher_reached']),int(row['instances'])
        cells.append(dict(cell=f"{row['set']}|{row['family']}|{row['params']}|{row['slot_mix']}",
                          instances=weight,SR_IG=rate,headroom=None if rate is None else 1-rate))
        families[f"{row['set']}|{row['family']}"].append((rate,weight))
    by_family = []
    for stratum,values in sorted(families.items()):
        total = sum(weight for rate,weight in values if rate is not None)
        rate = sum(rate*weight for rate,weight in values if rate is not None)/total if total else None
        by_family.append(dict(stratum=stratum,instances=total,SR_IG=None if rate is None else round(rate,4),
                              headroom=None if rate is None else round(1-rate,4)))
    primary = [dict(stratum=f'{split}|{row["family"]}',n_instances=row['n_instances'],
                    n_distinct_canon=row['n_distinct_canon'])
               for split in ('test_id','test_disc','test_comp') if split in diversity
               for row in diversity[split]['primary_strata']]
    report = dict(r_max=int(frozen),cells=cells,by_family=by_family,primary_strata=primary,n_cells=len(cells),
                  headroom_cells=sorted(cell['cell'] for cell in cells if cell['SR_IG'] is not None and cell['SR_IG'] < 0.9))
    output.mkdir(parents=True,exist_ok=True)
    (output/'ig_ceiling_summary.json').write_text(json.dumps(report,indent=1)+'\n')
    return report


def gate_a_report(cfg: dict, output: Path, candidate: Path, root: Path) -> str:
    """Assemble the Gate-A candidate, ceiling summary, report and provenance from the finished artifacts."""
    from .pilot import pilot_cells
    require_inputs(output,None,tuple(name for name in REQUIRED if name != 'ig_ceiling_summary.json'))
    from .e7 import load_manifest
    load_manifest(output)
    rows = list(csv.DictReader((output/'results.csv').open()))
    diversity = json.loads((output/'canonical_diversity.json').read_text())
    ig_ceiling_summary(rows,diversity,output)
    digest = candidate_from_results(output,candidate,len(pilot_cells(cfg)))
    report = write_report(output,candidate)
    from ..protocol import write_provenance
    artifacts = sorted(path for path in output.iterdir() if path.is_file() and path.name != 'provenance.json')+[candidate]
    write_provenance(root,output/'provenance.json',artifacts,cfg=cfg)
    return digest


def main() -> None:
    """Run one Gate-A stage: the pre-generation structural dry-run, or the final report assembly."""
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=('dry-run','report'))
    parser.add_argument('--output',type=Path,default=Path('out/pilot'))
    parser.add_argument('--candidate',type=Path,default=Path('frozen_candidate.json'))
    args = parser.parse_args()
    cfg,began = _configs(),time.perf_counter()
    if args.stage == 'dry-run':
        print(json.dumps(shape_pool_dry_run(cfg,args.output),indent=1))
    else:
        print('candidate',gate_a_report(cfg,args.output,args.candidate,Path('.').resolve()))
    print(f'{args.stage} done in {time.perf_counter()-began:.0f} s',flush=True)


if __name__ == '__main__':
    main()
