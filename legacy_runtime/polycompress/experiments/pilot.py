"""Phase-one exhaustive recall, teacher trajectories, and per-cell pilot measurements."""

import argparse
from collections import Counter, deque
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time

from ..calculus import Action,Task,apply,group,strip,verify
from ..config import CalcCfg,PilotCfg
from ..enumerate import enumerate_actions
from ..handles import Arena,Mono,canonical_start,cost,items,poly
from ..heuristics import ImmediateGain,CostRanker
from ..search import Budget,beam_search,peak_rss_mb,run_tasks,sized_compute
from ..state import StateInfo, state_info
from .reference import trajectory_record, write_trajectories
from .. import generator as g
from . import e7
from .. import poly as p

NUMERIC = ('n_vars','occurrences','distinct','start_cost','C_gen','candidates_at_start','enumerate_seconds',
           'apply_seconds','truncated','bound_recall','gen_node_hit','trajectory_length','reached_C_gen',
           'reference_eligible','reference_attempted','exhaustive_covered','exhaustive_reached','exhaustive_seconds',
           'teacher_length','teacher_reached','teacher_cost','teacher_seconds','teacher_expansions','teacher_applies')
STATUSES = ('accepted','rejected','uncovered','ineligible')


def _path(parents: dict, root: int) -> tuple[list[int],list[Action]]:
    roots,actions = [root],[]
    while parents[root] is not None:
        parent,action = parents[root]
        actions.append(action)
        roots.append(parent)
        root = parent
    return roots[::-1],actions[::-1]


def exhaustive_trajectory(arena: Arena, task: Task, start: int, cfg: PilotCfg) -> dict:
    """Find a shortest successful calculus trajectory with exact breadth-first search."""
    began = time.perf_counter()
    queue,parents = deque([(start,0)]),{start:None}
    covered,reached,last = True,False,start
    p.timed_exhaustive(True)
    while queue:
        if time.perf_counter()-began >= cfg.exhaustive_time_s:
            covered = False
            break
        root,depth = queue.popleft()
        if cost(arena,root) <= task.c_gen:
            reached,last = True,root
            break
        if depth >= cfg.exhaustive_max_depth:
            continue
        info = state_info(arena,root)
        r_max = max((len(items(arena.handles[c])) for c in info.contexts),default=0)
        actions = enumerate_actions(arena,info,root,CalcCfg(r_max=r_max,n_max=sys.maxsize))
        for action in actions:
            if time.perf_counter()-began >= cfg.exhaustive_time_s:
                covered = False
                break
            successor = apply(arena,task,info,root,action)
            if successor is None or successor in parents:
                continue
            verify(arena,task,successor)
            parents[successor] = root,action
            queue.append((successor,depth+1))
        if not covered:
            break
    # Status and path information are fixed here; reconstruction and export follow (section 17.A).
    p.timed_exhaustive(False)
    roots,actions = _path(parents,last)
    return dict(covered=covered,reached=reached,roots=roots,actions=actions,
                seconds=time.perf_counter()-began)


def trajectory_recall(arena: Arena, trajectory: dict, cfg: CalcCfg) -> float | None:
    """Measure bounded action inclusion along a covered successful path."""
    actions = trajectory['actions']
    if not trajectory['covered'] or not trajectory['reached'] or not actions:
        return None
    present = sum(action in enumerate_actions(arena,state_info(arena,root),root,cfg)
                  for root,action in zip(trajectory['roots'],actions))
    return present/len(actions)


def bounded_best(arena: Arena, task: Task, start: int, horizon: int, cfg: CalcCfg,
                 deadline: float | None = None) -> int | None:
    """Exhaust all bounded candidates to a fixed continuation horizon."""
    frontier,seen,best = [start],{start},cost(arena,start)
    for _ in range(horizon):
        successors = []
        for root in frontier:
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            info = state_info(arena,root)
            for action in enumerate_actions(arena,info,root,cfg):
                if deadline is not None and time.perf_counter() >= deadline:
                    return None
                nxt = apply(arena,task,info,root,action)
                if nxt is None or nxt in seen:
                    continue
                verify(arena,task,nxt)
                seen.add(nxt)
                successors.append(nxt)
                best = min(best,cost(arena,nxt))
        frontier = successors
        if not frontier:
            break
    return None if deadline is not None and time.perf_counter() >= deadline else best


def generator_nodes(arena: Arena, gen: int) -> list[p.Poly]:
    """List the polynomials of every non-monomial generator node."""
    return [poly(arena,h) for h in sorted(state_info(arena,gen).reach_h) if not isinstance(arena.handles[h],Mono)]


def _gen_node_hit(arena: Arena, info: StateInfo, actions: list[Action], targets: list[p.Poly]) -> bool:
    for a in actions:
        if a.kind == 'FACTOR':
            rem: Counter[int] = Counter()
            for h,c in a.region:
                stripped = strip(arena,h,a.arg)
                if stripped is None:
                    break
                rem[stripped] += c
            else:
                if poly(arena,group(arena,tuple(sorted(rem.items())))) in targets:
                    return True
        elif a.src is not None and poly(arena,group(arena,a.src)) in targets:
            return True
    return False


def measure_bounded(task: Task, targets: list[p.Poly], calc: CalcCfg, budget: Budget) -> dict:
    """Measure start diagnostics on a probe arena and the teacher from its own fresh canonical arena."""
    probe = Arena()
    probe.calc_cfg = calc
    start = canonical_start(probe,task.target)
    info = state_info(probe,start)
    began = time.perf_counter()
    actions = enumerate_actions(probe,info,start,calc)
    enum_seconds = time.perf_counter()-began
    truncated = bool(probe.truncations)
    hit = _gen_node_hit(probe,info,actions,targets)
    began = time.perf_counter()
    for action in actions:
        apply(probe,task,info,start,action)
    apply_seconds = time.perf_counter()-began
    arena = Arena()
    arena.calc_cfg = calc
    start = canonical_start(arena,task.target)
    assert arena.apply_count == 0 and not arena.apply_memo, 'teacher counters begin before any probe'
    incumbent,stats,tree = beam_search(arena,task,start,ImmediateGain(task),CostRanker(),budget,True)
    parents = {h:None if node['parent'] is None else (node['parent'],node['action']) for h,node in tree.items()}
    return dict(n_vars=task.n_vars,occurrences=sum(task.target.values()),distinct=len(task.target),
                start_cost=cost(arena,start),C_gen=task.c_gen,candidates_at_start=len(actions),
                enumerate_seconds=enum_seconds,apply_seconds=apply_seconds,truncated=truncated,gen_node_hit=hit,
                teacher_length=len(_path(parents,incumbent)[1]),teacher_reached=cost(arena,incumbent) <= task.c_gen,
                teacher_cost=cost(arena,incumbent),teacher_seconds=stats.wall,teacher_expansions=stats.expansions,
                teacher_applies=stats.applies,peak_rss_mb=peak_rss_mb())


def measure_instance(task: Task, targets: list[p.Poly], calc: CalcCfg, pilot: PilotCfg,
                     budget: Budget) -> dict[int,dict]:
    """Run the r_max-independent exhaustive reference once if eligible, then every bounded r_max measurement."""
    return measure_reference(task,targets,calc,pilot,budget,None)[0]


def measure_reference(task: Task, targets: list[p.Poly], calc: CalcCfg, pilot: PilotCfg,
                      budget: Budget, cell: dict | None) -> tuple[dict[int,dict],dict | None]:
    """Measure one instance and, for a named cell, export its accepted trajectory after the timed search."""
    arena = Arena()
    start = canonical_start(arena,task.target)
    eligible = sum(task.target.values()) <= pilot.exhaustive_max_occurrences
    # An ineligible instance is never attempted: it is neither covered nor a time cap.
    exhaustive = dict(covered=None,reached=None,roots=[start],actions=[],seconds=0.0)
    if eligible:
        exhaustive = exhaustive_trajectory(arena,task,start,pilot)
    result = {}
    for r_max in pilot.r_max_grid:
        began = time.perf_counter()
        bounded = replace(calc,r_max=r_max)
        row = measure_bounded(task,targets,bounded,budget)
        row.update(bound_recall=trajectory_recall(arena,exhaustive,bounded),exhaustive_covered=exhaustive['covered'],
                   exhaustive_reached=exhaustive['reached'],exhaustive_seconds=exhaustive['seconds'],
                   reference_eligible=eligible,reference_attempted=eligible)
        row['reference_status'] = search_status(exhaustive['covered'],exhaustive['reached']) if eligible else 'ineligible'
        if exhaustive['covered'] and exhaustive['reached']:
            row.update(trajectory_length=len(exhaustive['actions']),reached_C_gen=True,trajectory_source='exhaustive')
        else:
            row.update(trajectory_length=row['teacher_length'],reached_C_gen=row['teacher_reached'],
                       trajectory_source='ImmediateGain')
        row['seconds'] = exhaustive['seconds']+time.perf_counter()-began
        result[r_max] = row
    exported = None
    if cell is not None and eligible and exhaustive['covered'] and exhaustive['reached']:
        exported = trajectory_record(arena,task,exhaustive,targets,cell)
    return result,exported


def search_status(covered: bool, accepted: bool) -> str:
    """Name a time-limited search outcome: a timeout is uncovered, never a proved rejection."""
    return 'uncovered' if not covered else 'accepted' if accepted else 'rejected'


def measure_cell(job: dict) -> list[dict]:
    """Aggregate the supplied instances into one complete pilot row per r_max."""
    return measure_cell_export(dict(job,export=False))['rows']


def measure_cell_export(job: dict) -> dict:
    """Aggregate one cell's rows and, unless disabled, its accepted reference trajectories (section 17.A)."""
    cell = None if job.get('export') is False else {key:job[key] for key in ('cell_id','set','family','params','slot_mix')}
    measured = [measure_reference(task,targets,job['calc'],job['pilot'],job['budget'],cell)
                for task,targets in job['instances']]
    results = [result for result,exported in measured]
    trajectories = [exported for result,exported in measured if exported is not None]
    rows = []
    for r_max in job['pilot'].r_max_grid:
        per = [result[r_max] for result in results]
        row = {key:job[key] for key in ('cell_id','set','family')}
        row.update(params=json.dumps(job['params']),slot_mix=json.dumps(job['slot_mix']),r_max=r_max,
                   n_workers=job['n_workers'],instances=len(per))
        for key in NUMERIC:
            values = [r[key] for r in per if r[key] is not None]
            row[key] = sum(values)/len(values) if values else None
        row['peak_rss_mb'] = max((r['peak_rss_mb'] for r in per),default=peak_rss_mb())
        row['trajectory_source'] = json.dumps({source:sum(r['trajectory_source'] == source for r in per)
                                               for source in ('exhaustive','ImmediateGain')},sort_keys=True)
        row['reference_status'] = json.dumps({status:sum(r['reference_status'] == status for r in per)
                                              for status in STATUSES},sort_keys=True)
        row['rejected_by_reason'] = json.dumps(job['rejected_by_reason'],sort_keys=True)
        row['seconds'] = sum(r['seconds'] for r in per)
        rows.append(row)
    return dict(rows=rows,trajectories=trajectories)


def _draw(cell: dict, rng: random.Random, cfg: dict) -> list[tuple[Task,list[p.Poly]]]:
    groups = [cell['shapes'][shape] for shape in sorted(cell['shapes'])]
    instances = g.emit_shapes(cell,groups,cfg['pilot'].instances_per_cell,rng,cfg['gen'],'pilot')
    return [(task,generator_nodes(arena,gen)) for arena,task,gen in instances]


def pilot_cells(cfg: dict) -> list[dict]:
    """Draw the calibration pool over every set construction's slot-mix cells from the pilot stream."""
    rng,jobs = random.Random(g.seed('pilot')),[]
    grids = (('train',g.family_cells(cfg['gen']),False),('test_disc',[(f,(2,2)) for f in ('F1','F2','F3')],True))
    for name,grid,disc in grids:
        for family,params in grid:
            for sub in g.mix_cells(g.enumerate_cell(family,params,cfg['gen'],disc)):
                jobs.append(dict(set=name,family=family,params=params,slot_mix=sub['mix'],
                                 instances=_draw(sub,rng,cfg) if sub['shapes'] else [],
                                 rejected_by_reason=dict(sub['rejected'])))
    pools = g.composition_pools(cfg['gen'])
    for outer,inner in g.COMPOSITIONS:
        owned = pools[f'{outer}∘{inner}']['shapes']
        for params in g.FAMILY_GRID[outer]:
            shapes = sorted(shape for shape,d in owned.items() if d[0] == params)
            mix = tuple(sorted((inner,)+('x',)*(g.n_slots(outer,params)-1)))
            built = []
            quota = g.even_split(cfg['pilot'].instances_per_cell,len(shapes)) if shapes else []
            for count,shape in zip(quota,shapes):
                op,ip,slot = owned[shape]
                built += [g.build_composition(outer,op,inner,ip,slot,rng) for _ in range(count)]
            jobs.append(dict(set='test_comp',family=outer,params=params,slot_mix=mix,rejected_by_reason={},
                             instances=[(task,generator_nodes(arena,gen)) for arena,task,gen in built]))
    for sub in g.mix_cells(g.enumerate_cell('F1',(3,3),cfg['gen'])):
        jobs.append(dict(set='test_e4',family='F1',params=(3,3),slot_mix=sub['mix'],instances=_draw(sub,rng,cfg),
                         rejected_by_reason=dict(sub['rejected'])))
    for i,job in enumerate(jobs):
        job.update(cell_id=i,calc=cfg['calc'],pilot=cfg['pilot'],budget=cfg['search'].teacher,
                   n_workers=cfg['compute'].n_workers)
        for task,targets in job['instances']:
            task.meta['id'] = sum(len(j['instances']) for j in jobs[:i])+job['instances'].index((task,targets))
    return jobs


def shard_indices(shard: str, length: int) -> range:
    """Parse the sole declared cluster partition option."""
    parts = shard.split('/')
    if len(parts) != 2 or not all(part.isdecimal() for part in parts):
        raise ValueError('shard must be i/n')
    i,n = map(int,parts)
    if n < 1 or i >= n:
        raise ValueError('shard requires 0 <= i < n')
    return range(i,length,n)


def write_shard(rows: list[dict], output: Path, shard: str) -> Path:
    """Write one shard's measured rows."""
    shard_indices(shard,len(rows))
    if not rows:
        raise ValueError('cannot write an empty pilot shard')
    output.mkdir(parents=True,exist_ok=True)
    path = output/f"shard_{shard.split('/')[0]}.csv"
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _census(cfg: dict, shard: str, output: Path) -> None:
    """Measure this shard's pilot cells and write the candidate for an unsharded run."""
    compute,sizing = sized_compute(cfg['compute'],'census')
    cells = pilot_cells(cfg)
    selected = [cells[i] for i in shard_indices(shard,len(cells))]
    for job in selected:
        job['n_workers'] = sizing['workers']
    measured = run_tasks(measure_cell_export,[(job['cell_id'],job) for job in selected],compute)
    rows = [row for cell in measured for row in cell['rows']]
    write_shard(rows,output,shard)
    write_trajectories([record for cell in measured for record in cell['trajectories']],output,shard)
    if shard == '0/1':
        from .freeze import write_candidate
        write_candidate(rows,output,Path('frozen_candidate.json'),len(cells))


def main() -> None:
    """Run one Phase-1 pilot stage: the sharded census, the split audit, or the two Gate-A diagnostics."""
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',nargs='?',default='census',choices=('census','audit','diagnostics'))
    parser.add_argument('--shard',default='0/1')
    parser.add_argument('--output',type=Path,default=Path('out/pilot'))
    args = parser.parse_args()
    shard_indices(args.shard,0)
    from ..protocol import _configs
    cfg,began = _configs(),time.perf_counter()
    if args.stage == 'census':
        _census(cfg,args.shard,args.output)
    elif args.stage == 'audit':
        summary = split_audit(cfg,args.output)
        print(json.dumps({key:value for key,value in summary.items() if key not in ('cells','random_ood','scale_stress','named_cases')},
                         indent=1,default=list))
    else:
        jobs = pilot_cells(cfg)
        print('rmax rows',len(rmax_diagnostic(jobs,cfg,args.output)),flush=True)
        print('truncation rows',len(truncation_diagnostic(jobs,cfg,args.output)),flush=True)
    print(f'{args.stage} done in {time.perf_counter()-began:.0f} s',flush=True)


def split_audit(cfg: dict, output: Path = Path('out/pilot')) -> dict:
    """Generate every specified set, run the cross-set audits, and record the split summary."""
    began = time.perf_counter()
    comp_pools,disc_pools,reserved = g.reserve_shapes(cfg['gen'])
    sets,e7b_statuses = g.generate_splits(cfg['gen'],cfg['calc'],cfg['pilot'],cfg['compute'],
                                          cfg['random_ood'],cfg['scale'])
    cells = g.shape_table(cfg['gen'],reserved)
    canons = g.leakage_audit(sets)
    collisions = canon_collisions(sets)
    assert not collisions, f'cross-set canonical collisions remain: {collisions}'
    comp = {p.canon(t.target) for t in g.set_tasks('test_comp',sets['test_comp'])}
    others = {p.canon(t.target) for name in sets if name not in ('test_comp','test_disc') for t in g.set_tasks(name,sets[name])}
    for family,cell in disc_pools.items():
        assert not set(cell['owned']) & others, f'discovery shape of {family} appears in another set'
    from .control import MANIFEST, exploratory_exclusion_audit
    exploratory = exploratory_exclusion_audit(sets,output.parent/'exploratory'/MANIFEST)
    hashes = e7.write_manifest(sets,e7b_statuses,cfg['calc'],output)
    verified = e7.verify_manifest(e7.load_manifest(output),cfg['calc'])
    comp_tasks = g.set_tasks('test_comp',sets['test_comp'])
    summary = dict(sets={name:len(entries) for name,entries in sets.items()},distinct_canons=canons,
                   canon_collisions=collisions,e7_manifest_sha256=hashes['manifest'],e7_hashes=hashes,e7_round_trip=verified,
                   cells=[dict(family=c['family'],params=c['params'],shapes=len(c['shapes']),
                               choices=sum(map(len,c['shapes'].values())),reserved=len(c['reserved']),
                               owned={s:len(g.owned_groups(c,s)) for s in ('train','val','test_id')},
                               rejected=[[list(mix),reason,n] for (mix,reason),n in sorted(c['rejected'].items())])
                          for c in cells],
                   test_comp={cell:dict(constructions=pool['constructions'],unique_canons=pool['unique_canons'],
                                        owned_shapes=len(pool['shapes']),
                                        emitted=sum(t.meta['composition_cell'] == cell for t in comp_tasks),
                                        emitted_shapes=len({p.canon(t.target) for t in comp_tasks if t.meta['composition_cell'] == cell}))
                              for cell,pool in comp_pools.items()},
                   test_disc={family:dict(shapes=len(cell['shapes']),excluded_composition_twins=len(cell['excluded']),
                                          owned_shapes=len(cell['owned']),
                                          emitted=sum(t.meta['family'] == family for t in g.set_tasks('test_disc',sets['test_disc'])),
                                          emitted_shapes=len({p.canon(t.target) for t in g.set_tasks('test_disc',sets['test_disc']) if t.meta['family'] == family}),
                                          mixes=len([m for m in g.mix_cells(cell) if m['shapes']]),
                                          rejected=[[list(mix),reason,n] for (mix,reason),n in sorted(cell['rejected'].items())])
                              for family,cell in disc_pools.items()},
                   e7a=dict(pairs=len(sets['e7a']),drawn=cfg['gen'].n_e7_pairs),
                   e7b=dict(grid=len(e7b_statuses),classes=len({s['canonical_class'] for s in e7b_statuses}),pairs=len(sets['e7b']),
                            statuses=dict(sorted(Counter(s['status'] for s in e7b_statuses).items())),
                            by_cell=dict(Counter(f"{e[1].meta['embedded_family']}|d={e[1].meta['params'][0]}" for e in sets['e7b']))),
                   test_comp_canonical_shapes=len(comp),pending=[],exploratory_only=exploratory,
                   random_ood=_suite_summary(sets['random_ood'],('hidden_dag_sha256',)),
                   scale_stress=_suite_summary(sets['scale_stress'],('hidden_dag_sha256','bucket')),
                   named_cases=_suite_summary(sets['named_cases'],('target_sha256',)),
                   oracle_eligible_random_ood=_oracle_eligible(sets['random_ood'],cfg['oracle']),
                   display_task_ids=_display_ids(sets,comp_tasks,comp_pools))
    canonical_diversity(sets,output)
    summary['seconds'] = time.perf_counter()-began
    output.mkdir(parents=True,exist_ok=True)
    (output/'splits.json').write_text(json.dumps(summary,indent=2,default=list)+'\n')
    return summary


def _display_ids(sets: dict[str,list], comp_tasks: list[Task], comp_pools: dict[str,dict]) -> dict:
    """Freeze the qualitative display ids from generator order alone, never from a method outcome."""
    return dict(test_comp=[min(t.meta['id'] for t in comp_tasks if t.meta['composition_cell'] == cell)
                           for cell in sorted(comp_pools)],
                random_ood=sorted(entry[1].meta['id'] for entry in sets['random_ood'])[:4],
                named_cases=sorted(entry[1].meta['id'] for entry in sets['named_cases']))


def _distribution(tasks: list[Task]) -> dict:
    """Tabulate the structural statistics frozen for oracle eligibility."""
    read = dict(n_vars=lambda t:t.n_vars,occurrences=lambda t:t.meta['occurrences'],
                C_start=lambda t:t.meta['start_cost'],C_gen=lambda t:t.c_gen)
    return {name:dict(sorted(Counter(get(t) for t in tasks).items())) for name,get in read.items()}


def _oracle_eligible(entries: list, oracle) -> dict:
    """Freeze the structurally oracle-eligible random-OOD ids before Phase 2; coverage is measured later."""
    tasks = [entry[1] for entry in entries]
    eligible = [t for t in tasks if t.meta['occurrences'] <= oracle.max_occurrences and t.n_vars <= oracle.max_vars]
    return dict(rule=dict(max_occurrences=oracle.max_occurrences,max_vars=oracle.max_vars),
                ids=sorted(t.meta['id'] for t in eligible),n_eligible=len(eligible),n_full=len(tasks),
                distributions=dict(eligible=_distribution(eligible),full=_distribution(tasks)))


def canonical_diversity(sets: dict[str,list], output: Path) -> dict:
    """Report instances, distinct canons, skeleton signatures and instances per canon for every stratum."""
    report = {}
    for name,entries in sorted(sets.items()):
        tasks = g.set_tasks(name,entries)
        strata: dict[tuple,list[Task]] = {}
        for task in tasks:
            key = (task.meta.get('composition_cell') or task.meta['family'],json.dumps(list(task.meta['params'])))
            strata.setdefault(key,[]).append(task)
        report[name] = dict(aggregate=_diversity(tasks),
                            strata=[dict(family=family,params=json.loads(params),**_diversity(group))
                                    for (family,params),group in sorted(strata.items())])
        if name in ('test_id','test_disc','test_comp'):
            report[name]['primary_strata'] = [dict(family=family,**_diversity(
                [task for (f,params),group in strata.items() if f == family for task in group]))
                for family in sorted({family for family,params in strata})]
    output.mkdir(parents=True,exist_ok=True)
    (output/'canonical_diversity.json').write_text(json.dumps(report,indent=1,default=list)+'\n')
    return report


def _diversity(tasks: list[Task]) -> dict:
    """Summarize one stratum's structural diversity."""
    shapes = Counter(p.canon(task.target) for task in tasks)
    signatures = {task.meta['skeleton_signature'] for task in tasks if task.meta['skeleton_signature'] is not None}
    return dict(n_instances=len(tasks),n_distinct_canon=len(shapes),
                n_skeleton_signatures=len(signatures) if signatures else None,
                instances_per_canon=len(tasks)/len(shapes),
                instances_per_canon_distribution=dict(sorted(Counter(shapes.values()).items())))


def _suite_summary(entries: list, extra: tuple[str,...]) -> dict:
    """Freeze a generated suite's per-task identity, reference cost and structural statistics."""
    tasks = [entry[1] for entry in entries]
    rows = [dict(id=t.meta['id'],family=t.meta['family'],params=list(t.meta['params']),n_vars=t.n_vars,
                 C_gen=t.c_gen,C_start=t.meta['start_cost'],occurrences=t.meta['occurrences'],
                 distinct=t.meta['distinct'],target=sorted(t.target.items()),
                 target_sha256=hashlib.sha256(repr(sorted(t.target.items())).encode()).hexdigest(),
                 **{key:t.meta[key] for key in extra if key != 'target_sha256'}) for t in tasks]
    return dict(n_tasks=len(rows),tasks=rows)


def canon_collisions(sets: dict[str,list]) -> list[list[str]]:
    """List every group of set names sharing a canonical target, without asserting disjointness."""
    owners: dict[tuple,set[str]] = {}
    for name,entries in sets.items():
        for task in g.set_tasks(name,entries):
            owners.setdefault(p.canon(task.target),set()).add(name)
    return sorted(sorted(names) for names in owners.values() if len(names) > 1)


def bounded_exhaustive(arena: Arena, task: Task, start: int, calc: CalcCfg, cfg: PilotCfg) -> dict:
    """Explore the r_max-bounded action space breadth-first and record the best reachable cost."""
    began = time.perf_counter()
    queue,seen = deque([(start,0)]),{start}
    best,best_depth,covered = cost(arena,start),0,True
    p.timed_exhaustive(True)
    while queue and covered:
        root,depth = queue.popleft()
        if depth >= cfg.exhaustive_max_depth:
            continue
        info = state_info(arena,root)
        for action in enumerate_actions(arena,info,root,calc):
            if time.perf_counter()-began >= cfg.exhaustive_time_s:
                covered = False
                break
            successor = apply(arena,task,info,root,action)
            if successor is None or successor in seen:
                continue
            verify(arena,task,successor)
            seen.add(successor)
            queue.append((successor,depth+1))
            if cost(arena,successor) < best:
                best,best_depth = cost(arena,successor),depth+1
    p.timed_exhaustive(False)
    return dict(best_cost=best,reached_C_gen=best <= task.c_gen,best_depth=best_depth,covered=covered,
                status=search_status(covered,best <= task.c_gen),states=len(seen),seconds=time.perf_counter()-began)


def _rmax_instance(payload: tuple) -> list[dict]:
    task,row,calc,pilot = payload
    rows = []
    for r_max in pilot.r_max_grid:
        arena = Arena()
        arena.calc_cfg = replace(calc,r_max=r_max)
        start = canonical_start(arena,task.target)
        rows.append(dict(row,r_max=r_max,start_cost=cost(arena,start),**bounded_exhaustive(arena,task,start,arena.calc_cfg,pilot)))
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rmax_diagnostic(jobs: list[dict], cfg: dict, output: Path) -> list[dict]:
    """Compare r_max values by bounded exhaustive reachability on every exhaustive-eligible pilot instance."""
    payloads = []
    for job in jobs:
        for i,(task,targets) in enumerate(job['instances']):
            occurrences = sum(task.target.values())
            if occurrences > cfg['pilot'].exhaustive_max_occurrences:
                continue
            oracle = occurrences <= cfg['oracle'].max_occurrences and task.n_vars <= cfg['oracle'].max_vars
            row = dict(cell_id=job['cell_id'],set=job['set'],family=job['family'],params=json.dumps(job['params']),
                       slot_mix=json.dumps(job['slot_mix']),instance=i,pilot_id=task.meta['id'],n_vars=task.n_vars,
                       occurrences=occurrences,C_gen=task.c_gen,oracle_eligible=oracle)
            payloads.append((len(payloads),(task,row,cfg['calc'],cfg['pilot'])))
    compute,sizing = sized_compute(cfg['compute'],'rmax_diagnostic')
    rows = [row for rows in run_tasks(_rmax_instance,payloads,compute) for row in rows]
    _write_csv(rows,output/'rmax_diagnostic.csv')
    return rows


def truncation_diagnostic(jobs: list[dict], cfg: dict, output: Path) -> list[dict]:
    """Measure dead and duplicate candidates before and after the n_max sample on truncated pilot cells."""
    rows = []
    for job in jobs:
        for i,(task,targets) in enumerate(job['instances'][:1]):
            for r_max in cfg['pilot'].r_max_grid:
                arena = Arena()
                arena.calc_cfg = calc = replace(cfg['calc'],r_max=r_max)
                start = canonical_start(arena,task.target)
                info = state_info(arena,start)
                full = enumerate_actions(arena,info,start,replace(calc,n_max=sys.maxsize))
                if len(full) <= calc.n_max:
                    continue
                sample = enumerate_actions(arena,info,start,calc)
                row = dict(cell_id=job['cell_id'],set=job['set'],family=job['family'],params=json.dumps(job['params']),
                           slot_mix=json.dumps(job['slot_mix']),instance=i,occurrences=sum(task.target.values()),
                           start_cost=cost(arena,start),C_gen=task.c_gen,r_max=r_max,n_max=calc.n_max)
                for name,actions in (('full',full),('sample',sample)):
                    successors = [apply(arena,task,info,start,a) for a in actions]
                    live = [s for s in successors if s is not None]
                    row.update({f'{name}_candidates':len(actions),f'{name}_dead':successors.count(None),
                                f'{name}_distinct_successors':len(set(live)),
                                f'{name}_best_gain':max((cost(arena,start)-cost(arena,s) for s in live),default=None)})
                rows.append(row)
    _write_csv(rows,output/'truncation_diagnostic.csv')
    return rows


if __name__ == '__main__':
    main()
