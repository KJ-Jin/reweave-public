"""From certified relations to a library: exact re-match support, alias classification, eligibility.

Every certified relation is re-matched exactly against the ground-pair pool. Support counts distinct tasks, distinct
canonical target shapes and distinct binding signatures over non-alias instances. An instance is an alias when one
primitive action already produces the same successor (bounded three-state check: alias / unknown / non-alias).
Eligibility: task_support >= 5, shape_support >= 3, binding_signature_support >= 2, not an alias relation (thresholds
are configuration). Fold libraries (task_id % 5) support out-of-fold training labels.
"""

import dataclasses
import json
import os
import signal
import time
from collections import Counter, defaultdict
from multiprocessing import get_context

from . import v1

# Unmemoized inherited `calculus._apply` (the executed MVP-1 local-replay semantics). Reached through the declared
# adapter table rather than a new adapter symbol.
V1_APPLY_UNMEMOIZED = v1.MODULES['calculus']._apply
from . import rewrite_schema as S
from .hashing import sha256_json
from .rewrite_certify import target_unique_ops
from .rewrite_extract import chain_and_decisions, extract_task, parent_map
from .rewrite_propose import propose

N_FOLDS = 5
LIBRARY_VERSION = 'rewrite-library-1'


class LibraryCapExceeded(RuntimeError):
    """More eligible relations than active_cap: library construction stops (there is no selection rule)."""


def fold_of(task_id: int) -> int:
    return task_id % N_FOLDS


# ---------------------------------------------------------------- re-match and alias (MVP-1 §4)


class AliasTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise AliasTimeout()


def rebuild_pair(arena, chain, decisions, pair: dict):
    """Recompute (S_k, C, image, R, T) for a recorded pair by the same local replay as extraction."""
    k, L = pair['k'], pair['L']
    S_k = chain[k]
    C = next(h for h in v1.state_info(arena, S_k).reach_h if v1.digest(arena, h).hex() == pair['C_digest'])
    image = C
    for j in range(k, k+L):
        image = V1_APPLY_UNMEMOIZED(arena, v1.state_info(arena, image), image, decisions[j])
    M = Counter(dict(v1.items(arena.handles[C])))
    M2 = Counter(dict(v1.items(arena.handles[image]))) if isinstance(arena.handles[image], v1.Sum) else Counter({image: 1})
    U = M & M2
    return S_k, C, image, M-U, M2-U


def rematch(sch: S.Schema, arena, R: Counter, T: Counter, cfg) -> tuple[str, list]:
    """Exact region-anchored re-match: all bindings with R_inst == R whose lowered target equals T."""
    budget = S.Budget(cfg.rematch_unify_states_max)
    index = S.ChildIndex(arena, dict(R))
    valid = []
    try:
        for beta in S.assign(sch.source, index, dict(R), {}, arena, cfg, budget, exact=True, simultaneous=S.uses_simultaneous(sch)):
            try:
                if S.lower_region(sch.target, beta, arena) == T:
                    valid.append(dict(beta))
            except (S.CapError, AssertionError):
                continue
    except S.Cut:
        return 'rematch_incomplete', []
    return ('instance' if valid else 'no_match'), valid


def alias_status(arena, S_k: int, C: int, target: int, cfg) -> tuple[str, dict]:
    info = v1.state_info(arena, S_k)
    reach_c = v1.state_info(arena, C).reach_h
    par = parent_map(arena, S_k)
    anc, frontier = set(), [C]
    while frontier:
        nxt = []
        for h in frontier:
            for p in par.get(h, ()):
                if p not in anc:
                    anc.add(p)
                    nxt.append(p)
        frontier = nxt
    contexts = [h for h in info.contexts if h in reach_c or h in anc]
    rinfo = dataclasses.replace(info, contexts=contexts)
    calc = dataclasses.replace(arena.calc_cfg, n_max=cfg.alias_n_max)
    began = time.perf_counter()
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, cfg.alias_time_cap_s)
    n_actions, truncated = 0, False
    try:
        before = len(arena.truncations)
        actions = v1.enumerate_actions(arena, rinfo, S_k, calc)
        n_actions = len(actions)
        truncated = len(arena.truncations) > before or n_actions >= cfg.alias_n_max
        for a in actions:
            try:
                succ = V1_APPLY_UNMEMOIZED(arena, info, S_k, a)
            except AssertionError:
                succ = None
            if succ == target:
                return 'primitive_alias', dict(seconds=time.perf_counter()-began, actions=n_actions, contexts=len(contexts))
        status = 'alias_unknown' if truncated else 'non_alias'
        return status, dict(seconds=time.perf_counter()-began, actions=n_actions, contexts=len(contexts), truncated=truncated)
    except AliasTimeout:
        return 'alias_unknown', dict(seconds=time.perf_counter()-began, actions=n_actions, contexts=len(contexts), timeout=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def support_task(record: dict, row: dict, pairs: list, schemas_by_mult: dict, cfg) -> tuple[list, list, Counter]:
    """For one task: re-match every candidate schema on every pair; alias-check pairs with >= 1 instance."""
    counts: Counter = Counter()
    arena, chain, decisions = chain_and_decisions(record)
    instances, pair_rows = [], []
    for pi, pair in pairs:
        S_k, C, image, R, T = rebuild_pair(arena, chain, decisions, pair)
        found = []
        for sch in schemas_by_mult.get((pair['R_multiplicity'], pair['T_multiplicity']), ()):
            status, valid = rematch(sch, arena, R, T, cfg)
            counts[f'rematch.{status}'] += 1
            if status == 'rematch_incomplete':
                found.append((sch.schema_hash, status, None, False))
            elif status == 'instance':
                sigs = sorted({S.normalized_signature(sch, b) for b in valid})
                found.append((sch.schema_hash, status, sigs[0], len(sigs) > 1))
        alias, info = (None, {})
        if any(st == 'instance' for h, st, sig, amb in found):
            alias, info = alias_status(arena, S_k, C, chain[pair['k']+pair['L']], cfg)
            counts[f'alias.{alias}'] += 1
        pair_rows.append(dict(pair_index=pi, task_id=pair['task_id'], alias=alias, alias_info=info))
        for h, st, sig, amb in found:
            instances.append(dict(schema_hash=h, pair_index=pi, task_id=pair['task_id'], shape=pair['shape'],
                                  family=pair['family'], k=pair['k'], L=pair['L'], provenance=pair['provenance'],
                                  status=st, signature=sig, ambiguous=amb, alias=alias))
    return instances, pair_rows, counts


def aggregate_support(schemas: dict, instances: list) -> dict:
    by = defaultdict(list)
    for r in instances:
        by[r['schema_hash']].append(r)
    out = {}
    for h in schemas:
        rs = by.get(h, [])
        good = [r for r in rs if r['status'] == 'instance' and r['alias'] == 'non_alias']
        sigs = sorted({json.dumps(r['signature']) for r in good})
        entry = dict(
            instances=sum(r['status'] == 'instance' for r in rs),
            rematch_incomplete=sum(r['status'] == 'rematch_incomplete' for r in rs),
            alias=Counter(r['alias'] for r in rs if r['status'] == 'instance'),
            task_support=len({r['task_id'] for r in good}), shape_support=len({r['shape'] for r in good}),
            binding_signature_support=len(sigs), signatures=sigs,
            families=Counter(r['family'] for r in good), provenance=Counter(r['provenance'] for r in good),
            ambiguous=sum(r['ambiguous'] for r in good))
        a = entry['alias']
        entry['is_alias'] = a.get('non_alias', 0) == 0 and a.get('primitive_alias', 0) >= 1
        entry['support_2_2_2'] = entry['task_support'] >= 2 and entry['shape_support'] >= 2 and entry['binding_signature_support'] >= 2
        entry['support_5_3_3'] = entry['task_support'] >= 5 and entry['shape_support'] >= 3 and entry['binding_signature_support'] >= 3
        entry['alias'] = dict(a)
        entry['families'] = dict(entry['families'])
        entry['provenance'] = dict(entry['provenance'])
        out[h] = entry
    return out


def eligible(entry: dict, cfg) -> bool:
    return (not entry['is_alias'] and entry['task_support'] >= cfg.task_support_min
            and entry['shape_support'] >= cfg.shape_support_min
            and entry['binding_signature_support'] >= cfg.binding_signature_support_min)


# ---------------------------------------------------------------- pools and hashes


def pair_pool_hash(pairs: list) -> str:
    return sha256_json(pairs)


def certified_schema_pool_hash(schemas: dict) -> str:
    return sha256_json({h: schemas[h]['schema'] for h in sorted(schemas)})


def schema_features(schema: S.Schema, support_entry: dict) -> list[float]:
    """(target unique ops, n_holes, log1p(task_support)); task_support is library-valued."""
    import math
    return [float(target_unique_ops(schema)), float(len(schema.types)), math.log1p(support_entry['task_support'])]


# ---------------------------------------------------------------- parallel helpers (fork; module-level workers)


_G: dict = {}


def _imap(fn, items, workers: int):
    if workers <= 1:
        for x in items:
            yield fn(x)
        return
    with get_context('fork').Pool(workers) as p:
        yield from p.imap_unordered(fn, items, chunksize=1)


def _extract_worker(item):
    rec, row = item
    pairs, counts = extract_task(rec, row, _G['cfg'])
    return pairs, counts


def _support_worker(item):
    rec, row, task_pairs = item
    return support_task(rec, row, task_pairs, _G['by_mult'], _G['cfg'])


def extract_pairs(items: list, cfg, workers: int = 1) -> tuple[list, Counter]:
    """items: [(record, row)] of train_fit tasks. Pairs sorted by (task_id, k, L)."""
    _G['cfg'] = cfg
    pairs, counts = [], Counter()
    for p, c in _imap(_extract_worker, items, workers):
        pairs += p
        counts.update(c)
    pairs.sort(key=lambda p: (p['task_id'], p['k'], p['L']))
    return pairs, counts


def run_support(items: list, pairs: list, schemas: dict, cfg, workers: int = 1) -> tuple[list, list, Counter]:
    by_mult = defaultdict(list)
    for h in sorted(schemas):
        sch = S.Schema.from_json(schemas[h]['schema'])
        by_mult[(sum(c for p, c in sch.source), sum(c for p, c in sch.target))].append(sch)
    _G['by_mult'], _G['cfg'] = dict(by_mult), cfg
    by_task = defaultdict(list)
    for i, p in enumerate(pairs):
        by_task[p['task_id']].append((i, p))
    work = [(rec, row, by_task[rec['task_id']]) for rec, row in items if rec['task_id'] in by_task]
    work.sort(key=lambda x: -len(x[2]))
    instances, prows, counts = [], [], Counter()
    for inst, prow, c in _imap(_support_worker, work, workers):
        instances += inst
        prows += prow
        counts.update(c)
    order = lambda r: (r['pair_index'], r.get('schema_hash', ''))
    instances.sort(key=order)
    prows.sort(key=lambda r: r['pair_index'])
    return instances, prows, counts


# ---------------------------------------------------------------- library construction


def library_body(name: str, task_ids: list, pairs: list, schemas: dict, support: dict, cfg, counts: dict) -> dict:
    elig = sorted(h for h in schemas if eligible(support[h], cfg))
    body = dict(version=LIBRARY_VERSION, name=name, config_hash=cfg.hash(),
                eligibility=dict(task_support=cfg.task_support_min, shape_support=cfg.shape_support_min,
                                 binding_signature_support=cfg.binding_signature_support_min, alias_excluded=True),
                active_cap=cfg.active_cap, train_task_ids_hash=sha256_json(sorted(task_ids)), n_tasks=len(task_ids),
                pair_pool_hash=pair_pool_hash(pairs), n_pairs=len(pairs),
                certified_schema_pool_hash=certified_schema_pool_hash(schemas), n_certified=len(schemas),
                eligible=elig, schemas={h: schemas[h]['schema'] for h in elig},
                support={h: {k: support[h][k] for k in ('task_support', 'shape_support', 'binding_signature_support',
                                                        'families', 'provenance', 'alias', 'instances')} for h in elig},
                counts=counts)
    body['library_hash'] = sha256_json(dict(version=LIBRARY_VERSION, eligible=elig, schemas=body['schemas'],
                                            support={h: {k: body['support'][h][k] for k in ('task_support', 'shape_support', 'binding_signature_support')}
                                                     for h in elig}, eligibility=body['eligibility'], config_hash=body['config_hash']))
    if len(elig) > cfg.active_cap:
        raise LibraryCapExceeded(f'{name}: {len(elig)} eligible schemas > active_cap {cfg.active_cap}')
    return body


def build_library(items: list, cfg, name: str, workers: int = 1) -> tuple[dict, dict]:
    """The whole pipeline on exactly `items` ([(record, row)] of train_fit tasks). Returns (library, artifacts)."""
    pairs, ex_counts = extract_pairs(items, cfg, workers)
    schemas, prop_counts, logs = propose(pairs, cfg)
    instances, prows, sup_counts = run_support(items, pairs, schemas, cfg, workers)
    support = aggregate_support(schemas, instances)
    counts = dict(extract=dict(ex_counts), propose=dict(prop_counts), support=dict(sup_counts))
    lib = library_body(name, [rec['task_id'] for rec, row in items], pairs, schemas, support, cfg, counts)
    return lib, dict(pairs=pairs, schemas=schemas, support=support, instances=instances, pair_rows=prows)


def build_fold_libraries(items: list, cfg, workers: int = 1) -> tuple[dict, dict]:
    """`L_all` and every `L_-f` with shared per-task extraction and a shared (pair, schema) support computation.

    Identical to calling `build_library` separately on each task subset: extraction is per task; re-match is per
    (pair, schema); alias status is a function of the pair only; each library aggregates only its own pairs and its
    own certified schemas (tested)."""
    all_pairs, ex_counts = extract_pairs(items, cfg, workers)
    subsets = {'all': None}
    subsets.update({f'minus{f}': f for f in range(N_FOLDS)})
    proposals = {}
    for name, f in subsets.items():
        pairs = [p for p in all_pairs if f is None or fold_of(p['task_id']) != f]
        schemas, prop_counts, logs = propose(pairs, cfg)
        proposals[name] = (pairs, schemas, prop_counts)
    union: dict = {}
    for name in subsets:
        for h, e in proposals[name][1].items():
            union.setdefault(h, dict(schema=e['schema'], seeds=[]))
    instances, prows, sup_counts = run_support(items, all_pairs, union, cfg, workers)
    index_of = {(p['task_id'], p['k'], p['L']): i for i, p in enumerate(all_pairs)}
    libraries, artifacts = {}, dict(all_pairs=all_pairs, union=sorted(union))
    for name, f in subsets.items():
        pairs, schemas, prop_counts = proposals[name]
        local = {index_of[(p['task_id'], p['k'], p['L'])]: i for i, p in enumerate(pairs)}
        inst = [dict(r, pair_index=local[r['pair_index']]) for r in instances
                if r['pair_index'] in local and r['schema_hash'] in schemas]
        support = aggregate_support(schemas, inst)
        task_ids = [rec['task_id'] for rec, row in items if f is None or fold_of(rec['task_id']) != f]
        counts = dict(propose=dict(prop_counts), extract_all=dict(ex_counts) if f is None else None)
        libraries[name] = library_body(name, task_ids, pairs, schemas, support, cfg, counts)
        artifacts[name] = dict(schemas=schemas, support=support)
    artifacts['support_counts'] = dict(sup_counts)
    artifacts['instances'] = instances
    artifacts['pair_rows'] = prows
    return libraries, artifacts


def vocabulary(libraries: dict) -> list[str]:
    """Schema embedding vocabulary: sorted union of L_all and every L_-f."""
    return sorted({h for lib in libraries.values() for h in lib['eligible']})


def library_for_task(libraries: dict, task_id: int) -> dict:
    return libraries[f'minus{fold_of(task_id)}']


def deploy_library(library_all: dict, oof_label_counts: dict) -> dict:
    """L_deploy = {s in L_all : OOF_label_count(s) > 0} (a relation with no out-of-fold label is never deployed)."""
    keep = [h for h in library_all['eligible'] if oof_label_counts.get(h, 0) > 0]
    removed = [h for h in library_all['eligible'] if oof_label_counts.get(h, 0) == 0]
    body = dict(library_all, name='deploy', eligible=keep, schemas={h: library_all['schemas'][h] for h in keep},
                support={h: library_all['support'][h] for h in keep}, removed_zero_oof=removed,
                removed_support={h: {k: library_all['support'][h][k] for k in ('task_support', 'shape_support', 'binding_signature_support')} for h in removed},
                oof_label_counts={h: oof_label_counts.get(h, 0) for h in library_all['eligible']},
                all_library_hash=library_all['library_hash'])
    body['library_hash'] = sha256_json(dict(kind='deploy', all_library_hash=library_all['library_hash'], eligible=keep))
    return body


def schemas_of(library: dict) -> list[S.Schema]:
    return [S.Schema.from_json(library['schemas'][h]) for h in library['eligible']]
