"""Proposal of parameterized relations from ground pairs (monomial parameters).

Ground pairs are bucketed by target skeleton and source shape. For each seed pair, a basis of <= 4 target monomial
leaves is chosen and every leaf is decomposed over the basis; each decomposition yields a candidate relation over
parameters, which is normalized, checked against the structural bounds and the binding plan, and certified exactly.
Certified candidates are deduplicated by `schema_hash`.
"""

import itertools
from collections import Counter, defaultdict

from .hashing import canonical_json
from .rewrite_certify import bounds_violation, certify
from .rewrite_schema import NV, Schema, binding_plan, canonicalize


def _skel(g):
    if g[0] == 'M':
        return 'mono'
    if g[2]:
        return 'handle'
    return [g[0], sorted(([_skel(c), n] for c, n in g[1]), key=canonical_json)]


def _kind(g) -> str:
    return 'mono' if g[0] == 'M' else ('sum' if g[0] == 'S' else 'prod')


def bucket_key(pair: dict) -> str:
    return canonical_json([sorted(([_skel(g), n] for g, n in pair['T']), key=canonical_json),
                           sorted(n for g, n in pair['R']), sorted(_kind(g) for g, n in pair['R'])])


def _target_leaves(g, out: set) -> None:
    if g[0] == 'M':
        out.add(tuple(g[1]))
        return
    if g[2]:
        return
    for c, n in g[1]:
        _target_leaves(c, out)


def _all_leaves(g, out: set) -> None:
    if g[0] == 'M':
        out.add(tuple(g[1]))
        return
    for c, n in g[1]:
        _all_leaves(c, out)


def _pack(v) -> int:
    return sum(e << (4*i) for i, e in enumerate(v))


def decompositions(v: tuple, basis: list, emax: int, cap: int, counts: Counter) -> list:
    out = []
    for e in itertools.product(range(emax+1), repeat=len(basis)):
        if not any(e):
            continue
        if all(sum(e[b]*basis[b][j] for b in range(len(basis))) == v[j] for j in range(NV)):
            out.append(e)
            if len(out) > cap:
                counts['proposal_cut.decompositions_per_term'] += 1
                return out[:cap]
    return out


def _pattern(g, decomp: dict):
    if g[0] == 'M':
        e = decomp[tuple(g[1])]
        return ('M', tuple((i, x) for i, x in enumerate(e) if x))
    return (g[0], tuple((_pattern(c, decomp), n) for c, n in g[1]))


def propose_from_seed(pair: dict, cfg, counts: Counter) -> list[tuple[Schema, str]]:
    """Return [(schema, status)] for every candidate this seed produced."""
    if pair['requires_ref']:
        counts['proposal.requires_ref'] += 1
        return []
    tl: set = set()
    for g, n in pair['T']:
        _target_leaves(g, tl)
    sl: set = set()
    for g, n in pair['R']:
        _all_leaves(g, sl)
    tleaves = sorted(tl, key=_pack)
    out = []
    tested = valid = 0
    bases = []
    for size in range(1, min(cfg.basis_size_max, len(tleaves))+1):
        for subset in itertools.combinations(tleaves, size):
            if tested >= cfg.basis_subsets_tested:
                counts['proposal_cut.basis_subsets_tested'] += 1
                break
            tested += 1
            basis = list(subset)
            others = [v for v in tleaves if v not in subset]
            if all(decompositions(v, basis, cfg.basis_exponent_max, 1, Counter()) for v in others):
                bases.append(basis)
                if len(bases) >= cfg.bases_per_seed:
                    break
        if len(bases) >= cfg.bases_per_seed or tested >= cfg.basis_subsets_tested:
            if len(bases) >= cfg.bases_per_seed:
                counts['proposal_cut.bases_per_seed'] += 1
            break
    if not bases:
        counts['proposal.no_basis'] += 1
    produced = 0
    for basis in bases:
        leaf_values = sorted(set(tleaves) | sl, key=_pack)
        options = []
        ok = True
        for v in leaf_values:
            if v in basis:
                # a basis value is its own hole wherever it occurs (source and target share one decomposition)
                ds = [tuple(1 if basis[b] == v else 0 for b in range(len(basis)))]
            else:
                ds = decompositions(v, basis, cfg.basis_exponent_max, cfg.decompositions_per_term, counts)
            if not ds:
                ok = False
                break
            options.append(ds)
        if not ok:
            counts['proposal.no_decomposition'] += 1
            continue
        for choice in itertools.product(*options):
            if produced >= cfg.candidates_per_seed:
                counts['proposal_cut.candidates_per_seed'] += 1
                return out
            produced += 1
            decomp = dict(zip(leaf_values, choice))
            src = tuple((_pattern(g, decomp), n) for g, n in pair['R'])
            tgt = tuple((_pattern(g, decomp), n) for g, n in pair['T'])
            sch = canonicalize(tuple(['mon']*len(basis)), src, tgt)
            bad = bounds_violation(sch, cfg)
            if bad:
                out.append((sch, f'schema_bound.{bad}'))
                continue
            plan = binding_plan(sch)
            if plan is None:
                out.append((sch, 'no_binding_plan'))
                continue
            if plan == 'simultaneous':
                counts['candidate.simultaneous_binding_plan'] += 1
            cert, status, polys = certify(sch)
            out.append((sch, status))
    return out


def propose(pairs: list[dict], cfg) -> tuple[dict, Counter, dict]:
    counts: Counter = Counter()
    freq = Counter(p['canon_R'] for p in pairs)
    buckets = defaultdict(list)
    for idx, p in enumerate(pairs):
        buckets[bucket_key(p)].append(idx)
    order = sorted(buckets, key=lambda b: (-len(buckets[b]), b))
    schemas: dict = {}
    seed_log: dict = {}
    total = 0
    for bi, b in enumerate(order):
        members = sorted(buckets[b], key=lambda i: (-freq[pairs[i]['canon_R']], pairs[i]['task_id'], pairs[i]['k'], pairs[i]['L']))
        seeds = members[:cfg.seeds_per_bucket]
        if len(members) > cfg.seeds_per_bucket:
            counts['proposal_cut.seeds_per_bucket'] += len(members)-cfg.seeds_per_bucket
        per_bucket = 0
        for i in seeds:
            if total >= cfg.candidates_total:
                counts['proposal_cut.candidates_total_seeds_skipped'] += 1
                seed_log[i] = dict(bucket=bi, skipped='candidates_total')
                continue
            if per_bucket >= cfg.candidates_per_bucket:
                counts['proposal_cut.candidates_per_bucket_seeds_skipped'] += 1
                seed_log[i] = dict(bucket=bi, skipped='candidates_per_bucket')
                continue
            results = propose_from_seed(pairs[i], cfg, counts)
            per_bucket += len(results)
            total += len(results)
            log = dict(bucket=bi, candidates=len(results), statuses=Counter(st for s, st in results),
                       hashes=sorted({s.schema_hash for s, st in results if st == 'certified'}))
            seed_log[i] = log
            for sch, st in results:
                counts[f'candidate.{st}'] += 1
                if st != 'certified':
                    continue
                entry = schemas.setdefault(sch.schema_hash, dict(schema=sch.to_json(), seeds=[]))
                entry['seeds'].append(dict(task_id=pairs[i]['task_id'], k=pairs[i]['k'], L=pairs[i]['L'], bucket=bi))
    counts['buckets'] = len(buckets)
    counts['certified_unique'] = len(schemas)
    return schemas, counts, dict(bucket_order=order, seed_log={str(k): dict(v, statuses=dict(v.get('statuses', {})))
                                                                 for k, v in seed_log.items()})
