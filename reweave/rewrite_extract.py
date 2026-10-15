"""Ground-pair extraction from successful exact-search trajectories.

For every window of 2-4 consecutive decisions on a trajectory, the window is replayed locally from its context (and
up to two enclosing Sum contexts); if the local replay reproduces the recorded successor, the changed children give a
ground pair (R -> T) with poly(R) == poly(T). Pairs whose target reuses pre-existing compound structure are marked
`requires_ref`; they are the evidence that compound (handle-bound) parameters are needed.
"""

from collections import Counter

from . import v1

# Unmemoized inherited `calculus._apply` (the executed MVP-1 local-replay semantics). Reached through the declared
# adapter table rather than a new adapter symbol.
V1_APPLY_UNMEMOIZED = v1.MODULES['calculus']._apply

REASONS = ('local_apply_failed', 'final_state_mismatch', 'empty_or_identity', 'ground_size_bound', 'duplicate')


def chain_and_decisions(record: dict):
    arena, ids = v1.rebuild_record(record)
    arena.calc_cfg = v1.CalcCfg()
    chain = [ids[s['root']] for s in record['labelled']]+[ids[record['incumbent']]]
    decisions = [v1.action_from_record(s['actions'][s['positive']], ids) for s in record['labelled']]
    return arena, chain, decisions


def parent_map(arena, root: int) -> dict:
    par: dict = {}
    for h in v1.state_info(arena, root).reach_h:
        for c, n in v1.items(arena.handles[h]):
            par.setdefault(c, set()).add(h)
    return par


def sum_ancestors(arena, root: int, ctx: int, limit: int) -> list:
    """Nearest Sum ancestors of `ctx` in `root`, by upward breadth-first level then digest (lowest first)."""
    par = parent_map(arena, root)
    out, frontier, seen = [], [ctx], {ctx}
    while frontier and len(out) < limit:
        level = sorted({p for h in frontier for p in par.get(h, ()) if p not in seen}, key=lambda h: v1.digest(arena, h))
        seen.update(level)
        for p in level:
            if isinstance(arena.handles[p], v1.Sum) and len(out) < limit:
                out.append(p)
        frontier = level
    return out


def ground(arena, h: int, pre: frozenset) -> list:
    """Ground term: ['M', exps] or [kind, [[child, count], ...], pre_existing]."""
    node = arena.handles[h]
    if isinstance(node, v1.Mono):
        return ['M', list(v1.unpack(node.m))]
    kind = 'S' if isinstance(node, v1.Sum) else 'P'
    kids = sorted(v1.items(node), key=lambda hc: v1.digest(arena, hc[0]))
    return [kind, [[ground(arena, c, pre), n] for c, n in kids], h in pre]


def _new_structure(arena, h: int, pre: frozenset, new_handles: set, leaves: set) -> None:
    node = arena.handles[h]
    if isinstance(node, v1.Mono):
        leaves.add(node.m)
        return
    if h in pre:
        return
    new_handles.add(h)
    for c, n in v1.items(node):
        _new_structure(arena, c, pre, new_handles, leaves)


def extract_task(record: dict, row: dict, cfg) -> tuple[list, Counter]:
    counts: Counter = Counter()
    pairs = []
    if any(s.get('positive') is None for s in record['labelled']):
        counts['record_without_positive'] += 1
        return pairs, counts
    arena, chain, decisions = chain_and_decisions(record)
    n = len(decisions)
    seen: set = set()
    reach_cache: dict = {}
    for k in range(n):
        S_k = chain[k]
        if k not in reach_cache:
            reach_cache[k] = frozenset(v1.state_info(arena, S_k).reach_h)
        pre = reach_cache[k]
        for L in range(cfg.L_min, cfg.L_max+1):
            if k+L > n:
                break
            counts['windows'] += 1
            ctx0 = decisions[k].ctx
            candidates = [ctx0]+sum_ancestors(arena, S_k, ctx0, cfg.anc_max)
            accepted = False
            for C in candidates:
                counts['attempts'] += 1
                image = C
                failed = False
                for j in range(k, k+L):
                    try:
                        image = V1_APPLY_UNMEMOIZED(arena, v1.state_info(arena, image), image, decisions[j])
                    except Exception:
                        image = None
                    if image is None:
                        failed = True
                        break
                if failed:
                    counts['reject.local_apply_failed'] += 1
                    continue
                if v1.substitute(arena, S_k, C, image) != chain[k+L]:
                    counts['reject.final_state_mismatch'] += 1
                    continue
                M = Counter(dict(v1.items(arena.handles[C])))
                M2 = Counter(dict(v1.items(arena.handles[image]))) if isinstance(arena.handles[image], v1.Sum) else Counter({image: 1})
                U = M & M2
                R, T = M-U, M2-U
                if not R or not T or R == T:
                    counts['reject.empty_or_identity'] += 1
                    continue
                pR = {}
                for h, c in R.items():
                    pR = v1.poly_add(pR, v1.poly_scale(v1.hpoly(arena, h), c))
                pT = {}
                for h, c in T.items():
                    pT = v1.poly_add(pT, v1.poly_scale(v1.hpoly(arena, h), c))
                assert v1.poly_equal(pR, pT), 'ground pair polynomial mismatch (real bug)'
                new_handles, leaves = set(), set()
                for h in T:
                    _new_structure(arena, h, pre, new_handles, leaves)
                if (len(R) > cfg.gp_source_children or sum(R.values()) > cfg.gp_source_multiplicity
                        or len(T) > cfg.gp_target_children or len(new_handles) > cfg.gp_target_new_handles
                        or len(leaves) > cfg.gp_target_mono_leaves):
                    counts['reject.ground_size_bound'] += 1
                    continue
                dkey = (tuple(sorted((v1.digest(arena, h), c) for h, c in R.items())),
                        tuple(sorted((v1.digest(arena, h), c) for h, c in T.items())))
                accepted = True
                if dkey in seen:
                    counts['reject.duplicate'] += 1
                    break
                seen.add(dkey)
                root = chain[k]
                prov = ('root_whole' if not U else 'root_partial') if C == root else 'interior'
                nodes, index = v1.serialize(arena, [S_k], with_digests=False)
                pairs.append(dict(
                    task_id=record['task_id'], family=row['family'], shape=row['shape'], k=k, L=L,
                    kinds=[decisions[j].kind if decisions[j].src is None or decisions[j].kind != 'REUSE' else 'REUSE_SRC'
                           for j in range(k, k+L)],
                    provenance=prov, C_digest=v1.digest(arena, C).hex(), C_pos=index[C], S_k=nodes, S_k_root=index[S_k],
                    R=[[ground(arena, h, pre), c] for h, c in sorted(R.items(), key=lambda hc: v1.digest(arena, hc[0]))],
                    T=[[ground(arena, h, pre), c] for h, c in sorted(T.items(), key=lambda hc: v1.digest(arena, hc[0]))],
                    R_multiplicity=sum(R.values()), T_multiplicity=sum(T.values()),
                    requires_ref=any(h in pre and not isinstance(arena.handles[h], v1.Mono)
                                     for t in T for h in _pre_nonmono(arena, t, pre)),
                    canon_R=repr(v1.canon(pR))))
                counts[f'accepted.L{L}'] += 1
                counts[f'accepted.{prov}'] += 1
                break
            counts['windows_with_pair' if accepted else 'windows_without_pair'] += 1
    return pairs, counts


def _pre_nonmono(arena, h: int, pre: frozenset) -> list:
    """Pre-existing non-Mono handles inside the new structure of a target child (these would need ref holes)."""
    node = arena.handles[h]
    if isinstance(node, v1.Mono):
        return []
    if h in pre:
        return [h]
    return [x for c, n in v1.items(node) for x in _pre_nonmono(arena, c, pre)]


def pair_digest(pair: dict) -> str:
    from .hashing import sha256_json
    return sha256_json(pair)
