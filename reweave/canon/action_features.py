"""Scorer-facing features of an ActionRecord (no learned component).

The schema-agnostic scorer reads a rewrite action only through its `ActionRecord` and the state it is scored in.
Handles are resolved from the record's STRUCTURAL digests (never from arena ids, creator ids, surface ids, family ids
or allocation order), and every feature is a deterministic function of the exact canonical relation (family
structure), the parameter values in canonical order, the latent multiplier d, the concrete lowered target, the
realizing structure of each parameter, and structural lookups in the state.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass

from .. import rewrite_schema as S
from .. import v1

T7_FEATURE_VERSION = 't7-features-v1'
KINDS = ('PRIMITIVE', 'VISIBLE_REWRITE', 'LATENT_REWRITE', 'VIEW_REWRITE')
NV = S.NV

FAMILY_FIELDS = (
    'arity',
    *(f'{side}_{q}' for side in ('src', 'tgt') for q in (
        'top_children', 'top_multiplicity', 'n_sum', 'n_prod', 'n_leaf', 'depth', 'max_multiplicity',
        'repeated_occurrences', 'max_leaf_exponent', 'multi_param_leaves', 'formal_gates', 'formal_add',
        'formal_mul')),
    'relation_terms', 'relation_degree', 'relation_min_degree', 'formal_compression')
ROLE_FIELDS = ('src_occurrences', 'src_exponent', 'src_max_exponent', 'tgt_occurrences', 'tgt_exponent',
               'tgt_max_exponent', 'relation_degree', 'relation_term_share', 'co_occurring_roles')
VALUE_FIELDS = ('terms', 'max_degree', 'min_degree', 'support', 'max_exponent', 'coef_sum', 'coef_max',
                'is_monomial')
REALIZATION_FIELDS = ('existing_handle', 'in_state_graph', 'cost', 'nodes', 'depth', 'fanout', 'shared')
D_FIELDS = ('degree', 'support', 'max_exponent', 'exp_1', 'exp_2', 'exp_3')
CONCRETE_FIELDS = ('region_children', 'region_multiplicity', 'source_gates', 'source_gates_not_reused',
                   'target_nodes', 'target_gates_isolated', 'target_gates_new', 'target_existing_fraction',
                   'reused_handles', 'context_gates_before', 'context_gates_after', 'context_collapses',
                   'context_exists', 'local_delta', 'target_depth')
N_FAMILY, N_ROLE = len(FAMILY_FIELDS), len(ROLE_FIELDS)+len(VALUE_FIELDS)+len(REALIZATION_FIELDS)
N_D, N_CONCRETE = len(D_FIELDS), len(CONCRETE_FIELDS)


def _l(x: float) -> float:
    return math.log1p(x)


def _sl(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x)


@dataclass
class ActionFeatures:
    """One rewrite action's scorer inputs in one state. Handle ids are this arena's, resolved from digests."""
    kind: int
    family: tuple                 # N_FAMILY floats — FAMILY STRUCTURE (payload only)
    roles: tuple                  # per canonical role: N_ROLE floats
    role_nodes: tuple             # per canonical role: ('h', handle id) state-graph node | ('m', packed) off-graph
    d: tuple                      # N_D floats (zeros for a visible action)
    concrete: tuple               # N_CONCRETE floats
    ctx: int
    region: tuple                 # ((handle id, count), ...)


# ---------------------------------------------------------------- family structure (payload only)


_FAMILY: dict = {}


def _walk(region, mult: int, depth: int, acc: dict, occ: list, exp: list, maxexp: list) -> int:
    """Tree statistics of an AC-normalized pattern region (multiplicities multiplied along paths)."""
    deepest = depth
    for node, n in region:
        m = mult*n
        acc['max_multiplicity'] = max(acc['max_multiplicity'], n)
        acc['repeated_occurrences'] += n-1
        if node[0] == 'M':
            acc['n_leaf'] += m
            acc['max_leaf_exponent'] = max([acc['max_leaf_exponent']]+[e for i, e in node[1]])
            acc['multi_param_leaves'] += m*(len(node[1]) > 1)
            for i, e in node[1]:
                occ[i] += m
                exp[i] += m*e
                maxexp[i] = max(maxexp[i], e)
            deepest = max(deepest, depth+1)
        else:
            acc['n_sum' if node[0] == 'S' else 'n_prod'] += m
            deepest = max(deepest, _walk(node[1], m, depth+1, acc, occ, exp, maxexp))
    return deepest


def _gate_ops(arena, h: int) -> tuple:
    v1.cost(arena, h)
    ops = Counter(arena.gates[g].op for g in arena.gate_reach_memo[h])
    return ops['ADD'], ops['MUL']


def family_features(relation) -> tuple:
    """(family vector, per-role vectors) of an exact canonical relation payload ["ac-relation-v1", k, src, tgt].
    Memoized by the payload itself (exact key)."""
    key = json.dumps(relation, sort_keys=True, separators=(',', ':'))
    hit = _FAMILY.get(key)
    if hit is not None:
        return hit
    version, k, src_j, tgt_j = relation
    src, tgt = S.deser_region(src_j), S.deser_region(tgt_j)
    arena = v1.Arena()
    arena.calc_cfg = v1.CalcCfg()
    beta = {i: tuple(1 if j == i else 0 for j in range(NV)) for i in range(k)}
    fam = [float(k)]
    per_side = {}
    for side, region in (('src', src), ('tgt', tgt)):
        acc = Counter(max_multiplicity=0, repeated_occurrences=0, max_leaf_exponent=0)
        occ, exp, maxexp = [0]*k, [0]*k, [0]*k
        depth = _walk(region, 1, 0, acc, occ, exp, maxexp)
        top = S.lower_region(region, beta, arena)
        h = v1.mk_sum(arena, Counter(top))
        add, mul = _gate_ops(arena, h)
        per_side[side] = dict(h=h, occ=occ, exp=exp, maxexp=maxexp, gates=v1.cost(arena, h))
        fam += [_l(len(region)), _l(sum(n for c, n in region)), _l(acc['n_sum']), _l(acc['n_prod']),
                _l(acc['n_leaf']), float(depth), float(acc['max_multiplicity']), _l(acc['repeated_occurrences']),
                float(acc['max_leaf_exponent']), _l(acc['multi_param_leaves']), _l(per_side[side]['gates']),
                _l(add), _l(mul)]
    poly = v1.hpoly(arena, per_side['src']['h'])
    degs = [sum(v1.unpack(m)) for m in poly]
    fam += [_l(len(poly)), float(max(degs)), float(min(degs)),
            _sl(per_side['src']['gates']-per_side['tgt']['gates'])]
    co = [set() for _ in range(k)]
    for node, n in S.norm_region(src):
        for leaf in _leaves(node):
            ids = [i for i, e in leaf[1]]
            for i in ids:
                co[i] |= set(ids)-{i}
    roles = []
    for i in range(k):
        s, t = per_side['src'], per_side['tgt']
        deg_i = max(v1.unpack(m)[i] for m in poly)
        share = sum(1 for m in poly if v1.unpack(m)[i])/len(poly)
        roles.append((_l(s['occ'][i]), _l(s['exp'][i]), float(s['maxexp'][i]), _l(t['occ'][i]), _l(t['exp'][i]),
                      float(t['maxexp'][i]), float(deg_i), share, float(len(co[i]))))
    out = _FAMILY[key] = (tuple(fam), tuple(roles))
    return out


def _leaves(node):
    if node[0] == 'M':
        yield node
        return
    for c, n in node[1]:
        yield from _leaves(c)


# ---------------------------------------------------------------- binding values (exact polynomials)


def value_features(value) -> tuple:
    """Symmetric statistics of an exact parameter value [[exponents(8)], coef] ...: identical for every binding mode
    and every realization of the same value."""
    degs = [sum(e) for e, c in value]
    support = {j for e, c in value for j, x in enumerate(e) if x}
    coefs = [c for e, c in value]
    return (_l(len(value)), float(max(degs)), float(min(degs)), float(len(support)),
            float(max(max(e) for e, c in value)), _l(sum(coefs)), _l(max(coefs)),
            float(len(value) == 1 and coefs[0] == 1))


# ---------------------------------------------------------------- state lookups by digest


class StateView:
    """Structural lookups in one state (digest -> handle id over the reachable set)."""

    def __init__(self, arena, root: int) -> None:
        self.arena, self.root = arena, root
        self.info = v1.state_info(arena, root)
        self.by_digest = {v1.digest(arena, h): h for h in self.info.reach_h}
        self._depth: dict = {}

    def handle(self, dig: str) -> int:
        return self.by_digest[bytes.fromhex(dig)]

    def gates(self, h: int) -> frozenset:
        v1.cost(self.arena, h)
        return self.arena.gate_reach_memo[h]

    def depth(self, h: int) -> int:
        d = self._depth.get(h)
        if d is None:
            kids = v1.items(self.arena.handles[h])
            d = self._depth[h] = 0 if not kids else 1+max(self.depth(c) for c, n in kids)
        return d


def _mono_digest(m: int) -> bytes:
    return hashlib.sha256(b'M'+m.to_bytes(4, 'big')).digest()


def realization_features(sv: StateView, mode: str, handle_dig, exponents) -> tuple:
    """Concrete structure realizing one parameter: an existing handle (REACHABLE_HANDLE_BIND) or a monomial."""
    a = sv.arena
    if mode == 'REACHABLE_HANDLE_BIND':
        h = sv.handle(handle_dig)
        node = ('h', h)
        feats = (1.0, 1.0, _l(v1.cost(a, h)), _l(len(v1.state_info(a, h).reach_h)), float(sv.depth(h)),
                 _l(sv.info.fanout.get(h, 0)), float(sv.info.fanout.get(h, 0) > 1))
        return feats, node
    m = v1.pack(tuple(exponents))
    h = sv.by_digest.get(_mono_digest(m))
    node = ('h', h) if h is not None else ('m', m)
    fan = sv.info.fanout.get(h, 0) if h is not None else 0
    return (0.0, float(h is not None), _l(v1.MONO_COST(m)), 0.0, 0.0, _l(fan), float(fan > 1)), node


# ---------------------------------------------------------------- concrete realization (exact local structure)


def _pnode(j):
    """Payload JSON node -> hashable tuple."""
    if j[0] in ('m', 'H'):
        return (j[0], tuple(j[1]) if j[0] == 'm' else j[1])
    return (j[0], tuple((_pnode(c), n) for c, n in j[1]))


def _local_gates(node) -> int:
    if node[0] == 'm':
        m = v1.pack(node[1])
        return v1.MONO_COST(m) if m else 0
    if node[0] == 'H':
        return 0
    return sum(len(v1.addchain(n)) for c, n in node[1])+len(node[1])-1


def _node_digest(node, memo: dict) -> bytes:
    d = memo.get(node)
    if d is None:
        if node[0] == 'm':
            d = _mono_digest(v1.pack(node[1]))
        elif node[0] == 'H':
            d = bytes.fromhex(node[1])
        else:
            parts = sorted(_node_digest(c, memo)+n.to_bytes(2, 'big') for c, n in node[1])
            d = hashlib.sha256((b'S' if node[0] == 'S' else b'P')+b''.join(parts)).digest()
        memo[node] = d
    return d


def _sum_of(kids: Counter):
    items = tuple(sorted(kids.items()))
    return items[0][0] if len(items) == 1 and items[0][1] == 1 else ('S', items)


def _prod_with(node, v: tuple):
    """`mk_prod(node, mono(v))` on payload nodes: monomial factors merge, other boundaries are kept."""
    if node[0] == 'm':
        return ('m', tuple(a+b for a, b in zip(node[1], v)))
    if not any(v):
        return node
    items = tuple(sorted(Counter({node: 1, ('m', tuple(v)): 1}).items()))
    return ('P', items)


def _pdepth(node) -> int:
    if node[0] in ('m', 'H'):
        return 0
    return 1+max(_pdepth(c) for c, n in node[1])


def concrete_features(sv: StateView, rec) -> tuple:
    """Local, exact structure of the concrete successor: consumed region, lowered target (d applied for a latent
    action), reuse of existing handles and the rebuilt context. No action is applied, nothing is interned."""
    a = sv.arena
    con = rec.concrete
    ctx = sv.handle(con['site']['context'])
    region = [(sv.handle(dg), n) for dg, n in con['site']['region']]
    top = Counter()
    for j, n in con['target']:
        top[_pnode(j)] += n
    d = con['site']['d']
    if d is not None:                                     # repl = mk_prod(mono(d), mk_sum(T)) replaces the region
        top = Counter({_prod_with(_sum_of(top), tuple(d)): 1})
    memo: dict = {}
    reach = set(sv.by_digest)
    new_gates, iso_gates, nodes, existing, reused = 0, 0, 0, 0, 0
    seen: set = set()
    stack = list(top)
    while stack:
        node = stack.pop()
        dg = _node_digest(node, memo)
        if dg in seen:
            continue
        seen.add(dg)
        nodes += 1
        iso_gates += _local_gates(node)
        if node[0] == 'H':
            reused += 1
            existing += 1
            continue
        if dg in reach:
            existing += 1
            continue
        new_gates += _local_gates(node)
        if node[0] in ('S', 'P'):
            stack.extend(c for c, n in node[1])
    src = frozenset().union(*(sv.gates(h) for h, n in region))
    kept = frozenset().union(*(sv.gates(sv.handle(n[1])) for n in seen_h(top))) if reused else frozenset()
    before = Counter(dict(v1.items(a.handles[ctx])))
    after = Counter(before)
    for h, n in region:
        after[h] -= n
    after = Counter({('D', v1.digest(a, h)): n for h, n in after.items() if n > 0})
    for node, n in top.items():
        after[('D', _node_digest(node, memo))] += n
    items_after = sorted(after.items())
    collapses = len(items_after) == 1 and items_after[0][1] == 1
    g_before = sum(len(v1.addchain(n)) for h, n in before.items())+len(before)-1
    g_after = 0 if collapses else sum(len(v1.addchain(n)) for x, n in items_after)+len(items_after)-1
    if collapses:
        ctx2 = items_after[0][0][1]
    else:
        ctx2 = hashlib.sha256(b'S'+b''.join(sorted(x[1]+n.to_bytes(2, 'big') for x, n in items_after))).digest()
    delta = new_gates+(g_after-g_before)-len(src-kept)
    return (_l(len(region)), _l(sum(n for h, n in region)), _l(len(src)), _l(len(src-kept)), _l(nodes),
            _l(iso_gates), _l(new_gates), existing/max(nodes, 1), _l(reused), _l(g_before), _l(g_after),
            float(collapses), float(ctx2 in reach), _sl(delta), float(max(_pdepth(n) for n in top)))


def seen_h(top: Counter):
    """Every opaque existing-handle leaf of the target payload."""
    out, stack = set(), list(top)
    while stack:
        node = stack.pop()
        if node[0] == 'H':
            out.add(node)
        elif node[0] in ('S', 'P'):
            stack.extend(c for c, n in node[1])
    return out


def d_features(d) -> tuple:
    if d is None:
        return (0.0,)*N_D
    nz = sorted((e for e in d if e), reverse=True)+[0, 0, 0]
    return (float(sum(d)), float(sum(1 for e in d if e)), float(max(d)), float(nz[0]), float(nz[1]), float(nz[2]))


# ---------------------------------------------------------------- one record


def rewrite_features(rec, sv: StateView) -> ActionFeatures:
    """Features of a VISIBLE_REWRITE / LATENT_REWRITE record in its state."""
    sem, prov = rec.semantic, rec.provenance
    fam, role_struct = family_features(sem['family']['relation'])
    role_param = {j: int(p) for p, j in prov['role_of_parameter'].items()}
    by_param = {b['param']: b for b in prov['replay']['bindings']}
    roles, nodes = [], []
    for j, par in enumerate(sem['parameters']):
        b = by_param[role_param[j]]
        rf, node = realization_features(sv, b['mode'], b['handle'], b['exponents'])
        roles.append(role_struct[j]+value_features(par['value'])+rf)
        nodes.append(node)
    return ActionFeatures(KINDS.index(rec.kind), fam, tuple(roles), tuple(nodes), d_features(sem['d']),
                          concrete_features(sv, rec), sv.handle(sem['site']['context']),
                          tuple((sv.handle(dg), n) for dg, n in sem['site']['region']))


def view_features(rec, arena, substituted_root: int) -> ActionFeatures:
    """A VIEW_REWRITE: its inner rewrite's features in the substituted state V (where the inner rewrite lives),
    marked as a view action. Interface support only: views are not behind the record boundary until T6B."""
    from .action_record import rewrite_record
    assert rec.kind == 'VIEW_REWRITE' and rec.live is not None
    sv = StateView(arena, substituted_root)
    f = rewrite_features(rewrite_record(arena, rec.live.cand, 'view', origin='view_rewrite'), sv)
    f.kind = KINDS.index('VIEW_REWRITE')
    return f


def record_features(recs: list, arena, root: int) -> list:
    """ActionFeatures for every rewrite record of a state (None for PRIMITIVE records, which the inherited V1 head
    scores from the state graph)."""
    sv = StateView(arena, root)
    return [None if r.kind == 'PRIMITIVE' else rewrite_features(r, sv) for r in recs]
