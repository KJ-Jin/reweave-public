"""Parameterized relations: patterns, normal form, canonical serialization and `schema_hash`, binding plan,
bounded unification, target lowering, signatures.

Pattern := ('M', ((param, e), ...))       product of parameters; ('M', ()) is One
         | ('S', ((Pattern, count), ...))
         | ('P', ((Pattern, count), ...))
Region  := ((Pattern, count), ...)         a multiset
Monomial bindings map parameter -> exponent vector (8-tuple); `canon/exprparam.py` generalizes them to compound
handles. `schema_hash` is the SHA-256 of the canonical body minimized over parameter permutations.

Matching uses a leaf-greedy plan when one exists, and otherwise the bounded exact simultaneous binding solver
(`simultaneous_bindings`) for leaves that introduce several unbound parameters at once.
"""

import hashlib
import itertools
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from . import v1
from .hashing import canonical_json

NV = 8
VERSION = 'rewrite-schema-1'


class Cut(Exception):
    """A work cap was reached."""


class CapError(Exception):
    """An instantiation exceeded inherited monomial caps."""


# ---------------------------------------------------------------- serialization and normal form


def ser(p) -> list:
    if p[0] == 'M':
        return ['M', [[i, e] for i, e in p[1]]]
    return [p[0], [[ser(c), n] for c, n in p[1]]]


def ser_region(r) -> list:
    return [[ser(c), n] for c, n in r]


def deser(j) -> tuple:
    if j[0] == 'M':
        return ('M', tuple((i, e) for i, e in j[1]))
    return (j[0], tuple((deser(c), n) for c, n in j[1]))


def deser_region(j) -> tuple:
    return tuple((deser(c), n) for c, n in j)


@lru_cache(maxsize=None)
def key(p) -> str:
    return canonical_json(ser(p))


def _sorted_items(counter: Counter) -> tuple:
    return tuple(sorted(counter.items(), key=lambda cn: (key(cn[0]), cn[1])))


@lru_cache(maxsize=None)
def norm(p) -> tuple:
    kind = p[0]
    if kind == 'M':
        acc = Counter()
        for i, e in p[1]:
            acc[i] += e
        return ('M', tuple(sorted((i, e) for i, e in acc.items() if e)))
    kids: Counter = Counter()
    mon: Counter = Counter()
    for c, n in p[1]:
        c = norm(c)
        if kind == 'P' and c[0] == 'M':
            for i, e in c[1]:
                mon[i] += e*n
        else:
            kids[c] += n
    if kind == 'P':
        if mon:
            kids[('M', tuple(sorted(mon.items())))] += 1
        if not kids:
            return ('M', ())
    items = _sorted_items(kids)
    if len(items) == 1 and items[0][1] == 1:
        return items[0][0]
    return (kind, items)


def norm_region(r) -> tuple:
    kids: Counter = Counter()
    for c, n in r:
        kids[norm(c)] += n
    return _sorted_items(kids)


def relabel(p, perm: dict) -> tuple:
    if p[0] == 'M':
        return ('M', tuple((perm[i], e) for i, e in p[1]))
    return (p[0], tuple((relabel(c, perm), n) for c, n in p[1]))


def relabel_region(r, perm: dict) -> tuple:
    return tuple((relabel(c, perm), n) for c, n in r)


@lru_cache(maxsize=None)
def holes(p) -> frozenset:
    if p[0] == 'M':
        return frozenset(i for i, e in p[1])
    return frozenset().union(*(holes(c) for c, n in p[1]))


def region_holes(r) -> frozenset:
    return frozenset().union(*(holes(c) for c, n in r)) if r else frozenset()


# ---------------------------------------------------------------- canonical schema


@dataclass
class Schema:
    types: tuple
    source: tuple
    target: tuple
    body: dict
    schema_hash: str
    automorphisms: list = field(default_factory=list)

    def to_json(self) -> dict:
        return dict(schema_hash=self.schema_hash, holes=list(self.types), source=ser_region(self.source),
                    target=ser_region(self.target), automorphisms=[list(a) for a in self.automorphisms])

    @staticmethod
    def from_json(j: dict) -> 'Schema':
        s = Schema(tuple(j['holes']), deser_region(j['source']), deser_region(j['target']), {}, j['schema_hash'],
                   [tuple(a) for a in j['automorphisms']])
        s.body = dict(v=VERSION, holes=list(s.types), source=ser_region(s.source), target=ser_region(s.target))
        return s


def _body(types, source, target) -> dict:
    return dict(v=VERSION, holes=list(types), source=ser_region(source), target=ser_region(target))


def canonicalize(types: tuple, source: tuple, target: tuple) -> Schema:
    n = len(types)
    best = None
    for pi in itertools.permutations(range(n)):
        perm = {i: pi[i] for i in range(n)}
        t2 = [None]*n
        for i in range(n):
            t2[pi[i]] = types[i]
        src, tgt = norm_region(relabel_region(source, perm)), norm_region(relabel_region(target, perm))
        s = canonical_json(_body(t2, src, tgt))
        if best is None or s < best[0]:
            best = (s, tuple(t2), src, tgt)
    s, t2, src, tgt = best
    autos = []
    for pi in itertools.permutations(range(n)):
        perm = {i: pi[i] for i in range(n)}
        if canonical_json(_body(t2, norm_region(relabel_region(src, perm)), norm_region(relabel_region(tgt, perm)))) == s:
            autos.append(tuple(pi))
    body = _body(t2, src, tgt)
    return Schema(t2, src, tgt, body, hashlib.sha256(canonical_json(body).encode()).hexdigest(), autos)


# ---------------------------------------------------------------- binding plan


@lru_cache(maxsize=None)
def leaves(p) -> tuple:
    if p[0] == 'M':
        return (holes(p),)
    return tuple(l for c, n in p[1] for l in leaves(c))


SIMULTANEOUS_LEAF_HOLES_MAX = 4          # = RewriteConfig.n_holes_max; the solver never binds more holes than a schema may have


def processable(leaf_sets, bound: frozenset, max_new: int = 1) -> bool:
    """Greedy simulation: every Mon leaf introduces at most `max_new` unbound holes when reached (1 = leaf-greedy)."""
    todo, b = list(leaf_sets), set(bound)
    while todo:
        progress = False
        for l in list(todo):
            if len(l-b) <= max_new:
                b |= l
                todo.remove(l)
                progress = True
        if not progress:
            return False
    return True


def has_plan(schema: Schema) -> bool:
    """Leaf-greedy binding plan (the MVP-1 rule)."""
    return processable([l for c, n in schema.source for l in leaves(c)], frozenset())


def binding_plan(schema: Schema) -> str | None:
    """'leaf_greedy' if the MVP-1 plan exists; else 'simultaneous' if every Mon leaf introduces at most
    SIMULTANEOUS_LEAF_HOLES_MAX unbound holes when reached; else None."""
    leaf_sets = [l for c, n in schema.source for l in leaves(c)]
    if processable(leaf_sets, frozenset()):
        return 'leaf_greedy'
    if processable(leaf_sets, frozenset(), SIMULTANEOUS_LEAF_HOLES_MAX):
        return 'simultaneous'
    return None


def uses_simultaneous(schema: Schema) -> bool:
    return binding_plan(schema) == 'simultaneous'


def max_exp(p) -> int:
    if p[0] == 'M':
        return max((e for i, e in p[1]), default=0)
    return max(max_exp(c) for c, n in p[1])


# ---------------------------------------------------------------- lowering and signatures


def mon_vector(p, beta: dict) -> tuple:
    v = [0]*NV
    for i, e in p[1]:
        b = beta[i]
        for j in range(NV):
            v[j] += e*b[j]
    return tuple(v)


def lower(p, beta: dict, arena) -> int:
    if p[0] == 'M':
        v = mon_vector(p, beta)
        if max(v) > 6:
            raise CapError('monomial exponent above inherited cap')
        return v1.mk_mono(arena, v1.pack(v))
    kids: Counter = Counter()
    for c, n in p[1]:
        kids[lower(c, beta, arena)] += n
    return (v1.mk_sum if p[0] == 'S' else v1.mk_prod)(arena, kids)


def lower_region(r, beta: dict, arena) -> Counter:
    out: Counter = Counter()
    for c, n in r:
        out[lower(c, beta, arena)] += n
    return out


def _inst_sig(p, beta: dict):
    if p[0] == 'M':
        return ['m', list(mon_vector(p, beta))]
    kids = sorted(([_inst_sig(c, beta), n] for c, n in p[1]), key=canonical_json)
    return [p[0], kids]


def target_signature(schema: Schema, beta: dict) -> str:
    """Signature of the instantiated target without interning any handle."""
    kids = sorted(([_inst_sig(c, beta), n] for c, n in schema.target), key=canonical_json)
    return hashlib.sha256(canonical_json(kids).encode()).hexdigest()


def binding_digest(beta: dict) -> str:
    return hashlib.sha256(canonical_json([[i, list(beta[i])] for i in sorted(beta)]).encode()).hexdigest()


def normalized_signature(schema: Schema, beta: dict) -> tuple:
    """Binding signature: min over variable permutations (column sort) and schema automorphisms."""
    n = len(schema.types)
    best = None
    for pi in schema.automorphisms or [tuple(range(n))]:
        rows = [None]*n
        for i in range(n):
            rows[pi[i]] = beta[i]
        cols = sorted(tuple(rows[h][j] for h in range(n)) for j in range(NV))
        sig = tuple(tuple(cols[j][h] for j in range(NV)) for h in range(n))
        if best is None or sig < best:
            best = sig
    return best


# ---------------------------------------------------------------- bounded unification


class Budget:
    __slots__ = ('states', 'cap')

    def __init__(self, cap: int) -> None:
        self.states, self.cap = 0, cap

    def tick(self) -> None:
        self.states += 1
        if self.states > self.cap:
            raise Cut()


class ChildIndex:
    """Children of one Sum/Prod node or region: counts, digest order, Mono lookup and divisibility classes."""

    def __init__(self, arena, counts: dict) -> None:
        self.arena = arena
        self.order = sorted(counts, key=lambda h: v1.digest(arena, h))
        self.mono = {}
        self.vec = {}
        for h in self.order:
            node = arena.handles[h]
            if isinstance(node, v1.Mono):
                self.mono[node.m] = h
                self.vec[h] = v1.unpack(node.m)
        self._div: dict = {}

    def divisible(self, e: int) -> list:
        if e not in self._div:
            self._div[e] = [h for h in self.order if h in self.vec and any(self.vec[h]) and all(x % e == 0 for x in self.vec[h])]
        return self._div[e]

    def candidates(self, p, beta: dict) -> list:
        if p[0] == 'M':
            unbound = [(i, e) for i, e in p[1] if i not in beta]
            if not unbound:
                v = mon_vector(p, beta)
                if max(v) > 6:
                    return []
                h = self.mono.get(v1.pack(v))
                return [h] if h is not None else []
            if len(unbound) == 1 and len(p[1]) == 1:
                return self.divisible(unbound[0][1])
            return [h for h in self.order if h in self.vec]
        cls = v1.Sum if p[0] == 'S' else v1.Prod
        return [h for h in self.order if isinstance(self.arena.handles[h], cls)]


def _entry_order(entries, beta: dict) -> list:
    bound = frozenset(beta)

    def k(entry):
        p, n = entry
        return (0 if processable(leaves(p), bound) else 1, len(holes(p)-bound), -max_exp(p), key(p), n)
    return sorted(entries, key=k)


def simultaneous_bindings(unbound: list, res: list, beta: dict, cfg, budget: Budget):
    """Bounded exact simultaneous binding of one Mon leaf: every assignment of exponent vectors b_i to the unbound
    holes (i, e_i) with sum_i e_i*b_i == res, each b_i nonconstant, max(b_i) <= binding_exponent_max,
    sum(b_i) <= binding_degree_max, pairwise distinct and distinct from existing bindings. Enumeration order is
    deterministic (coordinate-wise lexicographic); every enumerated combination ticks the work budget (fail-closed Cut)."""
    assert 2 <= len(unbound) <= SIMULTANEOUS_LEAF_HOLES_MAX
    per_coord = []
    for j in range(NV):
        opts = [combo for combo in itertools.product(*(range(min(cfg.binding_exponent_max, res[j]//e)+1) for i, e in unbound))
                if sum(e*x for (i, e), x in zip(unbound, combo)) == res[j]]
        if not opts:
            return
        per_coord.append(opts)
    taken = set(beta.values())
    for choice in itertools.product(*per_coord):
        budget.tick()
        bs = [tuple(choice[j][k] for j in range(NV)) for k in range(len(unbound))]
        if any(not any(b) or max(b) > cfg.binding_exponent_max or sum(b) > cfg.binding_degree_max or b in taken for b in bs):
            continue
        if len(set(bs)) < len(bs):
            continue
        nb = dict(beta)
        for (i, e), b in zip(unbound, bs):
            nb[i] = b
        yield nb


def unify(p, h: int, beta: dict, arena, cfg, budget: Budget, simultaneous: bool = False):
    budget.tick()
    node = arena.handles[h]
    if p[0] == 'M':
        if not isinstance(node, v1.Mono):
            return
        res = list(v1.unpack(node.m))
        unbound = []
        for i, e in p[1]:
            if i in beta:
                b = beta[i]
                for j in range(NV):
                    res[j] -= e*b[j]
            else:
                unbound.append((i, e))
        if min(res) < 0:
            return
        if not unbound:
            if not any(res):
                yield beta
            return
        if len(unbound) > 1:
            if simultaneous and len(unbound) <= SIMULTANEOUS_LEAF_HOLES_MAX:
                yield from simultaneous_bindings(unbound, res, beta, cfg, budget)
            return
        i, e = unbound[0]
        if any(x % e for x in res):
            return
        b = tuple(x//e for x in res)
        if not any(b) or max(b) > cfg.binding_exponent_max or sum(b) > cfg.binding_degree_max or b in beta.values():
            return
        nb = dict(beta)
        nb[i] = b
        yield nb
        return
    if (p[0] == 'S' and not isinstance(node, v1.Sum)) or (p[0] == 'P' and not isinstance(node, v1.Prod)):
        return
    counts = dict(v1.items(node))
    yield from assign(tuple(p[1]), ChildIndex(arena, counts), counts, beta, arena, cfg, budget, exact=True, simultaneous=simultaneous)


def assign(entries: tuple, index: ChildIndex, remaining: dict, beta: dict, arena, cfg, budget: Budget, exact: bool,
           trace: list | None = None, simultaneous: bool = False):
    """Backtrack over pattern entries; each entry (pattern, count) takes a child with remaining count >= count.
    exact=True requires every remaining count to reach zero (R_inst == children); exact=False allows R_inst ⊆ children.
    `trace` (if given) receives (entry, child) choices of the current solution when yielding. `simultaneous` enables
    the simultaneous binding solver; callers set it only for schemas without a leaf-greedy plan."""
    if not entries:
        if not exact or not any(remaining.values()):
            yield beta
        return
    entry = _entry_order(entries, beta)[0]
    rest = list(entries)
    rest.remove(entry)
    p, n = entry
    for h in index.candidates(p, beta):
        if remaining.get(h, 0) < n:
            continue
        for nb in unify(p, h, beta, arena, cfg, budget, simultaneous):
            remaining[h] -= n
            if trace is not None:
                trace.append((entry, h))
            yield from assign(tuple(rest), index, remaining, nb, arena, cfg, budget, exact, trace, simultaneous)
            if trace is not None:
                trace.pop()
            remaining[h] += n
