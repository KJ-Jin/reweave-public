"""Generic ExprParam binding: one semantic parameter type, two runtime binding modes.

A relation leaf ('M', ((i, e_i), ...)) denotes the product  prod_i A_i^{e_i}  of ExprParams. At runtime each ExprParam
binds in one of two modes, and the modes may be mixed within one match:

  MONOMIAL_BIND          A := x^b, an exponent vector;
  REACHABLE_HANDLE_BIND  A := H, a compound handle ALREADY PRESENT in the current state (`HB`).

Boundedness: a parameter binds only units already materialized at its matched position (a context child, or a factor
of a child product). The matcher never synthesizes, regroups or searches for compound expressions, so handle binding
stays within the ordinary matcher budgets. Repeated occurrences of a parameter must bind the same exact polynomial
(semantic equality, never arena identity). Lowering reuses bound handles as children; every successor passes an
acyclicity / no-rebuild check and exact verification.

Principle: learn from structure, reuse through handles.
"""

from __future__ import annotations

import hashlib
import itertools
import weakref
from collections import Counter
from dataclasses import dataclass, field

from .. import rewrite_latent as L
from .. import rewrite_match as RM
from .. import rewrite_schema as S
from .. import v1
from ..hashing import canonical_json, sha256_json
from ..rewrite_latent import LatentCandidate
from .labelling import key_of

NV = S.NV
REALIZATIONS_MAX = 4                                   # concrete realizations per match and per semantic class (= K_REP)


# ---------------------------------------------------------------- binding values and semantic keys


@dataclass(frozen=True)
class HB:
    """A REACHABLE_HANDLE_BIND value. `key` (exact semantic key of the handle's polynomial) is the binding's semantic
    value; `h` / `dig` name the structural handle that REALIZES it (replay, concrete realization key). `alts` is
    matching-time only: every existing unit consumed with this value (excluded from equality and hashing)."""
    h: int
    key: str
    dig: str
    alts: frozenset = field(default=frozenset(), compare=False)


def has_hb(beta: dict) -> bool:
    return any(isinstance(b, HB) for b in beta.values())


def poly_key(poly: dict) -> str:
    """Semantic key of a polynomial in its own variables (no renaming quotient). A prefilter: equality of values is
    decided by exact polynomial comparison."""
    return key_of(tuple(sorted(poly.items())))


_KEYS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def hkey(arena, h: int) -> str:
    """Per-arena memo of handle keys (a read-only cache: it creates no handle). Bounded work: every bindable unit is
    a sub-handle of the state, and with positive coefficients a sub-handle's support is at most the support of the
    task polynomial (supp(Q)*m is contained in supp(Q*R) for m in supp(R); supp(child) in supp(sum)); its exact
    polynomial is already memoized by the state's exact verification."""
    memo = _KEYS.get(arena)
    if memo is None:
        memo = _KEYS[arena] = {}
    k = memo.get(h)
    if k is None:
        k = memo[h] = poly_key(v1.hpoly(arena, h))
    return k


def hb(arena, h: int) -> HB:
    return HB(h, hkey(arena, h), v1.digest(arena, h).hex())


def param_key(b) -> str:
    """Semantic key of one ExprParam binding, uniform over binding modes (a monomial binding b denotes x^b)."""
    return b.key if isinstance(b, HB) else poly_key({v1.pack(tuple(b)): 1})


def mode(b) -> str:
    return 'REACHABLE_HANDLE_BIND' if isinstance(b, HB) else 'MONOMIAL_BIND'


def _grow(b: HB, units) -> HB:
    return HB(b.h, b.key, b.dig, b.alts | frozenset(units))


class Binder:
    """Per-state binding context. `allowed` is the concrete state's reachable handle set; `within` (latent views) is
    the set of handles below the view's original children — a bound unit must lie inside the consumed region, so a
    view's stripped product is never bindable even when an equal handle exists elsewhere in the state."""

    def __init__(self, arena, handles: bool, allowed=frozenset()) -> None:
        self.arena, self.handles, self.allowed = arena, handles, allowed
        self.within = None
        self.stats = Counter()

    def unit_ok(self, h: int) -> bool:
        a = self.arena
        return (not isinstance(a.handles[h], v1.Mono) and not a.constant[h] and h in self.allowed
                and (self.within is None or h in self.within))

    def key(self, h: int) -> str:
        memo = _KEYS.get(self.arena)
        if memo is None or h not in memo:
            self.stats['keys_computed'] += 1
        return hkey(self.arena, h)

    def same(self, key: str, h_ref: int, h: int) -> bool:
        """Exact semantic equality of two handles: the key prefilter, then exact polynomial comparison."""
        if h == h_ref:
            return True
        if self.key(h) != key:
            return False
        if v1.poly_equal(v1.hpoly(self.arena, h_ref), v1.hpoly(self.arena, h)):
            return True
        self.stats['key_collisions'] += 1
        return False


# ---------------------------------------------------------------- matching (the mainline matcher, generalized)


class GIndex(S.ChildIndex):
    """`ChildIndex` plus the compound children a handle-capable leaf may match. Identical to it when handles are off."""

    def __init__(self, arena, counts: dict, bd: Binder) -> None:
        super().__init__(arena, counts)
        self.bd = bd
        self.compound = [h for h in self.order if not isinstance(arena.handles[h], v1.Mono)] if bd.handles else []

    def candidates(self, p, beta: dict) -> list:
        if p[0] != 'M' or not self.bd.handles:
            return S.ChildIndex.candidates(self, p, beta)
        bound = [beta[i] for i, e in p[1] if i in beta]
        if any(isinstance(b, HB) for b in bound):
            return self.compound                     # a leaf with a handle-bound parameter lowers to a compound
        base = S.ChildIndex.candidates(self, p, beta)
        return base if len(bound) == len(p[1]) else base+self.compound


def _lone(leaf) -> bool:
    return len(leaf[1]) == 1 and leaf[1][0][1] == 1


class _Pause:
    """Yielded through the matcher generators when a fair-scheduled lane's current quantum is spent (T6A.2). Never
    produced under the mainline budgets, which have no `pause_due`."""
    __slots__ = ()

    def __repr__(self) -> str:
        return 'PAUSE'


PAUSE = _Pause()


def _tick(budget) -> bool:
    """Tick the budget; True iff a fair scheduler asks the generator to pause here (resumable, no work lost)."""
    budget.tick()
    due = getattr(budget, 'pause_due', None)
    return due is not None and due()


def unify_g(p, h: int, beta: dict, arena, cfg, budget, simultaneous: bool, bd: Binder):
    """`rewrite_schema.unify`, with handle-capable leaves routed to `_leaf` and product leaves inside product patterns
    matched flat (associative MUL) when REACHABLE_HANDLE_BIND is on."""
    if bd.handles and p[0] == 'M' and (not isinstance(arena.handles[h], v1.Mono)
                                       or any(isinstance(beta.get(i), HB) for i, e in p[1])):
        yield from _leaf(p, h, beta, arena, cfg, budget, simultaneous, bd)
        return
    if _tick(budget):
        yield PAUSE
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
            if simultaneous and len(unbound) <= S.SIMULTANEOUS_LEAF_HOLES_MAX:
                yield from S.simultaneous_bindings(unbound, res, beta, cfg, budget)
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
    if bd.handles and p[0] == 'P':
        leaves = [(c, n) for c, n in p[1] if c[0] == 'M']
        if leaves and not (_lone(leaves[0][0]) and leaves[0][1] == 1):
            yield from _flat_product(p, leaves[0], counts, beta, arena, cfg, budget, simultaneous, bd)
            return
    yield from assign_g(tuple(p[1]), GIndex(arena, counts, bd), counts, beta, arena, cfg, budget, exact=True,
                        simultaneous=simultaneous, bd=bd)


def _flat_product(p, leaf_entry, counts: dict, beta: dict, arena, cfg, budget, simultaneous: bool, bd: Binder):
    """Associative MUL: the pattern's non-leaf factors take factors of this product node, then the product leaf must
    exactly cover the REMAINING factors (its units and one merged monomial). V1 keeps one flat factor list, so this
    is the structural inverse of the flat lowering in `lower_g`. `rewrite_schema.norm` merges the monomial children
    of a product into one leaf, so there is at most one."""
    leaf, n = leaf_entry
    params = tuple((i, e*n) for i, e in leaf[1])
    rest = tuple(e for e in p[1] if e[0][0] != 'M')
    trace: list = []
    for nb in assign_g(rest, GIndex(arena, counts, bd), dict(counts), beta, arena, cfg, budget, exact=False,
                       trace=trace, simultaneous=simultaneous, bd=bd):
        if nb is PAUSE:
            yield nb
            continue
        left = Counter(counts)
        for (pp, nn), h in trace:
            left[h] -= nn
        left = {h: c for h, c in left.items() if c > 0}
        if left:
            yield from _cover(params, left, nb, arena, cfg, budget, simultaneous, bd)


def _mono_finish(mono: list, res: list, beta: dict, cfg, budget, simultaneous: bool):
    """Bind the leaf's monomial-mode parameters to the residual monomial exactly as the mainline leaf does."""
    if not mono:
        if not any(res):
            yield beta
        return
    if len(mono) > 1:
        if simultaneous and len(mono) <= S.SIMULTANEOUS_LEAF_HOLES_MAX:
            yield from S.simultaneous_bindings(mono, res, beta, cfg, budget)
        return
    i, e = mono[0]
    if any(x % e for x in res):
        return
    b = tuple(x//e for x in res)
    if not any(b) or max(b) > cfg.binding_exponent_max or sum(b) > cfg.binding_degree_max or b in beta.values():
        return
    nb = dict(beta)
    nb[i] = b
    yield nb


def _leaf(p, h: int, beta: dict, arena, cfg, budget, simultaneous: bool, bd: Binder):
    """A leaf at a compound node (or with a handle-bound parameter)."""
    if _tick(budget):
        yield PAUSE
    node = arena.handles[h]
    if isinstance(node, v1.Mono):
        return                                        # V1: a compound handle is never a monomial
    params = p[1]
    if _lone(p):                                      # the leaf is one parameter: the node itself is the unit
        i = params[0][0]
        b = beta.get(i)
        if b is None:
            if bd.unit_ok(h):
                nb = dict(beta)
                nb[i] = HB(h, bd.key(h), v1.digest(arena, h).hex(), frozenset((h,)))
                bd.stats['handle_bindings'] += 1
                yield nb
        elif isinstance(b, HB) and bd.same(b.key, b.h, h):
            nb = dict(beta)
            nb[i] = _grow(b, (h,) if bd.unit_ok(h) else ())
            yield nb
        return
    if not isinstance(node, v1.Prod):
        return                                        # A^e (e >= 2) and multi-parameter leaves lower to a product
    yield from _cover(params, dict(v1.items(node)), beta, arena, cfg, budget, simultaneous, bd)


def _cover(params, factors: dict, beta: dict, arena, cfg, budget, simultaneous: bool, bd: Binder):
    """Exactly cover a factor multiset by one leaf's parameters. Non-monomial factors are grouped into exact semantic
    value slots; each unbound parameter chooses one slot (consuming its exponent's worth of units) or the monomial
    mode; every choice ticks the budget. A slot's bindable units are the parameter's realization alternatives."""
    res = [0]*NV
    slots: dict = {}                                  # slot id -> [remaining, representative, units, reference, key]
    for c in sorted(factors, key=lambda c: v1.digest(arena, c)):
        n = factors[c]
        if isinstance(arena.handles[c], v1.Mono):
            for j, x in enumerate(v1.unpack(arena.handles[c].m)):
                res[j] += n*x
            continue
        k = bd.key(c)
        # exact grouping: the key digest indexes, exact polynomial comparison decides (a colliding different value
        # gets its own slot; a later handle joins whichever slot holds its exact value)
        s = next((s for s in slots.values() if s[4] == k and bd.same(k, s[3], c)), None)
        if s is None:
            s = slots[(k, len(slots))] = [0, None, [], c, k]
        s[0] += n
        if bd.unit_ok(c):
            s[2].append(c)
            if s[1] is None:
                s[1] = c
    base = dict(beta)
    unbound = []
    for i, e in params:
        b = base.get(i)
        if b is None:
            unbound.append((i, e))
        elif isinstance(b, HB):
            s = next((s for s in slots.values() if s[4] == b.key and bd.same(b.key, b.h, s[3])), None)
            if s is None or s[0] < e:
                return
            s[0] -= e
            base[i] = _grow(b, s[2])
        else:
            for j in range(NV):
                res[j] -= e*b[j]
    if min(res) < 0:
        return
    sids = list(slots)

    def choose(k: int, nb: dict, mono: list):
        if k == len(unbound):
            if any(slots[x][0] for x in sids):
                return                                # every existing unit must be consumed: no regrouping
            yield from _mono_finish(mono, res, nb, cfg, budget, simultaneous)
            return
        i, e = unbound[k]
        for sid in sids:
            s = slots[sid]
            if s[1] is None or s[0] < e:
                continue
            if _tick(budget):
                yield PAUSE
            bd.stats['handle_choices'] += 1
            s[0] -= e
            nb2 = dict(nb)
            nb2[i] = HB(s[1], s[4], v1.digest(arena, s[1]).hex(), frozenset(s[2]))
            yield from choose(k+1, nb2, mono)
            s[0] += e
        if _tick(budget):
            yield PAUSE
        yield from choose(k+1, nb, mono+[(i, e)])
    for out in choose(0, base, []):
        if out is PAUSE:
            yield out
            continue
        if has_hb(out):
            bd.stats['handle_bindings'] += 1
        yield out


def assign_g(entries: tuple, index, remaining: dict, beta: dict, arena, cfg, budget, exact: bool,
             trace: list | None = None, simultaneous: bool = False, bd: Binder | None = None):
    """`rewrite_schema.assign` over `GIndex` / `unify_g` (line for line otherwise)."""
    if not entries:
        if not exact or not any(remaining.values()):
            yield beta
        return
    entry = S._entry_order(entries, beta)[0]
    rest = list(entries)
    rest.remove(entry)
    p, n = entry
    for h in index.candidates(p, beta):
        if remaining.get(h, 0) < n:
            continue
        for nb in unify_g(p, h, beta, arena, cfg, budget, simultaneous, bd):
            if nb is PAUSE:
                yield nb
                continue
            remaining[h] -= n
            if trace is not None:
                trace.append((entry, h))
            yield from assign_g(tuple(rest), index, remaining, nb, arena, cfg, budget, exact, trace, simultaneous, bd)
            if trace is not None:
                trace.pop()
            remaining[h] += n


def below(arena, R) -> frozenset:
    """Every handle at or below the consumed children R: the only handles a binding of THIS candidate may name."""
    return frozenset().union(*(v1.state_info(arena, r).reach_h for r in R))


def realizations(arena, beta: dict, inside=None) -> tuple:
    """The distinct concrete realizations of one matched semantic binding: each handle-bound parameter may be
    realized by any existing unit it consumed with its exact value AND that lies inside this candidate's consumed
    region (`inside`, the apply-time predicate's region). Deterministic (digest order), at most REALIZATIONS_MAX;
    returns (betas without `alts`, truncated). A parameter with no admissible realization yields no candidate."""
    holes = sorted(i for i, b in beta.items() if isinstance(b, HB))
    if not holes:
        return [beta], False
    opts = []
    for i in holes:
        b = beta[i]
        units = sorted(u for u in set(b.alts) | {b.h} if inside is None or u in inside)
        units.sort(key=lambda u: v1.digest(arena, u))
        if not units:
            return [], False
        opts.append([HB(u, b.key, v1.digest(arena, u).hex()) for u in units])
    out = []
    for combo in itertools.product(*opts):
        nb = dict(beta)
        for i, x in zip(holes, combo):
            nb[i] = x
        out.append(nb)
        if len(out) > REALIZATIONS_MAX:
            break
    return out[:REALIZATIONS_MAX], len(out) > REALIZATIONS_MAX


# ---------------------------------------------------------------- signatures, digests, candidate records


def _prod_node(kids: Counter, v) -> tuple:
    """`mk_prod` semantics: monomial factors merge into one (dropped when it is One), a single factor of multiplicity 1
    is the factor itself, every other boundary is kept."""
    if not kids:
        return ('m', tuple(v))
    kids = Counter(kids)
    if any(v):
        kids[('m', tuple(v))] += 1
    items = tuple(sorted(kids.items()))
    return items[0][0] if len(items) == 1 and items[0][1] == 1 else ('P', items)


def _sum_node(kids: Counter) -> tuple:
    """`mk_sum` semantics: no flattening, a single child of multiplicity 1 is the child itself."""
    items = tuple(sorted(kids.items()))
    return items[0][0] if len(items) == 1 and items[0][1] == 1 else ('S', items)


def _tnode(p, beta: dict, ref) -> tuple:
    """Canonical structure of the lowered pattern node (exactly `lower_g`'s DAG, V1 constructor semantics); a
    handle-bound unit is the opaque leaf ('H', ref(b))."""
    if p[0] == 'M':
        units: Counter = Counter()
        v = [0]*NV
        for i, e in p[1]:
            b = beta[i]
            if isinstance(b, HB):
                units[('H', ref(b))] += e
            else:
                for j in range(NV):
                    v[j] += e*b[j]
        return _prod_node(units, v) if units else ('m', tuple(v))
    kids: Counter = Counter()
    if p[0] == 'P':
        v = [0]*NV
        for c, n in p[1]:
            if c[0] == 'M' and any(isinstance(beta[i], HB) for i, e in c[1]):
                for i, e in c[1]:                     # associative MUL: the product leaf is spliced
                    b = beta[i]
                    if isinstance(b, HB):
                        kids[('H', ref(b))] += e*n
                    else:
                        for j in range(NV):
                            v[j] += e*n*b[j]
                continue
            node = _tnode(c, beta, ref)
            if node[0] == 'm':                        # mk_prod merges monomial factors
                for j in range(NV):
                    v[j] += n*node[1][j]
            else:
                kids[node] += n
        return _prod_node(kids, v)
    for c, n in p[1]:
        kids[_tnode(c, beta, ref)] += n
    return _sum_node(kids)


def target_payload(sch, beta: dict, ref=lambda b: b.dig) -> tuple:
    """EXACT concrete instantiated target: the multiset of lowered region nodes (surface-independent — two surfaces
    whose instantiations lower to the same DAG give the same payload)."""
    kids: Counter = Counter()
    for c, n in sch.target:
        kids[_tnode(c, beta, ref)] += n
    return tuple(sorted(kids.items()))


def target_signature_g(sch, beta: dict) -> str:
    """`rewrite_schema.target_signature` for monomial bindings; for handle bindings the digest of the exact concrete
    target payload (T6A.2: independent of the recognition surface)."""
    if not has_hb(beta):
        return S.target_signature(sch, beta)
    return hashlib.sha256(canonical_json(['t6-target-v2', target_payload(sch, beta)]).encode()).hexdigest()


def concrete_key(cand) -> tuple:
    """EXACT in-arena concrete successor key: context, consumed region, d and the exact target (handle ids). Equal
    keys give the identical successor DAG; the recognition surface is not part of it."""
    return (cand.ctx, tuple(sorted(cand.R_inst.items())), getattr(cand, 'd', None),
            target_payload(cand.schema, cand.beta, ref=lambda b: b.h))


_PTUPLES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def value_tuple(arena, b) -> tuple:
    """EXACT semantic value of one binding: the sorted term tuple of its polynomial (memoized per arena)."""
    if not isinstance(b, HB):
        return ((v1.pack(tuple(b)), 1),)
    memo = _PTUPLES.get(arena)
    if memo is None:
        memo = _PTUPLES[arena] = {}
    t = memo.get(b.h)
    if t is None:
        t = memo[b.h] = tuple(sorted(v1.hpoly(arena, b.h).items()))
    return t


def canonical_roles(arena, cand) -> tuple:
    """(frame, exact parameter values in the family's canonical parameter order). Among the relation's canonical
    frames (its symmetries) the smallest exact value tuple is chosen, ties by the realizing structures' digests, so
    the order is a function of the action's semantics."""
    from .relation import ac_forms
    k, best, frames = ac_forms(cand.schema)
    beta = cand.beta
    best_frame, best_key = None, None
    for img in frames:
        holes = [img.index(j) for j in range(k)]
        key = (tuple(value_tuple(arena, beta[i]) for i in holes),
               tuple(beta[i].dig if isinstance(beta[i], HB) else '' for i in holes))
        if best_key is None or key < best_key:
            best_frame, best_key = holes, key
    return best_frame, best_key[0]


def semantic_key_exact(arena, cand) -> tuple:
    """EXACT in-arena semantic action key: family (exact relation payload), parameter values in canonical order, d,
    context and consumed region. Binding modes, realizing structures and the surface are not part of it."""
    from .relation import family_payload
    return (family_payload(cand.schema), canonical_roles(arena, cand)[1], getattr(cand, 'd', None), cand.ctx,
            tuple(sorted(cand.R_inst.items())))


@dataclass
class GCandidate(RM.Candidate):
    """A handle-lane visible candidate: `surfaces` lists every recognition surface producing this exact concrete
    successor (provenance; the live `schema` is the one of the smallest order key)."""
    surfaces: tuple = ()


@dataclass
class GLatentCandidate(LatentCandidate):
    surfaces: tuple = ()


def binding_digest_g(arena, beta: dict) -> str:
    """`rewrite_schema.binding_digest`; a handle binding enters by its structural digest."""
    if not has_hb(beta):
        return S.binding_digest(beta)
    return hashlib.sha256(canonical_json([[i, ['H', beta[i].dig] if isinstance(beta[i], HB) else list(beta[i])]
                                          for i in sorted(beta)]).encode()).hexdigest()


def _region_key(arena, R) -> tuple:
    return tuple(sorted((v1.digest(arena, h).hex(), c) for h, c in R.items()))


def cand_g(arena, ctx: int, sch, beta: dict, R: Counter) -> RM.Candidate:
    """A visible candidate with the frozen site key and order key (`match_state`'s definitions)."""
    site_key = ('rw', v1.digest(arena, ctx).hex(), _region_key(arena, R))
    tsig, bdig = target_signature_g(sch, beta), binding_digest_g(arena, beta)
    order = (hashlib.sha256(canonical_json([site_key, sch.schema_hash, tsig, bdig]).encode()).hexdigest(),
             site_key, sch.schema_hash, tsig, bdig)
    return RM.Candidate(ctx, sch, dict(beta), Counter(R), site_key, tsig, bdig, order)


def lcand_g(arena, ctx: int, d, sch, beta: dict, R: Counter) -> LatentCandidate:
    """A latent candidate with `latent_candidates`' site key (d included) and order key."""
    site_key = ('lx', v1.digest(arena, ctx).hex(), _region_key(arena, R), tuple(d))
    tsig, bdig = target_signature_g(sch, beta), binding_digest_g(arena, beta)
    order = (hashlib.sha256(repr((site_key, sch.schema_hash, tsig, bdig)).encode()).hexdigest(),
             site_key, sch.schema_hash, tsig, bdig)
    return LatentCandidate(ctx, tuple(d), sch, dict(beta), Counter(R), site_key, tsig, bdig, order)


# ---------------------------------------------------------------- the two generators


def match_state_g(arena, root: int, schemas: list, cfg, handles: bool, hb_only: bool = False,
                  truncate: bool = True) -> tuple:
    """`rewrite_match.match_state` over the generic binder (the same contexts, caps, dedupe and order). With
    `hb_only` (the handle lane) only candidates with a handle-bound parameter are emitted, one per concrete
    realization; `truncate=False` leaves the output cap to the lane's admission."""
    info = v1.state_info(arena, root)
    bd = Binder(arena, handles, info.reach_h if handles else frozenset())
    contexts = sorted(info.contexts, key=lambda h: v1.digest(arena, h))
    work = dict(contexts=0, attempts=0, cap=cfg.match_attempts_max, unify_states=0, cut_contexts=0,
                cut_unify_states=0, cut_attempts=0, cut_output=0, generated=0)
    if len(contexts) > cfg.match_contexts_max:
        work['cut_contexts'] = len(contexts)-cfg.match_contexts_max
        contexts = contexts[:cfg.match_contexts_max]
    found: dict = {}
    stop = False
    rst = Counter()
    simultaneous = {sch.schema_hash: S.uses_simultaneous(sch) for sch in schemas}
    for ctx in contexts:
        if stop:
            break
        work['contexts'] += 1
        counts = dict(v1.items(arena.handles[ctx]))
        index = GIndex(arena, counts, bd)
        ctx_digest = v1.digest(arena, ctx).hex()
        for sch in schemas:
            budget = RM.WorkBudget(cfg.unify_states_max, work)
            trace: list = []
            try:
                for beta0 in assign_g(sch.source, index, dict(counts), {}, arena, cfg, budget, exact=False, trace=trace,
                                      simultaneous=simultaneous[sch.schema_hash], bd=bd):
                    if beta0 is PAUSE:
                        continue
                    if hb_only and not has_hb(beta0):
                        continue
                    R_inst: Counter = Counter()
                    for (p, n), h in trace:
                        R_inst[h] += n
                    work['generated'] += 1
                    region_key = tuple(sorted((v1.digest(arena, h).hex(), c) for h, c in R_inst.items()))
                    site_key = ('rw', ctx_digest, region_key)
                    reals, cut = realizations(arena, beta0, below(arena, R_inst) if has_hb(beta0) else None)
                    rst['realizations'] += len(reals)
                    rst['realizations_truncated'] += cut
                    for beta in reals:
                        tsig = target_signature_g(sch, beta)
                        bdig = binding_digest_g(arena, beta)
                        dkey = (site_key, sch.schema_hash, tsig)
                        order = (hashlib.sha256(canonical_json([site_key, sch.schema_hash, tsig, bdig]).encode()).hexdigest(),
                                 site_key, sch.schema_hash, tsig, bdig)
                        c = RM.Candidate(ctx, sch, dict(beta), Counter(R_inst), site_key, tsig, bdig, order)
                        if has_hb(beta) and dependency_check(arena, root, c) is not None:
                            rst['rejected_inadmissible'] += 1
                            continue
                        prev = found.get(dkey)
                        if prev is None or bdig < prev.binding_digest:
                            found[dkey] = c
            except S.Cut as cut:
                work['unify_states'] += budget.states
                if cut.args and cut.args[0] == 'attempts':
                    work['cut_attempts'] += 1
                    stop = True
                    break
                work['cut_unify_states'] += 1
                continue
            work['unify_states'] += budget.states
    cands = sorted(found.values(), key=lambda c: c.order_key)
    work['deduped'] = len(cands)
    if truncate and len(cands) > cfg.rewrite_n_max:
        work['cut_output'] = len(cands)-cfg.rewrite_n_max
        cands = cands[:cfg.rewrite_n_max]
    work.pop('cap')
    if handles:
        work.update(hb_candidates=sum(has_hb(c.beta) for c in cands), handle_bindings=bd.stats['handle_bindings'],
                    handle_choices=bd.stats['handle_choices'], keys_computed=bd.stats['keys_computed'],
                    key_collisions=bd.stats['key_collisions'], **rst)
    return cands, work


def latent_candidates_g(arena, root: int, schemas: list, cfg, lcfg, handles: bool, hb_only: bool = False) -> tuple:
    """`rewrite_latent.latent_candidates` over the generic binder. The quotient views are the frozen ones; a view's
    stripped product is matched but a bound unit must lie below the view's original children."""
    info = v1.state_info(arena, root)
    bd = Binder(arena, handles, info.reach_h if handles else frozenset())
    contexts = sorted(info.contexts, key=lambda h: v1.digest(arena, h))
    work = dict(contexts=0, views=0, divisors_tested=0, assignments=0, unify_states=0, generated=0,
                cut_attempts=0, cut_unify_states=0, cut_contexts=max(0, len(contexts)-cfg.match_contexts_max),
                cut_candidates=0)
    shared = L._Shared(lcfg.attempts_max)
    found: dict = {}
    rst = Counter()
    simultaneous = {sch.schema_hash: S.uses_simultaneous(sch) for sch in schemas}
    stop = False
    for ctx in contexts[:cfg.match_contexts_max]:
        if stop:
            break
        work['contexts'] += 1
        ctx_digest = v1.digest(arena, ctx).hex()
        for d, view, origin in L.views(arena, ctx, lcfg):
            if stop:
                break
            work['views'] += 1
            work['divisors_tested'] += 1
            if handles:
                bd.within = frozenset().union(*(v1.state_info(arena, o).reach_h for q in origin for o, c in origin[q]))
            index = GIndex(arena, dict(view), bd)
            for sch in schemas:
                budget = L._ViewBudget(cfg.unify_states_max, shared)
                trace: list = []
                try:
                    for beta0 in assign_g(sch.source, index, dict(view), {}, arena, cfg, budget, exact=False,
                                          trace=trace, simultaneous=simultaneous[sch.schema_hash], bd=bd):
                        if beta0 is PAUSE:
                            continue
                        work['assignments'] += 1
                        if hb_only and not has_hb(beta0):
                            continue
                        vr: Counter = Counter()
                        for (p, n), h in trace:
                            vr[h] += n
                        R = L.allocate(arena, vr, origin)
                        if not R:
                            continue
                        work['generated'] += 1
                        region_key = tuple(sorted((v1.digest(arena, h).hex(), c) for h, c in R.items()))
                        site_key = ('lx', ctx_digest, region_key, tuple(d))
                        reals, cut = realizations(arena, beta0, below(arena, R) if has_hb(beta0) else None)
                        rst['realizations'] += len(reals)
                        rst['realizations_truncated'] += cut
                        for beta in reals:
                            tsig = target_signature_g(sch, beta)
                            bdig = binding_digest_g(arena, beta)
                            dkey = (site_key, sch.schema_hash, tsig)
                            order = (hashlib.sha256(repr((site_key, sch.schema_hash, tsig, bdig)).encode()).hexdigest(),
                                     site_key, sch.schema_hash, tsig, bdig)
                            c = LatentCandidate(ctx, tuple(d), sch, dict(beta), Counter(R), site_key, tsig, bdig, order)
                            if has_hb(beta) and dependency_check(arena, root, c) is not None:
                                rst['rejected_inadmissible'] += 1
                                continue
                            prev = found.get(dkey)
                            if prev is None or bdig < prev.binding_digest:
                                found[dkey] = c
                        if len(found) >= lcfg.candidates_max:
                            work['cut_candidates'] = 1
                            stop = True
                            break
                except S.Cut:
                    work['unify_states'] += budget.states
                    if shared.cut:
                        work['cut_attempts'] += 1
                        stop = True
                        break
                    work['cut_unify_states'] += 1
                    continue
                work['unify_states'] += budget.states
    work['attempts'] = shared.n
    work['deduped'] = len(found)
    out = sorted(found.values(), key=lambda c: c.order_key)
    if handles:
        work.update(hb_candidates=sum(has_hb(c.beta) for c in out), handle_bindings=bd.stats['handle_bindings'],
                    handle_choices=bd.stats['handle_choices'], keys_computed=bd.stats['keys_computed'],
                    key_collisions=bd.stats['key_collisions'], **rst)
    return out, work


# ---------------------------------------------------------------- the fair handle lane (T6A.2)


HANDLE_QUANTUM = 1024      # work ticks a family receives per round-robin turn in the handle lane


class FairLane:
    """A fair-scheduled handle generation lane: ONE global work allowance of the frozen per-state size (never
    widened), shared by the families round-robin, a fixed quantum per family per turn, in canonical family order."""

    def __init__(self, cap: int, quantum: int = HANDLE_QUANTUM) -> None:
        self.cap, self.quantum = cap, quantum
        self.used = 0
        self.quantum_used = 0
        self.stop_cause = None
        self.rounds = 0
        self.fam: dict = {}


class LaneBudget(S.Budget):
    """The frozen per-(context, surface[, view]) unify cap, charged to the lane's global allowance; a pause is
    requested at the next tick point once the family's quantum is spent (the generator resumes where it stopped)."""
    __slots__ = ('lane',)

    def __init__(self, cap: int, lane: FairLane) -> None:
        super().__init__(cap)
        self.lane = lane

    def tick(self) -> None:
        self.states += 1
        ln = self.lane
        ln.used += 1
        ln.quantum_used += 1
        if ln.used > ln.cap:
            raise S.Cut('lane_work')
        if self.states > self.cap:
            raise S.Cut('unify_states')

    def pause_due(self) -> bool:
        return self.lane.quantum_used >= self.lane.quantum


def families(schemas: list) -> list:
    """[(family id, exact relation payload, surfaces)] — surfaces grouped by EXACT AC relation, families in canonical
    order (id, payload), surfaces in legacy-id order. A function of the schema SET, never of its order."""
    from .relation import canonical_family_id_ac, family_payload
    fams: dict = {}
    for s in schemas:
        fams.setdefault(family_payload(s), []).append(s)
    return sorted(((canonical_family_id_ac(v[0]), p, sorted(v, key=lambda s: s.schema_hash)) for p, v in fams.items()),
                  key=lambda x: (x[0], x[1]))


def _schedule(gens: list, lane: FairLane, emit) -> None:
    """Deterministic round-robin over live family generators: each runs until its quantum is spent (it yields PAUSE)
    or it finishes. `emit` returns True when an output cap is reached. The global allowance ends the lane."""
    live = list(gens)
    while live and lane.stop_cause is None:
        lane.rounds += 1
        nxt = []
        for fkey, g in live:
            st = lane.fam[fkey]
            lane.quantum_used = 0
            before = lane.used
            done = True
            try:
                for item in g:
                    if item is PAUSE:
                        done = False
                        break
                    st['matches'] += 1
                    if emit(item):
                        lane.stop_cause = 'candidate_cap'
                        done = False
                        break
            except S.Cut:
                lane.stop_cause = 'work_cap'
                done = False
            st['work'] += lane.used-before
            # a pause is honoured at the first pause point after the quantum is spent; ticks inside the frozen mainline
            # enumerator `rewrite_schema.simultaneous_bindings` cannot pause between two of its yields: record the
            # largest excess of one turn over the quantum
            st['overrun'] = max(st['overrun'], lane.used-before-lane.quantum)
            st['turns'] += 1
            if done:
                st['complete'] = 1
            else:
                nxt.append((fkey, g))
            if lane.stop_cause is not None:
                break
        live = nxt


def _family_matches(arena, items, surfaces, cfg, lane, fst, simul, latent: bool):
    """One family's handle-bound matches over the lane's (context[, view]) items, pausable. Each item is
    (context, context digest, d or None, children multiset, view origin or None, index)."""
    for ctx, ctx_digest, d, counts, origin, index in items:
        for sch in surfaces:
            budget = LaneBudget(cfg.unify_states_max, lane)
            trace: list = []
            try:
                for beta0 in assign_g(sch.source, index, dict(counts), {}, arena, cfg, budget, exact=False,
                                      trace=trace, simultaneous=simul[sch.schema_hash], bd=index.bd):
                    if beta0 is PAUSE:
                        yield PAUSE
                        continue
                    if not has_hb(beta0):
                        continue
                    R: Counter = Counter()
                    for (p, n), h in trace:
                        R[h] += n
                    if latent:
                        R = L.allocate(arena, R, origin)
                        if not R:
                            continue
                    yield (ctx, ctx_digest, d, sch, beta0, R)
            except S.Cut as cut:
                fst['unify_states'] += budget.states
                if cut.args and cut.args[0] == 'lane_work':
                    raise
                fst['cut_unify_states'] += 1
                continue
            fst['unify_states'] += budget.states


def _emitter(arena, root: int, found: dict, st: Counter, cap: int | None):
    """Realization expansion inside THIS candidate's consumed region, the shared apply-time admissibility predicate,
    and exact surface-independent concrete dedup (all supporting surfaces kept as provenance)."""
    def emit(item) -> bool:
        ctx, ctx_digest, d, sch, beta0, R = item
        reals, cut = realizations(arena, beta0, below(arena, R))
        st['realizations_truncated'] += cut
        st['inadmissible_matches'] += not reals
        region_key = tuple(sorted((v1.digest(arena, h).hex(), c) for h, c in R.items()))
        for beta in reals:
            tsig, bdig = target_signature_g(sch, beta), binding_digest_g(arena, beta)
            if d is None:
                site_key = ('rw', ctx_digest, region_key)
                order = (hashlib.sha256(canonical_json([site_key, sch.schema_hash, tsig, bdig]).encode()).hexdigest(),
                         site_key, sch.schema_hash, tsig, bdig)
                c = GCandidate(ctx, sch, dict(beta), Counter(R), site_key, tsig, bdig, order)
            else:
                site_key = ('lx', ctx_digest, region_key, d)
                order = (hashlib.sha256(repr((site_key, sch.schema_hash, tsig, bdig)).encode()).hexdigest(),
                         site_key, sch.schema_hash, tsig, bdig)
                c = GLatentCandidate(ctx, d, sch, dict(beta), Counter(R), site_key, tsig, bdig, order)
            if dependency_check(arena, root, c) is not None:
                st['rejected_inadmissible'] += 1
                continue
            st['realizations'] += 1
            k = concrete_key(c)
            prev = found.get(k)
            if prev is None:
                c.surfaces = (sch.schema_hash,)
                found[k] = c
                continue
            st['merged_cross_surface' if sch.schema_hash not in prev.surfaces else 'merged_same_surface'] += 1
            surfaces = tuple(sorted(set(prev.surfaces) | {sch.schema_hash}))
            if c.order_key < prev.order_key:
                c.surfaces = surfaces
                found[k] = c
            else:
                prev.surfaces = surfaces
        return cap is not None and len(found) >= cap
    return emit


def _run_lane(arena, root, schemas, cfg, items, cap_work, quantum, latent, cap_out, extra) -> tuple:
    lane = FairLane(cap_work, quantum)
    simul = {s.schema_hash: S.uses_simultaneous(s) for s in schemas}
    found: dict = {}
    st: Counter = Counter()
    gens = []
    fams = families(schemas)
    for fid, payload, surfaces in fams:
        fkey = (fid, payload)
        lane.fam[fkey] = Counter()
        gens.append((fkey, _family_matches(arena, items, surfaces, cfg, lane, lane.fam[fkey], simul, latent)))
    _schedule(gens, lane, _emitter(arena, root, found, st, cap_out))
    cands = sorted(found.values(), key=lambda c: c.order_key)
    fam = [dict(family=fk[0], complete=bool(f['complete']), work=f['work'], turns=f['turns'], matches=f['matches'],
                overrun=f['overrun'], cut_unify_states=f['cut_unify_states'])
           for fk, f in ((fk, lane.fam[fk]) for fk, g in gens)]
    work = dict(extra, attempts=lane.used, cap=lane.cap, quantum=quantum, rounds=lane.rounds,
                work_cap=int(lane.stop_cause == 'work_cap'), candidate_cap=int(lane.stop_cause == 'candidate_cap'),
                cut_attempts=int(lane.stop_cause == 'work_cap'),
                unify_states=lane.used, cut_unify_states=sum(f['cut_unify_states'] for f in fam),
                matches=sum(f['matches'] for f in fam), generated=sum(f['matches'] for f in fam), deduped=len(cands),
                hb_candidates=len(cands),
                families=len(fam), families_complete=sum(f['complete'] for f in fam), family_records=fam, **st)
    return cands, work


def handle_visible(arena, root: int, schemas: list, cfg, quantum: int = HANDLE_QUANTUM) -> tuple:
    """Handle-bound VISIBLE candidates of one state under the fair scheduler (global allowance
    `match_attempts_max`, the frozen per-(context, surface) unify cap, contexts in digest order up to the frozen
    context cap)."""
    info = v1.state_info(arena, root)
    bd = Binder(arena, True, info.reach_h)
    contexts = sorted(info.contexts, key=lambda h: v1.digest(arena, h))
    cut_contexts = max(0, len(contexts)-cfg.match_contexts_max)
    items = []
    for ctx in contexts[:cfg.match_contexts_max]:
        counts = dict(v1.items(arena.handles[ctx]))
        items.append((ctx, v1.digest(arena, ctx).hex(), None, counts, None, GIndex(arena, counts, bd)))
    cands, work = _run_lane(arena, root, schemas, cfg, items, cfg.match_attempts_max, quantum, False, None,
                            dict(contexts=len(items), cut_contexts=cut_contexts))
    work.update(handle_choices=bd.stats['handle_choices'], keys_computed=bd.stats['keys_computed'],
                key_collisions=bd.stats['key_collisions'])
    return cands, work


def divisor_truncated(arena, ctx: int, lcfg) -> int:
    """1 iff the frozen quotient-view construction (`rewrite_latent.views`) cuts this context's eligible divisors at
    `divisors_max` (read-only replica of its share count)."""
    share: Counter = Counter()
    for h in dict(v1.items(arena.handles[ctx])):
        e = L.exposed_mono_vec(arena, h)
        if e is None:
            continue
        for d in L.divisors(e, lcfg.divisor_degree_max):
            share[d] += 1
    return int(sum(1 for d, n in share.items() if n >= lcfg.min_children) > lcfg.divisors_max)


def handle_latent(arena, root: int, schemas: list, cfg, lcfg, quantum: int = HANDLE_QUANTUM) -> tuple:
    """Handle-bound LATENT candidates of one state under the fair scheduler (global allowance `attempts_max`, output
    cap `candidates_max`, the frozen views). Each view has its own binder whose units must lie below the view's
    originals; the exact region check against the allocated originals is applied at emission."""
    info = v1.state_info(arena, root)
    contexts = sorted(info.contexts, key=lambda h: v1.digest(arena, h))
    cut_contexts = max(0, len(contexts)-cfg.match_contexts_max)
    items, views, divisor_cut = [], 0, 0
    for ctx in contexts[:cfg.match_contexts_max]:
        divisor_cut += divisor_truncated(arena, ctx, lcfg)
        ctx_digest = v1.digest(arena, ctx).hex()
        for d, view, origin in L.views(arena, ctx, lcfg):
            views += 1
            vb = Binder(arena, True, info.reach_h)
            vb.within = frozenset().union(*(v1.state_info(arena, o).reach_h for q in origin for o, c in origin[q]))
            items.append((ctx, ctx_digest, tuple(d), dict(view), origin, GIndex(arena, dict(view), vb)))
    cands, work = _run_lane(arena, root, schemas, cfg, items, lcfg.attempts_max, quantum, True, lcfg.candidates_max,
                            dict(contexts=len(contexts[:cfg.match_contexts_max]), cut_contexts=cut_contexts,
                                 views=views, divisor_cap=divisor_cut))
    work.update(handle_choices=sum(it[5].bd.stats['handle_choices'] for it in items),
                keys_computed=sum(it[5].bd.stats['keys_computed'] for it in items),
                key_collisions=sum(it[5].bd.stats['key_collisions'] for it in items))
    return cands, work


# ---------------------------------------------------------------- lowering and application


def lower_g(p, beta: dict, arena) -> int:
    """`rewrite_schema.lower`; a handle-bound unit is reused as a child, never rebuilt. A product leaf inside a product
    pattern is spliced into that product (associative MUL)."""
    if p[0] == 'M':
        if not any(isinstance(beta[i], HB) for i, e in p[1]):
            return S.lower(p, beta, arena)
        units: Counter = Counter()
        v = [0]*NV
        for i, e in p[1]:
            b = beta[i]
            if isinstance(b, HB):
                units[b.h] += e
            else:
                for j in range(NV):
                    v[j] += e*b[j]
        if max(v) > 6:
            raise S.CapError('monomial exponent above inherited cap')
        if any(v):
            units[v1.mk_mono(arena, v1.pack(tuple(v)))] += 1
        return v1.mk_prod(arena, units)
    kids: Counter = Counter()
    for c, n in p[1]:
        if p[0] == 'P' and c[0] == 'M' and any(isinstance(beta[i], HB) for i, e in c[1]):
            v = [0]*NV
            for i, e in c[1]:
                b = beta[i]
                if isinstance(b, HB):
                    kids[b.h] += e*n
                else:
                    for j in range(NV):
                        v[j] += e*n*b[j]
            if max(v) > 6:
                raise S.CapError('monomial exponent above inherited cap')
            if any(v):
                kids[v1.mk_mono(arena, v1.pack(tuple(v)))] += 1
            continue
        kids[lower_g(c, beta, arena)] += n
    return (v1.mk_sum if p[0] == 'S' else v1.mk_prod)(arena, kids)


def lower_region_g(r, beta: dict, arena) -> Counter:
    if not has_hb(beta):
        return S.lower_region(r, beta, arena)
    out: Counter = Counter()
    for c, n in r:
        out[lower_g(c, beta, arena)] += n
    return out


def dependency_check(arena, root: int, cand) -> str | None:
    """Fail-closed apply-time checks for handle-bound parameters: the bound handle is a compound handle of the
    concrete state, lies inside the consumed region, does not contain the rewritten context (no capture / self
    reference), and still has the exact value it was matched with."""
    hbs = [b for b in cand.beta.values() if isinstance(b, HB)]
    if not hbs:
        return None
    reach_root = v1.state_info(arena, root).reach_h
    below: set = set()
    for r in cand.R_inst:
        below |= v1.state_info(arena, r).reach_h
    for b in hbs:
        if not (0 <= b.h < len(arena.handles)) or b.h not in reach_root:
            return 'dep_unreachable'
        if b.h == cand.ctx or cand.ctx in v1.state_info(arena, b.h).reach_h:
            return 'dep_capture'
        if b.h not in below:
            return 'dep_outside_region'
        if isinstance(arena.handles[b.h], v1.Mono) or arena.constant[b.h]:
            return 'dep_not_compound'
        if hkey(arena, b.h) != b.key or v1.digest(arena, b.h).hex() != b.dig:
            return 'dep_key_mismatch'
    return None


def reuse_check(arena, sch, beta: dict, T_inst: Counter) -> str | None:
    """No operand rebuild: every handle-bound parameter that occurs in the target is a node of the lowered target."""
    used = {i for c, n in sch.target for i in S.holes(c)}
    need = {beta[i].h for i in used if isinstance(beta[i], HB)}
    return None if need <= v1.reachable(arena, list(T_inst)) else 'rebuild'


def apply_candidate_g(arena, root: int, cand, target_poly: dict) -> tuple:
    if not has_hb(cand.beta):
        return RM.apply_candidate(arena, root, cand, target_poly)
    why = dependency_check(arena, root, cand)
    if why:
        return None, why
    try:
        T_inst = lower_region_g(cand.schema.target, cand.beta, arena)
    except (S.CapError, AssertionError):
        return None, 'cap'
    why = reuse_check(arena, cand.schema, cand.beta, T_inst)
    if why:
        return None, why
    counts = Counter(dict(v1.items(arena.handles[cand.ctx])))
    counts.subtract(cand.R_inst)
    if any(c < 0 for c in counts.values()):
        return None, 'not_a_match'
    counts.update(T_inst)
    ctx2 = v1.mk_sum(arena, Counter({h: c for h, c in counts.items() if c > 0}))
    if ctx2 == cand.ctx:
        return None, 'noop'
    root2 = v1.substitute(arena, root, cand.ctx, ctx2)
    if root2 == root:
        return None, 'noop'
    if not v1.poly_equal(v1.hpoly(arena, root2), target_poly):
        return None, 'inexact'
    return root2, 'ok'


def apply_latent_g(arena, root: int, cand, target_poly: dict) -> tuple:
    if not has_hb(cand.beta):
        return L.apply_latent(arena, root, cand, target_poly)
    why = dependency_check(arena, root, cand)
    if why:
        return None, why
    try:
        T = lower_region_g(cand.schema.target, cand.beta, arena)
    except (S.CapError, AssertionError):
        return None, 'cap'
    why = reuse_check(arena, cand.schema, cand.beta, T)
    if why:
        return None, why
    tsum = v1.mk_sum(arena, Counter(T))
    dm = v1.mk_mono(arena, v1.pack(cand.d))
    repl = v1.mk_prod(arena, Counter({dm: 1, tsum: 1}))
    counts = Counter(dict(v1.items(arena.handles[cand.ctx])))
    counts.subtract(cand.R_inst)
    if any(c < 0 for c in counts.values()):
        return None, 'not_a_match'
    counts[repl] += 1
    ctx2 = v1.mk_sum(arena, Counter({h: c for h, c in counts.items() if c > 0}))
    if ctx2 == cand.ctx:
        return None, 'noop'
    root2 = v1.substitute(arena, root, cand.ctx, ctx2)
    if root2 == root:
        return None, 'noop'
    if not v1.poly_equal(v1.hpoly(arena, root2), target_poly):
        return None, 'inexact'
    return root2, 'ok'


def apply_any_g(arena, root: int, cand, target_poly: dict) -> tuple:
    if isinstance(cand, LatentCandidate):
        return apply_latent_g(arena, root, cand, target_poly)
    return apply_candidate_g(arena, root, cand, target_poly)


# ---------------------------------------------------------------- semantic action identity and the action record


def family_frames(sch) -> tuple:
    """(CANONICAL_SEMANTIC_FAMILY_ID v2 — the AC-normalized relation of `canon.relation` —, the parameter permutations
    realizing its canonical form; they fix the canonical parameter order)."""
    from .relation import ac_forms, canonical_family_id_ac
    k, best, frames = ac_forms(sch)
    return canonical_family_id_ac(sch), frames


def param_keys(sch, beta: dict) -> tuple:
    """The parameters' semantic keys in the family's canonical parameter order (min over the relation's symmetries)."""
    fam, frames = family_frames(sch)
    k = len(sch.types)
    keys = [param_key(beta[i]) for i in range(k)]
    return min(tuple(keys[img.index(j)] for j in range(k)) for img in frames)


def action_identity(arena, cand) -> str:
    """SEMANTIC_ACTION_KEY of a rewrite candidate (the digest of its exact semantic payload — `canon.action_record`)."""
    from .action_record import rewrite_keys
    return rewrite_keys(arena, cand)[0]


def realization_key(arena, cand) -> str:
    """CONCRETE_REALIZATION_KEY of a rewrite candidate (the digest of its exact, surface-independent concrete
    successor payload — `canon.action_record`)."""
    from .action_record import rewrite_keys
    return rewrite_keys(arena, cand)[1]


def origin(cand) -> str:
    return 'latent' if isinstance(cand, LatentCandidate) else 'visible'


def describe(arena, cand) -> dict:
    """Provenance of one action instance (binding metadata lives here, never in the family identity)."""
    from .archive import canonical_family_id
    fam, frames = family_frames(cand.schema)
    return dict(kind='LX' if isinstance(cand, LatentCandidate) else 'RW', legacy_surface_id=cand.schema.schema_hash,
                canonical_semantic_family_id=canonical_family_id(cand.schema), canonical_semantic_family_id_ac=fam,
                bindings={str(i): dict(mode=mode(b), key=param_key(b),
                                       value=v1.handles_mod.to_expr(arena, b.h) if isinstance(b, HB) else list(b))
                          for i, b in sorted(cand.beta.items())},
                parameter_keys_family_order=list(param_keys(cand.schema, cand.beta)),
                d=list(cand.d) if isinstance(cand, LatentCandidate) else None,
                context=v1.digest(arena, cand.ctx).hex(), region=[list(x) for x in _region_key(arena, cand.R_inst)],
                action_identity=action_identity(arena, cand), realization_key=realization_key(arena, cand),
                order_key=cand.order_key[0])


# ---------------------------------------------------------------- the handle lane: separate budget and admission


def admit(arena, cands: list, cap: int, per_class: int = REALIZATIONS_MAX) -> tuple:
    """Semantic-class-level admission of the handle lane. Candidates are grouped into semantic action classes by the
    EXACT semantic key (`semantic_key_exact`: digests index nothing here), members = distinct concrete
    realizations, at most `per_class` in order-key order. Classes are bucketed by (origin, AC family: id then exact
    relation payload) — never by surface or enumeration position — and admitted whole, one class per non-empty
    bucket per round in canonical bucket order, until at least `cap` realizations are admitted. Deterministic;
    returns (admitted in order-key order, stats); each admitted candidate carries `class_size` (provenance)."""
    from .relation import family_payload
    classes: dict = {}
    for c in sorted(cands, key=lambda c: c.order_key):
        classes.setdefault(semantic_key_exact(arena, c), []).append(c)
    st = Counter(classes=len(classes), realizations=len(cands))
    buckets: dict = {}
    for sk, mem in classes.items():
        if len(mem) > per_class:
            st['class_realizations_dropped'] += len(mem)-per_class
            mem = mem[:per_class]
        for c in mem:
            c.class_size = len(mem)
        s0 = mem[0].schema
        buckets.setdefault((origin(mem[0]), family_frames(s0)[0], family_payload(s0)), []).append(mem)
    for b in buckets.values():
        b.sort(key=lambda mem: mem[0].order_key)
    taken: list = []
    keys = sorted(buckets)
    while len(taken) < cap and any(buckets[k] for k in keys):
        for k in keys:
            if len(taken) >= cap:
                break
            if buckets[k]:
                mem = buckets[k].pop(0)
                taken += mem
                st['classes_admitted'] += 1
    st['admitted'] = len(taken)
    st['dropped'] = len(cands)-len(taken)
    return sorted(taken, key=lambda c: c.order_key), st


def handle_lane(arena, root: int, schemas: list, cfg, lcfg, quantum: int = HANDLE_QUANTUM) -> tuple:
    """REACHABLE_HANDLE_BIND candidates of one concrete state: the fair-scheduled generic binder with its OWN work
    allowances (visible `match_attempts_max`, latent `attempts_max`: the frozen sizes, separate from the legacy
    lane's), handle-bound candidates only, admissible for their exact consumed region, deduplicated by exact concrete
    successor, then semantic-class admission with its own cap (`rewrite_n_max`)."""
    vis, w = handle_visible(arena, root, schemas, cfg, quantum)
    lat, lw = [], {}
    if lcfg is not None:
        lat, lw = handle_latent(arena, root, schemas, cfg, lcfg, quantum)
        n0 = len(lat)
        lat = L.dedupe_against_visible(arena, lat, vis)
        lw['deduped_against_visible'] = n0-len(lat)
    adm, st = admit(arena, vis+lat, cfg.rewrite_n_max)
    return adm, dict(visible=w, latent=lw, admission=st, n_visible=len(vis), n_latent=len(lat))


def count_lane(s, st: dict) -> None:
    """Handle-lane bookkeeping on a search summary (the legacy counters are the mainline ones, untouched)."""
    w, lw, a = st['visible'], st['latent'], st['admission']
    s.gen_matcher_contexts += w.get('contexts', 0)
    s.gen_matcher_bindings += w.get('matches', 0)
    s.gen_matcher_unify_states += w.get('unify_states', 0)
    s.gen_latent_bindings += lw.get('matches', 0)
    s.gen_latent_unify_states += lw.get('unify_states', 0)
    s.gen_handle_choices += w.get('handle_choices', 0)+lw.get('handle_choices', 0)
    s.gen_keys_computed += w.get('keys_computed', 0)+lw.get('keys_computed', 0)
    s.gen_key_collisions += w.get('key_collisions', 0)+lw.get('key_collisions', 0)
    s.gen_matcher_cuts += w.get('cut_unify_states', 0)+w.get('work_cap', 0)
    s.gen_latent_cuts += lw.get('cut_unify_states', 0)+lw.get('work_cap', 0)+lw.get('candidate_cap', 0)
    s.hb_visible_candidates += st['n_visible']
    s.hb_latent_candidates += st['n_latent']
    s.hb_realizations_truncated += w.get('realizations_truncated', 0)+lw.get('realizations_truncated', 0)
    s.hb_rejected_inadmissible += (w.get('rejected_inadmissible', 0)+lw.get('rejected_inadmissible', 0)
                                   + w.get('inadmissible_matches', 0)+lw.get('inadmissible_matches', 0))
    s.hb_merged_cross_surface += w.get('merged_cross_surface', 0)+lw.get('merged_cross_surface', 0)
    s.hb_families_truncated += ((w.get('families', 0)-w.get('families_complete', 0))
                                + (lw.get('families', 0)-lw.get('families_complete', 0)))
    s.hb_lane_rounds += w.get('rounds', 0)+lw.get('rounds', 0)
    s.hb_semantic_classes += a['classes']
    s.hb_admitted += a['admitted']
    s.hb_admission_dropped += a['dropped']
    s.hb_class_realizations_dropped += a['class_realizations_dropped']


# ---------------------------------------------------------------- temporary scorer adapter (until T7)


def score_combined_g(scorer, arena, info, root, actions, cands):
    """The fixed-vocabulary scorer. Exactly `scorer.score_combined` when no candidate has a handle-bound parameter.
    Otherwise one encoder pass with the same rewrite head; a handle-bound parameter's binding input is the encoder
    embedding of the handle realizing it — V1's own convention for a reachable FACTOR argument — and a monomial
    parameter's is the input embedding of its off-graph monomial row, as today. Temporary until T7 (it is
    fixed-surface, pools parameters by a commutative mean and does not see d, origin, lane or class size)."""
    if not any(has_hb(c.beta) for c in cands) or getattr(scorer, 'model', None) is None:
        return scorer.score_combined(arena, info, root, actions, cands)
    import torch
    from ..rewrite_model import graph_rows
    model = scorer.model
    batch = v1.make_batch([(arena, info, root, actions)])
    with torch.no_grad():
        logits, values = v1.forward(model, batch)
        h = model._last_h
        rows = graph_rows(info, batch.offsets[0])
        feats = []
        for c in cands:
            members = tuple(c.R_inst.items())
            pooled = v1.pool_rows(h, [rows[m] for m, n in members], [n for m, n in members])
            items = sorted(c.beta.items())
            monos = [v1.pack(tuple(b)) for i, b in items if not isinstance(b, HB)]
            parts = []
            if monos:
                parts.append(model.input(torch.tensor([v1.off_graph_mono(m) for m in monos], dtype=h.dtype)))
            hbs = [b for i, b in items if isinstance(b, HB)]
            if hbs:
                parts.append(torch.stack([h[rows[b.h]] for b in hbs]))
            bind = torch.cat(parts).mean(dim=0)
            emb = model.schema_emb(torch.tensor([scorer.schema_index[c.schema.schema_hash]],
                                                dtype=torch.long)).squeeze(0).to(h.dtype)
            feats.append(torch.cat([h[rows[c.ctx]], pooled, emb,
                                    torch.tensor(scorer.features[c.schema.schema_hash], dtype=h.dtype), bind]))
        rw = model.rw_scorer(torch.stack(feats)).squeeze(1)
    return [float(x) for x in logits], [float(x) for x in rw]


# ---------------------------------------------------------------- serialization helpers (used by canon.latent_identity)


def enc_binding(b, index: dict):
    return ['H', index[b.h]] if isinstance(b, HB) else list(b)


def dec_binding(b, ids: list, arena):
    if b and b[0] == 'H':
        return hb(arena, ids[b[1]])
    return tuple(b)
