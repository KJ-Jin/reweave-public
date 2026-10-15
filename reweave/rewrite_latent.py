"""Latent exposure: matching a relation that is present only up to a monomial multiplier.

The relation matcher normally requires a relation's source to exist as a sub-multiset of the children of a reachable
Sum context. This module additionally runs the UNCHANGED matcher on a bounded, deterministic set of monomial-quotient
VIEWS of a context's children. A match found in view `d` is applied by replacing the matched original children with
`d * T(theta)`, built with ordinary constructors and exact-verified.

In the free commutative semiring nothing cancels, so a hidden factor leaves exactly detectable scaled copies of the
source terms; the quotient view recovers them. `d = 1` is the ordinary visible view and is never constructed here.
Depth is exactly 1: while considering a relation this module never invokes another relation.
"""

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass, field

from . import v1
from . import rewrite_schema as S
from .hashing import sha256_json


@dataclass(frozen=True)
class LatentConfig:
    """Frozen bounds for latent exposure. Separate from `RewriteConfig` by design: adding a field there would change
    `RewriteConfig.hash()` and therefore every frozen stage key, for a capability that is off by default."""
    divisors_max: int = 16          # quotient views per context, in the deterministic order below
    divisor_degree_max: int = 4     # total degree of an exposure multiplier
    min_children: int = 2           # a view is built only if the multiplier divides at least this many children
    attempts_max: int = 200_000     # shared fail-closed work counter per state
    candidates_max: int = 4096      # raw output bound before visible dedupe

    def as_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        return sha256_json(self.as_dict())


@dataclass
class LatentCandidate:
    """Structurally compatible with `rewrite_match.Candidate` for scoring and ordering, but a distinct type: it must
    never be handed to `apply_candidate`, which would drop the multiplier and fail exact verification."""
    ctx: int
    d: tuple
    schema: object
    beta: dict
    R_inst: Counter
    site_key: tuple
    target_signature: str
    binding_digest: str
    order_key: tuple = field(default=())

    @property
    def stable_hash(self) -> str:
        return self.order_key[0] if self.order_key else ''


class _Shared:
    """Fail-closed exposure work counter, shared across every view and schema at one state."""
    __slots__ = ('n', 'cap', 'cut')

    def __init__(self, cap: int) -> None:
        self.n, self.cap, self.cut = 0, cap, False

    def tick(self) -> None:
        self.n += 1
        if self.n > self.cap:
            self.cut = True
            raise S.Cut('exposure_attempts')


class _ViewBudget(S.Budget):
    """Per (context, view, schema) unification budget that also charges the shared exposure counter."""
    __slots__ = ('shared',)

    def __init__(self, cap: int, shared: _Shared) -> None:
        super().__init__(cap)
        self.shared = shared

    def tick(self) -> None:
        self.states += 1
        self.shared.tick()
        if self.states > self.cap:
            raise S.Cut('unify_states')


def exposed_mono_vec(arena, h):
    """The single exposed monomial factor of a child, or None. Mirrors V1 `enumerate.exposed_mono`."""
    node = arena.handles[h]
    if isinstance(node, v1.Mono):
        return v1.unpack(node.m)
    if isinstance(node, v1.Prod):
        for c, n in v1.items(node):
            if isinstance(arena.handles[c], v1.Mono):
                return v1.unpack(arena.handles[c].m)
    return None


def strip_mono(arena, h, d):
    """Remove one exposed monomial factor `d` from a child without crossing a Sum boundary, or None.

    Mirrors V1 `calculus.strip` for a Mono divisor, rebuilt from exported primitives because the V1 adapter does not
    export `strip`. The adapter is not modified."""
    node = arena.handles[h]
    if isinstance(node, v1.Mono):
        e = v1.unpack(node.m)
        if any(e[j] < d[j] for j in range(len(d))):
            return None
        return v1.mk_mono(arena, v1.pack(tuple(e[j]-d[j] for j in range(len(e)))))
    if not isinstance(node, v1.Prod):
        return None
    q = next((c for c, n in v1.items(node) if isinstance(arena.handles[c], v1.Mono)), None)
    if q is None:
        return None
    e = v1.unpack(arena.handles[q].m)
    if any(e[j] < d[j] for j in range(len(d))):
        return None
    fac = Counter(dict(v1.items(node)))
    fac[q] -= 1
    if fac[q] <= 0:
        del fac[q]
    fac[v1.mk_mono(arena, v1.pack(tuple(e[j]-d[j] for j in range(len(e)))))] += 1
    return v1.mk_prod(arena, fac)


def divisors(e, degree_max: int) -> list:
    """Every nonunit vector 0 <= x <= e with total degree <= degree_max, deterministic order."""
    out, cur = [], [0]*len(e)
    idx = [j for j in range(len(e)) if e[j]]

    def go(k, deg):
        if k == len(idx):
            if deg:
                out.append(tuple(cur))
            return
        j = idx[k]
        for x in range(0, e[j]+1):
            if deg+x > degree_max:
                break
            cur[j] = x
            go(k+1, deg+x)
        cur[j] = 0
    go(0, 0)
    return sorted(out, key=lambda x: (sum(x), x))


def views(arena, ctx: int, lcfg: LatentConfig) -> list:
    """The bounded deterministic monomial-quotient views of one Sum context.

    Returns [(d, view_counts, origin)]. `origin[q]` is [(original child, available count), ...] ordered by structural
    digest, so the mapping back to originals preserves multiplicity exactly."""
    counts = dict(v1.items(arena.handles[ctx]))
    emono = {h: exposed_mono_vec(arena, h) for h in counts}
    share: Counter = Counter()
    for h, e in emono.items():
        if e is None:
            continue
        for d in divisors(e, lcfg.divisor_degree_max):
            share[d] += 1
    cand = [d for d, n in share.items() if n >= lcfg.min_children]
    cand.sort(key=lambda d: (-share[d], -sum(d), d))
    out = []
    for d in cand[:lcfg.divisors_max]:
        view: Counter = Counter()
        origin: dict = {}
        for h, c in counts.items():
            if emono[h] is None or any(emono[h][j] < d[j] for j in range(len(d))):
                continue
            q = strip_mono(arena, h, d)
            if q is None:
                continue
            view[q] += c
            origin.setdefault(q, Counter())[h] += c
        if not view or sum(view.values()) < lcfg.min_children:
            continue
        for q in list(origin):
            origin[q] = sorted(origin[q].items(), key=lambda hc: v1.digest(arena, hc[0]))
        out.append((d, view, origin))
    return out


def allocate(arena, view_region: Counter, origin: dict) -> Counter:
    """Map a matched quotient-view region back to original children, preserving multiplicity exactly.

    A single original child may supply several copies of the same quotient; that is precisely what happens when a
    source term carries a coefficient above one. Allocating one copy per distinct original child (the bug found in
    the feasibility prototype) silently discards every such match. Every original that strips to the same quotient
    has the same polynomial, so the choice among them cannot affect exactness; it is fixed by digest for
    determinism."""
    out: Counter = Counter()
    for q, n in view_region.items():
        need = n
        for h, avail in origin.get(q, []):
            if need <= 0:
                break
            take = min(need, avail)
            out[h] += take
            need -= take
        if need > 0:
            return Counter()
    return out


def latent_candidates(arena, root: int, schemas: list, cfg, lcfg: LatentConfig) -> tuple:
    """Bounded depth-1 latent source exposure at one state.

    The unchanged `rewrite_schema.assign` (unchanged binding planner, injectivity, degree bounds and sub-multiset
    semantics) is run on each quotient view. Fail-closed on the shared work counter."""
    info = v1.state_info(arena, root)
    contexts = sorted(info.contexts, key=lambda h: v1.digest(arena, h))
    work = dict(contexts=0, views=0, divisors_tested=0, assignments=0, unify_states=0, generated=0,
                cut_attempts=0, cut_unify_states=0, cut_contexts=max(0, len(contexts)-cfg.match_contexts_max),
                cut_candidates=0)
    shared = _Shared(lcfg.attempts_max)
    found: dict = {}
    simultaneous = {sch.schema_hash: S.uses_simultaneous(sch) for sch in schemas}
    stop = False
    for ctx in contexts[:cfg.match_contexts_max]:
        if stop:
            break
        work['contexts'] += 1
        ctx_digest = v1.digest(arena, ctx).hex()
        for d, view, origin in views(arena, ctx, lcfg):
            if stop:
                break
            work['views'] += 1
            work['divisors_tested'] += 1
            index = S.ChildIndex(arena, dict(view))
            for sch in schemas:
                budget = _ViewBudget(cfg.unify_states_max, shared)
                trace: list = []
                try:
                    for beta in S.assign(sch.source, index, dict(view), {}, arena, cfg, budget, exact=False,
                                         trace=trace, simultaneous=simultaneous[sch.schema_hash]):
                        work['assignments'] += 1
                        vr: Counter = Counter()
                        for (p, n), h in trace:
                            vr[h] += n
                        R = allocate(arena, vr, origin)
                        if not R:
                            continue
                        work['generated'] += 1
                        tsig = S.target_signature(sch, beta)
                        bdig = S.binding_digest(beta)
                        region_key = tuple(sorted((v1.digest(arena, h).hex(), c) for h, c in R.items()))
                        site_key = ('lx', ctx_digest, region_key, tuple(d))
                        dkey = (site_key, sch.schema_hash, tsig)
                        order = (hashlib.sha256(repr((site_key, sch.schema_hash, tsig, bdig)).encode()).hexdigest(),
                                 site_key, sch.schema_hash, tsig, bdig)
                        prev = found.get(dkey)
                        if prev is None or bdig < prev.binding_digest:
                            found[dkey] = LatentCandidate(ctx, tuple(d), sch, dict(beta), R, site_key, tsig, bdig,
                                                          order)
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
    return sorted(found.values(), key=lambda c: c.order_key), work


def dedupe_against_visible(arena, latent: list, visible: list) -> list:
    """The visible candidate always wins. A latent candidate is dropped when an ordinary visible candidate already
    consumes the same region of the same context with the same schema and instantiated target."""
    seen = {(c.ctx, tuple(sorted((v1.digest(arena, h).hex(), n) for h, n in c.R_inst.items())),
             c.schema.schema_hash, c.target_signature) for c in visible}
    return [c for c in latent
            if (c.ctx, tuple(sorted((v1.digest(arena, h).hex(), n) for h, n in c.R_inst.items())),
                c.schema.schema_hash, c.target_signature) not in seen]


def apply_latent(arena, root: int, cand: LatentCandidate, target_poly: dict) -> tuple:
    """Apply one latent candidate with inherited constructors and exact verification.

    `d * T(theta)` replaces exactly the consumed original children. The quotient-view handles used for matching are
    never part of the successor. Nothing is re-canonicalised, no handle is free, and the untouched part of the
    context is carried over unchanged."""
    try:
        T = S.lower_region(cand.schema.target, cand.beta, arena)
    except (S.CapError, AssertionError):
        return None, 'cap'
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


def apply_any(arena, root: int, cand, target_poly: dict) -> tuple:
    """Dispatch to the latent or the ordinary visible application by candidate type."""
    from .rewrite_match import apply_candidate
    if isinstance(cand, LatentCandidate):
        return apply_latent(arena, root, cand, target_poly)
    return apply_candidate(arena, root, cand, target_poly)


def source_poly(arena, cand: LatentCandidate) -> dict:
    """Polynomial of the exposed source S(theta): poly(consumed originals) divided by the multiplier."""
    acc: dict = {}
    for h, c in cand.R_inst.items():
        acc = v1.poly_add(acc, v1.poly_scale(v1.hpoly(arena, h), c))
    dm = {v1.pack(cand.d): 1}
    out: dict = {}
    for m, c in acc.items():
        e = v1.unpack(m)
        out[v1.pack(tuple(e[j]-cand.d[j] for j in range(len(e))))] = c
    return out
