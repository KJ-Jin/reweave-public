"""Semantic interning and cheapest-representative normalization.

Reachable handles are grouped into semantic classes by the key of their exact canonical polynomial. Each class keeps a
bounded portfolio of structurally distinct representatives (K_REP = 4). Normalization replaces a handle by a cheaper
representative of its class and is accepted only on a strict whole-state gate-cost decrease.

Correctness never depends on the semantic key: `Portfolio.instantiate` exact-checks every relabelled representative
against the handle it would replace.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace

from .. import v1
from .labelling import IDENT, NV, compose, inverse, key_of, poly_frame, rdigest, relabel_into

K_REP, V_GUARD = 4, 6                    # frozen C7.2I claim §3, §6
O = SimpleNamespace(IDENT=IDENT, NV=NV, rdigest=rdigest, inverse=inverse, relabel_into=relabel_into, compose=compose)


# ---------------------------------------------------------------- handle keys (c7_2i_intern @ 0bf1da7)

@dataclass
class HKey:
    key: str
    T: tuple
    n_used: int
    D: bytes
    f0: tuple
    f0inv: tuple
    n_frames: int


class HandleKeys:
    """Per-arena memo of handle keys and canonical structure, with the frozen variable guard (§3)."""

    def __init__(self, arena) -> None:
        self.a = arena
        self.pcache: dict = {}
        self.full_memo: dict = {}
        self.used_memo: dict = {}
        self.rmemo: dict = {}
        self.stats = Counter()

    def used(self, h: int) -> int:
        n = self.used_memo.get(h)
        if n is None:
            mask = 0
            for m in v1.hpoly(self.a, h):
                for i, x in enumerate(v1.unpack(m)):
                    if x:
                        mask |= 1 << i
            n = self.used_memo[h] = bin(mask).count('1')
        return n

    def full(self, h: int):
        """The key of h, or None if the variable guard or a frozen cap excludes it."""
        r = self.full_memo.get(h)
        if r is not None:
            return r if r is not False else None
        if self.used(h) > V_GUARD:
            self.stats['var_guard_skipped'] += 1
            self.full_memo[h] = False
            return None
        poly = v1.hpoly(self.a, h)
        pk = frozenset(poly.items())
        fr = self.pcache.get(pk)
        if fr is None:
            try:
                fr = self.pcache[pk] = poly_frame(poly)
            except AssertionError:
                self.stats['frame_excluded'] += 1
                self.full_memo[h] = False
                return None
            self.stats['frames_computed'] += 1
        pi, T, G, frames, n = fr
        digs = [O.rdigest(self.a, h, f, self.rmemo.setdefault(f, {})) for f in frames]
        D = min(digs)
        f0 = frames[digs.index(D)]
        self.stats['keys_computed'] += 1
        r = self.full_memo[h] = HKey(key_of(T), T, n, D, f0, O.inverse(f0), len(frames))
        return r



# ---------------------------------------------------------------- frozen helpers (c7_2i_intern @ 0bf1da7)


def built_handles(arena, roots, pre) -> set:
    out, todo = set(), list(roots)
    while todo:
        h = todo.pop()
        if h in pre or h in out or isinstance(arena.handles[h], v1.Mono):
            continue
        out.add(h)
        todo.extend(c for c, n in v1.items(arena.handles[h]))
    return out


def reach(arena, h) -> set:
    return v1.state_info(arena, h).reach_h


def _nonmono(arena, h) -> bool:
    return not isinstance(arena.handles[h], v1.Mono)


# ---------------------------------------------------------------- the bounded portfolio (c7_2i_intern @ 0bf1da7)


@dataclass
class Rep:
    D: str
    cost: int
    arena: object
    h: int
    origin: str                      # CURRENT | LEARNED_REWRITE | SEMANTIC_NORMALIZATION | HISTORICAL
    source: str
    step: int
    f0: tuple = IDENT                # the frame mapping this representative's own variables into the key frame
    norm_uses: int = 0
    view_uses: int = 0
    useful: int = 0


@dataclass
class SemClass:
    key: str
    T: tuple
    created: int
    reps: dict = field(default_factory=dict)
    evicted: list = field(default_factory=list)
    dup: int = 0
    historical: int = 0
    inserted: int = 0


class Portfolio:
    def __init__(self, ca, hk: HandleKeys, archive, arch_arena) -> None:
        self.ca, self.hk, self.archive, self.arch_arena = ca, hk, archive, arch_arena
        self.classes: dict = {}
        self.interned: set = set()
        self.stats = Counter()
        self.failures: list = []
        self.max_reps = 0

    # ---- interning
    def intern_state(self, root: int, step: int, origin: str, norm_roots=()) -> None:
        ca = self.ca
        norm_reach = set()
        for x in norm_roots:
            norm_reach |= reach(ca, x)
        for h in sorted(reach(ca, root)):
            if h == root or not _nonmono(ca, h) or h in self.interned:
                continue
            self.interned.add(h)
            self.stats['lookups'] += 1
            k = self.hk.full(h)
            if k is None:
                continue
            cls = self.classes.get(k.key)
            if cls is None:
                cls = self.classes[k.key] = SemClass(k.key, k.T, step)
                self.stats['classes_created'] += 1
                self._seed(cls, k)
            elif cls.T != k.T:
                self.stats['hash_collision'] += 1          # exact-equality control (§3)
                continue
            org = 'SEMANTIC_NORMALIZATION' if h in norm_reach else origin
            self._insert(cls, Rep(k.D.hex(), v1.cost(ca, h), ca, h, org, 'current', step, f0=k.f0))

    def _seed(self, cls: SemClass, k: HKey) -> None:
        e = self.archive.get(cls.key) if self.archive else None
        if not e:
            return
        if tuple((tuple(v), c) for v, c in e['T']) != cls.T:
            self.stats['archive_collision'] += 1
            return
        for r in e['reps']:
            self._insert(cls, Rep(r['D'], r['cost'], self.arch_arena, r['_root'], 'HISTORICAL',
                                  ','.join(r['sources'][:3]), -1))
            cls.historical += 1
            self.stats['historical_reps'] += 1

    def _insert(self, cls: SemClass, rep: Rep) -> None:
        cls.inserted += 1
        if rep.D in cls.reps:
            cls.dup += 1
            self.stats['structural_dup'] += 1
            return
        cls.reps[rep.D] = rep
        self.stats['reps_inserted'] += 1
        if len(cls.reps) > K_REP:
            worst = max(cls.reps.values(), key=lambda r: (r.cost, r.D))
            del cls.reps[worst.D]
            cls.evicted.append(dict(D=worst.D, cost=worst.cost, origin=worst.origin, at=rep.step))
            self.stats['evicted'] += 1
        self.max_reps = max(self.max_reps, len(cls.reps))
        assert len(cls.reps) <= K_REP, 'portfolio bound (§6)'

    # ---- lookups
    def cls_of(self, h: int):
        k = self.hk.full(h)
        if k is None:
            return None, None
        cls = self.classes.get(k.key)
        if cls is None or cls.T != k.T:
            return None, k
        return cls, k

    def best(self, h: int):
        cls, k = self.cls_of(h)
        if cls is None:
            return None, k
        c_h = v1.cost(self.ca, h)
        cands = [r for r in cls.reps.values() if r.cost < c_h and r.D != k.D.hex()]
        return (min(cands, key=lambda r: (r.cost, r.D)) if cands else None), k

    def alts(self, h: int):
        cls, k = self.cls_of(h)
        if cls is None:
            return [], k
        return sorted([r for r in cls.reps.values() if r.D != k.D.hex()], key=lambda r: (r.cost, r.D))[:K_REP-1], k

    def instantiate(self, rep: Rep, k: HKey, h: int):
        """Build rep in the current frame as an ordinary concrete DAG and exact-verify ⟦r_h⟧ = ⟦h⟧ (§5C).

        The representative lives in its own frame f0(rep), so the mapping into h's variables is
        f0(h)⁻¹ ∘ f0(rep): key frame first, then back into the current frame.
        """
        ca = self.ca
        self.stats['instantiations'] += 1
        h2 = O.relabel_into(rep.arena, rep.h, O.compose(k.f0inv, rep.f0), ca, {})
        if not v1.poly_equal(v1.hpoly(ca, h2), v1.hpoly(ca, h)):
            self._fail(dict(kind='instantiate', key=k.key, D=rep.D, origin=rep.origin))
            return None
        k2 = self.hk.full(h2)
        if k2 is not None and k2.D.hex() != rep.D:
            self._fail(dict(kind='instantiate_structure', key=k.key, D=rep.D, got=k2.D.hex(), origin=rep.origin))
            return None
        return h2

    def _fail(self, rec) -> None:
        self.stats['failures'] += 1
        if len(self.failures) < 50:
            self.failures.append(rec)


# ---------------------------------------------------------------- eager normalization (c7_2i_intern @ 0bf1da7)


def normalize(ctx, state: int, new_handles, step: int):
    """One pass over newly built handles; accept only strict whole-state C_sf decreases."""
    ca, P = ctx.ca, ctx.port
    cur, events = state, []
    t0 = time.process_time()
    for h in sorted(new_handles, key=lambda x: (-v1.cost(ca, x), v1.digest(ca, x))):
        if h == cur:
            continue
        live = reach(ca, cur)
        if h not in live:
            continue
        rep, k = P.best(h)
        if rep is None:
            continue
        P.stats['norm_attempts'] += 1
        h2 = P.instantiate(rep, k, h)
        if h2 is None or h2 == h:
            continue
        cand = v1.substitute(ca, cur, h, h2)
        c0, c1 = v1.cost(ca, cur), v1.cost(ca, cand)
        if c1 >= c0:
            P.stats['norm_rejected_not_cheaper'] += 1
            continue
        if not v1.poly_equal(v1.hpoly(ca, cand), ctx.Tpoly):
            P._fail(dict(kind='normalize', key=k.key, D=rep.D))
            continue
        P.stats['norm_success'] += 1
        rep.norm_uses += 1
        events.append(dict(h=h, h2=h2, key=k.key, D_old=k.D.hex(), D_new=rep.D, origin=rep.origin, source=rep.source,
                           c_before=c0, c_after=c1, shared=h2 in live, step=step))
        cur = cand
    ctx.norm_cpu += time.process_time()-t0
    return cur, events



# ---------------------------------------------------------------- integration context (new)


class SemanticCtx:
    """The fields `normalize` and the portfolio read (the C7.2I `_Ctx` subset used without views)."""

    def __init__(self, arena, target: dict) -> None:
        self.ca = arena
        self.Tpoly = target
        self.hk = HandleKeys(arena)
        self.port = Portfolio(arena, self.hk, None, None)
        self.norm_cpu = 0.0
