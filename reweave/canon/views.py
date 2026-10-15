"""Bounded alternative representations as VIEW actions, and indexed view retrieval.

A view action substitutes an alternative representative of a semantic class (`canon.semantic`) and then applies a
rewrite that matches only in the alternative: "substitute h -> h', then rewrite a context that exists only in h'".
Retrieval first filters (class, relation) pairs by a cheap structural compatibility index that is a necessary
condition, so it never misses a candidate the eager matcher would find; the exact matcher decides.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

from .. import rewrite_match as RM
from .. import rewrite_schema as S
from .. import v1
from . import identity as Qm
from .labelling import _UF
from .semantic import _nonmono, reach

K_REP, V_GUARD, VIEW_PAIRS_MAX, VIEW_ATTEMPTS = 4, 6, 16, 20000      # frozen C7.2I claim §3, §6, §9
K_VIEW_ACTIVE = 4
O = SimpleNamespace(_UF=_UF)


# ---------------------------------------------------------------- bounded local views (c7_2i_intern @ 0bf1da7)


def match_contexts(arena, root: int, ctxs, schemas, cfg, attempt_cap: int):
    """The frozen `match_state` body, restricted to the given contexts, with a per-view attempt cap.

    Identical dedup (site_key, schema, target signature; smallest binding digest) and identical order key; the
    C7.1b exact hash replicas are used for the signatures.
    """
    info = v1.state_info(arena, root)
    work = dict(contexts=0, attempts=0, cap=attempt_cap, unify_states=0, cut_contexts=0, cut_unify_states=0,
                cut_attempts=0, cut_output=0, generated=0)
    found: dict = {}
    simultaneous = {sch.schema_hash: S.uses_simultaneous(sch) for sch in schemas}
    for ctx in sorted(ctxs, key=lambda h: v1.digest(arena, h)):
        if ctx not in info.contexts:
            continue
        work['contexts'] += 1
        counts = dict(v1.items(arena.handles[ctx]))
        index = S.ChildIndex(arena, counts)
        ctx_digest = v1.digest(arena, ctx).hex()
        for sch in schemas:
            budget = RM.WorkBudget(cfg.unify_states_max, work)
            trace: list = []
            try:
                for beta in S.assign(sch.source, index, dict(counts), {}, arena, cfg, budget, exact=False, trace=trace,
                                     simultaneous=simultaneous[sch.schema_hash]):
                    R_inst: Counter = Counter()
                    for (p, n), h in trace:
                        R_inst[h] += n
                    work['generated'] += 1
                    region_key = tuple(sorted((v1.digest(arena, h).hex(), c) for h, c in R_inst.items()))
                    site_key = ('rw', ctx_digest, region_key)
                    tsig = Qm.fast_target_signature(sch, beta)
                    bdig = Qm.fast_binding_digest(beta)
                    dkey = (site_key, sch.schema_hash, tsig)
                    prev = found.get(dkey)
                    if prev is None or bdig < prev.binding_digest:
                        found[dkey] = RM.Candidate(ctx, sch, dict(beta), R_inst, site_key, tsig, bdig,
                                                   Qm.fast_order_key(site_key, sch.schema_hash, tsig, bdig))
            except S.Cut as cut:
                work['unify_states'] += budget.states
                if cut.args and cut.args[0] == 'attempts':
                    work['cut_attempts'] += 1
                    return sorted(found.values(), key=lambda c: c.order_key), work
                work['cut_unify_states'] += 1
                continue
            work['unify_states'] += budget.states
    return sorted(found.values(), key=lambda c: c.order_key), work


def ctx_candidates(ctx, c: int, budget: list):
    """The frozen matcher's candidates at one context, cached: they depend only on that context's children.

    `budget` is the view's remaining attempt allowance (§9). A cached context costs nothing; a fresh one draws
    from it, and a view stops matching once its allowance is gone.
    """
    r = ctx.ctx_cache.get(c)
    if r is not None:
        ctx.port.stats['view_cache_hits'] += 1
        return r
    if budget[0] <= 0:
        ctx.port.stats['view_budget_exhausted'] += 1
        return []
    r, w = match_contexts(ctx.ca, c, {c}, ctx.schemas, ctx.cfg, budget[0])
    budget[0] -= w['attempts']
    ctx.ctx_cache[c] = r
    ctx.port.stats['view_matcher_contexts'] += w['contexts']
    ctx.port.stats['view_matcher_attempts'] += w['attempts']
    return r


@dataclass
class ViewAction:
    h: int
    h2: int
    key: str
    D_old: str
    D_new: str
    size: int
    cand: object
    succ: int
    canon: bytes
    origin: str
    source: str
    p: bytes = b''
    s: bytes = b''
    parent_cost: int = 0
    succ_cost: int = 0


def view_actions(ctx, cs, root: int, info, step: int):
    """Bounded alternative-representation views of one handle at a time (§7, §9, §11)."""
    ca, P = ctx.ca, ctx.port
    t0 = time.process_time()
    out, seen_canon = [], {}
    handles = sorted((h for h in info.reach_h if h != root and _nonmono(ca, h)), key=lambda h: v1.digest(ca, h))
    orbit = _orbits(ctx, cs, handles)
    pairs = 0
    live = info.reach_h
    for h, size in orbit:
        alts, k = P.alts(h)
        if not alts:
            continue
        P.stats['nontrivial_class_hits'] += 1
        ctx.nontrivial.add(h)
        for rep in alts:
            if pairs >= VIEW_PAIRS_MAX:
                P.stats['view_pairs_capped'] += 1
                break
            pairs += 1
            P.stats['view_pairs'] += 1
            h2 = P.instantiate(rep, k, h)
            if h2 is None or h2 == h:
                continue
            V = v1.substitute(ca, root, h, h2)
            if V == root:
                continue
            # A context's matches depend only on that context's own children, so they are cached per context
            # handle: inside the alternative, and at the rebuilt parents of the viewed handle. Contexts that
            # already exist in the state are served by the ordinary action space and are skipped.
            ctxs = {x for x in reach(ca, h2) if isinstance(ca.handles[x], v1.Sum)} - live
            ctxs |= {v1.substitute(ca, p, h, h2) for p in live
                     if isinstance(ca.handles[p], v1.Sum) and any(c == h for c, n in v1.items(ca.handles[p]))}
            inner_ctx = {x for x in ctxs if x in reach(ca, h2)}      # only these are local to the alternative
            budget = [VIEW_ATTEMPTS]
            cands = sorted((c for x in sorted(ctxs, key=lambda x: v1.digest(ca, x)) for c in ctx_candidates(ctx, x, budget)),
                           key=lambda c: c.order_key)
            P.stats['view_candidates'] += len(cands)
            seen_succ = set()
            for c in cands:
                if c.ctx in inner_ctx:
                    # The rewrite is local to the alternative, and the alternative's rewritten image does not
                    # depend on the surrounding state, so it is computed once and spliced. Identical successor:
                    # the context exists only inside h2, so replacing it inside h2 and then substituting h → h2'
                    # is the same state as rewriting it in V.
                    ck = (h2, c.order_key)
                    h2p = ctx.local_cache.get(ck)
                    if h2p is None:
                        img, why = RM.apply_candidate(ca, h2, c, v1.hpoly(ca, h2))
                        h2p = ctx.local_cache[ck] = (img if why == 'ok' else False)
                        if why != 'ok':
                            P.stats['view_rejected_'+why] += 1
                    elif h2p is not False:
                        P.stats['view_local_cache_hits'] += 1
                    if h2p is False:
                        continue
                    r3 = v1.substitute(ca, root, h, h2p)
                else:
                    r3, why = RM.apply_candidate(ca, V, c, ctx.Tpoly)
                    if why != 'ok':
                        P.stats['view_rejected_'+why] += 1
                        continue
                if r3 == root or r3 in seen_succ:
                    P.stats['view_dedup'] += 1
                    continue
                seen_succ.add(r3)
                try:
                    v1.verify(ca, ctx.task_c, r3)
                except AssertionError as ex:
                    P._fail(dict(kind='view', key=k.key, why=repr(ex)[:200]))
                    continue
                d = ctx.tfc.canon_digest(ca, r3)
                if d == cs.digest:
                    P.stats['view_noop'] += 1
                    continue
                prev = seen_canon.get(d)
                if prev is not None:
                    P.stats['view_dedup'] += 1                  # §11: one class per effective action
                    if c.order_key < prev.cand.order_key:
                        prev.cand, prev.succ, prev.h, prev.h2 = c, r3, h, h2
                    continue
                va = ViewAction(h, h2, k.key, k.D.hex(), rep.D, size, c, r3, d, rep.origin, rep.source,
                                parent_cost=v1.cost(ca, root), succ_cost=v1.cost(ca, r3))
                seen_canon[d] = va
                out.append(va)
                P.stats['view_verified'] += 1
                rep.view_uses += 1
    ctx.view_cpu += time.process_time()-t0
    return out



def _orbits(ctx, cs, handles):
    """Aut*(s) orbits of queried handles; one representative per orbit (§11)."""
    gm = [(g, ctx.tfc.cimg.setdefault(g, {})) for g in cs.gens]
    idx = {h: i for i, h in enumerate(handles)}
    uf = O._UF()
    for _ in handles:
        uf.add()
    for h in handles:
        for g, memo in gm:
            j = idx.get(memo[h] if h in memo else ctx.tfc.img(h, g))
            if j is not None:
                uf.union(idx[h], j)
    groups: dict = {}
    for h in handles:
        groups.setdefault(uf.find(idx[h]), []).append(h)
    return [(min(m, key=lambda x: v1.digest(ctx.ca, x)), len(m)) for m in groups.values()]



# ---------------------------------------------------------------- retrieval-indexed views (c7_2r_runtime @ 63d84aa)


def node_sig(arena, h):
    """('M', degree, divisor mask) | ('S', arity) | ('P', arity). Renaming-invariant: kinds, arities, degrees."""
    node = arena.handles[h]
    if isinstance(node, v1.Mono):
        deg, mask = mono_class(arena, h)
        return ('M', deg, mask)
    tot = sum(n for c, n in v1.items(node))
    return ('S' if isinstance(node, v1.Sum) else 'P', tot)


def _bucket(n: int) -> str:
    return str(n) if n <= 4 else '05-08' if n <= 8 else '09-16' if n <= 16 else '17+'


def sig_cache(memo: dict, arena):
    """Signature caches are keyed per arena: a handle is an index, so the same int is a different node elsewhere.

    A representative is read in its own frame, which is a different arena from the search state. The entry keeps
    a reference to the arena, so its `id` cannot be recycled underneath us.
    """
    e = memo.get(id(arena))
    if e is None:
        e = memo[id(arena)] = (arena, {})
    return e[1]


def ctx_sig(arena, c, memo: dict):
    """The multiset of child signatures of a context, cached per (arena, handle)."""
    cache = sig_cache(memo, arena)
    s = cache.get(c)
    if s is None:
        s = cache[c] = Counter()
        for child, n in v1.items(arena.handles[c]):
            s[node_sig(arena, child)] += n
    return s


def mono_class(arena, h):
    """(degree, divisor mask) of a Mono child: which hole exponents 1..6 could divide it. Cheap, pre-matching."""
    e = v1.unpack(arena.handles[h].m)
    deg = sum(e)
    mask = 0
    for d in range(1, 7):
        if all(x % d == 0 for x in e):
            mask |= 1 << d
    return deg, mask


def pattern_sig(p):
    """The anchor signature a source pattern requires of the child it matches.

    For a monomial pattern the requirement is arithmetic as well as structural: every hole binds a nonconstant
    vector, so the matched monomial has degree ≥ Σ exponents, and a single-hole pattern with exponent e can only
    match a monomial all of whose exponents are divisible by e (the frozen `ChildIndex.divisible` rule).
    """
    if p[0] == 'M':
        exps = [e for i, e in p[1]]
        div = exps[0] if len(p[1]) == 1 else 1
        return ('M', sum(exps), div)
    return (p[0], sum(n for c, n in p[1]))


class SchemaIndex:
    """Compiled anchor signatures of the frozen library's rewrite sources (built once)."""

    def __init__(self, schemas) -> None:
        self.need = []
        for sch in schemas:
            need = Counter()
            for pat, mult in sch.source:
                need[pattern_sig(pat)] += mult
            self.need.append((sch, need))
        self.cache: dict = {}
        self.stats = Counter()

    def compatible(self, sig: Counter):
        """The schemas whose source anchors this context could possibly supply (a necessary condition).

        A monomial requirement (degree d, divisor e) is supplied by a Mono child of degree ≥ d that e divides; an
        S / P requirement by a child of that kind and arity. Counted per requirement, so it stays a necessary
        condition and can never hide a real match.
        """
        key = tuple(sorted(sig.items()))
        r = self.cache.get(key)
        if r is not None:
            return r
        self.stats['signatures_compiled'] += 1
        out = []
        for sch, need in self.need:
            ok = True
            for s, n in need.items():
                if s[0] == 'M':
                    have = sum(c for t, c in sig.items() if t[0] == 'M' and t[1] >= s[1] and (t[2] >> s[2]) & 1)
                else:
                    have = sig.get(s, 0)
                if have < n:
                    ok = False
                    break
            if ok:
                out.append(sch)
        r = self.cache[key] = tuple(out)
        return r


def rep_signatures(rep, idx: SchemaIndex, memo: dict):
    """(root signature, inner context signatures) of a portfolio representative, in its own frame, cached on it."""
    s = getattr(rep, '_sigs', None)
    if s is None:
        a = rep.arena
        inner = [ctx_sig(a, x, memo) for x in v1.state_info(a, rep.h).reach_h if isinstance(a.handles[x], v1.Sum)]
        s = (node_sig(a, rep.h), inner)
        try:
            rep._sigs = s
        except AttributeError:
            pass
    return s


# ---------------------------------------------------------------- retrieval-indexed view actions


@dataclass
class Retrieval:
    idx: SchemaIndex
    k_active: int = None          # None = index only (behaviour-preserving); an int enables truncation
    b_threshold: int = 0          # truncate only when the filtered workload M(s) exceeds this
    sig_memo: dict = None
    stats: Counter = None
    parent_hit: dict = None

    def __post_init__(self):
        self.sig_memo = {} if self.sig_memo is None else self.sig_memo
        self.stats = Counter() if self.stats is None else self.stats
        self.parent_hit = {} if self.parent_hit is None else self.parent_hit


def make_view_fn(schemas, k_active=None, b_threshold: int = 0):
    """A drop-in replacement for `c7_2i_intern.view_actions` that retrieves before matching."""
    R = Retrieval(SchemaIndex(schemas), k_active, b_threshold)

    def view_fn(ctx, cs, root, info, step):
        return view_actions_indexed(ctx, cs, root, info, step, R)
    view_fn.retrieval = R
    return view_fn


def view_actions_indexed(ctx, cs, root, info, step, R: Retrieval):
    """C7.2I's view actions, with a cheap retrieval stage in front of every expensive step."""
    ca, P = ctx.ca, ctx.port
    t0 = time.process_time()
    out, seen_canon = [], {}
    live = info.reach_h
    handles = sorted((h for h in live if h != root and I._nonmono(ca, h)), key=lambda h: v1.digest(ca, h))
    # ---- parents of each handle, and their context signatures, computed once per state
    parents: dict = {}
    for p in live:
        if isinstance(ca.handles[p], v1.Sum):
            for c, n in v1.items(ca.handles[p]):
                if I._nonmono(ca, c):
                    parents.setdefault(c, []).append((p, n))
    # ---- pair enumeration: exactly the eager order and the same per-state pair cap, so that the index can only
    #      elide provably empty work and never change which views exist
    plan = []
    pairs = 0
    for h, size in I._orbits(ctx, cs, handles):
        alts, k = P.alts(h)
        if not alts:
            continue
        P.stats['nontrivial_class_hits'] += 1
        ctx.nontrivial.add(h)
        h_sig = node_sig(ca, h)
        stop = False
        for rep in alts:
            if pairs >= I.VIEW_PAIRS_MAX:
                P.stats['view_pairs_capped'] += 1
                stop = True
                break
            pairs += 1
            P.stats['view_pairs'] += 1
            R.stats['pairs_considered'] += 1
            root_sig, inner = rep_signatures(rep, R.idx, R.sig_memo)
            hit = any(R.idx.compatible(s) for s in inner)
            if not hit:
                for p, n in parents.get(h, ()):                     # the rebuilt parent, without building it
                    pk = (p, h_sig, root_sig)
                    got = R.parent_hit.get(pk)
                    if got is None:
                        s = Counter(ctx_sig(ca, p, R.sig_memo))
                        s[h_sig] -= n
                        if s[h_sig] <= 0:
                            del s[h_sig]
                        s[root_sig] += n
                        got = R.parent_hit[pk] = bool(R.idx.compatible(s))
                    if got:
                        hit = True
                        break
            if not hit:
                R.stats['pairs_skipped_by_index'] += 1
                continue
            plan.append((h, size, rep, k))
        if stop:
            break
    R.stats['pairs_filtered'] += len(plan)
    # M(s) is the workload *after* the sound filter: the distribution a truncation rule has to be chosen against.
    R.stats['M='+_bucket(len(plan))] += 1
    R.stats['raw='+_bucket(pairs)] += 1
    # ---- optional truncation: a separate knob, on the FILTERED workload M(s), never on raw state size
    if R.k_active is not None and len(plan) > R.b_threshold:
        R.stats['states_truncated'] += 1
        R.stats['pairs_dropped_by_budget'] += len(plan)-R.k_active
        plan = plan[:R.k_active]
    # ---- exact stage, only for retrieved pairs
    for h, size, rep, k in plan:
        R.stats['pairs_active'] += 1
        h2 = P.instantiate(rep, k, h)
        if h2 is None or h2 == h:
            continue
        V = v1.substitute(ca, root, h, h2)
        if V == root:
            continue
        ctxs = {x for x in I.reach(ca, h2) if isinstance(ca.handles[x], v1.Sum)} - live
        ctxs |= {v1.substitute(ca, p, h, h2) for p, n in parents.get(h, ())}
        inner_ctx = {x for x in ctxs if x in I.reach(ca, h2)}
        budget = [I.VIEW_ATTEMPTS]
        cands = []
        for x in sorted(ctxs, key=lambda x: v1.digest(ca, x)):
            sch = R.idx.compatible(ctx_sig(ca, x, R.sig_memo))
            R.stats['contexts_probed'] += 1
            if not sch:
                R.stats['contexts_skipped_by_index'] += 1
                continue
            R.stats['contexts_matched'] += 1
            R.stats['schema_slots'] += len(sch)
            cands += _ctx_candidates(ctx, x, sch, budget)
        cands.sort(key=lambda c: c.order_key)
        P.stats['view_candidates'] += len(cands)
        seen_succ = set()
        for c in cands:
            if c.ctx in inner_ctx:
                ck = (h2, c.order_key)
                h2p = ctx.local_cache.get(ck)
                if h2p is None:
                    img, why = RM.apply_candidate(ca, h2, c, v1.hpoly(ca, h2))
                    h2p = ctx.local_cache[ck] = (img if why == 'ok' else False)
                    if why != 'ok':
                        P.stats['view_rejected_'+why] += 1
                elif h2p is not False:
                    P.stats['view_local_cache_hits'] += 1
                if h2p is False:
                    continue
                r3 = v1.substitute(ca, root, h, h2p)
            else:
                r3, why = RM.apply_candidate(ca, V, c, ctx.Tpoly)
                if why != 'ok':
                    P.stats['view_rejected_'+why] += 1
                    continue
            if r3 == root or r3 in seen_succ:
                P.stats['view_dedup'] += 1
                continue
            seen_succ.add(r3)
            try:
                v1.verify(ca, ctx.task_c, r3)
            except AssertionError as ex:
                P._fail(dict(kind='view', key=k.key, why=repr(ex)[:200]))
                continue
            d = ctx.tfc.canon_digest(ca, r3)
            if d == cs.digest:
                P.stats['view_noop'] += 1
                continue
            prev = seen_canon.get(d)
            if prev is not None:
                P.stats['view_dedup'] += 1                     # C7.1b quotient action identity
                if c.order_key < prev.cand.order_key:
                    prev.cand, prev.succ, prev.h, prev.h2 = c, r3, h, h2
                continue
            va = I.ViewAction(h, h2, k.key, k.D.hex(), rep.D, size, c, r3, d, rep.origin, rep.source,
                              parent_cost=v1.cost(ca, root), succ_cost=v1.cost(ca, r3))
            seen_canon[d] = va
            out.append(va)
            P.stats['view_verified'] += 1
            rep.view_uses += 1
    ctx.view_cpu += time.process_time()-t0
    return out



def _ctx_candidates(ctx, c: int, schemas, budget: list):
    """The frozen matcher at one context, restricted to the retrieved schemas and cached per context."""
    r = ctx.ctx_cache.get(c)
    if r is not None:
        ctx.port.stats['view_cache_hits'] += 1
        return r
    if budget[0] <= 0:
        ctx.port.stats['view_budget_exhausted'] += 1
        return []
    r, w = I.match_contexts(ctx.ca, c, {c}, schemas, ctx.cfg, budget[0])
    budget[0] -= w['attempts']
    ctx.ctx_cache[c] = r
    ctx.port.stats['view_matcher_contexts'] += w['contexts']
    ctx.port.stats['view_matcher_attempts'] += w['attempts']
    return r


# ---------------------------------------------------------------- integration layer (new)

# the C7.2R bodies above reference the C7.2I module as `I`
I = SimpleNamespace(_nonmono=_nonmono, _orbits=_orbits, reach=reach, VIEW_PAIRS_MAX=VIEW_PAIRS_MAX,
                    VIEW_ATTEMPTS=VIEW_ATTEMPTS, ViewAction=ViewAction, match_contexts=match_contexts)


class ViewCtx:
    """The fields the ported view code reads (the C7.2I `_Ctx` subset for views), plus the scorer for the adapter."""

    def __init__(self, ca, tfc, task_c, Tpoly, schemas, cfg, sem, scorer, index: bool) -> None:
        self.ca, self.tfc, self.task_c, self.Tpoly = ca, tfc, task_c, Tpoly
        self.schemas, self.cfg, self.port, self.scorer = schemas, cfg, sem.port, scorer
        self.view_cpu = 0.0
        self.nontrivial: set = set()
        self.ctx_cache: dict = {}
        self.local_cache: dict = {}
        self.retrieval = Retrieval(SchemaIndex(schemas)) if index else None
        self.score_cache: dict = {}


def context(ca, tfc, task_c, Tpoly, schemas, cfg, sem, scorer, index: bool = True) -> ViewCtx:
    assert sem is not None, 'views read the semantic portfolio'
    return ViewCtx(ca, tfc, task_c, Tpoly, schemas, cfg, sem, scorer, index)


def actions(vctx: ViewCtx, cs, root: int, info, step: int) -> list:
    """Indexed retrieval when enabled (C7.2R, search-identical to eager by construction), else the eager C7.2I views."""
    if vctx.retrieval is not None:
        return view_actions_indexed(vctx, cs, root, info, step, vctx.retrieval)
    return view_actions(vctx, cs, root, info, step)


def scores(vctx: ViewCtx, root: int, vac: list) -> list:
    """Temporary scorer adapter (existing families only). A view's action is its rewrite candidate applied in the
    substituted state V = root[h -> h2]; it is scored exactly like an ordinary rewrite candidate of V by the current
    trained scorer (one encoder pass per distinct V). Replaced by the schema-agnostic scorer at T7."""
    ca = vctx.ca
    byV: dict = {}
    for i, va in enumerate(vac):
        V = v1.substitute(ca, root, va.h, va.h2)
        byV.setdefault(V, []).append(i)
    out = [0.0]*len(vac)
    for V in sorted(byV, key=lambda x: v1.digest(ca, x)):
        idx = byV[V]
        _p, rw = vctx.scorer.score_combined(ca, v1.state_info(ca, V), V, [], [vac[i].cand for i in idx])
        for i, x in zip(idx, rw):
            out[i] = x
    return out
