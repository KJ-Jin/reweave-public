"""Symmetry-safe canonical identity: canonical task frame, canonical states, quotient action classes.

Bounded candidate generation (enumeration caps, orientation, matcher order and truncation) must commute with variable
renaming. The task frame is the lexicographically smallest relabelling of the task; states are canonical DAGs in that
frame; actions are classes closed under the automorphisms of the state, so renaming neither splits nor merges actions.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

from .. import rewrite_match as RM
from .. import rewrite_schema as S
from .. import v1
from ..hashing import canonical_json
from .labelling import (IDENT, NV, _UF, compose, fast_labelling, generators, inverse, rdigest, relabel_into, relabel_m,
                        relabel_vec)

BIG = 10**9
SAFE_PRIM, SAFE_RW = 2000, 256
# The frozen generators, taken from their defining modules (never from the names a module patches).
V1_ENUMERATE = v1.enumerate_mod.enumerate_actions
V1_SAMPLING_KEY = v1.enumerate_mod.sampling_key
V1_MATCH = RM.match_state


# ---------------------------------------------------------------- canonical states (c7_1_equivariant @ bc5600a)

class CanonState:
    def __init__(self, tf: TaskFrame, arena, root: int) -> None:
        digs = [rdigest(arena, root, f, tf.rmemo[k]) for k, f in enumerate(tf.frames)]
        self.digest = min(digs)
        F = [k for k, d in enumerate(digs) if d == self.digest]
        k0 = F[0]
        self.f0 = tf.frames[k0]
        self.f0inv = inverse(self.f0)
        u0inv = inverse(tf.G[k0])
        self.aut = sorted({compose(tf.G[k], u0inv) for k in F})
        self.gens = generators(self.aut)
        fwd = tf.fwd.setdefault(k0, {})
        self.croot = relabel_into(arena, root, self.f0, tf.carena, fwd)
        assert v1.digest(tf.carena, self.croot) == self.digest, 'relabelled digest replica must equal the interned digest'
        self.back = {fwd[h]: h for h in v1.state_info(arena, root).reach_h}
        assert len(self.back) == len(v1.state_info(arena, root).reach_h)

    def back_h(self, tf: TaskFrame, arena, h: int) -> int:
        o = self.back.get(h)
        if o is not None:
            return o
        node = tf.carena.handles[h]
        assert isinstance(node, v1.Mono), 'only unreachable monomial FACTOR arguments live outside the state'
        return v1.mk_mono(arena, relabel_m(node.m, self.f0inv))


def _hash(xs) -> str:
    return hashlib.sha256(repr(sorted(xs)).encode()).hexdigest()[:24]


def _admit(keyed: list, cap: int, safe: int):
    """keyed: [(class_key, members)] sorted by key. Whole classes until the retained count reaches cap."""
    kept, total, max_cls, explosion = [], 0, 0, False
    for key, members in keyed:
        if total >= cap:
            break
        kept += members
        total += len(members)
        max_cls = max(max_cls, len(members))
        explosion |= len(members) > safe
    return kept, max(0, total-cap), max_cls, explosion


# ---------------------------------------------------------------- primitive generation


def _cand(arena, ctx: int, sch, beta: dict, R: Counter):
    """A Candidate with the frozen site key, target signature, binding digest and order key (as in match_state)."""
    region_key = tuple(sorted((v1.digest(arena, h).hex(), c) for h, c in R.items()))
    site_key = ('rw', v1.digest(arena, ctx).hex(), region_key)
    tsig = S.target_signature(sch, beta)
    bdig = S.binding_digest(beta)
    order = (hashlib.sha256(canonical_json([site_key, sch.schema_hash, tsig, bdig]).encode()).hexdigest(),
             site_key, sch.schema_hash, tsig, bdig)
    return RM.Candidate(ctx, sch, dict(beta), Counter(R), site_key, tsig, bdig, order)


def _dkey(c):
    return (c.site_key, c.schema.schema_hash, c.target_signature)



# the copied C7.1b bodies below reference the oracle module as `O`
O = SimpleNamespace(NV=NV, BIG=BIG, IDENT=IDENT, SAFE_PRIM=SAFE_PRIM, SAFE_RW=SAFE_RW, V1_ENUMERATE=V1_ENUMERATE,
                    V1_SAMPLING_KEY=V1_SAMPLING_KEY, V1_MATCH=V1_MATCH, relabel_vec=relabel_vec, relabel_m=relabel_m,
                    compose=compose, inverse=inverse, rdigest=rdigest, relabel_into=relabel_into, _UF=_UF,
                    CanonState=CanonState, _hash=_hash, _admit=_admit, _cand=_cand, _dkey=_dkey)
_ENC = json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False)


# ---------------------------------------------------------------- exact hash replicas (c7_1b_quotient @ c7db59b)


def _cj(x) -> str:
    """`hashing.canonical_json` for values built only from str / int / list / tuple. `_plain` is then the identity
    up to tuple → list, which the encoder applies itself; the encoder settings are canonical_json's."""
    return _ENC.encode(x)


def _fsig(p, beta: dict):
    """Replica of `rewrite_schema._inst_sig`."""
    if p[0] == 'M':
        return ['m', list(S.mon_vector(p, beta))]
    return [p[0], sorted(([_fsig(c, beta), n] for c, n in p[1]), key=_cj)]


def fast_target_signature(sch, beta: dict) -> str:
    """Replica of `rewrite_schema.target_signature`."""
    kids = sorted(([_fsig(c, beta), n] for c, n in sch.target), key=_cj)
    return hashlib.sha256(_cj(kids).encode()).hexdigest()


def fast_binding_digest(beta: dict) -> str:
    """Replica of `rewrite_schema.binding_digest`."""
    return hashlib.sha256(_cj([[i, list(beta[i])] for i in sorted(beta)]).encode()).hexdigest()


def fast_order_key(site_key, sh: str, tsig: str, bdig: str) -> tuple:
    """Replica of the frozen `match_state` order key."""
    return (hashlib.sha256(_cj([site_key, sh, tsig, bdig]).encode()).hexdigest(), site_key, sh, tsig, bdig)




# ---------------------------------------------------------------- canonical task frame (c7_1b_quotient @ c7db59b)


class TaskFrameQ:
    """The oracle's per-task frame (π extended to a bijection, T*, G sorted, frames = υ ∘ π), from one fast pass."""

    def __init__(self, target: dict | None = None, *, canonical_of: 'TaskFrameQ | None' = None) -> None:
        t0 = time.process_time()
        if canonical_of is not None:
            # The canonical task T*: its P1 labelling is the identity (T* is the minimal form, and the identity is
            # the smallest permutation), and its symmetry group is the same G.
            self.pi, self.n, self.T, self.G = IDENT, canonical_of.n, canonical_of.T, canonical_of.G
        else:
            pi, used, mins = fast_labelling(target)
            full = list(IDENT)
            taken = {pi[i] for i in used}
            for i in used:
                full[i] = pi[i]
            for i, j in zip([i for i in range(NV) if i not in used], [j for j in range(NV) if j not in taken]):
                full[i] = j
            self.pi = tuple(full)
            self.n = len(used)
            self.T = {O.relabel_vec(v1.unpack(m), self.pi): c for m, c in target.items()}
            G = set()
            for img in mins:
                u = list(IDENT)
                for k, i in enumerate(used):
                    u[self.pi[i]] = img[k]
                G.add(tuple(u))
            self.G = sorted(G)
        assert self.G and self.G[0] == IDENT
        self.frames = [O.compose(u, self.pi) for u in self.G]
        self.carena = v1.Arena()
        self.carena.calc_cfg = v1.CalcCfg()
        self.arena = None
        self._reset()
        self.cimg: dict = {}
        self.tsig_cache: dict = {}
        self.bdig_cache: dict = {}
        self.schemas: dict = {}
        self.stats = Counter()
        self.setup_seconds = time.process_time()-t0

    def _reset(self) -> None:
        self.rmemo = [dict() for _ in self.frames]
        self.fwd: dict = {}
        self.cache: dict = {}

    def state(self, arena, root: int) -> 'O.CanonState':
        """The oracle's CanonState (identical definition), computed once per (arena, state) and cached."""
        if self.arena is not arena:
            self.arena = arena
            self._reset()
        cs = self.cache.get(root)
        if cs is None:
            self.stats['canon_computed'] += 1
            cs = self.cache[root] = O.CanonState(self, arena, root)
        else:
            self.stats['canon_cache_hit'] += 1
        return cs

    def img(self, h: int, u) -> int:
        return O.relabel_into(self.carena, h, u, self.carena, self.cimg.setdefault(u, {}))

    def canon_digest(self, arena, root: int) -> bytes:
        """D*(s) without interning (min relabelled digest over the frames)."""
        if self.arena is not arena:
            self.arena = arena
            self._reset()
        return min(O.rdigest(arena, root, f, self.rmemo[k]) for k, f in enumerate(self.frames))

    def tsig(self, sch, beta_t) -> str:
        k = (sch.schema_hash, beta_t)
        r = self.tsig_cache.get(k)
        if r is None:
            self.stats['tsig_computed'] += 1
            self.schemas.setdefault(sch.schema_hash, sch)
            r = self.tsig_cache[k] = fast_target_signature(sch, dict(beta_t))
        else:
            self.stats['tsig_cache_hit'] += 1
        return r

    def bdig(self, beta_t) -> str:
        r = self.bdig_cache.get(beta_t)
        if r is None:
            self.stats['bdig_computed'] += 1
            r = self.bdig_cache[beta_t] = fast_binding_digest(dict(beta_t))
        else:
            self.stats['bdig_cache_hit'] += 1
        return r

    def seed(self, c, beta_t) -> None:
        """Values the frozen matcher computed for a found candidate (exact by construction)."""
        self.schemas.setdefault(c.schema.schema_hash, c.schema)
        self.tsig_cache.setdefault((c.schema.schema_hash, beta_t), c.target_signature)
        self.bdig_cache.setdefault(beta_t, c.binding_digest)


def verify_fast_hashes(tf: TaskFrameQ) -> dict:
    """Every cached target signature / binding digest recomputed with the frozen functions (test harness)."""
    bad_t = sum(S.target_signature(tf.schemas[sh], dict(bt)) != r for (sh, bt), r in tf.tsig_cache.items())
    bad_b = sum(S.binding_digest(dict(bt)) != r for bt, r in tf.bdig_cache.items())
    return dict(tsig_checked=len(tf.tsig_cache), tsig_mismatch=bad_t, bdig_checked=len(tf.bdig_cache), bdig_mismatch=bad_b)



# ---------------------------------------------------------------- action classes (c7_1b_quotient @ c7db59b)


@dataclass
class ActionClass:
    kind: str                     # 'prim' | 'rw'
    key: object                   # class key (oracle: min sampling key / min order_key); None if the cap did not bind
    size: int                     # orbit size (number of concrete members)
    rep: object                   # canonical representative (v1.Action or RM.Candidate, canonical arena)
    members: list = field(default_factory=list)   # member descriptors (test harness only)


def _akey(a):
    return (a.kind, a.ctx, a.region, a.arg, a.src)


def q_primitive(tf: TaskFrameQ, cs, cfg, log: list | None = None, keep_members: bool = False) -> list:
    t0 = time.process_time()
    ca, root = tf.carena, cs.croot
    full = V1_ENUMERATE(ca, v1.state_info(ca, root), root, replace(cfg, n_max=BIG))
    acts, index, uf = [], {}, O._UF()
    for a in full:
        k = _akey(a)
        if k not in index:
            index[k] = uf.add()
            acts.append(a)
    n_enum = len(acts)
    materialized = 0
    gmaps = [(g, tf.cimg.setdefault(g, {})) for g in cs.gens]
    i = 0
    while i < len(acts):
        a = acts[i]
        for g, memo in gmaps:
            f = lambda h: memo[h] if h in memo else tf.img(h, g)
            region = tuple(sorted((f(h), c) for h, c in a.region))
            src = tuple(sorted((f(h), c) for h, c in a.src)) if a.src else a.src
            k = (a.kind, f(a.ctx), region, f(a.arg) if a.arg >= 0 else a.arg, src)
            j = index.get(k)
            if j is None:
                j = index[k] = uf.add()
                acts.append(v1.Action(k[0], k[1], k[2], k[3], k[4]))
                materialized += 1
            uf.union(i, j)
        i += 1
    groups: dict = {}
    for i in range(len(acts)):
        groups.setdefault(uf.find(i), []).append(i)
    if len(acts) <= cfg.n_max:
        kept_groups = list(groups.values())
        slack, max_cls, explosion, truncated = 0, max((len(m) for m in kept_groups), default=0), False, False
        keyed = None
    else:
        keyed = sorted(((min(V1_SAMPLING_KEY(ca, root, acts[i]) for i in m), m) for m in groups.values()), key=lambda x: x[0])
        kept, slack, max_cls, explosion = O._admit(keyed, cfg.n_max, SAFE_PRIM)
        kept_set = set(kept)
        kept_groups = [m for key, m in keyed if m[0] in kept_set]
        truncated = True
    key_of = {id(m): key for key, m in keyed} if keyed is not None else {}
    classes = []
    for m in kept_groups:
        rep = min((acts[i] for i in m), key=lambda a: v1.action_key(ca, a))
        classes.append(ActionClass('prim', key_of.get(id(m)), len(m), rep, [acts[i] for i in m] if keep_members else []))
    classes.sort(key=lambda c: v1.action_key(ca, c.rep))
    if log is not None:
        log.append(dict(stage='prim', canon_digest=cs.digest.hex(), aut=len(cs.aut), gens=len(cs.gens), precap=n_enum,
                        closure=len(acts), materialized=materialized, classes=len(groups), retained_classes=len(classes),
                        retained_members=sum(c.size for c in classes), slack=slack, max_class=max_cls, explosion=explosion,
                        truncated=truncated, duplicates_avoided=sum(c.size-1 for c in classes),
                        retained_canon_hash=O._hash(v1.action_key(ca, acts[i]) for m in kept_groups for i in m),
                        cpu=time.process_time()-t0))
    return classes


def q_rewrite(tf: TaskFrameQ, cs, schemas: list, cfg, log: list | None = None, keep_members: bool = False) -> list:
    t0 = time.process_time()
    ca, root = tf.carena, cs.croot
    cands, work = V1_MATCH(ca, root, schemas, replace(cfg, rewrite_n_max=BIG))
    dhex: dict = {}

    def dh(h):
        r = dhex.get(h)
        if r is None:
            r = dhex[h] = v1.digest(ca, h).hex()
        return r
    keys, rep, idx, uf = [], {}, {}, O._UF()

    def add(site_key, sch, tsig, bdig, ctx, beta_t, R_t) -> int:
        d = (site_key, sch.schema_hash, tsig)
        j = idx.get(d)
        if j is None:
            j = idx[d] = uf.add()
            keys.append(d)
            rep[d] = (bdig, ctx, sch, beta_t, R_t)
        elif bdig < rep[d][0]:
            rep[d] = (bdig, ctx, sch, beta_t, R_t)
        return j
    okc: dict = {}
    for c in cands:
        beta_t = tuple(sorted(c.beta.items()))
        tf.seed(c, beta_t)
        okc[(O._dkey(c), c.binding_digest)] = c.order_key
        add(c.site_key, c.schema, c.target_signature, c.binding_digest, c.ctx, beta_t, tuple(sorted(c.R_inst.items())))
    n_found = len(keys)
    images = 0
    gmaps = [(g, tf.cimg.setdefault(g, {})) for g in cs.gens]
    i = 0
    while i < len(keys):
        bdig, ctx, sch, beta_t, R_t = rep[keys[i]]
        for g, memo in gmaps:
            f = lambda h: memo[h] if h in memo else tf.img(h, g)
            ctx2 = f(ctx)
            R2 = tuple(sorted((f(h), n) for h, n in R_t))
            beta2 = tuple((hole, O.relabel_vec(b, g)) for hole, b in beta_t)
            site2 = ('rw', dh(ctx2), tuple(sorted((dh(h), n) for h, n in R2)))
            images += 1
            uf.union(i, add(site2, sch, tf.tsig(sch, beta2), tf.bdig(beta2), ctx2, beta2, R2))
        i += 1

    reused = 0

    def order_key(d):
        nonlocal reused
        bdig = rep[d][0]
        r = okc.get((d, bdig))
        if r is not None:
            reused += 1
            return r
        return fast_order_key(d[0], d[1], d[2], bdig)
    ok = {d: order_key(d) for d in keys}
    groups: dict = {}
    for i in range(len(keys)):
        groups.setdefault(uf.find(i), []).append(i)
    keyed = sorted(((min(ok[keys[i]] for i in m), m) for m in groups.values()), key=lambda x: x[0])
    class_hash = O._hash(k for k, m in keyed)
    if len(keys) <= cfg.rewrite_n_max:
        kept_keyed, slack, max_cls, explosion, truncated = keyed, 0, max((len(m) for k, m in keyed), default=0), False, False
    else:
        kept, slack, max_cls, explosion = O._admit(keyed, cfg.rewrite_n_max, SAFE_RW)
        kept_set = set(kept)
        kept_keyed = [(k, m) for k, m in keyed if m[0] in kept_set]
        truncated = True
    classes = []
    for key, m in kept_keyed:
        d = min((keys[i] for i in m), key=lambda d: ok[d])
        bdig, ctx, sch, beta_t, R_t = rep[d]
        cand = RM.Candidate(ctx, sch, dict(beta_t), Counter(dict(R_t)), d[0], d[2], bdig, ok[d])
        mem = []
        if keep_members:
            for i in m:
                b2, c2, s2, bt2, r2 = rep[keys[i]]
                mem.append(RM.Candidate(c2, s2, dict(bt2), Counter(dict(r2)), keys[i][0], keys[i][2], b2, ok[keys[i]]))
        classes.append(ActionClass('rw', key if truncated else key, len(m), cand, mem))
    classes.sort(key=lambda c: c.rep.order_key)
    if log is not None:
        log.append(dict(stage='rw', canon_digest=cs.digest.hex(), aut=len(cs.aut),
                        canon_work={k: work[k] for k in ('contexts', 'attempts', 'unify_states', 'cut_contexts',
                                                          'cut_unify_states', 'cut_attempts', 'generated', 'deduped')},
                        found_hash=O._hash((O._dkey(c), c.binding_digest) for c in cands), class_hash=class_hash,
                        retained_canon_hash=O._hash(ok[keys[i]] for k, m in kept_keyed for i in m),
                        found=n_found, closure=len(keys), images=images, materialized=len(keys)-n_found,
                        classes=len(groups), retained_classes=len(classes), retained_members=sum(c.size for c in classes),
                        slack=slack, max_class=max_cls, explosion=explosion, truncated=truncated,
                        duplicates_avoided=sum(c.size-1 for c in classes), tsig_cache=len(tf.tsig_cache),
                        order_keys_reused=reused, order_keys_computed=len(keys)-reused,
                        task_cache_stats_cumulative={k: tf.stats[k] for k in ('tsig_computed', 'tsig_cache_hit', 'bdig_computed', 'bdig_cache_hit')},
                        cpu=time.process_time()-t0))
    return classes


def generate(tf: TaskFrameQ, arena, root: int, schemas: list, cfg, ccfg, log: list | None = None, keep_members: bool = False):
    """Canonicalize the state once (cached), then generate primitive and rewrite classes on Canon(s)."""
    t0 = time.process_time()
    cs = tf.state(arena, root)
    t1 = time.process_time()
    prim = q_primitive(tf, cs, ccfg, log, keep_members)
    rw = q_rewrite(tf, cs, schemas, cfg, log, keep_members)
    if log is not None:
        log.append(dict(stage='canon', cpu=t1-t0))
    return cs, prim, rw
