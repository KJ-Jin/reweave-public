"""Generic ExprParam binding in symmetry-safe identity mode: generic-binder action classes on the canonical state.

`q_rewrite_g` / `q_latent_g` are the identity-mode rewrite and latent class generators with the binder replaced by the
generic binder (`canon.exprparam`); under a symmetry g of the canonical state a handle-bound parameter maps to the
relabelled handle with its semantic key recomputed. With REACHABLE_HANDLE_BIND off both produce exactly the
monomial-only classes (tested). `handle_lane_identity` is the identity-mode handle lane.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import replace

from .. import rewrite_match as RM
from .. import v1
from . import exprparam as EP
from .identity import BIG, SAFE_RW, ActionClass, _admit, _dkey, _hash, fast_order_key
from .labelling import _UF, relabel_vec
from .latent_identity import LATENT_CLASS_CAP, dedupe_key


def _image(ca, beta_t, g, f) -> tuple:
    return tuple((hole, EP.hb(ca, f(b.h)) if isinstance(b, EP.HB) else relabel_vec(b, g)) for hole, b in beta_t)


def q_rewrite_g(tf, cs, schemas: list, cfg, handles: bool, log: list | None = None, keep_members: bool = False,
                hb_only: bool = False, work_out: dict | None = None) -> list:
    t0 = time.process_time()
    ca, root = tf.carena, cs.croot
    if hb_only:                            # the handle lane: the fair scheduler (T6A.2)
        cands, work = EP.handle_visible(ca, root, schemas, cfg)
    else:
        cands, work = EP.match_state_g(ca, root, schemas, replace(cfg, rewrite_n_max=BIG), handles)
    if work_out is not None:
        work_out.update(work)
    dhex: dict = {}

    def dh(h):
        r = dhex.get(h)
        if r is None:
            r = dhex[h] = v1.digest(ca, h).hex()
        return r
    keys, rep, idx, uf = [], {}, {}, _UF()

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
        if not EP.has_hb(c.beta):
            tf.seed(c, beta_t)
        okc[(_dkey(c), c.binding_digest)] = c.order_key
        add(c.site_key, c.schema, c.target_signature, c.binding_digest, c.ctx, beta_t, tuple(sorted(c.R_inst.items())))
    n_found = len(keys)
    images = 0
    gmaps = [(g, tf.cimg.setdefault(g, {})) for g in cs.gens]
    i = 0
    while i < len(keys):
        bdig, ctx, sch, beta_t, R_t = rep[keys[i]]
        generic = any(isinstance(b, EP.HB) for hole, b in beta_t)
        for g, memo in gmaps:
            f = lambda h: memo[h] if h in memo else tf.img(h, g)
            ctx2 = f(ctx)
            R2 = tuple(sorted((f(h), n) for h, n in R_t))
            site2 = ('rw', dh(ctx2), tuple(sorted((dh(h), n) for h, n in R2)))
            images += 1
            if generic:
                beta2 = _image(ca, beta_t, g, f)
                uf.union(i, add(site2, sch, EP.target_signature_g(sch, dict(beta2)), EP.binding_digest_g(ca, dict(beta2)),
                                ctx2, beta2, R2))
            else:
                beta2 = tuple((hole, relabel_vec(b, g)) for hole, b in beta_t)
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
    class_hash = _hash(k for k, m in keyed)
    if len(keys) <= cfg.rewrite_n_max:
        kept_keyed, slack, max_cls, explosion, truncated = keyed, 0, max((len(m) for k, m in keyed), default=0), False, False
    else:
        kept, slack, max_cls, explosion = _admit(keyed, cfg.rewrite_n_max, SAFE_RW)
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
        entry = dict(stage='rw', canon_digest=cs.digest.hex(), aut=len(cs.aut),
                     canon_work={k: work[k] for k in ('contexts', 'attempts', 'unify_states', 'cut_contexts',
                                                       'cut_unify_states', 'cut_attempts', 'generated', 'deduped')},
                     found_hash=_hash((_dkey(c), c.binding_digest) for c in cands), class_hash=class_hash,
                     retained_canon_hash=_hash(ok[keys[i]] for k, m in kept_keyed for i in m),
                     found=n_found, closure=len(keys), images=images, materialized=len(keys)-n_found,
                     classes=len(groups), retained_classes=len(classes), retained_members=sum(c.size for c in classes),
                     slack=slack, max_class=max_cls, explosion=explosion, truncated=truncated,
                     duplicates_avoided=sum(c.size-1 for c in classes), tsig_cache=len(tf.tsig_cache),
                     order_keys_reused=reused, order_keys_computed=len(keys)-reused,
                     task_cache_stats_cumulative={k: tf.stats[k] for k in ('tsig_computed', 'tsig_cache_hit', 'bdig_computed', 'bdig_cache_hit')},
                     cpu=time.process_time()-t0)
        if handles:
            entry.update(hb_found=work.get('hb_candidates', 0), hb_classes=sum(EP.has_hb(c.rep.beta) for c in classes),
                         handle_choices=work.get('handle_choices', 0))
        log.append(entry)
    return classes


def q_latent_g(tf, cs, schemas: list, cfg, lcfg, visible_keys: set, handles: bool, log: list | None = None,
               keep_members: bool = False, hb_only: bool = False, work_out: dict | None = None) -> list:
    t0 = time.process_time()
    ca, root = tf.carena, cs.croot
    if hb_only:                            # the handle lane: the fair scheduler (T6A.2)
        lat, work = EP.handle_latent(ca, root, schemas, cfg, lcfg)
    else:
        lat, work = EP.latent_candidates_g(ca, root, schemas, cfg, lcfg, handles)
    if work_out is not None:
        work_out.update(work)
    keys, rep, idx, uf = [], {}, {}, _UF()

    def add(c) -> int:
        k = (c.site_key, c.schema.schema_hash, c.target_signature)
        j = idx.get(k)
        if j is None:
            j = idx[k] = uf.add()
            keys.append(k)
            rep[k] = c
        elif c.binding_digest < rep[k].binding_digest:
            rep[k] = c
        return j
    for c in lat:
        add(c)
    n_found = len(keys)
    gmaps = [(g, tf.cimg.setdefault(g, {})) for g in cs.gens]
    i = 0
    while i < len(keys):
        c = rep[keys[i]]
        for g, memo in gmaps:
            f = lambda h: memo[h] if h in memo else tf.img(h, g)
            img = EP.lcand_g(ca, f(c.ctx), relabel_vec(c.d, g), c.schema, dict(_image(ca, tuple(c.beta.items()), g, f)),
                             Counter({f(h): n for h, n in c.R_inst.items()}))
            uf.union(i, add(img))
        i += 1
    groups: dict = {}
    for i in range(len(keys)):
        groups.setdefault(uf.find(i), []).append(i)
    dropped = 0
    live = []
    for m in groups.values():
        if any(dedupe_key(ca, rep[keys[i]]) in visible_keys for i in m):
            dropped += 1
            continue
        live.append(m)
    keyed = sorted(((min(rep[keys[i]].order_key for i in m), m) for m in live), key=lambda x: x[0])
    total = sum(len(m) for k, m in keyed)
    if total <= LATENT_CLASS_CAP:
        kept_keyed, slack, max_cls, explosion, truncated = keyed, 0, max((len(m) for k, m in keyed), default=0), False, False
    else:
        kept, slack, max_cls, explosion = _admit(keyed, LATENT_CLASS_CAP, LATENT_CLASS_CAP)
        kept_set = set(kept)
        kept_keyed = [(k, m) for k, m in keyed if m[0] in kept_set]
        truncated = True
    if work_out is not None:
        work_out['orbit_cap'] = int(truncated)
    classes = []
    for key, m in kept_keyed:
        c = min((rep[keys[i]] for i in m), key=lambda x: x.order_key)
        classes.append(ActionClass('lx', key, len(m), c, [rep[keys[i]] for i in m] if keep_members else []))
    classes.sort(key=lambda x: x.rep.order_key)
    if log is not None:
        entry = dict(stage='lx', canon_digest=cs.digest.hex(), aut=len(cs.aut), found=n_found, closure=len(keys),
                     materialized=len(keys)-n_found, classes=len(groups), dedup_dropped_classes=dropped,
                     retained_classes=len(classes), retained_members=sum(x.size for x in classes), slack=slack,
                     max_class=max_cls, explosion=explosion, truncated=truncated,
                     class_hash=_hash(k for k, m in keyed),
                     cut_attempts=work.get('cut_attempts', 0), cut_candidates=work.get('cut_candidates', 0),
                     cut_contexts=work.get('cut_contexts', 0), cut_unify_states=work.get('cut_unify_states', 0),
                     cpu=time.process_time()-t0)
        if handles:
            entry.update(hb_found=work.get('hb_candidates', 0), hb_classes=sum(EP.has_hb(x.rep.beta) for x in classes),
                         handle_choices=work.get('handle_choices', 0))
        log.append(entry)
    return classes


def _merge_surfaces(classes: list) -> tuple:
    """Orbit classes of different recognition surfaces that realize exactly the same concrete successors (the same
    set of exact concrete member keys) are ONE selectable class; every surface is kept as provenance."""
    groups: dict = {}
    for c in classes:
        groups.setdefault(frozenset(EP.concrete_key(m) for m in c.members), []).append(c)
    out, merged = [], 0
    for g in groups.values():
        g.sort(key=lambda c: c.rep.order_key)
        surfaces = sorted({s for c in g for s in (getattr(c.rep, 'surfaces', ()) or (c.rep.schema.schema_hash,))})
        g[0].rep.surfaces = tuple(surfaces)
        merged += len(g)-1
        out.append(g[0])
    out.sort(key=lambda c: c.rep.order_key)
    return out, merged


def handle_lane_identity(tf, cs, schemas: list, cfg, lcfg, log: list | None = None) -> tuple:
    """The identity-mode handle lane of one canonical state (see the module docstring)."""
    w: dict = {}
    lw: dict = {}
    wcl = q_rewrite_g(tf, cs, schemas, replace(cfg, rewrite_n_max=BIG), True, log, keep_members=True,
                      hb_only=True, work_out=w)
    wcl, mw = _merge_surfaces(wcl)
    w['merged_cross_surface'] = w.get('merged_cross_surface', 0)+mw
    lcl = []
    if lcfg is not None:
        vk = {dedupe_key(tf.carena, m) for c in wcl for m in c.members}
        lcl = q_latent_g(tf, cs, schemas, cfg, lcfg, vk, True, log, keep_members=True, hb_only=True, work_out=lw)
        lcl, ml = _merge_surfaces(lcl)
        lw['merged_cross_surface'] = lw.get('merged_cross_surface', 0)+ml
    for c in wcl+lcl:
        c.rep.class_orbit_size = c.size
    adm, st = EP.admit(tf.carena, [c.rep for c in wcl+lcl], cfg.rewrite_n_max)
    return adm, dict(visible=w, latent=lw, admission=st, n_visible=len(wcl), n_latent=len(lcl), classes=wcl+lcl)


def orbit_keys_g(tf, cs, c) -> set:
    """Every image of a (possibly handle-bound) action under the FULL automorphism group of the state (tests)."""
    from ..rewrite_latent import LatentCandidate
    ca = tf.carena
    out = set()
    for g in cs.aut:
        f = lambda h: tf.img(h, g)
        beta = dict(_image(ca, tuple(c.beta.items()), g, f))
        R = Counter({f(h): n for h, n in c.R_inst.items()})
        x = (EP.lcand_g(ca, f(c.ctx), relabel_vec(c.d, g), c.schema, beta, R) if isinstance(c, LatentCandidate)
             else EP.cand_g(ca, f(c.ctx), c.schema, beta, R))
        out.add((x.site_key, x.schema.schema_hash, x.target_signature))
    return out
