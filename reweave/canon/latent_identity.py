"""Latent quotient actions inside symmetry-safe identity; concretization, serialization and replay.

A monomial-quotient action is identified by
    site_key = ('lx', context digest, consumed original region (digest, multiplicity)*, d)  and  (relation, target sig)
so the multiplier d participates semantically. Under a symmetry g of the canonical state the image of a latent action
relabels its context, region, bindings AND d; `q_latent` closes the found latent actions under Aut(state), so an
action class is exactly an orbit. Recorded trajectories (including d, views and normalization steps) replay exactly.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter

from .. import rewrite_match as RM
from .. import rewrite_schema as S
from .. import v1
from ..rewrite_latent import LatentCandidate, apply_latent, latent_candidates
from . import exprparam as EP
from .identity import ActionClass, _admit, _cand, _hash
from .labelling import IDENT, _UF, compose, inverse, relabel_into, relabel_vec

LATENT_CLASS_CAP = 4096          # = LatentConfig.candidates_max: whole-class admission over the closure


def _region_key(arena, R) -> tuple:
    return tuple(sorted((v1.digest(arena, h).hex(), c) for h, c in R.items()))


def lcand(arena, ctx: int, d, sch, beta: dict, R: Counter) -> LatentCandidate:
    """A LatentCandidate with exactly `rewrite_latent.latent_candidates`' site key, signature, digest and order key."""
    site_key = ('lx', v1.digest(arena, ctx).hex(), _region_key(arena, R), tuple(d))
    tsig = S.target_signature(sch, beta)
    bdig = S.binding_digest(beta)
    order = (hashlib.sha256(repr((site_key, sch.schema_hash, tsig, bdig)).encode()).hexdigest(),
             site_key, sch.schema_hash, tsig, bdig)
    return LatentCandidate(ctx, tuple(d), sch, dict(beta), Counter(R), site_key, tsig, bdig, order)


def dedupe_key(arena, c) -> tuple:
    """The mainline `dedupe_against_visible` key (context, consumed region, schema, target signature)."""
    return (c.ctx, _region_key(arena, c.R_inst), c.schema.schema_hash, c.target_signature)


def q_latent(tf, cs, schemas: list, cfg, lcfg, visible_keys: set, log: list | None = None,
             keep_members: bool = False) -> list:
    """Latent quotient action classes on the canonical state: orbits under Aut(state) of the latent actions."""
    t0 = time.process_time()
    ca, root = tf.carena, cs.croot
    lat, work = latent_candidates(ca, root, schemas, cfg, lcfg)
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
            img = lcand(ca, f(c.ctx), relabel_vec(c.d, g), c.schema, {h: relabel_vec(b, g) for h, b in c.beta.items()},
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
    classes = []
    for key, m in kept_keyed:
        c = min((rep[keys[i]] for i in m), key=lambda x: x.order_key)
        classes.append(ActionClass('lx', key, len(m), c, [rep[keys[i]] for i in m] if keep_members else []))
    classes.sort(key=lambda x: x.rep.order_key)
    if log is not None:
        log.append(dict(stage='lx', canon_digest=cs.digest.hex(), aut=len(cs.aut), found=n_found, closure=len(keys),
                        materialized=len(keys)-n_found, classes=len(groups), dedup_dropped_classes=dropped,
                        retained_classes=len(classes), retained_members=sum(x.size for x in classes), slack=slack,
                        max_class=max_cls, explosion=explosion, truncated=truncated,
                        class_hash=_hash(k for k, m in keyed),
                        cut_attempts=work.get('cut_attempts', 0), cut_candidates=work.get('cut_candidates', 0),
                        cut_contexts=work.get('cut_contexts', 0), cut_unify_states=work.get('cut_unify_states', 0),
                        cpu=time.process_time()-t0))
    return classes


def orbit_keys(tf, cs, c) -> set:
    """Every image of an action under the FULL automorphism group of the state (test harness)."""
    ca = tf.carena
    out = set()
    for g in cs.aut:
        f = lambda h: tf.img(h, g)
        if isinstance(c, LatentCandidate):
            x = lcand(ca, f(c.ctx), relabel_vec(c.d, g), c.schema, {h: relabel_vec(b, g) for h, b in c.beta.items()},
                      Counter({f(h): n for h, n in c.R_inst.items()}))
        else:
            x = _cand(ca, f(c.ctx), c.schema, {h: relabel_vec(b, g) for h, b in c.beta.items()},
                      Counter({f(h): n for h, n in c.R_inst.items()}))
        out.add((x.site_key, x.schema.schema_hash, x.target_signature))
    return out


# ---------------------------------------------------------------- concretization into the original variables


def map_action(a, src, perm, dst, memo):
    """Relabel an action by `perm` from arena `src` into `dst`: primitives, visible rewrites and latent rewrites (d is
    relabelled too), with identities recomputed in the destination. A T6 handle binding maps to the relabelled handle
    with its semantic key recomputed in `dst`."""
    f = lambda h: relabel_into(src, h, perm, dst, memo)
    if isinstance(a, (LatentCandidate, RM.Candidate)) and EP.has_hb(a.beta):
        beta = {i: EP.hb(dst, f(b.h)) if isinstance(b, EP.HB) else relabel_vec(b, perm) for i, b in a.beta.items()}
        R = Counter({f(h): n for h, n in a.R_inst.items()})
        if isinstance(a, LatentCandidate):
            return EP.lcand_g(dst, f(a.ctx), relabel_vec(a.d, perm), a.schema, beta, R)
        return EP.cand_g(dst, f(a.ctx), a.schema, beta, R)
    beta = lambda c: {i: relabel_vec(b, perm) for i, b in c.beta.items()}
    if isinstance(a, LatentCandidate):
        return lcand(dst, f(a.ctx), relabel_vec(a.d, perm), a.schema, beta(a), Counter({f(h): n for h, n in a.R_inst.items()}))
    if isinstance(a, RM.Candidate):
        return _cand(dst, f(a.ctx), a.schema, beta(a), Counter({f(h): n for h, n in a.R_inst.items()}))
    if isinstance(a, tuple) and a and a[0] == 'SUB':
        return ('SUB', f(a[1]), f(a[2]), a[3])
    return v1.Action(a.kind, f(a.ctx), tuple(sorted((f(h), c) for h, c in a.region)),
                     f(a.arg) if a.arg >= 0 else a.arg, tuple(sorted((f(h), c) for h, c in a.src)) if a.src else a.src)


def lineage(res) -> tuple:
    tree = res['tree']
    x, states = res['incumbent'], []
    while x is not None:
        states.append(x)
        x = tree[x]['parent']
    states.reverse()
    return states


def concretize(res, arena_out) -> tuple:
    """The canonical lineage as ordinary concrete steps in the ORIGINAL variables (C7.1b's construction, extended to
    latent actions and to normalization substitutions): returns (states, steps) where steps[j] is the list of
    concrete steps from states[j] to states[j+1]."""
    ca, tf0, tree = res['arena'], res['tf0'], res['tree']
    states = lineage(res)
    pinv = inverse(tf0.pi)
    U = IDENT
    out_s, out_steps = [], []
    for j, C in enumerate(states):
        f = compose(pinv, inverse(U))
        memo: dict = {}
        out_s.append(relabel_into(ca, C, f, arena_out, memo))
        if j+1 < len(states):
            node = tree[states[j+1]]
            act = node['action']
            if hasattr(act, 'h2') and hasattr(act, 'cand'):          # a VIEW: substitute h -> h2, then its rewrite in V
                steps = [map_action(('SUB', act.h, act.h2, 'VIEW'), ca, f, arena_out, memo),
                         map_action(act.cand, ca, f, arena_out, memo)]
            else:
                steps = [map_action(act, ca, f, arena_out, memo)]
            steps += [map_action(('SUB', e['h'], e['h2'], 'NORM'), ca, f, arena_out, memo) for e in node.get('norm', [])]
            out_steps.append(steps)
            U = compose(node['upsilon'], U)
    return out_s, out_steps


def apply_step(arena, target: dict, root: int, a):
    """Exact successor of one concrete step (primitive, visible rewrite, latent rewrite or substitution)."""
    if isinstance(a, tuple) and a[0] == 'SUB':
        assert a[1] in v1.state_info(arena, root).reach_h, 'substitution of an unreachable handle'
        return v1.substitute(arena, root, a[1], a[2])
    if isinstance(a, (LatentCandidate, RM.Candidate)) and EP.has_hb(a.beta):
        r, why = EP.apply_any_g(arena, root, a, target)
        return r if why == 'ok' else None
    if isinstance(a, LatentCandidate):
        r, why = apply_latent(arena, root, a, target)
        return r if why == 'ok' else None
    if isinstance(a, RM.Candidate):
        r, why = RM.apply_candidate(arena, root, a, target)
        return r if why == 'ok' else None
    return v1.MODULES['calculus']._apply(arena, v1.state_info(arena, root), root, a)


def replay(arena, target: dict, states: list, steps: list) -> bool:
    """Every concrete step applied in order reproduces the next concrete state exactly."""
    cur = states[0]
    for j, el in enumerate(steps):
        for a in el:
            cur = apply_step(arena, target, cur, a)
            if cur is None:
                return False
        if v1.digest(arena, cur) != v1.digest(arena, states[j+1]):
            return False
    return v1.poly_equal(v1.hpoly(arena, cur), target)


# ---------------------------------------------------------------- serialization (the multiplier travels explicitly)


def encode_step(arena, a, index) -> list:
    if isinstance(a, tuple) and a[0] == 'SUB':
        return ['SUB', index[a[1]], index[a[2]], a[3]]
    if isinstance(a, (LatentCandidate, RM.Candidate)) and EP.has_hb(a.beta):
        beta = sorted([i, EP.enc_binding(b, index)] for i, b in a.beta.items())
        region = sorted([index[h], c] for h, c in a.R_inst.items())
        if isinstance(a, LatentCandidate):
            return ['LX', a.schema.schema_hash, index[a.ctx], region, beta, list(a.d), a.target_signature]
        return ['RW', a.schema.schema_hash, index[a.ctx], region, beta, a.target_signature]
    if isinstance(a, LatentCandidate):
        return ['LX', a.schema.schema_hash, index[a.ctx], sorted([index[h], c] for h, c in a.R_inst.items()),
                sorted([i, list(b)] for i, b in a.beta.items()), list(a.d), a.target_signature]
    if isinstance(a, RM.Candidate):
        return ['RW', a.schema.schema_hash, index[a.ctx], sorted([index[h], c] for h, c in a.R_inst.items()),
                sorted([i, list(b)] for i, b in a.beta.items()), a.target_signature]
    return v1.action_record(a, index)


def decode_step(rec: list, ids: list, by_hash: dict, arena):
    if rec[0] == 'SUB':
        return ('SUB', ids[rec[1]], ids[rec[2]], rec[3])
    if rec[0] in ('LX', 'RW'):
        sch = by_hash[rec[1]]
        beta = {i: EP.dec_binding(b, ids, arena) for i, b in rec[4]}
        R = Counter({ids[p]: c for p, c in rec[3]})
        if EP.has_hb(beta):
            c = (EP.lcand_g(arena, ids[rec[2]], tuple(rec[5]), sch, beta, R) if rec[0] == 'LX'
                 else EP.cand_g(arena, ids[rec[2]], sch, beta, R))
        else:
            c = (lcand(arena, ids[rec[2]], tuple(rec[5]), sch, beta, R) if rec[0] == 'LX' else _cand(arena, ids[rec[2]], sch, beta, R))
        assert c.target_signature == rec[-1], 'decoded action does not reproduce its recorded target signature'
        return c
    return v1.action_from_record(rec, ids)


def flatten(arena, target: dict, states: list, steps: list) -> tuple:
    """Every intermediate state of the concrete lineage (a normalization substitution may replace a handle that
    exists only between the action and the substitution), with each step's lineage element."""
    chain, flat, elem, cur = [states[0]], [], [], states[0]
    for j, el in enumerate(steps):
        for a in el:
            cur = apply_step(arena, target, cur, a)
            assert cur is not None, f'step {len(flat)} does not apply'
            chain.append(cur)
            flat.append(a)
            elem.append(j)
        assert v1.digest(arena, cur) == v1.digest(arena, states[j+1]), f'lineage element {j} does not reproduce'
    return chain, flat, elem


def serialize(arena, target: dict, states: list, steps: list) -> dict:
    chain, flat, elem = flatten(arena, target, states, steps)
    nodes, index = v1.serialize(arena, chain, with_digests=True)
    return dict(nodes=nodes, roots=[index[s] for s in chain], elements=elem,
                steps=[encode_step(arena, a, index) for a in flat])


def deserialize_and_replay(body: dict, by_hash: dict, target: dict) -> bool:
    arena = v1.Arena()
    arena.calc_cfg = v1.CalcCfg()
    ids = v1.deserialize(arena, body['nodes'])
    chain = [ids[r] for r in body['roots']]
    cur = chain[0]
    for i, rec in enumerate(body['steps']):
        nxt = apply_step(arena, target, cur, decode_step(rec, ids, by_hash, arena))
        if nxt != chain[i+1]:
            return False
        cur = nxt
    return v1.poly_equal(v1.hpoly(arena, cur), target)
