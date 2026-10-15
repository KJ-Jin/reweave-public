"""The versioned, scorer-facing ActionRecord: one generic representation for every selectable action.

    PRIMITIVE          a FACTOR / REUSE action (a stable structural descriptor; no relation family)
    VISIBLE_REWRITE    a certified relation applied at a visible site
    LATENT_REWRITE     the same inside a monomial quotient view, with the latent multiplier d
    VIEW_REWRITE       a rewrite enabled by substituting an alternative representation

A record has four strictly separated parts:

    semantic     SEMANTIC IDENTITY: the AC family (id + the exact canonical relation), the parameter values in
                 canonical family parameter order (exact polynomials), d, and the site (context digest and region)
    concrete     CONCRETE REALIZATION: the exact lowered target DAG and the site
    provenance   origin, lane, surfaces, replay information: never identity
    features     scorer inputs; adding one never changes a key

Never in any part: arena ids, creator ids, representative ids, allocation order.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field

from .. import v1
from ..hashing import canonical_json
from ..rewrite_latent import LatentCandidate
from . import exprparam as EP

RECORD_VERSION = 'reweave-action-record-v1'
KINDS = ('PRIMITIVE', 'VISIBLE_REWRITE', 'LATENT_REWRITE', 'VIEW_REWRITE')
LANES = ('primitive', 'legacy', 'handle', 'view')


@dataclass
class ActionRecord:
    kind: str
    semantic_key: str
    semantic: dict
    concrete_key: str
    concrete: dict
    provenance: dict
    features: dict
    live: object = field(default=None, repr=False, compare=False)

    def to_json(self) -> dict:
        return dict(v=RECORD_VERSION, kind=self.kind, semantic_key=self.semantic_key, semantic=self.semantic,
                    concrete_key=self.concrete_key, concrete=self.concrete, provenance=self.provenance,
                    features=self.features)

    def dumps(self) -> str:
        return canonical_json(self.to_json())

    @staticmethod
    def from_json(j: dict) -> 'ActionRecord':
        assert j['v'] == RECORD_VERSION, j['v']
        return ActionRecord(j['kind'], j['semantic_key'], j['semantic'], j['concrete_key'], j['concrete'],
                            j['provenance'], j['features'])


def _key(kind: str, part: str, payload) -> str:
    """`sha256_json([RECORD_VERSION, part, kind, payload])`, byte for byte. Record payloads are already plain JSON
    (str-keyed dicts, lists, str, int, None), for which `hashing.canonical_json` reduces to this `json.dumps`; the
    generic walker is skipped (it dominated the record cost). Tests pin the equality."""
    return hashlib.sha256(json.dumps([RECORD_VERSION, part, kind, payload], sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


_FID_V1: dict = {}


def family_id_v1(sch) -> str:
    """The historical v1 family id of a surface (provenance only), memoized per surface."""
    from .archive import canonical_family_id
    from .relation import surface_memo
    return surface_memo(_FID_V1, sch, canonical_family_id)


def _dig(arena, h: int) -> str:
    return v1.digest(arena, h).hex()


def _pairs(arena, pairs) -> list:
    return [[d, c] for d, c in sorted((_dig(arena, h), c) for h, c in pairs)]


def _poly(t: tuple) -> list:
    return [[list(v1.unpack(m)), c] for m, c in t]


def _plain(x):
    """Nested tuples -> lists (JSON-stable)."""
    if isinstance(x, tuple):
        return [_plain(y) for y in x]
    return x


def _count_nodes(x) -> int:
    if isinstance(x, (list, tuple)) and x and x[0] in ('S', 'P'):
        return 1+sum(_count_nodes(c)*n for c, n in x[1])
    return 1


# ---------------------------------------------------------------- rewrite actions


def rewrite_semantic(arena, cand, roles: tuple | None = None) -> dict:
    from .relation import family_payload
    holes, vals = roles or EP.canonical_roles(arena, cand)
    return dict(family=dict(id=EP.family_frames(cand.schema)[0], relation=json.loads(family_payload(cand.schema))),
                parameters=[dict(role=j, value=_poly(v)) for j, v in enumerate(vals)],
                d=list(cand.d) if isinstance(cand, LatentCandidate) else None,
                site=dict(context=_dig(arena, cand.ctx), region=_pairs(arena, cand.R_inst.items())))


def rewrite_concrete(arena, cand) -> dict:
    return dict(site=dict(context=_dig(arena, cand.ctx), region=_pairs(arena, cand.R_inst.items()),
                          d=list(cand.d) if isinstance(cand, LatentCandidate) else None),
                target=_plain(EP.target_payload(cand.schema, cand.beta, ref=lambda b: b.dig)))


def rewrite_kind(cand) -> str:
    return 'LATENT_REWRITE' if isinstance(cand, LatentCandidate) else 'VISIBLE_REWRITE'


def rewrite_keys(arena, cand) -> tuple:
    kind = rewrite_kind(cand)
    return _key(kind, 'semantic', rewrite_semantic(arena, cand)), _key(kind, 'concrete', rewrite_concrete(arena, cand))


def rewrite_record(arena, cand, lane: str, origin: str | None = None) -> ActionRecord:
    kind = rewrite_kind(cand)
    holes, vals = roles = EP.canonical_roles(arena, cand)
    sem, con = rewrite_semantic(arena, cand, roles), rewrite_concrete(arena, cand)
    beta = cand.beta
    prov = dict(origin=origin or ('mainline_latent' if kind == 'LATENT_REWRITE' else 'mainline_match'), lane=lane,
                surfaces=sorted(getattr(cand, 'surfaces', ()) or (cand.schema.schema_hash,)),
                family_id_v1=family_id_v1(cand.schema), class_size=getattr(cand, 'class_size', None),
                orbit_size=getattr(cand, 'class_orbit_size', None),
                role_of_parameter={str(h): j for j, h in enumerate(holes)},
                replay=dict(surface=cand.schema.schema_hash, context=_dig(arena, cand.ctx),
                            region=_pairs(arena, cand.R_inst.items()),
                            d=list(cand.d) if kind == 'LATENT_REWRITE' else None,
                            bindings=[dict(param=i, mode=EP.mode(b), handle=b.dig if isinstance(b, EP.HB) else None,
                                           exponents=None if isinstance(b, EP.HB) else list(b))
                                      for i, b in sorted(beta.items())]))
    feats = dict(arity=len(holes), role_modes=[EP.mode(beta[h]) for h in holes],
                 role_degrees=[max(sum(v1.unpack(m)) for m, c in v) for v in vals],
                 role_terms=[len(v) for v in vals], d_degree=sum(cand.d) if kind == 'LATENT_REWRITE' else 0,
                 region_children=len(cand.R_inst), region_multiplicity=sum(cand.R_inst.values()),
                 target_nodes=sum(_count_nodes(x)*n for x, n in con['target']))
    return ActionRecord(kind, _key(kind, 'semantic', sem), sem, _key(kind, 'concrete', con), con, prov, feats, cand)


# ---------------------------------------------------------------- primitive and view actions


def primitive_record(arena, a) -> ActionRecord:
    node = arena.handles[a.arg] if a.arg >= 0 else None
    desc = dict(op=a.kind, context=_dig(arena, a.ctx), region=_pairs(arena, a.region),
                arg=_dig(arena, a.arg) if a.arg >= 0 else None,
                arg_monomial=list(v1.unpack(node.m)) if isinstance(node, v1.Mono) else None,
                source=_pairs(arena, a.src) if a.src else None)
    feats = dict(op=a.kind, region_children=len(a.region), region_multiplicity=sum(c for h, c in a.region),
                 has_source=a.src is not None)
    return ActionRecord('PRIMITIVE', _key('PRIMITIVE', 'semantic', dict(descriptor=desc)), dict(descriptor=desc),
                        _key('PRIMITIVE', 'concrete', dict(descriptor=desc)), dict(descriptor=desc),
                        dict(origin='v1_enumerate', lane='primitive', replay=dict(descriptor=desc)), feats, a)


def view_record(arena, va) -> ActionRecord:
    """A T5 view action: substitute h -> h2 (same exact value, another representation), then a rewrite in V."""
    inner = rewrite_record(arena, va.cand, 'view', origin='view_rewrite')
    value = EP.value_tuple(arena, EP.hb(arena, va.h))
    sem = dict(substituted_value=_poly(value), rewrite=inner.semantic)
    canon = va.canon.hex() if isinstance(va.canon, bytes) else str(va.canon)
    con = dict(successor_canonical=canon, substitution=dict(original=_dig(arena, va.h), alternative=_dig(arena, va.h2)),
               rewrite=inner.concrete)
    prov = dict(origin=f'view:{va.origin}', source=va.source, lane='view', class_size=va.size,
                replay=dict(original=_dig(arena, va.h), alternative=_dig(arena, va.h2), rewrite=inner.provenance['replay']))
    feats = dict(inner.features, view_class_size=va.size)
    return ActionRecord('VIEW_REWRITE', _key('VIEW_REWRITE', 'semantic', sem), sem,
                        _key('VIEW_REWRITE', 'concrete', con), con, prov, feats, va)


def records(arena, actions: list, cands: list, views: list = ()) -> list:
    """Records of one state's selectable actions, in the order the policy scores them: primitives, rewrites (legacy
    lane first, then the handle lane, exactly as the candidate list), views."""
    out = [primitive_record(arena, a) for a in actions]
    out += [rewrite_record(arena, c, 'handle' if EP.has_hb(c.beta) else 'legacy',
                           origin=None if not EP.has_hb(c.beta) else
                           ('handle_latent' if isinstance(c, LatentCandidate) else 'handle_visible')) for c in cands]
    out += [view_record(arena, va) for va in views]
    return out


# ---------------------------------------------------------------- validation, rebinding, the legacy scorer adapter


def validate(rec: ActionRecord) -> list:
    """Schema check of one record; returns the list of violations (empty = valid)."""
    bad = []
    if rec.kind not in KINDS:
        bad.append('kind')
    if rec.semantic_key != _key(rec.kind, 'semantic', rec.semantic):
        bad.append('semantic_key')
    if rec.concrete_key != _key(rec.kind, 'concrete', rec.concrete):
        bad.append('concrete_key')
    if rec.provenance.get('lane') not in LANES:
        bad.append('lane')
    if rec.kind in ('VISIBLE_REWRITE', 'LATENT_REWRITE'):
        sem = rec.semantic
        if set(sem) != {'family', 'parameters', 'd', 'site'}:
            bad.append('semantic_fields')
        if [p['role'] for p in sem['parameters']] != list(range(sem['family']['relation'][1])):
            bad.append('roles')
        if (sem['d'] is None) != (rec.kind == 'VISIBLE_REWRITE'):
            bad.append('d')
        if set(rec.concrete) != {'site', 'target'}:
            bad.append('concrete_fields')
    elif rec.kind == 'PRIMITIVE':
        if set(rec.semantic) != {'descriptor'}:
            bad.append('semantic_fields')
    elif set(rec.semantic) != {'substituted_value', 'rewrite'}:
        bad.append('semantic_fields')
    j = rec.to_json()
    if ActionRecord.from_json(json.loads(canonical_json(j))).to_json() != j:
        bad.append('roundtrip')
    return bad


def rebind(rec: ActionRecord, arena, root: int, by_hash: dict):
    """The live action of a record in a state, from structural digests alone (replay sufficiency of the record)."""
    dmap = {v1.digest(arena, h).hex(): h for h in v1.state_info(arena, root).reach_h}

    def H(d):
        return dmap[d]
    if rec.kind == 'PRIMITIVE':
        d = rec.provenance['replay']['descriptor']
        if d['arg'] is None:
            arg = -1
        elif d['arg'] in dmap:
            arg = dmap[d['arg']]
        else:
            arg = v1.mk_mono(arena, v1.pack(tuple(d['arg_monomial'])))
        ctx = H(d['context'])
        pos = {h: i for i, (h, c) in enumerate(v1.items(arena.handles[ctx]))}    # V1 regions keep the items order

        def region(pairs):
            return tuple(sorted(((H(x), c) for x, c in pairs), key=lambda hc: pos[hc[0]]))
        return v1.Action(d['op'], ctx, region(d['region']), arg, None if d['source'] is None else region(d['source']))
    if rec.kind == 'VIEW_REWRITE':
        raise ValueError('a view action rebinds in its substituted state (T6B); its rewrite part rebinds there')
    r = rec.provenance['replay']
    sch = by_hash[r['surface']]
    beta = {b['param']: (EP.hb(arena, H(b['handle'])) if b['mode'] == 'REACHABLE_HANDLE_BIND' else tuple(b['exponents']))
            for b in r['bindings']}
    R = Counter({H(x): c for x, c in r['region']})
    if rec.kind == 'LATENT_REWRITE':
        return EP.lcand_g(arena, H(r['context']), tuple(r['d']), sch, beta, R)
    return EP.cand_g(arena, H(r['context']), sch, beta, R)


def score_records(scorer, arena, info, root: int, recs: list, views=None, vctx=None) -> list:
    """The LEGACY adapter: the fixed-vocabulary scorer consumes records. It calls exactly the frozen scoring
    functions, with the live objects in the records' order (primitives and rewrites in one encoder pass through
    `exprparam.score_combined_g`, which is `scorer.score_combined` whenever no handle binding is present; views
    through `views.scores`), and returns one score per record in record order. A scorer that consumes records
    natively (T7 SCHEMA_AGNOSTIC_POLICY) is called with them directly; the legacy path below is unchanged."""
    if hasattr(scorer, 'score_action_records'):
        return scorer.score_action_records(arena, info, root, recs, views, vctx)
    prim = [i for i, r in enumerate(recs) if r.kind == 'PRIMITIVE']
    rw = [i for i, r in enumerate(recs) if r.kind in ('VISIBLE_REWRITE', 'LATENT_REWRITE')]
    vw = [i for i, r in enumerate(recs) if r.kind == 'VIEW_REWRITE']
    p, q = EP.score_combined_g(scorer, arena, info, root, [recs[i].live for i in prim], [recs[i].live for i in rw])
    w = views.scores(vctx, root, [recs[i].live for i in vw]) if vw else []
    out = [0.0]*len(recs)
    for idx, vals in ((prim, p), (rw, q), (vw, w)):
        assert len(idx) == len(vals)
        for i, x in zip(idx, vals):
            out[i] = x
    return out
