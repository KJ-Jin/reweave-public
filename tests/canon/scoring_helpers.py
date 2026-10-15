"""Shared helpers of the action-scoring tests (fixture states, records, features, renaming maps)."""

import json
import random
import sys
from collections import Counter
from pathlib import Path


from reweave import candidate_policy as CP
from reweave import v1
from reweave.canon import action_record as ARec
from reweave.canon import exprparam as EP
from reweave.canon import relation as REL
from reweave.canon import action_features as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixtures as F  # noqa: E402
import exprparam_fixtures as T6  # noqa: E402

POLICY = CP.CandidatePolicy(CP.ORIGIN_SCHEMA_DIVERSITY)
BUDGET = (6, 24, 6, 60)
PERMS = [T6.IDENT8]+[tuple(random.Random(s).sample(range(8), 8)) for s in (1, 2, 3)]
ALIAS = ('71a318fc91f1d184125b103b62046debaad201e70450854c9d8fccb23a84f903',
         'af576ff519480dca4fcd2b3371ce26aa2399a1d1c3a049bc41011ed8a9b908b4')
M = lambda *ie: ('M', tuple(ie))  # noqa: E731


def fam_of(s) -> tuple:
    return TF.family_features(json.loads(REL.family_payload(s)))


def numeric(f) -> tuple:
    """Everything the head reads except the state-graph rows; roles as a multiset (the head pools them
    symmetrically; under a variable renaming, value tie-breaks may reorder roles that the relation's own
    symmetry makes structurally identical)."""
    return (f.kind, f.family, tuple(sorted(f.roles)), f.d, f.concrete)


def records_at(a, root, schemas, latent=True):
    info = v1.state_info(a, root)
    prims = v1.enumerate_actions(a, info, root, a.calc_cfg)
    cands = EP.match_state_g(a, root, schemas, F.CFG, True)[0]
    if latent:
        cands += EP.latent_candidates_g(a, root, schemas, F.CFG, F.LCFG, True)[0]
    return ARec.records(a, prims, cands)


def logits(model, a, root, recs):
    import torch
    from reweave.canon import action_model as TM
    info = v1.state_info(a, root)
    feats = TF.record_features(recs, a, root)
    prim = [r.live for r in recs if r.kind == 'PRIMITIVE']
    with torch.no_grad():
        p, vals, rw, _ = TM.t7_forward(model, [(a, info, root, prim)], [[f for f in feats if f is not None]])
    return [float(x) for x in p], [float(x) for x in rw[0]]


def _rename(a, h, b, perm, memo):
    """The handle of `a` rebuilt in `b` with variable i renamed to perm[i]."""
    if h in memo:
        return memo[h]
    node = a.handles[h]
    if isinstance(node, v1.Mono):
        e = [0]*8
        for i, x in enumerate(v1.unpack(node.m)):
            e[perm[i]] += x
        out = v1.mk_mono(b, v1.pack(tuple(e)))
    else:
        kids = Counter({_rename(a, c, b, perm, memo): n for c, n in v1.items(node)})
        out = (v1.mk_sum if isinstance(node, v1.Sum) else v1.mk_prod)(b, kids)
    memo[h] = out
    return out


def _renamed_candidate(a, c, b, perm, memo):
    pv = lambda v: tuple(sum(v[i] for i in range(8) if perm[i] == j) for j in range(8))  # noqa: E731
    beta = {i: (EP.hb(b, _rename(a, x.h, b, perm, memo)) if isinstance(x, EP.HB) else pv(x)) for i, x in c.beta.items()}
    R = Counter({_rename(a, h, b, perm, memo): n for h, n in c.R_inst.items()})
    ctx = _rename(a, c.ctx, b, perm, memo)
    if isinstance(c, EP.LatentCandidate):
        return EP.lcand_g(b, ctx, pv(c.d), c.schema, beta, R)
    return EP.cand_g(b, ctx, c.schema, beta, R)
