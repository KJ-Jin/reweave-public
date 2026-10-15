"""The schema-agnostic scorer head. Roles pooled symmetrically; finite, deterministic scores; no relation-id table
anywhere (the only embedding is the closed action-kind vocabulary); an unseen certified family scores with no table
extension; VIEW_REWRITE interface support."""

import math
import random
from collections import Counter

import pytest
import torch

from reweave import rewrite_schema as S
from reweave import v1
from reweave.canon import action_record as ARec
from reweave.canon import archive as AR
from reweave.canon import exprparam as EP
from reweave.canon import relation as REL
from reweave.canon import action_features as TF
from reweave.canon import action_model as TM
from reweave.recipe import Recipe
from reweave.rewrite_certify import certify

from scoring_helpers import F, M, T6, logits, records_at


@pytest.fixture(scope='module')
def lib():
    return T6.l0()


@pytest.fixture(scope='module')
def model():
    torch.set_num_threads(1)
    return TM.build_t7_model(Recipe(lr=5e-5, layernorm=True, residual=True), 0).eval()


def test_role_pooling_is_parameter_renaming_invariant(lib, model):
    a = T6.arena()
    root, _ = T6.mixed(a)
    recs = [r for r in records_at(a, root, lib, latent=False) if r.kind != 'PRIMITIVE']
    feats = TF.record_features(recs, a, root)
    info = v1.state_info(a, root)
    multi = [f for f in feats if len(f.roles) > 1]
    assert multi
    batch = v1.make_batch([(a, info, root, [])])
    with torch.no_grad():
        v1.forward(model, batch)
        h = model._last_h
        rows = TM.graph_rows(info, batch.offsets[0])
        base = TM.rewrite_logits(model, h, rows, multi)
        for seed in (1, 2):
            perm = []
            for f in multi:
                idx = list(range(len(f.roles)))
                random.Random(seed).shuffle(idx)
                g = TF.ActionFeatures(**{**f.__dict__, 'roles': tuple(f.roles[i] for i in idx),
                                         'role_nodes': tuple(f.role_nodes[i] for i in idx)})
                perm.append(g)
            out = TM.rewrite_logits(model, h, rows, perm)
            assert torch.allclose(base, out, rtol=0, atol=1e-12), 'roles are pooled symmetrically'


def test_scores_are_finite_and_deterministic(lib, model):
    a = T6.arena()
    root, _ = T6.latent_square(a)
    recs = records_at(a, root, lib)
    p1, r1 = logits(model, a, root, recs)
    p2, r2 = logits(model, a, root, recs)
    assert (p1, r1) == (p2, r2) and r1 and all(math.isfinite(x) for x in p1+r1)
    b = T6.arena(17)
    rb, _ = T6.latent_square(b)
    p3, r3 = logits(model, b, rb, records_at(b, rb, lib))
    # The inherited V1 encoder orders graph nodes by handle id: another allocation changes only the floating-point
    # summation order of message passing (the T7 features themselves are allocation-exact; identity mode, whose
    # arena allocation is canonical, is bit-exact).
    assert len(p3) == len(p1) and len(r3) == len(r1)
    assert all(abs(x-y) <= 1e-12*max(1.0, abs(x)) for x, y in zip(p1+r1, p3+r3))


def test_model_has_no_schema_vocabulary(model):
    embs = {n: m.num_embeddings for n, m in model.named_modules() if isinstance(m, torch.nn.Embedding)}
    assert embs == {'kind_emb': len(TF.KINDS)}, 'the only embedding table is the closed action-kind vocabulary'
    legacy = {'schema_emb', 'rw_scorer'}
    assert not legacy & {n.split('.')[0] for n, p in model.named_parameters()}


def test_unseen_certified_family_scores_without_extension(lib, model):
    """A*B + A*C + A*D -> A*(B + C + D): certified, not in the fixed historical vocabulary."""
    new = S.canonicalize(('mon',)*4, ((M((0, 1), (1, 1)), 1), (M((0, 1), (2, 1)), 1), (M((0, 1), (3, 1)), 1)),
                         ((('P', ((M((0, 1)), 1), (('S', ((M((1, 1)), 1), (M((2, 1)), 1), (M((3, 1)), 1))), 1))), 1),))
    assert certify(new)[0]
    assert REL.family_payload(new) not in {REL.family_payload(s) for s in lib}, 'not a known family'
    archive = AR.build_archive(lib+[new], F.CFG)
    assert REL.lookup_surface_exact(archive, new) is not None, 'the archive represents it'
    keys_before = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    n_before = TM.parameter_count(model)
    a = T6.arena()
    H = T6.lin(a, (0, 1))
    root = v1.mk_sum(a, Counter({v1.mk_prod(a, Counter({H: 1, T6.mono(a, {j: 1}): 1})): 1 for j in (2, 3, 4)}))
    recs = records_at(a, root, [new], latent=False)
    new_recs = [r for r in recs if r.kind != 'PRIMITIVE']
    assert new_recs, 'the new relation matches (A := the existing handle H)'
    for r in new_recs:
        assert ARec.validate(r) == [] and r.semantic['family']['id'] == REL.canonical_family_id_ac(new)
    p1, r1 = logits(model, a, root, recs)
    assert len(r1) == len(new_recs) and all(math.isfinite(x) for x in r1)
    assert logits(model, a, root, recs) == (p1, r1)
    assert {k: tuple(v.shape) for k, v in model.state_dict().items()} == keys_before
    assert TM.parameter_count(model) == n_before, 'no embedding-table extension'


def test_view_rewrite_interface(lib, model):
    from reweave.canon import views as VW
    b = T6.arena()
    root, hs = T6.repeated_semantic(b)
    V = v1.substitute(b, root, hs['H2'], hs['H'])
    cand = [c for c in EP.match_state_g(b, V, lib, F.CFG, True)[0] if EP.has_hb(c.beta)][0]
    succ, why = EP.apply_any_g(b, V, cand, v1.hpoly(b, V))
    assert why == 'ok'
    va = VW.ViewAction(hs['H2'], hs['H'], 'k', 'D0', 'D1', 1, cand, succ, v1.digest(b, succ), 'CURRENT', 'fixture')
    rec = ARec.view_record(b, va)
    f = TF.view_features(rec, b, V)
    assert f.kind == TF.KINDS.index('VIEW_REWRITE')
    info = v1.state_info(b, V)
    with torch.no_grad():
        p, vals, rw, _ = TM.t7_forward(model, [(b, info, V, [])], [[f]])
    assert len(rw[0]) == 1 and math.isfinite(float(rw[0][0]))
