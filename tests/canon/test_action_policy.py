"""The schema-agnostic policy behind the ActionRecord boundary (exact, deterministic, refused outside a generic
profile); the fixed-vocabulary scorer keeps its parity-exact adapter; explicit policy identities."""

import pytest
import torch

from reweave import v1
from reweave.canon import action_record as ARec
from reweave.canon import exprparam as EP
from reweave.canon import search as CS
from reweave.canon import action_model as TM
from reweave.canon import action_policy as TP
from reweave.canon.config import PROFILES
from reweave.recipe import Recipe
from reweave.rewrite_model import RewriteValueRanker

from scoring_helpers import BUDGET, F, POLICY, T6, records_at


@pytest.fixture(scope='module')
def lib():
    return T6.l0()


@pytest.fixture(scope='module')
def model():
    torch.set_num_threads(1)
    return TM.build_t7_model(Recipe(lr=5e-5, layernorm=True, residual=True), 0).eval()


def test_search_integration_behind_the_record_boundary(lib, model):
    sc, ranker = TP.SchemaAgnosticScorer(model), RewriteValueRanker(model)

    def run(icfg):
        return CS.run(v1.Task(F.FIXTURES['TWO_SQ'], 8, 1, {}), sc, ranker, v1.Budget(*BUDGET), lib, F.CFG, True,
                      latent=F.LCFG, policy=POLICY, icfg=icfg)
    r1, r2 = run(PROFILES['GENERIC_HANDLES']), run(PROFILES['GENERIC_HANDLES'])
    a = r1['arena']
    assert v1.poly_equal(v1.hpoly(a, r1['incumbent']), F.FIXTURES['TWO_SQ'])
    sig = lambda res: [(n.get('scores'), n.get('rewrite_scores'), n['cost']) for n in res['tree'].values()]  # noqa
    assert sig(r1) == sig(r2), 'deterministic'
    assert sc.cpu['states'] > 0 and sc.cpu['rewrite_records'] > 0
    with pytest.raises(RuntimeError):
        run(PROFILES['INTEGRATION_BASELINE'])


def test_legacy_adapter_is_untouched_for_the_fixed_scorer(lib):
    from reweave.rewrite_model import RewriteScorer, build_rewrite_model
    m = build_rewrite_model(Recipe(lr=5e-5, layernorm=True, residual=True), 0, len(lib))
    sc = RewriteScorer(m, {s.schema_hash: i for i, s in enumerate(lib)},
                       {s.schema_hash: [1.0, float(len(s.types)), 0.5] for s in lib})
    assert not hasattr(sc, 'score_action_records')
    a = T6.arena()
    root, _ = T6.latent_square(a)
    info = v1.state_info(a, root)
    recs = records_at(a, root, lib)
    mono = [r for r in recs if r.kind == 'PRIMITIVE' or not EP.has_hb(r.live.beta)]
    p, q = sc.score_combined(a, info, root, [r.live for r in mono if r.kind == 'PRIMITIVE'],
                             [r.live for r in mono if r.kind != 'PRIMITIVE'])
    assert ARec.score_records(sc, a, info, root, mono) == p+q


def test_policy_identities_are_explicit(model):
    a = TP.policy_identity('SCHEMA_AGNOSTIC_POLICY', model)
    b = TP.policy_identity('LEGACY_FIXED_VOCAB_POLICY', None, checkpoint='x')
    assert a['policy_hash'] != b['policy_hash'] and a['model_version'] == TM.T7_MODEL_VERSION


def test_library_order_never_changes_a_score(lib, model):
    """The schema-agnostic scorer reads no library table: a shuffled library gives the identical recorded tree."""
    import random

    def sig(schemas):
        sc = TP.SchemaAgnosticScorer(model)
        res = CS.run(v1.Task(F.FIXTURES['CUBE'], 8, 1, {}), sc, RewriteValueRanker(model), v1.Budget(*BUDGET), schemas,
                     F.CFG, True, latent=F.LCFG, policy=POLICY, icfg=PROFILES['GENERIC_HANDLES'])
        a = res['arena']
        return sorted((v1.digest(a, h).hex(), n.get('scores'), sorted(zip((r.concrete_key for r in
                       n.get('action_records', []) if r.kind != 'PRIMITIVE'), n.get('rewrite_scores') or [])))
                      for h, n in res['tree'].items())
    base = sig(list(lib))
    for seed in (1, 2):
        perm = list(lib)
        random.Random(seed).shuffle(perm)
        assert sig(perm) == base
