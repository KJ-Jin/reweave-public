"""Parameterized relations: canonicalization / hash, the Square fixture, certification, runtime source matching, exact
target lowering and application. Fixtures are tests only; they never enter a mined library."""

from collections import Counter

from reweave import v1
from reweave import rewrite_schema as S
from reweave.rewrite_certify import bounds_violation, certify
from reweave.rewrite_config import RewriteConfig
from reweave.rewrite_library import rematch
from reweave.rewrite_match import action_of, apply_candidate, candidate_of, match_state, provenance

SQUARE_HASH = '3cac496ef8bd6e7bd9eac9103429ab4a9f741635462e844c6731b1c74cadfbea'
CFG = RewriteConfig()


def M(*ie):
    return ('M', tuple(ie))


def square():
    src = ((M((0, 2)), 1), (M((0, 1), (1, 1)), 2), (M((1, 2)), 1))
    tgt = ((('P', ((('S', ((M((0, 1)), 1), (M((1, 1)), 1))), 2),)), 1),)
    return S.canonicalize(('mon', 'mon'), src, tgt)


def mono(*e):
    return v1.poly_mod.pack(tuple(e)+(0,)*(8-len(e)))


def test_square_hash_certificate_plan_bounds():
    sch = square()
    assert sch.schema_hash == SQUARE_HASH
    assert sorted(sch.automorphisms) == [(0, 1), (1, 0)]
    assert certify(sch)[0] and S.has_plan(sch) and bounds_violation(sch, CFG) is None
    wrong = S.canonicalize(('mon', 'mon'), ((M((0, 2)), 1), (M((0, 1), (1, 1)), 1), (M((1, 2)), 1)), sch.target)
    assert not certify(wrong)[0]


def test_canonical_hash_is_invariant_under_hole_renaming_and_child_order():
    src = ((M((1, 2)), 1), (M((1, 1), (0, 1)), 2), (M((0, 2)), 1))
    tgt = ((('P', ((('S', ((M((1, 1)), 1), (M((0, 1)), 1))), 2),)), 1),)
    assert S.canonicalize(('mon', 'mon'), src, tgt).schema_hash == SQUARE_HASH
    j = square().to_json()
    assert S.Schema.from_json(j).schema_hash == SQUARE_HASH and S.Schema.from_json(j).body == square().body


def _run(target_terms):
    arena = v1.Arena()
    arena.calc_cfg = v1.CalcCfg()
    target = dict(target_terms)
    root = v1.canonical_start(arena, target)
    cands, work = match_state(arena, root, [square()], CFG)
    return arena, root, target, cands, work


def test_held_out_square_direct_match_and_exact_application():
    arena, root, target, cands, work = _run({mono(0, 0, 0, 0, 0, 2): 1, mono(0, 0, 0, 0, 0, 1, 1): 2,
                                             mono(0, 0, 0, 0, 0, 0, 2): 1, mono(0, 0, 0, 0, 0, 0, 0, 1): 1})
    assert len(cands) == 1 and work['cut_attempts'] == 0 and work['deduped'] == 1
    before = len(arena.handles)
    root2, reason = apply_candidate(arena, root, cands[0], target)
    assert reason == 'ok' and v1.cost(arena, root) == 7 and v1.cost(arena, root2) == 3
    assert v1.poly_equal(v1.hpoly(arena, root2), target)
    assert provenance(arena, root, cands[0]) == 'root_partial'
    assert len(arena.handles) > before          # the target structure is created only at application
    # the runtime action record re-applies to the identical successor
    again, why = apply_candidate(arena, root, candidate_of(action_of(arena, cands[0]), square()), target)
    assert (again, why) == (root2, 'ok')


def test_monomial_binding_xy_z_and_partial_multiplicity():
    arena, root, target, cands, work = _run({mono(2, 2): 1, mono(1, 1, 1): 3, mono(0, 0, 2): 1})
    assert len(cands) == 1
    assert sorted(cands[0].beta.values()) == sorted([(1, 1, 0, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0, 0)])
    root2, reason = apply_candidate(arena, root, cands[0], target)
    assert reason == 'ok' and v1.cost(arena, root2) == 6


def test_negatives_and_injectivity():
    for terms in ({mono(2): 1, mono(1, 1): 1, mono(0, 2): 1},
                  {mono(2): 1, mono(1, 1): 2, mono(0, 3): 1},
                  {mono(2): 4}):
        arena, root, target, cands, work = _run(terms)
        assert cands == []


def test_inexact_splice_is_rejected():
    arena, root, target, cands, work = _run({mono(0, 0, 0, 0, 0, 2): 1, mono(0, 0, 0, 0, 0, 1, 1): 2,
                                             mono(0, 0, 0, 0, 0, 0, 2): 1, mono(0, 0, 0, 0, 0, 0, 0, 1): 1})
    wrong_target = dict(target)
    wrong_target[mono(1)] = 1
    assert apply_candidate(arena, root, cands[0], wrong_target) == (None, 'inexact')


def test_region_anchored_rematch_is_exact():
    arena = v1.Arena()
    a, b = v1.mk_mono(arena, mono(1)), v1.mk_mono(arena, mono(0, 1))
    R = Counter({v1.mk_mono(arena, mono(2)): 1, v1.mk_mono(arena, mono(1, 1)): 2, v1.mk_mono(arena, mono(0, 2)): 1})
    T = Counter({v1.mk_prod(arena, Counter({v1.mk_sum(arena, Counter({a: 1, b: 1})): 2})): 1})
    status, valid = rematch(square(), arena, R, T, CFG)
    assert status == 'instance' and len(valid) == 2
    assert len({S.normalized_signature(square(), bb) for bb in valid}) == 1
