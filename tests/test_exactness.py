"""The two invariants the whole design rests on, checked on the current mechanism with the reference library.

1. Exactness: every admitted successor (primitive, visible rewrite or latent rewrite) lowers to the ordinary
   ADD/MUL circuit language and is exactly equal to the target polynomial; an inexact candidate is never admitted.
2. No cost discount: the gate cost of a successor is the cost of the ordinary circuit, identical to the cost of the
   same structure rebuilt in a fresh arena with no library present.
"""

from collections import Counter

import pytest

from reweave import v1
from reweave.candidate_policy import CandidatePolicy
from reweave.current_config import CURRENT
from reweave.rewrite_config import RewriteConfig
from reweave.rewrite_latent import LatentCandidate, LatentConfig, apply_latent, latent_candidates
from reweave.rewrite_match import Candidate, apply_candidate, match_state
from reweave.rewrite_search_latent import beam_search

ORDINARY = (v1.Mono, v1.Sum, v1.Prod)
CFG = RewriteConfig()
VARS = 'abcdefgh'


def poly(*terms):
    out = {}
    for spec, c in terms:
        e = [0]*8
        for v in spec.split('*'):
            e[VARS.index(v)] += 1
        m = v1.pack(tuple(e))
        out[m] = out.get(m, 0)+c
    return out


TARGETS = [
    poly(('a*a', 1), ('a*b', 2), ('b*b', 1)),                                              # (a+b)^2
    poly(('f*a*a*b*b', 1), ('f*a*b*c', 2), ('f*c*c', 1), ('d*d', 1), ('d*e', 2), ('e*e', 1)),  # f(ab+c)^2+(d+e)^2
    poly(('a*a*c', 1), ('a*b*c', 2), ('b*b*c', 1), ('a*a*d', 1), ('a*b*d', 2), ('b*b*d', 1)),  # (c+d)(a+b)^2
    poly(('a*b', 1), ('a*c', 1), ('b*c', 2), ('b*b', 1), ('c*c', 1)),                       # (a+b+c)(b+c)
]


class Flat:
    """Every action scores 0: ties break by the deterministic order (primitives first), so no learned model is
    involved and both primitive and rewrite successors are exercised."""

    def score(self, arena, info, root, actions):
        return [0.0]*len(actions)

    def score_combined(self, arena, info, root, actions, cands):
        return [0.0]*len(actions), [1.0]*len(cands)


@pytest.fixture(scope='module')
def library():
    return CURRENT.library.schemas()


@pytest.fixture(scope='module')
def searched(library):
    out = []
    for i, target in enumerate(TARGETS):
        task = v1.Task(target, 8, 0, dict(id=i))
        arena = v1.Arena()
        arena.calc_cfg = v1.CalcCfg()
        start = v1.canonical_start(arena, target)
        best, summary, tree = beam_search(arena, task, start, Flat(), v1.CostRanker(), v1.Budget(4, 8, 6, 80),
                                          library, CFG, True, LatentConfig(), CandidatePolicy())
        out.append((task, arena, start, best, summary, tree))
    return out


def test_every_admitted_successor_is_exact(searched):
    rewrites = 0
    for task, arena, start, best, summary, tree in searched:
        assert summary.exact_verification_failures == 0
        for h, node in tree.items():
            assert v1.poly_equal(v1.hpoly(arena, h), task.target)
            rewrites += isinstance(node.get('action'), (Candidate, LatentCandidate))
        assert v1.poly_equal(v1.hpoly(arena, best), task.target)
    assert rewrites > 0, 'the fixture must exercise admitted rewrite successors'


def test_costed_circuits_contain_only_ordinary_nodes(searched):
    for task, arena, start, best, summary, tree in searched:
        for h in tree:
            assert all(type(arena.handles[x]) in ORDINARY for x in v1.reachable(arena, [h]))


def test_library_use_cannot_change_costs(searched):
    """Cost of a rewrite successor == cost of the same circuit rebuilt in a fresh arena with no library."""
    checked = 0
    for task, arena, start, best, summary, tree in searched:
        for h, node in tree.items():
            if not isinstance(node.get('action'), (Candidate, LatentCandidate)):
                continue
            nodes, index = v1.serialize(arena, [h], with_digests=True)
            fresh = v1.Arena()
            g = v1.deserialize(fresh, nodes)[index[h]]
            assert v1.cost(fresh, g) == v1.cost(arena, h) == node['cost']
            assert v1.standard_cost(fresh, g) == v1.standard_cost(arena, h)
            checked += 1
    assert checked > 0


def test_corrupted_binding_is_rejected_by_exact_verification(library):
    """Tamper with a valid candidate's monomial binding: the lowered successor is inexact and is refused."""
    target = TARGETS[0]
    arena = v1.Arena()
    arena.calc_cfg = v1.CalcCfg()
    root = v1.canonical_start(arena, target)
    cands, _ = match_state(arena, root, library, CFG)
    assert cands
    cand = cands[0]
    hole = sorted(cand.beta)[0]
    wrong = list(cand.beta[hole])
    wrong[7] += 1                                    # multiply that parameter by an unrelated variable h
    bad = Candidate(cand.ctx, cand.schema, {**cand.beta, hole: tuple(wrong)}, Counter(cand.R_inst), cand.site_key,
                    cand.target_signature, cand.binding_digest, cand.order_key)
    good, why_good = apply_candidate(arena, root, cand, target)
    succ, why = apply_candidate(arena, root, bad, target)
    assert why_good == 'ok' and v1.poly_equal(v1.hpoly(arena, good), target)
    assert succ is None and why == 'inexact'


def test_latent_rewrite_is_exact_and_keeps_the_multiplier(library):
    target = poly(('f*a*a', 1), ('f*a*c', 2), ('f*c*c', 1), ('b', 1))     # f(a+c)^2 + b
    arena = v1.Arena()
    arena.calc_cfg = v1.CalcCfg()
    root = v1.canonical_start(arena, target)
    lat, _ = latent_candidates(arena, root, library, CFG, LatentConfig())
    assert lat
    for cand in lat:
        succ, why = apply_latent(arena, root, cand, target)
        assert succ is None or v1.poly_equal(v1.hpoly(arena, succ), target)
