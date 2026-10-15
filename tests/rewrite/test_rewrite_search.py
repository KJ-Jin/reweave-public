"""Search with relations: identical to primitive search without relations; one admitted primitive or rewrite successor
counts as exactly one expansion; exact verification."""

from collections import Counter

from reweave import v1
from reweave.rewrite_config import RewriteConfig
from reweave.rewrite_match import Candidate
from reweave.rewrite_search import beam_search

from test_rewrite_schema import mono, square


def _task():
    target = {mono(0, 0, 0, 0, 0, 2): 1, mono(0, 0, 0, 0, 0, 1, 1): 2, mono(0, 0, 0, 0, 0, 0, 2): 1,
              mono(1, 1): 2, mono(2): 1, mono(0, 2): 1, mono(0, 0, 0, 0, 0, 0, 0, 1): 1}
    return v1.Task(target, 8, 5, {})


class PreferRewrites:
    def __init__(self, primitive_scorer):
        self.p = primitive_scorer

    def score(self, arena, info, root, actions):
        return self.p.score(arena, info, root, actions)

    def score_combined(self, arena, info, root, actions, cands):
        return self.score(arena, info, root, actions), [1e9]*len(cands)


def test_identical_to_v1_search_without_schemas():
    task = _task()
    budget = v1.Budget(4, 4, 6, 40)
    a1, a2 = v1.Arena(), v1.Arena()
    a1.calc_cfg = a2.calc_cfg = v1.CalcCfg()
    s1, s2 = v1.canonical_start(a1, task.target), v1.canonical_start(a2, task.target)
    inc1, stats, _ = v1.beam_search(a1, task, s1, v1.ImmediateGain(task), v1.CostRanker(), budget, False)
    inc2, summ, _ = beam_search(a2, task, s2, v1.ImmediateGain(task), v1.CostRanker(), budget)
    assert v1.digest(a1, inc1) == v1.digest(a2, inc2)
    assert (stats.expansions, stats.steps) == (summ.outer_expansions, summ.steps)
    assert summ.rewrite_applies == 0 and summ.primitive_applies == summ.outer_expansions


def test_outer_expansion_accounting_primitive_and_rewrite_count_one_each():
    task = _task()
    arena = v1.Arena()
    arena.calc_cfg = v1.CalcCfg()
    start = v1.canonical_start(arena, task.target)
    budget = v1.Budget(3, 4, 4, 25)
    inc, s, tree = beam_search(arena, task, start, PreferRewrites(v1.ImmediateGain(task)), v1.CostRanker(), budget,
                               [square()], RewriteConfig(), record=True)
    kinds = Counter('rewrite' if isinstance(node['action'], Candidate) else 'primitive'
                    for h, node in tree.items() if node['parent'] is not None)
    assert s.rewrite_applies >= 1 and s.primitive_applies >= 1
    assert s.outer_expansions == s.primitive_applies+s.rewrite_applies == len(tree)-1
    assert kinds == Counter(rewrite=s.rewrite_applies, primitive=s.primitive_applies)
    assert s.exact_verification_failures == 0 and s.rewrite_selected >= s.rewrite_applies
    for h in tree:
        assert v1.poly_equal(v1.hpoly(arena, h), task.target)
    assert v1.cost(arena, inc) <= v1.cost(arena, start)
