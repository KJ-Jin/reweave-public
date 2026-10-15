"""Semantic interning: OFF parity, interning is behaviour-neutral, exact normalization, provenance, no arena-id
dependence."""

import ast
import sys
from collections import Counter
from pathlib import Path

import pytest

from reweave import candidate_policy as CP
from reweave import v1
from reweave.canon import semantic as SN
from reweave.canon.config import IntegrationConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixtures as F  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
TIMING = ('cpu_time', 'wall_time', 'peak_rss', 'norm_cpu')
PORTED = ('HKey', 'HandleKeys', 'built_handles', 'reach', '_nonmono', 'Rep', 'SemClass', 'Portfolio', 'normalize')
ORIGINS = {'CURRENT', 'LEARNED_REWRITE', 'SEMANTIC_NORMALIZATION'}


def defs(src):
    return {n.name: ast.dump(n) for n in ast.parse(src).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def search(poly, icfg, arena=None, record=True):
    from reweave.canon.search import beam_search
    if arena is None:
        arena = v1.Arena()
        arena.calc_cfg = v1.CalcCfg()
    root = v1.canonical_start(arena, poly)
    task = v1.Task(poly, 8, 1, {})
    inc, st, tree = beam_search(arena, task, root, F.Flat(), v1.CostRanker(), v1.Budget(6, 24, 6, 60),
                                [F.square(), F.distrib()], F.CFG, record, latent=F.LCFG,
                                policy=CP.CandidatePolicy(CP.ORIGIN_SCHEMA_DIVERSITY), icfg=icfg)
    return arena, task, inc, st, tree


@pytest.mark.parametrize('name', sorted(F.FIXTURES))
def test_interning_alone_is_behaviour_neutral(name):
    a1, _, i1, s1, t1 = search(F.FIXTURES[name], IntegrationConfig())
    a2, _, i2, s2, t2 = search(F.FIXTURES[name], IntegrationConfig(semantic_interning=True))
    assert v1.digest(a1, i1) == v1.digest(a2, i2)
    base = {k: v for k, v in s1.as_dict().items() if k not in TIMING and not k.startswith(('portfolio_', 'keys_'))}
    assert base == {k: v for k, v in s2.as_dict().items() if k in base}
    assert [(n, t1[n]['action'], t1[n]['cost']) for n in sorted(t1)] == [(n, t2[n]['action'], t2[n]['cost']) for n in sorted(t2)]


@pytest.mark.parametrize('name', sorted(F.FIXTURES))
def test_normalization_is_exact_strict_and_provenanced(name):
    a, task, inc, st, tree = search(F.FIXTURES[name], IntegrationConfig(normalization=True))
    assert v1.poly_equal(v1.hpoly(a, inc), task.target)
    assert st.exact_verification_failures == 0 and st.norm_failures == 0
    assert st.portfolio_max_reps <= SN.K_REP
    for n, v in tree.items():
        for e in v.get('norm', []):
            assert e['c_after'] < e['c_before']
            assert e['key'] and e['D_old'] and e['D_new'] and e['origin'] in ORIGINS


def test_normalize_reuses_a_renamed_representative_exactly():
    a = v1.Arena()
    a.calc_cfg = v1.CalcCfg()
    mk = lambda *e: v1.mk_mono(a, F.mono(*e))
    cheap = v1.mk_prod(a, Counter({v1.mk_sum(a, Counter({mk(1): 1, mk(0, 1): 1})): 2}))
    first = v1.mk_prod(a, Counter({mk(0, 0, 0, 0, 1): 1, cheap: 1}))
    exp2 = v1.mk_sum(a, Counter({mk(0, 0, 2): 1, mk(0, 0, 1, 1): 2, mk(0, 0, 0, 2): 1}))
    state = v1.mk_prod(a, Counter({mk(0, 0, 0, 0, 0, 1): 1, exp2: 1}))
    ctx = SN.SemanticCtx(a, v1.hpoly(a, state))
    ctx.port.intern_state(first, 0, 'CURRENT')
    out, ev = SN.normalize(ctx, state, SN.built_handles(a, [state], set()), 1)
    assert ev and out != state and v1.poly_equal(v1.hpoly(a, out), v1.hpoly(a, state))
    assert v1.cost(a, out) < v1.cost(a, state) and ctx.port.stats['failures'] == 0


def junk_arena():
    """An arena with unrelated handles interned first, so every later handle id is shifted."""
    a = v1.Arena()
    a.calc_cfg = v1.CalcCfg()
    ms = [v1.mk_mono(a, F.mono(*(i % 3, (i+1) % 4, i % 2, 0, (i*7) % 3))) for i in range(40)]
    for i in range(0, 38, 2):
        v1.mk_prod(a, Counter({v1.mk_sum(a, Counter({ms[i]: 1, ms[i+1]: 1})): 1, ms[i+2]: 1}))
    return a


@pytest.mark.parametrize('name', sorted(F.FIXTURES))
def test_no_arena_id_dependence(name):
    """Baseline and normalization give the same endpoint and the same normalization events (by semantic key and
    structural digest) whether or not unrelated handles were interned first."""
    ev = lambda tree: sorted((e['key'], e['D_old'], e['D_new'], e['c_before'], e['c_after'])
                             for v in tree.values() for e in v.get('norm', []))
    for icfg in (IntegrationConfig(), IntegrationConfig(normalization=True)):
        a1, _, i1, s1, t1 = search(F.FIXTURES[name], icfg)
        a2, _, i2, s2, t2 = search(F.FIXTURES[name], icfg, arena=junk_arena())
        assert v1.digest(a1, i1) == v1.digest(a2, i2), f'{icfg}: endpoint depends on arena ids'
        assert s1.outer_expansions == s2.outer_expansions and s1.norm_events == s2.norm_events
        assert ev(t1) == ev(t2)
