"""Latent exposure: bounded depth-1 matching through monomial-quotient views.

Fixtures are tests only; they never enter a mined library. Every assertion here is about `reweave/rewrite_latent.py`
and its additive wiring into search and labels."""

import re
from collections import Counter
from pathlib import Path

from reweave import v1
from reweave import rewrite_latent as L
from reweave import rewrite_schema as S
from reweave.rewrite_config import RewriteConfig
from reweave.rewrite_match import match_state

REPO = Path(__file__).resolve().parent.parent.parent
CFG = RewriteConfig()
LCFG = L.LatentConfig()


def M(*ie):
    return ('M', tuple(ie))


def mono(*e):
    return v1.pack(tuple(e)+(0,)*(8-len(e)))


def square():
    """(H0+H1)^2 -> H0^2 + 2 H0H1 + H1^2 read right to left: source is the expanded square, target the product.

    Source multiplicity 2 on the cross term; used for the multiplicity regression test."""
    src = ((M((0, 2)), 1), (M((0, 1), (1, 1)), 2), (M((1, 2)), 1))
    tgt = ((('P', ((('S', ((M((0, 1)), 1), (M((1, 1)), 1))), 2),)), 1),)
    return S.canonicalize(('mon', 'mon'), src, tgt)


def cube():
    """(H0+H1)^3 expanded: coefficients 1, 3, 3, 1 — source multiplicity 3."""
    src = ((M((0, 3)), 1), (M((0, 2), (1, 1)), 3), (M((0, 1), (1, 2)), 3), (M((1, 3)), 1))
    tgt = ((('P', ((('S', ((M((0, 1)), 1), (M((1, 1)), 1))), 3),)), 1),)
    return S.canonicalize(('mon', 'mon'), src, tgt)


def arena_of(poly):
    a = v1.Arena()
    a.calc_cfg = v1.CalcCfg()
    return a, v1.canonical_start(a, poly)


def times(poly, e):
    """Multiply a polynomial by a monomial given as an exponent tuple."""
    out = {}
    for m, c in poly.items():
        v = v1.unpack(m)
        out[v1.pack(tuple(v[j]+e[j] for j in range(len(v))))] = c
    return out


def expanded_square(x=0, y=1):
    return {mono(*(2 if j == x else 0 for j in range(max(x, y)+1))): 1,
            mono(*((1 if j in (x, y) else 0) for j in range(max(x, y)+1))): 2,
            mono(*(2 if j == y else 0 for j in range(max(x, y)+1))): 1}


def expanded_cube():
    return {mono(3): 1, mono(2, 1): 3, mono(1, 2): 3, mono(0, 3): 1}


# ---------------------------------------------------------------- 1 / 2: multiplicity regression


def test_quotient_view_allocation_preserves_multiplicity_regression():
    """REGRESSION for the feasibility-prototype bug: the quotient view mapped back to originals one copy per
    distinct child, silently discarding every match that consumes a term with coefficient > 1.

    `allocate` must draw the full available multiplicity from each original. The commented `buggy` allocation below
    is the old behaviour and is asserted to fail on the same input."""
    a = v1.Arena()
    a.calc_cfg = v1.CalcCfg()
    h1, h2 = v1.mk_mono(a, mono(2)), v1.mk_mono(a, mono(1, 1))
    origin = {h1: [(h1, 1)], h2: [(h2, 3)]}
    got = L.allocate(a, Counter({h1: 1, h2: 3}), origin)
    assert got == Counter({h1: 1, h2: 3}), got

    def buggy(view_region, origin):
        out = Counter()
        for q, n in view_region.items():
            need = n
            for h, _avail in origin.get(q, []):
                if need <= 0:
                    break
                out[h] += 1
                need -= 1
            if need > 0:
                return Counter()
        return out
    assert buggy(Counter({h1: 1, h2: 3}), origin) == Counter(), 'the old allocation must fail this case'


def test_latent_recovery_of_a_source_with_multiplicity_three():
    """A coefficient-3 source is recovered latently under a monomial multiplier."""
    sch = cube()
    d = (0, 0, 1)+(0,)*5
    poly = times(expanded_cube(), d)
    a, root = arena_of(poly)
    cands, work = L.latent_candidates(a, root, [sch], CFG, LCFG)
    assert cands, 'a coefficient-3 latent source must be recovered'
    hit = [c for c in cands if c.d == d]
    assert hit, f'expected the multiplier {d}, saw {[c.d for c in cands]}'
    c = hit[0]
    assert sum(c.R_inst.values()) == 8, c.R_inst        # 1 + 3 + 3 + 1 original occurrences
    succ, why = L.apply_latent(a, root, c, poly)
    assert why == 'ok' and v1.poly_equal(v1.hpoly(a, succ), poly)


# ---------------------------------------------------------------- 3 / 4: determinism


def test_divisor_and_view_enumeration_is_deterministic():
    poly = times(expanded_square(), (0, 0, 2)+(0,)*5)
    a, root = arena_of(poly)
    ctx = sorted(v1.state_info(a, root).contexts, key=lambda h: v1.digest(a, h))[0]
    runs = [[d for d, _v, _o in L.views(a, ctx, LCFG)] for _ in range(4)]
    assert all(r == runs[0] for r in runs)
    assert all(any(x) for x in runs[0]), 'the unit multiplier is never a latent view'
    assert len(runs[0]) <= LCFG.divisors_max


def test_latent_binding_enumeration_is_deterministic():
    sch = square()
    poly = times(expanded_square(), (0, 0, 1)+(0,)*5)
    a, root = arena_of(poly)
    keys = [[c.order_key for c in L.latent_candidates(a, root, [sch], CFG, LCFG)[0]] for _ in range(3)]
    assert all(k == keys[0] for k in keys)
    assert keys[0] == sorted(keys[0]), 'candidates are emitted in their deterministic total order'


# ---------------------------------------------------------------- 5: bounded work


def test_exposure_work_bound_fails_closed():
    sch = square()
    poly = times(expanded_square(), (0, 0, 1)+(0,)*5)
    a, root = arena_of(poly)
    full, _ = L.latent_candidates(a, root, [sch], CFG, LCFG)
    tight = L.LatentConfig(attempts_max=5)
    cut, work = L.latent_candidates(a, root, [sch], CFG, tight)
    assert work['cut_attempts'] >= 1 and work['attempts'] <= tight.attempts_max+1
    assert len(cut) <= len(full), 'a work cut may only remove candidates, never invent them'


# ---------------------------------------------------------------- 6 / 7 / 8: parity and dedupe


def test_visible_path_is_untouched_when_latent_is_disabled():
    """The visible matcher is never routed through the latent module, and latent exposure never mutates its output."""
    sch = square()
    poly = times(expanded_square(), (0, 0, 1)+(0,)*5)
    a, root = arena_of(poly)
    before, wbefore = match_state(a, root, [sch], CFG)
    L.latent_candidates(a, root, [sch], CFG, LCFG)
    after, wafter = match_state(a, root, [sch], CFG)
    assert [c.order_key for c in before] == [c.order_key for c in after]
    assert wbefore == wafter
    assert 'match_state' not in _executable_source(), 'the latent module must never call the visible matcher'


def test_visible_candidate_wins_over_a_latent_duplicate():
    sch = square()
    poly = expanded_square()                                    # visible at d = 1
    a, root = arena_of(poly)
    vis, _ = match_state(a, root, [sch], CFG)
    assert vis, 'fixture must have a visible match'
    lat, _ = L.latent_candidates(a, root, [sch], CFG, LCFG)
    kept = L.dedupe_against_visible(a, lat, vis)
    keys = {(c.ctx, tuple(sorted((v1.digest(a, h).hex(), n) for h, n in c.R_inst.items())),
             c.schema.schema_hash, c.target_signature) for c in vis}
    assert not any((c.ctx, tuple(sorted((v1.digest(a, h).hex(), n) for h, n in c.R_inst.items())),
                    c.schema.schema_hash, c.target_signature) in keys for c in kept)


def test_latent_candidates_producing_the_same_successor_are_deduped_by_site_key():
    sch = square()
    poly = times(expanded_square(), (0, 0, 1)+(0,)*5)
    a, root = arena_of(poly)
    cands, _ = L.latent_candidates(a, root, [sch], CFG, LCFG)
    keys = [(c.site_key, c.schema.schema_hash, c.target_signature) for c in cands]
    assert len(keys) == len(set(keys)), 'one candidate per (site, schema, instantiated target)'


# ---------------------------------------------------------------- 9 / 10 / 11 / 12: exactness and cost


def test_latent_successor_is_exact_and_passes_the_inherited_verifier():
    sch = square()
    poly = times(expanded_square(), (0, 0, 1)+(0,)*5)
    a, root = arena_of(poly)
    task = v1.Task(poly, 8, 1, {})
    cands, _ = L.latent_candidates(a, root, [sch], CFG, LCFG)
    applied = 0
    for c in cands:
        succ, why = L.apply_latent(a, root, c, poly)
        if why != 'ok':
            continue
        applied += 1
        assert v1.poly_equal(v1.hpoly(a, succ), poly)
        v1.verify(a, task, succ)
    assert applied, 'fixture must apply at least one latent candidate'


def test_no_library_symbol_in_costed_circuit():
    """Every node reachable from a latent successor is an ordinary Mono/Sum/Prod handle: no schema, hole, binding or
    library object survives into the costed circuit."""
    sch = square()
    poly = times(expanded_square(), (0, 0, 1)+(0,)*5)
    a, root = arena_of(poly)
    cands, _ = L.latent_candidates(a, root, [sch], CFG, LCFG)
    succ = next(s for s, w in (L.apply_latent(a, root, c, poly) for c in cands) if w == 'ok')
    for h in v1.state_info(a, succ).reach_h:
        assert isinstance(a.handles[h], (v1.Mono, v1.Sum, v1.Prod)), a.handles[h]


def test_solution_language_unchanged():
    """A latent successor is an ordinary state of the inherited solution language: the inherited verifier accepts it
    and the inherited primitive enumerator operates on it normally."""
    sch = square()
    poly = times(expanded_square(), (0, 0, 1)+(0,)*5)
    a, root = arena_of(poly)
    task = v1.Task(poly, 8, 1, {})
    cands, _ = L.latent_candidates(a, root, [sch], CFG, LCFG)
    succ = next(s for s, w in (L.apply_latent(a, root, c, poly) for c in cands) if w == 'ok')
    v1.verify(a, task, succ)
    info = v1.state_info(a, succ)
    acts = v1.enumerate_actions(a, info, succ, a.calc_cfg)
    for act in acts[:8]:
        nxt = v1.apply_action(a, task, info, succ, act)
        if nxt is not None:
            v1.verify(a, task, nxt)


def test_latent_exposure_adds_no_cost_discount():
    """The replacement is charged normally: removing the consumed children and adding d * T(theta) must move the cost
    by exactly the inherited cost of the rebuilt state, with no discount applied anywhere."""
    sch = square()
    poly = times(expanded_square(), (0, 0, 1)+(0,)*5)
    a, root = arena_of(poly)
    cands, _ = L.latent_candidates(a, root, [sch], CFG, LCFG)
    checked = 0
    for c in cands:
        succ, why = L.apply_latent(a, root, c, poly)
        if why != 'ok':
            continue
        checked += 1
        # Re-intern the successor alone in a fresh arena: if the reported cost depended on sharing with unrelated
        # handles that happen to live in the working arena, the two costs would differ.
        fresh = v1.Arena()
        fresh.calc_cfg = v1.CalcCfg()
        nodes, index = v1.serialize(a, [succ], with_digests=True)
        ids = v1.deserialize(fresh, nodes)
        assert v1.cost(fresh, ids[index[succ]]) == v1.cost(a, succ)
    assert checked, 'fixture must apply at least one latent candidate'


# ---------------------------------------------------------------- 13 / 14: hygiene invariants


def _executable_source() -> str:
    """The latent module with docstrings and comments stripped: prose may mention anything, logic may not."""
    src = (REPO/'reweave'/'rewrite_latent.py').read_text()
    src = re.sub(r'"""(?:.|\n)*?"""', '', src)
    return re.sub(r'#.*', '', src)


def test_beam_search_without_latent_config_generates_no_latent_candidates():
    """The additive wiring is off by default: with `latent=None` no latent code runs and no counter moves."""
    from reweave.rewrite_search_latent import beam_search

    class _Flat:                                   # the combined-scorer interface, no model involved
        def score(self, arena, info, root, actions):
            return [0.0]*len(actions)

        def score_combined(self, arena, info, root, actions, cands):
            return [0.0]*len(actions), [0.0]*len(cands)

    sch = square()
    poly = times(expanded_square(), (0, 0, 1)+(0,)*5)
    a, root = arena_of(poly)
    task = v1.Task(poly, 8, 1, {})
    inc, st, _ = beam_search(a, task, root, _Flat(), v1.CostRanker(),
                             v1.Budget(4, 4, 4, 20), [sch], CFG, False, latent=None)
    d = st.as_dict()
    assert d['latent_generated'] == 0 and d['latent_selected'] == 0 and d['latent_applies'] == 0
    assert d['latent_attempts'] == 0


def test_runtime_module_has_no_generator_or_family_awareness():
    """The production latent module must contain no benchmark family name, schema hash, task id or generator concept
    in executable code. Prose in the module docstring is allowed; conditional logic is not."""
    body = _executable_source()
    banned = [r'\bF1\b', r'\bF2\b', r'\bF3\b', r'\bF4\b', r'generator', r'task_id', r'\bfamily\b',
              r'product[_ ]of[_ ]sums', r'\b[0-9a-f]{12,}\b']
    for pat in banned:
        assert not re.search(pat, body, re.IGNORECASE), f'{pat} appears in runtime latent-exposure logic'


def test_primitive_r_max_and_region_enumeration_are_unchanged():
    """Latent exposure must not widen generic primitive enumeration: r_max stays 3 and `regions` is untouched."""
    assert v1.CalcCfg().r_max == 3
    body = _executable_source()
    assert 'r_max' not in body and 'enumerate_actions' not in body
    poly = times(expanded_square(), (0, 0, 1)+(0,)*5)
    a, root = arena_of(poly)
    info = v1.state_info(a, root)
    before = [v1.action_key(a, x) for x in v1.enumerate_actions(a, info, root, a.calc_cfg)]
    L.latent_candidates(a, root, [square()], CFG, LCFG)
    after = [v1.action_key(a, x) for x in v1.enumerate_actions(a, info, root, a.calc_cfg)]
    assert before == after


# ---------------------------------------------------------------- 15: machine-class guard
