"""Generic ExprParam binding. MONOMIAL_BIND reproduces the monomial matcher candidate for candidate;
REACHABLE_HANDLE_BIND binds concrete reachable compound handles only (no synthesis, no regrouping); positive /
negative / mixed / repeated / non-injective / latent fixtures; fail-closed dependency checks; family and action
identity; renaming and allocation invariance; configuration identity."""

import dataclasses
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

from reweave import candidate_policy as CP
from reweave import rewrite_latent as L
from reweave import rewrite_match as RM
from reweave import v1
from reweave.canon import action_record as ARec
from reweave.canon import archive as AR
from reweave.canon import candidate_set as CSR
from reweave.canon import exprparam as EP
from reweave.canon import exprparam_identity as EPI
from reweave.canon import identity as ID
from reweave.canon import latent_identity as LI
from reweave.canon import relation as REL
from reweave.canon import search as CS
from reweave.canon.config import IntegrationConfig, PROFILES, config_identity
from reweave.canon.labelling import relabel_into

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixtures as F  # noqa: E402
import exprparam_fixtures as T6  # noqa: E402

TIMING = ('cpu_time', 'wall_time', 'peak_rss', 'norm_cpu', 'identity_cpu', 'view_cpu')
T6_FIELDS = tuple(f.name for f in dataclasses.fields(CS.CanonSearchSummary)
                  if f.name.startswith(('gen_', 'hb_', 'csr_')) or f.name == 'action_records')
POLICY = CP.CandidatePolicy(CP.ORIGIN_SCHEMA_DIVERSITY)
BUDGET = (6, 24, 6, 60)
PERMS = [T6.IDENT8]+[tuple(random.Random(s).sample(range(8), 8)) for s in (1, 2, 3)]
RECORDED = {       # configuration hashes of the earlier profiles, pinned
    'INTEGRATION_BASELINE': '79095aeadf41b60388f7ac8e2de913877b429f9663a646e59848f78637deb573',
    'NORMALIZATION': '979ee3c9fab9f0c537d18a1b64eae697b38ae0749c806d4dd431cc772c5bc0fe',
    'ARCHIVE_RETRIEVAL': 'c2405f3726cb874860be3f9dfb65821b6ce614d7cd21b13d9430583d6e76bdd2',
    'IDENTITY': '1b6bd4e603252baf4e4c47a0f45fa11bb89c717ffe198411e593904814ded3b7',
    'IDENTITY_VIEWS': '8497c4aaf7879017c69eff6078b6e1bc0a30138b76f6951d3cc1f71772ccbd10',
    'IDENTITY_VIEWS_EAGER': '66c7212c90cceeb9de593eac5270b58011dc83c8e9bd38d405fe92099ef6810a',
}


@pytest.fixture(scope='module')
def lib():
    return T6.l0()


@pytest.fixture(scope='module')
def sq(lib):
    return next(s for s in lib if s.schema_hash == T6.SQUARE)


def with_fixture_schemas(lib):
    return list({s.schema_hash: s for s in lib+[F.square(), F.distrib()]}.values())


def csig(a, c) -> tuple:
    return (type(c).__name__, v1.digest(a, c.ctx).hex(), c.schema.schema_hash, tuple(sorted(c.beta.items())),
            tuple(sorted((v1.digest(a, h).hex(), n) for h, n in c.R_inst.items())), c.site_key, c.target_signature,
            c.binding_digest, c.order_key, getattr(c, 'd', None))


def gen(a, root, schemas, handles):
    vis, w = EP.match_state_g(a, root, schemas, F.CFG, handles)
    lat, lw = EP.latent_candidates_g(a, root, schemas, F.CFG, F.LCFG, handles)
    return vis, lat, w, lw


def square_hb(cands, sq):
    return [c for c in cands if c.schema.schema_hash == sq.schema_hash and EP.has_hb(c.beta)]


def reach(a, h) -> set:
    return set(v1.state_info(a, h).reach_h)


# ---------------------------------------------------------------- configuration


def test_t0_t5_identities_unchanged_and_t6_switches_explicit():
    for name, h in RECORDED.items():
        assert config_identity(PROFILES[name], name)['config_hash'] == h, f'{name}: T0-T5 identity must not move'
    e = IntegrationConfig().effective()
    assert not (e.generic_exprparam_binding or e.reachable_handle_bind), 'T6 defaults OFF'
    new = {config_identity(PROFILES[n])['config_hash'] for n in
           ('GENERIC_MONOMIAL', 'GENERIC_HANDLES', 'IDENTITY_GENERIC_MONOMIAL', 'IDENTITY_GENERIC_HANDLES')}
    assert len(new) == 4 and not new & set(RECORDED.values())
    assert config_identity(PROFILES['GENERIC_MONOMIAL'])['behaviour_equivalent_to'] is None
    with pytest.raises(ValueError):
        IntegrationConfig(reachable_handle_bind=True).effective()
    with pytest.raises(ValueError):
        IntegrationConfig(generic_exprparam_binding=True, archive_retrieval=True).effective()
    with pytest.raises(ValueError):
        IntegrationConfig(identity_mode=True, generic_exprparam_binding=True, view_reuse=True).effective()


# ---------------------------------------------------------------- MONOMIAL_BIND parity (hard gate)


def _states(poly, schemas):
    task = v1.Task(poly, 8, 1, {})
    res = CS.run(task, F.Flat(), v1.CostRanker(), v1.Budget(*BUDGET), schemas, F.CFG, True, latent=F.LCFG,
                 policy=POLICY)
    return res['arena'], [h for h, n in res['tree'].items() if 'actions' in n]


def _parity_at(a, st, schemas):
    c0, w0 = RM.match_state(a, st, schemas, F.CFG)
    c1, w1 = EP.match_state_g(a, st, schemas, F.CFG, False)
    assert [csig(a, c) for c in c0] == [csig(a, c) for c in c1] and w0 == w1
    l0, lw0 = L.latent_candidates(a, st, schemas, F.CFG, F.LCFG)
    l1, lw1 = EP.latent_candidates_g(a, st, schemas, F.CFG, F.LCFG, False)
    assert [csig(a, c) for c in l0] == [csig(a, c) for c in l1] and lw0 == lw1
    assert [EP.action_identity(a, c) for c in c0+l0] == [EP.action_identity(a, c) for c in c1+l1]
    assert not any(EP.has_hb(c.beta) for c in c1+l1)
    return len(c0)+len(l0)


@pytest.mark.parametrize('name', sorted(F.FIXTURES))
def test_monomial_bind_reproduces_mainline_candidates_on_search_states(name, lib):
    schemas = with_fixture_schemas(lib)
    a, states = _states(F.FIXTURES[name], schemas)
    assert states
    assert sum(_parity_at(a, st, schemas) for st in states) > 0


@pytest.mark.parametrize('name', sorted(T6.BUILDERS))
def test_monomial_bind_reproduces_mainline_candidates_on_compound_states(name, lib):
    a = T6.arena()
    root, _ = T6.BUILDERS[name](a)
    _parity_at(a, root, with_fixture_schemas(lib))


def _adesc(a, x):
    if isinstance(x, (RM.Candidate, L.LatentCandidate)):
        return ('R', x.order_key, EP.action_identity(a, x))
    return ('P', v1.action_key(a, x))


def tree_sig(res) -> list:
    a, out = res['arena'], []
    for h, n in res['tree'].items():
        out.append((v1.digest(a, h).hex(), n['cost'], None if n['parent'] is None else v1.digest(a, n['parent']).hex(),
                    None if n['action'] is None else _adesc(a, n['action']), [_adesc(a, x) for x in n.get('chosen', [])],
                    [csig(a, c) for c in n.get('rewrites', [])], n.get('scores'), n.get('rewrite_scores'),
                    [v1.digest(a, x).hex() for x in n.get('successors', [])]))
    return out


def plain(summary) -> dict:
    return {k: v for k, v in summary.as_dict().items() if k not in TIMING and k not in T6_FIELDS}


@pytest.mark.parametrize('name', sorted(F.FIXTURES))
def test_generic_monomial_search_equals_baseline_search(name, lib):
    schemas = with_fixture_schemas(lib)

    def run(icfg):
        return CS.run(v1.Task(F.FIXTURES[name], 8, 1, {}), F.Flat(), v1.CostRanker(), v1.Budget(*BUDGET), schemas,
                      F.CFG, True, latent=F.LCFG, policy=POLICY, icfg=icfg)
    b, g = run(IntegrationConfig()), run(PROFILES['GENERIC_MONOMIAL'])
    assert tree_sig(b) == tree_sig(g)
    assert plain(b['summary']) == plain(g['summary'])
    sg = g['summary']
    assert (sg.hb_visible_candidates, sg.hb_latent_candidates, sg.hb_selected, sg.hb_applies, sg.gen_handle_choices,
            sg.hb_rejected_dependency) == (0, 0, 0, 0, 0, 0)


@pytest.mark.parametrize('name', ['Q1', 'Q2', 'TWO_SQ', 'CUBE'])
def test_identity_generic_monomial_equals_identity(name):
    schemas = [F.square(), F.distrib()]

    def run(icfg):
        res = CS.run(v1.Task(F.FIXTURES[name], 8, 1, {}), F.Flat(), v1.CostRanker(), v1.Budget(*BUDGET), schemas,
                     F.CFG, True, latent=F.LCFG, policy=POLICY, icfg=icfg)
        ca = res['arena']
        return ([v1.digest(ca, x).hex() for x in LI.lineage(res)],
                [{k: v for k, v in e.items() if k != 'cpu'} for e in res['gen_log'] if e['stage'] != 'canon'],
                plain(res['summary']))
    assert run(IntegrationConfig(identity_mode=True)) == run(PROFILES['IDENTITY_GENERIC_MONOMIAL'])


@pytest.mark.parametrize('name', sorted(T6.BUILDERS))
def test_generic_monomial_classes_equal_t4_classes(name, lib):
    def classes(fn_rw, fn_lx):
        a = T6.arena()
        root, _ = T6.BUILDERS[name](a)
        tf = ID.TaskFrameQ(v1.hpoly(a, root))
        cs = tf.state(a, root)
        log: list = []
        w = fn_rw(tf, cs, log)
        vk = {LI.dedupe_key(tf.carena, m) for c in w for m in c.members}
        x = fn_lx(tf, cs, vk, log)
        ca = tf.carena
        return ([(c.kind, c.size, c.key, csig(ca, c.rep)) for c in w+x],
                [{k: v for k, v in e.items() if k != 'cpu'} for e in log])
    schemas = with_fixture_schemas(lib)
    t4 = classes(lambda tf, cs, log: ID.q_rewrite(tf, cs, schemas, F.CFG, log, keep_members=True),
                 lambda tf, cs, vk, log: LI.q_latent(tf, cs, schemas, F.CFG, F.LCFG, vk, log))
    t6 = classes(lambda tf, cs, log: EPI.q_rewrite_g(tf, cs, schemas, F.CFG, False, log, keep_members=True),
                 lambda tf, cs, vk, log: EPI.q_latent_g(tf, cs, schemas, F.CFG, F.LCFG, vk, False, log))
    assert t4 == t6


def test_l0_product_patterns_hold_only_single_parameter_leaves(lib):
    """Scope fact for the nested-product lowering: in the 19 L0 families every M leaf directly inside a P pattern is a
    single parameter with exponent 1 (it lowers to the bound unit itself), so V1's nested-vs-flat product boundary
    never decides a handle match on L0. Multi-parameter leaves occur only as context children."""
    def walk(p, parent):
        if p[0] == 'M':
            if parent == 'P':
                assert len(p[1]) == 1 and p[1][0][1] == 1, p
            return
        for c, n in p[1]:
            walk(c, p[0])
    for s in lib:
        for c, n in s.source+s.target:
            walk(c, 'S')


def test_monomial_bind_keeps_the_mainline_pairwise_distinct_rule(sq, lib):
    """Reported conflict: the mainline binds distinct monomial holes to distinct monomials; parity keeps that rule on the
    monomial path, so 4a^2 offers no monomial Square(a, a) (the handle path is non-injective, below)."""
    a = T6.arena()
    root = v1.canonical_start(a, {F.mono(2): 4})
    for handles in (False, True):
        vis, lat, w, lw = gen(a, root, lib, handles)
        assert not [c for c in vis+lat if c.schema.schema_hash == sq.schema_hash]


# ---------------------------------------------------------------- REACHABLE_HANDLE_BIND: required fixtures


def _apply(a, root, c):
    return EP.apply_any_g(a, root, c, v1.hpoly(a, root))


def test_positive_square_H_K_from_existing_handles(sq, lib):
    a = T6.arena()
    root, hs = T6.positive(a)
    H, K = hs['H'], hs['K']
    reach0 = reach(a, root)
    assert {H, K} <= reach0, 'H and K exist before matching'
    n0 = len(a.handles)
    vis, w = EP.match_state_g(a, root, lib, F.CFG, True)
    assert len(a.handles) == n0, 'matching interns nothing'
    hits = square_hb(vis, sq)
    assert len(hits) == 1
    c = hits[0]
    assert {b.h for b in c.beta.values()} == {H, K}
    assert {EP.mode(b) for b in c.beta.values()} == {'REACHABLE_HANDLE_BIND'}
    assert c.ctx == root and c.R_inst == Counter(dict(v1.items(a.handles[root])))
    root2, why = _apply(a, root, c)
    assert why == 'ok'
    assert root2 == v1.mk_prod(a, Counter({v1.mk_sum(a, Counter({H: 1, K: 1})): 2})), 'exactly (H + K)^2'
    assert {H, K} <= reach(a, root2), 'bound handles reused, not rebuilt'
    assert EP.describe(a, c)['canonical_semantic_family_id'] == AR.canonical_family_id(sq)


@pytest.mark.parametrize('builder', ['negative_flat', 'negative_partial'])
def test_no_materialized_handle_no_square_H_K(builder, sq, lib):
    p = T6.arena()
    proot, phs = T6.positive(p)
    target_HK = square_hb(EP.match_state_g(p, proot, lib, F.CFG, True)[0], sq)[0].target_signature
    K_key = EP.hkey(p, phs['K'])
    a = T6.arena()
    root, hs = T6.BUILDERS[builder](a)
    reach0 = reach(a, root)
    vis, lat, w, lw = gen(a, root, lib, True)
    assert all(c.target_signature != target_HK for c in vis+lat), 'Square(H, K) is never offered from semantics alone'
    for c in vis+lat:
        for b in c.beta.values():
            if isinstance(b, EP.HB):
                assert b.h in reach0 and b.key != K_key, 'K = c+e is never invented'
    if builder == 'negative_flat':
        assert not any(EP.has_hb(c.beta) for c in vis+lat), 'a flat state has no compound unit to bind'


def test_no_product_regrouping(sq, lib):
    a = T6.arena()
    root, hs = T6.flat_product(a)
    reach0 = reach(a, root)
    vis, lat, w, lw = gen(a, root, lib, True)
    assert not square_hb(vis+lat, sq), 'A*B over h1*h2*h3*h4 is never A = h1*h3'
    for c in vis+lat:
        assert EP.dependency_check(a, root, c) is None
        assert all(b.h in reach0 for b in c.beta.values() if isinstance(b, EP.HB))
    b = T6.arena()
    root2, hs2 = T6.grouped_product(b)
    hits = square_hb(EP.match_state_g(b, root2, lib, F.CFG, True)[0], sq)
    assert len(hits) == 1 and {x.h for x in hits[0].beta.values()} == {hs2['P13'], hs2['P24']}
    assert _apply(b, root2, hits[0])[1] == 'ok'


def test_mixed_bind_compound_handle_and_monomial(sq, lib):
    def one(junk=0, pre=False):
        a = T6.arena(junk)
        if pre:
            T6.mono(a, {2: 1, 3: 1})
            T6.lin(a, (6, 7))
        root, hs = T6.mixed(a)
        hits = square_hb(EP.match_state_g(a, root, lib, F.CFG, True)[0], sq)
        assert len(hits) == 1
        c = hits[0]
        modes = {EP.mode(b): b for b in c.beta.values()}
        assert set(modes) == {'MONOMIAL_BIND', 'REACHABLE_HANDLE_BIND'}
        assert modes['REACHABLE_HANDLE_BIND'].h == hs['H'] and v1.pack(modes['MONOMIAL_BIND']) == a.handles[hs['m']].m
        root2, why = _apply(a, root, c)
        assert why == 'ok'
        assert root2 == v1.mk_prod(a, Counter({v1.mk_sum(a, Counter({hs['H']: 1, hs['m']: 1})): 2}))
        return c.order_key, c.target_signature, EP.action_identity(a, c), v1.digest(a, root2)
    base = one()
    assert one(junk=31) == base == one(pre=True), 'identity and successor independent of allocation'


def test_repeated_parameter_binds_semantically_equal_handles(sq, lib):
    """A binds H at A^2 and H2 (same polynomial, different handle) at AB: one semantic action, TWO concrete
    realizations (the target reuses H or H2), both offered, both exact, different successor DAGs."""
    a = T6.arena()
    root, hs = T6.repeated_semantic(a)
    hits = square_hb(EP.match_state_g(a, root, lib, F.CFG, True)[0], sq)
    assert len(hits) == 2
    assert all({b.key for b in c.beta.values()} == {EP.hkey(a, hs['H']), EP.hkey(a, hs['K'])} for c in hits)
    assert EP.hkey(a, hs['H']) == EP.hkey(a, hs['H2']) and hs['H'] != hs['H2']
    assert len({EP.action_identity(a, c) for c in hits}) == 1, 'one semantic action'
    assert len({EP.realization_key(a, c) for c in hits}) == 2, 'two concrete realizations'
    realized = {next(b.h for b in c.beta.values() if b.key == EP.hkey(a, hs['H'])) for c in hits}
    assert realized == {hs['H'], hs['H2']}
    succ = [_apply(a, root, c) for c in hits]
    assert all(w == 'ok' for r, w in succ) and succ[0][0] != succ[1][0], 'structurally different successors'
    b = T6.arena()
    root2, _ = T6.repeated_mismatch(b)
    assert not square_hb(EP.match_state_g(b, root2, lib, F.CFG, True)[0], sq), \
        'one ExprParam never binds two different polynomials'


def test_distinct_parameters_may_bind_the_same_value(sq, lib):
    a = T6.arena()
    root, hs = T6.noninjective(a)
    hits = square_hb(EP.match_state_g(a, root, lib, F.CFG, True)[0], sq)
    assert len(hits) == 1
    c = hits[0]
    assert [b.h for i, b in sorted(c.beta.items())] == [hs['H'], hs['H']], 'A := H and B := H'
    root2, why = _apply(a, root, c)
    assert why == 'ok' and root2 == v1.mk_prod(a, Counter({v1.mk_sum(a, Counter({hs['H']: 2})): 2})), '(H + H)^2'


def test_non_injective_value_with_two_realizing_handles(sq, lib):
    """A = B = one value realized by H or H2: every distinct concrete target is offered ((H+H)^2, (H+H2)^2,
    (H2+H2)^2), all in one semantic action class, all exact."""
    a = T6.arena()
    root, hs = T6.noninjective_two(a)
    hits = square_hb(EP.match_state_g(a, root, lib, F.CFG, True)[0], sq)
    targets = {tuple(sorted(b.h for b in c.beta.values())) for c in hits}
    assert targets == {(hs['H'], hs['H']), tuple(sorted((hs['H'], hs['H2']))), (hs['H2'], hs['H2'])}
    assert len({EP.action_identity(a, c) for c in hits}) == 1
    assert len({EP.realization_key(a, c) for c in hits}) == len(hits) == 3
    assert all(_apply(a, root, c)[1] == 'ok' for c in hits)
    adm, st = EP.admit(a, hits, 1)
    assert st['classes'] == 1 and st['classes_admitted'] == 1 and len(adm) == 3, 'a class is admitted whole'


def test_latent_d_times_square_H_K(sq, lib):
    a, b = T6.arena(), T6.arena()
    root, hs = T6.latent_square(a)
    rb, _ = T6.latent_square(b)
    reach0 = reach(a, root)
    vis, lat, w, lw = gen(a, root, lib, True)
    RM.match_state(b, rb, lib, F.CFG)
    L.latent_candidates(b, rb, lib, F.CFG, F.LCFG)
    assert len(a.handles) == len(b.handles), 'only the frozen quotient views intern anything'
    assert not square_hb(vis, sq), 'visible matching needs the multiplier stripped'
    hits = square_hb(lat, sq)
    assert len(hits) == 1
    c = hits[0]
    assert isinstance(c, L.LatentCandidate) and c.d == tuple(1 if j == 5 else 0 for j in range(8))
    assert {x.h for x in c.beta.values()} == {hs['H'], hs['K']} <= reach0, 'strip d, bind existing H, K'
    root2, why = _apply(a, root, c)
    assert why == 'ok' and v1.poly_equal(v1.hpoly(a, root2), v1.hpoly(a, root))
    tsum = v1.mk_prod(a, Counter({v1.mk_sum(a, Counter({hs['H']: 1, hs['K']: 1})): 2}))
    assert root2 == v1.mk_prod(a, Counter({T6.mono(a, {5: 1}): 1, tsum: 1})), 'd * (H + K)^2'
    d2 = dataclasses.replace(c, d=tuple(1 if j == 6 else 0 for j in range(8)))
    assert EP.describe(a, c)['d'] == list(c.d) and EP.action_identity(a, c) != EP.action_identity(a, d2), 'd identifies'


def test_dependency_checks_fail_closed(sq, lib):
    a = T6.arena()
    root, hs = T6.nested(a)
    tgt = v1.hpoly(a, root)
    hits = [c for c in square_hb(EP.match_state_g(a, root, lib, F.CFG, True)[0], sq) if c.ctx == hs['inner']]
    assert len(hits) == 1
    c = hits[0]
    hole = next(i for i, b in c.beta.items() if b.h == hs['H'])

    def forged(h, key=None):
        beta = dict(c.beta)
        beta[hole] = EP.HB(h, key or EP.hkey(a, h), v1.digest(a, h).hex())
        return dataclasses.replace(c, beta=beta)
    outside = T6.lin(a, (5, 6))
    cases = {'dep_capture': [forged(root), forged(hs['inner'])],                       # ancestor capture, self reference
             'dep_outside_region': [forged(hs['Y']), forged(hs['Z'])],                 # sibling; depends on a consumed child
             'dep_key_mismatch': [forged(hs['H'], key='0'*64)],
             'dep_unreachable': [forged(outside)]}
    for why, xs in cases.items():
        for x in xs:
            r, w = _apply(a, root, x)
            assert r is None and w == why
    assert EP.reuse_check(a, sq, c.beta, Counter({hs['Y']: 1})) == 'rebuild'
    root2, why = _apply(a, root, c)
    assert why == 'ok' and hs['inner'] not in reach(a, root2) and {hs['H'], hs['K']} <= reach(a, root2)


# ---------------------------------------------------------------- no synthesis, bounded work, identity


@pytest.mark.parametrize('name', sorted(T6.BUILDERS))
def test_candidate_generation_synthesizes_nothing(name, lib):
    a, b = T6.arena(), T6.arena()
    ra, _ = T6.BUILDERS[name](a)
    rb, _ = T6.BUILDERS[name](b)
    reach0 = reach(a, ra)
    n0 = len(a.handles)
    vis, w = EP.match_state_g(a, ra, lib, F.CFG, True)
    assert len(a.handles) == n0
    lat, lw = EP.latent_candidates_g(a, ra, lib, F.CFG, F.LCFG, True)
    _, wb = RM.match_state(b, rb, lib, F.CFG)
    _, lwb = L.latent_candidates(b, rb, lib, F.CFG, F.LCFG)
    assert len(a.handles) == len(b.handles), 'the generic binder interns nothing beyond the frozen views'
    assert reach(a, ra) == reach0
    for c in vis+lat:
        assert EP.dependency_check(a, ra, c) is None, 'every handle binding is a reachable unit inside its region'
    for k in ('cut_contexts', 'cut_unify_states', 'cut_attempts'):
        assert w[k] == wb[k] and lw[k] == lwb[k], f'{k}: no additional cut on the fixtures'


def test_family_identity_is_binding_mode_independent(sq, lib):
    fam = AR.canonical_family_id(sq)
    alias = AR.build_archive(lib, F.CFG).alias_map()[sq.schema_hash]
    assert alias['canonical_semantic_family_id'] == fam and alias['legacy_surface_id'] == sq.schema_hash
    rows = {}
    for label, build in (('Square(a,b)', lambda a: (v1.canonical_start(a, F.SQ), {})),
                         ('Square(ab,c)', lambda a: (v1.canonical_start(a, {F.mono(2, 2): 1, F.mono(1, 1, 1): 2,
                                                                                F.mono(0, 0, 2): 1}), {})),
                         ('Square(H,K)', T6.positive), ('Square(H,H)', T6.noninjective)):
        a = T6.arena()
        root, _ = build(a)
        cands = [c for c in EP.match_state_g(a, root, lib, F.CFG, True)[0] if c.schema.schema_hash == sq.schema_hash]
        assert cands, label
        d = EP.describe(a, cands[0])
        assert d['canonical_semantic_family_id'] == fam and d['legacy_surface_id'] == sq.schema_hash
        assert d['canonical_semantic_family_id_ac'] == REL.canonical_family_id_ac(sq)
        rows[label] = d
    assert {m['mode'] for m in rows['Square(a,b)']['bindings'].values()} == {'MONOMIAL_BIND'}
    assert {m['mode'] for m in rows['Square(H,H)']['bindings'].values()} == {'REACHABLE_HANDLE_BIND'}
    assert len({r['action_identity'] for r in rows.values()}) == 4
    a = T6.arena()
    m = v1.mk_mono(a, F.mono(1, 1))
    assert EP.param_key((1, 1, 0, 0, 0, 0, 0, 0)) == EP.hkey(a, m), 'one key space for both binding modes'


def canon_view(builder, perm, schemas) -> tuple:
    a = T6.arena()
    root, _ = builder(a, perm)
    tf = ID.TaskFrameQ(v1.hpoly(a, root))
    cs = tf.state(a, root)
    log: list = []
    wcl = EPI.q_rewrite_g(tf, cs, schemas, F.CFG, True, log, keep_members=True)
    lcl = EPI.q_latent_g(tf, cs, schemas, F.CFG, F.LCFG, {LI.dedupe_key(tf.carena, m) for c in wcl for m in c.members},
                         True, log, keep_members=True)
    ca = tf.carena
    tgt = v1.hpoly(ca, cs.croot)
    rows = []
    for c in wcl+lcl:
        r = c.rep
        if not EP.has_hb(r.beta):
            continue
        assert {(m.site_key, m.schema.schema_hash, m.target_signature) for m in c.members} == \
            EPI.orbit_keys_g(tf, cs, r), 'a class is exactly an orbit'
        r2, why = EP.apply_any_g(ca, cs.croot, r, tgt)
        rows.append((c.kind, c.size, r.site_key, r.schema.schema_hash, r.target_signature, EP.action_identity(ca, r),
                     EP.param_keys(r.schema, r.beta), tuple(EP.mode(b) for h, b in sorted(r.beta.items())),
                     getattr(r, 'd', None), why, None if r2 is None else v1.digest(ca, r2).hex()))
    return cs.digest.hex(), sorted(rows), [(e['stage'], e['class_hash'], e['retained_classes']) for e in log]


@pytest.mark.parametrize('name', ['positive', 'mixed', 'noninjective', 'grouped_product', 'latent_square',
                                  'repeated_semantic'])
def test_identity_mode_handle_classes_are_renaming_invariant(name, lib):
    base = canon_view(T6.BUILDERS[name], T6.IDENT8, lib)
    assert base[1], 'the fixture offers handle-bound classes'
    assert all(r[9] == 'ok' for r in base[1])
    for p in PERMS[1:]:
        assert canon_view(T6.BUILDERS[name], p, lib) == base


@pytest.mark.parametrize('name', ['positive', 'mixed', 'noninjective', 'grouped_product', 'latent_square'])
def test_handle_actions_are_allocation_independent_and_replay(name, lib):
    by_hash = {s.schema_hash: s for s in lib}

    def view(junk, pre):
        a = T6.arena(junk)
        if pre:
            T6.lin(a, (2, 4))
            T6.lin(a, (6, 7))
        root, _ = T6.BUILDERS[name](a)
        tgt = v1.hpoly(a, root)
        vis, lat, w, lw = gen(a, root, lib, True)
        out = []
        for c in vis+lat:
            if not EP.has_hb(c.beta):
                continue
            r2, why = EP.apply_any_g(a, root, c, tgt)
            out.append((c.order_key, c.target_signature, EP.action_identity(a, c), why,
                        None if r2 is None else v1.digest(a, r2).hex()))
            if why == 'ok':
                assert LI.replay(a, tgt, [root, r2], [[c]])
                body = LI.serialize(a, tgt, [root, r2], [[c]])
                assert any(isinstance(x, list) and x[:1] == ['H'] for i, x in body['steps'][0][4])
                assert LI.deserialize_and_replay(body, by_hash, tgt)
                perm = (3, 0, 7, 1, 6, 2, 5, 4)
                b = T6.arena()
                memo: dict = {}
                rb = relabel_into(a, root, perm, b, memo)
                cb = LI.map_action(c, a, perm, b, memo)
                r2b, whyb = EP.apply_any_g(b, rb, cb, v1.hpoly(b, rb))
                assert whyb == 'ok' and r2b == relabel_into(a, r2, perm, b, memo)
                dk = (cb.site_key, cb.schema.schema_hash, cb.target_signature)
                direct = [x for x in sum(gen(b, rb, lib, True)[:2], [])
                          if (x.site_key, x.schema.schema_hash, x.target_signature) == dk]
                assert len(direct) == 1 and EP.action_identity(b, direct[0]) == EP.action_identity(b, cb), \
                    'the renamed action is the action matched directly on the renamed state'
        return out
    base = view(0, False)
    assert base
    assert view(41, False) == base == view(17, True)


@pytest.mark.parametrize('name', ['Q2', 'TWO_SQ', 'CUBE'])
def test_identity_generic_handles_search_is_equivariant_and_replays(name):
    schemas = [F.square(), F.distrib()]
    base = None
    for p in PERMS[:3]:
        poly = F.relabel_poly(F.FIXTURES[name], p)
        res = CS.run(v1.Task(poly, 8, 1, {}), F.Flat(), v1.CostRanker(), v1.Budget(*BUDGET), schemas, F.CFG, True,
                     latent=F.LCFG, policy=POLICY, icfg=PROFILES['IDENTITY_GENERIC_HANDLES'])
        ca = res['arena']
        sig = ([v1.digest(ca, x).hex() for x in LI.lineage(res)],
               [(e['stage'], e.get('class_hash')) for e in res['gen_log'] if e['stage'] != 'canon'],
               v1.cost(ca, res['incumbent']))
        a = v1.Arena()
        a.calc_cfg = v1.CalcCfg()
        st, steps = LI.concretize(res, a)
        assert LI.replay(a, poly, st, steps)
        assert LI.deserialize_and_replay(LI.serialize(a, poly, st, steps), {x.schema_hash: x for x in schemas}, poly)
        s = res['summary']
        assert s.exact_verification_failures == 0 and s.hb_rejected_dependency == 0
        base = base or sig
        assert sig == base


def test_concrete_generic_handles_search_is_exact_and_fail_closed(lib):
    for name in ('Q2', 'TWO_SQ', 'CUBE'):
        res = CS.run(v1.Task(F.FIXTURES[name], 8, 1, {}), F.Flat(), v1.CostRanker(), v1.Budget(*BUDGET), lib, F.CFG,
                     True, latent=F.LCFG, policy=POLICY, icfg=PROFILES['GENERIC_HANDLES'])
        a, s = res['arena'], res['summary']
        assert v1.poly_equal(v1.hpoly(a, res['incumbent']), F.FIXTURES[name])
        assert s.exact_verification_failures == 0 and s.hb_rejected_dependency == 0


# ---------------------------------------------------------------- temporary scorer adapter


def test_scorer_adapter_passthrough_and_handle_rows(lib):
    from reweave.recipe import Recipe
    from reweave.rewrite_model import RewriteScorer, build_rewrite_model
    model = build_rewrite_model(Recipe(lr=5e-5, layernorm=True, residual=True), 0, len(lib))
    sc = RewriteScorer(model, {s.schema_hash: i for i, s in enumerate(lib)},
                       {s.schema_hash: [1.0, float(len(s.types)), 0.5] for s in lib})
    a = T6.arena()
    H, K = T6.lin(a, (0, 1)), T6.lin(a, (2, 4))
    kids = Counter({v1.mk_prod(a, Counter({H: 2})): 1, v1.mk_prod(a, Counter({H: 1, K: 1})): 2,
                    v1.mk_prod(a, Counter({K: 2})): 1, T6.mono(a, {5: 2}): 1, T6.mono(a, {5: 1, 6: 1}): 2,
                    T6.mono(a, {6: 2}): 1})
    root = v1.mk_sum(a, kids)
    info = v1.state_info(a, root)
    actions = v1.enumerate_actions(a, info, root, a.calc_cfg)
    cands = EP.match_state_g(a, root, lib, F.CFG, True)[0]
    mono = [c for c in cands if not EP.has_hb(c.beta)]
    hbc = [c for c in cands if EP.has_hb(c.beta)]
    assert mono and hbc
    base = sc.score_combined(a, info, root, actions, mono)
    assert EP.score_combined_g(sc, a, info, root, actions, mono) == base, 'exact passthrough without handle bindings'
    p, r = EP.score_combined_g(sc, a, info, root, actions, mono+hbc)
    assert p == base[0] and all(abs(x-y) < 1e-9 for x, y in zip(r[:len(mono)], base[1]))
    assert len(r) == len(mono)+len(hbc) and all(math.isfinite(x) for x in r)
    assert EP.score_combined_g(sc, a, info, root, actions, mono+hbc) == (p, r), 'deterministic'


# ---------------------------------------------------------------- T6A.1: review corrections


def test_flat_mul_product_leaf_inside_a_product_pattern():
    """A*B*(C+D) over ONE flat V1 product H*K*(c+d): the product leaf A*B binds the remaining factors H, K directly
    (associative MUL); the lowering is flat; the nested DAG (H*K)*(c+d) is not regrouped to match it."""
    sch = T6.flat_product_schema()
    a = T6.arena()
    root, hs = T6.flat_product_state(a)
    hits = [c for c in EP.match_state_g(a, root, [sch], F.CFG, True, hb_only=True)[0]]
    # H, K and c+d are all sums: the (C+D) factor may be any of the three existing factors, the product leaf A*B then
    # takes the other two — three exact, bounded rewrites of the one flat product
    assert len(hits) == 3
    assert {frozenset(b.h for b in c.beta.values() if isinstance(b, EP.HB)) for c in hits} == \
        {frozenset((hs['H'], hs['K'])), frozenset((hs['H'], hs['cd'])), frozenset((hs['K'], hs['cd']))}
    assert all(_apply(a, root, c)[1] == 'ok' for c in hits)
    c = next(c for c in hits if {b.h for b in c.beta.values() if isinstance(b, EP.HB)} == {hs['H'], hs['K']})
    root2, why = _apply(a, root, c)
    assert why == 'ok'
    kids = dict(v1.items(a.handles[root2]))
    c_, d_ = T6.mono(a, {2: 1}), T6.mono(a, {3: 1})
    assert v1.mk_prod(a, Counter({hs['H']: 1, hs['K']: 1, c_: 1})) in kids, 'flat lowering H*K*c'
    assert v1.mk_prod(a, Counter({hs['H']: 1, hs['K']: 1, d_: 1})) in kids
    b = T6.arena()
    H, K, cd = T6.lin(b, (0, 1)), T6.lin(b, (4, 5)), T6.lin(b, (2, 3))
    nested = v1.mk_sum(b, Counter({v1.mk_prod(b, Counter({v1.mk_prod(b, Counter({H: 1, K: 1})): 1, cd: 1})): 1,
                                   T6.mono(b, {6: 1}): 1}))
    assert not EP.match_state_g(b, nested, [sch], F.CFG, True, hb_only=True)[0], 'no regrouping of (H*K) into A, B'


def test_ac_family_id_is_association_independent_and_versioned(lib):
    M = lambda *ie: ('M', tuple(ie))
    from reweave import rewrite_schema as S
    src = ((M((0, 1), (1, 1), (2, 1)), 1), (M((0, 1), (1, 1), (3, 1)), 1))
    flat = S.canonicalize(('mon',)*4, src, ((('P', ((M((0, 1), (1, 1)), 1), (('S', ((M((2, 1)), 1), (M((3, 1)), 1))), 1))), 1),))
    nest = S.canonicalize(('mon',)*4, src, ((('P', ((('P', ((M((0, 1)), 1), (('S', ((M((2, 1)), 1), (M((3, 1)), 1))), 1))), 1),
                                                     (M((1, 1)), 1))), 1),))
    assert AR.canonical_family_id(flat) != AR.canonical_family_id(nest), 'v1 hashes the written structure'
    assert REL.canonical_family_id_ac(flat) == REL.canonical_family_id_ac(nest), 'v2: associativity is not identity'
    s3 = S.canonicalize(('mon',)*3, ((M((0, 1)), 1), (M((1, 1)), 1), (M((2, 1)), 1)),
                        ((('S', ((('S', ((M((0, 1)), 1), (M((1, 1)), 1))), 1), (M((2, 1)), 1))), 1),))
    s3f = S.canonicalize(('mon',)*3, ((M((0, 1)), 1), (M((1, 1)), 1), (M((2, 1)), 1)),
                         ((('S', ((M((0, 1)), 1), (M((1, 1)), 1), (M((2, 1)), 1))), 1),))
    assert REL.canonical_family_id_ac(s3) == REL.canonical_family_id_ac(s3f)
    # L0: v1 has 19 families; modulo AC two surfaces are ONE relation (identical sources, targets differing only in
    # association / sharing of a sub-sum). They stay two surfaces (two realizations) of one v2 family.
    groups: dict = {}
    for s in lib:
        groups.setdefault(REL.canonical_family_id_ac(s), []).append(s)
    assert len({AR.canonical_family_id(s) for s in lib}) == 19
    merged = [g for g in groups.values() if len(g) > 1]
    assert len(groups) == 18 and len(merged) == 1 and len(merged[0]) == 2
    s1, s2 = merged[0]
    assert REL.ac_forms(s1)[1] == REL.ac_forms(s2)[1] and AR.relation_form(s1) != AR.relation_form(s2)
    assert AR.relation_form(s1)[1][0] == AR.relation_form(s2)[1][0], \
        'the two surfaces share the source; they differ only in target structure'
    assert REL.canonical_family_id_ac(F.distrib()) != REL.canonical_family_id_ac(F.square())


def _legacy_at(a, st, schemas):
    c0, w0 = RM.match_state(a, st, schemas, F.CFG)
    l0, lw0 = L.latent_candidates(a, st, schemas, F.CFG, F.LCFG)
    l0 = L.dedupe_against_visible(a, l0, c0)
    return CP.select(c0+l0, F.CFG.rewrite_n_max, POLICY)


@pytest.mark.parametrize('name', ['Q2', 'TWO_SQ', 'CUBE', 'A0'])
def test_handle_lane_never_removes_a_legacy_action_before_scoring(name, lib):
    """GENERIC_HANDLES: at every expanded state the scored rewrite set is exactly the mainline admitted set plus the
    handle lane's admitted set (separate work allowance, separate admission)."""
    res = CS.run(v1.Task(F.FIXTURES[name], 8, 1, {}), F.Flat(), v1.CostRanker(), v1.Budget(*BUDGET), lib, F.CFG, True,
                 latent=F.LCFG, policy=POLICY, icfg=PROFILES['GENERIC_HANDLES'])
    a = res['arena']
    n = 0
    for h, node in res['tree'].items():
        if 'rewrites' not in node:
            continue
        legacy = [c for c in node['rewrites'] if not EP.has_hb(c.beta)]
        assert [csig(a, c) for c in legacy] == [csig(a, c) for c in _legacy_at(a, h, lib)]
        n += 1
    assert n


@pytest.mark.parametrize('name', ['negative_partial', 'noninjective_two', 'latent_view_reachable', 'positive'])
def test_semantic_class_admission_is_deterministic_and_whole_class(name, lib):
    a = T6.arena()
    root, _ = T6.BUILDERS[name](a)
    vis, lat, w, lw = gen(a, root, lib, True)
    hc = [c for c in vis+lat if EP.has_hb(c.beta)]
    assert hc
    for cap in (1, 3, 8, 64):
        adm, st = EP.admit(a, hc, cap)
        adm2, st2 = EP.admit(a, list(reversed(hc)), cap)
        assert [c.order_key for c in adm] == [c.order_key for c in adm2] and st == st2, 'input-order independent'
        by_class = Counter(EP.action_identity(a, c) for c in adm)
        full = Counter(EP.action_identity(a, c) for c in hc)
        assert all(by_class[k] == min(full[k], EP.REALIZATIONS_MAX) for k in by_class), 'classes admitted whole'
        assert len(adm) >= min(cap, len(hc)) or st['class_realizations_dropped']


def test_binding_equality_is_exact_not_digest(sq, lib):
    """Force a key-digest collision between J (= a+d) and H: the exact polynomial check refuses the equality."""
    a = T6.arena()
    root, hs = T6.repeated_mismatch(a)
    EP.hkey(a, hs['H'])
    EP._KEYS[a][hs['J']] = EP._KEYS[a][hs['H']]
    vis, w = EP.match_state_g(a, root, lib, F.CFG, True)
    assert not square_hb(vis, sq), 'a colliding digest never makes two different polynomials one value'
    assert w['key_collisions'] > 0


def json_dump(x) -> str:
    return json.dumps(x, sort_keys=True)


# ---------------------------------------------------------------- T6A.2: action-space interface freeze


def _fresh_scorer(lib):
    from reweave.recipe import Recipe
    from reweave.rewrite_model import RewriteScorer, build_rewrite_model
    model = build_rewrite_model(Recipe(lr=5e-5, layernorm=True, residual=True), 0, len(lib))
    return RewriteScorer(model, {s.schema_hash: i for i, s in enumerate(lib)},
                         {s.schema_hash: [1.0, float(len(s.types)), 0.5] for s in lib})


def _view_action(a, lib):
    """A structurally valid T5 ViewAction: substitute H2 -> H (same exact value), then a rewrite in V."""
    from reweave.canon import views as VW
    root, hs = T6.repeated_semantic(a)
    V = v1.substitute(a, root, hs['H2'], hs['H'])
    cand = square_hb(EP.match_state_g(a, V, lib, F.CFG, True)[0], T6.l0_square())[0]
    succ, why = EP.apply_any_g(a, V, cand, v1.hpoly(a, V))
    assert why == 'ok'
    return root, VW.ViewAction(hs['H2'], hs['H'], 'k', 'D0', 'D1', 1, cand, succ, v1.digest(a, succ), 'CURRENT', 'fixture')


def test_every_selectable_action_kind_has_a_valid_record(lib):
    a = T6.arena()
    root = _rich(a)
    info = v1.state_info(a, root)
    prims = v1.enumerate_actions(a, info, root, a.calc_cfg)
    vis = EP.match_state_g(a, root, lib, F.CFG, True)[0]
    assert any(EP.has_hb(c.beta) for c in vis) and not all(EP.has_hb(c.beta) for c in vis)
    c = T6.arena()
    lroot, _ = T6.latent_square(c)
    lat = EP.latent_candidates_g(c, lroot, lib, F.CFG, F.LCFG, True)[0]
    b = T6.arena()
    vroot, va = _view_action(b, lib)
    recs = ARec.records(a, prims, vis)+ARec.records(c, [], lat)+[ARec.view_record(b, va)]
    assert {r.kind for r in recs} == set(ARec.KINDS)
    assert {r.provenance['lane'] for r in recs} == set(ARec.LANES)
    for r in recs:
        assert ARec.validate(r) == [], (r.kind, ARec.validate(r))
        assert set(r.to_json()) == {'v', 'kind', 'semantic_key', 'semantic', 'concrete_key', 'concrete',
                                    'provenance', 'features'}
        for part in ('semantic', 'concrete'):
            tokens = set(json.dumps(r.to_json()[part]).split('"'))
            assert not {'lane', 'class_size', 'truncation', 'surfaces', 'surface'} & tokens


def test_records_are_deterministic_allocation_free_and_rebindable(lib):
    by_hash = {s.schema_hash: s for s in lib}

    def dumps(junk):
        a = T6.arena(junk)
        root, _ = T6.noninjective_two(a)
        info = v1.state_info(a, root)
        prims = v1.enumerate_actions(a, info, root, a.calc_cfg)
        recs = ARec.records(a, prims, EP.match_state_g(a, root, lib, F.CFG, True)[0])
        for r in recs:
            live = ARec.rebind(r, a, root, by_hash)
            if r.kind == 'PRIMITIVE':
                assert live == r.live
            else:
                assert csig(a, live) == csig(a, r.live), 'the record alone reproduces the live action'
            assert ARec.ActionRecord.from_json(json.loads(r.dumps())).dumps() == r.dumps()
        return sorted(r.dumps() for r in recs)
    base = dumps(0)
    assert base == dumps(0) == dumps(29), 'deterministic and independent of allocation'


def test_semantic_and_concrete_keys_are_two_levels(sq, lib):
    a = T6.arena()
    root, hs = T6.repeated_semantic(a)
    recs = ARec.records(a, [], square_hb(EP.match_state_g(a, root, lib, F.CFG, True)[0], sq))
    assert len({r.semantic_key for r in recs}) == 1 and len({r.concrete_key for r in recs}) == 2
    b = T6.arena()
    root2, _ = T6.noninjective_two(b)
    recs = ARec.records(b, [], square_hb(EP.match_state_g(b, root2, lib, F.CFG, True)[0], sq))
    assert len({r.semantic_key for r in recs}) == 1 and len({r.concrete_key for r in recs}) == 3
    r = recs[0]
    assert [p['role'] for p in r.semantic['parameters']] == [0, 1]
    assert r.semantic['family']['id'] == REL.canonical_family_id_ac(sq)
    assert json.dumps(r.semantic['family']['relation']) == json.dumps(json.loads(REL.family_payload(sq)))


def test_legacy_adapter_reproduces_the_fixed_scorer_exactly(lib):
    sc = _fresh_scorer(lib)
    a = T6.arena()
    H, K = T6.lin(a, (0, 1)), T6.lin(a, (2, 4))
    kids = Counter({v1.mk_prod(a, Counter({H: 2})): 1, v1.mk_prod(a, Counter({H: 1, K: 1})): 2,
                    v1.mk_prod(a, Counter({K: 2})): 1, T6.mono(a, {5: 2}): 1, T6.mono(a, {5: 1, 6: 1}): 2,
                    T6.mono(a, {6: 2}): 1})
    root = v1.mk_sum(a, kids)
    info = v1.state_info(a, root)
    actions = v1.enumerate_actions(a, info, root, a.calc_cfg)
    cands = EP.match_state_g(a, root, lib, F.CFG, True)[0]
    mono = [c for c in cands if not EP.has_hb(c.beta)]
    for cl in (mono, cands):
        p, q = EP.score_combined_g(sc, a, info, root, actions, cl)
        s = ARec.score_records(sc, a, info, root, ARec.records(a, actions, cl))
        assert s == p+q, 'exactly the frozen scorer outputs, in record order'
    p, q = sc.score_combined(a, info, root, actions, mono)
    assert ARec.score_records(sc, a, info, root, ARec.records(a, actions, mono)) == p+q


def _small(cfg_attempts=60, unify=30, lat_attempts=60):
    return (dataclasses.replace(F.CFG, match_attempts_max=cfg_attempts, unify_states_max=unify),
            dataclasses.replace(F.LCFG, attempts_max=lat_attempts))


@pytest.mark.parametrize('name', ['TWO_SQ', 'CUBE', 'Q2'])
def test_candidate_set_records_identify_every_truncation(name, lib):
    """Tiny frozen-shape caps force truncation: every expanded state has a CandidateSetRecord whose legacy-lane causes
    equal the mainline generators' own work records recomputed at that state; `no_action_exists` only when complete."""
    cfg, lcfg = _small()
    res = CS.run(v1.Task(F.FIXTURES[name], 8, 1, {}), F.Flat(), v1.CostRanker(), v1.Budget(*BUDGET), lib, cfg, True,
                 latent=lcfg, policy=POLICY, icfg=PROFILES['GENERIC_HANDLES'])
    a = res['arena']
    seen_incomplete = 0
    for h, node in res['tree'].items():
        if 'rewrites' not in node:
            continue
        csr = node['candidate_set']
        c0, w0 = RM.match_state(a, h, lib, cfg)
        l0, lw0 = L.latent_candidates(a, h, lib, cfg, lcfg)
        ref = CSR.legacy_lane_concrete(a, h, cfg, lcfg, w0, lw0, len(L.dedupe_against_visible(a, l0, c0)),
                                       csr['lanes']['legacy_rewrite']['admitted'])
        assert csr['lanes']['legacy_rewrite'] == ref
        for ln in csr['lanes'].values():
            assert ln['no_action_exists'] == (ln['enabled'] and ln['generation_complete'] and ln['considered'] == 0)
        assert csr['selectable']['PRIMITIVE'] == len(node['actions'])
        assert csr['selectable']['VISIBLE_REWRITE']+csr['selectable']['LATENT_REWRITE'] == len(node['rewrites'])
        seen_incomplete += not csr['generation_complete']
    assert seen_incomplete, 'the tiny caps must truncate somewhere'
    s = res['summary']
    assert s.csr_generation_incomplete_states == seen_incomplete and s.csr_states >= seen_incomplete


def _rich(a):
    """H^2 + 2HK + K^2 over existing handles next to a monomial square: several families have handle work here."""
    H, K = T6.lin(a, (0, 1)), T6.lin(a, (2, 4))
    return v1.mk_sum(a, Counter({v1.mk_prod(a, Counter({H: 2})): 1, v1.mk_prod(a, Counter({H: 1, K: 1})): 2,
                                 v1.mk_prod(a, Counter({K: 2})): 1, T6.mono(a, {5: 2}): 1, T6.mono(a, {5: 1, 6: 1}): 2,
                                 T6.mono(a, {6: 2}): 1}))


def test_handle_lane_is_invariant_to_schema_order_and_fair(lib):
    """Half of the lane's full work as the global allowance, a small quantum: permuting the schema list changes
    neither the admitted handle set nor the (deterministic) schedule and incompleteness record; every family got a
    turn; the families still incomplete when the allowance ran out received equal quanta (no family monopolizes the
    lane by schema order)."""
    q = 8
    a = T6.arena()
    full, fw = EP.handle_visible(a, _rich(a), lib, F.CFG, quantum=q)
    W = fw['attempts']
    assert fw['work_cap'] == 0 and fw['families_complete'] == fw['families'] and W >= 4 * q
    cfg = dataclasses.replace(F.CFG, match_attempts_max=W // 2)

    def run(order):
        b = T6.arena()
        adm, st = EP.handle_lane(b, _rich(b), order, cfg, F.LCFG, quantum=q)
        return ([(c.order_key, c.surfaces) for c in adm], st['visible'], st['latent'], st['admission'],
                CSR.handle_lane_record(st, len(adm)))
    base = run(list(lib))
    for seed in (1, 2, 3):
        perm = list(lib)
        random.Random(seed).shuffle(perm)
        assert run(perm) == base
    w, rec = base[1], base[4]
    assert w['work_cap'] == 1 and w['attempts'] == W // 2 + 1, 'the allowance binds; the overrunning tick is counted'
    assert rec['truncation'].get('work_cap') == 1 and not rec['generation_complete'] and not rec['no_action_exists']
    fams = w['family_records']
    assert len(fams) == 18 and all(f['turns'] >= 1 for f in fams), 'every family got a turn'
    inc = [f for f in fams if not f['complete']]
    assert inc
    assert max(f['turns'] for f in inc) - min(f['turns'] for f in inc) <= 1, 'one turn per live family per round'
    for f in inc:     # every turn but the last one ran its full quantum (plus the recorded non-pausable overrun)
        assert (f['turns']-1)*q <= f['work'] <= f['turns']*(q+f['overrun'])
    slack = max(f['turns']*f['overrun'] for f in inc)
    assert max(f['work'] for f in inc) - min(f['work'] for f in inc) <= q + slack, 'equal quanta'
    assert sum(f['work'] for f in fams) == w['attempts']


def test_latent_binding_must_lie_inside_the_allocated_originals(lib):
    """P = H*K lies below o2 only. Whenever the view allocation of a match that binds P takes only the flat f*H*K
    originals, the candidate is refused at generation by the apply-time predicate; nothing admitted can fail."""
    a = T6.arena()
    root, hs = T6.latent_unconsumed_original(a)
    sch = T6.double_schema()
    lat, lw = EP.handle_latent(a, root, [sch]+lib, F.CFG, F.LCFG)
    assert lat and all(EP.dependency_check(a, root, c) is None for c in lat)
    binds_P = [c for c in lat if any(isinstance(b, EP.HB) and b.h == hs['P'] for b in c.beta.values())]
    assert all(hs['o2'] in c.R_inst for c in binds_P), 'P may only be bound when o2 is consumed'
    f = tuple(1 if j == 5 else 0 for j in range(8))
    bad = EP.lcand_g(a, root, f, sch, {0: EP.hb(a, hs['P'])}, Counter({hs['fHK']: 2}))
    assert EP.dependency_check(a, root, bad) == 'dep_outside_region'
    view = next(v for v in L.views(a, root, F.LCFG) if tuple(v[0]) == f)
    assert view[1][hs['P']] == 3 and {o for o, n in view[2][hs['P']]} == {hs['fHK'], hs['o2']}
    if view[2][hs['P']][0][0] == hs['fHK']:
        # allocation takes the two flat originals for A + A := P: that match must be refused, never emitted
        assert lw.get('inadmissible_matches', 0)+lw.get('rejected_inadmissible', 0) >= 1
        assert not [c for c in binds_P if c.schema.schema_hash == sch.schema_hash]
    else:
        assert [c for c in binds_P if c.schema.schema_hash == sch.schema_hash], 'o2 allocated first: admissible'
    # the sequential generator applies the same predicate
    seq = EP.latent_candidates_g(a, root, [sch]+lib, F.CFG, F.LCFG, True, hb_only=True)[0]
    assert all(EP.dependency_check(a, root, c) is None for c in seq)
    for c in lat:
        assert EP.apply_any_g(a, root, c, v1.hpoly(a, root))[1] == 'ok'


def test_identical_concrete_successors_merge_across_surfaces(sq, lib):
    sx = T6.square_expr_surface()
    assert sx.schema_hash != sq.schema_hash and REL.same_family(sx, sq)
    a = T6.arena()
    root, hs = T6.positive(a)
    cands, w = EP.handle_visible(a, root, [sq, sx], F.CFG)
    assert len(cands) == 1 and set(cands[0].surfaces) == {sq.schema_hash, sx.schema_hash}
    assert w['merged_cross_surface'] >= 1
    b = T6.arena()
    root2, _ = T6.repeated_semantic(b)
    two, _ = EP.handle_visible(b, root2, [sq, sx], F.CFG)
    assert len(two) == 2 and all(len(c.surfaces) == 2 for c in two), 'distinct realizations stay distinct'
    assert len({EP.realization_key(b, c) for c in two}) == 2 and len({EP.action_identity(b, c) for c in two}) == 1
    rec = ARec.rewrite_record(a, cands[0], 'handle')
    assert rec.provenance['surfaces'] == sorted((sq.schema_hash, sx.schema_hash))


def test_hashes_index_exact_payloads_decide(lib, monkeypatch):
    """Force every handle-bound target signature and every v2 family id to collide: the generated concrete set, the
    semantic classes and the alias map are unchanged — digests index, exact payloads decide."""
    def gen_all():
        a = T6.arena()
        root = _rich(a)
        vis, w = EP.handle_visible(a, root, lib, F.CFG)
        lat, lw = EP.handle_latent(a, root, lib, F.CFG, F.LCFG)
        adm, st = EP.admit(a, vis+lat, 10**6)
        return (sorted(repr(EP.concrete_key(c)) for c in vis+lat), sorted(sorted(c.surfaces) for c in vis+lat),
                st['classes'], len(adm))
    base = gen_all()
    assert base[2] >= 2
    monkeypatch.setattr(EP, 'target_signature_g', lambda sch, beta: 'collide')
    monkeypatch.setattr(REL, 'canonical_family_id_ac', lambda s: 'f'*64)
    assert gen_all() == base, 'colliding digests never merge different targets, families or classes'
    am = REL.ac_alias_map(lib)
    assert len(am['families']) == 18 and len(am['surfaces']) == 19


def test_ac_alias_map_is_reversible(lib):
    am = REL.ac_alias_map(lib)
    assert len(am['surfaces']) == 19 and len(am['families']) == 18
    for fk, fam in am['families'].items():
        for sid in fam['surfaces']:
            assert am['surfaces'][sid]['family_key'] == fk
    pair = [f for f in am['families'].values() if len(f['surfaces']) == 2]
    assert len(pair) == 1 and set(pair[0]['surfaces']) == {
        '71a318fc91f1d184125b103b62046debaad201e70450854c9d8fccb23a84f903',
        'af576ff519480dca4fcd2b3371ce26aa2399a1d1c3a049bc41011ed8a9b908b4'}
    archive = AR.build_archive(lib, F.CFG)
    assert all(REL.lookup_surface_exact(archive, s) is not None for s in lib)
    forged = dataclasses.replace(lib[0], body=dict(lib[0].body, target=[]))
    assert REL.lookup_surface_exact(archive, forged) is None, 'a legacy id is an index, the body decides'


def test_record_keys_are_sha256_json_and_surface_memos_check_the_body(lib):
    """The fast key path is `sha256_json` byte for byte; per-surface memos are indexed by the legacy id but a hit
    requires the identical canonical body."""
    from reweave.hashing import sha256_json
    a = T6.arena()
    root = _rich(a)
    info = v1.state_info(a, root)
    c = T6.arena()
    lroot, _ = T6.latent_square(c)
    prims = v1.enumerate_actions(a, info, root, a.calc_cfg)
    recs = (ARec.records(a, prims, EP.match_state_g(a, root, lib, F.CFG, True)[0])
            + ARec.records(c, [], EP.latent_candidates_g(c, lroot, lib, F.CFG, F.LCFG, True)[0]))
    assert {r.kind for r in recs} == {'PRIMITIVE', 'VISIBLE_REWRITE', 'LATENT_REWRITE'}
    for r in recs:
        assert r.semantic_key == sha256_json([ARec.RECORD_VERSION, 'semantic', r.kind, r.semantic])
        assert r.concrete_key == sha256_json([ARec.RECORD_VERSION, 'concrete', r.kind, r.concrete])
    s = lib[0]
    cache: dict = {}
    calls: list = []

    def fn(x):
        calls.append(x)
        return len(calls)
    assert REL.surface_memo(cache, s, fn) == 1 and REL.surface_memo(cache, s, fn) == 1
    forged = dataclasses.replace(s, body=dict(s.body, target=[]))
    assert REL.surface_memo(cache, forged, fn) == 2, 'same legacy id, different body: recomputed, never reused'
    assert REL.surface_memo(cache, s, fn) == 1
    assert all(ARec.family_id_v1(x) == AR.canonical_family_id(x) for x in lib)
