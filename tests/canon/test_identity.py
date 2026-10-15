"""Symmetry-safe identity: renaming equivariance incl. latent actions, classes = exact orbits (no split, no collapse),
multiplier-aware latent identity survives replay / serialization, deterministic class caps."""

import ast
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

from reweave import candidate_policy as CP
from reweave import v1
from reweave.canon import identity as ID
from reweave.canon import latent_identity as LI
from reweave.canon.config import IntegrationConfig
from reweave.canon.identity_search import run_identity
from reweave.rewrite_latent import LatentCandidate

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixtures as F  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
TIMING = ('cpu_time', 'wall_time', 'peak_rss', 'norm_cpu', 'identity_cpu')
PORTED = {
    ('bc5600a', 'reweave/c7_1_equivariant.py'): ('CanonState', '_hash', '_admit', '_cand', '_dkey'),
    ('c7db59b', 'reweave/c7_1b_quotient.py'): ('_cj', '_fsig', 'fast_target_signature', 'fast_binding_digest',
                                              'fast_order_key', 'TaskFrameQ', 'verify_fast_hashes', 'ActionClass',
                                              '_akey', 'q_primitive', 'q_rewrite', 'generate'),
}
SCHEMAS = [F.square(), F.distrib()]


def defs(src):
    return {n.name: ast.dump(n) for n in ast.parse(src).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def ident_run(poly, icfg=None, record=True):
    task = v1.Task(poly, 8, 1, {})
    return run_identity(task, F.Flat(), v1.CostRanker(), v1.Budget(6, 24, 6, 60), SCHEMAS, F.CFG, record, latent=F.LCFG,
                        policy=CP.CandidatePolicy(CP.ORIGIN_SCHEMA_DIVERSITY),
                        icfg=icfg or IntegrationConfig(identity_mode=True))


def signature(res):
    ca = res['arena']
    lin = [v1.digest(ca, x).hex() for x in LI.lineage(res)]
    log = [(e['stage'], e.get('canon_digest'), e.get('retained_canon_hash'), e.get('class_hash'), e.get('retained_classes'))
           for e in res['gen_log'] if e['stage'] != 'canon']
    summ = {k: v for k, v in res['summary'].as_dict().items() if k not in TIMING}
    return lin, log, summ, v1.digest(ca, res['incumbent']).hex()


def renamed(poly, rng, n=6):
    p = list(range(8))
    img = rng.sample(range(n), n)
    p[:n] = img
    return F.relabel_poly(poly, tuple(p))


@pytest.mark.parametrize('name', sorted(F.FIXTURES))
def test_identity_mode_is_renaming_equivariant_including_latent(name):
    rng = random.Random(name)
    base = signature(ident_run(F.FIXTURES[name]))
    for _ in range(3):
        assert signature(ident_run(renamed(F.FIXTURES[name], rng))) == base


@pytest.mark.parametrize('name', sorted(F.FIXTURES))
def test_classes_are_exact_orbits(name):
    tf0 = ID.TaskFrameQ(F.FIXTURES[name])
    tfc = ID.TaskFrameQ(canonical_of=tf0)
    ca = tfc.carena
    start = v1.canonical_start(ca, {v1.pack(vec): c for vec, c in tf0.T.items()})
    cs = tfc.state(ca, start)
    wcl = ID.q_rewrite(tfc, cs, SCHEMAS, F.CFG, None, keep_members=True)
    vkeys = {LI.dedupe_key(ca, m) for c in wcl for m in c.members}
    lcl = LI.q_latent(tfc, cs, SCHEMAS, F.CFG, F.LCFG, vkeys, None, keep_members=True)
    seen = set()
    for c in wcl+lcl:
        members = {(m.site_key, m.schema.schema_hash, m.target_signature) for m in c.members}
        assert members == LI.orbit_keys(tfc, cs, c.rep), 'a class is exactly one orbit (no split, no collapse)'
        assert not (members & seen), 'classes are disjoint'
        seen |= members
    for c in lcl:
        assert all(isinstance(m, LatentCandidate) and any(m.d) for m in c.members)
        assert all(m.site_key[0] == 'lx' and m.site_key[3] == tuple(m.d) for m in c.members), 'd is in the identity'
    assert ID.verify_fast_hashes(tfc) == dict(ID.verify_fast_hashes(tfc), tsig_mismatch=0, bdig_mismatch=0)


def test_latent_classes_exist_where_the_quotient_applies():
    for name in ('Q1', 'Q2', 'A0', 'CUBE'):
        res = ident_run(F.FIXTURES[name])
        assert res['summary'].lx_classes > 0, name


@pytest.mark.parametrize('name', sorted(F.FIXTURES))
@pytest.mark.parametrize('norm', [False, True])
def test_concretized_lineage_replays_exactly_and_survives_serialization(name, norm):
    poly = F.FIXTURES[name]
    res = ident_run(poly, IntegrationConfig(identity_mode=True, normalization=norm))
    assert res['summary'].exact_verification_failures == 0 and res['summary'].norm_failures == 0
    a = v1.Arena()
    a.calc_cfg = v1.CalcCfg()
    states, steps = LI.concretize(res, a)
    assert v1.poly_equal(v1.hpoly(a, states[0]), poly)
    assert LI.replay(a, poly, states, steps)
    body = LI.serialize(a, poly, states, steps)
    assert LI.deserialize_and_replay(body, {s.schema_hash: s for s in SCHEMAS}, poly)


def test_latent_step_codec_keeps_the_multiplier():
    a, root = F.arena_of(F.Q1)
    from reweave.rewrite_latent import latent_candidates
    lat, _w = latent_candidates(a, root, SCHEMAS, F.CFG, F.LCFG)
    assert lat
    nodes, index = v1.serialize(a, [root], with_digests=True)
    b = v1.Arena()
    b.calc_cfg = v1.CalcCfg()
    ids = v1.deserialize(b, nodes)
    for c in lat:
        rec = LI.encode_step(a, c, index)
        assert rec[0] == 'LX' and tuple(rec[5]) == tuple(c.d)
        c2 = LI.decode_step(rec, ids, {s.schema_hash: s for s in SCHEMAS}, b)
        assert (c2.site_key, c2.order_key, c2.d) == (c.site_key, c.order_key, c.d)


def test_class_caps_and_generation_are_deterministic():
    for name in ('TWO_SQ', 'CUBE', 'Q2'):
        assert signature(ident_run(F.FIXTURES[name])) == signature(ident_run(F.FIXTURES[name]))


def test_identity_off_is_the_concrete_loop():
    from reweave.canon.search import run
    task = v1.Task(F.Q1, 8, 1, {})
    res = run(task, F.Flat(), v1.CostRanker(), v1.Budget(6, 24, 6, 60), SCHEMAS, F.CFG, False, latent=F.LCFG,
              policy=CP.CandidatePolicy(CP.ORIGIN_SCHEMA_DIVERSITY), icfg=IntegrationConfig())
    assert 'tf0' not in res
