"""Deterministic candidate-admission policies.

Fixtures are tests only; they never enter a mined library. Every assertion is about `reweave/candidate_policy.py`
and its wiring at the two admission stages S-CAP and S-RET."""

import random
import re
from collections import Counter
from pathlib import Path

from reweave import candidate_policy as CP
from reweave import v1
from reweave.rewrite_config import RewriteConfig
from reweave.rewrite_latent import LatentCandidate
from reweave.rewrite_search import RewriteSearchSummary

REPO = Path(__file__).resolve().parent.parent.parent
CFG = RewriteConfig()
FROZEN_SUMMARY_FIELDS = frozenset(RewriteSearchSummary().as_dict())
HASH, SCH, ORG = CP.CandidatePolicy(CP.HASH), CP.CandidatePolicy(CP.SCHEMA_DIVERSITY), \
    CP.CandidatePolicy(CP.ORIGIN_SCHEMA_DIVERSITY)


class _Schema:
    def __init__(self, h):
        self.schema_hash = h


class _Vis:
    """Stands in for `rewrite_match.Candidate`: the policy only reads `schema`, `order_key`, `stable_hash`."""
    def __init__(self, sh, key):
        self.schema, self.order_key = _Schema(sh), (key,)

    @property
    def stable_hash(self):
        return self.order_key[0]


def _lat(sh, key):
    return LatentCandidate(0, (1,), _Schema(sh), {}, Counter(), (), '', '', (key,))


def corpus():
    """40 visible from 2 schemas and 40 latent from 2 schemas, with interleaved hash keys."""
    out = []
    for i in range(40):
        out.append(_Vis('AA' if i % 2 else 'BB', f'{i*7 % 97:04d}v{i}'))
    for i in range(40):
        out.append(_lat('AA' if i % 2 else 'CC', f'{i*11 % 97:04d}l{i}'))
    return out


# ---------------------------------------------------------------- 1: control parity


def test_cm_hash_is_exactly_the_frozen_stable_hash_truncation():
    c = corpus()
    for cap in (1, 7, 16, 40, 256):
        assert CP.select(c, cap, HASH) == sorted(c, key=lambda x: x.order_key)[:cap]


# ---------------------------------------------------------------- 2 / 3: round-robin


def test_schema_diversity_round_robin_is_deterministic_and_balanced():
    c = corpus()
    got = CP.select(c, 12, SCH)
    assert len(got) == 12
    per = Counter(x.schema.schema_hash for x in got)
    assert set(per) == {'AA', 'BB', 'CC'}
    assert max(per.values())-min(per.values()) <= 1, per
    assert CP.select(c, 12, SCH) == got


def test_origin_schema_diversity_separates_visible_and_latent_of_the_same_schema():
    c = corpus()
    got = CP.select(c, 12, ORG)
    keys = Counter(CP.bucket_key(x, ORG) for x in got)
    assert set(keys) == {('visible', 'AA'), ('visible', 'BB'), ('latent', 'AA'), ('latent', 'CC')}
    assert max(keys.values())-min(keys.values()) <= 1, keys
    # the same schema appearing with two origins is two buckets here, one bucket under the ablation
    assert len(set(Counter(CP.bucket_key(x, SCH) for x in c))) == 3
    assert len(set(Counter(CP.bucket_key(x, ORG) for x in c))) == 4


# ---------------------------------------------------------------- 4 / 5: ordering and permutation invariance


def test_stable_hash_ordering_is_preserved_inside_every_bucket():
    c = corpus()
    for pol in (SCH, ORG):
        got = CP.select(c, 20, pol)
        per = {}
        for x in got:
            per.setdefault(CP.bucket_key(x, pol), []).append(x.order_key)
        for k, v in per.items():
            assert v == sorted(v), (k, v)
            full = sorted([y.order_key for y in c if CP.bucket_key(y, pol) == k])
            assert v == full[:len(v)], 'a bucket must be consumed in frozen stable-hash order, prefix first'


def test_raw_candidate_permutation_does_not_change_the_retained_set_or_order():
    c = corpus()
    rng = random.Random(0)
    for pol in (HASH, SCH, ORG):
        base = [x.order_key for x in CP.select(c, 17, pol)]
        for _ in range(5):
            p = c[:]
            rng.shuffle(p)
            assert [x.order_key for x in CP.select(p, 17, pol)] == base


# ---------------------------------------------------------------- 6 / 7: budgets and generation


def test_every_policy_admits_the_same_number_of_candidates():
    c = corpus()
    for cap in (1, 5, 16, 33, 80, 256):
        sizes = {pol.name: len(CP.select(c, cap, pol)) for pol in (HASH, SCH, ORG)}
        assert len(set(sizes.values())) == 1, sizes
        assert sizes[CP.HASH] == min(cap, len(c))


def test_policies_never_add_or_invent_candidates():
    c = corpus()
    ident = {id(x) for x in c}
    for pol in (HASH, SCH, ORG):
        got = CP.select(c, 30, pol)
        assert all(id(x) in ident for x in got)
        assert len({id(x) for x in got}) == len(got), 'no duplicates'


# ---------------------------------------------------------------- 8 / 9: no score, no family logic


def test_policy_never_touches_score_utility_or_outcome():
    class Trap:
        def __init__(self, sh, key):
            self.schema, self.order_key = _Schema(sh), (key,)

        @property
        def stable_hash(self):
            return self.order_key[0]

        def __getattr__(self, name):
            raise AssertionError(f'candidate policy touched forbidden attribute {name!r}')
    items = [Trap('AA' if i % 2 else 'BB', f'{i:03d}') for i in range(20)]
    for pol in (HASH, SCH):
        assert len(CP.select(items, 8, pol)) == 8


def test_no_family_or_task_specific_logic_in_policy_module():
    src = (REPO/'reweave'/'candidate_policy.py').read_text()
    body = re.sub(r'"""(?:.|\n)*?"""', '', src)
    body = re.sub(r'#.*', '', body)
    for pat in (r'\bF1\b', r'\bF2\b', r'\bF3\b', r'\bF4\b', r'generator', r'task_id', r'\bfamily\b',
                r'\b[0-9a-f]{12,}\b'):
        assert not re.search(pat, body, re.IGNORECASE), f'{pat} appears in candidate-policy logic'


# ---------------------------------------------------------------- 10 / 11: bucket classification


def test_visible_latent_origin_classification_is_exact():
    v, l = _Vis('AA', '1'), _lat('AA', '2')
    assert CP.is_latent(l) and not CP.is_latent(v)
    assert CP.bucket_key(v, ORG) == ('visible', 'AA')
    assert CP.bucket_key(l, ORG) == ('latent', 'AA')
    assert CP.bucket_key(v, SCH) == CP.bucket_key(l, SCH) == ('AA',)


def test_primitives_do_not_participate_in_either_admission_stage():
    """The pipeline map records that primitives are capped at enumeration and re-enter only at K_act, so no policy
    here creates a primitive bucket. This test pins that: the admission stages see rewrite candidates only."""
    src = (REPO/'reweave'/'candidate_policy.py').read_text()
    body = re.sub(r'"""(?:.|\n)*?"""', '', src)
    assert 'GROUP' not in body and 'FACTOR' not in body and 'REUSE' not in body
    search = (REPO/'reweave'/'rewrite_search_latent.py').read_text()
    assert 'select(cands+lat, cfg.rewrite_n_max' in search, 'S-CAP must admit rewrite candidates only'


# ---------------------------------------------------------------- 12 / 13: identity and cache keys


def test_policy_semantic_hashes_are_distinct():
    hs = {pol.name: pol.hash() for pol in (HASH, SCH, ORG)}
    assert len(set(hs.values())) == 3, hs
    assert CP.CandidatePolicy(CP.HASH).hash() == HASH.hash(), 'policy hash is a pure function of its semantics'


def test_policy_specific_label_cache_keys_cannot_collide():
    keys = {f'{pol.name}:{pol.hash()}' for pol in (HASH, SCH, ORG)}
    assert len(keys) == 3
    assert len({k.split(':')[1] for k in keys}) == 3


# ---------------------------------------------------------------- 14 / 15: exactness and CPU model


def test_admission_policy_cannot_change_exactness():
    """A policy only chooses among already-generated candidates, so it cannot alter any successor. Applying every
    policy to the same corpus yields subsets of one another's source set with identical members."""
    c = corpus()
    got = {pol.name: {x.order_key for x in CP.select(c, 80, pol)} for pol in (HASH, SCH, ORG)}
    assert got[CP.HASH] == got[CP.SCHEMA_DIVERSITY] == got[CP.ORIGIN_SCHEMA_DIVERSITY] == {x.order_key for x in c}


# ---------------------------------------------------------------- frozen-module equivalence (provenance)


def _fixtures():
    """The latent-exposure fixture polynomials, loaded from the sibling module rather than duplicated."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('_lat_fix', Path(__file__).with_name('test_latent_exposure.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_latent_summary_only_adds_fields_to_the_frozen_one():
    """The new module may add latent counters but must not rename, reorder or reinterpret a frozen field."""
    from reweave.rewrite_search import RewriteSearchSummary
    from reweave.rewrite_search_latent import LatentSearchSummary
    a, b = list(RewriteSearchSummary().as_dict()), list(LatentSearchSummary().as_dict())
    assert b[:len(a)] == a, 'frozen summary fields must keep their names and order'
    assert set(b)-set(a) == {'latent_attempts', 'latent_generated', 'latent_selected', 'latent_applies'}


def test_latent_search_matches_frozen_beam_search_when_disabled():
    """The duplicated loop must reproduce the frozen one exactly when latent exposure is off: same summary (bar
    timing), same incumbent cost, same recorded search tree."""
    from reweave.rewrite_search import beam_search as frozen
    from reweave.rewrite_search_latent import beam_search as latent_aware
    F = _fixtures()
    sch = F.square()
    poly = F.times(F.expanded_square(), (0, 0, 1)+(0,)*5)

    class _Flat:
        def score(self, arena, info, root, actions):
            return [0.0]*len(actions)

        def score_combined(self, arena, info, root, actions, cands):
            return [0.0]*len(actions), [0.0]*len(cands)

    timing = ('cpu_time', 'wall_time', 'peak_rss')
    applied = 0
    for poly in (F.expanded_square(), F.times(F.expanded_square(), (0, 0, 1)+(0,)*5)):
        out = []
        for fn, kw in ((frozen, {}), (latent_aware, dict(latent=None, policy=None))):
            a, root = F.arena_of(poly)
            task = v1.Task(poly, 8, 1, {})
            inc, st, tree = fn(a, task, root, _Flat(), v1.CostRanker(), v1.Budget(6, 24, 6, 40), [sch], CFG, True, **kw)
            d = {k: v for k, v in st.as_dict().items() if k not in timing and k in FROZEN_SUMMARY_FIELDS}
            out.append((v1.cost(a, inc), d, sorted((n, v['action'], v['cost']) for n, v in tree.items())))
        assert out[0][0] == out[1][0], 'incumbent cost differs'
        assert out[0][1] == out[1][1], [(k, out[0][1][k], out[1][1][k])
                                        for k in out[0][1] if out[0][1][k] != out[1][1][k]]
        assert out[0][2] == out[1][2], 'recorded search tree differs'
        applied += out[0][1]['rewrite_applies']
    assert applied > 0, 'at least one fixture must exercise the rewrite path'
