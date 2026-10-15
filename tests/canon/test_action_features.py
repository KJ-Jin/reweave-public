"""ActionRecord -> scorer features, no learned component. Family structure is a function of the exact relation payload
alone (19 reference surfaces -> 18 distinct family vectors, the one AC-aliased pair identical, multiplicity not
collapsed); each action and its variable-renamed counterpart get identical features; allocation-exact; binding values
independent of binding mode and realization; latent d structural."""

import math

import pytest

from reweave import rewrite_schema as S
from reweave import v1
from reweave.canon import action_record as ARec
from reweave.canon import relation as REL
from reweave.canon import action_features as TF
from reweave.rewrite_certify import certify

from scoring_helpers import ALIAS, M, PERMS, T6, _rename, _renamed_candidate, fam_of, numeric, records_at


@pytest.fixture(scope='module')
def lib():
    return T6.l0()


@pytest.fixture(scope='module')
def sq(lib):
    return next(s for s in lib if s.schema_hash == T6.SQUARE)


def test_family_vector_is_payload_only_and_separates_the_18_l0_families(lib):
    vecs = {s.schema_hash: fam_of(s) for s in lib}
    assert len(set(vecs.values())) == 18
    assert vecs[ALIAS[0]] == vecs[ALIAS[1]], 'two surfaces of one AC family: identical FAMILY STRUCTURE'
    by_payload: dict = {}
    for s in lib:
        by_payload.setdefault(REL.family_payload(s), set()).add(vecs[s.schema_hash])
    assert all(len(v) == 1 for v in by_payload.values()) and len(by_payload) == 18
    sx = T6.square_expr_surface()
    assert fam_of(sx) == vecs[T6.SQUARE], 'another recognition surface of Square: same family vector'
    for v in vecs.values():
        assert len(v[0]) == TF.N_FAMILY and all(math.isfinite(x) for x in v[0])


def test_multiplicity_is_not_collapsed():
    sq = S.canonicalize(('mon', 'mon'), ((M((0, 2)), 1), (M((0, 1), (1, 1)), 2), (M((1, 2)), 1)),
                        ((('P', ((('S', ((M((0, 1)), 1), (M((1, 1)), 1))), 2),)), 1),))
    dbl = S.canonicalize(('mon', 'mon'), ((M((0, 2)), 2), (M((0, 1), (1, 1)), 4), (M((1, 2)), 2)),
                         ((('P', ((('S', ((M((0, 1)), 1), (M((1, 1)), 1))), 2),)), 2),))
    assert certify(sq)[0] and certify(dbl)[0]
    assert REL.family_payload(sq) != REL.family_payload(dbl)
    assert fam_of(sq) != fam_of(dbl), 'relations that differ only in multiplicity keep different structure vectors'


@pytest.mark.parametrize('name', ['positive', 'mixed', 'noninjective_two', 'latent_square', 'repeated_semantic',
                                  'grouped_product'])
def test_features_are_variable_renaming_and_allocation_invariant(name, lib):
    """Each action mapped to its variable-renamed counterpart gets identical T7 features (the renamed candidate
    SET may differ where generation truncates by digest order — a T6 property, not a feature property)."""
    a = T6.arena()
    root, _ = T6.BUILDERS[name](a)
    recs = [r for r in records_at(a, root, lib) if r.kind != 'PRIMITIVE']
    assert recs
    base = [numeric(f) for f in TF.record_features(recs, a, root)]
    for p in PERMS[1:]:
        b, memo = T6.arena(), {}
        rb = _rename(a, root, b, p, memo)
        rb2, _ = T6.BUILDERS[name](b, p)
        assert rb == rb2, 'the renamed state is the builder state under the renaming'
        cands = [_renamed_candidate(a, r.live, b, p, memo) for r in recs]
        fb = TF.record_features(ARec.records(b, [], cands), b, rb)
        assert [numeric(f) for f in fb] == base, 'T7 features are symmetric in the variables'
    c = T6.arena(29)
    rc, _ = T6.BUILDERS[name](c)
    recs_c = [r for r in records_at(c, rc, lib) if r.kind != 'PRIMITIVE']
    assert [numeric(f) for f in TF.record_features(recs_c, c, rc)] == base, 'allocation-independent'


def test_binding_value_is_mode_and_realization_independent(sq, lib):
    a = T6.arena()
    root, hs = T6.repeated_semantic(a)
    recs = [r for r in records_at(a, root, lib, latent=False) if r.kind != 'PRIMITIVE'
            and r.provenance['replay']['surface'] == sq.schema_hash]
    feats = TF.record_features(recs, a, root)
    assert len(feats) == 2
    nv = len(TF.ROLE_FIELDS)
    vals = [tuple(r[nv:nv+len(TF.VALUE_FIELDS)] for r in f.roles) for f in feats]
    assert vals[0] == vals[1], 'H and H2 have the same exact value: same value features'
    assert feats[0].family == feats[1].family
    assert feats[0].concrete != feats[1].concrete or feats[0].roles != feats[1].roles, \
        'the two concrete realizations remain distinguishable'
    b = T6.arena()
    r2, _ = T6.mixed(b)
    mixed = [r for r in records_at(b, r2, lib, latent=False) if r.kind != 'PRIMITIVE']
    fm = TF.record_features(mixed, b, r2)
    modes = {tuple(x['mode'] for x in r.provenance['replay']['bindings']) for r in mixed}
    assert len(modes) > 1, 'handle-bound and monomial-bound actions in one state'
    fams = {f.family for f, r in zip(fm, mixed) if r.provenance['replay']['surface'] == sq.schema_hash}
    assert len(fams) == 1, 'binding mode never changes the family representation'


def test_latent_d_is_structural(lib):
    outs = []
    for p in PERMS:
        a = T6.arena()
        root, _ = T6.latent_square(a, p)
        recs = [r for r in records_at(a, root, lib) if r.kind == 'LATENT_REWRITE']
        assert recs
        outs.append(sorted(f.d for f in TF.record_features(recs, a, root)))
    assert all(o == outs[0] for o in outs) and outs[0][0][0] == 1.0
