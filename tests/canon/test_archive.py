"""Certified archive: canonical semantic family identity, surface aliases, exact JSON round trip, retrieval == exhaustive
visible matching, behaviour unchanged with retrieval OFF."""

import ast
import random
import sys
from pathlib import Path

import pytest

from reweave import candidate_policy as CP
from reweave import rewrite_schema as S
from reweave import v1
from reweave.canon import archive as AR
from reweave.canon.config import IntegrationConfig
from reweave.rewrite_match import match_state

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixtures as F  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
PORTED = {
    ('5f6af3d', 'reweave/c7_library.py'): ('_nodes', '_depth', '_sig', '_triple', 'region_features', 'features',
                                          'specificity', '_pattern_profile', '_Compiled', 'compile_schema', '_kind_of',
                                          '_node_triple', '_ContextView', 'passes', 'generic_instance', 'subsumed_by'),
    ('7ab4341', 'reweave/c7_3_memory.py'): ('required_keys', 'available_keys', 'Surface', 'Family', 'Archive',
                                           'match_state_retrieved'),
}
NEW = {'legacy_surface_id', 'relation_form', 'canonical_family_id', 'CanonicalArchive', 'build_archive'}


class _Level(ast.NodeTransformer):
    """`from . import rewrite_match` (sibling of the source module) == `from .. import rewrite_match` (parent here)."""

    def visit_ImportFrom(self, node):
        if node.module is None and node.level == 2:
            node.level = 1
        return node


def defs(src, normalize=False):
    tree = ast.parse(src)
    if normalize:
        tree = _Level().visit(tree)
    return {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def square_variant(types=('mon', 'mon'), swap=False):
    a, b = (1, 0) if swap else (0, 1)
    src = ((F.M((a, 2)), 1), (F.M((a, 1), (b, 1)), 2), (F.M((b, 2)), 1))
    tgt = ((('P', ((('S', ((F.M((a, 1)), 1), (F.M((b, 1)), 1))), 2),)), 1),)
    return S.canonicalize(types, src, tgt)


def test_family_id_is_binding_mode_and_parameter_order_independent():
    base = square_variant()
    expr = square_variant(types=('expr', 'expr'))
    swapped = square_variant(swap=True)
    assert expr.schema_hash != base.schema_hash, 'a different hole-type label is a different LEGACY id'
    fid = AR.canonical_family_id(base)
    assert AR.canonical_family_id(expr) == fid, 'binding mode must not create a separate semantic family'
    assert AR.canonical_family_id(swapped) == fid
    assert AR.canonical_family_id(F.distrib()) != fid
    assert AR.legacy_surface_id(base) == base.schema_hash


def test_family_ids_are_deterministic_across_arenas_and_calls():
    s = [F.square(), F.distrib()]
    first = [AR.canonical_family_id(x) for x in s]
    junk = v1.Arena()
    junk.calc_cfg = v1.CalcCfg()
    for i in range(50):
        v1.mk_mono(junk, F.mono(i % 4, i % 3, i % 2))
    assert [AR.canonical_family_id(x) for x in s] == first


def l0():
    from reweave.current_config import CURRENT
    from reweave.rewrite_library import schemas_of
    return schemas_of(CURRENT.library.assert_identity())


def test_l0_archive_identity_aliases_and_exact_round_trip():
    schemas = l0()
    assert len(schemas) == 19
    a1 = AR.build_archive(schemas, F.CFG)
    shuffled = list(schemas)
    random.Random(7).shuffle(shuffled)
    a2 = AR.build_archive(shuffled, F.CFG)
    assert a1.alias_map() == a2.alias_map() and a1.digest() == a2.digest(), 'insertion is in legacy-id order'
    am = a1.alias_map()
    assert set(am) == {s.schema_hash for s in schemas}, 'every legacy surface id is kept, unchanged'
    for sid, e in am.items():
        assert e['legacy_surface_id'] == sid and len(e['canonical_semantic_family_id']) == 64
    j = a1.to_json()
    a3 = AR.CanonicalArchive.from_json(j, F.CFG)
    assert a3.to_json() == j
    assert all(m['certification'] == 'certified' for m in a1.meta.values())


def test_archive_admits_certified_relations_only():
    bad_src = ((F.M((0, 2)), 1),)
    bad_tgt = ((F.M((0, 1)), 2),)
    bad = S.canonicalize(('mon',), bad_src, bad_tgt)
    a = AR.CanonicalArchive()
    with pytest.raises(ValueError):
        a.add_certified(bad, dict(source='test'), F.CFG)
    assert len(a.surfaces) == 0


@pytest.mark.parametrize('name', sorted(F.FIXTURES))
def test_retrieval_equals_exhaustive_visible_matching(name):
    schemas = l0()+[F.square(), F.distrib()]
    arch = AR.build_archive(schemas, F.CFG)
    a, root = F.arena_of(F.FIXTURES[name])
    exh, _ = match_state(a, root, sorted(schemas, key=lambda s: s.schema_hash), F.CFG)
    ret, _ = AR.match_state_retrieved(a, root, arch, F.CFG)
    assert [c.order_key for c in exh] == [c.order_key for c in ret]


@pytest.mark.parametrize('name', sorted(F.FIXTURES))
def test_search_with_retrieval_equals_search_without(name):
    from reweave.canon.search import beam_search
    schemas = sorted([F.square(), F.distrib()], key=lambda s: s.schema_hash)
    arch = AR.build_archive(schemas, F.CFG)
    out = []
    for icfg in (IntegrationConfig(), IntegrationConfig(archive_retrieval=True)):
        a, root = F.arena_of(F.FIXTURES[name])
        task = v1.Task(F.FIXTURES[name], 8, 1, {})
        inc, st, tree = beam_search(a, task, root, F.Flat(), v1.CostRanker(), v1.Budget(6, 24, 6, 60), schemas, F.CFG,
                                    True, latent=F.LCFG, policy=CP.CandidatePolicy(CP.ORIGIN_SCHEMA_DIVERSITY),
                                    icfg=icfg, archive=arch)
        out.append((v1.digest(a, inc), [(n, tree[n]['action'], tree[n]['cost']) for n in sorted(tree)]))
    assert out[0] == out[1]
