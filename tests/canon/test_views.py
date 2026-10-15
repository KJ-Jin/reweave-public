"""View actions: views OFF == identity mode without views, indexed retrieval == eager matching, exact and replayable view
actions, deterministic."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from reweave import candidate_policy as CP
from reweave import v1
from reweave.canon import latent_identity as LI
from reweave.canon import views as VW
from reweave.canon.config import IntegrationConfig
from reweave.canon.identity_search import run_identity

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixtures as F  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
TIMING = ('cpu_time', 'wall_time', 'peak_rss', 'norm_cpu', 'identity_cpu')
VERBATIM = {
    ('0bf1da7', 'reweave/c7_2i_intern.py'): ('match_contexts', 'ctx_candidates', 'ViewAction', '_orbits'),
    ('63d84aa', 'reweave/c7_2r_runtime.py'): ('node_sig', '_bucket', 'sig_cache', 'ctx_sig', 'mono_class', 'pattern_sig',
                                             'SchemaIndex', 'rep_signatures', 'Retrieval', 'make_view_fn',
                                             '_ctx_candidates'),
}
MODIFIED = {('0bf1da7', 'reweave/c7_2i_intern.py'): 'view_actions',
            ('63d84aa', 'reweave/c7_2r_runtime.py'): 'view_actions_indexed'}
SCHEMAS = [F.square(), F.distrib()]


class _DropP1(ast.NodeTransformer):
    """Remove exactly the P1 score-key statements: the c7_policy import, `K = ctx.sc._keys(ca)`, `va.p = …`, `va.s = …`."""

    def visit_ImportFrom(self, node):
        return None if any(al.name == 'c7_policy' for al in node.names) or node.module == 'c7_policy' else node

    def visit_Assign(self, node):
        t = node.targets[0]
        if isinstance(t, ast.Name) and t.id == 'K':
            return None
        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'va' and t.attr in ('p', 's'):
            return None
        return node


def defs(src, drop=False):
    tree = ast.parse(src)
    if drop:
        tree = ast.fix_missing_locations(_DropP1().visit(tree))
    return {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def show(commit, path):
    return subprocess.run(['git', '-C', str(REPO), 'show', f'{commit}:{path}'], capture_output=True, text=True,
                          check=True).stdout


def ident(poly, icfg, record=True):
    task = v1.Task(poly, 8, 1, {})
    views = VW if icfg.effective().view_reuse else None
    return run_identity(task, F.Flat(), v1.CostRanker(), v1.Budget(6, 24, 6, 60), SCHEMAS, F.CFG, record, latent=F.LCFG,
                        policy=CP.CandidatePolicy(CP.ORIGIN_SCHEMA_DIVERSITY), icfg=icfg, views=views)


def sig(res):
    ca = res['arena']
    return ([v1.digest(ca, x).hex() for x in LI.lineage(res)],
            {k: v for k, v in res['summary'].as_dict().items() if k not in TIMING and k != 'view_cpu'})


@pytest.mark.parametrize('name', sorted(F.FIXTURES))
def test_views_off_equals_t4_identity(name):
    base = ident(F.FIXTURES[name], IntegrationConfig(identity_mode=True, normalization=True))
    task = v1.Task(F.FIXTURES[name], 8, 1, {})
    again = run_identity(task, F.Flat(), v1.CostRanker(), v1.Budget(6, 24, 6, 60), SCHEMAS, F.CFG, True, latent=F.LCFG,
                         policy=CP.CandidatePolicy(CP.ORIGIN_SCHEMA_DIVERSITY),
                         icfg=IntegrationConfig(identity_mode=True, normalization=True), views=VW)
    assert sig(base) == sig(again)


@pytest.mark.parametrize('name', sorted(F.FIXTURES))
def test_indexed_views_equal_eager_views_and_are_exact(name):
    poly = F.FIXTURES[name]
    idx = ident(poly, IntegrationConfig(identity_mode=True, normalization=True, view_reuse=True, view_index=True))
    eag = ident(poly, IntegrationConfig(identity_mode=True, normalization=True, view_reuse=True, view_index=False))
    assert sig(idx)[0] == sig(eag)[0]
    for res in (idx, eag):
        s = res['summary']
        assert s.exact_verification_failures == 0 and s.norm_failures == 0 and res['sem'].port.stats['failures'] == 0
        a = v1.Arena()
        a.calc_cfg = v1.CalcCfg()
        states, steps = LI.concretize(res, a)
        assert LI.replay(a, poly, states, steps)
        assert LI.deserialize_and_replay(LI.serialize(a, poly, states, steps), {x.schema_hash: x for x in SCHEMAS}, poly)


def test_views_are_deterministic():
    for name in ('TWO_SQ', 'CUBE'):
        c = IntegrationConfig(identity_mode=True, normalization=True, view_reuse=True)
        assert sig(ident(F.FIXTURES[name], c)) == sig(ident(F.FIXTURES[name], c))


def test_view_defaults_are_off_and_require_identity():
    e = IntegrationConfig().effective()
    assert not (e.view_reuse or e.portfolio_alternatives or e.view_index)
    with pytest.raises(ValueError):
        IntegrationConfig(view_reuse=True).effective()
    e = IntegrationConfig(identity_mode=True, view_reuse=True).effective()
    assert e.portfolio_alternatives and e.semantic_interning and e.view_index
