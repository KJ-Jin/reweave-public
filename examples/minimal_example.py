"""ReWeave in miniature: one turn of the abstraction loop, using the real implementation.

    1. SEARCH    primitive-only exact search on a few small training polynomials
    2. INDUCE    mine the successful trajectories for reusable parameterized relations
    3. CERTIFY   keep only relations that are formal identities, and insert them into a certified archive
    4. REUSE     apply the induced relation to unseen instances: visibly, through a hidden monomial
                 multiplier (latent quotient), and over compound sub-expressions (ExprParam handle binding)
    5. SEARCH    search again with the enlarged action language, under the same small budget

Every successor is lowered to the ordinary ADD/MUL circuit and exact-verified against the target polynomial.
Cost is the number of unique ADD/MUL gates of that ordinary circuit; abstractions get no discount.

Run:  python examples/minimal_example.py        (a few seconds on a laptop CPU)
"""

import dataclasses
from collections import Counter

from reweave import v1
from reweave.candidate_policy import CandidatePolicy
from reweave.canon import action_record, archive, exprparam
from reweave.rewrite_config import RewriteConfig
from reweave.rewrite_latent import LatentConfig, apply_any, apply_latent, latent_candidates
from reweave.rewrite_library import build_library, schemas_of
from reweave.rewrite_match import apply_candidate, match_state
from reweave.rewrite_search_latent import beam_search

VARS = 'abcdefgh'


def poly(*terms):
    """poly(('a*a*b', 1), ('c', 2)) = a^2 b + 2c, as an exact polynomial over the variables a..h."""
    out = {}
    for spec, coeff in terms:
        e = [0]*len(VARS)
        for v in spec.split('*'):
            e[VARS.index(v)] += 1
        m = v1.pack(tuple(e))
        out[m] = out.get(m, 0)+coeff
    return out


def square(x, y, *extra):
    """x^2 + 2xy + y^2 (+ extra terms), with x and y monomials such as 'a' or 'c*d'."""
    return poly((f'{x}*{x}', 1), (f'{x}*{y}', 2), (f'{y}*{y}', 1), *extra)


def flat_start(target):
    """A fresh arena holding the canonical flat (sum-of-monomials) circuit of `target`."""
    arena = v1.Arena()
    arena.calc_cfg = v1.CalcCfg()
    return arena, v1.canonical_start(arena, target)


def show(schema):
    """Render a relation's source and target with parameters A, B, C, D."""
    def pat(p):
        if p[0] == 'M':
            return '*'.join('ABCD'[i]+(f'^{e}' if e > 1 else '') for i, e in p[1]) or '1'
        op = ' + ' if p[0] == 'S' else '*'
        parts = [(f'{n}*' if p[0] == 'S' and n > 1 else '')+pat(c)+(f'^{n}' if p[0] == 'P' and n > 1 else '')
                 for c, n in p[1]]
        return '('+op.join(parts)+')'
    region = lambda r: ' + '.join((f'{n}*' if n > 1 else '')+pat(c) for c, n in r)
    return f'{region(schema.source)}  ->  {region(schema.target)}'


class ProbeScorer:
    """A hand-written stand-in for the learned action-selection policy: score every candidate action by the exact
    one-step change in gate cost. (The learned scorers are `reweave.rewrite_model` and `reweave.canon.action_policy`.)
    Scores only order the candidates; correctness is decided by exact verification, never by the scorer."""

    def __init__(self, task):
        self.task, self.primitive = task, v1.ImmediateGain(task)

    def score(self, arena, info, root, actions):
        return self.primitive.score(arena, info, root, actions)

    def score_combined(self, arena, info, root, actions, rewrites):
        gains = []
        for cand in rewrites:
            succ, _ = apply_any(arena, root, cand, self.task.target)
            gains.append(float('-inf') if succ is None else float(v1.cost(arena, root)-v1.cost(arena, succ)))
        return self.score(arena, info, root, actions), gains


def main():
    cfg = RewriteConfig()

    print('1. SEARCH: primitive-only exact search on training polynomials')
    train = [square('a', 'b'), square('c*d', 'e'), square('a*b', 'c', ('g', 1)), square('d', 'e*f', ('h*g', 1)),
             square('b', 'g'), square('d*h', 'a', ('e*e', 1))]
    items = []
    for i, target in enumerate(train):
        task = v1.Task(target, 8, 0, dict(id=i))
        arena, start = flat_start(target)
        best, _, tree = v1.beam_search(arena, task, start, v1.ImmediateGain(task), v1.CostRanker(),
                                       v1.Budget(8, 8, 20, 400), True)
        print(f'   task {i}: {v1.cost(arena, start):2d} gates -> {v1.cost(arena, best):2d} gates')
        items.append((v1.extract_record(arena, task, start, best, tree), dict(family='square', shape=f's{i % 3}')))

    print('\n2. INDUCE + 3. CERTIFY: trajectory windows -> parameterized relations -> exact certification')
    # Support thresholds are lowered from the defaults (5 tasks / 3 shapes) because this corpus has only 6 tasks.
    small = dataclasses.replace(cfg, task_support_min=3, shape_support_min=2)
    library, _ = build_library(items, small, 'example')
    learned = schemas_of(library)
    print(f'   {library["n_pairs"]} ground pairs, {library["n_certified"]} certified relations, '
          f'{len(learned)} admitted after support / alias / eligibility checks')
    # The certified archive re-certifies every relation on insertion and keys it by its semantic family id.
    lib = archive.build_archive(learned, cfg, dict(source='minimal_example'))
    for s in learned:
        meta = lib.meta[s.schema_hash]
        print(f'   family {meta["relation_id"][:10]}  {meta["certification"]:9s}  {show(s)}')
    sq = next(s for s in learned if len(s.target) == 1)         # the Square relation (A + B)^2

    print('\n4. REUSE on unseen instances (each successor is lowered and exact-verified)')
    target = square('d', 'g', ('e*h', 1))                        # (d + g)^2 + eh: visible
    arena, root = flat_start(target)
    cands, _ = match_state(arena, root, [sq], cfg)
    succ, verdict = apply_candidate(arena, root, cands[0], target)
    print(f'   visible   (d+g)^2 + eh         : {v1.cost(arena, root):2d} -> {v1.cost(arena, succ):2d} gates  [{verdict}]')

    target = poly(('f*a*a', 1), ('f*a*c', 2), ('f*c*c', 1), ('b', 1))   # f(a + c)^2 + b: hidden behind f
    arena, root = flat_start(target)
    visible, _ = match_state(arena, root, [sq], cfg)
    lat, _ = latent_candidates(arena, root, [sq], cfg, LatentConfig())
    succ, verdict = apply_latent(arena, root, lat[0], target)
    print(f'   latent    f(a+c)^2 + b         : {v1.cost(arena, root):2d} -> {v1.cost(arena, succ):2d} gates  [{verdict}]'
          f'  ({len(visible)} visible matches; quotient d = {"".join(VARS[i] for i, e in enumerate(lat[0].d) if e)})')

    def compound_square(arena):
        """H^2 + 2HK + K^2 with H = a + b and K = c + e already present as compound handles."""
        mono = lambda v: v1.mk_mono(arena, v1.pack(tuple(int(x == v) for x in VARS)))
        H = v1.mk_sum(arena, Counter({mono('a'): 1, mono('b'): 1}))
        K = v1.mk_sum(arena, Counter({mono('c'): 1, mono('e'): 1}))
        prod = lambda kids: v1.mk_prod(arena, Counter(kids))
        return v1.mk_sum(arena, Counter({prod({H: 2}): 1, prod({H: 1, K: 1}): 2, prod({K: 2}): 1}))

    arena = v1.Arena()
    arena.calc_cfg = v1.CalcCfg()
    root = compound_square(arena)
    target = v1.hpoly(arena, root)
    handle, _ = exprparam.handle_visible(arena, root, [sq], cfg)
    succ, verdict = exprparam.apply_candidate_g(arena, root, handle[0], target)
    print(f'   compound  H^2+2HK+K^2, H=a+b.. : {v1.cost(arena, root):2d} -> {v1.cost(arena, succ):2d} gates  [{verdict}]'
          f'  ({len(match_state(arena, root, [sq], cfg)[0])} monomial-only matches)')

    # Semantic identity: the same action built in a differently allocated arena has the same semantic key.
    other = v1.Arena()
    other.calc_cfg = v1.CalcCfg()
    for i in range(5):                                           # unrelated handles shift every allocation id
        v1.mk_mono(other, v1.pack((0,)*7+(i+1,)))
    root2 = compound_square(other)
    keys = [action_record.rewrite_record(a, exprparam.handle_visible(a, r, [sq], cfg)[0][0], 'handle').semantic_key
            for a, r in ((arena, root), (other, root2))]
    print(f'   semantic action key independent of arena allocation: {keys[0] == keys[1]}  ({keys[0][:16]}…)')

    print('\n5. SEARCH AGAIN: same small budget (B=2, K_act=2, T=6, 20 expansions), primitive vs. enlarged language')
    budget = v1.Budget(2, 2, 6, 20)
    tests = {'f(ab+c)^2 + (d+e)^2': poly(('f*a*a*b*b', 1), ('f*a*b*c', 2), ('f*c*c', 1), ('d*d', 1), ('d*e', 2), ('e*e', 1)),
             'c(a+b)^2 + d(a+b)^2': poly(('a*a*c', 1), ('a*b*c', 2), ('b*b*c', 1), ('a*a*d', 1), ('a*b*d', 2), ('b*b*d', 1)),
             'g(c+d)^2 + h(c+d)^2 + e': poly(('g*c*c', 1), ('g*c*d', 2), ('g*d*d', 1), ('h*c*c', 1), ('h*c*d', 2),
                                              ('h*d*d', 1), ('e', 1))}
    for name, target in tests.items():
        task = v1.Task(target, 8, 0, dict(id=100))
        arena, start = flat_start(target)
        prim, _, _ = v1.beam_search(arena, task, start, v1.ImmediateGain(task), v1.CostRanker(), budget, False)
        c_start, c_prim = v1.cost(arena, start), v1.cost(arena, prim)
        arena, start = flat_start(target)
        best, stats, _ = beam_search(arena, task, start, ProbeScorer(task), v1.CostRanker(), budget, learned, cfg,
                                     False, LatentConfig(), CandidatePolicy())
        assert v1.poly_equal(v1.hpoly(arena, best), target)
        print(f'   {name:26s} start {c_start:2d}   primitive {c_prim:2d}   with library {v1.cost(arena, best):2d}'
              f'   (rewrites applied {stats.rewrite_applies}, exact-verification failures '
              f'{stats.exact_verification_failures})')


if __name__ == '__main__':
    main()
