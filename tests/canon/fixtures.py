"""Checkpoint-free fixtures shared by the canonical-architecture tests (tests only; never enter a library)."""

from collections import Counter

from reweave import rewrite_latent as L
from reweave import rewrite_schema as S
from reweave import v1
from reweave.rewrite_config import RewriteConfig

CFG = RewriteConfig()
LCFG = L.LatentConfig()


def M(*ie):
    return ('M', tuple(ie))


def mono(*e):
    return v1.pack(tuple(e)+(0,)*(8-len(e)))


def square():
    """(H0+H1)^2 read right to left: source 2ab + a^2 + b^2, target (a+b)^2 — the deployed Square's shape."""
    src = ((M((0, 2)), 1), (M((0, 1), (1, 1)), 2), (M((1, 2)), 1))
    tgt = ((('P', ((('S', ((M((0, 1)), 1), (M((1, 1)), 1))), 2),)), 1),)
    return S.canonicalize(('mon', 'mon'), src, tgt)


def distrib():
    """ab + ac -> a(b+c)."""
    src = ((M((0, 1), (1, 1)), 1), (M((0, 1), (2, 1)), 1))
    tgt = ((('P', ((M((0, 1)), 1), (('S', ((M((1, 1)), 1), (M((2, 1)), 1))), 1))), 1),)
    return S.canonicalize(('mon', 'mon', 'mon'), src, tgt)


def arena_of(poly):
    a = v1.Arena()
    a.calc_cfg = v1.CalcCfg()
    return a, v1.canonical_start(a, poly)


class Flat:
    """A model-free scorer: every action scores 0, so ties break by the frozen order (primitive first, then key)."""

    def score(self, arena, info, root, actions):
        return [0.0]*len(actions)

    def score_combined(self, arena, info, root, actions, cands):
        return [0.0]*len(actions), [0.0]*len(cands)


# polynomials: exponent tuples over a..f (a=0 … f=5)
def P(d: dict):
    return {mono(*e): c for e, c in d.items()}


Q1 = P({(2, 0, 0, 1): 1, (1, 1, 0, 1): 2, (0, 2, 0, 1): 1})                                        # d(a+b)^2
Q2 = P({(2, 2, 1, 1): 1, (1, 1, 1, 1, 1): 2, (0, 0, 1, 1, 2): 1, (0, 0, 0, 0, 0, 1): 1})           # cd(ab+e)^2 + f
A0 = P({(2, 0, 0, 2): 1, (1, 1, 0, 2): 2, (0, 2, 0, 2): 1})                                        # d^2(a+b)^2
N1 = P({(2, 0, 0, 1): 1, (1, 1, 0, 0, 1): 2, (0, 2, 0, 0, 0, 1): 1})                               # da^2+2eab+fb^2
N2 = P({(2, 0, 0, 1): 1, (1, 1, 0, 1): 2, (0, 2, 0, 0, 1): 1})                                     # da^2+2dab+eb^2
SQ = P({(2,): 1, (1, 1): 2, (0, 2): 1})                                                            # (a+b)^2
TWO_SQ = P({(2,): 1, (1, 1): 2, (0, 2): 1, (0, 0, 2): 1, (0, 0, 1, 1): 2, (0, 0, 0, 2): 1})        # (a+b)^2+(c+d)^2
CUBE = P({(3,): 1, (2, 1): 3, (1, 2): 3, (0, 3): 1})
FIXTURES = dict(Q1=Q1, Q2=Q2, A0=A0, N1=N1, N2=N2, SQ=SQ, TWO_SQ=TWO_SQ, CUBE=CUBE)


def relabel_poly(poly: dict, perm) -> dict:
    """Rename variables: variable i becomes perm[i]."""
    out = {}
    for m, c in poly.items():
        e = v1.unpack(m)
        f = [0]*8
        for i, x in enumerate(e):
            if x:
                f[perm[i]] = x
        out[v1.pack(tuple(f))] = c
    return out


def compound_square_state(arena, multiplier=None):
    """H^2 + 2HK + K^2 with H = a+b, K = c+e (compound parameters), optionally times a monomial."""
    mk = lambda *e: v1.mk_mono(arena, mono(*e))
    H = v1.mk_sum(arena, Counter({mk(1): 1, mk(0, 1): 1}))
    K = v1.mk_sum(arena, Counter({mk(0, 0, 1): 1, mk(0, 0, 0, 0, 1): 1}))
    extra = Counter({mk(*multiplier): 1}) if multiplier else Counter()
    return v1.mk_sum(arena, Counter({v1.mk_prod(arena, Counter({H: 2})+extra): 1,
                                     v1.mk_prod(arena, Counter({H: 1, K: 1})+extra): 2,
                                     v1.mk_prod(arena, Counter({K: 2})+extra): 1}))
