"""Fixtures: concrete DAGs whose compound parameters ALREADY EXIST as handles before matching. Built only with the
runtime constructors (`mk_mono` / `mk_sum` / `mk_prod`). Tests only; never enter a library.

Variables: a..h = 0..7. Every builder takes a variable renaming `perm` (variable i becomes perm[i]) and returns
(root, named handles)."""

from collections import Counter

from reweave import v1

SQUARE = '3cac496ef8bd6e7bd9eac9103429ab4a9f741635462e844c6731b1c74cadfbea'   # L0 Square: A^2 + 2AB + B^2 -> (A+B)^2
IDENT8 = tuple(range(8))


def l0():
    from reweave.current_config import CURRENT
    from reweave.rewrite_library import schemas_of
    return schemas_of(CURRENT.library.assert_identity())


def l0_square():
    return next(s for s in l0() if s.schema_hash == SQUARE)


def mono(a, exps: dict, perm=IDENT8) -> int:
    e = [0]*8
    for i, x in exps.items():
        e[perm[i]] += x
    return v1.mk_mono(a, v1.pack(tuple(e)))


def arena(junk: int = 0):
    """A fresh arena, optionally pre-filled with `junk` unrelated handles (a different allocation order)."""
    a = v1.Arena()
    a.calc_cfg = v1.CalcCfg()
    for i in range(junk):
        v1.mk_sum(a, Counter({mono(a, {i % 7: 1+i % 3}): 1, mono(a, {7: 1}): 1+i % 2}))
    return a


def lin(a, idxs, perm=IDENT8) -> int:
    """x_i + x_j + ... as ONE compound Sum handle."""
    return v1.mk_sum(a, Counter({mono(a, {i: 1}, perm): 1 for i in idxs}))


def square_ctx(a, A: int, B: int, mult=None, perm=IDENT8) -> int:
    """A^2 + 2AB + B^2 over two DISTINCT existing handles (compound, or a Mono for a monomial parameter), optionally
    times the monomial `mult`."""
    assert A != B
    extra = Counter({mono(a, mult, perm): 1}) if mult else Counter()
    return v1.mk_sum(a, Counter({v1.mk_prod(a, Counter({A: 2})+extra): 1,
                                 v1.mk_prod(a, Counter({A: 1, B: 1})+extra): 2,
                                 v1.mk_prod(a, Counter({B: 2})+extra): 1}))


# ---------------------------------------------------------------- positive, negative, mixed


def positive(a, perm=IDENT8):
    """H^2 + HK + HK + K^2 with H = a+b and K = c+e existing compound handles."""
    H, K = lin(a, (0, 1), perm), lin(a, (2, 4), perm)
    return square_ctx(a, H, K, perm=perm), dict(H=H, K=K)


def negative_flat(a, perm=IDENT8):
    """The same polynomial as `positive`, flat: no compound handle exists anywhere."""
    b = arena()
    root, _ = positive(b, perm)
    return v1.canonical_start(a, v1.hpoly(b, root)), {}


def negative_partial(a, perm=IDENT8):
    """H exists; K = c+e does NOT (its terms are spread over separate children): H^2 + 2Hc + 2He + c^2 + 2ce + e^2."""
    H = lin(a, (0, 1), perm)
    c, e = mono(a, {2: 1}, perm), mono(a, {4: 1}, perm)
    kids = Counter({v1.mk_prod(a, Counter({H: 2})): 1, v1.mk_prod(a, Counter({H: 1, c: 1})): 2,
                    v1.mk_prod(a, Counter({H: 1, e: 1})): 2, mono(a, {2: 2}, perm): 1,
                    mono(a, {2: 1, 4: 1}, perm): 2, mono(a, {4: 2}, perm): 1})
    return v1.mk_sum(a, kids), dict(H=H)


def _four(a, perm):
    return [lin(a, p, perm) for p in ((0, 1), (2, 3), (4, 5), (6, 7))]


def flat_product(a, perm=IDENT8):
    """(h1 h3)^2 + 2 (h1 h3)(h2 h4) + (h2 h4)^2 with FLAT products: h1*h3 and h2*h4 are NOT handles."""
    h1, h2, h3, h4 = _four(a, perm)
    kids = Counter({v1.mk_prod(a, Counter({h1: 2, h3: 2})): 1, v1.mk_prod(a, Counter({h1: 1, h2: 1, h3: 1, h4: 1})): 2,
                    v1.mk_prod(a, Counter({h2: 2, h4: 2})): 1})
    return v1.mk_sum(a, kids), dict(h1=h1, h2=h2, h3=h3, h4=h4)


def grouped_product(a, perm=IDENT8):
    """The same polynomial with P13 = h1*h3 and P24 = h2*h4 materialized as handles."""
    h1, h2, h3, h4 = _four(a, perm)
    P13, P24 = v1.mk_prod(a, Counter({h1: 1, h3: 1})), v1.mk_prod(a, Counter({h2: 1, h4: 1}))
    return square_ctx(a, P13, P24, perm=perm), dict(P13=P13, P24=P24)


def mixed(a, perm=IDENT8):
    """H^2 + 2Hm + m^2 with H = a+b a compound handle and m = cd a monomial."""
    H, m = lin(a, (0, 1), perm), mono(a, {2: 1, 3: 1}, perm)
    return square_ctx(a, H, m, perm=perm), dict(H=H, m=m)


# ---------------------------------------------------------------- repeated / non-injective


def repeated_semantic(a, perm=IDENT8):
    """A occurs in A^2 and in AB; the AB child carries H2 = (a+b)+c, a DIFFERENT handle with the SAME polynomial as
    H = a+b+c. Repeated occurrences of A must agree semantically, not by handle id."""
    H = lin(a, (0, 1, 2), perm)
    H2 = v1.mk_sum(a, Counter({lin(a, (0, 1), perm): 1, mono(a, {2: 1}, perm): 1}))
    K = lin(a, (3, 4), perm)
    assert H != H2 and v1.poly_equal(v1.hpoly(a, H), v1.hpoly(a, H2))
    kids = Counter({v1.mk_prod(a, Counter({H: 2})): 1, v1.mk_prod(a, Counter({H2: 1, K: 1})): 2,
                    v1.mk_prod(a, Counter({K: 2})): 1})
    return v1.mk_sum(a, kids), dict(H=H, H2=H2, K=K)


def repeated_mismatch(a, perm=IDENT8):
    """As `repeated_semantic`, but the AB child carries J = a+d, a different polynomial: A cannot bind both."""
    H, J, K = lin(a, (0, 1, 2), perm), lin(a, (0, 3), perm), lin(a, (3, 4), perm)
    kids = Counter({v1.mk_prod(a, Counter({H: 2})): 1, v1.mk_prod(a, Counter({J: 1, K: 1})): 2,
                    v1.mk_prod(a, Counter({K: 2})): 1})
    return v1.mk_sum(a, kids), dict(H=H, J=J, K=K)


def noninjective(a, perm=IDENT8):
    """4 H^2 = H^2 + 2 H H + H^2: Square with A := H and B := H (distinct parameters, one value)."""
    H = lin(a, (0, 1), perm)
    return v1.mk_sum(a, Counter({v1.mk_prod(a, Counter({H: 2})): 4})), dict(H=H)


def noninjective_two(a, perm=IDENT8):
    """H^2 + 2 H H2 + H2^2 with H = a+b+c and H2 = (a+b)+c: one semantic value, two existing structural realizations.
    Square(A, B) with A = B = that value; the target may reuse H or H2 for each parameter."""
    H = lin(a, (0, 1, 2), perm)
    H2 = v1.mk_sum(a, Counter({lin(a, (0, 1), perm): 1, mono(a, {2: 1}, perm): 1}))
    kids = Counter({v1.mk_prod(a, Counter({H: 2})): 1, v1.mk_prod(a, Counter({H: 1, H2: 1})): 2,
                    v1.mk_prod(a, Counter({H2: 2})): 1})
    return v1.mk_sum(a, kids), dict(H=H, H2=H2)


def latent_view_reachable(a, perm=IDENT8):
    """f*(H^2 + 2HK + K^2) spread over three children, plus a child H*K elsewhere in the same context: the latent view's
    stripped product H*K then also exists as a reachable handle, but lies outside the consumed originals."""
    root, hs = latent_square(a, perm)
    HK = v1.mk_prod(a, Counter({hs['H']: 1, hs['K']: 1}))
    kids = Counter(dict(v1.items(a.handles[root])))
    kids[HK] += 1
    return v1.mk_sum(a, kids), dict(hs, HK=HK)


# ---------------------------------------------------------------- latent quotient, dependency safety


def latent_square(a, perm=IDENT8):
    """f * (H^2 + 2HK + K^2) spread over three children: f H^2, 2 f H K, f K^2 (d = f)."""
    H, K = lin(a, (0, 1), perm), lin(a, (2, 4), perm)
    return square_ctx(a, H, K, mult={5: 1}, perm=perm), dict(H=H, K=K)


def nested(a, perm=IDENT8):
    """The positive Square context inside a larger state, with a sibling handle Y and a handle Z that contains a
    consumed child (Z = (H^2) * h), for the fail-closed dependency negatives."""
    H, K = lin(a, (0, 1), perm), lin(a, (2, 4), perm)
    inner = square_ctx(a, H, K, perm=perm)
    Y = lin(a, (3, 6), perm)
    Z = v1.mk_prod(a, Counter({v1.mk_prod(a, Counter({H: 2})): 1, mono(a, {7: 1}, perm): 1}))
    root = v1.mk_sum(a, Counter({v1.mk_prod(a, Counter({inner: 1, mono(a, {6: 1}, perm): 1})): 1, Y: 1, Z: 1}))
    return root, dict(H=H, K=K, inner=inner, Y=Y, Z=Z)


def latent_unconsumed_original(a, perm=IDENT8):
    """f*H^2 + 2 f*H*K + f*K^2 plus o2 = f*P with P = H*K an existing nested product. The quotient view for d = f has
    the stripped child P three times — twice from the flat f*H*K children, once from o2 — but P lies below o2 only.
    A binding A := P is admissible only if THIS candidate's allocated originals include o2."""
    H, K = lin(a, (0, 1), perm), lin(a, (2, 4), perm)
    f = mono(a, {5: 1}, perm)
    P = v1.mk_prod(a, Counter({H: 1, K: 1}))
    o2 = v1.mk_prod(a, Counter({P: 1, f: 1}))
    fHK = v1.mk_prod(a, Counter({H: 1, K: 1, f: 1}))
    kids = Counter({v1.mk_prod(a, Counter({H: 2, f: 1})): 1, fHK: 2, v1.mk_prod(a, Counter({K: 2, f: 1})): 1, o2: 1})
    return v1.mk_sum(a, kids), dict(H=H, K=K, P=P, o2=o2, fHK=fHK, f=f)


def double_schema():
    """A + A -> (A + A) as a sum node: a lone-parameter leaf of multiplicity 2 (fixture only)."""
    from reweave import rewrite_schema as S
    return S.canonicalize(('mon',), ((('M', ((0, 1),)), 2),), ((('S', ((('M', ((0, 1),)), 2),)), 1),))


def square_expr_surface():
    """A second recognition surface of the Square family: the same relation with hole types ('expr', 'expr') — a
    different LEGACY_SURFACE_ID, the same family, the same lowering."""
    from reweave import rewrite_schema as S
    M = lambda *ie: ('M', tuple(ie))
    src = ((M((0, 2)), 1), (M((0, 1), (1, 1)), 2), (M((1, 2)), 1))
    tgt = ((('P', ((('S', ((M((0, 1)), 1), (M((1, 1)), 1))), 2),)), 1),)
    return S.canonicalize(('expr', 'expr'), src, tgt)


BUILDERS = dict(positive=positive, negative_flat=negative_flat, negative_partial=negative_partial,
                flat_product=flat_product, grouped_product=grouped_product, mixed=mixed,
                repeated_semantic=repeated_semantic, repeated_mismatch=repeated_mismatch, noninjective=noninjective,
                noninjective_two=noninjective_two, latent_square=latent_square,
                latent_view_reachable=latent_view_reachable, latent_unconsumed_original=latent_unconsumed_original,
                nested=nested)


def flat_product_schema():
    """A*B*(C+D) -> A*B*C + A*B*D: a multi-parameter leaf directly inside a product pattern (no L0 family has one)."""
    M = lambda *ie: ('M', tuple(ie))
    src = ((('P', ((M((0, 1), (1, 1)), 1), (('S', ((M((2, 1)), 1), (M((3, 1)), 1))), 1))), 1),)
    tgt = ((M((0, 1), (1, 1), (2, 1)), 1), (M((0, 1), (1, 1), (3, 1)), 1))
    from reweave import rewrite_schema as S
    return S.canonicalize(('mon',)*4, src, tgt)


def flat_product_state(a, perm=IDENT8):
    """H*K*(c+d) + g with H = a+b, K = e+f existing handles, as ONE flat V1 product."""
    H, K = lin(a, (0, 1), perm), lin(a, (4, 5), perm)
    cd = lin(a, (2, 3), perm)
    return v1.mk_sum(a, Counter({v1.mk_prod(a, Counter({H: 1, K: 1, cd: 1})): 1, mono(a, {6: 1}, perm): 1})), \
        dict(H=H, K=K, cd=cd)
