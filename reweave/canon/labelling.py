"""Canonical labelling and relabelling utilities (no search, no policy, no runtime state).

Used by semantic interning, semantic family identity and symmetry-safe identity: variable relabelling of exponent
vectors and polynomials, permutation algebra, automorphism groups, and the canonical polynomial frame / key.
"""

from __future__ import annotations

import hashlib
import itertools
from collections import Counter
from types import SimpleNamespace

from .. import v1

NV = 8
IDENT = tuple(range(NV))


# ---------------------------------------------------------------- permutations and relabelling (c7_1_equivariant @ bc5600a)

def relabel_vec(e, p) -> tuple:
    out = [0]*NV
    for i, x in enumerate(e):
        if x:
            out[p[i]] = x
    return tuple(out)


def relabel_m(m: int, p) -> int:
    return v1.pack(relabel_vec(v1.unpack(m), p))


def compose(p, q) -> tuple:
    """p ∘ q: apply q, then p (relabel(relabel(e, q), p) == relabel(e, compose(p, q)))."""
    return tuple(p[q[i]] for i in range(NV))


def inverse(p) -> tuple:
    inv = [0]*NV
    for i, j in enumerate(p):
        inv[j] = i
    return tuple(inv)


def rdigest(arena, h: int, p, memo: dict) -> bytes:
    """V1 structural digest of handle h relabelled by p, without interning (replica of V1 handles._digest)."""
    d = memo.get(h)
    if d is None:
        node = arena.handles[h]
        if isinstance(node, v1.Mono):
            d = hashlib.sha256(b'M'+relabel_m(node.m, p).to_bytes(4, 'big')).digest()
        else:
            parts = sorted(rdigest(arena, c, p, memo)+n.to_bytes(2, 'big') for c, n in v1.items(node))
            d = hashlib.sha256((b'S' if isinstance(node, v1.Sum) else b'P')+b''.join(parts)).digest()
        memo[h] = d
    return d


def relabel_into(src, h: int, p, dst, memo: dict) -> int:
    """Intern the structure of handle h (arena src) relabelled by p into arena dst."""
    r = memo.get(h)
    if r is None:
        node = src.handles[h]
        if isinstance(node, v1.Mono):
            r = v1.mk_mono(dst, relabel_m(node.m, p))
        else:
            kids = Counter()
            for c, n in v1.items(node):
                kids[relabel_into(src, c, p, dst, memo)] += n
            r = (v1.mk_sum if isinstance(node, v1.Sum) else v1.mk_prod)(dst, kids)
        memo[h] = r
    return r


def automorphisms(T: dict, n: int) -> list:
    """Every permutation of canonical positions 0..n-1 (identity elsewhere) fixing T (exponent vector -> coefficient)."""
    sig = {i: tuple(sorted((e[i], c, sum(e)) for e, c in T.items() if e[i])) for i in range(n)}
    blocks = {}
    for i in range(n):
        blocks.setdefault(sig[i], []).append(i)
    blist = [blocks[k] for k in sorted(blocks)]
    out = []
    for choice in itertools.product(*(itertools.permutations(b) for b in blist)):
        p = list(IDENT)
        for b, img in zip(blist, choice):
            for i, j in zip(b, img):
                p[i] = j
        p = tuple(p)
        if all(T.get(relabel_vec(e, p)) == c for e, c in T.items()):
            out.append(p)
    return sorted(out)


def group_closure(gens: list) -> set:
    span, frontier = {IDENT}, [IDENT]
    while frontier:
        x = frontier.pop()
        for g in gens:
            y = compose(g, x)
            if y not in span:
                span.add(y)
                frontier.append(y)
    return span


def generators(group: list) -> list:
    gens, span = [], {IDENT}
    for g in group:
        if g not in span:
            gens.append(g)
            span = group_closure(gens)
    assert span == set(group), 'generators must span the automorphism group'
    return gens


class _UF:
    def __init__(self) -> None:
        self.p: list = []

    def add(self) -> int:
        self.p.append(len(self.p))
        return len(self.p)-1

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i: int, j: int) -> None:
        a, b = self.find(i), self.find(j)
        if a != b:
            self.p[max(a, b)] = min(a, b)




# ---------------------------------------------------------------- canonical labelling of one polynomial

_W = [7**(NV-1-j)*(1 << 16) for j in range(NV)]


def fast_labelling(target: dict):
    """(π in P1 format, list of all minimizing images).

    Identical to `c7_policy.canonical_labelling`: the lexicographically smallest permutation minimizing
    sorted((relabel_vec(e, π), c)). The key tuple (vector, c) is encoded as sum_j v_j·7^(7-j)·2^16 + c. That encoding
    is injective and order-preserving because exponents are ≤ 6 and coefficients < 2^16.
    """
    used = sorted({i for m in target for i, x in enumerate(v1.unpack(m)) if x})
    n = len(used)
    mons = [(v1.unpack(m), c) for m, c in target.items()]
    assert all(max(e) <= 6 for e, c in mons) and all(0 < c < (1 << 16) for e, c in mons)
    base = [c for e, c in mons]
    T = [[[e[used[k]]*_W[j] for e, c in mons] for j in range(n)] for k in range(n)]
    best, best_img, mins = None, None, []
    for img in itertools.permutations(range(n)):
        vals = sorted(map(sum, zip(base, *[T[k][img[k]] for k in range(n)])))
        if best is None or vals < best:
            best, best_img, mins = vals, img, [img]
        elif vals == best:
            mins.append(img)
    perm = list(IDENT)
    for k, i in enumerate(used):
        perm[i] = best_img[k]
    return tuple(perm), used, mins


def canonical_labelling(target: dict) -> tuple:
    """π: task variable i -> canonical position π[i]. Lexicographically smallest minimizer of the relabelled target."""
    used = sorted({i for m in target for i, x in enumerate(v1.unpack(m)) if x})
    best = None
    for img in itertools.permutations(range(len(used))):
        perm = list(range(NV))
        for k, i in enumerate(used):
            perm[i] = img[k]
        key = tuple(sorted((relabel_vec(v1.unpack(m), perm), c) for m, c in target.items()))
        cand = (key, tuple(perm))
        if best is None or cand < best:
            best = cand
    return best[1] if best else tuple(range(NV))



# the copied bodies below reference these module aliases exactly as in their source module
O = SimpleNamespace(NV=NV, IDENT=IDENT, relabel_vec=relabel_vec, relabel_m=relabel_m, compose=compose, inverse=inverse,
                    rdigest=rdigest, relabel_into=relabel_into, automorphisms=automorphisms, generators=generators,
                    group_closure=group_closure, _UF=_UF)
Qm = SimpleNamespace(fast_labelling=fast_labelling)


# ---------------------------------------------------------------- renaming-invariant polynomial key (c7_2i_intern @ 0bf1da7)

def poly_frame(poly: dict):
    """(pi, T, G, frames, n_used) of an exact polynomial: the C7.1b TaskFrameQ construction applied to ⟦h⟧."""
    pi1, used, mins = Qm.fast_labelling(poly)
    full = list(IDENT)
    taken = {pi1[i] for i in used}
    for i in used:
        full[i] = pi1[i]
    for i, j in zip([i for i in range(NV) if i not in used], [j for j in range(NV) if j not in taken]):
        full[i] = j
    pi = tuple(full)
    T = tuple(sorted((O.relabel_vec(v1.unpack(m), pi), c) for m, c in poly.items()))
    G = set()
    for img in mins:
        u = list(IDENT)
        for k, i in enumerate(used):
            u[pi[i]] = img[k]
        G.add(tuple(u))
    G = sorted(G)
    return pi, T, G, [O.compose(u, pi) for u in G], len(used)


def key_of(T) -> str:
    return hashlib.sha256(repr(T).encode()).hexdigest()
