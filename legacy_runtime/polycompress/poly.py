"""Exact positive sparse polynomials over packed monomials."""

from functools import lru_cache
from itertools import permutations
import random

MAX_VARS = 8
H = 0x8888_8888
P = 2**61 - 1
_rng = random.Random(20260905)
V = [_rng.randrange(1, P) for _ in range(MAX_VARS)]
assert len(set(V)) == MAX_VARS
Poly = dict[int, int]

# Diagnostic guard (spec section 17.B): canon is expensive enough that calling it from inside a
# timed exhaustive reference search could consume that search's deadline and change its status.
TIMED_EXHAUSTIVE_ACTIVE = False


def timed_exhaustive(active: bool) -> None:
    """Open or close the guard that forbids canon while a timed exhaustive search owns the deadline."""
    global TIMED_EXHAUSTIVE_ACTIVE
    assert TIMED_EXHAUSTIVE_ACTIVE != active, 'the timed-exhaustive guard never nests or double-clears'
    TIMED_EXHAUSTIVE_ACTIVE = active


def pack(exps: tuple[int, ...]) -> int:
    """Pack up to eight exponent fields."""
    assert len(exps) <= MAX_VARS and all(0 <= e <= 6 for e in exps)
    return sum(e << (4*i) for i, e in enumerate(exps))


def unpack(m: int) -> tuple[int, ...]:
    """Return all eight exponent fields."""
    return tuple((m >> (4*i)) & 15 for i in range(MAX_VARS))


def one() -> Poly:
    """Return literal one."""
    return {0: 1}


def mono_mul(a: int, b: int) -> int:
    """Multiply packed monomials without field carries."""
    assert all(x+y < 8 for x, y in zip(unpack(a), unpack(b)))
    return a+b


def divides(a: int, b: int) -> bool:
    """Test fieldwise divisibility."""
    return ((b | H)-a) & H == H


def mono_div(b: int, a: int) -> int:
    """Divide a known multiple by a monomial."""
    assert divides(a, b)
    return b-a


def mono_gcd(a: int, b: int) -> int:
    """Return the fieldwise minimum."""
    return pack(tuple(min(x, y) for x, y in zip(unpack(a), unpack(b))))


def mono_degree(m: int) -> int:
    """Return total monomial degree."""
    return sum(unpack(m))


def mono_support(m: int) -> frozenset[int]:
    """Return indices of used variables."""
    return frozenset(i for i, e in enumerate(unpack(m)) if e)


def add(p: Poly, q: Poly) -> Poly:
    """Add positive sparse polynomials."""
    result = p.copy()
    for m, c in q.items():
        result[m] = result.get(m, 0)+c
    return result


def mul(p: Poly, q: Poly) -> Poly:
    """Multiply by the declared double loop."""
    result: Poly = {}
    for a, c in p.items():
        for b, d in q.items():
            m = mono_mul(a, b)
            result[m] = result.get(m, 0)+c*d
    return result


def scale(p: Poly, c: int) -> Poly:
    """Scale coefficients by a nonnegative integer."""
    assert c >= 0
    return {m: n*c for m, n in p.items()} if c else {}


def equal(p: Poly, q: Poly) -> bool:
    """Confirm exact equality."""
    return p == q


def degree(p: Poly) -> int:
    """Return maximum total degree."""
    return max(map(mono_degree, p), default=0)


def support(p: Poly) -> frozenset[int]:
    """Return all used variable indices."""
    return frozenset().union(*(mono_support(m) for m in p))


@lru_cache(maxsize=None)
def fp_mono(m: int) -> int:
    """Evaluate one monomial at the fixed point."""
    result = 1
    for v, e in zip(V, unpack(m)):
        result = result*pow(v, e, P) % P
    return result


def fp(p: Poly) -> int:
    """Return an equality bucket, never an equality certificate."""
    return sum(c*fp_mono(m) for m, c in p.items()) % P


def canon(p: Poly) -> tuple[tuple[int, int], ...]:
    """Minimize the polynomial over every variable renaming."""
    assert not TIMED_EXHAUSTIVE_ACTIVE, 'canon may not run inside a timed exhaustive search'
    used = sorted(support(p))
    terms = [(unpack(m), c) for m, c in p.items()]
    return min(tuple(sorted((sum(e[i] << (4*j) for i, j in zip(used, perm)), c)
                            for e, c in terms)) for perm in permutations(range(len(used))))
