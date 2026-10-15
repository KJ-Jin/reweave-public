"""Exact certification of parameterized relations, and the structural bounds on admissible relations.

Certification: parameter i -> fresh variable x_i; source and target are evaluated with the exact polynomial
arithmetic of the runtime; a relation is certified iff the two polynomials are equal. Representation overflow fails
closed. A certified relation is an identity of the free commutative semiring, so every instance of it is exact.
"""

from . import v1
from .rewrite_schema import NV, Schema, key, max_exp, region_holes


def _eval(p) -> dict:
    if p[0] == 'M':
        exps = [0]*NV
        for i, e in p[1]:
            exps[i] += e
        return {v1.pack(tuple(exps)): 1}
    if p[0] == 'S':
        acc: dict = {}
        for c, n in p[1]:
            acc = v1.poly_add(acc, v1.poly_scale(_eval(c), n))
        return acc
    acc = {v1.pack((0,)*NV): 1}          # inherited poly.one()
    for c, n in p[1]:
        q = _eval(c)
        for _ in range(n):
            acc = v1.poly_mul(acc, q)
    return acc


def eval_region(r) -> dict:
    acc: dict = {}
    for c, n in r:
        acc = v1.poly_add(acc, v1.poly_scale(_eval(c), n))
    return acc


def certify(schema: Schema) -> tuple[bool, str, dict]:
    """Formal identity with hole i -> x_i under inherited poly; fails closed on representation overflow."""
    try:
        src, tgt = eval_region(schema.source), eval_region(schema.target)
    except AssertionError:
        return False, 'certification_overflow', {}
    ok = v1.poly_equal(src, tgt)
    return ok, ('certified' if ok else 'certification_fail'), dict(source_poly=sorted(src.items()), target_poly=sorted(tgt.items()))


def _ops(p) -> int:
    return 0 if p[0] == 'M' else 1+sum(_ops(c) for c, n in p[1])


def _unique_ops(p, acc: set) -> None:
    if p[0] != 'M':
        acc.add(key(p))
        for c, n in p[1]:
            _unique_ops(c, acc)


def _depth(p) -> int:
    return 1 if p[0] == 'M' else 1+max(_depth(c) for c, n in p[1])


def _max_arity(p) -> int:
    if p[0] == 'M':
        return 0
    return max([len(p[1])]+[_max_arity(c) for c, n in p[1]])


def target_unique_ops(schema: Schema) -> int:
    uniq: set = set()
    for p, c in schema.target:
        _unique_ops(p, uniq)
    return len(uniq)


def bounds_violation(schema: Schema, cfg) -> str | None:
    n = len(schema.types)
    if not 1 <= n <= cfg.n_holes_max:
        return 'n_holes'
    if region_holes(schema.source) != frozenset(range(n)) or region_holes(schema.target) != frozenset(range(n)):
        return 'holes_mismatch'
    if schema.source == schema.target:
        return 'identity'
    if len(schema.source) > cfg.source_terms_max:
        return 'source_terms'
    if sum(c for p, c in schema.source) > cfg.source_multiplicity_max:
        return 'source_multiplicity'
    if sum(_ops(p) for p, c in schema.source) > cfg.source_operators_max:
        return 'source_operators'
    uniq: set = set()
    for p, c in schema.target:
        _unique_ops(p, uniq)
    if not 1 <= len(uniq) <= cfg.target_operators_max:
        return 'target_operators'
    allp = [p for p, c in schema.source+schema.target]
    if max(_depth(p) for p in allp) > cfg.pattern_depth_max:
        return 'pattern_depth'
    if max(max_exp(p) for p in allp) > cfg.mon_exponent_max:
        return 'mon_exponent'
    if max(_max_arity(p) for p in allp) > cfg.node_arity_max:
        return 'node_arity'
    return None
