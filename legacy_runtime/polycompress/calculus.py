"""Exact compound FACTOR/REUSE actions over exposed contexts."""

from collections import Counter
from dataclasses import dataclass

from . import poly as p
from .handles import Arena, Mono, Sum, Prod, digest, items, mk_mono, mk_sum, mk_prod, poly
from .state import StateInfo, state_info, match


@dataclass(frozen=True)
class Task:
    target: p.Poly
    n_vars: int
    c_gen: int
    meta: dict


@dataclass(frozen=True)
class Action:
    kind: str
    ctx: int
    region: tuple[tuple[int,int], ...]
    arg: int
    src: tuple[tuple[int,int], ...] | None = None


def region_key(arena: Arena, region: tuple[tuple[int,int], ...]) -> tuple:
    """Return the allocation-independent stable key of an exposed region."""
    return tuple(sorted((digest(arena,h),c) for h,c in region))


def action_key(arena: Arena, a: Action) -> tuple:
    """Return the allocation-independent canonical action order built from structural digests."""
    src = () if a.src is None else region_key(arena,a.src)
    return a.kind,digest(arena,a.ctx),region_key(arena,a.region),b'' if a.arg < 0 else digest(arena,a.arg),src


def group(arena: Arena, region: tuple[tuple[int,int], ...]) -> int:
    """Materialize an exposed additive multiset."""
    return mk_sum(arena,Counter(dict(region)))


def poly_region(arena: Arena, region: tuple[tuple[int,int], ...]) -> p.Poly:
    """Evaluate a region without materializing its root."""
    result: p.Poly = {}
    for h,c in region:
        result = p.add(result,p.scale(poly(arena,h),c))
    return result


def strip(arena: Arena, h: int, H: int) -> int | None:
    """Remove one exposed factor without crossing a Sum boundary."""
    node,factor = arena.handles[h],arena.handles[H]
    if isinstance(factor,Mono):
        if isinstance(node,Mono):
            return mk_mono(arena,p.mono_div(node.m,factor.m)) if p.divides(factor.m,node.m) else None
        if not isinstance(node,Prod):
            return None
        q = next((v for v,c in node.factors if isinstance(arena.handles[v],Mono)),None)
        if q is None or not p.divides(factor.m,arena.handles[q].m):
            return None
        factors = Counter(dict(node.factors))
        del factors[q]
        factors[mk_mono(arena,p.mono_div(arena.handles[q].m,factor.m))] += 1
    else:
        if not isinstance(node,Prod) or H not in dict(node.factors):
            return None
        factors = Counter(dict(node.factors))
        factors[H] -= 1
    return mk_prod(arena,factors)


def apply(arena: Arena, task: Task, info: StateInfo, root: int, a: Action) -> int | None:
    """Apply once per root/action without verifying the successor polynomial."""
    return _probe(arena,info,root,a)


def _probe(arena: Arena, info: StateInfo, root: int, a: Action) -> int | None:
    key = root,a
    if key not in arena.apply_memo:
        arena.apply_count += 1
        arena.apply_memo[key] = _apply(arena,info,root,a)
    return arena.apply_memo[key]


def _apply(arena: Arena, info: StateInfo, root: int, a: Action) -> int | None:
    assert a.ctx in info.contexts, 'action context is reachable'
    counts = Counter(dict(items(arena.handles[a.ctx])))
    region = Counter(dict(a.region))
    assert region and all(0 < c <= counts[h] for h,c in region.items())
    if a.kind == 'FACTOR':
        rem: Counter[int] = Counter()
        for h,c in a.region:
            stripped = strip(arena,h,a.arg)
            if stripped is None:
                return None
            rem[stripped] += c
        k = match(info,group(arena,tuple(sorted(rem.items()))))
        if p.degree(poly(arena,k)) == 0:
            return None
        factors = Counter([a.arg,k])
        replacement = mk_prod(arena,factors)
        counts.subtract(region)
        counts[replacement] += 1
    elif a.kind == 'REUSE' and a.src is None:
        v = a.arg
        if v == a.ctx or a.ctx in state_info(arena,v).reach_h:
            return None
        if not p.equal(poly_region(arena,a.region),poly(arena,v)) or v == group(arena,a.region):
            return None
        counts.subtract(region)
        counts[v] += 1
    elif a.kind == 'REUSE':
        source = Counter(dict(a.src))
        if any(c+region[h] > counts[h] for h,c in source.items()):
            return None
        if not p.equal(poly_region(arena,a.src),poly_region(arena,a.region)):
            return None
        g = match(info,group(arena,a.src))
        counts.subtract(source)
        counts.subtract(region)
        counts[g] += 2
    else:
        assert False, 'unknown action kind'
    ctx2 = mk_sum(arena,counts)
    if ctx2 == a.ctx:
        return None
    root2 = substitute(arena,root,a.ctx,ctx2)
    assert root2 != root, 'changed context must change the root'
    return root2


def verify(arena: Arena, task: Task, root: int) -> None:
    """Assert exact task equality once per admitted root."""
    if root not in arena.verified:
        assert p.equal(poly(arena,root),task.target), 'calculus changed the target polynomial'
        arena.verified.add(root)
        arena.verifier_count += 1


def substitute(arena: Arena, root: int, old: int, new: int) -> int:
    """Replace all references using a recursion memo local to this call."""
    memo: dict[int,int] = {}

    def visit(h: int) -> int:
        """Rebuild one ancestor through the per-call substitution memo."""
        if h == old:
            return new
        if old not in state_info(arena,h).reach_h:
            return h
        if h not in memo:
            children: Counter[int] = Counter()
            for child,c in items(arena.handles[h]):
                children[visit(child)] += c
            make = mk_sum if isinstance(arena.handles[h],Sum) else mk_prod
            memo[h] = make(arena,children)
        return memo[h]

    return visit(root)


def action_record(a: Action, index: dict[int,int]) -> list:
    """Encode an action over serialized node positions (arg -1 and src None are kept)."""
    region = sorted([index[h],c] for h,c in a.region)
    src = None if a.src is None else sorted([index[h],c] for h,c in a.src)
    return [a.kind,index[a.ctx],region,-1 if a.arg < 0 else index[a.arg],src]


def action_from_record(record: list, ids: list[int]) -> Action:
    """Decode an action over rebuilt handle ids, restoring the id-sorted region order."""
    kind,ctx,region,arg,src = record
    return Action(kind,ids[ctx],tuple(sorted((ids[i],c) for i,c in region)),-1 if arg < 0 else ids[arg],
                  None if src is None else tuple(sorted((ids[i],c) for i,c in src)))
