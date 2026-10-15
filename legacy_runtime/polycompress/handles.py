"""Interned handles, their concrete scalar-free binary gate DAGs, and structural serialization."""

from collections import Counter
from dataclasses import dataclass
import hashlib
from itertools import combinations_with_replacement

from . import poly as p


@dataclass(frozen=True)
class Mono:
    m: int


@dataclass(frozen=True)
class Sum:
    terms: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Prod:
    factors: tuple[tuple[int, int], ...]


Handle = Mono | Sum | Prod


@dataclass(frozen=True)
class Gate:
    op: str
    a: int = -1
    b: int = -1
    var: int = -1


def addchain(k: int) -> tuple[tuple[int, int], ...]:
    """Find a shortest increasing addition chain with lexicographic ties."""
    assert 1 <= k <= 128
    for bound in range(k):
        result = _chain((1,), k, bound)
        if result is not None:
            return result
    assert False, 'addition chain exists'


def _chain(values: tuple[int, ...], target: int, left: int) -> tuple[tuple[int, int], ...] | None:
    if target in values:
        return ()
    if left == 0:
        return None
    if max(values) * 2**left < target:
        return None
    candidates = []
    for a, b in combinations_with_replacement(values, 2):
        c = a+b
        if c <= values[-1] or c > target:
            continue
        candidates.append((c, a, b))
    for c, a, b in sorted(candidates):
        tail = _chain(values+(c,), target, left-1)
        if tail is not None:
            return ((a,b),)+tail
    return None


def mono_chain(m: int) -> tuple[tuple[int, int], ...]:
    """Find a shortest vector addition chain from the used variables."""
    if m == 0:
        return ()
    leaves = tuple(1 << (4*i) for i in sorted(p.mono_support(m)))
    # Free leaves need not precede the first operation in numeric order.
    for bound in range(p.mono_degree(m)):
        result = _vector_chain(leaves, m, bound, 0)
        if result is not None:
            return result
    assert False, 'vector addition chain exists'


def _vector_chain(values: tuple[int, ...], target: int, left: int,
                  last: int) -> tuple[tuple[int, int], ...] | None:
    if target in values:
        return ()
    if left == 0 or max(map(p.mono_degree,values))*2**left < p.mono_degree(target):
        return None
    candidates = []
    for a,b in combinations_with_replacement(sorted(values),2):
        c = a+b
        if c <= last or c in values:
            continue
        if any(x+y > z for x,y,z in zip(p.unpack(a),p.unpack(b),p.unpack(target))):
            continue
        candidates.append((c,a,b))
    for c,a,b in sorted(candidates):
        tail = _vector_chain(values+(c,),target,left-1,c)
        if tail is not None:
            return ((a,b),)+tail
    return None


def MONO_COST(m: int) -> int:
    """Return the exact private monomial realization cost."""
    return len(mono_chain(m))


class Arena:
    """Own handles, gates, and all per-task memos."""

    def __init__(self) -> None:
        from .config import CalcCfg
        self.calc_cfg = CalcCfg()
        self.handles: list[Handle] = []
        self.key2id: dict[tuple, int] = {}
        self.gates = [Gate('VAR',var=i) for i in range(8)]+[Gate('ONE')]
        self.var_ids = tuple(range(8))
        self.one_id = 8
        self.gate_root: list[int] = []
        self.poly_memo: dict[int, p.Poly] = {}
        self.fps: list[int] = []
        self.digests: list[bytes] = []
        self.constant: list[bool] = []
        self.gate_reach_memo: dict[int, frozenset[int]] = {}
        self.info_memo: dict = {}
        self.apply_memo: dict = {}
        self.verified: set[int] = set()
        self.apply_count = 0
        self.verifier_count = 0
        self.truncations: list[tuple[int,int]] = []

    def intern(self, h: Handle) -> int:
        """Intern a normalized handle and materialize it once."""
        key = ('M',h.m) if isinstance(h,Mono) else ('S' if isinstance(h,Sum) else 'P',items(h))
        if key in self.key2id:
            return self.key2id[key]
        hid = len(self.handles)
        self.key2id[key] = hid
        self.handles.append(h)
        self.gate_root.append(_materialize(self,h))
        if isinstance(h,Mono):
            value = p.fp_mono(h.m)
        elif isinstance(h,Sum):
            value = sum(c*self.fps[v] for v,c in h.terms) % p.P
        else:
            value = 1
            for v,c in h.factors:
                value = value*pow(self.fps[v],c,p.P) % p.P
        self.fps.append(value)
        self.digests.append(_digest(self,h))
        self.constant.append(h.m == 0 if isinstance(h,Mono) else all(self.constant[v] for v,c in items(h)))
        return hid


def _digest(arena: 'Arena', h: Handle) -> bytes:
    if isinstance(h,Mono):
        return hashlib.sha256(b'M'+h.m.to_bytes(4,'big')).digest()
    parts = sorted(arena.digests[v]+c.to_bytes(2,'big') for v,c in items(h))
    return hashlib.sha256((b'S' if isinstance(h,Sum) else b'P')+b''.join(parts)).digest()


def digest(arena: 'Arena', h: int) -> bytes:
    """Return the allocation-independent structural key of a handle."""
    return arena.digests[h]


def items(h: Handle) -> tuple[tuple[int,int], ...]:
    """Return exposed children with multiplicities."""
    return h.terms if isinstance(h,Sum) else h.factors if isinstance(h,Prod) else ()


def mk_mono(arena: Arena, m: int) -> int:
    """Intern a monomial."""
    assert m >= 0 and m < 2**32 and all(e <= 6 for e in p.unpack(m))
    return arena.intern(Mono(m))


def mk_sum(arena: Arena, terms: Counter[int]) -> int:
    """Normalize an exposed additive multiset without flattening."""
    assert all(isinstance(c,int) and c >= 0 for c in terms.values())
    pairs = tuple(sorted((h,c) for h,c in terms.items() if c))
    assert pairs, 'empty sum'
    if len(pairs) == 1 and pairs[0][1] == 1:
        return pairs[0][0]
    return arena.intern(Sum(pairs))


def mk_prod(arena: Arena, factors: Counter[int]) -> int:
    """Merge exposed monomials and preserve every other boundary."""
    assert all(isinstance(c,int) and c >= 0 for c in factors.values())
    rest: Counter[int] = Counter()
    m = 0
    for h,c in factors.items():
        if isinstance(arena.handles[h],Mono):
            for _ in range(c):
                m = p.mono_mul(m,arena.handles[h].m)
        elif c:
            assert not arena.constant[h], 'a nontrivial constant circuit may not be a MUL operand'
            rest[h] += c
    if not rest:
        return mk_mono(arena,m)
    if m:
        rest[mk_mono(arena,m)] += 1
    pairs = tuple(sorted(rest.items()))
    if len(pairs) == 1 and pairs[0][1] == 1:
        return pairs[0][0]
    return arena.intern(Prod(pairs))


def _gate(arena: Arena, op: str, a: int, b: int) -> int:
    arena.gates.append(Gate(op,a,b))
    return len(arena.gates)-1


def _materialize(arena: Arena, h: Handle) -> int:
    if isinstance(h,Mono):
        if h.m == 0:
            return arena.one_id
        roots = {1 << (4*i):arena.var_ids[i] for i in p.mono_support(h.m)}
        for a,b in mono_chain(h.m):
            roots[a+b] = _gate(arena,'MUL',roots[a],roots[b])
        return roots[h.m]
    op = 'ADD' if isinstance(h,Sum) else 'MUL'
    roots = []
    for child,count in items(h):
        multiples = {1:arena.gate_root[child]}
        for a,b in addchain(count):
            multiples[a+b] = _gate(arena,op,multiples[a],multiples[b])
        roots.append(multiples[count])
    result = roots[0]
    for other in roots[1:]:
        result = _gate(arena,op,result,other)
    return result


def poly(arena: Arena, h: int) -> p.Poly:
    """Expand lazily and memoize exact handle semantics."""
    if h not in arena.poly_memo:
        node = arena.handles[h]
        if isinstance(node,Mono):
            result = {node.m:1}
        elif isinstance(node,Sum):
            result = {}
            for child,count in node.terms:
                result = p.add(result,p.scale(poly(arena,child),count))
        else:
            result = p.one()
            for child,count in node.factors:
                for _ in range(count):
                    result = p.mul(result,poly(arena,child))
        arena.poly_memo[h] = result
    return arena.poly_memo[h]


def fp(arena: Arena, h: int) -> int:
    """Return the eager homomorphic equality prefilter."""
    return arena.fps[h]


def cost(arena: Arena, h: int) -> int:
    """Count unique reachable arithmetic gates."""
    if h not in arena.gate_reach_memo:
        seen: set[int] = set()
        todo = [arena.gate_root[h]]
        while todo:
            g = todo.pop()
            node = arena.gates[g]
            if g in seen or node.op in ('VAR','ONE'):
                continue
            seen.add(g)
            todo.extend((node.a,node.b))
        arena.gate_reach_memo[h] = frozenset(seen)
    return len(arena.gate_reach_memo[h])


def canonical_start(arena: Arena, f: p.Poly) -> int:
    """Build the canonical flat start in packed-monomial order."""
    return mk_sum(arena,Counter({mk_mono(arena,m):c for m,c in sorted(f.items())}))


def heights(arena: Arena, closure: set[int]) -> dict[int,int]:
    """Return the longest child chain below every handle of a closed set (0 for a monomial)."""
    result: dict[int,int] = {}
    for h in sorted(closure):
        children = items(arena.handles[h])
        result[h] = 0 if not children else 1+max(result[v] for v,c in children)
    return result


def serialize(arena: Arena, roots: list[int], with_digests: bool = True) -> tuple[list[list],dict[int,int]]:
    """Serialize the reachable closure of roots children-first in (height, digest) order; ids never appear."""
    closure: set[int] = set()
    todo = list(roots)
    while todo:
        h = todo.pop()
        if h not in closure:
            closure.add(h)
            todo.extend(v for v,c in items(arena.handles[h]))
    level = heights(arena,closure)
    order = sorted(closure,key=lambda h:(level[h],digest(arena,h)))
    index = {h:i for i,h in enumerate(order)}
    nodes = []
    for h in order:
        node = arena.handles[h]
        body = node.m if isinstance(node,Mono) else sorted([index[v],c] for v,c in items(node))
        kind = 'M' if isinstance(node,Mono) else 'S' if isinstance(node,Sum) else 'P'
        nodes.append([kind,body,digest(arena,h).hex()] if with_digests else [kind,body])
    return nodes,index


def deserialize(arena: Arena, nodes: list[list]) -> list[int]:
    """Rebuild serialized nodes through the guarded constructors and assert every recorded digest."""
    ids: list[int] = []
    for kind,body,*recorded in nodes:
        if kind == 'M':
            h = mk_mono(arena,body)
        else:
            children = Counter({ids[i]:c for i,c in body})
            h = mk_sum(arena,children) if kind == 'S' else mk_prod(arena,children)
        assert not recorded or digest(arena,h).hex() == recorded[0], 'serialized structure reproduces its digest'
        ids.append(h)
    return ids


def _own_standard(arena: Arena, h: int) -> int:
    """Count the standard-model gates one handle contributes; integer constants are free."""
    node = arena.handles[h]
    if isinstance(node,Mono):
        return MONO_COST(node.m)
    pairs = items(node)
    if isinstance(node,Sum):
        # A multiplicity is one multiplication by the free integer constant, not an addition chain.
        return sum(1 for v,c in pairs if c > 1)+len(pairs)-1
    return sum(len(addchain(c)) for v,c in pairs if c > 1)+len(pairs)-1


def reachable(arena: Arena, roots: list[int]) -> set[int]:
    """Return the handles reachable from roots, roots included."""
    closure: set[int] = set()
    todo = list(roots)
    while todo:
        h = todo.pop()
        if h not in closure:
            closure.add(h)
            todo.extend(v for v,c in items(arena.handles[h]))
    return closure


def standard_cost(arena: Arena, h: int) -> int:
    """Return the reporting-only standard arithmetic cost; search and every S decision ignore it."""
    return sum(_own_standard(arena,v) for v in reachable(arena,[h]))


def _atom(text: str, named: bool, wrapped: bool) -> str:
    """Parenthesize only a sum, or a compound child carrying a power or coefficient."""
    if named or not any(c in text for c in '+*'):
        return text
    return f'({text})' if ' + ' in text or wrapped else text


def _render(arena: Arena, h: int, names: dict[int,str]) -> str:
    node = arena.handles[h]
    if isinstance(node,Mono):
        return '1' if node.m == 0 else '*'.join(f'x{i}' if e == 1 else f'x{i}^{e}'
                                                for i,e in enumerate(p.unpack(node.m)) if e)
    parts = []
    for child,c in sorted(items(node),key=lambda pair:digest(arena,pair[0])):
        atom = _atom(names.get(child) or _render(arena,child,names),child in names,c > 1)
        parts.append(f'{c}*{atom}' if isinstance(node,Sum) and c > 1 else f'{atom}^{c}' if c > 1 else atom)
    return ' + '.join(parts) if isinstance(node,Sum) else '*'.join(parts)


def to_expr(arena: Arena, h: int) -> str:
    """Print a handle DAG readably, defining each shared non-leaf once; reporting only."""
    closure = reachable(arena,[h])
    level = heights(arena,closure)
    fanout: Counter[int] = Counter()
    for v in closure:
        for child,c in items(arena.handles[v]):
            fanout[child] += 1
    names,lines = {},[]
    for v in sorted(closure,key=lambda v:(level[v],digest(arena,v))):
        if v != h and fanout[v] > 1 and not isinstance(arena.handles[v],Mono):
            names[v] = f't{len(lines)}'
            lines.append(f'{names[v]} = {_render(arena,v,names)}')
    return '\n'.join(lines+[_render(arena,h,names)])
