"""Canonical exposed-region enumeration, stable pair orientation, and usefulness-blind truncation."""

from collections import Counter
import hashlib
from itertools import combinations, product
from typing import TYPE_CHECKING, Iterator

from .calculus import Action, action_key, poly_region, region_key
from .handles import Arena, Mono, Prod, digest, fp, items, mk_mono, poly
from . import poly as p
from .state import StateInfo, state_info

if TYPE_CHECKING:
    from .config import CalcCfg


def regions(pairs: tuple[tuple[int,int], ...], r_max: int,
            partial: bool) -> Iterator[tuple[tuple[int,int], ...]]:
    """Yield every permitted subset and occurrence allocation."""
    for k in range(1,min(r_max,len(pairs))+1):
        for subset in combinations(pairs,k):
            counts = [range(1,c+1) if partial else (c,) for h,c in subset]
            for vector in product(*counts):
                yield tuple((h,c) for (h,n),c in zip(subset,vector))


def exposed_mono(arena: Arena, h: int) -> int:
    """Read the single exposed monomial factor, or the unit."""
    node = arena.handles[h]
    if isinstance(node,Mono):
        return node.m
    if isinstance(node,Prod):
        return next((arena.handles[v].m for v,c in node.factors if isinstance(arena.handles[v],Mono)),0)
    return 0


def factors(arena: Arena, region: tuple[tuple[int,int], ...]) -> list[int]:
    """Enumerate all nonunit common exposed factors."""
    common = exposed_mono(arena,region[0][0])
    shared: set[int] | None = None
    for h,c in region:
        common = p.mono_gcd(common,exposed_mono(arena,h))
        node = arena.handles[h]
        here = {v for v,n in node.factors if not isinstance(arena.handles[v],Mono)} if isinstance(node,Prod) else set()
        shared = here if shared is None else shared & here
    monos = [p.pack(exps) for exps in product(*(range(e+1) for e in p.unpack(common)))]
    return [mk_mono(arena,m) for m in sorted(monos) if m]+sorted(shared or ())


def sampling_key(arena: Arena, root: int, a: Action) -> bytes:
    """Derive the truncation key of an action from the stable state and action keys."""
    return hashlib.sha256(digest(arena,root)+repr(action_key(arena,a)).encode()).digest()


def enumerate_actions(arena: Arena, info: StateInfo, root: int, cfg: 'CalcCfg') -> list[Action]:
    """Enumerate compound actions, order them structurally, and keep the smallest sampling keys at the cap."""
    actions = []
    for ctx in info.contexts:
        pairs = items(arena.handles[ctx])
        counts = dict(pairs)
        buckets: dict[int,list[tuple]] = {}
        for region in regions(pairs,cfg.r_max,cfg.allow_partial_multiplicity):
            total = sum(c for h,c in region)
            value = sum(c*fp(arena,h) for h,c in region) % p.P
            if total >= 2:
                actions.extend(Action('FACTOR',ctx,region,arg) for arg in factors(arena,region))
            for v in info.fp_index.get(value,[]):
                if v == ctx or ctx in state_info(arena,v).reach_h or region == ((v,1),):
                    continue
                if p.equal(poly_region(arena,region),poly(arena,v)):
                    actions.append(Action('REUSE',ctx,region,v))
            if total < 2:
                continue
            sources = buckets.setdefault(value,[])
            # Identical region descriptions can name disjoint occurrence allocations.
            for other in sources+[region]:
                combined = Counter(dict(other))+Counter(dict(region))
                if any(c > counts[h] for h,c in combined.items()):
                    continue
                if p.equal(poly_region(arena,other),poly_region(arena,region)):
                    # The smaller stable region key is grouped as the source, whatever was allocated first.
                    source,target = sorted((other,region),key=lambda r:region_key(arena,r))
                    actions.append(Action('REUSE',ctx,target,-1,source))
            sources.append(region)
    actions.sort(key=lambda a:action_key(arena,a))
    if len(actions) > cfg.n_max:
        arena.truncations.append((len(actions),cfg.n_max))
        kept = sorted(actions,key=lambda a:sampling_key(arena,root,a))[:cfg.n_max]
        return sorted(kept,key=lambda a:action_key(arena,a))
    return actions
