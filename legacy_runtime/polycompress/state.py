"""Lazy per-root exposed context and reachability information."""

from collections import Counter, deque
from dataclasses import dataclass

from .handles import Arena, Sum, digest, fp, items, poly
from .poly import equal


@dataclass
class StateInfo:
    arena: Arena
    reach_h: frozenset[int]
    contexts: list[int]
    fanout: Counter[int]
    depth: dict[int,int]
    fp_index: dict[int,list[int]]


def state_info(arena: Arena, root: int) -> StateInfo:
    """Compute breadth-first depths and count each parent's edges once."""
    if root in arena.info_memo:
        return arena.info_memo[root]
    depth = {root:0}
    queue = deque([root])
    fanout: Counter[int] = Counter()
    while queue:
        parent = queue.popleft()
        for child,count in items(arena.handles[parent]):
            fanout[child] += count
            if child not in depth:
                depth[child] = depth[parent]+1
                queue.append(child)
    contexts, index = [], {}
    for h in sorted(depth):
        if isinstance(arena.handles[h],Sum):
            contexts.append(h)
        index.setdefault(fp(arena,h),[]).append(h)
    info = StateInfo(arena,frozenset(depth),contexts,fanout,depth,index)
    arena.info_memo[root] = info
    return info


def match(info: StateInfo, h: int) -> int:
    """Select the reachable exactly equal handle with the smallest structural digest."""
    if h in info.reach_h:
        return h
    equals = [v for v in info.fp_index.get(fp(info.arena,h),[]) if equal(poly(info.arena,v),poly(info.arena,h))]
    return min(equals,key=lambda v:digest(info.arena,v)) if equals else h
