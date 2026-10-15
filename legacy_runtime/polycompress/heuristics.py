"""Immediate gain, gate-cost ranking, and the twelve declared hand features."""

import numpy as np

from collections import Counter

from .calculus import Action, Task, apply, _probe, group, strip
from .handles import Arena, Mono, Sum, Prod, cost, items, poly
from .poly import degree
from .state import StateInfo, match


class ImmediateGain:
    """Probe every candidate using exact concrete gate cost."""

    def __init__(self, task: Task) -> None:
        self.task = task

    def score(self, arena: Arena, info: StateInfo, root: int, actions: list[Action]) -> list[float]:
        """Return exact gains; invalid actions rank last."""
        result = []
        for action in actions:
            successor = apply(arena,self.task,info,root,action)
            result.append(float('-inf') if successor is None else float(cost(arena,root)-cost(arena,successor)))
        return result


class CostRanker:
    def rank(self, arena: Arena, info: StateInfo, root: int) -> float:
        """Keep the cheapest concrete gate DAGs."""
        return float(cost(arena,root))


def features(arena: Arena, info: StateInfo, root: int, a: Action) -> np.ndarray:
    """Compute the twelve pre-action features without probing."""
    context = items(arena.handles[a.ctx])
    total = sum(c for h,c in a.region)
    result = np.array([
        a.kind == 'FACTOR', total, len(a.region), total/sum(c for h,c in context),
        a.arg >= 0 and isinstance(arena.handles[a.arg],Mono),
        degree(poly(arena,a.arg)) if a.arg >= 0 else 0,
        a.arg in info.reach_h, a.kind == 'REUSE' and a.src is None,
        sum(info.fanout[h] for h,c in a.region)/len(a.region),
        sum(isinstance(arena.handles[h],Prod) for h,c in a.region)/len(a.region),
        len(context), info.depth[a.ctx],
    ],dtype=np.float64)
    if arena.calc_cfg.consequence_visible:
        return np.concatenate((result,_consequences(arena,info,root,a)))
    return result


def _consequences(arena: Arena, info: StateInfo, root: int, a: Action) -> np.ndarray:
    source = None
    if a.kind == 'FACTOR':
        remainders = Counter()
        for h,c in a.region:
            stripped = strip(arena,h,a.arg)
            if stripped is None:
                break
            remainders[stripped] += c
        else:
            source = match(info,group(arena,tuple(sorted(remainders.items()))))
    elif a.src is None:
        source = a.arg
    else:
        source = match(info,group(arena,a.src))
    successor = _probe(arena,info,root,a)
    size = 0 if source is None else len(arena.handles[source].terms) if isinstance(arena.handles[source],Sum) else 1
    gain = float('-inf') if successor is None else cost(arena,root)-cost(arena,successor)
    return np.array([size,source in info.reach_h,gain],dtype=np.float64)


def standardize(rows: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Return the declared feature statistics; a zero-variance feature keeps std 1."""
    mean = rows.mean(axis=0)
    std = rows.std(axis=0)
    return mean,np.where(std > 0,std,1.0)


class HandLinear:
    """Strong hand-featured same-candidate control over the twelve declared pre-action features."""

    def __init__(self, w: np.ndarray, b: float, mean: np.ndarray, std: np.ndarray) -> None:
        assert w.shape == mean.shape == std.shape, 'weights and statistics share the feature layout'
        self.w,self.b,self.mean,self.std = w,float(b),mean,std

    def rows(self, arena: Arena, info: StateInfo, root: int, actions: list[Action]) -> np.ndarray:
        """Compute the declared features for every candidate without probing."""
        return np.stack([features(arena,info,root,a) for a in actions]) if actions else np.zeros((0,len(self.w)))

    def score(self, arena: Arena, info: StateInfo, root: int, actions: list[Action]) -> list[float]:
        """Score candidates from the standardized hand features alone; no embeddings are read."""
        rows = self.rows(arena,info,root,actions)
        return list((rows-self.mean)/self.std @ self.w+self.b) if len(rows) else []


class NeuralScorer:
    """Score candidates with the learned message-passing model (section 8.1)."""

    def __init__(self, model) -> None:
        self.model = model

    def score(self, arena: Arena, info: StateInfo, root: int, actions: list[Action]) -> list[float]:
        """Run one encoder pass over this state and return one logit per candidate."""
        from .model import forward, make_batch
        if not actions:
            return []
        import torch
        with torch.no_grad():
            logits,values = forward(self.model,make_batch([(arena,info,root,actions)]))
        return [float(x) for x in logits]


class ValueRanker:
    """Keep the states the learned value head predicts will finish cheapest."""

    def __init__(self, model) -> None:
        self.model = model

    def rank(self, arena: Arena, info: StateInfo, root: int) -> float:
        """Return the predicted final-over-start cost ratio; lower first."""
        from .model import forward, make_batch
        import torch
        with torch.no_grad():
            logits,values = forward(self.model,make_batch([(arena,info,root,[])]))
        return float(values[0])
