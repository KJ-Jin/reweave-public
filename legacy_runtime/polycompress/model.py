"""Message-passing state encoder, candidate scorer and value head (specification section 8.1)."""

from dataclasses import dataclass, replace
import math

import numpy as np
import torch
from torch import nn

from .calculus import Action
from .handles import Arena, Mono, MONO_COST, Prod, Sum, cost, items, poly
from . import poly as p
from .state import StateInfo

NODE_DIM = 17
EDGE_DIM = 3
KINDS = ('FACTOR', 'REUSE_EXISTING', 'REUSE_REGION')


def action_kind(a: Action) -> int:
    """Return the candidate's kind index; the three forms differ in what `h_arg` means."""
    if a.kind == 'FACTOR':
        return 0
    return 1 if a.src is None else 2


def node_features(arena: Arena, h: int, fanout: int, is_root: bool) -> np.ndarray:
    """Build the declared 17-dimensional node input; Mono exponents are the only polynomial content."""
    node = arena.handles[h]
    value = poly(arena, h)
    row = np.zeros(NODE_DIM, dtype=np.float64)
    row[0 if isinstance(node, Mono) else 1 if isinstance(node, Sum) else 2] = 1.0
    row[3] = math.log1p(p.degree(value))
    row[4] = len(p.support(value))
    row[5] = math.log1p(cost(arena, h))
    row[6] = math.log1p(fanout)
    row[7] = float(is_root)
    if isinstance(node, Mono):
        row[8:16] = p.unpack(node.m)
    # row[16] is the declared spare zero.
    return row


def off_graph_mono(m: int) -> np.ndarray:
    """Embed a monomial factor that is not a reachable handle: fanout 0, not root, cost MONO_COST."""
    row = np.zeros(NODE_DIM, dtype=np.float64)
    row[0] = 1.0
    row[3] = math.log1p(p.mono_degree(m))
    row[4] = len(p.mono_support(m))
    row[5] = math.log1p(MONO_COST(m))
    row[8:16] = p.unpack(m)
    return row


@dataclass
class StateGraph:
    """One state's node table, child->parent edges and handle index."""
    nodes: np.ndarray
    edge_child: list[int]
    edge_parent: list[int]
    edge_attr: np.ndarray
    index: dict[int, int]
    root: int


def build_graph(arena: Arena, info: StateInfo, root: int) -> StateGraph:
    """Extract the state graph: nodes are reach_h, edges run child -> parent."""
    order = sorted(info.reach_h)
    index = {h: i for i, h in enumerate(order)}
    nodes = np.stack([node_features(arena, h, info.fanout[h], h == root) for h in order])
    child, parent, attr = [], [], []
    for h in order:
        node = arena.handles[h]
        term = isinstance(node, Sum)
        for value, count in items(node):
            child.append(index[value])
            parent.append(index[h])
            attr.append([math.log1p(count), float(term), float(not term)])
    return StateGraph(nodes, child, parent,
                      np.array(attr, dtype=np.float64).reshape(-1, EDGE_DIM), index, index[root])


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes)-2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class PolyModel(nn.Module):
    """The declared encoder, candidate scorer and value head; L = 0 is the flat-only control."""

    def __init__(self, layers: int = 3, hidden: int = 128) -> None:
        super().__init__()
        self.layers, self.hidden = layers, hidden
        self.input = _mlp([NODE_DIM, hidden, hidden])
        self.children_msg = nn.ModuleList(_mlp([hidden+EDGE_DIM, hidden, hidden]) for _ in range(layers))
        self.parents_msg = nn.ModuleList(_mlp([hidden+EDGE_DIM, hidden, hidden]) for _ in range(layers))
        self.update = nn.ModuleList(_mlp([3*hidden, hidden, hidden]) for _ in range(layers))
        self.scorer = _mlp([4*hidden+len(KINDS)+2, hidden, hidden, 1])
        self.value = _mlp([2*hidden, hidden, hidden, 1])

    def encode(self, nodes: torch.Tensor, child: torch.Tensor, parent: torch.Tensor,
               attr: torch.Tensor) -> torch.Tensor:
        """Run the input MLP and L message-passing layers; at L = 0 edges are never read."""
        h = self.input(nodes)
        for i in range(self.layers):
            up = torch.zeros_like(h)
            down = torch.zeros_like(h)
            if child.numel():
                up.index_add_(0, parent, self.children_msg[i](torch.cat([h[child], attr], dim=1)))
                down.index_add_(0, child, self.parents_msg[i](torch.cat([h[parent], attr], dim=1)))
            h = self.update[i](torch.cat([h, up, down], dim=1))
        return h


def pool(h: torch.Tensor, rows: list[int], counts: list[int]) -> torch.Tensor:
    """Return the declared count-weighted mean over a region or a context's items."""
    if not rows:
        return torch.zeros(h.shape[1], dtype=h.dtype, device=h.device)
    weights = torch.tensor(counts, dtype=h.dtype, device=h.device).unsqueeze(1)
    return (h[rows]*weights).sum(dim=0)/float(sum(counts))


@dataclass
class Batch:
    """A concatenation of state graphs plus their candidate lists with a state index per candidate."""
    nodes: torch.Tensor
    child: torch.Tensor
    parent: torch.Tensor
    attr: torch.Tensor
    roots: list[int]
    offsets: list[int]
    candidates: list[dict]
    state_of: list[int]
    extra: torch.Tensor


def to_device(batch: Batch, device: str) -> Batch:
    """Place one batch's tensors on the training device; the index lists stay plain Python."""
    if device == 'cpu':
        return batch
    return replace(batch, nodes=batch.nodes.to(device), child=batch.child.to(device),
                   parent=batch.parent.to(device), attr=batch.attr.to(device),
                   extra=batch.extra.to(device))


def candidate_rows(arena: Arena, info: StateInfo, graph: StateGraph, a: Action,
                   offset: int) -> tuple[dict, np.ndarray]:
    """Describe one candidate by node positions plus its declared scalar inputs."""
    ctx_items = items(arena.handles[a.ctx])
    total = sum(c for h, c in a.region)
    coverage = total/sum(c for h, c in ctx_items)
    entry = dict(ctx=graph.index[a.ctx]+offset,
                 ctx_rows=[graph.index[h]+offset for h, c in ctx_items],
                 ctx_counts=[c for h, c in ctx_items],
                 region_rows=[graph.index[h]+offset for h, c in a.region],
                 region_counts=[c for h, c in a.region], kind=action_kind(a))
    if a.kind == 'FACTOR':
        node = arena.handles[a.arg]
        if a.arg in graph.index:
            entry.update(arg_rows=[graph.index[a.arg]+offset], arg_counts=[1], arg_extra=None)
        else:
            assert isinstance(node, Mono), 'only a monomial factor can be off-graph'
            entry.update(arg_rows=[], arg_counts=[], arg_extra=off_graph_mono(node.m))
    elif a.src is None:
        entry.update(arg_rows=[graph.index[a.arg]+offset], arg_counts=[1], arg_extra=None)
    else:
        entry.update(arg_rows=[graph.index[h]+offset for h, c in a.src],
                     arg_counts=[c for h, c in a.src], arg_extra=None)
    return entry, np.array([total, coverage], dtype=np.float64)


def make_batch(states: list[tuple[Arena, StateInfo, int, list[Action]]]) -> Batch:
    """Concatenate state graphs with node-index offsets and their candidate lists."""
    nodes, child, parent, attr, roots, offsets = [], [], [], [], [], []
    candidates, state_of, extra = [], [], []
    offset = 0
    for index, (arena, info, root, actions) in enumerate(states):
        graph = build_graph(arena, info, root)
        nodes.append(graph.nodes)
        child.extend(v+offset for v in graph.edge_child)
        parent.extend(v+offset for v in graph.edge_parent)
        if len(graph.edge_attr):
            attr.append(graph.edge_attr)
        roots.append(graph.root+offset)
        offsets.append(offset)
        for a in actions:
            entry, scalars = candidate_rows(arena, info, graph, a, offset)
            candidates.append(entry)
            state_of.append(index)
            extra.append(scalars)
        offset += len(graph.nodes)
    return Batch(torch.tensor(np.concatenate(nodes), dtype=torch.float64),
                 torch.tensor(child, dtype=torch.long), torch.tensor(parent, dtype=torch.long),
                 torch.tensor(np.concatenate(attr) if attr else np.zeros((0, EDGE_DIM)), dtype=torch.float64),
                 roots, offsets, candidates, state_of,
                 torch.tensor(np.stack(extra) if extra else np.zeros((0, 2)), dtype=torch.float64))


def forward(model: PolyModel, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Return candidate logits and one value per state from a single encoder pass."""
    h = model.encode(batch.nodes, batch.child, batch.parent, batch.attr)
    rows = []
    for i, entry in enumerate(batch.candidates):
        if entry['arg_extra'] is None:
            arg = pool(h, entry['arg_rows'], entry['arg_counts'])
        else:
            arg = model.input(torch.tensor(entry['arg_extra'], dtype=h.dtype,
                                            device=h.device).unsqueeze(0)).squeeze(0)
        kind = torch.zeros(len(KINDS), dtype=h.dtype, device=h.device)
        kind[entry['kind']] = 1.0
        rows.append(torch.cat([h[entry['ctx']], pool(h, entry['ctx_rows'], entry['ctx_counts']),
                               pool(h, entry['region_rows'], entry['region_counts']), arg,
                               kind, batch.extra[i]]))
    logits = model.scorer(torch.stack(rows)).squeeze(1) if rows else torch.zeros(0, dtype=h.dtype,
                                                                                 device=h.device)
    values = []
    for index, root in enumerate(batch.roots):
        lo, hi = batch.offsets[index], batch.offsets[index+1] if index+1 < len(batch.offsets) else len(h)
        values.append(torch.cat([h[lo:hi].mean(dim=0), h[root]]))
    return logits, model.value(torch.stack(values)).squeeze(1)


def listwise_loss(logits: torch.Tensor, state_of: list[int], positives: dict[int, int]) -> torch.Tensor:
    """Cross-entropy over each state's own candidate slice."""
    total = torch.zeros((), dtype=logits.dtype, device=logits.device)
    n = 0
    for state, positive in sorted(positives.items()):
        rows = [i for i, s in enumerate(state_of) if s == state]
        if len(rows) < 2:
            continue
        slice_ = logits[torch.tensor(rows, dtype=torch.long)]
        total = total+nn.functional.cross_entropy(slice_.unsqueeze(0),
                                                  torch.tensor([positive], dtype=torch.long,
                                                               device=logits.device))
        n += 1
    return total/max(n, 1)


def new_model(layers: int = 3, hidden: int = 128, seed: int = 0) -> PolyModel:
    """Construct a model with an explicit generator; global RNG state is never touched."""
    generator = torch.Generator().manual_seed(seed)
    # nn.Linear seeds its own weights from the global RNG on construction. Every parameter is
    # overwritten from `generator` below, so the model is unchanged either way, but section 0.6
    # forbids touching global RNG state at all: restore what construction consumed.
    entropy = torch.get_rng_state()
    model = PolyModel(layers, hidden).double()
    torch.set_rng_state(entropy)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() > 1:
                bound = 1.0/math.sqrt(parameter.shape[1])
                parameter.copy_(torch.empty_like(parameter).uniform_(-bound, bound, generator=generator))
            else:
                parameter.zero_()
    return model
