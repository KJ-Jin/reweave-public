"""The schema-agnostic action scorer.

The inherited state encoder, primitive scorer and value head are unchanged. Only the rewrite head differs from the
fixed-vocabulary `rewrite_model.RewriteModel`: the per-relation embedding and the monomial-only binding mean are
replaced by encodings of the relation's structure and of each parameter role:

    fixed:   [h(ctx); pooled h(region); schema_emb[relation id] (32); (unique ops, holes, log1p support);
              mean input-MLP(bound monomials)]                                       -> MLP(H, H, 1)
    generic: [h(ctx); pooled h(region); FAM(relation) (32); sum/max over roles ROLE_j (2H); d (6); concrete (15);
              kind (4)]                                                              -> MLP(H, H, 1)

A newly certified relation family is therefore scorable with no new embedding and no change of model shape.
"""

from __future__ import annotations

import hashlib
import math

import torch
from torch import nn

from .. import v1
from ..recipe import Recipe, RecipeModel, init_base_parameters
from ..rewrite_model import _mlp, graph_rows
from . import action_features as TF

T7_MODEL_VERSION = 't7-schema-agnostic-v1'
FAM_OUT = 32
KIND_EMB = 4
HEAD_PREFIXES = ('fam_mlp.', 'role_mlp.', 'kind_emb.', 'rw_head.')


class SchemaAgnosticModel(RecipeModel):
    def __init__(self, layers: int, hidden: int, layernorm: bool, residual: bool) -> None:
        super().__init__(layers, hidden, layernorm, residual)
        H = hidden
        self.fam_mlp = _mlp([TF.N_FAMILY, FAM_OUT, FAM_OUT])
        self.role_mlp = _mlp([TF.N_ROLE+H, H, H])
        self.kind_emb = nn.Embedding(len(TF.KINDS), KIND_EMB)
        self.rw_head = _mlp([2*H+FAM_OUT+2*H+TF.N_D+TF.N_CONCRETE+KIND_EMB, H, H, 1])
        self._last_h = None

    def encode(self, nodes, child, parent, attr):
        h = super().encode(nodes, child, parent, attr)
        self._last_h = h
        return h


def build_t7_model(recipe: Recipe, seed: int) -> SchemaAgnosticModel:
    """Inherited parameters initialized exactly as V1 `new_model(seed)`; the T7 head from its own generator."""
    entropy = torch.get_rng_state()
    model = SchemaAgnosticModel(recipe.layers, recipe.hidden, recipe.layernorm, recipe.residual).double()
    torch.set_rng_state(entropy)
    init_base_parameters(model, seed, ('norms.',)+HEAD_PREFIXES)
    generator = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f't7|{seed}'.encode()).digest()[:7], 'big'))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not name.startswith(HEAD_PREFIXES):
                continue
            if parameter.dim() > 1:
                bound = 1.0/math.sqrt(parameter.shape[1])
                parameter.copy_(torch.empty_like(parameter).uniform_(-bound, bound, generator=generator))
            else:
                parameter.zero_()
    return model


def family_vector(model: SchemaAgnosticModel, family: tuple) -> torch.Tensor:
    """The learned FAMILY STRUCTURE representation (a function of the relation payload only)."""
    return model.fam_mlp(torch.tensor(family, dtype=torch.float64))


def rewrite_logits(model: SchemaAgnosticModel, h: torch.Tensor, rows: dict, feats: list) -> torch.Tensor:
    """One logit per ActionFeatures, in order."""
    dt, H = h.dtype, h.shape[1]
    zero = torch.zeros(H, dtype=dt)
    role_in, owner, heads = [], [], []
    for i, f in enumerate(feats):
        for rf, node in zip(f.roles, f.role_nodes):
            if node[0] == 'h' and node[1] in rows:
                emb = h[rows[node[1]]]
            elif node[0] == 'm':
                emb = model.input(torch.tensor(v1.off_graph_mono(node[1]), dtype=dt).unsqueeze(0)).squeeze(0)
            else:
                emb = zero
            role_in.append(torch.cat([torch.tensor(rf, dtype=dt), emb]))
            owner.append(i)
    r = model.role_mlp(torch.stack(role_in))
    own = torch.tensor(owner, dtype=torch.long)
    n = len(feats)
    rsum = torch.zeros(n, H, dtype=dt).index_add_(0, own, r)
    rmax = torch.full((n, H), -math.inf, dtype=dt).scatter_reduce(0, own.unsqueeze(1).expand(-1, H), r, 'amax')
    for i, f in enumerate(feats):
        ctx = h[rows[f.ctx]] if f.ctx in rows else zero
        reg = [(rows[m], c) for m, c in f.region if m in rows]
        pooled = v1.pool_rows(h, [x for x, c in reg], [c for x, c in reg]) if reg else zero
        heads.append(torch.cat([ctx, pooled, family_vector(model, f.family), rsum[i], rmax[i],
                                torch.tensor(f.d, dtype=dt), torch.tensor(f.concrete, dtype=dt),
                                model.kind_emb(torch.tensor(f.kind, dtype=torch.long)).to(dt)]))
    return model.rw_head(torch.stack(heads)).squeeze(1)


def t7_forward(model: SchemaAgnosticModel, states: list[tuple], blocks: list):
    """Encode once; return (primitive logits, values, rewrite logits per state, batch). blocks[i]: a list of
    ActionFeatures (or None / empty)."""
    batch = v1.make_batch(states)
    logits, values = v1.forward(model, batch)
    h = model._last_h
    out = []
    for i, ((arena, info, root, actions), feats) in enumerate(zip(states, blocks)):
        if not feats:
            out.append(torch.zeros(0, dtype=h.dtype))
            continue
        out.append(rewrite_logits(model, h, graph_rows(info, batch.offsets[i]), feats))
    return logits, values, out, batch


def t7_batch_loss(model: SchemaAgnosticModel, items: list[tuple]):
    """The inherited V1 listwise loss + value MSE, with each labelled state's slice extended by its rewrite rows and
    the target moved to a rewrite when the Case B label says so (exactly `rewrite_model.rewrite_batch_loss`).

    items: (arena, info, root, actions, value, positive, block), block = dict(features=[...], target=int|None)."""
    states = [(a, info, root, actions) for a, info, root, actions, value, positive, block in items]
    blocks = [block['features'] if block else None for *rest, block in items]
    logits, values, rw_logits, batch = t7_forward(model, states, blocks)
    targets = torch.tensor([it[4] for it in items], dtype=values.dtype)
    total = torch.zeros((), dtype=logits.dtype)
    n = 0
    for state, it in enumerate(items):
        positive, block = it[5], it[6]
        if positive is None:
            continue
        rows = [i for i, s in enumerate(batch.state_of) if s == state]
        rw = rw_logits[state]
        if len(rw) == 0:
            if len(rows) < 2:
                continue
            slice_, target = logits[torch.tensor(rows, dtype=torch.long)], positive
        else:
            if len(rows)+len(rw) < 2:
                continue
            slice_ = torch.cat([logits[torch.tensor(rows, dtype=torch.long)], rw])
            target = positive if block['target'] is None else len(rows)+block['target']
        total = total+nn.functional.cross_entropy(slice_.unsqueeze(0), torch.tensor([target], dtype=torch.long))
        n += 1
    return total/max(n, 1)+nn.functional.mse_loss(values, targets)


def parameter_count(model: nn.Module, prefixes: tuple | None = None) -> int:
    return sum(p.numel() for name, p in model.named_parameters() if prefixes is None or name.startswith(prefixes))


def load_t7_model(path) -> SchemaAgnosticModel:
    blob = torch.load(str(path), map_location='cpu', weights_only=False)
    assert blob['extra']['model_version'] == T7_MODEL_VERSION, blob['extra']
    assert blob['extra']['feature_version'] == TF.T7_FEATURE_VERSION, blob['extra']
    model = build_t7_model(Recipe(**blob['recipe']), int(blob['seed']))
    model.load_state_dict(blob['state_dict'])
    model.eval()
    return model
