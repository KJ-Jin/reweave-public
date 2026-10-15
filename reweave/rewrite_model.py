"""Learned scorer over primitive and rewrite candidates, and its listwise loss.

Base encoder, primitive scorer and value head are the inherited message-passing model (`recipe.RecipeModel`),
computed in one encoder pass. Rewrite candidates get one extra scorer:
    [h(context) ; count-weighted mean h(matched source region children) ; schema embedding (32) ;
     (target unique ops, n_holes, log1p(task_support)) ; mean input-MLP embedding of the bound monomials]
    -> MLP(H, H, 1)
This is the fixed-vocabulary scorer (one embedding per relation); `canon/action_model.py` is its schema-agnostic
successor. Loss: listwise cross-entropy over each labelled state's primitive + rewrite candidates plus value MSE.
"""

import hashlib
import math

import torch
from torch import nn

from . import v1
from .recipe import Recipe, RecipeModel, init_base_parameters

SCHEMA_EMB = 32
N_SCHEMA_FEATS = 3
RW_PREFIXES = ('schema_emb.', 'rw_scorer.')


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes)-2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class RewriteModel(RecipeModel):
    def __init__(self, layers: int, hidden: int, layernorm: bool, residual: bool, n_schemas: int) -> None:
        super().__init__(layers, hidden, layernorm, residual)
        H = hidden
        self.n_schemas = n_schemas
        self.schema_emb = nn.Embedding(max(1, n_schemas), SCHEMA_EMB)
        self.rw_scorer = _mlp([2*H+SCHEMA_EMB+N_SCHEMA_FEATS+H, H, H, 1])
        self._last_h = None

    def encode(self, nodes, child, parent, attr):
        h = super().encode(nodes, child, parent, attr)
        self._last_h = h
        return h


def build_rewrite_model(recipe: Recipe, seed: int, n_schemas: int) -> RewriteModel:
    """Inherited parameters initialized exactly as V1 `new_model(seed)`; the rewrite head from its own generator."""
    entropy = torch.get_rng_state()
    model = RewriteModel(recipe.layers, recipe.hidden, recipe.layernorm, recipe.residual, n_schemas).double()
    torch.set_rng_state(entropy)
    init_base_parameters(model, seed, ('norms.',)+RW_PREFIXES)
    generator = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f'rewrite|{seed}'.encode()).digest()[:7], 'big'))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not name.startswith(RW_PREFIXES):
                continue
            if parameter.dim() > 1:
                bound = 1.0/math.sqrt(parameter.shape[1])
                parameter.copy_(torch.empty_like(parameter).uniform_(-bound, bound, generator=generator))
            else:
                parameter.zero_()
    return model


def graph_rows(info, offset: int) -> dict[int, int]:
    """V1 `build_graph` node order is sorted(reach_h)."""
    return {h: offset+i for i, h in enumerate(sorted(info.reach_h))}


def rewrite_forward(model: RewriteModel, states: list[tuple], blocks: list[dict | None]):
    """Encode once; return (primitive logits, values, rewrite logits per state, batch).

    blocks[i]: dict(candidates=[(ctx handle, ((member handle, count), ...), schema index, feats, [packed monomial per
    hole])]) with live handle ids, or None."""
    batch = v1.make_batch(states)
    logits, values = v1.forward(model, batch)
    h = model._last_h
    out = []
    for i, ((arena, info, root, actions), block) in enumerate(zip(states, blocks)):
        if not block or not block['candidates']:
            out.append(torch.zeros(0, dtype=h.dtype))
            continue
        rows = graph_rows(info, batch.offsets[i])
        feats = []
        for ctx, members, schema_index, scal, monos in block['candidates']:
            pooled = v1.pool_rows(h, [rows[m] for m, c in members], [c for m, c in members])
            mono_x = torch.tensor([v1.off_graph_mono(m) for m in monos], dtype=h.dtype)
            bind = model.input(mono_x).mean(dim=0)
            emb = model.schema_emb(torch.tensor([schema_index], dtype=torch.long)).squeeze(0).to(h.dtype)
            feats.append(torch.cat([h[rows[ctx]], pooled, emb, torch.tensor(scal, dtype=h.dtype), bind]))
        out.append(model.rw_scorer(torch.stack(feats)).squeeze(1))
    return logits, values, out, batch


def rewrite_batch_loss(model, items: list[tuple]):
    """V1 `train.batch_loss` with Case B rewrite rows appended to each labelled state's slice.

    items: (arena, info, root, actions, value, positive, block) where block = dict(candidates=..., target=int|None)."""
    states = [(a, info, root, actions) for a, info, root, actions, value, positive, block in items]
    blocks = [block for *rest, block in items]
    logits, values, rw_logits, batch = rewrite_forward(model, states, blocks)
    targets = torch.tensor([it[4] for it in items], dtype=values.dtype)
    positives = {i: it[5] for i, it in enumerate(items) if it[5] is not None}
    total = torch.zeros((), dtype=logits.dtype)
    n = 0
    for state, positive in sorted(positives.items()):
        rows = [i for i, s in enumerate(batch.state_of) if s == state]
        rw = rw_logits[state]
        block = blocks[state]
        if len(rw) == 0:
            # Identical to V1 listwise_loss for a state without rewrite rows.
            if len(rows) < 2:
                continue
            slice_ = logits[torch.tensor(rows, dtype=torch.long)]
            target = positive
        else:
            if len(rows)+len(rw) < 2:
                continue
            slice_ = torch.cat([logits[torch.tensor(rows, dtype=torch.long)], rw])
            target = positive if block['target'] is None else len(rows)+block['target']
        total = total+nn.functional.cross_entropy(slice_.unsqueeze(0), torch.tensor([target], dtype=torch.long))
        n += 1
    return total/max(n, 1)+nn.functional.mse_loss(values, targets)


def mono_list(beta: list | tuple) -> list[int]:
    return [v1.pack(tuple(vec)) for hole, vec in sorted(beta)]


def rewrite_state_tuples(record: dict, schema_index: dict) -> list[tuple]:
    """V1 `train.state_tuples` plus the rewrite block of each labelled state."""
    arena, ids = v1.rebuild_record(record)
    labels = {entry['root']: (k, entry) for k, entry in enumerate(record['labelled'])}
    blocks = (record.get('rewrite') or {}).get('states', {})
    items = []
    for state in record['states']:
        root = ids[state['root']]
        k, entry = labels.get(state['root'], (None, None))
        actions = [v1.action_from_record(a, ids) for a in entry['actions']] if entry else []
        raw = blocks.get(k) if k is not None else None
        block = None
        if raw is not None:
            block = dict(candidates=[(ids[c['ctx']], tuple((ids[m], n) for m, n in c['region']), schema_index[c['schema']],
                                      c['feats'], mono_list(c['beta'])) for c in raw['candidates']],
                         target=raw['target'])
        items.append((arena, v1.state_info(arena, root), root, actions, state['value'],
                      entry['positive'] if entry else None, block))
    return items


class RewriteScorer:
    """Search-time combined scorer: inherited primitive logits + rewrite logits from one encoder pass."""

    def __init__(self, model: RewriteModel, schema_index: dict, features: dict) -> None:
        self.model, self.schema_index, self.features = model, schema_index, features

    def score(self, arena, info, root, actions):
        return self.score_combined(arena, info, root, actions, [])[0]

    def score_combined(self, arena, info, root, actions, cands):
        if not actions and not cands:
            return [], []
        block = None
        if cands:
            block = dict(candidates=[(c.ctx, tuple(c.R_inst.items()), self.schema_index[c.schema.schema_hash],
                                      self.features[c.schema.schema_hash],
                                      mono_list([(i, b) for i, b in c.beta.items()])) for c in cands], target=None)
        with torch.no_grad():
            logits, values, rw, _ = rewrite_forward(self.model, [(arena, info, root, actions)], [block])
        return [float(x) for x in logits], [float(x) for x in rw[0]]


class RewriteValueRanker:
    def __init__(self, model: RewriteModel) -> None:
        self.model = model

    def rank(self, arena, info, root) -> float:
        with torch.no_grad():
            logits, values = v1.forward(self.model, v1.make_batch([(arena, info, root, [])]))
        return float(values[0])


def load_rewrite_model(path) -> RewriteModel:
    blob = torch.load(str(path), map_location='cpu', weights_only=False)
    model = build_rewrite_model(Recipe(**blob['recipe']), int(blob['seed']), int(blob['extra']['n_schemas']))
    model.load_state_dict(blob['state_dict'])
    model.eval()
    return model
