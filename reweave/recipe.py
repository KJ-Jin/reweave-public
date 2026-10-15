"""The stabilized encoder recipe and a resumable training loop equivalent to the inherited one.

The recipe changes only learning rate, LayerNorm after each message-passing update and a residual connection;
everything else is the inherited training rule: Adam, batches of 32 states shuffled in windows of 64 records with
`random.Random(f'train|{seed}|{epoch}')`, listwise action cross-entropy + value MSE, and the lowest held-in validation
loss checkpoint with patience. With `layernorm=False, residual=False` the model and batch order are identical to the
inherited `train.fit` (tested).
"""

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import torch
from torch import nn

from . import v1
from .hashing import sha256_json


@dataclass(frozen=True)
class Recipe:
    lr: float
    layernorm: bool
    residual: bool
    epochs: int = 20
    patience: int = 3
    hidden: int = 128
    layers: int = 3
    batch: int = 32
    window: int = 64
    optimizer: str = 'adam'
    clip: float = 0.0
    source: str = ''            # provenance note only; excluded from the hash

    def semantic(self) -> dict:
        d = asdict(self)
        d.pop('source')
        return d

    def hash(self) -> str:
        return sha256_json(self.semantic())


SMOKE_PLACEHOLDER = Recipe(lr=1e-3, layernorm=False, residual=False, epochs=1, patience=1,
                           source='SMOKE PLACEHOLDER: plumbing only, never a scientific recipe')


def load_recipe(path: str | Path) -> Recipe:
    data = json.loads(Path(path).read_text())
    return Recipe(**{k: v for k, v in data.items() if k in Recipe.__dataclass_fields__})


class RecipeModel(v1.PolyModel):
    """V1 PolyModel with the stabilization study's optional residual connection and post-update LayerNorm."""

    def __init__(self, layers: int, hidden: int, layernorm: bool, residual: bool) -> None:
        super().__init__(layers, hidden)
        self.residual = residual
        self.use_layernorm = layernorm
        if layernorm:
            self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))

    def encode(self, nodes, child, parent, attr):
        h = self.input(nodes)
        for i in range(self.layers):
            up = torch.zeros_like(h)
            down = torch.zeros_like(h)
            if child.numel():
                up.index_add_(0, parent, self.children_msg[i](torch.cat([h[child], attr], dim=1)))
                down.index_add_(0, child, self.parents_msg[i](torch.cat([h[parent], attr], dim=1)))
            out = self.update[i](torch.cat([h, up, down], dim=1))
            h = h+out if self.residual else out
            if self.use_layernorm:
                h = self.norms[i](h)
        return h


def init_base_parameters(model: nn.Module, seed: int, skip_prefixes: tuple[str, ...]) -> None:
    """Exactly V1 `new_model` initialization over the inherited parameters, in their registration order."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith(skip_prefixes):
                continue
            if parameter.dim() > 1:
                bound = 1.0/math.sqrt(parameter.shape[1])
                parameter.copy_(torch.empty_like(parameter).uniform_(-bound, bound, generator=generator))
            else:
                parameter.zero_()


def build_model(recipe: Recipe, seed: int) -> RecipeModel:
    entropy = torch.get_rng_state()
    model = RecipeModel(recipe.layers, recipe.hidden, recipe.layernorm, recipe.residual).double()
    torch.set_rng_state(entropy)
    init_base_parameters(model, seed, ('norms.',))
    return model


def save_model(path: Path, model: nn.Module, recipe: Recipe, seed: int, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    recipe=recipe.semantic(), seed=seed, extra=extra or {}), tmp)
    os.replace(tmp, path)


_LOADED: dict[str, RecipeModel] = {}


def load_model(path: str | Path) -> RecipeModel:
    key = str(Path(path).resolve())
    if key not in _LOADED:
        blob = torch.load(key, map_location='cpu', weights_only=False)
        recipe = Recipe(**blob['recipe'])
        model = build_model(recipe, int(blob['seed']))
        model.load_state_dict(blob['state_dict'])
        model.eval()
        _LOADED[key] = model
    return _LOADED[key]


# ---------------------------------------------------------------- V1-identical batch order with fast-forward


def batch_tokens(index: list[dict], rng: random.Random | None, batch: int, window: int) -> Iterator[list[tuple[int, int]]]:
    """Yield batches as (index position, state position) tokens in exactly V1 `train.batches` order.

    V1 shuffles record order, then fills a buffer window by window and shuffles the buffer; `random.shuffle`
    consumes randomness that depends only on the list length, so tokens reproduce the order without rebuilding.
    """
    order = list(range(len(index)))
    if rng is not None:
        rng.shuffle(order)
    buffer, start = [], 0
    for first in range(0, len(order), window):
        buffer, start = buffer[start:], 0
        for pos in order[first:first+window]:
            buffer += [(pos, s) for s in range(index[pos]['states'])]
        if rng is not None:
            rng.shuffle(buffer)
        while len(buffer)-start >= batch:
            yield buffer[start:start+batch]
            start += batch
    if len(buffer) > start:
        yield buffer[start:]


class RecordCache:
    """Rebuild each record once while any pending token still refers to it."""

    def __init__(self, path: Path, index: list[dict], build: Callable[[dict], list]) -> None:
        self.stream = path.open('rb')
        self.index, self.build = index, build
        self.cache: dict[int, list] = {}

    def items(self, tokens: list[tuple[int, int]]) -> list:
        out = []
        for pos, s in tokens:
            if pos not in self.cache:
                self.cache[pos] = self.build(v1.read_record_at(self.stream, self.index[pos]['offset']))
            out.append(self.cache[pos][s])
        return out

    def evict_except(self, keep: set[int]) -> None:
        for pos in [p for p in self.cache if p not in keep]:
            del self.cache[pos]

    def close(self) -> None:
        self.stream.close()


def iterate_items(path: Path, index: list[dict], rng: random.Random | None, batch: int, window: int,
                  build: Callable[[dict], list], skip: int = 0) -> Iterator[tuple[int, list]]:
    """Yield (batch number, items) from batch `skip` on, rebuilding only records that later batches need."""
    all_batches = list(batch_tokens(index, rng, batch, window))
    remaining: dict[int, int] = {}
    for tokens in all_batches[skip:]:
        for pos, s in tokens:
            remaining[pos] = remaining.get(pos, 0)+1
    cache = RecordCache(path, index, build)
    try:
        for number, tokens in enumerate(all_batches):
            if number < skip:
                continue
            items = cache.items(tokens)
            for pos, s in tokens:
                remaining[pos] -= 1
                if remaining[pos] == 0:
                    del remaining[pos]
                    cache.cache.pop(pos, None)
            yield number, items
    finally:
        cache.close()


# ---------------------------------------------------------------- training


def finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


@dataclass
class FitState:
    epoch: int = 0
    batch: int = 0
    step: int = 0
    total: float = 0.0
    count: int = 0
    history: list = field(default_factory=list)
    best: float | None = None
    best_epoch: int | None = None
    done: bool = False


PROBE_STEPS = frozenset([0, 1, 10, 50, 100, 250, 500, 1000, 2000])
PROBE_EVERY = 2500
EARLY_STEPS = 100


class HealthProbe:
    """Read-only training-health measurements (never alters numerics): at probe steps, forward hooks record each
    message/update MLP's hidden ReLU (dead-unit fraction = units zero on every row; zero-entry fraction; mean/std) and
    each layer's node states (update output, post-LayerNorm state); after backward, per-group gradient norms and a
    non-finite check over parameters and gradients. Lines go to `workdir/health.jsonl`."""

    GROUPS = ('input', 'children_msg', 'parents_msg', 'update', 'norms', 'scorer', 'value')

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handles: list = []
        self.row: dict | None = None
        self.early: list[float] = []

    def wanted(self, step: int) -> bool:
        return step in PROBE_STEPS or step % PROBE_EVERY == 0

    def begin(self, model: nn.Module, epoch: int, step: int) -> None:
        self.row = dict(epoch=epoch, step=step, activations={}, node_states={})
        for name, module in model.named_modules():
            parts = name.split('.')
            if parts[0] in ('children_msg', 'parents_msg', 'update', 'input') and isinstance(module, nn.ReLU):
                self.handles.append(module.register_forward_hook(self._activation(name)))
            elif parts[0] in ('update', 'norms') and len(parts) == 2:
                self.handles.append(module.register_forward_hook(self._state(name)))

    def _activation(self, name: str):
        def hook(module, inputs, output):
            with torch.no_grad():
                x = output.detach()
                if x.numel():
                    self.row['activations'][name] = dict(
                        dead_unit_fraction=float((x == 0).all(dim=0).double().mean()),
                        zero_fraction=float((x == 0).double().mean()), mean=float(x.mean()), std=float(x.std()))
        return hook

    def _state(self, name: str):
        def hook(module, inputs, output):
            with torch.no_grad():
                x = output.detach()
                if x.numel():
                    self.row['node_states'][name] = dict(std=float(x.std()), feature_std_mean=float(x.std(dim=0).mean()),
                                                         mean=float(x.mean()), abs_max=float(x.abs().max()))
        return hook

    def after_backward(self, model: nn.Module, loss: float) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        grads: dict[str, float] = {}
        finite_params, finite_grads = True, True
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                group = name.split('.')[0]
                finite_params &= bool(torch.isfinite(parameter).all())
                if parameter.grad is not None:
                    finite_grads &= bool(torch.isfinite(parameter.grad).all())
                    grads[group] = grads.get(group, 0.0)+float(parameter.grad.double().pow(2).sum())
        self.row.update(loss=loss, grad_norm={k: math.sqrt(v) for k, v in sorted(grads.items())},
                        message_passing_grad_norm=math.sqrt(sum(grads.get(g, 0.0) for g in
                                                                ('children_msg', 'parents_msg', 'update', 'norms'))),
                        parameters_finite=finite_params, gradients_finite=finite_grads)
        self.write(self.row)
        self.row = None

    def loss(self, step: int, value: float) -> None:
        if step < EARLY_STEPS:
            self.early.append(value)
            if len(self.early) == EARLY_STEPS:
                self.write(dict(early_loss_mean=sum(self.early)/len(self.early), early_loss_first=self.early[0],
                                early_loss_last=self.early[-1], steps=EARLY_STEPS))

    def write(self, row: dict) -> None:
        with self.path.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=True)+'\n')


def fit(dataset: Path, index: list[dict], seed: int, recipe: Recipe, workdir: Path,
        model: nn.Module | None = None, loss_fn: Callable | None = None, build: Callable[[dict], list] | None = None,
        checkpoint_minutes: float = 10.0, log: Callable[[str], None] = print,
        max_steps: int | None = None) -> dict:
    """Train with the inherited selection rule; resumable at batch granularity from `workdir/resume.pt`.

    `loss_fn(model, items)` defaults to V1 `train.batch_loss`; library arms pass their own loss and record builder.
    The checkpoint is written only after the epoch's best/patience bookkeeping is updated.
    """
    from polycompress import train as v1train
    loss_fn = loss_fn or v1train.batch_loss
    build = build or v1.state_tuples
    workdir.mkdir(parents=True, exist_ok=True)
    v1.configure_determinism('cpu')
    fit_index, val_index = v1.split_index(index)
    if not fit_index or not val_index:
        raise ValueError(f'{dataset} does not cover both train_fit and train_modelval')
    model = model if model is not None else build_model(recipe, seed)
    if recipe.optimizer != 'adam':
        raise ValueError('only the inherited Adam optimizer is implemented')
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe.lr)
    state = FitState()
    resume = workdir/'resume.pt'
    kept_path = workdir/'best.pt'
    if resume.is_file():
        blob = torch.load(resume, map_location='cpu', weights_only=False)
        model.load_state_dict(blob['model'])
        optimizer.load_state_dict(blob['optimizer'])
        state = FitState(**blob['state'])
        log(f'[fit] resumed at epoch {state.epoch} batch {state.batch} step {state.step} best {state.best}')
    last_save = time.perf_counter()
    began = time.perf_counter()
    probe = HealthProbe(workdir/'health.jsonl')

    def save() -> None:
        tmp = resume.with_suffix('.pt.tmp')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), state=asdict(state),
                        recipe=recipe.semantic(), seed=seed), tmp)
        os.replace(tmp, resume)

    while not state.done and state.epoch < recipe.epochs:
        model.train()
        rng = random.Random(f'train|{seed}|{state.epoch}')
        for number, items in iterate_items(dataset, fit_index, rng, recipe.batch, recipe.window, build, state.batch):
            optimizer.zero_grad()
            probing = probe.wanted(state.step)
            if probing:
                probe.begin(model, state.epoch, state.step)
            loss = loss_fn(model, items)
            value = float(loss.detach())
            probe.loss(state.step, value)
            if not finite(value):
                if probing:
                    probe.after_backward(model, value)
                raise FloatingPointError(f'non-finite training loss at epoch {state.epoch} batch {number}')
            loss.backward()
            if probing:
                probe.after_backward(model, value)
            if recipe.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.clip)
            optimizer.step()
            state.total += value*len(items)
            state.count += len(items)
            state.batch = number+1
            state.step += 1
            if max_steps is not None and state.step >= max_steps:
                save()
                return dict(stopped='max_steps', state=asdict(state))
            if time.perf_counter()-last_save > 60*checkpoint_minutes:
                save()
                last_save = time.perf_counter()
        model.eval()
        with torch.no_grad():
            vtotal, vcount = 0.0, 0
            for number, items in iterate_items(dataset, val_index, None, recipe.batch, recipe.window, build):
                vtotal += float(loss_fn(model, items))*len(items)
                vcount += len(items)
        measured = vtotal/vcount
        if not finite(measured):
            raise FloatingPointError(f'non-finite train_modelval loss at epoch {state.epoch}')
        record = dict(epoch=state.epoch, train_fit_loss=state.total/state.count, train_modelval_loss=measured,
                      steps=state.step, seconds=time.perf_counter()-began)
        state.history.append(record)
        log(f'[fit] seed {seed} epoch {state.epoch} fit {record["train_fit_loss"]:.6f} modelval {measured:.6f}')
        if state.best is None or measured < state.best:
            state.best, state.best_epoch = measured, state.epoch
            save_model(kept_path, model, recipe, seed, dict(epoch=state.epoch, train_modelval_loss=measured))
        elif state.epoch-state.best_epoch >= recipe.patience:
            state.done = True
        state.epoch += 1
        state.batch, state.total, state.count = 0, 0.0, 0
        save()
        last_save = time.perf_counter()
    state.done = True
    save()
    return dict(seed=seed, recipe=recipe.semantic(), recipe_hash=recipe.hash(), selected_epoch=state.best_epoch,
                train_modelval_loss=state.best, epochs_run=len(state.history), history=state.history,
                train_fit_records=len(fit_index), train_modelval_records=len(val_index), best=str(kept_path),
                health=str(probe.path))
