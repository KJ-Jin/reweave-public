"""Paired five-seed bootstrap with canonical-shape primary inference (specification section 13.1)."""

from dataclasses import dataclass
import random

from .config import StatsCfg

AUTHORED = ('test_id', 'test_disc', 'test_comp')


@dataclass(frozen=True)
class Paired:
    """One concrete instance's paired outcome: per-seed learned values against a control."""
    stratum: str
    shape: str
    ours: tuple[float, ...]
    control: tuple[float, ...]
    control_learned: bool = False

    def effect(self, ours: tuple[int, ...], control: tuple[int, ...]) -> float:
        """Return the paired effect under one seed draw; a deterministic control ignores its draw."""
        mine = sum(self.ours[i] for i in ours)/len(ours)
        if self.control_learned:
            theirs = sum(self.control[j] for j in control)/len(control)
        else:
            theirs = sum(self.control)/len(self.control)
        return mine-theirs


def _shape_means(samples: list[Paired], ours: tuple[int, ...],
                 control: tuple[int, ...]) -> dict[str, dict[str, float]]:
    """Average the paired effect within each canonical shape, after the seed mean is formed."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for sample in samples:
        grouped.setdefault(sample.stratum, {}).setdefault(sample.shape, []).append(
            sample.effect(ours, control))
    return {stratum: {shape: sum(v)/len(v) for shape, v in shapes.items()}
            for stratum, shapes in grouped.items()}


def _combine(per_stratum: dict[str, float], weights: dict[str, float] | None) -> float:
    """Combine strata with the frozen outer weights; the default is equal weight."""
    names = sorted(per_stratum)
    if weights is None:
        return sum(per_stratum[n] for n in names)/len(names)
    total = sum(weights[n] for n in names)
    return sum(per_stratum[n]*weights[n] for n in names)/total


def _point_shape(samples: list[Paired], ours: tuple[int, ...], control: tuple[int, ...],
                 weights: dict[str, float] | None, draw: dict[str, list[str]] | None) -> float:
    """Equal-weight shape mean inside each stratum, then the frozen outer combination."""
    means = _shape_means(samples, ours, control)
    per_stratum = {}
    for stratum, shapes in means.items():
        keys = draw[stratum] if draw is not None else sorted(shapes)
        per_stratum[stratum] = sum(shapes[k] for k in keys)/len(keys)
    return _combine(per_stratum, weights)


def _point_instance(samples: list[Paired], ours: tuple[int, ...], control: tuple[int, ...],
                    draw: list[int] | None) -> float:
    """Ordinary paired instance mean; the random-OOD and sensitivity estimand."""
    rows = samples if draw is None else [samples[i] for i in draw]
    return sum(s.effect(ours, control) for s in rows)/len(rows)


def paired_ci(samples: list[Paired], mode: str = 'shape', weights: dict[str, float] | None = None,
              cfg: StatsCfg | None = None) -> dict:
    """Return the point estimate, percentile interval, per-seed effects and structural sample size."""
    if mode not in ('shape', 'instance'):
        raise ValueError("mode must be 'shape' or 'instance'")
    if not samples:
        raise ValueError('paired_ci needs at least one sample')
    cfg = cfg or StatsCfg()
    n_ours = len(samples[0].ours)
    n_control = len(samples[0].control)
    if any(len(s.ours) != n_ours for s in samples):
        raise ValueError('every instance must carry the same number of learned seeds')
    all_ours, all_control = tuple(range(n_ours)), tuple(range(n_control))
    learned_control = samples[0].control_learned
    if mode == 'shape':
        point = _point_shape(samples, all_ours, all_control, weights, None)
    else:
        point = _point_instance(samples, all_ours, all_control, None)
    per_seed = [_point_shape([s for s in samples], (i,), all_control, weights, None) if mode == 'shape'
                else _point_instance(samples, (i,), all_control, None) for i in range(n_ours)]
    rng = random.Random(0)
    shapes_by_stratum: dict[str, list[str]] = {}
    for sample in samples:
        shapes_by_stratum.setdefault(sample.stratum, [])
        if sample.shape not in shapes_by_stratum[sample.stratum]:
            shapes_by_stratum[sample.stratum].append(sample.shape)
    draws = []
    for _ in range(cfg.bootstrap_reps):
        ours = tuple(rng.randrange(n_ours) for _ in range(n_ours))
        control = tuple(rng.randrange(n_control) for _ in range(n_control)) if learned_control else all_control
        if mode == 'shape':
            draw = {stratum: [rng.choice(keys) for _ in keys]
                    for stratum, keys in sorted(shapes_by_stratum.items())}
            draws.append(_point_shape(samples, ours, control, weights, draw))
        else:
            draw = [rng.randrange(len(samples)) for _ in samples]
            draws.append(_point_instance(samples, ours, control, draw))
    draws.sort()
    lo = draws[int(cfg.alpha/2*len(draws))]
    hi = draws[min(len(draws)-1, int((1-cfg.alpha/2)*len(draws)))]
    summary = dict(mode=mode, point=point, lo=lo, hi=hi, per_seed=per_seed, n_seeds=n_ours,
                   n_instances=len(samples), reps=cfg.bootstrap_reps, alpha=cfg.alpha)
    return dict(summary, **_size(mode, shapes_by_stratum))


def _size(mode: str, shapes_by_stratum: dict[str, list[str]]) -> dict:
    """Attach the structural sample size that every primary CI must print beside its interval."""
    if mode != 'shape':
        return dict(n_shapes=None, n_shapes_total=None, n_shapes_by_stratum=None)
    counts = {stratum: len(keys) for stratum, keys in sorted(shapes_by_stratum.items())}
    if len(counts) == 1:
        return dict(n_shapes=next(iter(counts.values())), n_shapes_total=sum(counts.values()),
                    n_shapes_by_stratum=counts)
    return dict(n_shapes=None, n_shapes_total=sum(counts.values()), n_shapes_by_stratum=counts)


def solve_rate(samples: list[Paired], weights: dict[str, float] | None = None) -> float:
    """Single-method canonical-shape solve rate: equal-weight per shape, then the frozen outer mean."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for sample in samples:
        grouped.setdefault(sample.stratum, {}).setdefault(sample.shape, []).append(
            sum(sample.ours)/len(sample.ours))
    per_stratum = {stratum: sum(sum(v)/len(v) for v in shapes.values())/len(shapes)
                   for stratum, shapes in grouped.items()}
    return _combine(per_stratum, weights)
