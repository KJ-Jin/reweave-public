"""The complete provisional configuration declared in specification section 9."""

from dataclasses import dataclass, field
from .search import Budget


@dataclass
class CalcCfg:
    r_max: int = 3
    n_max: int = 2000
    allow_partial_multiplicity: bool = True
    consequence_visible: bool = False


@dataclass
class SearchCfg:
    learner: Budget = field(default_factory=lambda: Budget(8,8,20,400))
    teacher: Budget = field(default_factory=lambda: Budget(32,16,20,3200))


@dataclass
class GenCfg:
    max_vars: int = 8
    max_occurrences: int = 60
    max_distinct: int = 40
    max_degree: int = 6
    n_train_per_family: int = 500
    n_val: int = 300
    n_test: int = 300
    n_per_composition: int = 150
    n_e4: int = 200
    n_e7_pairs: int = 200
    d_max: int = 2


@dataclass
class TrainCfg:
    seeds: tuple = (0,1,2,3,4)
    rounds: int = 2
    lr: float = 1e-3
    epochs: int = 20
    patience: int = 3
    hidden: int = 128
    layers: int = 3


@dataclass
class StatsCfg:
    bootstrap_reps: int = 2000
    alpha: float = 0.05


@dataclass
class PilotCfg:
    exhaustive_max_occurrences: int = 12
    exhaustive_max_depth: int = 8
    exhaustive_time_s: float = 60.0
    r_max_grid: tuple = (2,3,4)
    instances_per_cell: int = 20


@dataclass
class OracleCfg:
    max_occurrences: int = 12
    max_vars: int = 5
    time_s: float = 300.0


@dataclass
class EGraphCfg:
    iter_limit: int = 12
    node_limit: int = 100_000
    time_s: float = 60.0
    sensitivity_time_s: float = 600.0
    sensitivity_n_val: int = 20


@dataclass
class RandomOODCfg:
    n_tasks: int = 300
    min_gates: int = 6
    max_gates: int = 12


@dataclass
class FormCfg:
    time_s: float = 60.0
    seed: int = 20260905


@dataclass
class ScaleCfg:
    occurrence_buckets: tuple = ((61,80),(81,100))
    n_per_bucket: int = 20
    min_gates: int = 8
    max_gates: int = 18
    sensitivity_n_max: int = 8000


@dataclass
class NamedCfg:
    long_horizon: Budget = field(default_factory=lambda: Budget(8,8,40,800))


@dataclass
class ComputeCfg:
    n_workers: int = 1
    device: str = 'cpu'


@dataclass
class ProtocolCfg:
    gen_seed_test: int = 20260905
    tau_disc: float | None = None
    delta_disc: float | None = None
    delta_s4: float | None = None
    b_same: str | None = None
    b_sym: str | None = None
