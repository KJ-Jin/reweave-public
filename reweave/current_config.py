"""The single configuration surface for the reference ReWeave configuration.

It records *which* library identity, search budget and admission policy the reference configuration uses, and
derives one `semantic_hash()` over all of them. It defines no mechanism: that lives in `rewrite_latent`,
`candidate_policy`, `rewrite_search_latent` and `canon/`.

Large artifacts (deployed libraries, checkpoints) are not part of this repository; they are looked up under
`$REWEAVE_ARTIFACTS` (default: `./artifacts`).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARTIFACTS = Path(os.environ.get('REWEAVE_ARTIFACTS', str(REPO/'artifacts')))
BUNDLED_LIBRARY = Path(__file__).resolve().parent/'data'/'reference_library.json'


@dataclass(frozen=True)
class FrozenLibrary:
    """Identity of the reference 19-surface library. An immutable input, never recomputed; the artifact itself is
    not part of this repository."""
    artifact: str = 'RW-deploy-21f0b946185d'
    library_hash: str = 'ea919e3e95a8f04259523463ed641ad6744b83587fe425315265deeca4d02f13'
    schema_count: int = 19
    schema_ids_digest: str = '61e68029352a4255a6e3cb3839e12c96862a95e232d6ef5736c180b028a4afeb'

    @property
    def path(self) -> Path:
        return ARTIFACTS/'rewrite_deploy'/self.artifact

    def body(self) -> dict:
        """The full deployment artifact when present under `$REWEAVE_ARTIFACTS`, otherwise the bundled copy of its
        certified relations (`reweave/data/reference_library.json`)."""
        import json
        full = self.path/'deploy.json'
        return json.loads((full if full.is_file() else BUNDLED_LIBRARY).read_text())

    def schemas(self) -> list:
        """The library's certified relations, after the identity check."""
        from .rewrite_library import schemas_of
        return schemas_of(self.assert_identity())

    def assert_identity(self, deploy: dict | None = None) -> dict:
        """Fail closed unless the library is exactly the frozen one."""
        from .hashing import sha256_json
        deploy = self.body() if deploy is None else deploy
        ids = sorted(deploy['eligible'])
        got = (deploy['library_hash'], len(ids), sha256_json(ids))
        want = (self.library_hash, self.schema_count, self.schema_ids_digest)
        assert got == want, f'deployed library is not the frozen one: {got} != {want}'
        return deploy


@dataclass(frozen=True)
class Budget:
    """The frozen search budget. Every arm of every comparison runs under exactly this."""
    B: int = 8
    K_act: int = 8
    T_max: int = 20
    max_expansions: int = 400

    def as_v1(self):
        from . import v1
        return v1.Budget(self.B, self.K_act, self.T_max, self.max_expansions)

    def as_tuple(self) -> tuple:
        return (self.B, self.K_act, self.T_max, self.max_expansions)


@dataclass(frozen=True)
class Machine:
    """The CPU model the reference results were produced on. Recorded because float noise in the learned scorer can
    reorder beam ties across CPU models; exactness never depends on it."""
    required_cpu_model: str = 'AMD EPYC 9654 96-Core Processor'
    insufficient_scheduler_features: tuple = ('genoa', 'gen2')


@dataclass(frozen=True)
class CurrentConfig:
    """The reference configuration."""
    library: FrozenLibrary = field(default_factory=FrozenLibrary)
    budget: Budget = field(default_factory=Budget)
    machine: Machine = field(default_factory=Machine)
    policy_name: str = 'CM-ORIGIN-SCHEMA-DIVERSITY'
    recipe_hash: str = '8cd0f4dd344039664dbef689f7d7dfcb4088965309dc9b2738ac64ea5c47dc3d'
    arm: str = 'Static-Param'
    # Namespace for artifacts written under this configuration. See `lineage_id()`.
    lineage: str = 'rw-r1'

    def latent(self):
        """The production latent mechanism configuration."""
        from .rewrite_latent import LatentConfig
        return LatentConfig()

    def policy(self, name: str | None = None):
        """The deployment candidate-admission policy. Defaults to the predeclared primary."""
        from . import candidate_policy as CP
        return CP.CandidatePolicy(name or self.policy_name)

    def rewrite_config(self):
        from .rewrite_config import RewriteConfig
        return RewriteConfig()

    def identities(self) -> dict:
        """Everything that defines the current system, for recording in a report or a stage key."""
        cfg = self.rewrite_config()
        return dict(lineage=self.lineage, arm=self.arm,
                    library_hash=self.library.library_hash, schema_count=self.library.schema_count,
                    schema_ids_digest=self.library.schema_ids_digest,
                    latent_config_hash=self.latent().hash(), latent_config=self.latent().as_dict(),
                    policy_name=self.policy_name, policy_hash=self.policy().hash(),
                    budget=dict(zip(('B', 'K_act', 'T_max', 'max_expansions'), self.budget.as_tuple())),
                    rewrite_n_max=cfg.rewrite_n_max,
                    rewrite_label_candidates_max=cfg.rewrite_label_candidates_max,
                    rewrite_config_hash=cfg.hash(), recipe_hash=self.recipe_hash,
                    required_cpu_model=self.machine.required_cpu_model)

    def semantic_hash(self) -> str:
        """One hash over everything that defines current behaviour. Records WHAT ran, and changes if any of it does."""
        from .hashing import sha256_json
        return sha256_json(self.identities())

    def lineage_id(self, seed: int, round_: int = 0) -> str:
        """Artifact name for a clean-lineage run. Carries the lineage tag, so it can never collide with a
        historical key such as `Static-Param-s0-r0-888e5283fd0b`."""
        return f'{self.arm}-{self.lineage}-s{seed}-r{round_}'


CURRENT = CurrentConfig()
