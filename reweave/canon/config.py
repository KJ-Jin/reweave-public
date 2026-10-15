"""Configuration switches of the canonical architecture, their dependency closure, profiles and identity.

Any switch that can change action equivalence classes, candidate caps, scoring classes, matcher-visible actions,
normalization, portfolio/view availability or latent identity is part of the configuration identity, so enabling it
always yields a new explicit hash. With every switch OFF (INTEGRATION_BASELINE) behaviour equals the reference
configuration (`current_config.CURRENT`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from ..hashing import sha256_json

VERSION = 'canon-config-v1'


@dataclass(frozen=True)
class IntegrationConfig:
    semantic_interning: bool = False       # T2: handle-level semantic classes + bounded representative store
    normalization: bool = False            # T2: cheapest-representative normalization (strict whole-state decrease)
    archive_retrieval: bool = False        # T3: per-context retrieval for the VISIBLE matcher only
    identity_mode: bool = False            # T4: symmetry-safe canonical states and action classes (incl. latent)
    portfolio_alternatives: bool = False   # T5: retained non-cheapest representatives are offered to views
    view_reuse: bool = False               # T5: alternative-representation views as actions
    view_index: bool = True                # T5: indexed view retrieval (effective only when view_reuse)
    generic_exprparam_binding: bool = False  # T6: the ExprParam matcher / lowering / action identity (canon.exprparam)
    reachable_handle_bind: bool = False    # T6: an ExprParam may bind an existing compound handle of the concrete state

    def effective(self) -> 'IntegrationConfig':
        """Dependency closure and validation. Never silently enables a behaviour-changing mode the caller did not ask
        for: views require identity mode to be requested explicitly, handle binding requires the generic binder."""
        c = self
        if c.view_reuse and not c.identity_mode:
            raise ValueError('view_reuse requires identity_mode (views dedup through canonical states)')
        if c.identity_mode and c.archive_retrieval:
            raise ValueError('identity_mode with archive_retrieval is not integrated yet (T3 hook is concrete-mode only)')
        if c.reachable_handle_bind and not c.generic_exprparam_binding:
            raise ValueError('reachable_handle_bind requires generic_exprparam_binding')
        if c.generic_exprparam_binding and c.archive_retrieval:
            raise ValueError('generic_exprparam_binding with archive_retrieval is not integrated (the T3 retrieval '
                             'matcher is a monomial-only transcription)')
        if c.generic_exprparam_binding and c.view_reuse:
            raise ValueError('generic_exprparam_binding with view_reuse is not integrated: view/portfolio handles are '
                             'T6B, held until the T5 index-vs-eager difference is resolved')
        if c.view_reuse:
            c = replace(c, portfolio_alternatives=True)
        if c.normalization or c.portfolio_alternatives:
            c = replace(c, semantic_interning=True)
        if not c.view_reuse:
            c = replace(c, view_index=False)
        return c

    def flags(self) -> dict:
        return asdict(self.effective())

    def is_baseline(self) -> bool:
        e = self.effective()
        return not any(v for k, v in asdict(e).items())


PROFILES = {
    'INTEGRATION_BASELINE': IntegrationConfig(),
    'INTERNING_ONLY': IntegrationConfig(semantic_interning=True),
    'NORMALIZATION': IntegrationConfig(normalization=True),
    'ARCHIVE_RETRIEVAL': IntegrationConfig(archive_retrieval=True),
    'IDENTITY': IntegrationConfig(identity_mode=True),
    'IDENTITY_NORMALIZATION': IntegrationConfig(identity_mode=True, normalization=True),
    'IDENTITY_VIEWS': IntegrationConfig(identity_mode=True, view_reuse=True, view_index=True),
    'IDENTITY_VIEWS_EAGER': IntegrationConfig(identity_mode=True, view_reuse=True, view_index=False),
    'IDENTITY_NORMALIZATION_VIEWS': IntegrationConfig(identity_mode=True, normalization=True, view_reuse=True),
    # T6: the generic binder with MONOMIAL_BIND only (the legacy monomial lane: the mainline generators, unchanged), and
    # with REACHABLE_HANDLE_BIND (a separate handle lane over concrete reachable compound handles only).
    'GENERIC_MONOMIAL': IntegrationConfig(generic_exprparam_binding=True),
    'GENERIC_HANDLES': IntegrationConfig(generic_exprparam_binding=True, reachable_handle_bind=True),
    'IDENTITY_GENERIC_MONOMIAL': IntegrationConfig(identity_mode=True, generic_exprparam_binding=True),
    'IDENTITY_GENERIC_HANDLES': IntegrationConfig(identity_mode=True, generic_exprparam_binding=True,
                                                  reachable_handle_bind=True),
}

# Switches added after T5 enter the configuration identity only when ON, so every T0–T5 profile keeps the identity
# recorded in its gate reports (tests/canon/test_t6_exprparam.py pins them).
POST_T5_FLAGS = ('generic_exprparam_binding', 'reachable_handle_bind')
# The T6 implementation revision enters the identity whenever a T6 switch is ON: T6A.1 changed the handle-mode action
# space (separate handle lane and admission, concrete realizations, flat MUL, AC family id in the action key); T6A.2
# froze the action interface (ActionRecord boundary, CandidateSetRecord, fair handle-lane scheduling, exact keys,
# surface-independent concrete dedup, exact-region latent admissibility).
T6_IMPLEMENTATION = 'T6A.2'


def mainline_identity() -> dict:
    """The MAINLINE_CURRENT identity: exactly the mainline's own record, not recomputed or renamed."""
    from ..current_config import CURRENT
    return dict(profile='MAINLINE_CURRENT', semantic_hash=CURRENT.semantic_hash(), identities=CURRENT.identities())


def config_identity(icfg: IntegrationConfig | None, name: str | None = None) -> dict:
    """The explicit identity of an integration profile. MAINLINE_CURRENT is `icfg is None`."""
    if icfg is None:
        return mainline_identity()
    from ..current_config import CURRENT
    e = icfg.effective()
    flags = {k: v for k, v in asdict(e).items() if k not in POST_T5_FLAGS or v}
    body = dict(v=VERSION, flags=flags, mainline_semantic_hash=CURRENT.semantic_hash())
    if any(getattr(e, k) for k in POST_T5_FLAGS):
        body['t6_implementation'] = T6_IMPLEMENTATION
    h = sha256_json(body)
    return dict(profile=name or ('INTEGRATION_BASELINE' if e.is_baseline() else 'CUSTOM'), config_hash=h, body=body,
                behaviour_equivalent_to='MAINLINE_CURRENT' if e.is_baseline() else None)
