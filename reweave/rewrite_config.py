"""Bounds and thresholds of the rewrite mechanism (extraction, proposal, certification, support, matching).

All values are configuration with named counters: a cap that binds is always counted, never silent.
"""

from dataclasses import asdict, dataclass

from .hashing import sha256_json


@dataclass(frozen=True)
class RewriteConfig:
    # extraction
    L_min: int = 2
    L_max: int = 4
    anc_max: int = 2
    gp_source_children: int = 6
    gp_source_multiplicity: int = 12
    gp_target_children: int = 4
    gp_target_new_handles: int = 8
    gp_target_mono_leaves: int = 6
    # proposal
    seeds_per_bucket: int = 64
    basis_size_max: int = 4
    basis_exponent_max: int = 6
    basis_subsets_tested: int = 64
    bases_per_seed: int = 16
    decompositions_per_term: int = 8
    candidates_per_seed: int = 32
    candidates_per_bucket: int = 512
    candidates_total: int = 50_000
    # SchemaBounds
    n_holes_max: int = 4
    source_terms_max: int = 6
    source_multiplicity_max: int = 12
    source_operators_max: int = 12
    target_operators_max: int = 10
    pattern_depth_max: int = 4
    mon_exponent_max: int = 6
    node_arity_max: int = 4
    # support / alias
    rematch_unify_states_max: int = 200_000
    alias_n_max: int = 1_000_000
    alias_time_cap_s: float = 30.0
    # runtime matcher
    match_contexts_max: int = 512
    unify_states_max: int = 4_096
    match_attempts_max: int = 200_000
    rewrite_n_max: int = 256
    # binding caps
    binding_degree_max: int = 6
    binding_exponent_max: int = 6
    # eligibility and library cap
    task_support_min: int = 5
    shape_support_min: int = 3
    binding_signature_support_min: int = 2
    active_cap: int = 32
    # bounded supervision
    rewrite_label_candidates_max: int = 16

    def as_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        return sha256_json(self.as_dict())


MVP1_FIELDS = ('L_min', 'L_max', 'anc_max', 'gp_source_children', 'gp_source_multiplicity', 'gp_target_children',
               'gp_target_new_handles', 'gp_target_mono_leaves', 'seeds_per_bucket', 'basis_size_max',
               'basis_exponent_max', 'basis_subsets_tested', 'bases_per_seed', 'decompositions_per_term',
               'candidates_per_seed', 'candidates_per_bucket', 'candidates_total', 'n_holes_max', 'source_terms_max',
               'source_multiplicity_max', 'source_operators_max', 'target_operators_max', 'pattern_depth_max',
               'mon_exponent_max', 'node_arity_max', 'rematch_unify_states_max', 'alias_n_max', 'alias_time_cap_s',
               'match_contexts_max', 'unify_states_max', 'match_attempts_max', 'rewrite_n_max', 'binding_degree_max',
               'binding_exponent_max')
