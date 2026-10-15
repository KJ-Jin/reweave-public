"""The canonical search loop.

With `icfg` None or all switches OFF this is `rewrite_search_latent.beam_search` line for line (tested). Each
architectural component (semantic interning, archive retrieval, generic ExprParam binding, ActionRecord scoring) is a
guarded block behind its own switch.
"""

import time
from collections import Counter
from dataclasses import dataclass

from .. import v1
from ..candidate_policy import CandidatePolicy, select
from ..rewrite_latent import LatentCandidate, apply_latent, dedupe_against_visible, latent_candidates
from ..rewrite_match import Candidate, apply_candidate, match_state
from ..rewrite_search import _peak_rss_mb
from ..rewrite_search_latent import LatentSearchSummary
from . import action_record as AR
from . import candidate_set as CSR
from . import exprparam as EP
from . import semantic as SN
from .archive import build_archive, match_state_retrieved
from .config import IntegrationConfig


@dataclass
class CanonSearchSummary(LatentSearchSummary):
    """The mainline summary plus integration counters (all zero while every switch is OFF). The normalization
    counter names are those of the evaluated A/B wrapper (`exp-normalization-ab`), so rows compare field by field."""
    norm_successors: int = 0              # successors (admitted or not) with >= 1 accepted normalization
    norm_events: int = 0                  # accepted normalizations
    norm_attempts: int = 0
    norm_rejected_not_cheaper: int = 0
    norm_failures: int = 0                # instantiate / exactness failures (must be 0)
    norm_collapsed_revisits: int = 0      # pre-normalization successor new, normalized successor already visited
    norm_cpu: float = 0.0
    portfolio_classes: int = 0
    portfolio_reps_inserted: int = 0
    portfolio_evicted: int = 0
    portfolio_max_reps: int = 0
    keys_var_guard_skipped: int = 0
    keys_frame_excluded: int = 0
    keys_hash_collision: int = 0
    retrieval_pairs_offered: int = 0      # T3: (context, surface) pairs that reached the exact visible matcher
    retrieval_contexts_no_schema: int = 0
    gen_matcher_contexts: int = 0         # T6 handle lane (its own work allowance): visible matcher contexts
    gen_matcher_bindings: int = 0         # handle-bound visible matches before realization / dedupe
    gen_matcher_unify_states: int = 0
    gen_latent_bindings: int = 0          # handle-bound latent matches mapped back to originals
    gen_latent_unify_states: int = 0
    gen_handle_choices: int = 0           # handle-unit choices ticked by the binder (visible + latent)
    gen_keys_computed: int = 0            # exact semantic keys computed (memoized per arena)
    gen_key_collisions: int = 0           # key digest equal, exact polynomials different (split, never merged)
    gen_matcher_cuts: int = 0             # handle-lane visible cut_unify_states + cut_attempts (frozen sizes)
    gen_latent_cuts: int = 0              # handle-lane latent cut_unify_states + cut_attempts + cut_candidates
    hb_lane_incomplete_states: int = 0    # states whose handle lane generation was incomplete (CandidateSetRecord)
    hb_rejected_inadmissible: int = 0     # matches / realizations refused at generation by the apply-time predicate
    hb_merged_cross_surface: int = 0      # identical concrete successors from several surfaces merged (provenance kept)
    hb_families_truncated: int = 0        # family schedules the lane allowance ended before they finished
    hb_lane_rounds: int = 0
    csr_states: int = 0                   # T6A.2 CandidateSetRecords (generic profiles)
    csr_generation_incomplete_states: int = 0
    csr_legacy_incomplete_states: int = 0
    csr_primitive_admission_truncated_states: int = 0
    action_records: int = 0               # ActionRecords built at the scoring boundary (generic profiles)
    hb_visible_candidates: int = 0        # distinct concrete handle-bound visible candidates (realizations)
    hb_latent_candidates: int = 0         # the same, latent (after dedupe against the handle lane's visible ones)
    hb_realizations_truncated: int = 0    # matches whose realization product exceeded REALIZATIONS_MAX
    hb_semantic_classes: int = 0          # semantic action classes offered to the handle-lane admission
    hb_admitted: int = 0                  # realizations admitted to scoring (separate cap: rewrite_n_max)
    hb_admission_dropped: int = 0
    hb_class_realizations_dropped: int = 0
    hb_selected: int = 0
    hb_applies: int = 0
    hb_rejected_dependency: int = 0       # fail-closed apply-time dependency / reuse rejections (must stay 0)


def beam_search(arena, task, start: int, scorer, ranker, budget, schemas: list | None = None, cfg=None,
                record: bool = False, latent=None, policy=None, icfg: IntegrationConfig | None = None,
                archive=None) -> tuple:
    icfg = (icfg or IntegrationConfig()).effective()
    assert not (icfg.identity_mode or icfg.view_reuse), \
        'concrete-mode loop: identity mode and views run in the identity loop'
    assert archive is not None or not icfg.archive_retrieval, 'archive_retrieval needs the certified archive'
    rcache: dict = {}
    rst = Counter()
    assert min(budget.B, budget.K_act) > 0 and min(budget.T_max, budget.max_expansions) >= 0
    wall0, cpu0 = time.perf_counter(), time.process_time()
    verify0 = arena.verifier_count
    s = CanonSearchSummary()
    tree = {start: dict(parent=None, action=None, cost=v1.cost(arena, start))} if record else {}
    beam, visited, incumbent = [start], {start}, start
    sem = None
    if icfg.semantic_interning:                                                    # T2 block 1 of 3
        sem = SN.SemanticCtx(arena, task.target)
        sem.port.intern_state(start, 0, 'CURRENT')
    s.stop_reason = 'T_max'
    for step in range(budget.T_max):
        if s.outer_expansions >= budget.max_expansions:
            s.stop_reason = 'max_expansions'
            break
        s.steps = step+1
        succ = {}
        for root in beam:
            info = v1.state_info(arena, root)
            if icfg.generic_exprparam_binding:                                        # T6 block 0: V1's truncation log
                ntr0 = len(arena.truncations)
            actions = v1.enumerate_actions(arena, info, root, arena.calc_cfg)
            if schemas is None:
                cands: list[Candidate] = []
                scores = scorer.score(arena, info, root, actions)
                rw_scores: list[float] = []
            else:
                if icfg.archive_retrieval:                                            # T3: visible matcher only
                    cands, work = match_state_retrieved(arena, root, archive, cfg, None, rst, rcache)
                else:
                    cands, work = match_state(arena, root, schemas, cfg)
                if latent is not None:
                    lat, lwork = latent_candidates(arena, root, schemas, cfg, latent)
                    lat = dedupe_against_visible(arena, lat, cands)
                    s.latent_attempts += lwork['attempts']
                    s.latent_generated += len(lat)
                    cands = select(cands+lat, cfg.rewrite_n_max, policy or CandidatePolicy())
                s.rewrite_generated += work['deduped']
                s.rewrite_surviving_cap += len(cands)
                s.rewrite_states_matched += bool(cands)
                s.rewrite_states_cut_output += work['cut_output'] > 0
                s.matcher_unify_states += work['unify_states']
                if icfg.reachable_handle_bind:                                        # T6 block 1 of 3: the handle lane
                    hc, hst = EP.handle_lane(arena, root, schemas, cfg, latent)
                    EP.count_lane(s, hst)
                    cands = cands+hc
                if icfg.generic_exprparam_binding:                                    # T6 block 2 of 3: the boundary
                    recs = AR.records(arena, actions, cands)                           # ActionRecords -> legacy adapter
                    sc = AR.score_records(scorer, arena, info, root, recs)
                    scores, rw_scores = sc[:len(actions)], sc[len(actions):]
                    lanes = dict(
                        primitive=CSR.primitive_lane_concrete(
                            len(actions), arena.truncations[ntr0] if len(arena.truncations) > ntr0 else None),
                        legacy_rewrite=CSR.legacy_lane_concrete(
                            arena, root, cfg, latent, work, lwork if latent is not None else None,
                            len(lat) if latent is not None else 0, sum(not EP.has_hb(c.beta) for c in cands)),
                        handle=(CSR.handle_lane_record(hst, len(hc)) if icfg.reachable_handle_bind else CSR.disabled()))
                    csr = CSR.record(v1.digest(arena, root).hex(), step, lanes, CSR.selectable(recs))
                    CSR.count(s, csr, len(recs))
                    if record:
                        tree[root].update(candidate_set=csr, action_records=recs)
                else:
                    scores, rw_scores = scorer.score_combined(arena, info, root, actions, cands)
            assert len(scores) == len(actions) and len(rw_scores) == len(cands)
            s.candidates_scored += len(actions)+len(cands)
            s.primitive_candidates += len(actions)
            ranked = sorted([(a, sc, 0, v1.action_key(arena, a)) for a, sc in zip(actions, scores)] +
                            [(c, sc, 1, c.order_key) for c, sc in zip(cands, rw_scores)],
                            key=lambda x: (-x[1], x[2], x[3]))
            chosen = [x[0] for x in ranked[:budget.K_act]]
            if record:
                tree[root].update(actions=actions, scores=scores, rewrites=cands, rewrite_scores=rw_scores,
                                  chosen=chosen, successors=[])
            for a in chosen:
                if isinstance(a, (Candidate, LatentCandidate)):
                    s.rewrite_selected += 1
                    s.latent_selected += isinstance(a, LatentCandidate)
                    if icfg.generic_exprparam_binding:                                # T6 block 3 of 3
                        root2, reason = EP.apply_any_g(arena, root, a, task.target)
                        s.hb_selected += EP.has_hb(a.beta)
                        s.hb_rejected_dependency += reason.startswith('dep_') or reason == 'rebuild'
                    else:
                        root2, reason = (apply_latent if isinstance(a, LatentCandidate) else
                                         apply_candidate)(arena, root, a, task.target)
                    if reason == 'inexact':
                        s.exact_verification_failures += 1
                    elif reason == 'noop':
                        s.rewrite_rejected_noop += 1
                    elif reason != 'ok':
                        s.rewrite_rejected_other += 1
                    if root2 is not None and root2 in visited:
                        s.rewrite_revisits += 1
                else:
                    root2 = v1.apply_action(arena, task, info, root, a)
                norm_ev = []
                if icfg.normalization and root2 is not None:                             # T2 block 2 of 3
                    v1.verify(arena, task, root2)
                    pre = root2
                    root2, norm_ev = SN.normalize(sem, root2, SN.built_handles(arena, [root2], info.reach_h), step)
                    if norm_ev:
                        v1.verify(arena, task, root2)
                        s.norm_successors += 1
                        s.norm_events += len(norm_ev)
                        s.norm_collapsed_revisits += (pre not in visited and root2 in visited)
                if root2 is None or root2 in visited:
                    continue
                v1.verify(arena, task, root2)
                visited.add(root2)
                s.outer_expansions += 1
                if isinstance(a, (Candidate, LatentCandidate)):
                    s.rewrite_applies += 1
                    s.latent_applies += isinstance(a, LatentCandidate)
                    if icfg.generic_exprparam_binding:
                        s.hb_applies += EP.has_hb(a.beta)
                else:
                    s.primitive_applies += 1
                c = v1.cost(arena, root2)
                succ[root2] = ranker.rank(arena, v1.state_info(arena, root2), root2), c, v1.digest(arena, root2)
                if record:
                    tree[root]['successors'].append(root2)
                    tree[root2] = dict(parent=root, action=a, cost=c)
                    if sem is not None:
                        tree[root2]['norm'] = norm_ev
                if sem is not None:                                                        # T2 block 3 of 3
                    org = 'LEARNED_REWRITE' if isinstance(a, (Candidate, LatentCandidate)) else 'CURRENT'
                    sem.port.intern_state(root2, step, org, norm_roots=[e['h2'] for e in norm_ev])
                if c < v1.cost(arena, incumbent):
                    incumbent = root2
                    s.incumbent_found_at = s.outer_expansions
                if s.outer_expansions >= budget.max_expansions:
                    break
            if s.outer_expansions >= budget.max_expansions:
                break
        if s.outer_expansions >= budget.max_expansions:
            s.stop_reason = 'max_expansions'
        if not succ:
            s.stop_reason = 'no_successors'
            break
        beam = sorted(succ, key=succ.get)[:budget.B]
    if sem is not None:
        ps, ks = sem.port.stats, sem.hk.stats
        s.norm_attempts, s.norm_rejected_not_cheaper = ps['norm_attempts'], ps['norm_rejected_not_cheaper']
        s.norm_failures, s.norm_cpu = ps['failures'], sem.norm_cpu
        s.portfolio_classes, s.portfolio_reps_inserted = len(sem.port.classes), ps['reps_inserted']
        s.portfolio_evicted, s.portfolio_max_reps = ps['evicted'], sem.port.max_reps
        s.keys_var_guard_skipped, s.keys_frame_excluded = ks['var_guard_skipped'], ks['frame_excluded']
        s.keys_hash_collision = ps['hash_collision']
    s.retrieval_pairs_offered, s.retrieval_contexts_no_schema = rst['pairs_offered'], rst['contexts_no_schema']
    s.verifier_calls = arena.verifier_count-verify0
    s.wall_time, s.cpu_time, s.peak_rss = time.perf_counter()-wall0, time.process_time()-cpu0, _peak_rss_mb()
    return incumbent, s, tree


_ARCHIVES: dict = {}


def archive_for(schemas: list, cfg):
    """The certified archive over the given schema list, built once per process (deterministic insertion order)."""
    key = (tuple(sorted(s.schema_hash for s in schemas)), cfg.hash())
    a = _ARCHIVES.get(key)
    if a is None:
        a = _ARCHIVES[key] = build_archive(schemas, cfg)
    return a


def run(task, scorer, ranker, budget, schemas: list | None = None, cfg=None, record: bool = False, latent=None,
        policy=None, icfg: IntegrationConfig | None = None, arena=None) -> dict:
    """Uniform entry point: {arena, start, incumbent, summary, tree, target}. Concrete mode uses a fresh arena (or the
    one given) at the task's canonical flat start."""
    icfg = (icfg or IntegrationConfig()).effective()
    if icfg.identity_mode:
        from .identity_search import run_identity
        from . import views as VW
        return run_identity(task, scorer, ranker, budget, schemas, cfg, record, latent, policy, icfg,
                            VW if icfg.view_reuse else None)
    if arena is None:
        arena = v1.Arena()
        arena.calc_cfg = v1.CalcCfg()
    start = v1.canonical_start(arena, task.target)
    archive = archive_for(schemas, cfg) if icfg.archive_retrieval else None
    inc, s, tree = beam_search(arena, task, start, scorer, ranker, budget, schemas, cfg, record, latent, policy, icfg,
                               archive)
    return dict(arena=arena, start=start, incumbent=inc, summary=s, tree=tree, target=task.target)
