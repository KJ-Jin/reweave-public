"""The canonical loop in symmetry-safe identity mode.

Same loop as `canon.search.beam_search` (same scorer, value ranker, admission policy, budget, latent configuration and
exact verification) with the state space and action identity made renaming-safe:

  * the task is relabelled into its canonical frame T* (`TaskFrameQ`); every state is a canonical DAG in one arena;
  * actions are quotient CLASSES (primitive, visible rewrite, latent rewrite), each closed under the automorphisms of
    the state; one representative per class is scored and applied;
  * every successor is exact-verified, optionally normalized, canonicalized, deduplicated and interned.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .. import v1
from ..candidate_policy import CandidatePolicy, select
from ..rewrite_latent import LatentCandidate, apply_latent
from ..rewrite_match import Candidate, apply_candidate
from ..rewrite_search import _peak_rss_mb
from . import action_record as AR
from . import candidate_set as CSR
from . import exprparam as EP
from . import semantic as SN
from .config import IntegrationConfig
from .exprparam_identity import handle_lane_identity
from .identity import IDENT, TaskFrameQ, q_primitive, q_rewrite, relabel_into
from .latent_identity import dedupe_key, q_latent
from .search import CanonSearchSummary


@dataclass
class IdentitySearchSummary(CanonSearchSummary):
    prim_classes: int = 0
    rw_classes: int = 0
    lx_classes: int = 0
    classes_scored: int = 0
    concrete_members_represented: int = 0
    canonical_revisits: int = 0
    view_classes: int = 0
    view_applies: int = 0
    identity_cpu: float = 0.0             # generation + canonicalization CPU (inside cpu_time)
    cross_generator_duplicates: int = 0   # same canonical successor from two generators (latent / view / visible / primitive)


def run_identity(task, scorer, ranker, budget, schemas: list, cfg, record: bool = False, latent=None, policy=None,
                 icfg: IntegrationConfig | None = None, views=None) -> dict:
    icfg = (icfg or IntegrationConfig(identity_mode=True)).effective()
    assert icfg.identity_mode and schemas is not None
    wall0, cpu0 = time.perf_counter(), time.process_time()
    tf0 = TaskFrameQ(task.target)
    tfc = TaskFrameQ(canonical_of=tf0)
    ca = tfc.carena
    Tpoly = {v1.pack(vec): c for vec, c in tf0.T.items()}
    task_c = v1.Task(Tpoly, task.n_vars, task.c_gen, task.meta)
    verify0 = ca.verifier_count
    s = IdentitySearchSummary()
    start = v1.canonical_start(ca, Tpoly)
    cs0 = tfc.state(ca, start)
    assert cs0.croot == start, 'the canonical start is its own canonical form'
    tree = {start: dict(parent=None, action=None, cost=v1.cost(ca, start), upsilon=IDENT, norm=[], kind='START')} if record else {}
    beam, visited, incumbent = [start], {start}, start
    sem = None
    if icfg.semantic_interning:
        sem = SN.SemanticCtx(ca, Tpoly)
        sem.port.intern_state(start, 0, 'CURRENT')
    vctx = (views.context(ca, tfc, task_c, Tpoly, schemas, cfg, sem, scorer, icfg.view_index)
            if (views is not None and icfg.view_reuse) else None)
    gen_log: list = []
    s.stop_reason = 'T_max'
    for step in range(budget.T_max):
        if s.outer_expansions >= budget.max_expansions:
            s.stop_reason = 'max_expansions'
            break
        s.steps = step+1
        succ = {}
        for root in beam:
            info = v1.state_info(ca, root)
            t0 = time.process_time()
            cs = tfc.state(ca, root)
            assert cs.croot == root
            n_log = len(gen_log)
            pcl = q_primitive(tfc, cs, ca.calc_cfg, gen_log)
            wcl = q_rewrite(tfc, cs, schemas, cfg, gen_log, keep_members=latent is not None)
            lcl = []
            if latent is not None:
                vkeys = {dedupe_key(ca, m) for c in wcl for m in c.members}
                lcl = q_latent(tfc, cs, schemas, cfg, latent, vkeys, gen_log)
            hreps, hcls = [], []
            if icfg.reachable_handle_bind:                                            # T6 block 1 of 3: the handle lane
                hreps, hst = handle_lane_identity(tfc, cs, schemas, cfg, latent, gen_log)
                EP.count_lane(s, hst)
                hcls = hst['classes']
            vac = views.actions(vctx, cs, root, info, step) if vctx is not None else []
            s.identity_cpu += time.process_time()-t0
            actions = [c.rep for c in pcl]
            vis = [c.rep for c in wcl]
            lat = [c.rep for c in lcl]
            s.prim_classes += len(pcl)
            s.rw_classes += len(wcl)
            s.lx_classes += len(lcl)
            s.view_classes += len(vac)
            s.latent_generated += len(lat)
            s.rewrite_generated += len(vis)
            cands = select(vis+lat, cfg.rewrite_n_max, policy or CandidatePolicy()) if latent is not None else vis
            s.rewrite_surviving_cap += len(cands)
            s.rewrite_states_matched += bool(cands)
            s.concrete_members_represented += sum(c.size for c in pcl)+sum(c.size for c in wcl)+sum(c.size for c in lcl)
            if icfg.generic_exprparam_binding:                                        # T6 block 2 of 3: the boundary
                legacy_admitted = len(cands)
                cands = cands+hreps
                s.concrete_members_represented += sum(c.size for c in hcls)
                recs = AR.records(ca, actions, cands)                                  # ActionRecords -> legacy adapter
                sc = AR.score_records(scorer, ca, info, root, recs)
                scores, rw_scores = sc[:len(actions)], sc[len(actions):]
                dg = cs.digest.hex()
                ents = [e for e in gen_log[n_log:] if e.get('canon_digest') == dg]      # this state's generation log
                pl, ll = CSR.identity_lanes(ents, len(actions), len(vis)+len(lat), legacy_admitted,
                                            CSR.divisor_cap_contexts(ca, root, cfg, latent) if latent is not None else 0,
                                            latent is not None)
                lanes = dict(primitive=pl, legacy_rewrite=ll,
                             handle=(CSR.handle_lane_record(hst, len(hreps)) if icfg.reachable_handle_bind
                                     else CSR.disabled()))
                csr = CSR.record(dg, step, lanes, CSR.selectable(recs))
                CSR.count(s, csr, len(recs))
                if record:
                    tree[root].update(candidate_set=csr, action_records=recs)
            else:
                scores, rw_scores = scorer.score_combined(ca, info, root, actions, cands)
            v_scores = views.scores(vctx, root, vac) if vac else []
            assert len(scores) == len(actions) and len(rw_scores) == len(cands) and len(v_scores) == len(vac)
            s.candidates_scored += len(actions)+len(cands)+len(vac)
            s.classes_scored += len(actions)+len(cands)+len(vac)
            s.primitive_candidates += len(actions)
            ranked = sorted([(a, x, 0, v1.action_key(ca, a)) for a, x in zip(actions, scores)] +
                            [(c, x, 1, c.order_key) for c, x in zip(cands, rw_scores)] +
                            [(va, x, 1, va.cand.order_key) for va, x in zip(vac, v_scores)],
                            key=lambda x: (-x[1], x[2], x[3]))
            chosen = [x[0] for x in ranked[:budget.K_act]]
            if record:
                tree[root].update(chosen=chosen, successors=[], n_prim_classes=len(pcl), n_rw_classes=len(wcl),
                                  n_lx_classes=len(lcl), n_view_classes=len(vac), scores=list(scores),
                                  rewrite_scores=list(rw_scores))
            produced: dict = {}
            for a in chosen:
                kind = 'ORD'
                if views is not None and isinstance(a, views.ViewAction):
                    r2, kind = a.succ, 'VIEW'
                elif isinstance(a, (Candidate, LatentCandidate)):
                    s.rewrite_selected += 1
                    s.latent_selected += isinstance(a, LatentCandidate)
                    if icfg.generic_exprparam_binding:                                # T6 block 3 of 3
                        r2, reason = EP.apply_any_g(ca, root, a, Tpoly)
                        s.hb_selected += EP.has_hb(a.beta)
                        s.hb_rejected_dependency += reason.startswith('dep_') or reason == 'rebuild'
                    else:
                        r2, reason = (apply_latent if isinstance(a, LatentCandidate) else apply_candidate)(ca, root, a, Tpoly)
                    if reason == 'inexact':
                        s.exact_verification_failures += 1
                    elif reason == 'noop':
                        s.rewrite_rejected_noop += 1
                    elif reason != 'ok':
                        s.rewrite_rejected_other += 1
                else:
                    r2 = v1.apply_action(ca, task_c, info, root, a)
                if r2 is None:
                    continue
                v1.verify(ca, task_c, r2)
                norm_ev = []
                if icfg.normalization:
                    pre = info.reach_h if kind != 'VIEW' else (info.reach_h | SN.reach(ca, a.h2))
                    r2, norm_ev = SN.normalize(sem, r2, SN.built_handles(ca, [r2], pre), step)
                    if norm_ev:
                        v1.verify(ca, task_c, r2)
                        s.norm_successors += 1
                        s.norm_events += len(norm_ev)
                t1 = time.process_time()
                cs2 = tfc.state(ca, r2)
                s.identity_cpu += time.process_time()-t1
                c2 = cs2.croot
                gen = 'VIEW' if kind == 'VIEW' else 'LX' if isinstance(a, LatentCandidate) else 'RW' if isinstance(a, Candidate) else 'PRIM'
                if c2 in produced and produced[c2] != gen:
                    s.cross_generator_duplicates += 1
                produced.setdefault(c2, gen)
                if c2 in visited:
                    s.canonical_revisits += 1
                    s.norm_collapsed_revisits += bool(norm_ev)
                    continue
                v1.verify(ca, task_c, c2)
                visited.add(c2)
                s.outer_expansions += 1
                if kind == 'VIEW':
                    s.view_applies += 1
                elif isinstance(a, (Candidate, LatentCandidate)):
                    s.rewrite_applies += 1
                    if icfg.generic_exprparam_binding:
                        s.hb_applies += EP.has_hb(a.beta)
                    s.latent_applies += isinstance(a, LatentCandidate)
                else:
                    s.primitive_applies += 1
                c = v1.cost(ca, c2)
                succ[c2] = ranker.rank(ca, v1.state_info(ca, c2), c2), c, v1.digest(ca, c2)
                if record:
                    tree[root]['successors'].append(c2)
                    tree[c2] = dict(parent=root, action=a, cost=c, upsilon=cs2.f0, concrete=r2, norm=norm_ev, kind=kind)
                if sem is not None:
                    org = 'LEARNED_REWRITE' if (kind == 'VIEW' or isinstance(a, (Candidate, LatentCandidate))) else 'CURRENT'
                    sem.port.intern_state(c2, step, org, norm_roots=[relabel_into(ca, e['h2'], cs2.f0, ca, {}) for e in norm_ev])
                if c < v1.cost(ca, incumbent):
                    incumbent = c2
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
    s.verifier_calls = ca.verifier_count-verify0
    s.wall_time, s.cpu_time, s.peak_rss = time.perf_counter()-wall0, time.process_time()-cpu0, _peak_rss_mb()
    return dict(arena=ca, start=start, incumbent=incumbent, summary=s, tree=tree, target=Tpoly, tf0=tf0, tfc=tfc,
                task_c=task_c, gen_log=gen_log, sem=sem, vctx=vctx)
