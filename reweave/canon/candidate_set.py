"""The per-state CandidateSetRecord: what was considered, what was admitted, and every truncation, per lane.

Completeness belongs to the CANDIDATE SET of a state, never to an action's identity. One record per expanded state,
independently per enabled generation lane:

    primitive       primitive enumeration (identity mode: primitive classes)
    legacy_rewrite  the visible matcher + latent generator + admission policy
    handle          the fair-scheduled REACHABLE_HANDLE_BIND lane + semantic-class admission

`no_action_exists` holds only when a lane is enabled, completely generated and empty; a truncated lane's absence of an
action is never evidence that none exists.
"""

from __future__ import annotations

from collections import Counter

from .. import v1

CSR_VERSION = 'reweave-candidate-set-v1'
GENERATION_CAUSES = ('context_cap', 'work_cap', 'unify_cap', 'divisor_cap', 'latent_context_cap', 'latent_work_cap',
                     'latent_unify_cap', 'latent_candidate_cap', 'orbit_cap', 'view_attempt_cap', 'view_pair_cap')
ADMISSION_CAUSES = ('enumeration_cap', 'output_cap', 'admission_cap', 'class_admission_cap', 'realization_cap',
                    'class_realization_cap')


def lane(enabled: bool, considered: int, admitted: int, truncation: dict, **extra) -> dict:
    trunc = {k: int(v) for k, v in sorted(truncation.items()) if v}
    unknown = set(trunc)-set(GENERATION_CAUSES)-set(ADMISSION_CAUSES)
    assert not unknown, f'undeclared truncation cause {unknown}'
    complete = not any(k in GENERATION_CAUSES for k in trunc)
    return dict(enabled=bool(enabled), generation_complete=complete,
                admission_truncated=any(k in ADMISSION_CAUSES for k in trunc), considered=int(considered),
                admitted=int(admitted), truncation=trunc,
                no_action_exists=bool(enabled and complete and considered == 0), **extra)


def disabled() -> dict:
    return lane(False, 0, 0, {})


def record(state_digest: str, step: int, lanes: dict, selectable: Counter) -> dict:
    live = [x for x in lanes.values() if x['enabled']]
    return dict(v=CSR_VERSION, state=dict(digest=state_digest, step=int(step)), lanes=lanes,
                selectable={k: int(selectable.get(k, 0)) for k in ('PRIMITIVE', 'VISIBLE_REWRITE', 'LATENT_REWRITE',
                                                                   'VIEW_REWRITE')},
                generation_complete=all(x['generation_complete'] for x in live))


def selectable(recs: list) -> Counter:
    return Counter(r.kind for r in recs)


def count(s, csr: dict, n_records: int) -> None:
    """CandidateSetRecord bookkeeping on a search summary."""
    ln = csr['lanes']
    s.csr_states += 1
    s.csr_generation_incomplete_states += not csr['generation_complete']
    s.csr_legacy_incomplete_states += not ln['legacy_rewrite']['generation_complete']
    s.csr_primitive_admission_truncated_states += ln['primitive']['admission_truncated']
    s.hb_lane_incomplete_states += ln['handle']['enabled'] and not ln['handle']['generation_complete']
    s.action_records += n_records


# ---------------------------------------------------------------- lane builders


def divisor_cap_contexts(arena, root: int, cfg, lcfg) -> int:
    """Contexts whose frozen quotient-view construction cut eligible divisors at `divisors_max` (read-only)."""
    from .exprparam import divisor_truncated
    ctxs = sorted(v1.state_info(arena, root).contexts, key=lambda h: v1.digest(arena, h))[:cfg.match_contexts_max]
    return sum(divisor_truncated(arena, c, lcfg) for c in ctxs)


def legacy_lane_concrete(arena, root: int, cfg, lcfg, work: dict, lwork: dict | None, n_latent: int,
                         admitted: int) -> dict:
    """The mainline generators' own work dicts, as they ran (nothing is re-run except the read-only divisor count)."""
    t = dict(context_cap=work.get('cut_contexts', 0), work_cap=work.get('cut_attempts', 0),
             unify_cap=work.get('cut_unify_states', 0), output_cap=work.get('cut_output', 0))
    considered = work.get('deduped', 0)
    if lwork is not None:
        t.update(latent_context_cap=lwork.get('cut_contexts', 0), latent_work_cap=lwork.get('cut_attempts', 0),
                 latent_unify_cap=lwork.get('cut_unify_states', 0), latent_candidate_cap=lwork.get('cut_candidates', 0),
                 divisor_cap=divisor_cap_contexts(arena, root, cfg, lcfg))
        considered += n_latent
    t['admission_cap'] = max(0, considered-work.get('cut_output', 0)-admitted)
    return lane(True, considered, admitted, t)


def handle_lane_record(st: dict, admitted: int, identity: bool = False) -> dict:
    """The fair handle lane: generation caps of both sub-lanes, realization / class / admission caps, and the
    per-family schedule (whether each family finished, its work, turns and matches)."""
    w, lw, a = st['visible'], st['latent'], st['admission']
    t = dict(context_cap=w.get('cut_contexts', 0), work_cap=w.get('work_cap', 0), unify_cap=w.get('cut_unify_states', 0),
             latent_context_cap=lw.get('cut_contexts', 0), latent_work_cap=lw.get('work_cap', 0),
             latent_unify_cap=lw.get('cut_unify_states', 0), latent_candidate_cap=lw.get('candidate_cap', 0),
             divisor_cap=lw.get('divisor_cap', 0), orbit_cap=lw.get('orbit_cap', 0)+w.get('orbit_cap', 0),
             realization_cap=w.get('realizations_truncated', 0)+lw.get('realizations_truncated', 0),
             class_realization_cap=a.get('class_realizations_dropped', 0),
             admission_cap=a.get('dropped', 0)-a.get('class_realizations_dropped', 0))
    fams = dict(visible=w.get('family_records', []), latent=lw.get('family_records', []))
    return lane(True, a.get('realizations', 0), admitted, t, families=fams,
                inadmissible_filtered=int(w.get('rejected_inadmissible', 0)+lw.get('rejected_inadmissible', 0)
                                          + w.get('inadmissible_matches', 0)+lw.get('inadmissible_matches', 0)),
                merged_cross_surface=int(w.get('merged_cross_surface', 0)+lw.get('merged_cross_surface', 0)),
                rounds=dict(visible=w.get('rounds', 0), latent=lw.get('rounds', 0)))


def primitive_lane_concrete(n_actions: int, truncation_entry) -> dict:
    """`truncation_entry` is V1's own `arena.truncations` entry (total, kept) appended by this state's enumeration, or
    None when it did not truncate."""
    if truncation_entry is None:
        return lane(True, n_actions, n_actions, {})
    total, kept = truncation_entry
    return lane(True, total, n_actions, dict(enumeration_cap=total-kept))


def identity_lanes(entries: list, n_prim: int, legacy_considered: int, legacy_admitted: int,
                   divisor_cap: int, latent_on: bool) -> tuple:
    """Primitive and legacy lanes of an identity-mode state from its generation-log entries (T4 classes)."""
    prim = next((e for e in entries if e.get('stage') == 'prim'), None)
    rw = next((e for e in entries if e.get('stage') == 'rw' and 'hb_found' not in e), None)
    lx = next((e for e in entries if e.get('stage') == 'lx' and 'hb_found' not in e), None)
    pl = lane(True, prim['classes'], n_prim, dict(enumeration_cap=prim['classes']-prim['retained_classes']
                                                   if prim['truncated'] else 0))
    cw = rw['canon_work']
    t = dict(context_cap=cw['cut_contexts'], work_cap=cw['cut_attempts'], unify_cap=cw['cut_unify_states'],
             class_admission_cap=(rw['classes']-rw['retained_classes']) if rw['truncated'] else 0)
    if latent_on and lx is not None:
        t.update(latent_context_cap=lx['cut_contexts'], latent_work_cap=lx['cut_attempts'],
                 latent_unify_cap=lx['cut_unify_states'], latent_candidate_cap=lx['cut_candidates'],
                 orbit_cap=int(lx['truncated']), divisor_cap=divisor_cap)
    t['admission_cap'] = max(0, legacy_considered-legacy_admitted)
    return pl, lane(True, legacy_considered, legacy_admitted, t)
