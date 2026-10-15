"""Beam search with one combined primitive + rewrite candidate list.

Same admission, visited, incumbent, tie-breaking and budget semantics as the inherited `search.beam_search`
(identical when no relations are given; tested). With relations, every expanded state is matched, primitive and
rewrite candidates are scored in one encoder pass, concatenated, ranked once and truncated once by K_act. A selected
rewrite is lowered, spliced and exact-verified at application; a verification failure is counted and never admitted.
One admitted successor = 1 outer expansion, whatever produced it. No rewrite quota, no second beam, no cost discount.
"""

import resource
import sys
import time
from dataclasses import asdict, dataclass

from . import v1
from .rewrite_match import Candidate, apply_candidate, match_state


@dataclass
class RewriteSearchSummary:
    outer_expansions: int = 0
    primitive_applies: int = 0
    rewrite_applies: int = 0
    candidates_scored: int = 0
    primitive_candidates: int = 0
    rewrite_generated: int = 0            # deduped matches before the output cap
    rewrite_surviving_cap: int = 0        # scored rewrite candidates
    rewrite_states_matched: int = 0
    rewrite_states_cut_output: int = 0
    rewrite_selected: int = 0             # rewrite candidates inside the K_act truncation
    rewrite_rejected_noop: int = 0
    rewrite_rejected_other: int = 0
    exact_verification_failures: int = 0
    rewrite_revisits: int = 0
    matcher_unify_states: int = 0
    verifier_calls: int = 0
    cpu_time: float = 0.0
    wall_time: float = 0.0
    peak_rss: float = 0.0
    stop_reason: str = ''
    steps: int = 0
    incumbent_found_at: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform == 'darwin' else 1024)


def beam_search(arena, task, start: int, scorer, ranker, budget, schemas: list | None = None, cfg=None,
                record: bool = False) -> tuple[int, RewriteSearchSummary, dict]:
    assert min(budget.B, budget.K_act) > 0 and min(budget.T_max, budget.max_expansions) >= 0
    wall0, cpu0 = time.perf_counter(), time.process_time()
    verify0 = arena.verifier_count
    s = RewriteSearchSummary()
    tree = {start: dict(parent=None, action=None, cost=v1.cost(arena, start))} if record else {}
    beam, visited, incumbent = [start], {start}, start
    s.stop_reason = 'T_max'
    for step in range(budget.T_max):
        if s.outer_expansions >= budget.max_expansions:
            s.stop_reason = 'max_expansions'
            break
        s.steps = step+1
        succ = {}
        for root in beam:
            info = v1.state_info(arena, root)
            actions = v1.enumerate_actions(arena, info, root, arena.calc_cfg)
            if schemas is None:
                cands: list[Candidate] = []
                scores = scorer.score(arena, info, root, actions)
                rw_scores: list[float] = []
            else:
                cands, work = match_state(arena, root, schemas, cfg)
                s.rewrite_generated += work['deduped']
                s.rewrite_surviving_cap += len(cands)
                s.rewrite_states_matched += bool(cands)
                s.rewrite_states_cut_output += work['cut_output'] > 0
                s.matcher_unify_states += work['unify_states']
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
                if isinstance(a, Candidate):
                    s.rewrite_selected += 1
                    root2, reason = apply_candidate(arena, root, a, task.target)
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
                if root2 is None or root2 in visited:
                    continue
                v1.verify(arena, task, root2)
                visited.add(root2)
                s.outer_expansions += 1
                if isinstance(a, Candidate):
                    s.rewrite_applies += 1
                else:
                    s.primitive_applies += 1
                c = v1.cost(arena, root2)
                succ[root2] = ranker.rank(arena, v1.state_info(arena, root2), root2), c, v1.digest(arena, root2)
                if record:
                    tree[root]['successors'].append(root2)
                    tree[root2] = dict(parent=root, action=a, cost=c)
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
    s.verifier_calls = arena.verifier_count-verify0
    s.wall_time, s.cpu_time, s.peak_rss = time.perf_counter()-wall0, time.process_time()-cpu0, _peak_rss_mb()
    return incumbent, s, tree
