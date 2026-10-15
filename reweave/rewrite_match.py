"""Runtime matching and exact application of parameterized relations.

Direct source matching at every reachable Sum context (digest order, `match_contexts_max`), bounded unification with
per-(context, relation) `unify_states_max` and per-state `match_attempts_max` caps; dedupe by
(site_key, schema_hash, target_signature) keeping the smallest binding digest; total order by stable hash; output cap
`rewrite_n_max`. Application: lower the target with the bindings, splice it into the context in place of the matched
source region, substitute through the runtime's state semantics, reject no-ops, exact-verify the whole successor.
"""

import hashlib
from collections import Counter
from dataclasses import dataclass

from . import v1
from .hashing import canonical_json
from . import rewrite_schema as S


class WorkBudget(S.Budget):
    """Per (context, schema) unification-state cap plus a per-state attempt cap shared across schemas."""
    __slots__ = ('state',)

    def __init__(self, cap: int, state: dict) -> None:
        super().__init__(cap)
        self.state = state

    def tick(self) -> None:
        self.states += 1
        self.state['attempts'] += 1
        if self.state['attempts'] > self.state['cap']:
            raise S.Cut('attempts')
        if self.states > self.cap:
            raise S.Cut('unify_states')


@dataclass
class Candidate:
    ctx: int
    schema: S.Schema
    beta: dict
    R_inst: Counter
    site_key: tuple
    target_signature: str
    binding_digest: str
    order_key: tuple

    @property
    def stable_hash(self) -> str:
        return self.order_key[0]


def match_state(arena, root: int, schemas: list, cfg) -> tuple[list, dict]:
    info = v1.state_info(arena, root)
    contexts = sorted(info.contexts, key=lambda h: v1.digest(arena, h))
    work = dict(contexts=0, attempts=0, cap=cfg.match_attempts_max, unify_states=0, cut_contexts=0,
                cut_unify_states=0, cut_attempts=0, cut_output=0, generated=0)
    if len(contexts) > cfg.match_contexts_max:
        work['cut_contexts'] = len(contexts)-cfg.match_contexts_max
        contexts = contexts[:cfg.match_contexts_max]
    found: dict = {}
    stop = False
    simultaneous = {sch.schema_hash: S.uses_simultaneous(sch) for sch in schemas}
    for ctx in contexts:
        if stop:
            break
        work['contexts'] += 1
        counts = dict(v1.items(arena.handles[ctx]))
        index = S.ChildIndex(arena, counts)
        ctx_digest = v1.digest(arena, ctx).hex()
        for sch in schemas:
            budget = WorkBudget(cfg.unify_states_max, work)
            trace: list = []
            try:
                for beta in S.assign(sch.source, index, dict(counts), {}, arena, cfg, budget, exact=False, trace=trace,
                                     simultaneous=simultaneous[sch.schema_hash]):
                    R_inst: Counter = Counter()
                    for (p, n), h in trace:
                        R_inst[h] += n
                    work['generated'] += 1
                    region_key = tuple(sorted((v1.digest(arena, h).hex(), c) for h, c in R_inst.items()))
                    site_key = ('rw', ctx_digest, region_key)
                    tsig = S.target_signature(sch, beta)
                    bdig = S.binding_digest(beta)
                    dkey = (site_key, sch.schema_hash, tsig)
                    order = (hashlib.sha256(canonical_json([site_key, sch.schema_hash, tsig, bdig]).encode()).hexdigest(),
                             site_key, sch.schema_hash, tsig, bdig)
                    prev = found.get(dkey)
                    if prev is None or bdig < prev.binding_digest:
                        found[dkey] = Candidate(ctx, sch, dict(beta), R_inst, site_key, tsig, bdig, order)
            except S.Cut as cut:
                work['unify_states'] += budget.states
                if cut.args and cut.args[0] == 'attempts':
                    work['cut_attempts'] += 1
                    stop = True
                    break
                work['cut_unify_states'] += 1
                continue
            work['unify_states'] += budget.states
    cands = sorted(found.values(), key=lambda c: c.order_key)
    work['deduped'] = len(cands)
    if len(cands) > cfg.rewrite_n_max:
        work['cut_output'] = len(cands)-cfg.rewrite_n_max
        cands = cands[:cfg.rewrite_n_max]
    work.pop('cap')
    return cands, work


def apply_candidate(arena, root: int, cand: Candidate, target_poly: dict) -> tuple[int | None, str]:
    try:
        T_inst = S.lower_region(cand.schema.target, cand.beta, arena)
    except (S.CapError, AssertionError):
        return None, 'cap'
    counts = Counter(dict(v1.items(arena.handles[cand.ctx])))
    counts.subtract(cand.R_inst)
    if any(c < 0 for c in counts.values()):
        return None, 'not_a_match'
    counts.update(T_inst)
    ctx2 = v1.mk_sum(arena, Counter({h: c for h, c in counts.items() if c > 0}))
    if ctx2 == cand.ctx:
        return None, 'noop'
    root2 = v1.substitute(arena, root, cand.ctx, ctx2)
    if root2 == root:
        return None, 'noop'
    if not v1.poly_equal(v1.hpoly(arena, root2), target_poly):
        return None, 'inexact'
    return root2, 'ok'


def provenance(arena, root: int, cand: Candidate) -> str:
    if cand.ctx != root:
        return 'interior'
    return 'root_whole' if cand.R_inst == Counter(dict(v1.items(arena.handles[cand.ctx]))) else 'root_partial'


# ---------------------------------------------------------------- runtime action record


@dataclass(frozen=True)
class RewriteAction:
    """Hashable runtime record of one rewrite candidate (identity = stable hash of site, schema, target, bindings)."""
    stable_hash: str
    schema_hash: str
    ctx: int
    region: tuple          # ((handle, count), ...) sorted by digest
    beta: tuple            # ((hole, exponent vector), ...)
    target_signature: str


def action_of(arena, cand: Candidate) -> RewriteAction:
    region = tuple(sorted(cand.R_inst.items(), key=lambda hc: v1.digest(arena, hc[0])))
    return RewriteAction(cand.stable_hash, cand.schema.schema_hash, cand.ctx, region,
                         tuple(sorted((i, tuple(b)) for i, b in cand.beta.items())), cand.target_signature)


def candidate_of(action: RewriteAction, schema: S.Schema) -> Candidate:
    """Rebuild the application view of a recorded action (for exact application only; ordering fields are empty)."""
    return Candidate(action.ctx, schema, {i: tuple(b) for i, b in action.beta}, Counter(dict(action.region)),
                     (), action.target_signature, '', (action.stable_hash,))
