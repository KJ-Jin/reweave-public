"""Deterministic candidate-admission policies.

A policy decides only WHICH candidates occupy a fixed number of slots at an admission stage. It never changes a
numeric cap, never generates or removes a candidate from the raw set, and never consults a model score, an immediate
gain, a continuation utility or any downstream outcome. It receives a candidate list and a cap, and nothing else.

Admission stages this applies to:

  S-CAP   the combined rewrite budget `RewriteConfig.rewrite_n_max`
  S-RET   the label-retention cap `RewriteConfig.rewrite_label_candidates_max`

Primitives participate in neither stage. `K_act` is downstream of both.

  CM-HASH                      the stable-hash prefix (the original behaviour)
  CM-SCHEMA-DIVERSITY          bucket by relation, stable-hash order inside each bucket, round-robin across buckets
  CM-ORIGIN-SCHEMA-DIVERSITY   bucket by (origin, schema), stable-hash order inside each bucket, round-robin across
                               buckets, so one prolific relation or origin cannot crowd out every other
"""

from collections import defaultdict
from dataclasses import dataclass

from .hashing import sha256_json

HASH = 'CM-HASH'
SCHEMA_DIVERSITY = 'CM-SCHEMA-DIVERSITY'
ORIGIN_SCHEMA_DIVERSITY = 'CM-ORIGIN-SCHEMA-DIVERSITY'
POLICIES = (HASH, SCHEMA_DIVERSITY, ORIGIN_SCHEMA_DIVERSITY)


@dataclass(frozen=True)
class CandidatePolicy:
    """`name` is the whole semantics; there are no tunable fields, by design."""
    name: str = HASH

    def __post_init__(self):
        assert self.name in POLICIES, self.name

    def as_dict(self) -> dict:
        return dict(policy=self.name, applies_to=['S-CAP', 'S-RET'], uses_score=False, uses_utility=False,
                    uses_outcome=False, within_bucket_order='frozen stable hash',
                    across_bucket_order='deterministic round-robin in canonical bucket-key order')

    def hash(self) -> str:
        return sha256_json(self.as_dict())


def is_latent(cand) -> bool:
    """Origin of a rewrite candidate. Structural type test, never a name or hash comparison."""
    from .rewrite_latent import LatentCandidate
    return isinstance(cand, LatentCandidate)


def bucket_key(cand, policy: CandidatePolicy) -> tuple:
    """Canonical, deterministic, sortable bucket key. Depends only on the candidate's semantic identity."""
    if policy.name == SCHEMA_DIVERSITY:
        return (cand.schema.schema_hash,)
    if policy.name == ORIGIN_SCHEMA_DIVERSITY:
        return ('latent' if is_latent(cand) else 'visible', cand.schema.schema_hash)
    raise AssertionError('CM-HASH has no buckets')


def select(cands: list, cap: int, policy: CandidatePolicy, order=None) -> list:
    """Admit at most `cap` candidates under `policy`.

    `order` is the frozen within-bucket ordering key (the stable hash, or the full order key which begins with it).
    CM-HASH is exactly the frozen behaviour: sort by that key and truncate. The diversity policies partition into
    canonical buckets, keep the same ordering inside each bucket, and take one candidate per non-empty bucket per
    round in canonical bucket-key order until the cap is filled or the candidates are exhausted. The admitted set is
    returned in the frozen global order so that every consumer downstream sees the same ordering discipline it saw
    before."""
    order = order or (lambda c: c.order_key)
    if policy.name == HASH:
        return sorted(cands, key=order)[:cap]
    buckets = defaultdict(list)
    for c in cands:
        buckets[bucket_key(c, policy)].append(c)
    for k in buckets:
        buckets[k].sort(key=order)
    keys = sorted(buckets)
    taken, i = [], 0
    while len(taken) < cap:
        progressed = False
        for k in keys:
            if i < len(buckets[k]):
                taken.append(buckets[k][i])
                progressed = True
                if len(taken) >= cap:
                    break
        if not progressed:
            break
        i += 1
    return sorted(taken, key=order)


def bucket_rank(cand, cands: list, policy: CandidatePolicy, order=None) -> int | None:
    """Rank of `cand` inside its own bucket under the frozen within-bucket ordering. Diagnostic only."""
    order = order or (lambda c: c.order_key)
    if policy.name == HASH:
        return sorted(cands, key=order).index(cand) if cand in cands else None
    k = bucket_key(cand, policy)
    same = sorted([c for c in cands if bucket_key(c, policy) == k], key=order)
    return same.index(cand) if cand in same else None
