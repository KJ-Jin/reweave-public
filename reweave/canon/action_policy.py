"""Policy configurations and the search-time schema-agnostic scorer.

    LEGACY_FIXED_VOCAB_POLICY   the fixed-vocabulary `rewrite_model.RewriteScorer`, reading ActionRecords through the
                                parity-exact adapter `action_record.score_records`
    SCHEMA_AGNOSTIC_POLICY      `SchemaAgnosticScorer`: ActionRecord -> score, no relation-id table anywhere

The schema-agnostic scorer exists only behind the ActionRecord boundary (generic ExprParam profiles). Its value ranker
is the inherited value head.
"""

from __future__ import annotations

import time
from collections import Counter

from ..hashing import sha256_json
from . import action_features as TF

POLICIES = ('LEGACY_FIXED_VOCAB_POLICY', 'SCHEMA_AGNOSTIC_POLICY')


class SchemaAgnosticScorer:
    """SCHEMA_AGNOSTIC_POLICY. One encoder pass per state: primitive logits from the inherited V1 head, rewrite
    logits from the T7 head over ActionRecord features. CPU is accounted separately for feature building and model
    inference (`self.cpu`)."""

    policy = 'SCHEMA_AGNOSTIC_POLICY'

    def __init__(self, model) -> None:
        self.model = model
        self.cpu: Counter = Counter()

    def score(self, arena, info, root, actions):
        """Primitive-only scoring (a search without a rewrite library)."""
        import torch
        from .. import v1
        if not actions:
            return []
        with torch.no_grad():
            logits, values = v1.forward(self.model, v1.make_batch([(arena, info, root, actions)]))
        return [float(x) for x in logits]

    def score_combined(self, *args, **kwargs):
        raise RuntimeError('SCHEMA_AGNOSTIC_POLICY scores ActionRecords only (a generic profile is required)')

    def score_action_records(self, arena, info, root, recs: list, views=None, vctx=None) -> list:
        import torch
        from .action_model import t7_forward
        t0 = time.process_time()
        if any(r.kind == 'VIEW_REWRITE' for r in recs):
            raise RuntimeError('VIEW_REWRITE is interface-only in T7 (views are behind the boundary at T6B)')
        prim = [i for i, r in enumerate(recs) if r.kind == 'PRIMITIVE']
        rw = [i for i, r in enumerate(recs) if r.kind != 'PRIMITIVE']
        sv = TF.StateView(arena, root) if rw else None
        feats = [TF.rewrite_features(recs[i], sv) for i in rw]
        t1 = time.process_time()
        with torch.no_grad():
            logits, values, rwl, _ = t7_forward(self.model, [(arena, info, root, [recs[i].live for i in prim])], [feats])
        t2 = time.process_time()
        out = [0.0]*len(recs)
        for i, x in zip(prim, logits):
            out[i] = float(x)
        for i, x in zip(rw, rwl[0]):
            out[i] = float(x)
        self.cpu['feature_seconds'] += t1-t0
        self.cpu['inference_seconds'] += t2-t1
        self.cpu['states'] += 1
        self.cpu['rewrite_records'] += len(rw)
        self.cpu['primitive_records'] += len(prim)
        return out


# ---------------------------------------------------------------- T7.4: the canonical scoring frame (identity mode)


def canonical_copy(arena, root: int) -> tuple:
    """(fresh arena, root, source handle -> copy handle) with the state interned in V1's own canonical serialization
    order (height, structural digest): every handle id — hence the inherited encoder's node order — is a function of
    the canonical DAG alone, never of the arena the state came from."""
    from .. import v1
    nodes, index = v1.serialize(arena, [root], with_digests=False)
    dst = v1.Arena()
    dst.calc_cfg = v1.CalcCfg()
    ids = v1.deserialize(dst, nodes)
    return dst, ids[index[root]], {h: ids[i] for h, i in index.items()}


def _map_action(arena, dst, fwd: dict, a):
    """A V1 primitive action in the canonical copy (off-state monomial arguments interned after the state)."""
    from .. import v1
    pos = {h: i for i, (h, c) in enumerate(v1.items(dst.handles[fwd[a.ctx]]))}

    def region(pairs):
        return tuple(sorted(((fwd[h], c) for h, c in pairs), key=lambda hc: pos[hc[0]]))
    if a.arg < 0:
        arg = a.arg
    elif a.arg in fwd:
        arg = fwd[a.arg]
    else:
        arg = v1.mk_mono(dst, arena.handles[a.arg].m)
    return v1.Action(a.kind, fwd[a.ctx], region(a.region), arg, None if a.src is None else region(a.src))


class CanonicalFrameScorer(SchemaAgnosticScorer):
    """SCHEMA_AGNOSTIC_POLICY in IDENTITY_MODE: every scored state is first copied into the canonical scoring frame,
    so renamed equivalent tasks give identical encoder inputs, identical logits (bit for bit), identical rankings and
    identical chosen actions. Records are digest-addressed, so they resolve in the copy unchanged."""

    policy = 'SCHEMA_AGNOSTIC_POLICY'

    def score_action_records(self, arena, info, root, recs: list, views=None, vctx=None) -> list:
        import torch
        from .. import v1
        from .action_model import t7_forward
        t0 = time.process_time()
        if any(r.kind == 'VIEW_REWRITE' for r in recs):
            raise RuntimeError('VIEW_REWRITE is interface-only in T7 (views are behind the boundary at T6B)')
        dst, droot, fwd = canonical_copy(arena, root)
        prim = [i for i, r in enumerate(recs) if r.kind == 'PRIMITIVE']
        rw = [i for i, r in enumerate(recs) if r.kind != 'PRIMITIVE']
        acts = [_map_action(arena, dst, fwd, recs[i].live) for i in prim]
        sv = TF.StateView(dst, droot) if rw else None
        feats = [TF.rewrite_features(recs[i], sv) for i in rw]
        dinfo = v1.state_info(dst, droot)
        t1 = time.process_time()
        with torch.no_grad():
            logits, values, rwl, _ = t7_forward(self.model, [(dst, dinfo, droot, acts)], [feats])
        t2 = time.process_time()
        out = [0.0]*len(recs)
        for i, x in zip(prim, logits):
            out[i] = float(x)
        for i, x in zip(rw, rwl[0]):
            out[i] = float(x)
        self.cpu['feature_seconds'] += t1-t0
        self.cpu['inference_seconds'] += t2-t1
        self.cpu['states'] += 1
        self.cpu['rewrite_records'] += len(rw)
        self.cpu['primitive_records'] += len(prim)
        self.last = dict(root_digest=v1.digest(arena, root).hex(), scores=out)
        return out


class CanonicalFrameRanker:
    """The inherited value head on the canonical scoring copy of the state (bit-identical under renaming)."""

    def __init__(self, model) -> None:
        self.model = model

    def rank(self, arena, info, root) -> float:
        import torch
        from .. import v1
        dst, droot, fwd = canonical_copy(arena, root)
        with torch.no_grad():
            logits, values = v1.forward(self.model, v1.make_batch([(dst, v1.state_info(dst, droot), droot, [])]))
        return float(values[0])


def policy_identity(name: str, model=None, checkpoint: str | None = None) -> dict:
    """The explicit identity of a scoring policy (recorded beside the generation configuration identity)."""
    from ..cache import state_dict_hash
    assert name in POLICIES, name
    body = dict(policy=name, checkpoint=checkpoint,
                model_state_hash=state_dict_hash(model.state_dict()) if model is not None else None)
    if name == 'SCHEMA_AGNOSTIC_POLICY':
        from .action_model import T7_MODEL_VERSION
        body.update(model_version=T7_MODEL_VERSION, feature_version=TF.T7_FEATURE_VERSION)
    return dict(body, policy_hash=sha256_json(body))
