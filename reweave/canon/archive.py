"""Certified library archive: semantic families, recognition surfaces, structural index, retrieval, aliases.

Three separated layers: a FAMILY is one exact parameterized relation; a SURFACE is one matchable source pattern of a
family (a family may have several); handles are interned separately. The archive is monotone and admits certified
relations only (`CanonicalArchive.add_certified` re-certifies on insertion).

  LEGACY_SURFACE_ID            = schema.schema_hash
  CANONICAL_SEMANTIC_FAMILY_ID = canonical key of the formal parameterized exact relation (`canonical_family_id`):
                                 the generic instance (parameter i -> fresh variable x_i), minimized over parameter
                                 permutations; binding modes never create separate families

Retrieval uses per-context structural necessary conditions (root kind, per-kind multiplicity, exponent divisibility,
anchor signatures) to choose which surfaces the exact matcher sees: sound by construction, no per-family embedding.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field

from .. import rewrite_match as RM
from .. import rewrite_schema as S
from .. import v1
from ..hashing import canonical_json, sha256_json
from .labelling import IDENT, rdigest

K_SCHEMA = 16
STATUSES = ('ACTIVE', 'DORMANT', 'ARCHIVED')
_RANK = {s: i for i, s in enumerate(STATUSES)}
_KIND = {'M': 0, 'S': 1, 'P': 2}


# ---------------------------------------------------------------- schema-agnostic structural features


def _nodes(p) -> int:
    if p[0] == 'M':
        return 1
    return 1+sum(n*_nodes(c) for c, n in p[1])


def _depth(p) -> int:
    if p[0] == 'M':
        return 0
    return 1+max(_depth(c) for c, n in p[1])


def _sig(p) -> str:
    """Local structural signature with hole identities abstracted (exponent profile only)."""
    if p[0] == 'M':
        return 'M'+repr(tuple(sorted(e for i, e in p[1])))
    return p[0]+'('+','.join(sorted(f'{_sig(c)}#{n}' for c, n in p[1]))+')'


def _triple(entries) -> tuple:
    t = [0, 0, 0]
    for c, n in entries:
        t[_KIND[c[0]]] += n
    return tuple(t)


def region_features(r) -> dict:
    occ: Counter = Counter()
    for c, n in r:
        for h in S.holes(c):
            occ[h] += n
    return dict(entries=len(r), multiplicity=sum(n for c, n in r), kinds=list(_triple(r)),
                nodes=sum(n*_nodes(c) for c, n in r), depth=max((_depth(c) for c, n in r), default=0),
                max_arity=max([len(r)]+[len(c[1]) for c, n in r if c[0] != 'M']),
                max_exponent=max((S.max_exp(c) for c, n in r), default=0),
                hole_occurrences=sorted(occ.values()),
                signature=sorted(f'{_sig(c)}#{n}' for c, n in r))


def features(s: S.Schema) -> dict:
    """Structural features only: operator structure, hole types and multiplicities, arity, sizes, local signatures."""
    return dict(n_holes=len(s.types), hole_types=sorted(Counter(s.types).items()),
                source=region_features(s.source), target=region_features(s.target),
                binding_plan=S.binding_plan(s), automorphisms=len(s.automorphisms))


def specificity(s: S.Schema) -> int:
    return sum(n*_nodes(c) for c, n in s.source)


# ---------------------------------------------------------------- necessary-condition profiles


def _pattern_profile(p) -> tuple:
    if p[0] == 'M':
        es = tuple(e for i, e in p[1])
        if not es:
            return ('ONE',)
        return ('M', sum(es), es[0] if len(es) == 1 else 1)
    return (p[0], _triple(p[1]))


@dataclass
class _Compiled:
    sid: str
    schema: S.Schema
    need: tuple                      # source per-kind multiplicity
    entries: tuple                   # ((profile, count), ...)
    spec: int


def compile_schema(s: S.Schema) -> _Compiled:
    return _Compiled(s.schema_hash, s, _triple(s.source), tuple((_pattern_profile(c), n) for c, n in s.source),
                     specificity(s))


def _kind_of(node) -> str:
    return 'M' if isinstance(node, v1.Mono) else 'S' if isinstance(node, v1.Sum) else 'P'


def _node_triple(arena, node) -> tuple:
    t = [0, 0, 0]
    for c, m in v1.items(node):
        t[_KIND[_kind_of(arena.handles[c])]] += m
    return tuple(t)


class _ContextView:
    """Per-context child summary for the necessary conditions (memoized per entry profile)."""
    __slots__ = ('have', 'monos', 'compound', 'memo')

    def __init__(self, arena, ctx: int) -> None:
        t = [0, 0, 0]
        self.monos: list = []
        self.compound: dict = {}
        for h, n in v1.items(arena.handles[ctx]):
            node = arena.handles[h]
            kind = _kind_of(node)
            t[_KIND[kind]] += n
            if kind == 'M':
                self.monos.append((v1.unpack(node.m), n))
            else:
                key = (kind, _node_triple(arena, node))
                self.compound[key] = max(self.compound.get(key, 0), n)
        self.have = tuple(t)
        self.memo: dict = {}

    def entry_ok(self, prof: tuple, n: int) -> bool:
        k = (prof, n)
        r = self.memo.get(k)
        if r is None:
            if prof[0] == 'ONE':
                r = any(c >= n and not any(v) for v, c in self.monos)
            elif prof[0] == 'M':
                E, e = prof[1], prof[2]
                r = any(c >= n and any(v) and sum(v) >= E and (e == 1 or all(x % e == 0 for x in v))
                        for v, c in self.monos)
            else:
                r = self.compound.get(prof, 0) >= n
            self.memo[k] = r
        return r


def passes(comp: _Compiled, view: _ContextView) -> bool:
    return (all(a <= b for a, b in zip(comp.need, view.have)) and
            all(view.entry_ok(p, n) for p, n in comp.entries))



# ---------------------------------------------------------------- generic-instance subsumption (c7_library @ 5f6af3d)


def generic_instance(s: S.Schema, arena):
    """Bind hole i to the fresh variable x_i; lower the source region. None if the region is not a Sum region."""
    beta = {i: tuple(1 if j == i else 0 for j in range(S.NV)) for i in range(len(s.types))}
    R = S.lower_region(s.source, beta, arena)
    if sum(R.values()) < 2:
        return None
    return beta, R, S.lower_region(s.target, beta, arena)


def subsumed_by(s: S.Schema, others: list, cfg) -> str | None:
    """Relation id of the first (sorted) archived relation that rewrites s's generic instance identically, else None."""
    if len(s.types) > S.NV:
        return None
    arena = v1.Arena()
    arena.calc_cfg = v1.CalcCfg()
    try:
        g = generic_instance(s, arena)
    except S.CapError:
        return None
    if g is None:
        return None
    beta, R, T = g
    ctx = v1.mk_sum(arena, Counter(R))
    for o in others:
        if o.schema_hash == s.schema_hash:
            continue
        cands, _w = RM.match_state(arena, ctx, [o], cfg)
        for c in cands:
            if c.ctx != ctx or c.R_inst != Counter(R):
                continue
            try:
                T2 = S.lower_region(o.target, c.beta, arena)
            except S.CapError:
                continue
            if T2 == Counter(T):
                return o.schema_hash
    return None



# ---------------------------------------------------------------- the C7.3 archive (c7_3_memory @ 7ab4341)

DEG_MAX = 6


def required_keys(comp: _Compiled) -> tuple:
    """The index keys a context must supply for this surface to have any chance of matching."""
    ks = []
    for p, n in comp.entries:
        ks.append(('ONE',) if p[0] == 'ONE' else ('M', p[1], p[2]) if p[0] == 'M' else (p[0], p[1]))
    return tuple(sorted(set(ks)))


def available_keys(view: _ContextView) -> set:
    """Every index key this context can supply. Mono requirements are ranges, so they expand; the range is
    bounded by the frozen degree cap, so this is O(children), not O(archive)."""
    A = set()
    for vec, _c in view.monos:
        if not any(vec):
            A.add(('ONE',))
            continue
        d = sum(vec)
        for e in range(1, DEG_MAX+1):
            if e > 1 and any(x % e for x in vec):
                continue
            for E in range(1, min(d, DEG_MAX)+1):
                A.add(('M', E, e))
    A |= set(view.compound)
    return A


# ---------------------------------------------------------------- archive


@dataclass
class Surface:
    """One concrete matchable source pattern of a relation family."""
    sid: str
    schema: object
    comp: object
    family: str
    origin: dict
    key: tuple = ()                                   # the rarest required key: this surface's posting slot
    stats: Counter = field(default_factory=Counter)   # §17 bookkeeping, never a retrieval input


@dataclass
class Family:
    """One exact semantic transformation. Several surfaces may recognize it; it is still one capability."""
    rid: str
    surfaces: list
    status: str = 'ACTIVE'
    feat: dict = field(default_factory=dict)
    utility: dict = field(default_factory=dict)
    last_useful_round: int | None = None
    keys: tuple = ()                                  # union of its surfaces' posting keys


class Archive:
    def __init__(self) -> None:
        self.families: dict = {}
        self.surfaces: dict = {}
        self.fam_post: dict = {}      # key -> [rid]      (stage 1)
        self.sur_post: dict = {}      # key -> [sid]      (stage 2)
        self._freq: Counter = Counter()
        self.labels: Counter = Counter()

        self._simul: dict = {}

    def __len__(self) -> int:
        return len(self.surfaces)

    def simultaneous(self, sch) -> bool:
        """`S.uses_simultaneous` per schema, cached: the frozen matcher computes it once per call, and a
        per-context loop would otherwise recompute it for every context."""
        r = self._simul.get(sch.schema_hash)
        if r is None:
            r = self._simul[sch.schema_hash] = S.uses_simultaneous(sch)
        return r

    # ---- insertion and exact semantic dedup (§4)
    def classify(self, s, cfg) -> tuple:
        rid = s.schema_hash
        if rid in self.families:
            return ('DUPLICATE' if rid in self.surfaces else 'NEW_RECOGNITION_SURFACE'), rid
        sub = subsumed_by(s, [self.surfaces[k].schema for k in sorted(self.surfaces)], cfg)
        if sub is not None:
            return 'SPECIALIZATION_SURFACE', self.surfaces[sub].family if sub in self.surfaces else sub
        return 'NEW_SEMANTIC_RELATION', rid

    def add(self, s, origin: dict, cfg, status: str = 'ACTIVE') -> tuple:
        """Insert a certified schema. Surfaces stay matchable whatever their label; only a new relation is a new
        capability. Nothing is deleted and nothing is collapsed on syntactic resemblance alone."""
        label, rid = self.classify(s, cfg)
        self.labels[label] += 1
        if label == 'DUPLICATE':
            return label, rid
        if label == 'NEW_SEMANTIC_RELATION':
            self.families[rid] = Family(rid, [], status, features(s))
        sur = Surface(s.schema_hash, s, compile_schema(s), rid, dict(origin, label=label))
        self.surfaces[sur.sid] = sur
        self.families[rid].surfaces.append(sur.sid)
        return label, rid

    # ---- index construction: post each surface under its rarest requirement
    def reindex(self) -> dict:
        self._freq.clear()
        need = {}
        for sid, sur in self.surfaces.items():
            need[sid] = required_keys(sur.comp)
            self._freq.update(need[sid])
        self.fam_post, self.sur_post = {}, {}
        unkeyed = 0
        for sid in sorted(self.surfaces):
            sur, ks = self.surfaces[sid], need[sid]
            if not ks:                                  # requires nothing: must always be considered
                sur.key = ()
                unkeyed += 1
            else:
                sur.key = min(ks, key=lambda k: (self._freq[k], k))
            self.sur_post.setdefault(sur.key, []).append(sid)
        for rid in sorted(self.families):
            f = self.families[rid]
            f.keys = tuple(sorted({self.surfaces[s].key for s in f.surfaces}))
            for k in f.keys:
                self.fam_post.setdefault(k, []).append(rid)
        return dict(surfaces=len(self.surfaces), families=len(self.families), keys=len(self.sur_post),
                    unkeyed=unkeyed, max_posting=max((len(v) for v in self.sur_post.values()), default=0))

    # ---- per-context retrieval: the unit the matcher actually works in
    def retrieve_at(self, arena, ctx: int, cfg, k_schema=None, st: Counter | None = None, cache: dict | None = None):
        """The surfaces that could match at *this* context.

        A per-state union is the wrong unit: the frozen matcher loops contexts x schemas, so a surface retrieved
        because it fits one context is still tried against every other. Retrieving per context is what stops
        archive growth from reaching the matcher.
        """
        st = Counter() if st is None else st
        if cache is not None and ctx in cache:
            st['context_cache_hits'] += 1
            return cache[ctx]
        view = _ContextView(arena, ctx)
        A = available_keys(view)
        st['contexts'] += 1
        cand_f = {rid for k in A if k in self.fam_post for rid in self.fam_post[k]}
        cand_f |= set(self.fam_post.get((), ()))
        st['family_probes'] += len(cand_f)
        st['family_untouched'] += len(self.families)-len(cand_f)
        chosen, probes = [], 0
        for rid in sorted(cand_f):
            hit = False
            for sid in self.families[rid].surfaces:
                sur = self.surfaces[sid]
                if sur.key and sur.key not in A:
                    continue
                probes += 1
                if passes(sur.comp, view):
                    chosen.append(sur)
                    hit = True
            st['families_hit'] += hit
        st['surface_probes'] += probes
        st['surfaces_retrieved'] += len(chosen)
        st['surfaces_untouched'] += len(self.surfaces)-probes
        chosen.sort(key=lambda s: (_RANK[self.families[s.family].status], -specificity(s.schema), s.sid))
        if k_schema is not None and len(chosen) > k_schema:
            st['truncated_contexts'] += 1
            st['surfaces_dropped_by_budget'] += len(chosen)-k_schema
            chosen = chosen[:k_schema]
        out = [s.schema for s in chosen]
        if cache is not None:
            cache[ctx] = out
        return out

    # ---- §17 bookkeeping: enough to make future compaction possible, no deletion now
    def note(self, kind: str, sids) -> None:
        for sid in sids:
            if sid in self.surfaces:
                self.surfaces[sid].stats[kind] += 1

    def usage(self) -> dict:
        return {sid: dict(s.stats) for sid, s in sorted(self.surfaces.items()) if s.stats}

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(sorted(
            (s.family, sid) for sid, s in self.surfaces.items())).encode()).hexdigest()

    def census(self) -> dict:
        per = sorted(len(f.surfaces) for f in self.families.values())
        return dict(surfaces=len(self.surfaces), families=len(self.families),
                    surfaces_per_family_max=max(per, default=0),
                    multi_surface_families=sum(1 for n in per if n > 1),
                    labels=dict(self.labels), digest=self.digest(),
                    bytes=len(json.dumps([s.schema.to_json() for s in self.surfaces.values()])))



# ---------------------------------------------------------------- the frozen matcher, behind retrieval (c7_3_memory @ 7ab4341)


def match_state_retrieved(arena, root: int, archive: Archive, cfg, k_schema=None, st: Counter | None = None,
                          cache: dict | None = None):
    """The frozen `rewrite_match.match_state`, with the schema list chosen per context.

    The body below is the frozen one, transcribed: the same digest-ordered context list and context cap, the same
    `assign(..., exact=False)` enumeration, the same `(site_key, schema_hash, tsig)` dedup keeping the smallest
    binding digest, the same sha256 order key, and the same shared attempt/unify budget across contexts. Only the
    inner `for sch in schemas` reads a per-context list instead of the whole archive, so a surface is offered to
    the matcher exactly where it could match. Nothing downstream can tell the difference — which the soundness
    harness checks against the unmodified matcher, candidate for candidate.
    """
    from .. import rewrite_match as RM
    st = Counter() if st is None else st
    info = v1.state_info(arena, root)
    contexts = sorted(info.contexts, key=lambda h: v1.digest(arena, h))
    work = dict(contexts=0, attempts=0, cap=cfg.match_attempts_max, unify_states=0, cut_contexts=0,
                cut_unify_states=0, cut_attempts=0, cut_output=0, generated=0)
    if len(contexts) > cfg.match_contexts_max:
        work['cut_contexts'] = len(contexts)-cfg.match_contexts_max
        contexts = contexts[:cfg.match_contexts_max]
    found: dict = {}
    stop = False
    for ctx in contexts:
        if stop:
            break
        schemas = archive.retrieve_at(arena, ctx, cfg, k_schema, st, cache)
        if not schemas:
            st['contexts_no_schema'] += 1
            continue
        work['contexts'] += 1
        st['pairs_offered'] += len(schemas)          # (context, surface) pairs that reach the exact matcher
        counts = dict(v1.items(arena.handles[ctx]))
        index = S.ChildIndex(arena, counts)
        ctx_digest = v1.digest(arena, ctx).hex()
        for sch in schemas:
            budget = RM.WorkBudget(cfg.unify_states_max, work)
            trace: list = []
            a0, got0 = work['attempts'], work['generated']
            try:
                for beta in S.assign(sch.source, index, dict(counts), {}, arena, cfg, budget, exact=False,
                                     trace=trace, simultaneous=archive.simultaneous(sch)):
                    R_inst: Counter = Counter()
                    for (p, n), h in trace:
                        R_inst[h] += n
                    work['generated'] += 1
                    region_key = tuple(sorted((v1.digest(arena, h).hex(), c) for h, c in R_inst.items()))
                    site_key = ('rw', ctx_digest, region_key)
                    tsig = S.target_signature(sch, beta)
                    bdig = S.binding_digest(beta)
                    dkey = (site_key, sch.schema_hash, tsig)
                    order = (hashlib.sha256(RM.canonical_json([site_key, sch.schema_hash, tsig, bdig]).encode()).hexdigest(),
                             site_key, sch.schema_hash, tsig, bdig)
                    prev = found.get(dkey)
                    if prev is None or bdig < prev.binding_digest:
                        found[dkey] = RM.Candidate(ctx, sch, dict(beta), R_inst, site_key, tsig, bdig, order)
            except S.Cut as cut:
                work['unify_states'] += budget.states
                if cut.args and cut.args[0] == 'attempts':
                    work['cut_attempts'] += 1
                    stop = True
                    break
                work['cut_unify_states'] += 1
                st['attempts_barren' if work['generated'] == got0 else 'attempts_productive'] += work['attempts']-a0
                continue
            work['unify_states'] += budget.states
            # the ceiling on any stronger filter: what a perfect one could have saved is exactly the attempts
            # spent on pairs that produced nothing
            st['attempts_barren' if work['generated'] == got0 else 'attempts_productive'] += work['attempts']-a0
    cands = sorted(found.values(), key=lambda c: c.order_key)
    work['deduped'] = len(cands)
    if len(cands) > cfg.rewrite_n_max:
        work['cut_output'] = len(cands)-cfg.rewrite_n_max
        cands = cands[:cfg.rewrite_n_max]
    work.pop('cap')
    # precision of the filter: of the pairs that reached the matcher, how many actually yielded a candidate. A low
    # number means the necessary condition is too weak; a high one means the remaining work is inherent.
    st['pairs_productive'] += len({(c.ctx, c.schema.schema_hash) for c in cands})
    return cands, work



# ---------------------------------------------------------------- integration identity layer (new)

FAMILY_ID_VERSION = 'canonical-semantic-family-v1'


def legacy_surface_id(s: S.Schema) -> str:
    """LEGACY_SURFACE_ID: the historical schema_hash, unchanged."""
    return s.schema_hash


def relation_form(s: S.Schema) -> tuple:
    """The formal parameterized exact relation: parameter i -> fresh variable x_i, source and target lowered, then the
    canonical form under every permutation of the k parameters (structural digests, never arena ids). Hole TYPES are
    not part of it, so a relation's identity is independent of how its parameters bind at runtime."""
    k = len(s.types)
    a = v1.Arena()
    a.calc_cfg = v1.CalcCfg()
    beta = {i: tuple(1 if j == i else 0 for j in range(S.NV)) for i in range(k)}
    R, T = S.lower_region(s.source, beta, a), S.lower_region(s.target, beta, a)
    best = None
    for img in itertools.permutations(range(k)):
        p = list(IDENT)
        for i, j in enumerate(img):
            p[i] = j
        p = tuple(p)
        memo: dict = {}
        form = (tuple(sorted((rdigest(a, h, p, memo).hex(), n) for h, n in R.items())),
                tuple(sorted((rdigest(a, h, p, memo).hex(), n) for h, n in T.items())))
        if best is None or form < best:
            best = form
    return k, best


def canonical_family_id(s: S.Schema) -> str:
    """CANONICAL_SEMANTIC_FAMILY_ID of the relation this schema expresses."""
    return sha256_json([FAMILY_ID_VERSION, *relation_form(s)])


class CanonicalArchive(Archive):
    """The C7.3 archive plus the integration identity layer. Monotone: nothing is ever removed.

    Every inserted surface must pass the mainline exact certification (formal identity with parameter i -> x_i,
    `rewrite_certify.certify`). A family's CANONICAL_SEMANTIC_FAMILY_ID is the formal relation id of the surface that
    founded it; every surface also records its own relation id. The legacy alias map is
    LEGACY_SURFACE_ID -> (legacy family rid, CANONICAL_SEMANTIC_FAMILY_ID)."""

    def __init__(self) -> None:
        super().__init__()
        self.canonical: dict = {}          # family rid -> CANONICAL_SEMANTIC_FAMILY_ID
        self.meta: dict = {}               # surface sid -> dict(label, certification, relation_id, origin, round)
        self.order: list = []              # insertion order of surfaces (defines the family structure)

    def add_certified(self, s: S.Schema, origin: dict, cfg, round_: int = 0) -> tuple:
        from ..rewrite_certify import certify
        ok, why, _detail = certify(s)
        if not ok:
            raise ValueError(f'{s.schema_hash[:12]}: not certified ({why}); the archive admits certified relations only')
        n_before = len(self.surfaces)
        label, rid = self.add(s, dict(origin, round=round_), cfg)
        if label == 'DUPLICATE':
            return label, rid
        assert len(self.surfaces) == n_before+1, 'monotone growth'
        rel = canonical_family_id(s)
        if label == 'NEW_SEMANTIC_RELATION':
            self.canonical[rid] = rel
        self.meta[s.schema_hash] = dict(label=label, certification=why, relation_id=rel, origin=origin, round=round_)
        self.order.append(s.schema_hash)
        return label, rid

    def alias_map(self) -> dict:
        return {sid: dict(legacy_surface_id=sid, legacy_family_rid=self.surfaces[sid].family,
                          canonical_semantic_family_id=self.canonical[self.surfaces[sid].family],
                          surface_relation_id=self.meta[sid]['relation_id'], label=self.meta[sid]['label'])
                for sid in self.order}

    def to_json(self) -> dict:
        return dict(v=FAMILY_ID_VERSION, surfaces=[dict(schema=self.surfaces[sid].schema.to_json(),
                                                        family=self.surfaces[sid].family,
                                                        status=self.families[self.surfaces[sid].family].status,
                                                        **self.meta[sid]) for sid in self.order],
                    alias_map=self.alias_map(), digest=self.digest())

    @staticmethod
    def from_json(j: dict, cfg) -> 'CanonicalArchive':
        """Rebuild by re-inserting in the recorded order; asserts every label, family and id reproduces exactly."""
        a = CanonicalArchive()
        for e in j['surfaces']:
            s = S.Schema.from_json(e['schema'])
            label, rid = a.add_certified(s, e['origin'], cfg, e['round'])
            assert (label, rid, a.meta[s.schema_hash]['relation_id']) == (e['label'], e['family'], e['relation_id']), \
                f'{s.schema_hash[:12]}: archive round trip does not reproduce'
        a.reindex()
        assert a.digest() == j['digest'] and a.alias_map() == j['alias_map']
        return a

    def retrieval_metadata(self) -> dict:
        return dict(reindex=self.reindex(), postings={repr(k): list(v) for k, v in sorted(self.sur_post.items(), key=lambda kv: repr(kv[0]))})


def build_archive(schemas: list, cfg, origin: dict | None = None) -> CanonicalArchive:
    """A certified archive over `schemas`, inserted in LEGACY_SURFACE_ID order (a deterministic family structure)."""
    a = CanonicalArchive()
    for s in sorted(schemas, key=lambda s: s.schema_hash):
        a.add_certified(s, dict(origin or dict(source='L0')), cfg)
    a.reindex()
    return a
