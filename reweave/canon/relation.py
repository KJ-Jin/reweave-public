"""The associative-commutative semantic relation IR and its versioned family identity.

`canon.archive.relation_form` lowers a relation's source and target patterns as written. Because the circuit
representation preserves every non-monomial Sum / Prod boundary, two relations that differ only by association —
P(P(A,X),Y) vs P(A,X,Y) — would receive different structural ids although they are the same exact relation.
`relation_form_ac` first normalizes both sides modulo associativity and commutativity of ADD and MUL (no
distributivity), then lowers and minimizes over parameter permutations. Digests index; exact payloads decide equality.
"""

from __future__ import annotations

import itertools
from collections import Counter

from .. import rewrite_schema as S
from .. import v1
from ..hashing import canonical_json, sha256_json
from .labelling import IDENT, rdigest

FAMILY_ID_AC_VERSION = 'canonical-semantic-family-v2-ac'


def ac_flatten(p) -> tuple:
    """A pattern modulo associativity / commutativity of its operators (normal form of `rewrite_schema.norm`)."""
    if p[0] == 'M':
        return p
    kids: Counter = Counter()
    for c, n in p[1]:
        c = ac_flatten(c)
        if c[0] == p[0]:
            for cc, nn in c[1]:
                kids[cc] += nn*n
        else:
            kids[c] += n
    return S.norm((p[0], tuple(sorted(kids.items(), key=lambda cn: (S.key(cn[0]), cn[1])))))


def ac_region(r) -> tuple:
    """A region is an additive multiset: nested sums are spliced into it."""
    kids: Counter = Counter()
    for c, n in r:
        c = ac_flatten(c)
        if c[0] == 'S':
            for cc, nn in c[1]:
                kids[cc] += nn*n
        else:
            kids[c] += n
    return S.norm_region(tuple(kids.items()))


def surface_memo(cache: dict, s, fn):
    """Per-surface memo of a pure function of a schema. The LEGACY_SURFACE_ID indexes the entry; the stored canonical
    body decides a hit, so a digest collision recomputes and never reuses another surface's value."""
    hit = cache.get(s.schema_hash)
    if hit is not None and hit[0] == s.body:
        return hit[1]
    r = fn(s)
    if hit is None:
        cache[s.schema_hash] = (s.body, r)
    return r


_FORMS: dict = {}


def _ac_forms(s) -> tuple:
    k = len(s.types)
    a = v1.Arena()
    a.calc_cfg = v1.CalcCfg()
    beta = {i: tuple(1 if j == i else 0 for j in range(S.NV)) for i in range(k)}
    R, T = S.lower_region(ac_region(s.source), beta, a), S.lower_region(ac_region(s.target), beta, a)
    forms = []
    for img in itertools.permutations(range(k)):
        p = list(IDENT)
        for i, j in enumerate(img):
            p[i] = j
        p = tuple(p)
        memo: dict = {}
        forms.append(((tuple(sorted((rdigest(a, h, p, memo).hex(), n) for h, n in R.items())),
                       tuple(sorted((rdigest(a, h, p, memo).hex(), n) for h, n in T.items()))), img))
    best = min(f for f, img in forms)
    return k, best, [img for f, img in forms if f == best]


def ac_forms(s) -> tuple:
    """(k, canonical AC form, [parameter permutations realizing it]) of a schema, cached by surface."""
    return surface_memo(_FORMS, s, _ac_forms)


def canonical_family_id_ac(s) -> str:
    """CANONICAL_SEMANTIC_FAMILY_ID v2: the exact relation modulo AC of ADD / MUL and parameter renaming."""
    k, best, frames = ac_forms(s)
    return sha256_json([FAMILY_ID_AC_VERSION, k, best])


# ---------------------------------------------------------------- exact payloads behind the digests (T6A.2)

RELATION_PAYLOAD_VERSION = 'ac-relation-v1'
_PAYLOADS: dict = {}


def family_payload(s) -> str:
    """The EXACT canonical relation of a schema's v2 family: its AC-normalized source and target regions relabelled
    into the family's canonical parameter frame, serialized. Normalized patterns and their fresh-variable lowerings
    determine each other, so two surfaces have equal payloads iff they have equal v2 forms; the v2 id indexes the
    payload, the payload decides equality (a digest collision can never merge two relations). Among the canonical
    frames (the relation's symmetries) the smallest serialization is taken."""
    return surface_memo(_PAYLOADS, s, _family_payload)


def _family_payload(s) -> str:
    k, best, frames = ac_forms(s)
    src, tgt = ac_region(s.source), ac_region(s.target)
    outs = []
    for img in frames:
        perm = {i: img[i] for i in range(k)}
        outs.append(canonical_json([RELATION_PAYLOAD_VERSION, k,
                                    S.ser_region(S.norm_region(S.relabel_region(src, perm))),
                                    S.ser_region(S.norm_region(S.relabel_region(tgt, perm)))]))
    return min(outs)


def same_family(s1, s2) -> bool:
    """Exact v2 family equality (payload comparison; the ids only index)."""
    return family_payload(s1) == family_payload(s2)


def ac_alias_map(schemas: list) -> dict:
    """The reversible v2 alias map of a surface set: every LEGACY_SURFACE_ID -> (v1 family id, v2 family id, exact
    relation payload digest), and every v2 family -> its surfaces. Families are formed by EXACT payload equality."""
    from .archive import canonical_family_id
    by_payload: dict = {}
    for s in sorted(schemas, key=lambda s: s.schema_hash):
        by_payload.setdefault(family_payload(s), []).append(s)
    surfaces, fams = {}, {}
    for p, members in sorted(by_payload.items(), key=lambda x: (canonical_family_id_ac(x[1][0]), x[0])):
        fid = canonical_family_id_ac(members[0])
        key = fid if fid not in fams else f'{fid}#{len(fams)}'        # a v2 digest collision stays two families
        fams[key] = dict(family_id=fid, relation_payload_sha256=sha256_json(p),
                         surfaces=[s.schema_hash for s in members])
        for s in members:
            surfaces[s.schema_hash] = dict(legacy_surface_id=s.schema_hash, family_id_v1=canonical_family_id(s),
                                           family_id=fid, family_key=key)
    return dict(v=RELATION_PAYLOAD_VERSION, surfaces=surfaces, families=fams)


def lookup_surface_exact(archive, s):
    """Exact archive identity lookup: the stored surface under `s`'s legacy id, only if its canonical body is equal
    (the legacy id is a digest of that body: an index, not the equality)."""
    stored = archive.surfaces.get(s.schema_hash)
    if stored is None or stored.schema.body != s.body:
        return None
    return stored
