# ReWeave — Architecture

This document describes the components of ReWeave and how they connect. The architecture is organized around one
requirement: **the library must be able to grow without bound, while every step stays exact.** That rules out
anything keyed by a fixed vocabulary, such as a schema-id embedding table, a per-family code path, or a scan of
every stored rewrite at every state. It also rules out anything that lets a learned component decide correctness.

[`method.md`](method.md) explains each mechanism step by step.

```
                         ┌──────────────────────────── induction ◄──────────── experience D_t ◄──┐
                         │  episodes → ExprParams → relation → certify                          │
                         ▼                                                                       │
  ┌───────────── certified archive  L_t ──────────────┐                                          │
  │ family (semantic relation, canonical family id)   │                                          │
  │   └─ surfaces (matchable source patterns)          │                                          │
  │ structural index ─► retrieval                      │                                          │
  └───────────────────────────┬────────────────────────┘                                          │
                              │ retrieved surfaces                                                │
                              ▼                                                                   │
   state ──► generation lanes: primitive │ visible │ latent │ handle (ExprParam) │ view           │
                              │                                                                   │
                              ▼                                                                   │
                ActionRecord per action  +  CandidateSetRecord per state                          │
                              │                                                                   │
                              ▼                                                                   │
            admission (fixed slots)  ──►  scoring π_t (schema-agnostic)  ──►  top-K_act           │
                              │                                                                   │
                              ▼                                                                   │
           application: lower → splice → substitute → EXACT VERIFY → cost (ordinary circuit)      │
                              │                                                                   │
                              ▼                                                                   │
              normalize → verify → canonicalize → semantic dedup → intern  ──►  beam search ──────┘
                                                                                     │
                              consolidation (replay with L_{t+1}, exact labels) ──► π_{t+1}
```

## Components

| component | role | module |
|---|---|---|
| exact runtime | polynomials, interned circuit DAG, gate cost, primitive calculus, verifier, beam search | `legacy_runtime/polycompress`, via `reweave/v1.py` |
| semantic relation | a parameterized exact relation over ExprParams, in normal form | `rewrite_schema.py` |
| relation IR and family identity | AC-normalized relation and `CANONICAL_SEMANTIC_FAMILY_ID` | `canon/relation.py`, `canon/archive.py` |
| certification | formal identity under exact arithmetic | `rewrite_certify.py` |
| certified archive | families, surfaces, monotone insertion, aliases, JSON round trip | `canon/archive.py` |
| retrieval | sound structural index over surfaces | `canon/archive.py`, `canon/views.py` |
| ExprParam binding | monomial and reachable-handle binding, fair handle lane | `canon/exprparam.py` |
| latent exposure | monomial-quotient views | `rewrite_latent.py` |
| semantic interning, views | semantic classes, portfolios, normalization, view actions | `canon/semantic.py`, `canon/views.py` |
| canonical identity | renaming-safe canonical states and quotient action classes | `canon/identity.py`, `canon/latent_identity.py` |
| ActionRecord | generic action representation | `canon/action_record.py` |
| CandidateSetRecord | per-state completeness and truncation | `canon/candidate_set.py` |
| admission | deterministic slot allocation across relations | `candidate_policy.py` |
| scoring | fixed-vocabulary and schema-agnostic scorers | `rewrite_model.py`, `canon/action_*.py` |
| application and verification | lower, splice, substitute, verify, cost | `rewrite_match.py`, `rewrite_latent.py`, `canon/exprparam.py` |
| search | beam search over all action kinds; explicit switches | `rewrite_search_latent.py`, `canon/search.py`, `canon/config.py` |
| induction | trajectory episodes → relations → support → library | `rewrite_extract.py`, `rewrite_propose.py`, `rewrite_library.py` |
| consolidation and iteration | replay with the enlarged library, policy update, rounds, genealogy | design (see [method §12–13](method.md#12-policy-consolidation)) |

## Semantic relation

The unit of the library is a **relation**: a directed pair of regions over ExprParams, certified as a polynomial
identity. It is not a recorded transition or a concrete subgraph. The same relation describes `x² + 2xy + y² →
(x + y)²`, `(ab)² + 2abc + c² → (ab + c)²` and `H² + 2HK + K² → (H + K)²` for compound `H` and `K`.

## Canonical family identity

Identity is layered so that only meaning defines an abstraction:

```
CANONICAL_SEMANTIC_FAMILY_ID    the relation itself: generic instance (param i ↦ x_i), AC-normalized,
                                minimized over parameter permutations; binding modes excluded
LEGACY_SURFACE_ID (schema_hash) one recognition surface of a family; never rewritten, so recorded
                                artifacts stay reproducible
semantic action key             (family relation, parameter values as exact polynomials in canonical
                                family order, latent multiplier d, site)
concrete realization key        (site, d, exact lowered target DAG)
```

None of these contains an arena id, a creator id, a representative id or allocation order. **Digests index, and exact
payloads decide.** Every digest-keyed grouping is confirmed by exact comparison, so a collision splits into two
entries instead of merging them.

## Certified archive

A monotone store with three separated layers:

- **Family**: one exact relation, with its certification record, provenance and round of birth.
- **Surface**: one matchable source pattern of a family. A family can have several, for example two associations
  of the same sub-sum.
- **Handle**: interned search-state structure, which lives in the arena and is never stored in the library.

`add_certified` re-certifies every relation on insertion and rejects anything uncertified. The archive serializes to
deterministic JSON and rebuilds exactly from it. Building it from a shuffled list produces the same families. The
reference library (`reweave/data/reference_library.json`) holds 19 surfaces. Modulo associativity and commutativity
they form 18 families.

## Retrieval

Scanning every surface at every state does not scale with an open-ended library. The archive keeps a **structural
index** of per-surface necessary conditions: root kind, per-kind multiplicities, exponent divisibility and anchor
signatures. At each context it retrieves only the surfaces whose conditions the context satisfies. Retrieval is
sound by construction: it can only remove surfaces that could not match. The exact matcher still decides every
match, and indexed retrieval is tested to equal exhaustive matching. The same pattern serves view retrieval: a
cheap compatibility index over semantic classes, followed by the exact matcher.

## ExprParam binding

ExprParams make a relation's parameters semantic, with binding modes chosen at runtime: monomial, reachable compound
handle, or a mix. View and portfolio representatives are designed as a further mode through the existing view
substitution. Binding is bounded by construction. Only units already materialized at the matched position can be
bound, so the handle lane runs within the ordinary matcher budgets.

The handle lane is scheduled **fairly**. Families are resumable generators that run round-robin with a fixed work
quantum under one global allowance. The admitted set is then a function of the *set* of relations, not of library
order: permuting the archive changes nothing, and adding a family changes the schedule in a recorded way.

## ActionRecord

Every selectable action is turned into one `ActionRecord` before scoring. This includes primitives, visible
rewrites, latent rewrites and view rewrites:

```
ActionRecord = { kind, semantic_key, semantic, concrete_key, concrete, provenance, features }
  semantic     what the action means      (defines "the same action")
  concrete     what it builds             (defines "the same successor")
  provenance   where it came from         (origin, lane, surfaces, replay information; never identity)
  features     admissible scorer inputs   (adding a feature never changes a key)
```

A record carries enough replay information to rebuild its live candidate from structural digests alone. A
`CandidateSetRecord` describes each expanded state: per generation lane, what was considered, what was admitted, and
every truncation cause. Completeness belongs to the candidate set, never to an action. "No action exists" is
reported only for a lane that was generated completely and found nothing, so absence from a truncated set is never
treated as evidence.

## Scoring

The policy consumes ActionRecords. It never consumes ids.

- **Fixed-vocabulary scorer** (`rewrite_model.py`). A message-passing state encoder with primitive, rewrite and
  value heads; its rewrite head includes a per-relation embedding. It is valid for a fixed library.
- **Schema-agnostic scorer** (`canon/action_features.py`, `action_model.py`, `action_policy.py`). The per-relation embedding is
  replaced by an encoding of the relation's structure, computed from the exact canonical relation. The monomial-only
  binding mean is replaced by per-role parameter encodings in canonical family order, plus the latent multiplier,
  the concrete target and the action kind. Features are invariant to variable renaming, library order, arena
  allocation and the choice among equivalent representatives. A newborn family needs no new parameters.

Primitive and rewrite actions share **one ranking on a common scale**, with one `K_act` truncation. Before scoring, a
deterministic **admission policy** decides which rewrite candidates occupy the fixed slots. It is fair across
(origin, relation) buckets and never reads a score.

## Application and verification

Application is the only place where new state structure is created, and every path through it ends in exact
verification. The steps are lower the target with the bindings, splice it into the context (times `d` for a latent
match), substitute through the state, and reject no-ops. Then exact-verify the whole successor against the task
polynomial and cost the ordinary circuit. ExprParam application adds an acyclicity check, a no-rebuild check and a
dependency check before verification. Rejections are counted by cause.

## Search

The canonical search loop (`canon/search.py`) is a beam search over all action kinds at once. Each architectural
component (semantic interning, normalization, archive retrieval, symmetry-safe identity, views, generic ExprParam
binding) sits behind an explicit switch in `canon/config.py`. With every switch off, the loop is behaviour-identical
to the reference loop. Every behaviour-changing switch enters the configuration hash, so a result always names the
exact configuration that produced it.

Per expanded state, the order is fixed:

```
generate actions → admission → score / select → apply → exact verify
                 → normalize (if enabled) → exact verify → canonicalize (identity mode)
                 → semantic dedup (visited) → intern / portfolio update
```

In **identity mode**, search runs over canonical states in the task's canonical variable frame. Actions are quotient
classes closed under the automorphisms of the state, so bounded generation commutes with variable renaming: renaming
neither splits nor merges actions.

## Induction

```
successful exact trajectory
  → windowed local replay           (rewrite_extract)   ground pairs R → T, poly(R) = poly(T)
  → proposal over parameters         (rewrite_propose)   basis, decomposition, normal form, binding plan
  → exact certification              (rewrite_certify)
  → re-match support, alias check    (rewrite_library)   tasks, shapes, binding signatures; non-alias only
  → eligibility                                          configured thresholds
  → archive insertion                (canon/archive)     family identity, monotone; visible next round
```

Pairs whose target reuses existing compound structure are marked `requires_ref`. The induction interface is
designed to extend to them: a semantic backtrace expands handles to exact polynomials and extracts at most four
independent ExprParams before certification. The induced relation then binds compound handles at runtime, which is
the principle *learn from structure, reuse through handles*.

## Consolidation

When a family enters the library, the experience it was induced from is replayed with the enlarged library. Origin
states and counterfactual successors are labelled by exact continuation, under a chronology rule that admits only
experience from rounds no later than the current one. The replayed records become ordinary training data for the
policy's existing objective. The relation is frozen, and only the policy that selects it changes.

## Invariants

These hold for every configuration, and the tests check them:

1. **Exactness.** Every admitted successor is an ordinary circuit whose polynomial equals the task polynomial.
2. **No cost discount.** Cost is computed on the ordinary lowered circuit. Rebuilding a successor without any
   library gives the same cost.
3. **Certified library only.** The archive admits only relations that pass exact certification.
4. **Semantic identity.** No identity, key or schedule depends on arena ids, creator ids, representative choice,
   variable names or library order.
5. **Counted truncation.** Every cap that binds is recorded, so incompleteness is never silent.
