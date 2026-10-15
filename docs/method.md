# ReWeave — Method

This document describes the mechanism step by step, at the level a research collaborator needs to read, extend or
challenge the code. [`architecture.md`](architecture.md) describes how the components fit together. Module names
refer to the `reweave/` package.

Contents:

1. [The optimization object](#1-the-optimization-object)
2. [Primitive exact search](#2-primitive-exact-search)
3. [Structural episodes from trajectories](#3-structural-episodes-from-trajectories)
4. [Parameterized relations](#4-parameterized-relations)
5. [Exact certification](#5-exact-certification)
6. [Support, aliasing and admission to the library](#6-support-aliasing-and-admission-to-the-library)
7. [Matching and binding](#7-matching-and-binding)
8. [Visible, latent and compositional reuse](#8-visible-latent-and-compositional-reuse)
9. [Exact lowering and verification](#9-exact-lowering-and-verification)
10. [Library update](#10-library-update)
11. [Policy: admission, scoring and selection](#11-policy-admission-scoring-and-selection)
12. [Policy consolidation](#12-policy-consolidation)
13. [The iterative closed loop](#13-the-iterative-closed-loop)

---

## 1. The optimization object

**Task.** A polynomial `f ∈ ℕ[x₁, …, x₈]` with positive integer coefficients. Coefficients are *scalar-free*: the
term `3xy` is three occurrences of the monomial `xy`, not a multiplication by a constant.

**State.** An exact, hash-consed DAG of three node kinds, held in an arena:

```
Mono(m)                       a monomial (packed exponent vector)
Sum((child, count), ...)      a multiset sum
Prod((child, count), ...)     a multiset product
```

Construction is interning (`mk_mono`, `mk_sum`, `mk_prod`), so equal structure is one handle. Every handle has a
*structural digest* that does not depend on allocation order. Arena ids are never used for identity.

**Cost.** Each state lowers to a binary ADD/MUL gate DAG. The objective `C_sf` is the number of **unique reachable
gates**. Shared sub-circuits are paid once, and monomials are built from multiplication gates. Search starts from the
canonical flat circuit, the sum of the task's monomials.

**Exactness.** The polynomial of every handle is computed with exact arithmetic (memoized, with a homomorphic
fingerprint as a prefilter only). A state is *correct* iff its polynomial equals the task polynomial exactly.

This domain is deliberately strict. Correctness is decidable for every step, the cost is a combinatorial property of
the circuit, and the hard part is purely search: finding short derivations through deep, recurring and often hidden
structure.

## 2. Primitive exact search

The primitive calculus (`legacy_runtime/polycompress/calculus.py`) acts on **exposed Sum contexts** of the state:

- **FACTOR** takes a region (a sub-multiset of a context's children, at most `r_max = 3` of them) and a common factor,
  and replaces the region by `factor · (region / factor)`.
- **REUSE** replaces a region by an existing handle with an equal polynomial, or makes two equal regions share one
  handle.

Every primitive successor is built with the ordinary constructors and verified exactly.

**Search** is a budgeted beam search with budget `(B, K_act, T_max, max_expansions)`. At each expanded state, all
candidate actions are scored by a *scorer*, ranked once and truncated to the top `K_act`. The chosen actions are
applied and verified, and successors are ranked by a *ranker*, which keeps the best `B`. Ties always break by
structural digest, never by id, so search is deterministic. The incumbent is the cheapest correct state found.

The round-0 teacher uses `ImmediateGain` (exact one-step cost gain) and `CostRanker` (current cost). Later rounds use
learned scorers (§11).

## 3. Structural episodes from trajectories

A successful search yields a *keep-best chain* `S₀ → S₁ → … → Sₙ` of exact states and the decisions between them.
`rewrite_extract.py` turns the chain into **ground structural pairs**:

1. For every window `(k, L)` with `2 ≤ L ≤ 4`, take the context `C` where decision `d_k` acts, and up to two
   enclosing Sum contexts.
2. **Replay locally.** Apply `d_k … d_{k+L-1}` starting from `C` alone. Accept the window only if substituting the
   local image back into `S_k` reproduces `S_{k+L}` exactly. Locality is checked, not assumed.
3. **Ground pair.** Let `U` be the children common to `C` and its image. The pair is `R = children(C) − U` and
   `T = children(image) − U`. `poly(R) = poly(T)` is asserted.
4. Pairs above fixed size bounds are rejected, and pairs are deduplicated per task.

A ground pair is one concrete compression episode. One example is the three primitive steps that regroup and factor
`x² + 2xy + y²` into `(x + y)²` inside a larger state. A pair whose target reuses a **pre-existing compound
sub-expression** is marked `requires_ref`. Such pairs are the direct evidence that abstractions need parameters which
bind compound structure, not just monomials (§7).

## 4. Parameterized relations

A relation (`rewrite_schema.py`) is a directed pair of regions over parameters:

```
Pattern := ('M', ((param, e), ...))       ∏ A_param^e   (a product of parameters; ('M', ()) is 1)
         | ('S', ((Pattern, count), ...))
         | ('P', ((Pattern, count), ...))
Region  := ((Pattern, count), ...)         a multiset

Square : { A², AB×2, B² }  →  { P(S(A, B)²) }        i.e.  A² + 2AB + B² → (A + B)²
```

Relations are kept in a **normal form**. Monomial children of a product merge, singleton sums and products collapse,
and children are sorted by canonical serialization. They are hashed by `schema_hash`, the SHA-256 of the canonical
body minimized over parameter permutations, and the automorphism group is recorded. Structural bounds keep relations
local: at most 4 parameters, 6 source terms and pattern depth 4.

**Proposal** (`rewrite_propose.py`). Ground pairs are bucketed by target skeleton and source shape, and the most
frequent pairs seed proposals. A seed proposes candidates in three steps. It chooses a *basis* of at most four target
monomial leaves, decomposes every leaf of the pair over that basis, and rewrites the pair over the resulting
parameters. Each candidate must have a **binding plan**: an order in which the matcher can bind every parameter
deterministically. That is either leaf-greedy, or a bounded exact simultaneous solver for leaves that introduce
several parameters at once.

## 5. Exact certification

`rewrite_certify.certify` replaces parameter `i` by a fresh variable `xᵢ`, evaluates both regions with exact
polynomial arithmetic, and certifies the relation iff the two polynomials are equal. Representation overflow fails
closed.

A certified relation is an identity of the free commutative semiring. Substitution is a semiring homomorphism, so
**every instance is exact, for any values of the parameters**: monomials, compound sub-expressions, or anything else.
This one fact makes compound-handle binding (§7) sound without new proofs.

## 6. Support, aliasing and admission to the library

Certification establishes that a relation is *true*. It says nothing about whether the relation is *worth keeping*.
`rewrite_library.py` decides that with exact evidence:

- **Re-match.** Every certified relation is re-matched exactly against the ground-pair pool, across all its
  bindings.
- **Alias check.** For every instance, a bounded primitive enumeration checks whether one primitive action already
  produces the same successor. The answer is three-valued (`primitive_alias` / `alias_unknown` / `non_alias`). A
  relation with no non-alias instance adds nothing to the primitive language and is excluded.
- **Support.** Over non-alias instances the library counts distinct tasks, distinct canonical target shapes and
  distinct *binding signatures*. A binding signature is the bound exponent matrix, normalized over variable order
  and the relation's automorphisms. It measures genuine *parametric* reuse, not repetition of one instance.
- **Eligibility.** A relation is eligible when its support clears configured thresholds (by default 5 tasks,
  3 shapes, 2 binding signatures) and it is not an alias. Fold-specific libraries, which leave out one fold of
  tasks, provide out-of-fold supervision for the learned scorer.

## 7. Matching and binding

**Sites.** A relation's source is matched as a sub-multiset of the children of a reachable Sum context. Contexts are
visited in digest order, unification is bounded per (context, relation) and per state, candidates are deduplicated
by (site, relation, target) and totally ordered by a stable hash, and the output is capped
(`rewrite_match.match_state`). A cap that binds is always counted.

**ExprParam binding** (`canon/exprparam.py`). Parameters are ExprParams: one semantic parameter type with runtime
binding modes that may be mixed within one match.

| mode | a parameter `A` binds | a leaf `A²·B` matches |
|---|---|---|
| `MONOMIAL_BIND` | a monomial `x^b` | one monomial with exponent `2b_A + b_B` |
| `REACHABLE_HANDLE_BIND` | a compound handle `H` already in the state | a product with children `H×2, K×1` |
| mixed | either | a product of the handle-bound parameters and one merged monomial |

The rules that keep this bounded and exact:

- **Learn from structure, reuse through handles.** A parameter binds only a semantic unit already *materialized* at
  its matched position: a context child, or a factor of a child product. The matcher never synthesizes, regroups or
  searches for compound expressions. For a pattern `A·B` over `h₁h₂h₃h₄` it never tries `A = h₁h₃` unless that
  product already exists as a handle. There is no global enumeration of reachable handles.
- **Semantic equality.** Repeated occurrences of one parameter must bind the *same exact polynomial*. "Nonconstant"
  and "distinct" are semantic tests, never arena-handle identity.
- **Apply-time checks.** Acyclicity (the target never contains the rewritten context), no operand rebuild (bound
  handles are reused as children), a dependency check (a bound unit must lie below the consumed region), and exact
  verification.

## 8. Visible, latent and compositional reuse

A relation should stay reusable when the structure it describes is present without being visible in one obvious
surface form. ReWeave covers four such cases. All of them use the same matcher and the same exact application.

**Visible.** The source is literally a sub-multiset of a context's children.

**Latent quotient** (`rewrite_latent.py`). The source is present only up to a monomial multiplier `d`:
`f·a² + 2f·ac + f·c²` contains `Square` behind `f`. For each context, a bounded deterministic set of divisors `d` is
tried: at most 16 per context, of total degree at most 4, each dividing at least two children. The matcher runs
unchanged on the quotient view `C/d`. A match consumes the original children and replaces them by
`d · T(θ)`. With positive coefficients nothing can cancel. So a hidden factor `H` in `P = H·Q + R` leaves
coefficient-wise detectable scaled copies `c·d·H ≤ P`, which is why the quotient view finds hidden instances
reliably. Depth is exactly one. Exposing one relation never invokes another.

**Compound handles.** Through `REACHABLE_HANDLE_BIND`, a relation induced over monomials applies unchanged to
`H² + 2HK + K²` with `H = a + b` and `K = c + e` existing handles. It combines with the latent quotient: the view
of `c·(H² + 2HK + K²)` is `{H², HK×2, K²}`, so `c·Square(H, K)` is offered within the same bounds.

**Semantic classes and views** (`canon/semantic.py`, `canon/views.py`). Reachable handles with equal exact
polynomials form a *semantic class* with a bounded portfolio of structurally distinct representatives.
**Normalization** swaps in a cheaper representative only on a strict whole-state cost decrease. A **view action**
substitutes an alternative representative and then applies a rewrite that matches only in that representative.
Indexed view retrieval uses cheap structural necessary conditions, so it never misses a candidate the eager matcher
would find.

## 9. Exact lowering and verification

Applying a rewrite candidate (`rewrite_match.apply_candidate`, `rewrite_latent.apply_latent`,
`exprparam.apply_candidate_g`) has six mandatory steps:

1. **Lower** the target region with the bindings to ordinary `Mono` / `Sum` / `Prod` handles.
2. **Splice**: `context' = mk_sum(children(context) − R_inst + T_inst)`, with `d · T_inst` for a latent match.
3. **Substitute** `context → context'` through the state.
4. **Reject no-ops.**
5. **Exact-verify** the whole successor against the task polynomial. An inexact successor is never admitted, and is
   counted as an exact-verification failure.
6. **Cost** the ordinary successor.

No library symbol survives into a costed circuit. Rebuilding a successor in a fresh arena with no library present
gives the same cost (tested in `tests/test_exactness.py`). Library use changes representation, retrieval and
effective horizon. It never changes the cost function, so **an abstraction gets no cost discount**. A rewrite may
lower, keep or raise the cost, and search decides whether to take it.

## 10. Library update

Admitted relations enter a **certified archive** (`canon/archive.py`). `add_certified` re-certifies every relation on
insertion. The archive is monotone and stores three layers separately:

- a **family**: one exact parameterized relation, identified by its `CANONICAL_SEMANTIC_FAMILY_ID`;
- a **surface**: one matchable source pattern of a family, identified by its `schema_hash` (a family may have
  several);
- interned **handles**, which belong to search states, not to the library.

The family id is computed on the generic instance (parameter `i ↦ xᵢ`), normalized modulo associativity and
commutativity of ADD and MUL (not distributivity) in `canon/relation.py`, and minimized over parameter
permutations. Binding modes never create separate families. Two surfaces that differ only in how a sub-sum is
associated are one family. Digests index, and exact payloads decide equality, so a digest collision stays two
families.

A newly admitted relation is visible to search **from the next round**, never within the round that produced it.

## 11. Policy: admission, scoring and selection

**Admission** (`candidate_policy.py`). Rewrite candidates of a state compete for a fixed number of slots under a
deterministic policy. `CM-ORIGIN-SCHEMA-DIVERSITY` buckets candidates by (origin, relation) and round-robins across
buckets, so one prolific relation cannot crowd out every other. A policy never changes a cap and never reads a
score.

**Scoring.** Primitive and rewrite candidates are scored in **one encoder pass**, ranked together, and truncated
once by `K_act`. There is no rewrite quota and no second beam. One admitted successor counts as one expansion,
whatever produced it.

- `rewrite_model.py` is a message-passing state encoder with primitive, rewrite and value heads. Its rewrite head
  combines context and region embeddings, a per-relation embedding, relation statistics and the bound monomials.
- `canon/action_model.py` / `action_policy.py` is the **schema-agnostic** successor. It reads each action only through its
  `ActionRecord`. The per-relation embedding is replaced by an encoding of the relation's *structure*, and the
  binding mean by per-*role* encodings of each parameter in canonical family order, plus the latent multiplier and
  the concrete target. A newly certified family is scorable with no new embedding and no change of model shape.

**Supervision.** Labels come from exact continuations. At a labelled teacher state, each exactly verified rewrite
successor is evaluated by a fixed primitive continuation search. The listwise target moves from the teacher's
primitive to a rewrite only when that rewrite's exact continuation outcome is strictly better.

## 12. Policy consolidation

Being able to *represent* a new family is not the same as knowing when it is *useful*. ReWeave does not assume that
a newly certified relation gets a good zero-shot score. The loop is therefore designed with an explicit
consolidation step:

```
(L_t, π_t) ──SEARCH──► D_t ──INDUCE──► R ──CERTIFY──► L_{t+1} = L_t ∪ {R} ──CONSOLIDATE──► π_{t+1}
```

1. **Origin evidence.** Record the episodes `R` was induced from: the tasks, trajectory states and windows on which
   a surface of `R` re-matched exactly.
2. **Retroactive replay.** Relabel those source trajectories with the library `L_t ∪ {R}`. Candidates are generated,
   applied and exactly verified as in search. Each state is labelled by the same exact continuation rule (§11),
   including counterfactual successors off the original trajectory.
3. **Chronology.** Only experience from rounds no later than `t`, from authorized training populations, may be used.
4. **Update.** The replayed records become ordinary training data for the same objective, with no special loss and
   no oversampling.

`R`'s semantics are frozen at certification. Consolidation only changes the policy that *selects* `R`.
Certification establishes correctness, and continuation labels establish usefulness.

## 13. The iterative closed loop

One round of the intended loop:

```
round t:   search with (L_t, π_t)             → experience D_t  (exact trajectories, states, outcomes)
           structural-episode induction on D_t → candidate relations
           exact certification                → L_{t+1}  (monotone; visible only from round t+1)
           consolidation                      → π_{t+1}
```

Three design rules govern the iteration:

- **Outcome-blind replay and genealogy.** Each relation records its *birth*: the round, the trajectories and the
  states it was induced from, and which existing relations those trajectories used. This genealogy is causal. A
  relation `B` depends on `A` when the experience `B` was induced from was reachable only because `A` was in the
  library. It does not matter whether `B`'s pattern mentions `A`.
- **No same-round feedback.** A relation never influences the search of the round it was born in.
- **Structural-episode induction over ExprParams.** To induce from trajectories that already use learned relations
  and compound handles, an episode is backtraced semantically to a self-contained source and target. Handles are
  expanded to their exact polynomials, and at most four genuinely independent ExprParams are extracted. Then the
  formal relation is built and certified exactly as in §5. Because identity is semantic, the same relation is
  recognized whether it was observed over monomials, over compound handles, or through a latent multiplier.

The intended outcome is an **abstraction curriculum**. Abstractions from round `t` reshape the states reached in
round `t+1`, and those states expose regularities that round `t+1` can turn into new abstractions. For example,
`Square` makes `A·(A+B)² + B·(A+B)²` cheap to reach, which exposes `Cube`. The library grows into an
increasingly expressive language, while every individual step stays exactly verified.
