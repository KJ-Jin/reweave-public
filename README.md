# ReWeave

**ReWeave is a learning-guided exact optimization framework. It turns successful search experience into certified,
reusable abstractions, and uses those abstractions to reshape future search. It then mines the experience that
results for further abstractions. Throughout, correctness stays outside the learned policy.**

The question behind it is simple to state. Can exact search bootstrap its own abstraction language? Can each
generation of learned abstractions change search enough to produce the experience that the next generation is
learned from?

---

## The problem

An exact optimizer usually searches over a fixed set of primitive moves. Every problem instance is solved from
those primitives, and every instance pays again to discover the same intermediate structure. Suppose ten thousand
problems all contain a perfect square in some disguise. A primitive-only optimizer rediscovers the multi-step route
to `(A + B)²` ten thousand times. It never learns that this route is one thing.

This has three costs:

- **Search effort is spent on structure that is already known.** The work grows with the depth of the primitive
  derivation, not with how new the problem is.
- **Effective horizon is wasted.** Under a fixed budget, the steps used to re-derive familiar structure are steps
  not used for the unfamiliar part of the problem.
- **What search experiences is limited by what its moves can reach.** Higher-level regularities only appear in states
  that are reachable cheaply enough. A fixed action language never reaches them, and so never sees them.

ReWeave treats the action language itself as something that should be learned and that should grow.

## The idea

```
        exact search
             │
             ▼
   successful structural experience
             │
             ▼
   induce reusable parameterized relations
             │
             ▼
   exact certification ──────────────► persistent abstraction library
                                                │
                                                ▼
                                  learn when / how to use them
                                                │
                                                ▼
                            search with the enlarged abstraction language
                                                │
                                                ▼
                           reach different states and trajectories
                                                │
                                                ▼
                            expose new recurring structure  ──► induce ... ↺
```

Formally, ReWeave maintains a certified library `L_t` and an action-selection policy `π_t`. Each round of the
intended closed loop is:

```
(L_t, π_t) ──SEARCH──► D_t ──INDUCE──► candidates ──CERTIFY──► L_{t+1}
          ──CONSOLIDATE──► π_{t+1} ──SEARCH──► D_{t+1} ──► ...
```

Here `D_t` is the search experience of round `t`: trajectories, intermediate states and outcomes.

The central point is not "learn a library of rewrites." A fixed library learned once is a cache. The point is that
**learned abstractions change future search enough to create the experience from which further abstractions can be
learned**. The library is the input to the next round's experience, not only the output of the last one.

## A motivating example: from square to cube

Here is one intuitive example of the abstraction curriculum ReWeave is designed to support. It is an illustration,
not the definition of the mechanism.

1. **Early experience.** Primitive search on many polynomials keeps making the same multi-step move. It regroups
   `A² + 2AB + B²` and factors it into `(A + B)²`, for many different `A` and `B`.
2. **Induction.** ReWeave abstracts these episodes into one parameterized relation:
   `Square(A, B) : A² + 2AB + B² → (A + B)²`.
   The relation is certified exactly as a polynomial identity and added to the library. (The minimal example below
   does exactly this, from primitive-search trajectories alone.)
3. **Changed search.** `Square` is now one action. Take a cube `A³ + 3A²B + 3AB² + B³`. Search can split it as
   `A·(A² + 2AB + B²) + B·(A² + 2AB + B²)`, collapse both squares in one step each into a shared handle
   `S = (A + B)²`, and factor to `(A + B)·S`. States like `A·S + B·S` were too deep to reach cheaply before.
   Now they are on ordinary trajectories.
4. **New structure.** Those trajectories expose a new recurring episode: a cube that collapses into
   `(A + B)·(A + B)²`. Induction can turn it into
   `Cube(A, B) : A³ + 3A²B + 3AB² + B³ → (A + B)³`.

`Cube` is defined over its own parameters `A` and `B`. Its source pattern does not need to contain the token
`Square`. What makes `Cube` dependent on `Square` is **causal, not syntactic**. Using `Square` changed which states
search reached, and that changed experience is what exposed `Cube`.

```
abstraction A  ──►  changes reachable search experience  ──►  new recurring regularities  ──►  abstraction B
```

Literal nesting such as `C(B(A, ·), ·)` is one possible form of this genealogy. It is not required. The genealogy
that matters runs through experience.

## Design principles

### 1. Exactness lives outside the learned component

```
symbolic induction    →  proposes a reusable relation
exact certification   →  establishes that it is valid
learned policy        →  decides when to use it
exact verification    →  protects every deployed successor
```

A relation enters the library only if it is a formal identity. Each parameter is replaced by a fresh variable and
the two sides are compared with exact polynomial arithmetic. At search time, every rewrite is *lowered* to the
ordinary concrete circuit and spliced into the state. The successor is then verified exactly against the target
polynomial. A successor that fails verification is counted and discarded, never admitted. The learned model only
ranks actions. It never decides what is correct.

Cost is always measured on the ordinary lowered computation. Here that is the number of unique ADD/MUL gates.
**Abstractions get no cost discount.** Using one changes representation, retrieval and effective search horizon. It
never changes the cost function or the class of solutions.

### 2. Abstractions are semantic relations, not memorized transitions

ReWeave is not a macro cache. A library entry is a **parameterized exact relation**. It is reusable across instances,
transferable across variables and problem families, composable with every other action, and exact by construction.
Its identity is its meaning, not its incidental form:

- a **canonical semantic family id** is computed from the generic instance of the relation. Associativity and
  commutativity of ADD and MUL are normalized, and the id is minimized over parameter permutations;
- surface syntax, variable names, arena allocation ids, the id of the action that created a sub-expression, or which
  equivalent representative happens to be present **never** define the abstraction.

Two recognition surfaces of one relation are one family. Two arenas that build the same state in different
allocation orders produce the same semantic action keys.

### 3. Learn from structure, reuse through handles

A relation's parameters are **ExprParams**: one semantic parameter type with several runtime binding modes. A
parameter can bind:

- a **monomial value** (`A := ab`), or
- an **existing, reachable compound DAG handle** (`A := (x + y)`, already built somewhere in the current state), or
- a semantic / view representative where alternative representations are retained.

Binding is bounded. A parameter binds only semantic units already *materialized* in the matcher-visible
representation, at its matched position. The matcher never synthesizes, regroups or searches for hidden
sub-expressions. So `Square` applies equally to `x² + 2xy + y²` and to `H² + 2HK + K²` with `H = a + b` and
`K = c + e`, within the same matcher budget.

### 4. Hidden and non-obvious structure

Useful structure is often present without being visible in one obvious surface form. ReWeave is designed so that a
learned abstraction stays reusable when that happens:

- **Latent quotient exposure.** A relation's source may be present only up to a monomial multiplier, as in
  `f·a² + 2f·ac + f·c²`. The unchanged matcher runs on bounded monomial-quotient views of a context, and the rewrite
  `d·S(θ) → d·T(θ)` is applied exactly. With positive coefficients nothing cancels, so a hidden factor leaves exactly
  detectable scaled copies of the source.
- **Semantic interning and normalization.** Sub-expressions with equal exact polynomials form one semantic class
  with a bounded portfolio of representatives. A cheaper representative replaces a costlier one only on a strict
  whole-state cost decrease.
- **Equivalent views.** An action can substitute an alternative representative and then apply a rewrite that only
  matches in that representation.
- **Compound-handle binding.** Through ExprParams (above), a relation learned on monomials applies unchanged over
  compound sub-expressions.

### 5. An open-ended library

The library is meant to grow over rounds. So the architecture does not depend on a fixed vocabulary of schema ids,
on one learned embedding per rewrite family, or on scanning every stored rewrite at every state. Instead it is built
on:

- **canonical semantic-family identity** and **semantic interning**;
- a monotone **certified archive** that separates families, recognition surfaces and handles;
- **indexed retrieval**: sound structural necessary conditions choose which surfaces the exact matcher sees;
- a generic **ActionRecord** for every action. It has separate semantic, concrete, provenance and feature parts, and
  never contains a runtime id;
- **schema-agnostic scoring**: the policy reads a relation's structure and its parameter roles, not its id, so a
  newly certified family can be represented and scored with no new embedding and no change of model shape.

### 6. Policy consolidation

Being able to *represent* a new family does not mean the policy knows when it is *useful*. ReWeave does not assume a
newly certified relation gets a good zero-shot score. The intended loop has an explicit consolidation step:

```
discover relation → certify relation → add to library
    → derive exact training / replay evidence → update action-selection policy → search again
```

The experience that produced a certified relation is replayed with the enlarged library. This includes its origin
states and their exact counterfactual continuations, and it becomes ordinary training data for the policy. The
relation's correctness is frozen at certification. Consolidation only changes the policy that *selects* it:
certification establishes correctness, and continuation labels establish usefulness.

## The current testbed

ReWeave is developed on **exact compression of monotone arithmetic circuits**. A task is a polynomial with positive
integer coefficients. A state is an exact, hash-consed ADD/MUL circuit DAG that computes it, and the objective is the
number of unique gates. The primitive actions are exact FACTOR and REUSE moves over exposed sub-sums. The domain is
small enough for every step to be verified exactly. It is still rich enough to have deep, recurring and hidden
structure: squares, cubes, shared factors and compositions of these.

## Quick start

Requirements: Python 3.10 or newer (tested with 3.11), plus `numpy` and `torch` (CPU is enough).

```bash
git clone https://github.com/KJ-Jin/reweave-public.git
cd reweave-public
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"

python examples/minimal_example.py     # one turn of the abstraction loop, a few seconds
pytest                                 # the test suite, about 20 seconds
```

Use an editable install (`-e`). ReWeave loads its vendored exact runtime from `legacy_runtime/` and checks its pin
there.

## Minimal example

[`examples/minimal_example.py`](examples/minimal_example.py) runs one turn of the loop on the real implementation.
Nothing in it is mocked:

```
1. SEARCH: primitive-only exact search on training polynomials
   task 0:  6 gates ->  2 gates
   ...
2. INDUCE + 3. CERTIFY: trajectory windows -> parameterized relations -> exact certification
   18 ground pairs, 2 certified relations, 2 admitted after support / alias / eligibility checks
   family 3bfa611ffc  certified  2*A*B + A^2 + B^2  ->  ((A + B)^2)
   family a3ab4b55bb  certified  2*A*B + A^2 + B^2  ->  (A*(A + B)) + (B*(A + B))

4. REUSE on unseen instances (each successor is lowered and exact-verified)
   visible   (d+g)^2 + eh         :  8 ->  4 gates  [ok]
   latent    f(a+c)^2 + b         : 10 ->  4 gates  [ok]  (0 visible matches; quotient d = f)
   compound  H^2+2HK+K^2, H=a+b.. :  8 ->  4 gates  [ok]  (0 monomial-only matches)
   semantic action key independent of arena allocation: True  (5d79fa7caa667067…)

5. SEARCH AGAIN: same small budget (B=2, K_act=2, T=6, 20 expansions), primitive vs. enlarged language
   f(ab+c)^2 + (d+e)^2        start 18   primitive 13   with library  7   (rewrites applied 5, exact-verification failures 0)
   c(a+b)^2 + d(a+b)^2        start 19   primitive 10   with library  4   (rewrites applied 2, exact-verification failures 0)
   g(c+d)^2 + h(c+d)^2 + e    start 20   primitive 13   with library  5   (rewrites applied 2, exact-verification failures 0)
```

The Square relation is induced from scratch, from primitive-search trajectories only. It is byte-identical to the
Square in the reference library (`schema_hash 3cac496e…`). It is then reused in three ways it was never shown:
visibly, behind a hidden multiplier `f`, and over compound sub-expressions. Search with the enlarged language reaches
smaller circuits under the same budget. Every successor is exact-verified, and every cost is the ordinary gate count.

## Repository layout

```text
reweave/                     the framework
  rewrite_schema.py          parameterized relations: patterns, canonical form, binding plan, lowering
  rewrite_extract.py         trajectory windows -> ground structural pairs
  rewrite_propose.py         ground pairs -> candidate parameterized relations
  rewrite_certify.py         exact certification (formal polynomial identity)
  rewrite_library.py         exact re-match support, alias check, eligibility -> library
  rewrite_match.py           runtime matching, lowering, splicing, exact verification
  rewrite_latent.py          latent monomial-quotient exposure
  rewrite_search_latent.py   beam search over primitive + visible + latent actions
  candidate_policy.py        deterministic candidate admission (fixed slots, fair across relations)
  rewrite_model.py, recipe.py  learned scorer (state encoder, action and value heads) and training recipe
  canon/                     the canonical architecture for an open-ended library
    relation.py              AC-normalized relation IR, semantic family identity
    archive.py               certified archive: families, surfaces, structural index, retrieval
    exprparam.py             generic ExprParam binding (monomials and reachable compound handles)
    action_record.py         ActionRecord: semantic / concrete / provenance / features
    candidate_set.py         per-state candidate-set records (completeness, every truncation)
    semantic.py, views.py    semantic interning, normalization, equivalent-view actions, view retrieval
    identity*.py, latent_identity.py   symmetry-safe canonical states and quotient action classes
    action_features.py, action_model.py, action_policy.py   schema-agnostic action scoring
    search.py, config.py     the canonical search loop and its explicit switches
  data/reference_library.json   the 19 certified relations of the reference library
legacy_runtime/polycompress/ the vendored exact runtime: polynomials, circuit DAGs, primitive calculus, verifier
examples/minimal_example.py  one turn of the abstraction loop
tests/                       exactness, rewrite, binding, identity, archive, scoring and example tests
docs/method.md               the mechanism, step by step
docs/architecture.md         the components and how they fit together
```

## Further reading

- [`docs/method.md`](docs/method.md): the state representation, primitive search, structural episodes, relations,
  certification, matching and binding, latent and compositional reuse, lowering and verification, library update,
  policy selection and consolidation, and the iterative loop.
- [`docs/architecture.md`](docs/architecture.md): semantic relation, family identity, certified archive, retrieval,
  ExprParam binding, ActionRecord, scoring, application, verification, search, induction and consolidation, and how
  they connect.
- [`legacy_runtime/README.md`](legacy_runtime/README.md): the vendored exact runtime.

## License

MIT. See [`LICENSE`](LICENSE).
