# ReWeave

**Learn how to search. Learn what to search with.**

ReWeave is a learning-guided exact optimization framework that studies whether an optimizer can improve both **how it searches** and **the abstraction language it searches with**.

Most optimization systems begin with a fixed vocabulary of primitive transformations. Even when search repeatedly rediscovers the same useful higher-level structure, that knowledge usually disappears with the trajectory that found it.

ReWeave takes a different view:

> **Successful search experience should become part of the optimizer.**

The system is designed to turn exact-search experience into two persistent forms of knowledge:

1. a better policy for deciding **how to search**, and
2. a growing library of certified abstractions that changes **what search can search with**.

The intended closed loop is

```text
                         exactness
                            │
                            ▼
                       (π_t, L_t)
                            │
                            ▼
                       exact search
                            │
                            ▼
                    experience D_t
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
      policy improvement            abstraction induction
       learn how to search          learn what to search with
             │                             │
             ▼                             ▼
          π_{t+1}                       L_{t+1}
             │                             │
             └──────────────┬──────────────┘
                            ▼
                       search again
                            ↺
```

or, compactly,

\[
\boxed{
(\pi_t,L_t)
\;\xrightarrow{\text{exact search}}\;
D_{t+1}
\;\xrightarrow{\text{learning}}\;
(\pi_{t+1},L_{t+1})
}
\]

where

- \(\pi_t\) is the current search/action-selection policy,
- \(L_t\) is the current certified abstraction library,
- \(D_t\) is the experience produced by exact search.

The long-term goal is an optimizer that becomes better at both **using its current language** and **expanding that language from experience**.

---

## Why ReWeave?

Search can improve in two fundamentally different ways.

One is to become better at choosing among the actions it already has.

The other is to acquire better actions.

ReWeave is interested in both.

### Learning how to search

A stronger search procedure can discover decisions and trajectories that are better than those suggested by the current policy alone.

Those outcomes can supervise the next policy:

\[
\pi_t
\rightarrow
\text{search}
\rightarrow
\text{better search evidence}
\rightarrow
\pi_{t+1}.
\]

This has the structure of **Expert Iteration**: search acts as an expert, and policy learning distills useful search decisions into a better future policy.

This outer loop is independent of any particular search algorithm. Bounded search, best-first graph search, MCTS, or another exact search procedure can play the expert role.

The important question is:

> **Can stronger search teach the next search policy to make better decisions?**

### Learning what to search with

The same search experience can contain reusable transformations.

Instead of forcing the optimizer to reconstruct those transformations from primitive actions on every new problem, ReWeave attempts to recover the reusable relation behind successful structural episodes:

\[
S(\theta)\rightarrow T(\theta).
\]

After exact certification, such a relation becomes a first-class search operator.

This gives a second form of improvement:

\[
L_t\rightarrow L_{t+1}.
\]

The optimizer's effective action language can therefore change over time.

---

## A Coupled Self-Improvement Loop

The two forms of learning are most useful when they interact.

A better policy makes better use of the abstractions currently available:

\[
\text{better policy}
\Rightarrow
\text{better abstraction consumption}
\Rightarrow
\text{different search trajectories}.
\]

A richer abstraction library changes what search can do:

\[
\text{better library}
\Rightarrow
\text{richer action space}
\Rightarrow
\text{different search experience}
\Rightarrow
\text{better policy data}.
\]

ReWeave is therefore not intended to train a policy forever over a permanently fixed action vocabulary, nor merely to learn a rewrite library once and stop.

Instead,

\[
\boxed{
\text{search experience}
\rightarrow
\begin{cases}
\text{better search policy}\\
\text{richer abstraction language}
\end{cases}
\rightarrow
\text{new search experience}
}
\]

The two loops share the same evolving source of experience and are intended to bootstrap one another.

---

## An Abstraction Curriculum

The central long-term goal is **abstraction growth across rounds of search**.

Suppose early search repeatedly discovers some useful structural relation. ReWeave may induce and certify an abstraction \(A\) from those trajectories.

Once \(A\) becomes available, future search changes. It no longer has to reconstruct the same structure entirely from primitive steps, and it may reach states that were previously too expensive or unlikely to reach.

Those new trajectories may expose another recurring relation \(B\).

```text
learn abstraction A
        │
        ▼
deploy A during search
        │
        ▼
reach different states and trajectories
        │
        ▼
observe new recurring structure
        │
        ▼
learn abstraction B
        │
        ▼
deploy B
        │
        ↺
```

The hierarchy here is **causal**, not necessarily syntactic.

A later abstraction does not have to literally contain an earlier abstraction as a symbolic token.

The important dependency may simply be

\[
A
\rightarrow
\text{changed search}
\rightarrow
\text{new experience}
\rightarrow
B.
\]

In this sense, learned abstractions help construct the future training distribution of the optimizer itself.

### A motivating example: square to cube

Suppose primitive search repeatedly discovers

\[
A^2+2AB+B^2
\rightarrow
(A+B)^2.
\]

ReWeave can abstract those episodes into a reusable Square relation and certify it exactly.

Once Square is available, search can collapse square-like regions directly. That changes which intermediate states fit inside a fixed search budget. On cube-like problems, search may now reach states containing shared \((A+B)^2\)-structure that primitive search had to spend several steps constructing.

Those changed trajectories can expose a higher-order recurring relation such as

\[
A^3+3A^2B+3AB^2+B^3
\rightarrow
(A+B)^3.
\]

The Cube abstraction does not need to contain a literal `Square` token.

Its dependence on Square can instead be:

```text
Square becomes available
        │
        ▼
Square changes future search
        │
        ▼
new structural experience appears
        │
        ▼
Cube-like structure becomes learnable
```

The square-to-cube example is only an illustration of the broader target:

> **Earlier abstractions should create the conditions under which later abstractions become discoverable.**

---

## A Cumulative Curriculum

Growing toward harder structure should not mean forgetting earlier competence.

ReWeave therefore treats curriculum growth as

\[
\boxed{\text{retention}+\text{expansion}}
\]

rather than

```text
Level 0 → forget → Level 1 → forget → Level 2.
```

As the search frontier moves outward, earlier experience remains useful.

Conceptually, later learning can draw on

```text
historical replay
        +
current search experience
        +
frontier / stretch experience
        +
exact counterfactual replay
```

The exact mixture is a design choice. The principle is that **new competence should extend earlier competence rather than replace it**.

This matters because a growing action language requires the policy to continue comparing

- primitive actions,
- earlier learned abstractions,
- newly introduced abstractions,

within the same search process.

The goal is therefore not a sequence of isolated levels, but an expanding competence frontier.

---

## From Search Experience to Reusable Abstraction

A successful trajectory is not itself the abstraction.

ReWeave aims to recover the reusable structural relation behind useful search experience.

```text
successful exact trajectory
        │
        ▼
identify a valuable structural episode
        │
        ▼
semantic backtrace to a self-contained transformation
        │
        ▼
extract reusable parameters
        │
        ▼
construct S(θ) → T(θ)
        │
        ▼
exact certification
        │
        ▼
canonical semantic identity
        │
        ▼
persistent abstraction memory
```

The goal is not to memorize

```text
concrete state A → concrete state B
```

but to infer something closer to

```text
S(θ) → T(θ)
```

whose parameters can be rebound in new contexts.

A useful abstraction should therefore be

- **parameterized**, rather than instance-specific;
- **transferable**, across problems;
- **composable**, with future search;
- **persistent**, across learning rounds;
- **semantically identified**, rather than tied to one surface form;
- **exactly certified**, before it becomes trusted.

---

## Design Principles

### 1. Exactness lives outside the learned component

ReWeave deliberately separates **strategic judgment** from **mathematical correctness**.

```text
structural induction
        │
        ▼
propose reusable relation

exact certification
        │
        ▼
establish semantic validity

learned policy
        │
        ▼
decide when the relation looks useful

lowering + exact verification
        │
        ▼
protect the resulting computation
```

A relation enters the library only after exact certification.

At search time, a learned rewrite is lowered to the ordinary concrete circuit representation, applied there, and the resulting computation is checked exactly.

The learned policy ranks actions. It does not decide what is mathematically correct.

Cost is also measured on the ordinary lowered computation.

**Abstractions receive no artificial cost discount.**

Their value comes from changing representation, retrieval, and effective search horizon—not from changing the optimization objective.

---

### 2. Semantic abstractions, not a macro cache

ReWeave is not intended to memorize successful state transitions.

A memorized macro says:

> repeat these concrete actions again.

A reusable abstraction says:

> this semantic relation holds under a family of bindings.

Two manifestations of the same relation should not become different concepts merely because they use

- different variable names,
- different graph node IDs,
- different allocation order,
- different temporary handles,
- different but semantically equivalent surface representations.

Conceptually,

```text
many concrete manifestations
            │
            ▼
canonical semantic relation
            │
            ▼
one reusable abstraction family
```

Semantic identity matters increasingly as the library grows: otherwise abstraction memory fragments into superficially different copies of the same idea.

---

### 3. Learn from structure, reuse through handles

Higher-level structure is not always visible as a flat collection of primitive terms.

A useful operand may already exist as a compound intermediate computation.

ReWeave therefore treats abstraction parameters as semantic operands that can bind to values already materialized in the current search representation.

These may include

- monomial values,
- reachable compound DAG handles,
- retained semantic representatives,
- equivalent views of existing computation.

This motivates a simple principle:

> **Learn from structure; reuse through handles.**

Binding remains bounded. The matcher does not perform an unrestricted inner search for arbitrary hidden expressions.

Instead, abstractions reuse semantic units already available in the matcher-visible representation.

This allows the same relation to operate over progressively richer intermediate structure without making matching itself an open-ended synthesis problem.

---

### 4. Hidden and algebraically obscured structure

Useful structure may not appear in exactly the same form in which an abstraction was first learned.

An applicable relation may be

- nested inside a larger computation,
- multiplied by a shared outer factor,
- represented through an equivalent semantic form,
- attached to a compound intermediate,
- obscured by earlier decomposition choices.

ReWeave therefore separates an abstraction's semantic identity from any one syntactic presentation.

Mechanisms such as latent quotient exposure, semantic interning, normalization, equivalent views, and compound-handle binding serve the same broader goal:

> **Reuse should depend on what computation is present, not only on whether one preferred syntax happens to be visible.**

---

### 5. A persistent, open-ended abstraction memory

A self-improving optimizer needs more than a small fixed rewrite table.

ReWeave treats certified abstractions as a persistent memory that can grow as new structure is discovered.

```text
certified relation
        │
        ▼
canonical semantic family
        │
        ▼
persistent archive
        │
        ▼
indexed retrieval
        │
        ▼
context-specific matching
        │
        ▼
generic action representation
        │
        ▼
shared policy scoring
```

The architecture is therefore designed around semantic identity and retrieval rather than a permanently fixed vocabulary of rewrite IDs.

A newly discovered family should not fundamentally require

- redesigning the policy output layer,
- allocating a permanent learned embedding slot,
- changing model dimensionality,
- scanning every stored abstraction at every state.

Instead, candidate actions can be represented through their relation structure, parameter roles, bindings, context, and provenance.

This allows the abstraction vocabulary to grow independently of the basic policy interface.

---

### 6. Policy consolidation

Exact certification answers:

> **Is this relation valid?**

It does not answer:

> **When should search use it?**

Those are separate problems.

A newly certified family should not be assumed to receive a magically correct zero-shot policy score.

The intended loop therefore includes explicit policy consolidation:

```text
discover relation R
        │
        ▼
certify R
        │
        ▼
add R to the library
        │
        ▼
derive search / replay supervision
        │
        ▼
update the action-selection policy
        │
        ▼
search again
```

One natural source of supervision is **exact counterfactual replay**.

Given a historical search state and a newly available abstraction, the system can ask:

> What would have happened if this abstraction had been available or prioritized here?

Because the transformation and its successor remain exactly evaluable, policy learning can change strategic preferences without changing the semantics of the certified relation.

---

## What Generalization Means

A system that only memorizes successful training trajectories is not learning the kind of abstraction ReWeave is after.

ReWeave therefore aims for progressively stronger forms of transfer.

### 1. Instantiation transfer

A known abstraction should transfer to new concrete realizations:

```text
same abstraction
        +
new variables / bindings / graph realizations
```

This includes new parameter assignments, compound operands, and semantically equivalent representations.

This is the minimum requirement for distinguishing a parameterized relation from a memorized transition.

### 2. Structural transfer

The system should transfer beyond the exact structural patterns seen during learning:

```text
learned structural principle
        │
        ▼
new parameter regimes
new structural patterns
new abstraction families
new compositions
new structural roles
```

Relevant evaluations include held-out parameter or pattern regimes, leave-one-pattern-out (LOPO), leave-one-family-out (LOFO), unseen compositions, hidden structural roles, and independently generated out-of-distribution structures.

These tests ask whether induction, representation, and action scoring depend on transferable structure rather than fixed surface templates or family IDs.

### 3. Curriculum emergence

The strongest target goes beyond applying an existing abstraction to a held-out instance.

An abstraction may initially be absent from the library entirely.

Earlier abstractions can change search so that the missing abstraction becomes learnable later:

```text
reuse existing abstractions
        │
        ▼
reach new search experience
        │
        ▼
observe a new recurring relation
        │
        ▼
induce a previously absent abstraction
```

This asks whether the system can create the experience required to expand its own abstraction vocabulary.

---

## Evaluating Self-Improvement

ReWeave's central claims require separating several possible sources of improvement.

A stronger policy can improve optimization even with a fixed action language. A richer action library can improve search even without better action selection. A memorized macro can also make search faster without demonstrating reusable abstraction.

The evaluation should distinguish these explanations.

| System | Policy improves? | Library grows? | Main question |
|---|---:|---:|---|
| Primitive / no-library search | No | No | What can the base exact search do? |
| Policy-improvement only | Yes | No | How much comes from learning how to search? |
| Library-growth only | No | Yes | How much comes from expanding the action language? |
| Full ReWeave | Yes | Yes | Do the two learning loops reinforce one another? |

Additional controls can ask whether

- a **fixed learned library** is sufficient, or continued abstraction growth matters;
- a **one-shot library update** is sufficient, or repeated rounds add capability;
- **parameterized abstractions** transfer beyond what could be obtained by memorizing useful trajectory macros.

For external context, relevant conventional comparators include symbolic factorization, greedy algebraic factorization with common-subexpression elimination, equality saturation with extraction, and multivariate Horner/CSE optimization.

The purpose of these comparisons is not simply to rank optimizers on one benchmark.

It is to isolate what ReWeave gains from

\[
\text{better search},
\qquad
\text{better abstractions},
\qquad
\text{and their interaction}.
\]

Together with the generalization ladder above, this gives a progression from ordinary transfer to the stronger question ReWeave ultimately cares about:

```text
new bindings
    ↓
held-out patterns / parameter regimes
    ↓
held-out abstraction families
    ↓
unseen compositions and structural roles
    ↓
generator-shift OOD structure
    ↓
new abstractions emerging through the curriculum
```

---

## The Current Testbed

ReWeave is developed on **exact compression of monotone arithmetic circuits**.

A task is a polynomial with positive integer coefficients. A state is an exact, hash-consed ADD/MUL circuit DAG that computes it, and the primary cost is the number of unique reachable ADD/MUL gates.

Primitive search operates through exact algebraic transformations such as factoring and reuse.

The domain is small enough that every deployed transformation can be verified exactly, while still containing recurring, compositional, and hidden structure.

This makes it a useful testbed for studying the broader question:

> Can search experience become reusable optimization knowledge without giving up exactness?

---

## Minimal Example

[`examples/minimal_example.py`](examples/minimal_example.py) runs one turn of the abstraction loop using the real implementation.

Nothing in the example is mocked.

### 1. Primitive search generates experience

```text
SEARCH: primitive-only exact search on training polynomials

task 0: 6 gates -> 2 gates
...
```

### 2. ReWeave induces and certifies relations

Trajectory windows are converted into structural pairs, parameterized, and certified exactly.

The example recovers a Square relation from primitive-search trajectories alone:

```text
2*A*B + A^2 + B^2  ->  ((A + B)^2)
```

The induced Square is byte-identical to the corresponding relation in the shipped reference library.

### 3. The relation transfers to unseen realizations

The same learned relation is reused in several settings:

```text
visible   (d+g)^2 + eh          :  8 -> 4 gates
latent    f(a+c)^2 + b          : 10 -> 4 gates
compound  H^2+2HK+K^2           :  8 -> 4 gates
```

The latent case has no visible match until the common multiplier is exposed.

The compound case binds the relation over already materialized compound sub-expressions rather than only monomial leaves.

Every successor is lowered and exact-verified.

### 4. Search runs again with the enlarged language

Under the same small search budget:

```text
                                  primitive    with library
f(ab+c)^2 + (d+e)^2                  13              7
c(a+b)^2 + d(a+b)^2                  10              4
g(c+d)^2 + h(c+d)^2 + e              13              5
```

The example demonstrates the core causal step:

```text
primitive search
    → structural experience
    → induced certified abstraction
    → reuse on new structures
    → changed future search
```

It is intended as a compact executable example of the mechanism, not as a benchmark result.

---

## Quick Start

Requirements: Python 3.10 or newer. The public release has been tested with Python 3.11.

```bash
git clone https://github.com/KJ-Jin/reweave-public.git
cd reweave-public

python -m venv .venv
source .venv/bin/activate

pip install -e ".[test]"
```

Run the minimal example:

```bash
python examples/minimal_example.py
```

Run the test suite:

```bash
pytest
```

ReWeave uses the vendored exact runtime under `legacy_runtime/`; the package loads it through the repository-local pin.

---

## Repository Layout

```text
reweave/
    rewrite_schema.py          parameterized relations, canonicalization, lowering
    rewrite_extract.py         search trajectories -> structural episodes
    rewrite_propose.py         structural pairs -> candidate relations
    rewrite_certify.py         exact relation certification
    rewrite_library.py         support, eligibility, library construction
    rewrite_match.py           matching, application, lowering, verification
    rewrite_latent.py          latent quotient exposure
    rewrite_search_latent.py   search over primitive and learned actions
    candidate_policy.py        candidate admission
    rewrite_model.py           learned scorer
    recipe.py                  model/training configuration

    canon/
        relation.py            semantic relation IR and family identity
        archive.py             certified archive and retrieval
        exprparam.py           generic parameter binding
        action_record.py       schema-agnostic action representation
        candidate_set.py       candidate-set completeness/provenance
        semantic.py            semantic interning and normalization
        views.py               equivalent semantic views
        identity*.py           symmetry-safe identities
        latent_identity.py     latent action identity
        action_features.py     structural action features
        action_model.py        schema-agnostic scorer model
        action_policy.py       action scoring / policy interface
        search.py              canonical search loop
        config.py              explicit architecture switches

    data/
        reference_library.json
                               certified reference relations

legacy_runtime/
    polycompress/
                               vendored exact polynomial/circuit runtime

examples/
    minimal_example.py         one executable turn of abstraction learning and reuse

tests/
                               exactness, matching, binding, identity,
                               archive, scoring and example tests

docs/
    method.md                  mechanism in detail
    architecture.md            system architecture and component interfaces
```

---

## System View

At a high level, ReWeave connects four persistent components.

### Exact Search

Explores optimization states using primitive and learned actions.

### Policy Learning

Learns which available actions are useful in which contexts.

### Structural Induction

Extracts parameterized semantic relations from successful search experience.

### Certified Abstraction Memory

Stores reusable exact relations and makes them available to future search.

Together:

```text
                     ┌─────────────────────┐
                     │ Certified Library   │
                     │        L_t          │
                     └─────────┬───────────┘
                               │
                               ▼
┌───────────────┐       ┌───────────────┐
│ Policy π_t    │──────▶│  Exact Search │
└───────────────┘       └───────┬───────┘
                                │
                                ▼
                       Search Experience
                                │
                     ┌──────────┴──────────┐
                     │                     │
                     ▼                     ▼
              Policy Learning       Structural Induction
                     │                     │
                     ▼                     ▼
                  π_{t+1}           Exact Certification
                                           │
                                           ▼
                                        L_{t+1}
```

Search discovers.

Induction proposes structure.

Certification establishes validity.

Policy learning acquires strategic preference.

The library preserves reusable concepts.

---

## Research Perspective

ReWeave is motivated by a broader question:

> **Can exact search progressively construct the concepts that make its own future search easier?**

If so, an optimizer need not remain permanently tied to the primitive transformation language chosen by its designer.

Successful experience can become reusable structure.

Reusable structure can change future search.

Changed search can create new experience.

That experience can support further abstraction.

The resulting process is not simply

\[
\text{learn a better policy},
\]

and not simply

\[
\text{learn a rewrite library}.
\]

It is

\[
\boxed{
\text{learn how to search}
\quad+\quad
\text{learn what to search with}
}
\]

inside an exactness boundary that remains outside the learned components.

That is the system ReWeave is designed to build.

---

## Further Reading

- [`docs/method.md`](docs/method.md) — state representation, primitive search, structural episodes, relation induction, certification, matching, lowering, and reuse.
- [`docs/architecture.md`](docs/architecture.md) — semantic identity, certified memory, retrieval, generic parameter binding, action representation, scoring, search, induction, and the intended iterative loop.
- [`legacy_runtime/README.md`](legacy_runtime/README.md) — the vendored exact polynomial/circuit runtime.

## License

MIT. See [`LICENSE`](LICENSE).
