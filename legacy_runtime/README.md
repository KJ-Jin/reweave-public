# legacy_runtime — the vendored exact runtime

ReWeave does not re-implement the exact substrate it optimizes over. It builds on `polycompress`, an earlier
project by the same author, vendored here at a pinned commit (`.v1_commit`):

| module | provides |
|---|---|
| `poly.py` | exact positive-integer polynomial arithmetic, exact equality, canonical form under variable renaming |
| `handles.py` | the interned `Mono` / `Sum` / `Prod` DAG, structural digests, lowering to binary gates, gate cost |
| `state.py` | exposed contexts and reachability of a state |
| `calculus.py`, `enumerate.py` | the primitive FACTOR / REUSE actions and their deterministic enumeration |
| `search.py` | budgeted, deterministic beam search |
| `heuristics.py` | the `ImmediateGain` scorer and `CostRanker` used for round-0 search |
| `model.py`, `train.py` | the message-passing state encoder and its listwise training |
| `generator.py` | the benchmark task families |
| `config.py`, `protocol.py`, `stats.py` | default configuration, configuration identity, paired bootstrap statistics |

ReWeave imports all of it through exactly one adapter, [`reweave/v1.py`](../reweave/v1.py). The adapter loads the
package from this directory, or from `POLYC_V1_IMPORT_PATH` if that is set. It refuses to import a copy whose
`.v1_commit` marker does not name the pinned commit, and refuses a `polycompress` package that resolves anywhere
else.

This snapshot contains the runtime package only. The original project's experiment launchers, baselines that need
external tools, and its own documentation are not included. The code is otherwise unchanged and is covered by the
repository's MIT license.
