"""The single adapter through which ReWeave imports the inherited exact runtime (`legacy_runtime/polycompress`).

The runtime is a pinned, vendored snapshot: exact polynomial arithmetic, the interned DAG and its reachable-gate
cost, the primitive FACTOR/REUSE calculus, the exact verifier and beam search. It is imported from the vendored
directory (or from `POLYC_V1_IMPORT_PATH`), and the `.v1_commit` marker must name the pinned commit, so the adapter
never silently follows a different copy. Every inherited name ReWeave uses is bound here.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
V1_CODE_COMMIT = 'd1343a1ceb9b66933b7a2db1f5bdd57441497738'


def resolution() -> dict:
    return dict(V1_CODE_COMMIT=V1_CODE_COMMIT, V1_IMPORT_PATH=str(REPO/'legacy_runtime'))


def import_path() -> Path:
    override = os.environ.get('POLYC_V1_IMPORT_PATH')
    if override:
        return Path(override)
    return Path(resolution()['V1_IMPORT_PATH'])


_path = str(import_path())
if _path not in sys.path:
    sys.path.insert(0, _path)
_marker = Path(_path)/'.v1_commit'
if not _marker.is_file() or _marker.read_text().strip() != V1_CODE_COMMIT:
    raise ImportError(f'V1 import path {_path} is not the pinned export of {V1_CODE_COMMIT}')

import polycompress  # noqa: E402

if Path(polycompress.__file__).resolve().parent.parent != Path(_path).resolve():
    raise ImportError(f'polycompress resolved to {polycompress.__file__}, not the pinned export {_path}')

from polycompress import poly as poly_mod  # noqa: E402
from polycompress import handles as handles_mod  # noqa: E402
from polycompress import state as state_mod  # noqa: E402
from polycompress import calculus as calculus_mod  # noqa: E402
from polycompress import enumerate as enumerate_mod  # noqa: E402
from polycompress import search as search_mod  # noqa: E402
from polycompress import generator as generator_mod  # noqa: E402
from polycompress import heuristics as heuristics_mod  # noqa: E402
from polycompress import stats as stats_mod  # noqa: E402
from polycompress import model as model_mod  # noqa: E402
from polycompress import train as train_mod  # noqa: E402
from polycompress import config as config_mod  # noqa: E402
from polycompress import protocol as protocol_mod  # noqa: E402

# poly: exact positive polynomial arithmetic, exact equality, canon under renaming
Poly = poly_mod.Poly
pack, unpack, canon, degree, support = poly_mod.pack, poly_mod.unpack, poly_mod.canon, poly_mod.degree, poly_mod.support
mono_degree, mono_support, poly_equal = poly_mod.mono_degree, poly_mod.mono_support, poly_mod.equal
poly_add, poly_mul, poly_scale, divides = poly_mod.add, poly_mod.mul, poly_mod.scale, poly_mod.divides
FP_PRIME = poly_mod.P

# handles: interned DAG, exact reachable gate cost, structural digest/serialization
Arena, Mono, Sum, Prod = handles_mod.Arena, handles_mod.Mono, handles_mod.Sum, handles_mod.Prod
mk_mono, mk_sum, mk_prod = handles_mod.mk_mono, handles_mod.mk_sum, handles_mod.mk_prod
items, digest, cost, standard_cost = handles_mod.items, handles_mod.digest, handles_mod.cost, handles_mod.standard_cost
hpoly, canonical_start, reachable = handles_mod.poly, handles_mod.canonical_start, handles_mod.reachable
serialize, deserialize, MONO_COST = handles_mod.serialize, handles_mod.deserialize, handles_mod.MONO_COST
Gate = handles_mod.Gate

# state: exposed context / reachability info and exact-equal handle matching
StateInfo, state_info, state_match = state_mod.StateInfo, state_mod.state_info, state_mod.match

# calculus: ordinary FACTOR/REUSE semantics and exact successor construction
Task, Action = calculus_mod.Task, calculus_mod.Action
action_key, apply_action, verify, substitute = (calculus_mod.action_key, calculus_mod.apply,
                                                calculus_mod.verify, calculus_mod.substitute)
action_record, action_from_record, region_key = (calculus_mod.action_record, calculus_mod.action_from_record,
                                                 calculus_mod.region_key)

# enumerate: deterministic primitive action enumeration/truncation
enumerate_actions = enumerate_mod.enumerate_actions

# search: beam/visited/incumbent semantics, budgets
Budget, Stats, beam_search, peak_rss_mb = search_mod.Budget, search_mod.Stats, search_mod.beam_search, search_mod.peak_rss_mb

# generator: F1-F4 constructors, grid, size filters, composition/discovery helpers, V1 set emitters
generator = generator_mod

# heuristics: ImmediateGain, CostRanker, ValueRanker, NeuralScorer
ImmediateGain, CostRanker = heuristics_mod.ImmediateGain, heuristics_mod.CostRanker
NeuralScorer, ValueRanker = heuristics_mod.NeuralScorer, heuristics_mod.ValueRanker

# stats: seed-aware paired bootstrap
Paired, paired_ci = stats_mod.Paired, stats_mod.paired_ci

# model / train: V1 encoder, batching, listwise loss, record format, keep-best lineage
PolyModel, new_model, make_batch, forward, listwise_loss = (model_mod.PolyModel, model_mod.new_model,
                                                            model_mod.make_batch, model_mod.forward,
                                                            model_mod.listwise_loss)
pool_rows, node_features, off_graph_mono, build_graph = (model_mod.pool, model_mod.node_features,
                                                         model_mod.off_graph_mono, model_mod.build_graph)
extract_record, rebuild_record, write_records, read_records = (train_mod.extract, train_mod.rebuild,
                                                               train_mod.write_records, train_mod.read_records)
scan_records, read_record_at, index_records, split_index = (train_mod.scan_records, train_mod.read_record_at,
                                                            train_mod.index_records, train_mod.split_index)
state_tuples, keep_best, merge_records, incumbent_cost = (train_mod.state_tuples, train_mod.keep_best,
                                                          train_mod.merge_records, train_mod.incumbent_cost)
train_tasks, configure_determinism, V1_BATCH, V1_WINDOW = (train_mod.train_tasks, train_mod.configure_determinism,
                                                           train_mod.BATCH, train_mod.WINDOW)

# config / protocol: frozen defaults, split rule, source-tree hash
configs, modelval, source_tree_hash = protocol_mod._configs, protocol_mod.modelval, protocol_mod.source_tree_hash
GenCfg, TrainCfg, CalcCfg, SearchCfg = config_mod.GenCfg, config_mod.TrainCfg, config_mod.CalcCfg, config_mod.SearchCfg
StatsCfg = config_mod.StatsCfg
addchain = handles_mod.addchain
teacher_mine = train_mod.mine
exposed_regions = enumerate_mod.regions

# V1 generator attributes V2 calls (constructors, grid, size filters, V1 set emitters).
GENERATOR_ATTRS = ('ATOMIC', 'DISC', 'FAMILY_GRID', 'WIDTH', '_easy_start', '_family', '_hidden_task', '_slot',
                   'benchmark_split', 'build_choice', 'build_composition', 'build_instance', 'composition_set',
                   'discovery_instance', 'discovery_set', 'enumerate_cell', 'n_slots', 'random_ood_set',
                   'reserve_shapes', 'seed', 'shape_table', 'size_rejection', 'skeleton_signature')

MODULES = dict(poly=poly_mod, handles=handles_mod, state=state_mod, calculus=calculus_mod,
               enumerate=enumerate_mod, search=search_mod, generator=generator_mod, heuristics=heuristics_mod,
               stats=stats_mod, model=model_mod, train=train_mod, config=config_mod, protocol=protocol_mod)

# (V2 alias, V1 module, V1 attribute) for every inherited symbol bound above.
SYMBOLS = [
    ('Poly', 'poly', 'Poly'), ('pack', 'poly', 'pack'), ('unpack', 'poly', 'unpack'), ('canon', 'poly', 'canon'),
    ('degree', 'poly', 'degree'), ('support', 'poly', 'support'), ('mono_degree', 'poly', 'mono_degree'),
    ('mono_support', 'poly', 'mono_support'), ('poly_equal', 'poly', 'equal'), ('poly_add', 'poly', 'add'),
    ('poly_mul', 'poly', 'mul'), ('poly_scale', 'poly', 'scale'), ('divides', 'poly', 'divides'),
    ('FP_PRIME', 'poly', 'P'),
    ('Arena', 'handles', 'Arena'), ('Mono', 'handles', 'Mono'), ('Sum', 'handles', 'Sum'), ('Prod', 'handles', 'Prod'),
    ('mk_mono', 'handles', 'mk_mono'), ('mk_sum', 'handles', 'mk_sum'), ('mk_prod', 'handles', 'mk_prod'),
    ('items', 'handles', 'items'), ('digest', 'handles', 'digest'), ('cost', 'handles', 'cost'),
    ('standard_cost', 'handles', 'standard_cost'), ('hpoly', 'handles', 'poly'),
    ('canonical_start', 'handles', 'canonical_start'), ('reachable', 'handles', 'reachable'),
    ('serialize', 'handles', 'serialize'), ('deserialize', 'handles', 'deserialize'),
    ('MONO_COST', 'handles', 'MONO_COST'), ('Gate', 'handles', 'Gate'),
    ('StateInfo', 'state', 'StateInfo'), ('state_info', 'state', 'state_info'), ('state_match', 'state', 'match'),
    ('Task', 'calculus', 'Task'), ('Action', 'calculus', 'Action'), ('action_key', 'calculus', 'action_key'),
    ('apply_action', 'calculus', 'apply'), ('verify', 'calculus', 'verify'), ('substitute', 'calculus', 'substitute'),
    ('action_record', 'calculus', 'action_record'), ('action_from_record', 'calculus', 'action_from_record'),
    ('region_key', 'calculus', 'region_key'),
    ('enumerate_actions', 'enumerate', 'enumerate_actions'),
    ('Budget', 'search', 'Budget'), ('Stats', 'search', 'Stats'), ('beam_search', 'search', 'beam_search'),
    ('peak_rss_mb', 'search', 'peak_rss_mb'),
    ('generator', 'generator', None),
    ('ImmediateGain', 'heuristics', 'ImmediateGain'), ('CostRanker', 'heuristics', 'CostRanker'),
    ('NeuralScorer', 'heuristics', 'NeuralScorer'), ('ValueRanker', 'heuristics', 'ValueRanker'),
    ('Paired', 'stats', 'Paired'), ('paired_ci', 'stats', 'paired_ci'),
    ('PolyModel', 'model', 'PolyModel'), ('new_model', 'model', 'new_model'), ('make_batch', 'model', 'make_batch'),
    ('forward', 'model', 'forward'), ('listwise_loss', 'model', 'listwise_loss'), ('pool_rows', 'model', 'pool'),
    ('node_features', 'model', 'node_features'), ('off_graph_mono', 'model', 'off_graph_mono'),
    ('build_graph', 'model', 'build_graph'),
    ('extract_record', 'train', 'extract'), ('rebuild_record', 'train', 'rebuild'),
    ('write_records', 'train', 'write_records'), ('read_records', 'train', 'read_records'),
    ('scan_records', 'train', 'scan_records'), ('read_record_at', 'train', 'read_record_at'),
    ('index_records', 'train', 'index_records'), ('split_index', 'train', 'split_index'),
    ('state_tuples', 'train', 'state_tuples'), ('keep_best', 'train', 'keep_best'),
    ('merge_records', 'train', 'merge_records'), ('incumbent_cost', 'train', 'incumbent_cost'),
    ('train_tasks', 'train', 'train_tasks'), ('configure_determinism', 'train', 'configure_determinism'),
    ('V1_BATCH', 'train', 'BATCH'), ('V1_WINDOW', 'train', 'WINDOW'),
    ('configs', 'protocol', '_configs'), ('modelval', 'protocol', 'modelval'),
    ('source_tree_hash', 'protocol', 'source_tree_hash'),
    ('GenCfg', 'config', 'GenCfg'), ('TrainCfg', 'config', 'TrainCfg'), ('CalcCfg', 'config', 'CalcCfg'),
    ('SearchCfg', 'config', 'SearchCfg'), ('StatsCfg', 'config', 'StatsCfg'), ('addchain', 'handles', 'addchain'),
    ('teacher_mine', 'train', 'mine'), ('exposed_regions', 'enumerate', 'regions'),
]+[(f'generator.{name}', 'generator', name) for name in GENERATOR_ATTRS]
