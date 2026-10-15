"""The README's minimal example runs end to end: induction recovers the Square relation from primitive search,
every reuse mode is exact, and search with the induced library is never worse than primitive-only search."""

import importlib.util
import re
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parent.parent/'examples'/'minimal_example.py'
SQUARE = '3cac496ef8bd6e7bd9eac9103429ab4a9f741635462e844c6731b1c74cadfbea'   # A^2 + 2AB + B^2 -> (A + B)^2


def test_minimal_example_runs(capsys):
    spec = importlib.util.spec_from_file_location('minimal_example', EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
    out = capsys.readouterr().out
    assert '->  ((A + B)^2)' in out
    assert out.count('[ok]') == 3
    assert 'independent of arena allocation: True' in out
    rows = re.findall(r'primitive\s+(\d+)\s+with library\s+(\d+).*failures (\d+)', out)
    assert len(rows) == 3
    for prim, lib, failures in rows:
        assert int(lib) <= int(prim) and failures == '0'


def test_induction_recovers_the_reference_square():
    """The relation induced from scratch is byte-identical (same LEGACY_SURFACE_ID) to the reference library's."""
    from reweave.current_config import CURRENT
    assert SQUARE in {s.schema_hash for s in CURRENT.library.schemas()}
    spec = importlib.util.spec_from_file_location('minimal_example', EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import dataclasses
    from reweave import v1
    from reweave.rewrite_config import RewriteConfig
    from reweave.rewrite_library import build_library
    items = []
    for i, target in enumerate([mod.square('a', 'b'), mod.square('c*d', 'e'), mod.square('a*b', 'c', ('g', 1)),
                                mod.square('d', 'e*f', ('h*g', 1)), mod.square('b', 'g'),
                                mod.square('d*h', 'a', ('e*e', 1))]):
        task = v1.Task(target, 8, 0, dict(id=i))
        arena, start = mod.flat_start(target)
        best, _, tree = v1.beam_search(arena, task, start, v1.ImmediateGain(task), v1.CostRanker(),
                                       v1.Budget(8, 8, 20, 400), True)
        items.append((v1.extract_record(arena, task, start, best, tree), dict(family='square', shape=f's{i % 3}')))
    cfg = dataclasses.replace(RewriteConfig(), task_support_min=3, shape_support_min=2)
    library, _ = build_library(items, cfg, 'test')
    assert SQUARE in library['eligible']
