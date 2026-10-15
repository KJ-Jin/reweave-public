"""Post-search export of deterministic shortest reference trajectories (spec section 17.A)."""

import hashlib
import json
from pathlib import Path

from ..calculus import Action, Task, action_key, action_record
from ..handles import Arena, cost, digest, serialize
from .. import poly as p

STEM = 'reference_trajectories'


def action_stable_key(arena: Arena, a: Action) -> str:
    """Hash the allocation-independent canonical action key into a stable hex string."""
    return hashlib.sha256(repr(action_key(arena,a)).encode()).hexdigest()


def target_key(target: p.Poly) -> str:
    """Hash an exact polynomial by its sorted packed terms."""
    return hashlib.sha256(repr(sorted(target.items())).encode()).hexdigest()


def trajectory_record(arena: Arena, task: Task, trajectory: dict, targets: list[p.Poly],
                      cell: dict) -> dict:
    """Serialize one accepted reference trajectory; the timed search has already terminated."""
    assert not p.TIMED_EXHAUSTIVE_ACTIVE, 'trajectory export runs strictly after the timed search'
    assert trajectory['covered'] and trajectory['reached'], 'only accepted references are exported'
    roots,actions = trajectory['roots'],trajectory['actions']
    args = [a.arg for a in actions if a.arg >= 0]
    nodes,index = serialize(arena,list(dict.fromkeys(roots))+args)
    path = [dict(step=step,state=index[root],state_digest=digest(arena,root).hex(),state_c_sf=cost(arena,root),
                 action=action_record(a,index),action_key_sha256=action_stable_key(arena,a),
                 next_state=index[roots[step+1]],next_digest=digest(arena,roots[step+1]).hex(),
                 next_c_sf=cost(arena,roots[step+1]))
            for step,(root,a) in enumerate(zip(roots,actions))]
    return dict(task_id=task.meta.get('id'),cell_id=cell['cell_id'],set=cell['set'],family=cell['family'],
                params=list(cell['params']),slot_mix=list(cell['slot_mix']),n_vars=task.n_vars,
                occurrences=sum(task.target.values()),distinct=len(task.target),
                start_cost=cost(arena,roots[0]),c_gen=task.c_gen,exhaustive_status='accepted',
                shortest_depth=len(actions),target=sorted(task.target.items()),
                target_sha256=target_key(task.target),nodes=nodes,path=path,
                generator_nodes=[sorted(q.items()) for q in targets])


def write_trajectories(records: list[dict], output: Path, shard: str) -> Path:
    """Write one shard's accepted reference trajectories as deterministic JSON lines."""
    output.mkdir(parents=True,exist_ok=True)
    path = output/f"{STEM}_shard_{shard.split('/')[0]}.jsonl"
    with path.open('w') as stream:
        for record in records:
            stream.write(json.dumps(record,sort_keys=True)+'\n')
    return path


def merge_trajectories(directory: Path, n_shards: int) -> Path:
    """Concatenate the per-shard trajectory files in shard order into one JSON-lines artifact."""
    if n_shards < 1:
        raise ValueError('n_shards must be positive')
    destination = directory/f'{STEM}.jsonl'
    with destination.open('w') as stream:
        for i in range(n_shards):
            path = directory/f'{STEM}_shard_{i}.jsonl'
            if not path.is_file():
                raise ValueError(f'missing trajectory shard: {path}')
            stream.write(path.read_text())
    return destination


def read_trajectories(path: Path) -> list[dict]:
    """Read every exported trajectory record in file order."""
    if not path.is_file():
        raise ValueError(f'reference trajectories missing: {path}')
    return [json.loads(line) for line in path.read_text().splitlines() if line]
