"""E7 exact-state manifest: serialized states and actions, per-set semantic hashes, loading and round-trip checks."""

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from ..calculus import Action, Task, action_from_record, action_record, apply
from ..config import CalcCfg
from ..enumerate import enumerate_actions
from ..handles import Arena, cost, deserialize, digest, poly, serialize
from ..state import state_info
from .. import generator as g


def _state(arena: Arena, roots: dict[str,int], actions: dict[str,Action]) -> dict:
    nodes,index = serialize(arena,list(roots.values())+[a.arg for a in actions.values() if a.arg >= 0])
    return dict(nodes=nodes,roots={name:index[h] for name,h in roots.items()},
                digests={name:digest(arena,h).hex() for name,h in roots.items()},
                actions={name:action_record(a,index) for name,a in actions.items()})


def e7a_entry(pair: list[tuple], calc: CalcCfg) -> dict:
    """Freeze one E7a pair: both exposed states, `a*`, `a_alt`, their successors and the exact margins."""
    useful,sham = g.e7a_margins(pair,calc)
    states = []
    for arena,task,root,designated in pair:
        info = state_info(arena,root)
        named = dict(a_star=designated,a_alt=task.meta['alternative_action'])
        after = {f'after_{name}':apply(arena,task,info,root,a) for name,a in named.items()}
        assert None not in after.values(), 'both designated E7a actions are legal'
        entry = _state(arena,dict(root=root,**after),named)
        entry.update(useful=task.meta['useful'],target=sorted(task.target.items()),n_vars=task.n_vars,
                     c_gen=task.c_gen,cost=cost(arena,root))
        states.append(entry)
    return dict(id=pair[0][1].meta['id'],margins=dict(useful=useful,sham=sham),states=states)


def e7b_entry(kept: tuple) -> dict:
    """Freeze one accepted E7b pair: start, useful and sham states with the actions that produce them."""
    arena,task,start,motif,x,xu,useful,sham,useful_action,sham_action = kept
    entry = _state(arena,dict(start=start,useful=useful,sham=sham),dict(useful=useful_action,sham=sham_action))
    entry.update(id=task.meta['id'],embedded_family=task.meta['embedded_family'],d=task.meta['params'][0],
                 x_index=task.meta['x_index'],slot_shapes=list(task.meta['slot_shapes']),target=sorted(task.target.items()),
                 n_vars=task.n_vars,c_gen=task.c_gen,status=task.meta['status'],
                 costs=dict(start=cost(arena,start),useful=cost(arena,useful),sham=cost(arena,sham)))
    return entry


def semantic_hash(value: object) -> str:
    """Hash canonical JSON of a semantic object (no timing or memory fields are ever included)."""
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write_manifest(sets: dict[str,list], statuses: list[dict], calc: CalcCfg, output: Path) -> dict[str,str]:
    """Write the exact-state E7 manifest and return its per-set and whole-file SHA-256 hashes."""
    e7a = [e7a_entry(pair,calc) for pair in sets['e7a']]
    e7b = [e7b_entry(entry) for entry in sets['e7b']]
    grid = [dict(s,slot_shapes=list(s['slot_shapes'])) for s in statuses]
    hashes = dict(e7a=semantic_hash(e7a),e7b=semantic_hash(e7b),e7b_grid=semantic_hash(grid))
    manifest = dict(calc=asdict(calc),e7a=e7a,e7b=e7b,e7b_grid_statuses=grid,hashes=hashes)
    text = json.dumps(manifest,sort_keys=True,indent=1)+'\n'
    output.mkdir(parents=True,exist_ok=True)
    (output/'e7_manifest.json').write_text(text)
    hashes = dict(hashes,manifest=hashlib.sha256(text.encode()).hexdigest())
    (output/'e7_manifest.sha256').write_text(''.join(f'{name} {value}\n' for name,value in sorted(hashes.items())))
    return hashes


def load_manifest(output: Path) -> dict:
    """Load the manifest and refuse it when any recorded semantic or file hash disagrees."""
    path = output/'e7_manifest.json'
    if not path.is_file() or not (output/'e7_manifest.sha256').is_file():
        raise ValueError(f'E7 manifest missing under {output}')
    text = path.read_text()
    manifest = json.loads(text)
    recorded = dict(line.split() for line in (output/'e7_manifest.sha256').read_text().splitlines())
    actual = dict(e7a=semantic_hash(manifest['e7a']),e7b=semantic_hash(manifest['e7b']),
                  e7b_grid=semantic_hash(manifest['e7b_grid_statuses']),manifest=hashlib.sha256(text.encode()).hexdigest())
    if recorded != actual or {k:v for k,v in actual.items() if k != 'manifest'} != manifest['hashes']:
        raise ValueError(f'E7 manifest hashes disagree: recorded {recorded}, actual {actual}')
    return manifest


def rebuild(entry: dict) -> tuple[Arena,dict[str,int],dict[str,Action]]:
    """Rebuild a frozen state entry in a fresh arena: named roots and named actions."""
    arena = Arena()
    ids = deserialize(arena,entry['nodes'])
    roots = {name:ids[i] for name,i in entry['roots'].items()}
    return arena,roots,{name:action_from_record(record,ids) for name,record in entry['actions'].items()}


def verify_manifest(manifest: dict, calc: CalcCfg) -> dict[str,int]:
    """Round-trip every frozen state: target, cost, digest, candidate membership and recorded successors."""
    counts: Counter[str] = Counter()
    for pair in manifest['e7a']:
        for state in pair['states']:
            arena,roots,actions = rebuild(state)
            root,target = roots['root'],{m:c for m,c in state['target']}
            task = Task(target,state['n_vars'],state['c_gen'],{})
            assert poly(arena,root) == target and cost(arena,root) == state['cost'], 'E7a state reproduces'
            info = state_info(arena,root)
            candidates = enumerate_actions(arena,info,root,calc)
            for name in ('a_star','a_alt'):
                assert actions[name] in candidates, f'E7a {name} is a frozen candidate'
                successor = apply(arena,task,info,root,actions[name])
                assert successor is not None and digest(arena,successor).hex() == state['digests'][f'after_{name}']
            counts['e7a_states'] += 1
    for entry in manifest['e7b']:
        arena,roots,actions = rebuild(entry)
        target = {m:c for m,c in entry['target']}
        task = Task(target,entry['n_vars'],entry['c_gen'],{})
        for name,root in roots.items():
            assert poly(arena,root) == target and cost(arena,root) == entry['costs'][name], f'E7b {name} reproduces'
            assert digest(arena,root).hex() == entry['digests'][name]
        info = state_info(arena,roots['start'])
        for name in ('useful','sham'):
            assert apply(arena,task,info,roots['start'],actions[name]) == roots[name], f'E7b {name} action reproduces its state'
        counts['e7b_pairs'] += 1
    return dict(counts)
