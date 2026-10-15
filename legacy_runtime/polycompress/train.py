"""Teacher trees to compact per-task records, streaming shard files, and the serialization smoke test."""

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import random
import struct
import time
from typing import Iterable, Iterator

import numpy as np

from .calculus import Task, action_record, action_from_record
from .config import TrainCfg
from .handles import Arena, canonical_start, cost, deserialize, digest, serialize
from .heuristics import ImmediateGain, CostRanker, HandLinear, features, standardize
from .protocol import modelval
from .search import Budget, beam_search, iter_tasks, peak_rss_mb, sized_compute
from .state import state_info

FRAME = struct.Struct('<Q')
DATASET = 'dataset.bin'
BATCH = 32                  # section 8.4: batches of 32 states
# Records held while one shuffle window is formed. The window bounds memory only; batch
# composition stays a deterministic function of the training seed either way.
WINDOW = 64


def extract(arena: Arena, task: Task, start: int, incumbent: int, tree: dict) -> dict:
    """Turn a recorded search tree into the section 8.2 record: shared subgraph, on-path labels, value targets."""
    chain,h = [],incumbent
    while h is not None:
        chain.append(h)
        h = tree[h]['parent']
    chain.reverse()
    depth = {}
    for h in tree:
        depth[h] = 0 if tree[h]['parent'] is None else depth[tree[h]['parent']]+1 if tree[h]['parent'] in depth else -1
    for h in sorted(tree,key=lambda h:len(_ancestors(tree,h))):
        depth[h] = len(_ancestors(tree,h))
    subtree = {h:tree[h]['cost'] for h in tree}
    for h in sorted(tree,key=lambda h:-depth[h]):
        if tree[h]['parent'] is not None:
            subtree[tree[h]['parent']] = min(subtree[tree[h]['parent']],subtree[h])
    labelled = [(s,tree[s]['actions'],tree[s]['actions'].index(tree[nxt]['action'])) for s,nxt in zip(chain,chain[1:])]
    args = [a.arg for s,actions,i in labelled for a in actions if a.arg >= 0]
    nodes,index = serialize(arena,list(tree)+args,with_digests=False)
    states = [dict(root=index[h],cost=tree[h]['cost'],value=subtree[h]/cost(arena,start),expanded='actions' in tree[h],
                   parent=None if tree[h]['parent'] is None else index[tree[h]['parent']]) for h in tree]
    return dict(task_id=task.meta.get('id'),target=sorted(task.target.items()),n_vars=task.n_vars,c_gen=task.c_gen,
                start=index[start],incumbent=index[incumbent],nodes=nodes,states=states,
                labelled=[dict(root=index[s],actions=[action_record(a,index) for a in actions],positive=i) for s,actions,i in labelled])


def _ancestors(tree: dict, h: int) -> list[int]:
    out = []
    while tree[h]['parent'] is not None:
        h = tree[h]['parent']
        out.append(h)
    return out


def rebuild(record: dict) -> tuple[Arena,list[int]]:
    """Rebuild a record's shared subgraph in a fresh arena and return the handle ids by position."""
    arena = Arena()
    return arena,deserialize(arena,record['nodes'])


def labelled_actions(record: dict, ids: list[int]) -> list[tuple[int,list,int]]:
    """Decode every labelled state as (root id, candidate actions, positive index)."""
    return [(ids[s['root']],[action_from_record(r,ids) for r in s['actions']],s['positive']) for s in record['labelled']]


def write_records(path: Path, records: Iterable[dict]) -> dict:
    """Stream length-framed pickles to one shard file, one record at a time; return count and bytes."""
    path.parent.mkdir(parents=True,exist_ok=True)
    count,total = 0,0
    with path.open('wb') as stream:
        for record in records:
            blob = pickle.dumps(record,protocol=pickle.HIGHEST_PROTOCOL)
            stream.write(FRAME.pack(len(blob)))
            stream.write(blob)
            count,total = count+1,total+len(blob)+FRAME.size
    return dict(records=count,bytes=total)


def scan_records(path: Path) -> Iterator[tuple[int,dict]]:
    """Yield every record with its file offset, so a later pass can re-read exactly that record."""
    with path.open('rb') as stream:
        while True:
            offset = stream.tell()
            head = stream.read(FRAME.size)
            if not head:
                return
            if len(head) != FRAME.size:
                raise ValueError(f'truncated record frame in {path}')
            yield offset,pickle.loads(stream.read(FRAME.unpack(head)[0]))


def read_records(path: Path) -> Iterator[dict]:
    """Yield the records of one shard file without holding more than one in memory."""
    for offset,record in scan_records(path):
        yield record


def read_record_at(stream, offset: int) -> dict:
    """Read exactly one record from an open shard file at a recorded offset."""
    stream.seek(offset)
    head = stream.read(FRAME.size)
    if len(head) != FRAME.size:
        raise ValueError(f'truncated record frame at offset {offset}')
    return pickle.loads(stream.read(FRAME.unpack(head)[0]))


_MODELS: dict[str,object] = {}


def load_model(path: str | Path):
    """Load a checkpoint once per process and keep it, exactly as section 16 requires of a worker."""
    import torch
    from .model import new_model
    key = str(Path(path).resolve())
    if key not in _MODELS:
        blob = torch.load(key,map_location='cpu',weights_only=True)
        model = new_model(int(blob['layers']),int(blob['hidden']),int(blob['seed']))
        model.load_state_dict(blob['state_dict'])
        model.eval()
        _MODELS[key] = model
    return _MODELS[key]


def teacher(task: Task, checkpoint: str | None) -> tuple[object,object]:
    """Round 0 is ImmediateGain + CostRanker; a learned round is that seed's own round t-1 model."""
    if checkpoint is None:
        return ImmediateGain(task),CostRanker()
    from .heuristics import NeuralScorer, ValueRanker
    model = load_model(checkpoint)
    return NeuralScorer(model),ValueRanker(model)


def mine(payload: tuple[Task,Budget,object,str | None]) -> dict:
    """Run one teacher search on a task from a fresh canonical arena and return its compact record."""
    task,budget,calc,checkpoint = payload
    arena = Arena()
    arena.calc_cfg = calc
    start = canonical_start(arena,task.target)
    scorer,ranker = teacher(task,checkpoint)
    began = time.perf_counter()
    incumbent,stats,tree = beam_search(arena,task,start,scorer,ranker,budget,True)
    record = extract(arena,task,start,incumbent,tree)
    record['stats'] = dict(expansions=stats.expansions,applies=stats.applies,candidates_scored=stats.candidates_scored,
                           seconds=time.perf_counter()-began,peak_rss_mb=peak_rss_mb(),handles=len(arena.handles))
    return record


# ---------------------------------------------------------------- round populations and shards


def train_tasks(cfg: dict) -> list[Task]:
    """Every benchmark train-split task in id order; section 8.3 runs the teacher on both sub-splits."""
    from . import generator as g
    comp,disc,reserved = g.reserve_shapes(cfg['gen'])
    split = g.benchmark_split(g.shape_table(cfg['gen'],reserved),cfg['gen'],'train')
    return sorted((task for arena,task,gen in split),key=lambda t:t.meta['id'])


def records_dir(output: Path, round_: int, seed: int | None) -> Path:
    """Round-0 records are shared by every seed; a learned round's records are seed-local (section 8.5)."""
    if round_ == 0:
        return output/'round0'
    if seed is None:
        raise ValueError('a learned-teacher round has one record set per seed')
    return output/f'round{round_}'/f'seed{seed}'


def model_dir(output: Path, round_: int, seed: int) -> Path:
    """Every round trains one model per seed, including the shared-record round 0."""
    return output/f'round{round_}'/f'seed{seed}'


def incumbent_cost(record: dict) -> int:
    """The cost this record's trajectory achieved, read from its own incumbent state."""
    for state in record['states']:
        if state['root'] == record['incumbent']:
            return state['cost']
    raise ValueError(f"record for task {record['task_id']} has no incumbent state")


def on_path_digests(record: dict) -> list[bytes]:
    """The section 3.4 structural digests of the record's on-path states, the section 8.5 signatures."""
    arena,ids = rebuild(record)
    return [digest(arena,ids[pos]) for pos in [s['root'] for s in record['labelled']]+[record['incumbent']]]


def mine_shard(tasks: list[Task], budget: Budget, calc: object, compute: object,
               checkpoint: str | None, path: Path) -> dict:
    """Mine one shard with this round's teacher and stream its records straight to one shard file."""
    compute,sizing = sized_compute(compute,'mine')
    rows: list[dict] = []
    began = time.perf_counter()

    def measured() -> Iterator[dict]:
        for record in iter_tasks(mine,[(t.meta['id'],(t,budget,calc,checkpoint)) for t in tasks],compute):
            rows.append(dict(task_id=record['task_id'],cost=incumbent_cost(record),c_gen=record['c_gen'],
                             states=len(record['states']),labelled=len(record['labelled']),**record['stats']))
            yield record
    written = write_records(path,measured())
    return dict(shard=str(path),tasks=len(tasks),checkpoint=checkpoint,worker_sizing=sizing,
                seconds=time.perf_counter()-began,rows=rows,**written)


def merge_records(sources: list[Path], destination: Path) -> dict:
    """Concatenate shard files into one task-id-ordered dataset, holding only one record per shard."""
    import heapq
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing shard files: {missing}')
    return write_records(destination,heapq.merge(*[read_records(path) for path in sources],
                                                 key=lambda record:record['task_id']))


def keep_best(previous: Path, current: Path, destination: Path) -> dict:
    """Keep each task's lowest-cost lineage trajectory (section 8.5); an exact tie keeps the earlier round."""
    beat,novel,on_path = 0,0,0

    def selected() -> Iterator[dict]:
        nonlocal beat,novel,on_path
        for before,after in zip(read_records(previous),read_records(current),strict=True):
            assert before['task_id'] == after['task_id'], 'a lineage shares one task order'
            seen = on_path_digests(before)
            improved = incumbent_cost(after) < incumbent_cost(before)
            beat += improved
            signatures = on_path_digests(after) if improved else seen
            on_path += len(signatures)
            novel += sum(signature not in set(seen) for signature in signatures)
            yield after if improved else before
    written = write_records(destination,selected())
    return dict(beat_parent_rate=beat/written['records'] if written['records'] else 0.0,
                novelty=novel/on_path if on_path else 0.0,tasks=written['records'],improved=beat,
                on_path_states=on_path,novel_states=novel,
                novelty_baseline='the previous kept lineage dataset of the same seed',**written)


# ---------------------------------------------------------------- dataset reading for training


def index_records(path: Path) -> list[dict]:
    """Index each record's offset, task id and state count; no record is retained."""
    return [dict(offset=offset,task_id=record['task_id'],states=len(record['states']),
                 labelled=len(record['labelled'])) for offset,record in scan_records(path)]


def state_tuples(record: dict) -> list[tuple]:
    """Rebuild one record once and expose every recorded state as a training item."""
    arena,ids = rebuild(record)
    labels = {entry['root']:entry for entry in record['labelled']}
    items: list[tuple] = []
    for state in record['states']:
        root = ids[state['root']]
        entry = labels.get(state['root'])
        actions = [action_from_record(a,ids) for a in entry['actions']] if entry else []
        items.append((arena,state_info(arena,root),root,actions,state['value'],
                      entry['positive'] if entry else None))
    return items


def batches(path: Path, index: list[dict], rng: random.Random | None) -> Iterator[list[tuple]]:
    """Yield batches of BATCH states, shuffling inside a bounded window of WINDOW records at a time."""
    order = list(index)
    if rng is not None:
        rng.shuffle(order)
    with path.open('rb') as stream:
        buffer,start = [],0
        for first in range(0,len(order),WINDOW):
            buffer,start = buffer[start:],0
            for entry in order[first:first+WINDOW]:
                buffer += state_tuples(read_record_at(stream,entry['offset']))
            if rng is not None:
                rng.shuffle(buffer)
            while len(buffer)-start >= BATCH:
                yield buffer[start:start+BATCH]
                start += BATCH
        if len(buffer) > start:
            yield buffer[start:]


def batch_loss(model, items: list[tuple], device: str = 'cpu'):
    """Section 8.4 loss: listwise cross-entropy over every labelled candidate list plus value MSE."""
    import torch
    from .model import forward, listwise_loss, make_batch, to_device
    batch = to_device(make_batch([(arena,info,root,actions)
                                  for arena,info,root,actions,value,positive in items]),device)
    logits,values = forward(model,batch)
    targets = torch.tensor([value for arena,info,root,actions,value,positive in items],
                           dtype=values.dtype,device=values.device)
    positives = {i:item[5] for i,item in enumerate(items) if item[5] is not None}
    return listwise_loss(logits,batch.state_of,positives)+torch.nn.functional.mse_loss(values,targets)


def mean_loss(model, path: Path, index: list[dict], device: str = 'cpu') -> float:
    """Mean section 8.4 loss over a split in recorded order, without shuffling or gradients."""
    import torch
    total,count = 0.0,0
    with torch.no_grad():
        for items in batches(path,index,None):
            total += float(batch_loss(model,items,device))*len(items)
            count += len(items)
    return total/count if count else float('nan')


# ---------------------------------------------------------------- execution placement


def resolve_device(setting: str | None = None) -> str:
    """Resolve the training device from the execution setting, never from calibration state.

    Section 16 puts a GPU on training when one is present, so `auto` selects CUDA when it is
    available and CPU otherwise. This is execution configuration (section 18.10): it is not part
    of the calibration identity, it changes no scientific rule, and the resolved value is recorded
    in the run's provenance.
    """
    import os
    import torch
    choice = (setting or os.environ.get('POLYCOMPRESS_DEVICE') or 'auto').strip().lower()
    if choice not in ('auto','cpu','cuda'):
        raise ValueError(f"POLYCOMPRESS_DEVICE must be auto, cpu or cuda, not {choice!r}")
    if choice == 'cuda' and not torch.cuda.is_available():
        raise ValueError('POLYCOMPRESS_DEVICE=cuda but this process has no CUDA device')
    return 'cuda' if choice == 'auto' and torch.cuda.is_available() else ('cuda' if choice == 'cuda' else 'cpu')


def configure_determinism(device: str) -> dict:
    """Hold section 0.6 on every backend: same inputs and seed must give identical results.

    The encoder's `index_add_` scatter is documented as nondeterministic on CUDA, because the
    atomics it uses accumulate in arbitrary order, so training enables deterministic algorithms
    and pins the cuBLAS workspace CUDA requires for reproducible reductions. The environment
    variable must be set before the CUDA context is created, which is why this runs before the
    model is placed. On CPU the call is a no-op guard: those kernels are already deterministic.
    """
    import os
    import torch
    if device == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    return dict(deterministic_algorithms=True,
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def device_record(device: str) -> dict:
    """Describe the backend a run actually used, for the section 18.6 provenance record."""
    import torch
    record = dict(device=device,torch=torch.__version__,torch_cuda_build=torch.version.cuda,
                  cuda_available=torch.cuda.is_available())
    if device == 'cuda':
        record.update(gpu=torch.cuda.get_device_name(0),
                      capability='.'.join(str(v) for v in torch.cuda.get_device_capability(0)))
    return record


# ---------------------------------------------------------------- training


def split_index(index: list[dict]) -> tuple[list[dict],list[dict]]:
    """Cut a dataset index into train_fit and train_modelval exactly as section 8.3 declares."""
    return [e for e in index if not modelval(e['task_id'])],[e for e in index if modelval(e['task_id'])]


def fit(dataset: Path, seed: int, layers: int, cfg: TrainCfg, destination: Path,
        device: str | None = None) -> dict:
    """Train one model from scratch (section 8.4) and keep the lowest train_modelval-loss checkpoint."""
    import torch
    from .model import new_model
    fit_index,val_index = split_index(index_records(dataset))
    if not fit_index or not val_index:
        raise ValueError(f'{dataset} does not cover both train_fit and train_modelval')
    device = resolve_device(device)
    backend = dict(device_record(device),**configure_determinism(device))
    print(f'[fit] backend {backend}',flush=True)
    model = new_model(layers,cfg.hidden,seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr)
    history,best,best_epoch,kept = [],None,None,None
    began = time.perf_counter()
    for epoch in range(cfg.epochs):
        total,count = 0.0,0
        for items in batches(dataset,fit_index,random.Random(f'train|{seed}|{epoch}')):
            optimizer.zero_grad()
            loss = batch_loss(model,items,device)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(items)
            count += len(items)
        measured = mean_loss(model,dataset,val_index,device)
        history.append(dict(epoch=epoch,train_fit_loss=total/count,train_modelval_loss=measured,
                            seconds=time.perf_counter()-began))
        print(f'[fit] seed {seed} L={layers} epoch {epoch} fit {total/count:.6f} '
              f'modelval {measured:.6f}',flush=True)
        if best is None or measured < best:
            best,best_epoch = measured,epoch
            kept = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
        elif epoch-best_epoch >= cfg.patience:
            break
    destination.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state_dict=kept,layers=layers,hidden=cfg.hidden,seed=seed),destination)
    return dict(checkpoint=str(destination),seed=seed,layers=layers,backend=backend,selected_epoch=best_epoch,
                train_modelval_loss=best,epochs_run=len(history),history=history,dataset=str(dataset),
                train_fit_tasks=len(fit_index),train_modelval_tasks=len(val_index),
                recipe=dict(lr=cfg.lr,epochs=cfg.epochs,patience=cfg.patience,hidden=cfg.hidden,
                            batch=BATCH,window=WINDOW),
                seconds=time.perf_counter()-began)


# ---------------------------------------------------------------- HandLinear control


def hand_rows(path: Path, index: list[dict]) -> tuple[np.ndarray,list[tuple[int,int,int]]]:
    """Collect the twelve declared pre-action features of every labelled candidate, with per-state slices."""
    rows,slices = [],[]
    for items in batches(path,index,None):
        for arena,info,root,actions,value,positive in items:
            if positive is None:
                continue
            slices.append((len(rows),len(actions),positive))
            rows += [features(arena,info,root,a) for a in actions]
    return (np.stack(rows) if rows else np.zeros((0,12))),slices


def hand_listwise(scores, slices: list[tuple[int,int,int]]):
    """The same listwise action loss the neural scorer uses, over each labelled state's own slice."""
    import torch
    total,n = torch.zeros((),dtype=torch.float64),0
    for start,width,positive in slices:
        if width < 2:
            continue
        total = total+torch.nn.functional.cross_entropy(scores[start:start+width].unsqueeze(0),
                                                        torch.tensor([positive],dtype=torch.long))
        n += 1
    return total/max(n,1)


def fit_hand_linear(dataset: Path, cfg: TrainCfg, destination: Path) -> dict:
    """Fit the section 7.3 HandLinear control on round-0 train_fit action labels; select on train_modelval."""
    import torch
    fit_index,val_index = split_index(index_records(dataset))
    rows,slices = hand_rows(dataset,fit_index)
    if not slices:
        raise ValueError(f'{dataset} holds no labelled round-0 action')
    # Section 7.3: the statistics come from the round-0 training states and travel with the model.
    mean,std = standardize(rows)
    xs = torch.tensor((rows-mean)/std,dtype=torch.float64)
    val_rows,val_slices = hand_rows(dataset,val_index)
    val_xs = torch.tensor((val_rows-mean)/std,dtype=torch.float64)
    w = torch.zeros(rows.shape[1],dtype=torch.float64,requires_grad=True)
    optimizer = torch.optim.Adam([w],lr=cfg.lr)
    history,best,best_epoch,kept = [],None,None,None
    began = time.perf_counter()
    for epoch in range(cfg.epochs):
        # Section 8.4's cadence, over the labelled states this control actually learns from.
        order = list(slices)
        random.Random(f'handlinear|{epoch}').shuffle(order)
        total,count = 0.0,0
        for first in range(0,len(order),BATCH):
            chunk = order[first:first+BATCH]
            optimizer.zero_grad()
            loss = hand_listwise(xs@w,chunk)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(chunk)
            count += len(chunk)
        with torch.no_grad():
            measured = float(hand_listwise(val_xs@w,val_slices))
        history.append(dict(epoch=epoch,train_fit_loss=total/count,train_modelval_loss=measured))
        print(f'[handlinear] epoch {epoch} fit {total/count:.6f} modelval {measured:.6f}',flush=True)
        if best is None or measured < best:
            best,best_epoch,kept = measured,epoch,w.detach().clone()
        elif epoch-best_epoch >= cfg.patience:
            break
    destination.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(w=kept,b=0.0,mean=torch.tensor(mean),std=torch.tensor(std),
                    selected_epoch=best_epoch,features=int(rows.shape[1])),destination)
    return dict(checkpoint=str(destination),selected_epoch=best_epoch,train_modelval_loss=best,
                epochs_run=len(history),history=history,dataset=str(dataset),
                labelled_states=len(slices),candidate_rows=int(rows.shape[0]),
                feature_mean=mean.tolist(),feature_std=std.tolist(),
                recipe=dict(lr=cfg.lr,epochs=cfg.epochs,patience=cfg.patience),
                seconds=time.perf_counter()-began)


def load_hand_linear(path: str | Path) -> HandLinear:
    """Rebuild the frozen HandLinear control from its checkpoint and stored feature statistics."""
    import torch
    blob = torch.load(str(path),map_location='cpu',weights_only=True)
    return HandLinear(blob['w'].numpy(),float(blob['b']),blob['mean'].numpy(),blob['std'].numpy())


def smoke(tasks: list[Task], budget: Budget, calc: object, compute: object, output: Path,
          projected_tasks: dict[str,int]) -> dict:
    """Mine a sample, stream it, and report sizes, throughput, round-0 and five-seed pipeline projections, and peak RAM."""
    tasks = sorted(tasks,key=lambda t:t.meta['id'])
    missing = projected_tasks.keys()-{t.meta['family'] for t in tasks}
    if missing:
        raise ValueError(f'serialization projection has no sample for {sorted(missing)}')
    compute,sizing = sized_compute(compute,'smoke')
    began = time.perf_counter()
    path = output/'serialization_smoke.bin'
    sizes: list[dict] = []

    def measured() -> Iterator[dict]:
        for task,record in zip(tasks,iter_tasks(mine,[(t.meta['id'],(t,budget,calc,None)) for t in tasks],compute)):
            blob = pickle.dumps(record,protocol=pickle.HIGHEST_PROTOCOL)
            sizes.append(dict(id=task.meta['id'],family=task.meta['family'],bytes=len(blob)+FRAME.size,states=len(record['states']),
                              labelled=len(record['labelled']),actions=sum(len(s['actions']) for s in record['labelled']),
                              nodes=len(record['nodes']),**record['stats']))
            yield record
    written = write_records(path,measured())
    wall = time.perf_counter()-began
    began = time.perf_counter()
    reread = sum(len(r['nodes']) > 0 for r in read_records(path))
    read_seconds = time.perf_counter()-began
    assert reread == written['records'], 'every streamed record reads back'
    began = time.perf_counter()
    copy = path.with_suffix('.copy.bin')
    assert write_records(copy,read_records(path)) == written, 'a streamed copy is byte-for-byte complete'
    io_seconds = time.perf_counter()-began
    copy.unlink()
    by_family = {family:[s for s in sizes if s['family'] == family] for family in sorted({s['family'] for s in sizes})}
    mean = {family:sum(s['bytes'] for s in rows)/len(rows) for family,rows in by_family.items()}
    projected = sum(mean[family]*projected_tasks[family] for family in projected_tasks)
    from .config import TrainCfg
    # One shared round-0 dataset plus two learned-teacher rounds per seed (§8.5).
    equivalents = 1+2*len(TrainCfg().seeds)
    report = dict(sample=sizes,mean_bytes_by_family=mean,mine_and_write_seconds=wall,
                  mine_and_write_bytes_per_second=written['bytes']/wall,read_seconds=read_seconds,
                  read_bytes_per_second=written['bytes']/read_seconds,
                  copy_seconds=io_seconds,write_bytes_per_second=written['bytes']/io_seconds,
                  projected_round_bytes=projected,projected_tasks=projected_tasks,
                  dataset_equivalents=equivalents,projected_pipeline_bytes=projected*equivalents,
                  train_seeds=list(TrainCfg().seeds),worker_sizing=sizing,
                  parent_peak_rss_mb=peak_rss_mb(),worker_peak_rss_mb=max(s['peak_rss_mb'] for s in sizes),file=str(path),**written)
    (output/'serialization_smoke.json').write_text(json.dumps(report,indent=1)+'\n')
    return report


def _shard_stem(shard: str) -> str:
    """Name a shard file by its index, exactly as the section 16 cluster option declares."""
    return f"shard_{shard.split('/')[0]}"


def _report(directory: Path, name: str, report: dict) -> Path:
    """Write one stage report next to the artifacts it describes."""
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/name
    path.write_text(json.dumps(report,indent=1,default=str)+'\n')
    return path


def _provenance(destination: Path, artifacts: list[Path], run: dict, parents: dict | None = None) -> None:
    """Record the section 18.6 production-run provenance beside this stage's outputs."""
    from .protocol import write_provenance
    write_provenance(Path(__file__).resolve().parent.parent,destination,artifacts,parents,run=run)


def main() -> None:
    """Run one Phase-2 pipeline stage: the serialization smoke test, mining, merging, training, or HandLinear."""
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',nargs='?',default='smoke',choices=('smoke','mine','merge','fit','handlinear'))
    parser.add_argument('--output',type=Path,default=None,help='where this stage writes')
    parser.add_argument('--datasets',type=Path,default=None,help='where this stage reads datasets and parents')
    parser.add_argument('--round',type=int,default=0)
    parser.add_argument('--seed',type=int,default=None)
    parser.add_argument('--layers',type=int,default=None,help='0 selects the required flat-only control')
    parser.add_argument('--shard',default='0/1')
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--every',type=int,default=100)
    args = parser.parse_args()
    from .experiments.pilot import shard_indices
    from .protocol import _configs
    cfg,began = _configs(),time.perf_counter()
    default = dict(smoke=Path('out/pilot'),handlinear=Path('out/controls')).get(args.stage,Path('out/train'))
    output = default if args.output is None else args.output
    datasets = output if args.datasets is None else args.datasets
    if args.round < 0 or args.round > cfg['train'].rounds:
        raise ValueError(f"round must lie in 0..{cfg['train'].rounds}")
    if args.stage in ('mine','merge','fit') and args.round > 0 and args.seed is None:
        raise ValueError('a learned-teacher round is seed-local; pass --seed')
    if args.stage == 'fit' and args.seed is None:
        raise ValueError('training is per seed; pass --seed')
    if args.seed is not None and args.seed not in cfg['train'].seeds:
        raise ValueError(f"seed must be one of {list(cfg['train'].seeds)}")

    if args.stage == 'smoke':
        from . import generator as g
        comp,disc,reserved = g.reserve_shapes(cfg['gen'])
        split = g.benchmark_splits(g.shape_table(cfg['gen'],reserved),cfg['gen'])['train']
        tasks = [task for arena,task,gen in split if task.meta['id'] % args.every == 0]
        projected = {family:cfg['gen'].n_train_per_family for family in ('F1','F2','F3','F4')}
        report = smoke(tasks,cfg['search'].teacher,cfg['calc'],cfg['compute'],output,projected)
        print(json.dumps({key:value for key,value in report.items() if key != 'sample'},indent=1))
    elif args.stage == 'mine':
        tasks = train_tasks(cfg)
        selected = [tasks[i] for i in shard_indices(args.shard,len(tasks))]
        parent = None if args.round == 0 else model_dir(datasets,args.round-1,args.seed)/'model.pt'
        directory = records_dir(output,args.round,args.seed)
        path = directory/f'{_shard_stem(args.shard)}.bin'
        report = mine_shard(selected,cfg['search'].teacher,cfg['calc'],cfg['compute'],
                            None if parent is None else str(parent),path)
        _report(directory,f'{_shard_stem(args.shard)}.json',report)
        _provenance(directory/f'{_shard_stem(args.shard)}.provenance.json',[path],
                    dict(stage='mine',round=args.round,seed=args.seed,shard=args.shard),
                    None if parent is None else dict(teacher=parent))
        print(f"mined {report['records']} records, {report['bytes']} bytes -> {path}")
    elif args.stage == 'merge':
        directory = records_dir(output,args.round,args.seed)
        shards = [directory/f'shard_{i}.bin' for i in range(args.shards)]
        dataset = directory/DATASET
        if args.round == 0:
            report = dict(stage='merge',round=0,shards=args.shards,**merge_records(shards,dataset))
        else:
            mined = directory/'mined.bin'
            written = merge_records(shards,mined)
            previous = records_dir(datasets,args.round-1,args.seed)/DATASET
            report = dict(stage='merge',round=args.round,seed=args.seed,shards=args.shards,
                          previous=str(previous),mined=dict(path=str(mined),**written),
                          **keep_best(previous,mined,dataset))
        _report(directory,'dataset.json',report)
        _provenance(directory/'dataset.provenance.json',[dataset],
                    dict(stage='merge',round=args.round,seed=args.seed,shards=args.shards))
        print(json.dumps({key:value for key,value in report.items() if key != 'mined'},indent=1))
    elif args.stage == 'fit':
        layers = cfg['train'].layers if args.layers is None else args.layers
        dataset = records_dir(datasets,args.round,args.seed)/DATASET
        directory = model_dir(output,args.round,args.seed)
        report = fit(dataset,args.seed,layers,cfg['train'],directory/'model.pt')
        _report(directory,'training.json',report)
        _provenance(directory/'model.provenance.json',[directory/'model.pt'],
                    dict(stage='fit',round=args.round,seed=args.seed,layers=layers,
                         execution=report['backend']),
                    dict(dataset=dataset) | ({} if args.round == 0 else
                                             dict(teacher=model_dir(datasets,args.round-1,args.seed)/'model.pt')))
        print(json.dumps({key:value for key,value in report.items() if key != 'history'},indent=1))
    else:
        dataset = records_dir(datasets,0,None)/DATASET
        report = fit_hand_linear(dataset,cfg['train'],output/'hand_linear.pt')
        _report(output,'hand_linear.json',report)
        _provenance(output/'hand_linear.provenance.json',[output/'hand_linear.pt'],
                    dict(stage='handlinear',round=0),dict(dataset=dataset))
        print(json.dumps({key:value for key,value in report.items() if key != 'history'},indent=1))
    print(f'{args.stage} done in {time.perf_counter()-began:.0f} s',flush=True)


if __name__ == '__main__':
    main()
