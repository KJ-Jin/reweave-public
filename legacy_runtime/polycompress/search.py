"""Budgeted beam search, deterministic independent task mapping, and memory-sized worker pools."""

from dataclasses import dataclass


@dataclass
class Budget:
    B: int
    K_act: int
    T_max: int
    max_expansions: int

import csv
import multiprocessing
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator, Protocol, TYPE_CHECKING, Callable, TypeVar

from .calculus import Task, Action, action_key, apply, verify
from .handles import Arena, cost, digest
from .state import StateInfo, state_info
from .enumerate import enumerate_actions

if TYPE_CHECKING:
    from .config import ComputeCfg


class Scorer(Protocol):
    def score(self, arena: Arena, info: StateInfo, root: int, actions: list[Action]) -> list[float]:
        """Score candidates, higher first."""
        ...


class Ranker(Protocol):
    def rank(self, arena: Arena, info: StateInfo, root: int) -> float:
        """Rank states, lower first."""
        ...


@dataclass
class Stats:
    candidates_scored: int = 0
    applies: int = 0
    verifier_calls: int = 0
    expansions: int = 0
    steps: int = 0
    wall: float = 0.0
    peak_rss_mb: float = 0.0


def peak_rss_mb() -> float:
    """Read this worker's measured peak resident memory."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform == 'darwin' else 1024)


def beam_search(arena: Arena, task: Task, start: int, scorer: Scorer,
                ranker: Ranker, budget: Budget, record: bool) -> tuple[int,Stats,dict]:
    """Run the admission-limited beam; ties on (rank, cost) break by structural digest, never by id."""
    assert min(budget.B,budget.K_act) > 0 and min(budget.T_max,budget.max_expansions) >= 0
    began,applies0,verify0 = time.perf_counter(),arena.apply_count,arena.verifier_count
    stats = Stats()
    tree = {start:dict(parent=None,action=None,cost=cost(arena,start))} if record else {}
    beam,visited,incumbent = [start],{start},start
    for step in range(budget.T_max):
        if stats.expansions >= budget.max_expansions:
            break
        stats.steps = step+1
        succ = {}
        for root in beam:
            info = state_info(arena,root)
            actions = enumerate_actions(arena,info,root,arena.calc_cfg)
            stats.candidates_scored += len(actions)
            scores = scorer.score(arena,info,root,actions)
            assert len(scores) == len(actions)
            ranked = sorted(zip(actions,scores),key=lambda pair:(-pair[1],action_key(arena,pair[0])))
            chosen = [a for a,s in ranked[:budget.K_act]]
            if record:
                tree[root].update(actions=actions,scores=scores,chosen=chosen,successors=[])
            for a in chosen:
                root2 = apply(arena,task,info,root,a)
                if root2 is None or root2 in visited:
                    continue
                verify(arena,task,root2)
                visited.add(root2)
                stats.expansions += 1
                c = cost(arena,root2)
                succ[root2] = ranker.rank(arena,state_info(arena,root2),root2),c,digest(arena,root2)
                if record:
                    tree[root]['successors'].append(root2)
                    tree[root2] = dict(parent=root,action=a,cost=c)
                if c < cost(arena,incumbent):
                    incumbent = root2
                if stats.expansions >= budget.max_expansions:
                    break
            if stats.expansions >= budget.max_expansions:
                break
        if not succ:
            break
        beam = sorted(succ,key=succ.get)[:budget.B]
    stats.applies,stats.verifier_calls = arena.apply_count-applies0,arena.verifier_count-verify0
    stats.wall,stats.peak_rss_mb = time.perf_counter()-began,peak_rss_mb()
    return incumbent,stats,tree


def _worker_init() -> None:
    import torch
    torch.set_num_threads(1)


T = TypeVar('T')
R = TypeVar('R')


def iter_tasks(fn: Callable[[T],R], tasks: list[tuple[int,T]], cfg: 'ComputeCfg') -> Iterator[R]:
    """Yield results of independent id/payload tasks in task-id order as they complete."""
    if cfg.n_workers < 1:
        raise ValueError('n_workers must be positive')
    payloads = [task for ident,task in sorted(tasks,key=lambda pair:pair[0])]
    if cfg.n_workers == 1:
        _worker_init()
        yield from (fn(task) for task in payloads)
        return
    with multiprocessing.get_context('spawn').Pool(cfg.n_workers,initializer=_worker_init,maxtasksperchild=1) as pool:
        yield from pool.imap(fn,payloads,chunksize=1)


def run_tasks(fn: Callable[[T],R], tasks: list[tuple[int,T]], cfg: 'ComputeCfg') -> list[R]:
    """Map independent id/payload tasks and return results in task-id order."""
    return list(iter_tasks(fn,tasks,cfg))


def physical_memory_mb() -> float:
    """Read installed physical memory."""
    if sys.platform == 'darwin':
        return int(subprocess.run(['sysctl','-n','hw.memsize'],capture_output=True,text=True,check=True).stdout)/2**20
    return os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/2**20


def size_workers(requested: int, per_worker_mb: float, reserve_mb: float = 2048.0,
                 total_mb: float | None = None) -> int:
    """Cap the worker count by memory: at least one, at most requested, at most (total - reserve) / peak."""
    if requested < 1 or per_worker_mb <= 0:
        raise ValueError('requested workers and per-worker memory must be positive')
    total = physical_memory_mb() if total_mb is None else total_mb
    return max(1,min(requested,int((total-reserve_mb)//per_worker_mb)))


CENSUS = Path('out/pilot/results.csv')


def measured_peak_rss_mb(results: Path = CENSUS) -> float | None:
    """Read the pilot census's measured per-worker peak RSS, the section 16 sizing basis."""
    if not results.is_file():
        return None
    with results.open(newline='') as stream:
        values = [float(row['peak_rss_mb']) for row in csv.DictReader(stream) if row.get('peak_rss_mb')]
    return max(values) if values else None


def sized_compute(cfg: 'ComputeCfg', stage: str, results: Path | None = None) -> tuple['ComputeCfg',dict]:
    """Size a pool by measured memory before it is opened and return the config plus the recorded basis."""
    from dataclasses import replace
    results = CENSUS if results is None else results
    measured = None if cfg.n_workers < 2 else measured_peak_rss_mb(results)
    if cfg.n_workers < 2:
        workers,basis = 1,'one worker requested; no pool is opened'
    elif measured is None:
        # Fail safe: never open a pool sized on an unmeasured guess.
        workers,basis = 1,f'no measured peak_rss_mb in {results}; sized down to one worker'
    else:
        workers = size_workers(cfg.n_workers,measured)
        basis = f'measured per-worker peak RSS {measured:.0f} MiB from {results}'
    record = dict(stage=stage,requested=cfg.n_workers,workers=workers,peak_rss_mb=measured,basis=basis)
    print(f'[workers] {stage}: requested {cfg.n_workers}, using {workers} ({basis})',flush=True)
    return replace(cfg,n_workers=workers),record
