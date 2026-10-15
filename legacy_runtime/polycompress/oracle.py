"""Bounded exact binary-DAG optimum certification (specification section 13.2)."""

import time

from .calculus import Task
from .config import OracleCfg
from . import poly as p


def _exponents(value: p.Poly) -> tuple[int, ...]:
    """Return the per-variable maximum exponent of a polynomial."""
    fields = [p.unpack(m) for m in value]
    return tuple(max(f[i] for f in fields) for i in range(p.MAX_VARS))


def _cap(m: int, target: p.Poly, memo: dict[int,int | None]) -> int | None:
    """Return the largest coefficient a useful gate may carry on monomial m."""
    if m not in memo:
        divisors = [c for t,c in target.items() if p.divides(m,t)]
        memo[m] = max(divisors) if divisors else None
    return memo[m]


def admissible(value: p.Poly, target: p.Poly, memo: dict[int,int | None]) -> bool:
    """Apply the declared safe monotone prunes to one candidate gate polynomial."""
    for m,c in value.items():
        cap = _cap(m,target,memo)
        if cap is None or c > cap:
            return False
    return True


def _reachable(values: list[p.Poly], target: p.Poly, left: int) -> bool:
    """Bound the size a monotone DAG can still reach in `left` gates; nothing ever cancels."""
    ceiling = max(sum(target.values()),len(target))+1
    occurrences = min(max(sum(v.values()) for v in values),ceiling)
    distinct = min(max(len(v) for v in values),ceiling)
    for _ in range(left):
        occurrences = min(max(2*occurrences,occurrences*occurrences),ceiling)
        distinct = min(max(2*distinct,distinct*distinct),ceiling)
    return occurrences >= sum(target.values()) and distinct >= len(target)


def _state(values: list[p.Poly]) -> frozenset:
    """Key a partial DAG by its set of node polynomials; gate order and identity never matter."""
    return frozenset(tuple(sorted(v.items())) for v in values)


def _dfs(target: p.Poly, values: list[p.Poly], caps: list[tuple], ceiling: tuple, gates: list,
         left: int, deadline: float, memo: dict[int,int | None], seen: set) -> list | None | str:
    """Append one more gate in deterministic order; return a witness gate list, None, or 'timeout'."""
    if left == 0 or not _reachable(values,target,left):
        return None
    if time.perf_counter() >= deadline:
        return 'timeout'
    state = (_state(values),left)
    if state in seen:
        return None
    n = len(values)
    for op in ('ADD','MUL'):
        for a in range(n):
            for b in range(a,n):
                if op == 'MUL':
                    # A nontrivial pure constant may not be a MUL operand; the unit is trivial.
                    if any(p.degree(values[i]) == 0 for i in (a,b)):
                        continue
                    # The largest product exponent is the sum of the operand maxima, so above the
                    # target's own ceiling the gate already has a monomial dividing no target
                    # monomial, and forming it would overflow a packed exponent field.
                    if any(x+y > t for x,y,t in zip(caps[a],caps[b],ceiling)):
                        continue
                value = p.add(values[a],values[b]) if op == 'ADD' else p.mul(values[a],values[b])
                # A node duplicating an existing polynomial can never shrink a minimum circuit,
                # which also subsumes the declared no-duplicate-(op, a, b) rule.
                if any(value == v for v in values) or not admissible(value,target,memo):
                    continue
                if value == target:
                    return gates+[(op,a,b)]
                result = _dfs(target,values+[value],caps+[_exponents(value)],ceiling,gates+[(op,a,b)],
                              left-1,deadline,memo,seen)
                if result is not None:
                    return result
    # Only an exhaustively searched state is memoized; a deadline propagates as 'timeout' instead.
    seen.add(state)
    return None


def certify(task: Task, cfg: OracleCfg) -> dict:
    """Certify the exact minimum gate count by iterative deepening, or report the time cap."""
    target = task.target
    variables = sorted(p.support(target))
    leaves = [{1 << (4*i):1} for i in variables]+[p.one()]
    began = time.perf_counter()
    deadline = began+cfg.time_s
    if any(target == leaf for leaf in leaves):
        return dict(c_star=0,covered=True,witness=[],seconds=time.perf_counter()-began)
    memo: dict[int,int | None] = {}
    seen: set = set()
    caps,ceiling = [_exponents(leaf) for leaf in leaves],_exponents(target)
    for bound in range(task.c_gen+1):
        witness = _dfs(target,leaves,caps,ceiling,[],bound,deadline,memo,seen)
        if witness == 'timeout':
            return dict(c_star=None,covered=False,witness=None,certified_bound=bound-1,
                        seconds=time.perf_counter()-began)
        if witness is not None:
            return dict(c_star=bound,covered=True,witness=[list(g) for g in witness],
                        seconds=time.perf_counter()-began)
    return dict(c_star=None,covered=False,witness=None,certified_bound=task.c_gen,
                seconds=time.perf_counter()-began)


def c_star_dag(task: Task, cfg: OracleCfg | None = None) -> int | None:
    """Return the exact bounded binary-DAG optimum, or None when the time cap expires."""
    return certify(task,cfg or OracleCfg())['c_star']


def eligible(task: Task, cfg: OracleCfg) -> bool:
    """Apply the frozen structural eligibility predicate; coverage is measured separately."""
    return sum(task.target.values()) <= cfg.max_occurrences and task.n_vars <= cfg.max_vars


def r_star(c_start: int, c_method: int, c_star: int | None) -> float | None:
    """Return certified-subset headroom recovery; an uncovered task never receives R*."""
    if c_star is None:
        return None
    if c_start == c_star:
        return None
    return (c_start-c_method)/(c_start-c_star)
