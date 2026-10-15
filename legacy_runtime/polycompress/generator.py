"""Declared skeletons, slots, compositions, discovery gap, splits, leakage checks, and the E7/random/scale/named sets."""

from collections import Counter
from dataclasses import replace
import hashlib
from itertools import combinations, combinations_with_replacement, product
import random
import sys
import time

from .calculus import Action, Task, apply, substitute
from .config import CalcCfg, ComputeCfg, GenCfg, PilotCfg, ProtocolCfg, RandomOODCfg, ScaleCfg
from .enumerate import enumerate_actions
from .handles import Arena,Mono,Sum,Prod,canonical_start,cost,items,mk_mono,mk_sum,mk_prod,poly
from . import poly as p
from .search import run_tasks, sized_compute
from .state import state_info

Instance = tuple[Arena,Task,int]
ATOMIC = ('x','xy')
DISC = ('zsum','twosums','square')
WIDTH = {'x':1,'xy':2,'zsum':3,'twosums':4,'square':2}
OWNERSHIP = ('val','test_id','train','train','train')
COMPOSITIONS = (('F1','F2'),('F2','F1'),('F2','F3'),('F3','F2'))
FAMILY_GRID = {'F1':((2,2),(2,3),(3,2)),'F2':((2,2),(2,3),(3,2),(3,3)),'F3':((2,2),(2,3),(3,2),(3,3))}


def seed(tag: str) -> int:
    """Derive one tagged stream from the single declared generation seed."""
    digest = hashlib.sha256(f'PolyCompress|{ProtocolCfg().gen_seed_test}|{tag}'.encode()).digest()
    return int.from_bytes(digest[:8],'big')


def _slot(arena: Arena, shape: str, variables: list[int]) -> int:
    nodes = [mk_mono(arena,1 << (4*i)) for i in variables]
    if shape in ATOMIC:
        return mk_prod(arena,Counter(nodes))
    if shape == 'zsum':
        return mk_prod(arena,Counter([nodes[0],mk_sum(arena,Counter(nodes[1:]))]))
    if shape == 'twosums':
        return mk_prod(arena,Counter(mk_sum(arena,Counter(nodes[i:i+2])) for i in (0,2)))
    if shape == 'square':
        return mk_prod(arena,Counter({mk_sum(arena,Counter(nodes)):2}))
    raise ValueError(f'unknown slot shape: {shape}')


def _family(arena: Arena, family: str, params: tuple[int,int], slots: list[int]) -> int:
    k,n = params
    if family not in ('F1','F2','F3') or k not in (2,3) or n not in (2,3):
        raise ValueError('unknown family or grid point')
    assert len(slots) == (k if family == 'F1' else k+n)
    s = mk_sum(arena,Counter(slots[:k]))
    if family == 'F1':
        return mk_prod(arena,Counter({s:n}))
    if family == 'F2':
        t = mk_sum(arena,Counter(slots[k:]))
        return mk_prod(arena,Counter([s,t]))
    t = mk_sum(arena,Counter([s,slots[-1]]))
    terms = [mk_prod(arena,Counter([s,b])) for b in slots[k:-1]]
    terms.append(mk_prod(arena,Counter({t:2})))
    return mk_sum(arena,Counter(terms))


def _allocate(arena: Arena, shapes: tuple[str,...], rng: random.Random | None,
              extra: int, cfg: GenCfg) -> tuple[list[int],list[int],list[int]]:
    n = sum(WIDTH[s] for s in shapes)+extra
    assert n <= cfg.max_vars, 'variable width is filtered before allocation'
    permutation = list(range(n)) if rng is None else rng.sample(range(n),n)
    offset,slots = 0,[]
    for shape in shapes:
        slots.append(_slot(arena,shape,permutation[offset:offset+WIDTH[shape]]))
        offset += WIDTH[shape]
    return slots,permutation[offset:],permutation


def build_instance(family: str, params: tuple[int,...], shapes: tuple[str,...],
                   rng: random.Random | None, cfg: GenCfg, embedded_family: str | None = None,
                   choice: int | None = None) -> Instance:
    """Build one explicit grid/slot instance; a None rng keeps the block variable order."""
    arena = Arena()
    d = params[0] if family == 'F4' else 0
    if family == 'F4' and (d not in range(1,cfg.d_max+1) or embedded_family not in ('F1','F2')):
        raise ValueError('invalid F4 motif or distractor count')
    slots,extra,permutation = _allocate(arena,shapes,rng,2*d,cfg)
    motif = _family(arena,embedded_family if family == 'F4' else family,
                    (2,2) if family == 'F4' else params,slots)
    gen = motif
    if family == 'F4':
        monomials = sorted(poly(arena,motif),key=lambda m:p.pack(tuple(p.unpack(m)[v] for v in permutation)))
        m = monomials[choice] if choice is not None else rng.choice(monomials)
        terms = [motif,mk_mono(arena,m)]
        terms.extend(mk_mono(arena,(1 << (4*extra[i]))+(1 << (4*extra[i+1]))) for i in range(0,len(extra),2))
        gen = mk_sum(arena,Counter(terms))
    meta = dict(family=family,params=params,slot_shapes=shapes,composition_depth=0,slots=slots,motif=motif,
                embedded_family=embedded_family,skeleton_signature=skeleton_signature(arena,gen),
                embedded_signature=skeleton_signature(arena,motif) if family == 'F4' else None)
    task = Task(poly(arena,gen),sum(WIDTH[s] for s in shapes)+2*d,cost(arena,gen),meta)
    return arena,task,gen


def n_slots(family: str, params: tuple[int,int]) -> int:
    """Count the named atomic slots of one outer skeleton."""
    return params[0] if family == 'F1' else sum(params)


def build_composition(outer: str, outer_params: tuple[int,int], inner: str, inner_params: tuple[int,int],
                      slot: int, rng: random.Random | None) -> Instance:
    """Build one depth-two construction with the inner family in one named outer slot."""
    if (outer,inner) not in COMPOSITIONS:
        raise ValueError('undeclared composition')
    inner_width,outer_width = n_slots(inner,inner_params),n_slots(outer,outer_params)
    assert slot in range(outer_width), 'insertion slot outside the outer skeleton'
    n = inner_width+outer_width-1
    arena = Arena()
    permutation = list(range(n)) if rng is None else rng.sample(range(n),n)
    leaves = [mk_mono(arena,1 << (4*i)) for i in permutation]
    nested = _family(arena,inner,inner_params,leaves[:inner_width])
    slots = leaves[inner_width:]
    slots.insert(slot,nested)
    gen = _family(arena,outer,outer_params,slots)
    meta = dict(family=outer,params=outer_params,slot_shapes=('x',)*n,composition_depth=2,
                composition=(outer,inner),inner_params=inner_params,insertion_slot=slot,
                embedded_family=None,slots=slots,skeleton_signature=skeleton_signature(arena,gen))
    return arena,Task(poly(arena,gen),n,cost(arena,gen),meta),gen


def composition_grid(outer: str, inner: str, cfg: GenCfg) -> list[tuple]:
    """List every declared outer-grid, inner-grid and insertion-slot construction whose allocation fits."""
    return [(op,ip,slot) for op in FAMILY_GRID[outer] for ip in FAMILY_GRID[inner]
            if n_slots(inner,ip)+n_slots(outer,op)-1 <= cfg.max_vars for slot in range(n_slots(outer,op))]


def composition_pools(cfg: GenCfg) -> dict[str,dict]:
    """Own each cell's canonical shapes, resolving duplicates by C_gen then descriptor, earlier cell first."""
    pools,owner = {},{}
    for outer,inner in COMPOSITIONS:
        cell,best,admissible = f'{outer}∘{inner}',{},0
        for descriptor in composition_grid(outer,inner,cfg):
            instance = build_composition(outer,descriptor[0],inner,descriptor[1],descriptor[2],None)
            if size_rejection(instance[1],cfg):
                continue
            admissible += 1
            shape,score = p.canon(instance[1].target),(instance[1].c_gen,descriptor)
            if shape not in best or score < best[shape][0]:
                best[shape] = score,descriptor
        pools[cell] = dict(shapes={shape:entry[1] for shape,entry in sorted(best.items())
                                   if owner.setdefault(shape,cell) == cell},
                           constructions=admissible,unique_canons=len(best))
    return pools


def composition_set(cfg: GenCfg, pools: dict[str,dict]) -> list[Instance]:
    """Emit each cell's quota over its owned canonical shapes, covering as many distinct shapes as possible."""
    rng,result = random.Random(seed('test_comp')),[]
    for outer,inner in COMPOSITIONS:
        cell = f'{outer}∘{inner}'
        owned = pools[cell]['shapes']
        shapes = sorted(owned)
        assert shapes, f'composition cell {cell} owns no admissible canonical shape'
        rng.shuffle(shapes)
        quota = cfg.n_per_composition
        chosen = shapes[:quota] if len(shapes) >= quota else shapes
        counts = [1]*len(chosen) if len(shapes) >= quota else [1+extra for extra in even_split(quota-len(shapes),len(shapes))]
        for shape,count in zip(chosen,counts):
            op,ip,slot = owned[shape]
            for _ in range(count):
                instance = build_composition(outer,op,inner,ip,slot,rng)
                assert p.canon(instance[1].target) == shape, 'renaming preserves the canonical shape'
                instance[1].meta.update(set='test_comp',composition_cell=cell)
                result.append(instance)
    return _identify(result)


def size_rejection(task: Task, cfg: GenCfg) -> str | None:
    """Return the first violated declared size bound."""
    for reason,value,cap in [('max_vars',task.n_vars,cfg.max_vars),
                             ('max_occurrences',sum(task.target.values()),cfg.max_occurrences),
                             ('max_distinct',len(task.target),cfg.max_distinct),
                             ('max_degree',p.degree(task.target),cfg.max_degree)]:
        if value > cap:
            return reason
    return None


def discovery_instance(family: str, params: tuple[int,int], shapes: tuple[str,...],
                       rng: random.Random | None, cfg: GenCfg) -> Instance | str:
    """Build the reference-flat gap and easy start, or name the rejection reason."""
    if family not in ('F1','F2','F3') or params != (2,2) or all(s in ATOMIC for s in shapes):
        raise ValueError('invalid discovery skeleton')
    arena,task,gen = build_instance(family,params,shapes,rng,cfg)
    reason,flat = size_rejection(task,cfg),gen
    for shape,slot in zip(shapes,task.meta['slots']):
        if shape in DISC:
            flat = substitute(arena,flat,slot,canonical_start(arena,poly(arena,slot)))
    if reason or cost(arena,flat) <= task.c_gen:
        return reason or 'discovery_gap'
    task.meta['gen_flat'] = flat
    task.meta['easy_start'] = _easy_start(arena,family,params,task.meta['slots'])
    task.meta['composite_slots'] = [h for s,h in zip(shapes,task.meta['slots']) if s in DISC]
    return arena,task,gen


def _easy_start(arena: Arena, family: str, params: tuple[int,int], slots: list[int]) -> int:
    symbolic = Arena()
    variables = [mk_mono(symbolic,1 << (4*i)) for i in range(len(slots))]
    outer = _family(symbolic,family,params,variables)
    terms: Counter[int] = Counter()
    for m,c in sorted(poly(symbolic,outer).items()):
        factors = Counter({slots[i]:e for i,e in enumerate(p.unpack(m)) if e})
        terms[mk_prod(arena,factors)] += c
    return mk_sum(arena,terms)


def slot_recovered(arena: Arena, roots: list[int], slots: list[int]) -> bool:
    """Check exact composite-slot recovery anywhere in a trajectory."""
    targets = [poly(arena,s) for s in slots]
    return any(not isinstance(arena.handles[h],Mono) and poly(arena,h) in targets
               for root in roots for h in state_info(arena,root).reach_h)


def skeleton_signature(arena: Arena, root: int) -> tuple:
    """Erase variable/family names while preserving nested operator structure."""
    node = arena.handles[root]
    if isinstance(node,Mono):
        return ('M',tuple(sorted(e for e in p.unpack(node.m) if e)))
    return ('S' if isinstance(node,Sum) else 'P',
            tuple(sorted((skeleton_signature(arena,h),c) for h,c in items(node))))


def family_cells(cfg: GenCfg) -> list[tuple[str,tuple[int,...]]]:
    """List every allowed family/parameter cell in lexicographic order."""
    grid = [(a,b) for a in (2,3) for b in (2,3)]
    cells = [('F1',g) for g in grid if g != (3,3)]+[(f,g) for f in ('F2','F3') for g in grid]
    return cells+[('F4',(d,)) for d in range(1,cfg.d_max+1)]


def cell_choices(family: str, params: tuple[int,...], disc: bool) -> list[tuple]:
    """List every ordered generator choice of a cell before variable permutation."""
    if disc:
        width = 2 if family == 'F1' else 4
        return [(s,None,None) for s in product(ATOMIC+DISC,repeat=width) if any(x in DISC for x in s)]
    if family == 'F4':
        return [(shapes,embedded,m) for embedded,width,count in (('F1',2,3),('F2',4,4))
                for shapes in product(ATOMIC,repeat=width) for m in range(count)]
    width = params[0] if family == 'F1' else sum(params)
    return [(shapes,None,None) for shapes in product(ATOMIC,repeat=width)]


def build_choice(cell: dict, choice: tuple, rng: random.Random | None, cfg: GenCfg) -> Instance | str:
    """Build one generator choice of a cell, or name its rejection reason."""
    shapes,embedded,m = choice
    family,params = cell['family'],cell['params']
    if sum(WIDTH[s] for s in shapes)+(2*params[0] if family == 'F4' else 0) > cfg.max_vars:
        return 'max_vars'
    if cell['disc']:
        return discovery_instance(family,params,shapes,rng,cfg)
    instance = build_instance(family,params,shapes,rng,cfg,embedded,m)
    return size_rejection(instance[1],cfg) or instance


def slot_mix(choice: tuple) -> tuple[str,...]:
    """Describe a choice by its unordered slot fillers and any embedded motif family."""
    return tuple(sorted(choice[0]))+(() if choice[1] is None else (choice[1],))


def enumerate_cell(family: str, params: tuple[int,...], cfg: GenCfg, disc: bool = False) -> dict:
    """Collapse one cell's admissible generator choices by canonical shape."""
    cell = dict(family=family,params=params,disc=disc,shapes={},rejected=Counter())
    for choice in cell_choices(family,params,disc):
        result = build_choice(cell,choice,None,cfg)
        if isinstance(result,str):
            cell['rejected'][(slot_mix(choice),result)] += 1
        else:
            cell['shapes'].setdefault(p.canon(result[1].target),[]).append(choice)
    return cell


def mix_cells(cell: dict) -> list[dict]:
    """Split one cell into its slot-mix sub-cells with per-reason rejections."""
    mixes: dict[tuple,dict] = {}
    for shape,choices in cell['shapes'].items():
        for choice in choices:
            sub = mixes.setdefault(slot_mix(choice),dict(cell,shapes={},rejected=Counter()))
            sub['shapes'].setdefault(shape,[]).append(choice)
    for (mix,reason),count in cell['rejected'].items():
        mixes.setdefault(mix,dict(cell,shapes={},rejected=Counter()))['rejected'][reason] += count
    return [dict(mixes[mix],mix=mix) for mix in sorted(mixes)]


def owners(cell: dict, reserved: frozenset = frozenset()) -> list[tuple[tuple,str]]:
    """Shuffle a cell's unreserved sorted shapes with its tagged stream and cycle split ownership."""
    order = [shape for shape in sorted(cell['shapes']) if shape not in reserved]
    random.Random(seed(f"shape|{cell['family']}|{','.join(map(str,cell['params']))}")).shuffle(order)
    return [(shape,OWNERSHIP[i % len(OWNERSHIP)]) for i,shape in enumerate(order)]


def owned_groups(cell: dict, split: str, exclude: tuple[str,...] = ()) -> list[list[tuple]]:
    """List the choice lists of the shapes a split owns, minus excluded embedded motifs."""
    groups = [[c for c in cell['shapes'][shape] if c[1] not in exclude]
              for shape,owner in cell['owners'] if owner == split]
    return [group for group in groups if group]


def even_split(count: int, bins: int) -> list[int]:
    """Spread a count over ordered bins, remainder to the first bins."""
    assert bins > 0 and count >= 0
    return [count//bins+(i < count % bins) for i in range(bins)]


def emit_shapes(cell: dict, groups: list[list[tuple]], quota: int, rng: random.Random,
                cfg: GenCfg, split: str) -> list[Instance]:
    """Draw a quota evenly over shape choice lists using one split stream."""
    result = []
    for count,choices in zip(even_split(quota,len(groups)),groups):
        for _ in range(count):
            instance = build_choice(cell,rng.choice(choices),rng,cfg)
            assert not isinstance(instance,str), 'shapes are size-filtered before emission'
            instance[1].meta['set'] = split
            result.append(instance)
    return result


def _identify(instances: list) -> list:
    for i,instance in enumerate(instances):
        instance[1].meta['id'] = i
    return instances


def shape_table(cfg: GenCfg, reserved: frozenset = frozenset()) -> list[dict]:
    """Enumerate every ordinary cell, drop higher-precedence canons, and fix each shape's owner globally."""
    cells,global_owner = [],{}
    for family,params in family_cells(cfg):
        cell = enumerate_cell(family,params,cfg)
        cell['reserved'] = sorted(shape for shape in cell['shapes'] if shape in reserved)
        cell['owners'] = [(shape,global_owner.setdefault(shape,owner)) for shape,owner in owners(cell,reserved)]
        for split in ('train','val','test_id'):
            assert owned_groups(cell,split), f'{family}{params} owns no {split} shape'
        cells.append(cell)
    return cells


def balanced(cells: list[dict], total: int, split: str, rng: random.Random, cfg: GenCfg,
             exclude: tuple[str,...] = (), tag: str | None = None) -> list[Instance]:
    """Spread a total evenly over families, then their cells, then the split's owned shapes."""
    families = sorted({cell['family'] for cell in cells})
    result = []
    for quota,family in zip(even_split(total,len(families)),families):
        members = [cell for cell in cells if cell['family'] == family]
        for count,cell in zip(even_split(quota,len(members)),members):
            result += emit_shapes(cell,owned_groups(cell,split,exclude),count,rng,cfg,tag or split)
    return result


BENCHMARK = ('train','val','test_id')


def benchmark_split(cells: list[dict], cfg: GenCfg, split: str) -> list[Instance]:
    """Emit exactly one benchmark population; each split draws only from its own seeded stream.

    Ownership is fixed globally by `shape_table`, so one split is byte-identical whether or not
    the others were generated. Phase 2 uses this to reach `val` without constructing `test_id`.
    """
    assert cfg.n_val % 4 == 0 and cfg.n_test % 4 == 0, 'per-family quotas must be exact'
    totals = dict(train=4*cfg.n_train_per_family,val=cfg.n_val,test_id=cfg.n_test)
    if split not in totals:
        raise ValueError(f'unknown benchmark split {split!r}; expected one of {BENCHMARK}')
    return _identify(balanced(cells,totals[split],split,random.Random(seed(split)),cfg))


def benchmark_splits(cells: list[dict], cfg: GenCfg) -> dict[str,list[Instance]]:
    """Emit the balanced train, val, and test_id populations from owned shapes."""
    return {split:benchmark_split(cells,cfg,split) for split in BENCHMARK}


def e4_set(cfg: GenCfg) -> list[Instance]:
    """Emit the held-out F1(3,3) population evenly over its canonical shapes."""
    cell = enumerate_cell('F1',(3,3),cfg)
    groups = [cell['shapes'][shape] for shape in sorted(cell['shapes'])]
    return _identify(emit_shapes(cell,groups,cfg.n_e4,random.Random(seed('test_e4')),cfg,'test_e4'))


def discovery_pools(cfg: GenCfg, reserved: frozenset) -> dict[str,dict]:
    """Reserve each discovery outer family's admissible canonical shapes after the test_comp exclusion."""
    pools = {}
    for family in ('F1','F2','F3'):
        cell = enumerate_cell(family,(2,2),cfg,True)
        cell['owned'] = [shape for shape in sorted(cell['shapes']) if shape not in reserved]
        cell['excluded'] = sorted(shape for shape in cell['shapes'] if shape in reserved)
        pools[family] = cell
    return pools


def discovery_set(cfg: GenCfg, pools: dict[str,dict]) -> list[Instance]:
    """Emit n_test/3 discovery instances per outer family, evenly over that family's reserved shapes."""
    assert cfg.n_test % 3 == 0, 'test_disc holds n_test / 3 instances per outer family'
    rng,result = random.Random(seed('test_disc')),[]
    for family in ('F1','F2','F3'):
        cell = pools[family]
        groups = [cell['shapes'][shape] for shape in cell['owned']]
        assert groups, f'{family} discovery cell owns no shape'
        result += emit_shapes(cell,groups,cfg.n_test//3,rng,cfg,'test_disc')
    return _identify(result)


def reserve_shapes(cfg: GenCfg) -> tuple[dict[str,dict],dict[str,dict],frozenset]:
    """Apply the frozen precedence test_comp > test_disc > val/test_id/train from generator structure only."""
    comp = composition_pools(cfg)
    reserved = frozenset(shape for pool in comp.values() for shape in pool['shapes'])
    disc = discovery_pools(cfg,reserved)
    return comp,disc,reserved | frozenset(shape for cell in disc.values() for shape in cell['owned'])


def generate_splits(cfg: GenCfg, calc: CalcCfg, pilot: PilotCfg, compute: ComputeCfg, ood: RandomOODCfg,
                    scale: ScaleCfg) -> tuple[dict[str,list],list[dict]]:
    """Emit every benchmark set under the declared cross-set precedence; return the E7b grid statuses too."""
    comp,disc,reserved = reserve_shapes(cfg)
    sets = benchmark_splits(shape_table(cfg,reserved),cfg)
    sets['test_comp'] = composition_set(cfg,comp)
    sets['test_disc'] = discovery_set(cfg,disc)
    sets['test_e4'] = e4_set(cfg)
    sets['e7a'] = e7a_set(cfg,calc)
    owned = {p.canon(state[1].target) for pair in sets['e7a'] for state in pair}
    sets['e7b'],statuses = e7b_set(cfg,calc,pilot,compute,owned)
    taken = {p.canon(t.target) for name in sets for t in set_tasks(name,sets[name])}
    sets['random_ood'] = random_ood_set(cfg,ood,taken)
    taken |= {p.canon(t.target) for t in set_tasks('random_ood',sets['random_ood'])}
    sets['scale_stress'] = scale_stress_set(cfg,scale,taken)
    taken |= {p.canon(t.target) for t in set_tasks('scale_stress',sets['scale_stress'])}
    sets['named_cases'] = named_cases(cfg,taken)
    return sets,statuses


def set_tasks(name: str, entries: list) -> list[Task]:
    """Read every task of one set, flattening E7a pairs."""
    if name == 'e7a':
        return [state[1] for pair in entries for state in pair]
    return [entry[1] for entry in entries]


def leakage_audit(sets: dict[str,list]) -> dict[str,int]:
    """Assert canonical disjointness and held-out structural-role rules; count canons per set."""
    owner: dict[tuple,str] = {}
    held = {t.meta['skeleton_signature'] for t in set_tasks('test_comp',sets.get('test_comp',[]))}
    for name,entries in sets.items():
        for task in set_tasks(name,entries):
            canonical = p.canon(task.target)
            assert owner.setdefault(canonical,name) == name, f'canon appears in {owner[canonical]} and {name}'
            meta = task.meta
            assert meta['composition_depth'] != 2 or name == 'test_comp', 'composition leakage'
            assert all(s in ATOMIC for s in meta['slot_shapes']) or name == 'test_disc', 'slot-role leakage'
            assert not (meta['family'] == 'F1' and meta['params'] == (3,3)) or name == 'test_e4', 'F1(3,3) leakage'
            if name == 'train':
                assert meta['skeleton_signature'] not in held, 'composition skeleton leakage'
                assert meta.get('embedded_signature') not in held, 'embedded composition leakage'
    return dict(Counter(owner.values()))


def e7a_pair(rng: random.Random) -> list[tuple[Arena,Task,int,Action]]:
    """Construct the specified useful/sham exposure pair with fresh variables."""
    permutation = rng.sample(range(5),5)
    result = []
    for useful in (True,False):
        arena = Arena()
        a,A,B,C,P = [mk_mono(arena,1 << (4*i)) for i in permutation]
        s = mk_sum(arena,Counter([A,B if useful else C]))
        aA,aB,aC = [mk_prod(arena,Counter([a,h])) for h in (A,B,C)]
        root = mk_sum(arena,Counter([aA,aB,aC,mk_prod(arena,Counter([s,P]))]))
        action = Action('FACTOR',root,tuple(sorted(((aA,1),(aB,1)))),a)
        alternative = Action('FACTOR',root,tuple(sorted(((aA,1),(aC,1)))),a)
        meta = dict(family='e7a',params=(),slot_shapes=('x',)*5,composition_depth=0,alternative_action=alternative,
                    embedded_family=None,skeleton_signature=skeleton_signature(arena,root),useful=useful)
        result.append((arena,Task(poly(arena,root),5,cost(arena,root),meta),root,action))
    return result


def e7a_margins(pair: list[tuple], cfg: CalcCfg) -> tuple[float,float]:
    """Evaluate every candidate with the exact three-step continuation margin."""
    from .experiments.pilot import bounded_best
    margins = []
    for arena,task,root,designated in pair:
        info,q = state_info(arena,root),{}
        for action in enumerate_actions(arena,info,root,cfg):
            successor = apply(arena,task,info,root,action)
            q[action] = float('inf') if successor is None else bounded_best(arena,task,successor,3,cfg)
        assert designated in q, 'designated E7a action must be enumerated'
        assert task.meta['alternative_action'] in q, 'matched E7a competitor must be enumerated'
        assert sum(value != float('inf') for value in q.values()) >= 2, 'E7a needs two legal candidates'
        margins.append(min(value for action,value in q.items() if action != designated)-q[designated])
    return tuple(margins)


def e7a_set(cfg: GenCfg, calc: CalcCfg) -> list[list[tuple]]:
    """Draw n_e7_pairs exposure pairs from the e7a stream and keep those passing the margin rule."""
    rng = random.Random(seed('e7a'))
    pairs = [e7a_pair(rng) for _ in range(cfg.n_e7_pairs)]
    margins = [e7a_margins(pair,calc) for pair in pairs]
    kept = [pair for pair,(useful,sham) in zip(pairs,margins) if useful-sham >= 1]
    for i,pair in enumerate(kept):
        pair[0][1].meta['id'] = pair[1][1].meta['id'] = i
    return kept


def _hidden_dag(rng: random.Random, n_vars: int, gates: int) -> list[tuple[str,int,int]]:
    """Draw one hidden scalar-free binary ADD/MUL proposal over the given variable leaves."""
    nodes,seen = [],set()
    while len(nodes) < gates:
        op = 'ADD' if rng.random() < 0.5 else 'MUL'
        size = n_vars+len(nodes)
        a,b = sorted((rng.randrange(size),rng.randrange(size)))
        if (op,a,b) not in seen:
            seen.add((op,a,b))
            nodes.append((op,a,b))
    return nodes


def _prune(nodes: list[tuple[str,int,int]], n_vars: int) -> list[tuple[str,int,int]]:
    """Keep only the gates reachable from the output, renumbered in the original order."""
    live = {n_vars+len(nodes)-1}
    for i in range(len(nodes)-1,-1,-1):
        if n_vars+i in live:
            live.update(nodes[i][1:])
    kept = [i for i in range(len(nodes)) if n_vars+i in live]
    index = {n_vars+i:n_vars+j for j,i in enumerate(kept)}
    return [(nodes[i][0],index.get(nodes[i][1],nodes[i][1]),index.get(nodes[i][2],nodes[i][2])) for i in kept]


def _dag_bounds(nodes: list[tuple[str,int,int]], n_vars: int) -> tuple[int,int]:
    """Return the exact expansion degree and occurrence count without expanding; nothing cancels."""
    degree,occurrences = [1]*n_vars,[1]*n_vars
    for op,a,b in nodes:
        degree.append(max(degree[a],degree[b]) if op == 'ADD' else degree[a]+degree[b])
        occurrences.append(occurrences[a]+occurrences[b] if op == 'ADD' else occurrences[a]*occurrences[b])
    return degree[-1],occurrences[-1]


def _dag_poly(nodes: list[tuple[str,int,int]], n_vars: int) -> p.Poly:
    """Expand a hidden DAG exactly."""
    value = [{1 << (4*i):1} for i in range(n_vars)]
    for op,a,b in nodes:
        value.append(p.add(value[a],value[b]) if op == 'ADD' else p.mul(value[a],value[b]))
    return value[-1]


def dag_digest(nodes: list[tuple[str,int,int]], n_vars: int) -> str:
    """Hash a pruned hidden DAG; it is generator metadata and never reaches search or the model."""
    return hashlib.sha256(repr((n_vars,nodes)).encode()).hexdigest()


def _hidden_task(name: str, tag: str, j: int, widths: tuple[int,...], gates: tuple[int,int],
                 cfg: GenCfg) -> Instance | None:
    """Propose one hidden-DAG task and return it only if every structural acceptance rule holds."""
    rng = random.Random(seed(f'{tag}|{j}'))
    n_vars = widths[rng.randrange(len(widths))]
    nodes = _prune(_hidden_dag(rng,n_vars,rng.randrange(gates[0],gates[1]+1)),n_vars)
    c_gen = len(nodes)
    fanout: Counter[int] = Counter()
    for op,a,b in nodes:
        fanout[a] += 1
        fanout[b] += 1
    if not gates[0] <= c_gen <= gates[1] or not any(fanout[n_vars+i] > 1 for i in range(c_gen)):
        return None
    degree,occurrences = _dag_bounds(nodes,n_vars)
    if degree > cfg.max_degree or occurrences > cfg.max_occurrences:
        return None
    target = _dag_poly(nodes,n_vars)
    arena = Arena()
    start = canonical_start(arena,target)
    if cost(arena,start) <= c_gen:
        return None
    meta = dict(family=name,params=(),slot_shapes=(),composition_depth=0,embedded_family=None,
                skeleton_signature=None,set=name,proposal=j,hidden_gates=nodes,hidden_leaves=n_vars,
                hidden_dag_sha256=dag_digest(nodes,n_vars),occurrences=sum(target.values()),
                distinct=len(target),start_cost=cost(arena,start))
    task = Task(target,len(p.support(target)),c_gen,meta)
    return arena,task,start


def random_ood_set(cfg: GenCfg, ood: RandomOODCfg, taken: set[tuple]) -> list[Instance]:
    """Accept hidden-DAG proposals in id order until n_tasks distinct canonical targets are frozen."""
    result,seen = [],set(taken)
    for j in range(100*ood.n_tasks):
        instance = _hidden_task('random_ood','random_ood',j,(3,4,5,6),(ood.min_gates,ood.max_gates),cfg)
        if instance is None or size_rejection(instance[1],cfg):
            continue
        shape = p.canon(instance[1].target)
        if shape in seen:
            continue
        seen.add(shape)
        result.append(instance)
        if len(result) == ood.n_tasks:
            return _identify(result)
    raise ValueError(f'random_ood accepted only {len(result)} of {ood.n_tasks} tasks')


def scale_stress_set(cfg: GenCfg, scale: ScaleCfg, taken: set[tuple]) -> list[Instance]:
    """Fill every declared occurrence bucket with hidden-DAG tasks beyond the controlled size regime."""
    result,seen = [],set(taken)
    for lo,hi in scale.occurrence_buckets:
        # The bucket filter is tighter than any bypassed §10.5 occurrence cap, so hi bounds the expansion.
        wide,kept = replace(cfg,max_occurrences=hi),0
        for j in range(2000*scale.n_per_bucket):
            instance = _hidden_task('scale_stress',f'scale_stress|{lo}|{hi}',j,(3,4,5,6,7,8),
                                    (scale.min_gates,scale.max_gates),wide)
            if instance is None or instance[1].n_vars > cfg.max_vars:
                continue
            task = instance[1]
            if not lo <= task.meta['occurrences'] <= hi or task.meta['distinct'] > hi:
                continue
            shape = p.canon(task.target)
            if shape in seen:
                continue
            seen.add(shape)
            task.meta['bucket'] = [lo,hi]
            result.append(instance)
            kept += 1
            if kept == scale.n_per_bucket:
                break
        if kept < scale.n_per_bucket:
            raise ValueError(f'scale_stress bucket {(lo,hi)} accepted only {kept} of {scale.n_per_bucket}')
    return _identify(result)


def _spans_k4(triple: tuple[int,int,int]) -> bool:
    """Report whether three K_4 edges form a spanning tree, by union-find over four vertices."""
    edges,group = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)],list(range(4))
    for i in triple:
        a,b = (group[v] for v in edges[i])
        if a == b:
            return False
        group = [b if g == a else g for g in group]
    return True


def _kirchhoff_k4() -> p.Poly:
    """Enumerate the sixteen spanning trees of K_4 over its six edge variables."""
    trees = [t for t in combinations(range(6),3) if _spans_k4(t)]
    assert len(trees) == 16, 'K_4 has sixteen spanning trees'
    return {p.pack(tuple(1 if i in t else 0 for i in range(6))):1 for t in trees}


def named_cases(cfg: GenCfg, taken: set[tuple] = frozenset()) -> list[Instance]:
    """Build the fixed seventeen-instance classical suite by exact combinatorial enumeration."""
    specs = [('e',n,k) for n in (4,5,6,7,8) for k in (2,3)]+[('h',n,k) for n in (3,4,5) for k in (2,3)]
    result = []
    for kind,n,k in specs+[('K4',6,3)]:
        if kind == 'K4':
            target = _kirchhoff_k4()
        else:
            picks = combinations(range(n),k) if kind == 'e' else combinations_with_replacement(range(n),k)
            target = {}
            for pick in picks:
                target = p.add(target,{p.pack(tuple(pick.count(i) for i in range(n))):1})
        assert n <= cfg.max_vars and p.degree(target) <= cfg.max_degree, 'named case fits the declared caps'
        assert sum(target.values()) <= cfg.max_occurrences, 'named case fits the occurrence cap'
        assert p.canon(target) not in taken, f'named case {kind}_{k} on {n} variables collides with another set'
        arena = Arena()
        start = canonical_start(arena,target)
        meta = dict(family=kind,params=(n,k),slot_shapes=(),composition_depth=0,embedded_family=None,
                    skeleton_signature=None,set='named_cases',has_generator_reference=False,
                    distinct=len(target),occurrences=sum(target.values()),start_cost=cost(arena,start),
                    target_sha256=hashlib.sha256(repr(sorted(target.items())).encode()).hexdigest())
        result.append((arena,Task(target,n,0,meta),start))
    return _identify(result)


def e7b_instance(embedded_family: str, shapes: tuple[str,...], d: int, x_index: int,
                  rng: random.Random | None, cfg: GenCfg) -> tuple[Arena,Task,int,int,int,int]:
    """Construct a specified motif and blocking variable from one explicit grid point."""
    assert embedded_family in ('F1','F2') and d in (0,1) and all(s in ATOMIC for s in shapes), 'E7b grid'
    arena = Arena()
    slots,extra,permutation = _allocate(arena,shapes,rng,1+2*d,cfg)
    motif = _family(arena,embedded_family,(2,2),slots)
    variables = permutation[:len(permutation)-1-2*d]
    assert x_index in range(len(variables)), 'blocking variable outside motif'
    x = mk_mono(arena,1 << (4*variables[x_index]))
    xu = mk_mono(arena,arena.handles[x].m+(1 << (4*extra[0])))
    terms = [motif,xu]+[mk_mono(arena,(1 << (4*extra[1]))+(1 << (4*extra[2]))) for _ in range(d)]
    gen = mk_sum(arena,Counter(terms))
    meta = dict(family='e7b',params=(d,),slot_shapes=shapes,composition_depth=0,x_index=x_index,
                embedded_family=embedded_family,skeleton_signature=skeleton_signature(arena,gen))
    task = Task(poly(arena,gen),len(variables)+1+2*d,cost(arena,gen),meta)
    assert size_rejection(task,cfg) is None, 'E7b grid points fit the ordinary size filter'
    return arena,task,canonical_start(arena,task.target),motif,x,xu


def e7b_pair(instance: tuple, calc: CalcCfg, pilot: PilotCfg) -> dict:
    """Evaluate the exact E7b keep rule; a time cap yields uncovered, never a proved rejection."""
    from .experiments.pilot import exhaustive_trajectory,bounded_best
    arena,task,start,motif,x,xu = instance
    motif_counts = {mk_mono(arena,m):c for m,c in poly(arena,motif).items()}
    info = state_info(arena,start)
    choices, began = [],time.perf_counter()
    unbounded = replace(calc,r_max=len(items(arena.handles[start])),n_max=sys.maxsize)
    for action in enumerate_actions(arena,info,start,unbounded):
        if any(c > motif_counts.get(h,0) for h,c in action.region):
            continue
        successor = apply(arena,task,info,start,action)
        if successor is None:
            continue
        remaining = pilot.exhaustive_time_s-(time.perf_counter()-began)
        continuation = replace(pilot,exhaustive_max_depth=pilot.exhaustive_max_depth-1,exhaustive_time_s=max(0,remaining))
        path = exhaustive_trajectory(arena,task,successor,continuation)
        if not path['covered']:
            return dict(status='uncovered',reason='exhaustive continuation hit the time cap')
        if path['reached']:
            choices.append((len(path['actions']),len(choices),successor,action))
    if not choices:
        return dict(status='rejected',reason='no motif-owned trajectory reaches C_gen')
    length,order,useful,useful_action = min(choices)
    xm = arena.handles[x].m
    m = min((m for m in poly(arena,motif) if p.divides(xm,m)),key=lambda m:(poly(arena,motif)[m],m))
    region = tuple(sorted(((mk_mono(arena,m),1),(xu,1))))
    sham_action = Action('FACTOR',start,region,x)
    sham = apply(arena,task,info,start,sham_action)
    if sham is None or abs(cost(arena,useful)-cost(arena,sham)) > 1:
        return dict(status='rejected',reason='sham state missing or cost gap above 1')
    deadline = began+pilot.exhaustive_time_s
    useful_best = bounded_best(arena,task,useful,3,calc,deadline)
    sham_best = bounded_best(arena,task,sham,3,calc,deadline)
    if useful_best is None or sham_best is None:
        return dict(status='uncovered',reason='three-step continuation hit the time cap')
    if useful_best > task.c_gen or sham_best <= task.c_gen:
        return dict(status='rejected',reason='three-step continuation does not separate useful from sham')
    return dict(status='accepted',useful=useful,sham=sham,useful_action=useful_action,sham_action=sham_action)


def e7b_grid(cfg: GenCfg) -> list[tuple]:
    """List every admissible E7b grid point in lexicographic order."""
    grid = []
    for embedded,width in (('F1',2),('F2',4)):
        for shapes in product(ATOMIC,repeat=width):
            n = sum(WIDTH[s] for s in shapes)
            grid.extend((embedded,shapes,d,i) for d in (0,1) if n+1+2*d <= cfg.max_vars for i in range(n))
    return grid


def _e7b_evaluate(payload: tuple) -> tuple[tuple,dict]:
    instance,calc,pilot = payload
    return instance,e7b_pair(instance,calc,pilot)


def e7b_set(cfg: GenCfg, calc: CalcCfg, pilot: PilotCfg, compute: ComputeCfg,
            exclude: set[tuple] = frozenset()) -> tuple[list[tuple],list[dict]]:
    """Draw shuffled grid points, skip classes owned elsewhere, and keep accepted distinct shapes (with their actions) and statuses."""
    rng = random.Random(seed('e7b'))
    grid = e7b_grid(cfg)
    rng.shuffle(grid)
    instances = [e7b_instance(embedded,shapes,d,i,rng,cfg) for embedded,shapes,d,i in grid]
    canons = [p.canon(instance[1].target) for instance in instances]
    classes = {c:i for i,c in enumerate(sorted(set(canons)))}
    eligible = [i for i,c in enumerate(canons) if c not in exclude]
    sized,sizing = sized_compute(compute,'e7b_set')
    evaluated = dict(zip(eligible,run_tasks(_e7b_evaluate,[(i,(instances[i],calc,pilot)) for i in eligible],sized)))
    kept,seen,statuses = [],set(),[]
    for i,(point,canonical) in enumerate(zip(grid,canons)):
        row = dict(order=i,embedded_family=point[0],slot_shapes=point[1],d=point[2],x_index=point[3],
                   canonical_class=classes[canonical],status='skipped',reason='canonical class owned by e7a',kept=False)
        if i in evaluated:
            instance,result = evaluated[i]
            row.update(status=result['status'],reason=result.get('reason',''))
            if result['status'] == 'accepted' and canonical not in seen and len(seen) < cfg.n_e7_pairs:
                seen.add(canonical)
                instance[1].meta.update(id=len(kept),set='e7b',status='accepted')
                kept.append(instance+(result['useful'],result['sham'],result['useful_action'],result['sham_action']))
                row.update(kept=True,useful_cost=cost(instance[0],result['useful']),sham_cost=cost(instance[0],result['sham']))
            elif result['status'] == 'accepted':
                row['reason'] = 'accepted duplicate of a kept canonical class'
        statuses.append(row)
    return kept,statuses
