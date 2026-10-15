"""Declared configuration loading, hash-bound calibration approval, comparator selection, and provenance."""

from dataclasses import asdict, fields, is_dataclass
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

from .config import (CalcCfg,SearchCfg,GenCfg,TrainCfg,StatsCfg,PilotCfg,OracleCfg,EGraphCfg,
                     RandomOODCfg,FormCfg,ScaleCfg,NamedCfg,ComputeCfg,ProtocolCfg)


def _configs() -> dict:
    return dict(calc=CalcCfg(),search=SearchCfg(),gen=GenCfg(),train=TrainCfg(),stats=StatsCfg(),
                pilot=PilotCfg(),oracle=OracleCfg(),egraph=EGraphCfg(),random_ood=RandomOODCfg(),
                form=FormCfg(),scale=ScaleCfg(),named=NamedCfg(),compute=ComputeCfg(),protocol=ProtocolCfg())


def defaults() -> dict:
    """Serialize precisely the section 9 defaults."""
    return {name:asdict(cfg) for name,cfg in _configs().items()}


def _override(instance: object, values: dict) -> None:
    if not isinstance(values,dict):
        raise ValueError('configuration sections must be objects')
    known = {f.name for f in fields(instance)}
    if values.keys()-known:
        raise ValueError(f'unknown configuration fields: {sorted(values.keys()-known)}')
    for name,value in values.items():
        current = getattr(instance,name)
        if is_dataclass(current):
            _override(current,value)
            continue
        if isinstance(current,tuple):
            if not isinstance(value,(list,tuple)):
                raise ValueError(f'{name} must be an array')
            value = tuple(value)
        elif current is not None and not isinstance(value,type(current)):
            if not (isinstance(current,float) and type(value) is int):
                raise ValueError(f'{name} has an invalid type')
        if type(current) is int and type(value) is bool:
            raise ValueError(f'{name} must be an integer')
        setattr(instance,name,value)


def load(path: str | Path = 'frozen.json') -> dict:
    """Load named configuration overrides and reject unknown fields."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'configuration file does not exist: {path}')
    values = json.loads(path.read_text())
    cfg = _configs()
    if not isinstance(values,dict) or values.keys()-cfg.keys():
        raise ValueError('unknown configuration sections')
    for name,overrides in values.items():
        _override(cfg[name],overrides)
    return cfg


def approve_calibration(candidate: str | Path, approvals: str | Path,
                        destination: str | Path = 'frozen.json') -> None:
    """Copy a calibration candidate atomically only after exact hash approval."""
    candidate,approvals,destination = Path(candidate),Path(approvals),Path(destination)
    load(candidate)
    data = candidate.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if not approvals.is_file() or f'CALIBRATION {digest} APPROVE' not in approvals.read_text().splitlines():
        raise ValueError(f'Gate A requires CALIBRATION {digest} APPROVE')
    temporary = destination.with_suffix(destination.suffix+'.tmp')
    temporary.write_bytes(data)
    temporary.replace(destination)


def modelval(task_id: int) -> bool:
    """Assign every training task id congruent to 9 modulo 10 to internal model validation."""
    return task_id % 10 == 9


def select_comparator(val_solve_rates: dict[str,float], tie: str) -> str:
    """Pick the highest benchmark-val solve rate; an exact tie goes to the declared tie winner."""
    if tie not in val_solve_rates:
        raise ValueError(f'tie winner {tie!r} is not a candidate')
    best = max(val_solve_rates.values())
    winners = sorted(name for name,rate in val_solve_rates.items() if rate == best)
    return tie if tie in winners else winners[0] if len(winners) == 1 else _first_declared(winners)


def _first_declared(winners: list[str]) -> str:
    raise ValueError(f'validation tie among {winners} has no declared winner')


# Section 11 requires the B3 crate's Cargo.lock to enter the protocol hash; the rule manifest,
# crate source, dependency stanza and declared build recipe are hashed with it. Build outputs
# under tools/egraph_baseline/target are not source and stay excluded.
#
# slurm/** is launch and build configuration: it sets PYTHONPATH, the conda environment and the
# default POLYCOMPRESS_FORM / POLYCOMPRESS_EGRAPH paths, so it decides which interpreter and which
# baseline executables a run actually uses. Editing it materially changes execution, so it is
# covered by the source hash and the dirty check.
SOURCE_GLOBS = ('polycompress/**/*.py','tests/**/*.py','IMPLEMENTATION_SPEC.md','QUESTIONS.md','DEVIATIONS.md',
                'APPROVALS.md','pytest.ini','.gitignore',
                'tools/egraph_baseline/Cargo.toml','tools/egraph_baseline/Cargo.lock',
                'tools/egraph_baseline/rules.txt','tools/egraph_baseline/BUILD.json',
                'tools/egraph_baseline/src/**/*.rs',
                'slurm/**/*.sh','slurm/**/*.py','slurm/**/*.sbatch')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files(root: Path) -> list[Path]:
    """List the tracked-source/config files that the source-tree hash covers (outputs excluded)."""
    return sorted({path for pattern in SOURCE_GLOBS for path in root.glob(pattern) if path.is_file()})


def source_tree_hash(root: Path) -> str:
    """Hash relative paths and contents of every source/config file in sorted order."""
    digest = hashlib.sha256()
    for path in source_files(root):
        digest.update(str(path.relative_to(root)).encode()+b'\0'+_sha256(path).encode()+b'\n')
    return digest.hexdigest()


def git_state(root: Path) -> dict:
    """Read the commit and whether any source/config file is modified or untracked; refuse outside git."""
    def run(*args: str) -> str:
        result = subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True)
        if result.returncode != 0:
            raise ValueError(f'git {args[0]} failed: {result.stderr.strip() or "not a git repository"}')
        return result.stdout
    commit = run('rev-parse','HEAD').strip()
    files = [str(path.relative_to(root)) for path in source_files(root)]
    status = [line for line in run('status','--porcelain','--untracked-files=all','--',
                                   *(f':(glob){pattern}' for pattern in SOURCE_GLOBS)).splitlines() if line]
    return dict(commit=commit,dirty=bool(status),dirty_files=[line[3:] for line in status],source_files=len(files))


# Environment variables that select which B3/B4 executable actually runs, and the Slurm fragment
# that supplies their defaults. They are not scientific source, but an override here changes which
# binary produces a baseline number, so provenance has to move with them.
BASELINE_OVERRIDES = ('POLYCOMPRESS_FORM','POLYCOMPRESS_EGRAPH')
BASELINE_LAUNCHER = Path('slurm/common.sh')
# A manifest may claim its recorded build or acquisition history only when it says so explicitly.
# Hash equality alone means "this manifest describes these bytes", never "these bytes were built
# or acquired the recorded way" -- a manifest generated while an override was active matches its
# own override by construction.
VERIFIED_PROVENANCE = ('default_build_output','unpacked_release')


def _identity(path: str | Path | None) -> dict:
    """Hash an executable's actual bytes, or say plainly that it is absent."""
    if path is None:
        return dict(path=None,sha256=None,present=False)
    resolved = Path(path)
    if not resolved.is_file():
        return dict(path=str(resolved),sha256=None,present=False)
    return dict(path=str(resolved.resolve()),sha256=_sha256(resolved),present=True)


def selected_executable(kind: str, manifest: dict) -> tuple[str | None, str | None]:
    """Resolve the executable a run would actually use, honouring the same override the wrapper does.

    The wrappers read POLYCOMPRESS_FORM / POLYCOMPRESS_EGRAPH before falling back to their default,
    so provenance has to resolve the override first too. Hashing only the path recorded in the
    manifest would describe a binary that never ran.
    """
    import os
    variable = 'POLYCOMPRESS_FORM' if kind == 'b4' else 'POLYCOMPRESS_EGRAPH'
    override = os.environ.get(variable)
    return override or manifest.get('executable'), override


def baseline_identity(root: Path) -> dict:
    """Bind the B3/B4 manifests, the executable a run would actually select, and the overrides.

    Section 11 locks each manifest before a test set is read; this ties that manifest, the binary
    that would really run and the environment and launcher that choose it into one record, so an
    override or a changed binary at the same path changes provenance. The manifest's build and
    acquisition history describes the default executable only: when the selected binary's bytes do
    not match the hash the manifest recorded, that history is explicitly not claimed for it.
    """
    import os
    record: dict = dict(overrides={name:os.environ.get(name) for name in BASELINE_OVERRIDES})
    launcher = root/BASELINE_LAUNCHER
    record['launcher'] = dict(path=str(BASELINE_LAUNCHER),
                              sha256=_sha256(launcher) if launcher.is_file() else None)
    for name in ('b3','b4'):
        path = root/'out'/'baselines'/f'{name}_manifest.json'
        entry: dict = dict(manifest_sha256=_sha256(path) if path.is_file() else None)
        stored = json.loads(path.read_text()) if path.is_file() else {}
        recorded = stored.get('sha256') or stored.get('executable_sha256')
        selected,override = selected_executable(name,stored)
        identity = _identity(selected)
        claimed = stored.get('executable_provenance')
        status = claimed.get('status') if isinstance(claimed,dict) else None
        entry.update(manifest_executable=stored.get('executable'),
                     manifest_executable_sha256=recorded,
                     selected_executable=identity,
                     selected_by_override=override is not None,
                     matches_manifest=bool(recorded) and identity['sha256'] == recorded,
                     executable_is_committed=False,
                     manifest_provenance_status=status)
        # Both are required: the manifest must claim a verified origin, and its recorded hash must
        # still describe the bytes that would run. An unrecognized or absent status fails closed.
        entry['manifest_provenance_applies'] = bool(status in VERIFIED_PROVENANCE and entry['matches_manifest'])
        if name == 'b3':
            entry['cargo_lock_sha256'] = stored.get('cargo_lock_sha256')
            entry['rule_manifest_sha256'] = stored.get('rule_manifest_sha256')
        else:
            entry['form_cfg'] = stored.get('form_cfg')
            entry['archive_sha256'] = (stored.get('acquisition') or {}).get('archive_sha256')
        record[name] = entry
    return record


def environment() -> dict:
    """Name the interpreter and the declared Python dependencies a run actually imported (section 18.6)."""
    import platform
    import numpy
    import sympy
    import torch
    return dict(python=platform.python_version(),numpy=numpy.__version__,torch=torch.__version__,
                sympy=sympy.__version__)


def _label(root: Path, path: Path) -> str:
    """Name an artifact relative to the repository, or by absolute path when it lives elsewhere."""
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def provenance(root: Path, artifacts: list[Path], parents: dict[str,Path] | None = None,
               cfg: dict | None = None, run: dict | None = None) -> dict:
    """Assemble commit, dirty state, hashes, resolved config, seeds, command, environment, parents and artifacts."""
    import os
    import sys
    cfg = cfg or _configs()
    root = Path(root).resolve()
    artifacts = [Path(path).resolve() for path in artifacts]
    record = dict(git=git_state(root),source_tree_sha256=source_tree_hash(root),
                  spec_sha256=_sha256(root/'IMPLEMENTATION_SPEC.md'),
                  config_defaults_sha256=hashlib.sha256((json.dumps(defaults(),sort_keys=True,indent=2)+'\n').encode()).hexdigest(),
                  frozen_candidate_sha256=_sha256(root/'frozen_candidate.json') if (root/'frozen_candidate.json').is_file() else None,
                  frozen_sha256=_sha256(root/'frozen.json') if (root/'frozen.json').is_file() else None,
                  seeds=dict(gen_seed_test=cfg['protocol'].gen_seed_test,train_seeds=list(cfg['train'].seeds),
                             stats_rng=0,tag_rule='seed(tag) = uint64_be(SHA256("PolyCompress|<gen_seed_test>|" + tag)[:8])'),
                  baselines=baseline_identity(root),
                  resolved_config={name:asdict(section) for name,section in cfg.items()},
                  run=run or {},
                  command=dict(argv=list(sys.argv),executable=sys.executable,cwd=os.getcwd()),
                  environment=environment(),
                  parent_checkpoints={name:dict(path=_label(root,Path(path).resolve()),sha256=_sha256(Path(path)))
                                      for name,path in (parents or {}).items()},
                  artifacts={_label(root,path):_sha256(path) for path in sorted(artifacts)},
                  created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'))
    return record


def write_provenance(root: Path, destination: Path, artifacts: list[Path], parents: dict[str,Path] | None = None,
                     cfg: dict | None = None, run: dict | None = None) -> dict:
    """Write the provenance record next to the artifacts it describes."""
    record = provenance(root,artifacts,parents,cfg,run)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(record,indent=1,sort_keys=True)+'\n')
    return record
