"""Semantic-dependency artifact keys and an atomic, resumable artifact store.

Every expensive artifact lives at `<root>/<kind>/<name>-<key12>/` with `key.json` (the full dependency dict) and a
`DONE` marker written last. A key is the SHA-256 of the dependency dict. Code dependencies enter as docstring-free AST
hashes of exactly the definitions that compute the artifact, so documentation-only edits never invalidate a key.
"""

import json
import os
from pathlib import Path

import torch

from .hashing import sha256_json

def artifact_key(deps: dict) -> str:
    return sha256_json(deps)


class Store:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, kind: str, name: str, deps: dict) -> Path:
        return self.root/kind/f'{name}-{artifact_key(deps)[:12]}'

    def prepare(self, kind: str, name: str, deps: dict) -> Path:
        p = self.path(kind, name, deps)
        p.mkdir(parents=True, exist_ok=True)
        key_file = p/'key.json'
        body = dict(key=artifact_key(deps), deps=deps)
        if key_file.is_file():
            if json.loads(key_file.read_text())['key'] != body['key']:
                raise RuntimeError(f'artifact directory collision at {p}')
        else:
            tmp = key_file.with_suffix('.tmp')
            tmp.write_text(json.dumps(body, indent=1, sort_keys=True)+'\n')
            os.replace(tmp, key_file)
        return p

    @staticmethod
    def done(p: Path) -> bool:
        return (p/'DONE').is_file()

    @staticmethod
    def finish(p: Path, summary: dict | None = None) -> None:
        tmp = p/'DONE.tmp'
        tmp.write_text(json.dumps(summary or {}, indent=1, sort_keys=True, default=str)+'\n')
        os.replace(tmp, p/'DONE')


def state_dict_hash(state_dict: dict) -> str:
    """Semantic checkpoint hash over tensor names, shapes, dtypes and bytes (independent of pickle framing)."""
    import hashlib
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        t = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode()+b'\0'+str(tuple(t.shape)).encode()+str(t.dtype).encode()+b'\0')
        digest.update(t.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_hash(path: str | Path) -> str:
    blob = torch.load(str(path), map_location='cpu', weights_only=False)
    return state_dict_hash(blob['state_dict'])


def atomic_write_json(path: Path, body) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(body, indent=1, sort_keys=True, default=str)+'\n')
    os.replace(tmp, path)
