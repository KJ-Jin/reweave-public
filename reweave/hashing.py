"""Canonical JSON, file hashes, and semantic (comment/docstring-free) source hashes.
"""

import ast
import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path


def _plain(value):
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(v) for v in value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value) -> str:
    """Key-sorted, whitespace-free JSON; tuples become lists, bytes hex."""
    return json.dumps(_plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha256_json(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path, relpaths: list[str]) -> str:
    """Hash sorted (relative path, content hash) pairs."""
    digest = hashlib.sha256()
    for rel in sorted(relpaths):
        digest.update(rel.encode()+b'\0'+file_sha256(root/rel).encode()+b'\n')
    return digest.hexdigest()


class _StripDocstrings(ast.NodeTransformer):
    def _strip(self, node):
        self.generic_visit(node)
        body = getattr(node, 'body', None)
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
        return node

    visit_Module = visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _strip


def semantic_source_hash(paths: list[str | Path]) -> str:
    """Hash Python sources by AST with docstrings removed; comments and formatting never count."""
    digest = hashlib.sha256()
    for path in sorted(str(p) for p in paths):
        tree = _StripDocstrings().visit(ast.parse(Path(path).read_text()))
        digest.update(Path(path).name.encode()+b'\0'+ast.dump(tree, annotate_fields=True,
                                                               include_attributes=False).encode()+b'\n')
    return digest.hexdigest()


def short(h: str, n: int = 12) -> str:
    return h[:n]


def semantic_definitions_hash(path: str | Path, names: list[str] | None) -> str:
    """Hash selected top-level definitions (functions, classes, assignments) of one module by docstring-free AST.

    `names=None` hashes the whole module (equivalent to `semantic_source_hash([path])`). A missing name fails loudly.
    """
    if names is None:
        return semantic_source_hash([path])
    tree = _StripDocstrings().visit(ast.parse(Path(path).read_text()))
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for leaf in ast.walk(target):
                    if isinstance(leaf, ast.Name):
                        found[leaf.id] = node
    missing = [n for n in names if n not in found]
    if missing:
        raise KeyError(f'{Path(path).name} has no top-level definitions {missing}')
    # Transitive closure over module-level names the selected definitions reference (globals, helpers, constants).
    closure, todo = set(), list(names)
    while todo:
        name = todo.pop()
        if name in closure:
            continue
        closure.add(name)
        for leaf in ast.walk(found[name]):
            if isinstance(leaf, ast.Name) and leaf.id in found and leaf.id not in closure:
                todo.append(leaf.id)
    digest = hashlib.sha256()
    for name in sorted(closure):
        digest.update(name.encode()+b'\0'+ast.dump(found[name], include_attributes=False).encode()+b'\n')
    return digest.hexdigest()
