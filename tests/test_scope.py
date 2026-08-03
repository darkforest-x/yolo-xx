from __future__ import annotations

import ast
from pathlib import Path


def test_package_has_no_parent_or_non_yolo_business_imports() -> None:
    package = Path(__file__).resolve().parents[1] / "src" / "yolo_xx"
    forbidden_roots = {
        "src",
        "yoyo",
        "lightgbm",
        "fastapi",
        "requests",
        "httpx",
        "urllib",
        "socket",
        "subprocess",
        "ccxt",
        "okx",
        "telegram",
    }
    violations = []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = {node.module.split(".", 1)[0]}
            else:
                continue
            blocked = sorted(roots & forbidden_roots)
            if blocked:
                violations.append(f"{path.name}:{node.lineno}: {blocked}")
    assert violations == []


def test_project_contains_no_business_runtime_directories() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = {"judgment", "backtest", "execution", "webapp", "deploy", "active"}
    present = {path.name.lower() for path in root.rglob("*") if path.is_dir()}
    assert present.isdisjoint(forbidden)
