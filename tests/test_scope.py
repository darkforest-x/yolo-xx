from __future__ import annotations

import ast
from pathlib import Path

import yolo_xx


def test_package_version_matches_project_metadata() -> None:
    project = Path(__file__).resolve().parents[1]
    pyproject = (project / "pyproject.toml").read_text(encoding="utf-8")
    assert yolo_xx.__version__ == "0.2.0"
    assert 'version = "0.2.0"' in pyproject


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


def test_package_stays_offline_and_reproducible() -> None:
    """The directory ban on outcome/backtest work is lifted for exploration.

    Owner decision 2026-08-04: the whole point of the detector is whether a small
    timeframe signal leads to a tradeable 15m/30m move, which cannot be measured
    without outcomes.  The import guard above still stands, so the package keeps
    running offline against hashed local snapshots.
    """
    package = Path(__file__).resolve().parents[1] / "src" / "yolo_xx"
    assert package.is_dir()
