"""The legacy readers (ephysalis.legacy) can be removed without touching the rest: nothing outside
ephysalis.legacy and tests/legacy imports it, and no test outside tests/legacy uses its files."""

import ast
from pathlib import Path

SRC = Path(__file__).parents[1] / "src" / "ephysalis"
TESTS = Path(__file__).parent


def _imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_core_does_not_import_legacy():
    offenders = {str(p.relative_to(SRC)): sorted(n for n in _imports(p) if n.startswith("ephysalis.legacy"))
                 for p in SRC.rglob("*.py") if "legacy" not in p.relative_to(SRC).parts}
    assert {k: v for k, v in offenders.items() if v} == {}


def test_tests_outside_legacy_do_not_use_it():
    legacy_tests = {p.stem for p in (TESTS / "legacy").glob("test_*.py")}
    offenders = {}
    for path in TESTS.glob("*.py"):
        used = {n for n in _imports(path) if n.startswith("ephysalis.legacy") or n in legacy_tests}
        if used:
            offenders[path.name] = sorted(used)
    assert offenders == {}
