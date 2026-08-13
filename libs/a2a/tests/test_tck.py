"""The official A2A TCK, as a gate.

Skipped unless a checkout is available, because it clones a repository and
builds a second virtualenv — right for CI and a deliberate local run, wrong for
every `make test`.

```bash
make tck                                          # clone the pin and run it
A2A_TCK_PATH=../a2a-tck uv run pytest tests/test_tck.py -m tck
```

The gate is not "everything passes". Five requirements fail, each recorded below
with its cause and the party that owns the fix; the check is that nothing *else*
fails, and that none of the five starts passing without the note being removed —
which is how `PUSH-DELIVER-001` left this list: the SDK's sender ignores the
credentials a caller registers, so this package ships one that does not.

A bare pass/fail assertion here would have to be either permanently red or
quietly deleted, and neither is a gate.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.tck.run import checkout

pytestmark = pytest.mark.tck

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

UPSTREAM = "a2a-sdk"
"""Owner of a failure whose fix is not in this package."""

DESIGN = "this package (deliberate)"
"""Owner of a failure that follows from a decision, not a defect."""

TCK_SUITE = "a2a-tck"
"""Owner of a failure caused by the suite's own behaviour."""

KNOWN_MUST_FAILURES = {
    "DM-MSG-001": (
        DESIGN,
        "Requires answering with a bare Message instead of a Task. This "
        "package always opens a task: results are artifacts, and a task is "
        "what survives a restart, a resume and a cancellation.",
    ),
    "JSONRPC-SSE-002": (
        UPSTREAM,
        "A request with `Content-Type: text/plain` should be refused with "
        "ContentTypeNotSupportedError (-32005). `a2a-sdk` 1.1.2's JSON-RPC "
        "dispatcher parses the body before looking at the content type, so it "
        "answers ParseError (-32700).",
    ),
}

KNOWN_SHOULD_FAILURES = {
    "CORE-HIST-005": (TCK_SUITE, "see CORE-HIST-006"),
    "CORE-HIST-006": (
        TCK_SUITE,
        "The TCK's `tck_id(name)` is stable for a session, so its multi-turn "
        "history scenario sends several turns under one `messageId`. "
        "`a2a-sdk` 1.1.2 de-duplicates `Task.history` by `messageId`, so the "
        "repeats are dropped. Reproduced directly against the server.",
    ),
    "DM-SERIAL-005": (
        UPSTREAM,
        "Unrecognised request fields should be ignored. `a2a-sdk` 1.1.2 parses "
        "strictly and answers InvalidParams.",
    ),
}


def _available() -> bool:
    return bool(os.environ.get("A2A_TCK_PATH")) or (PACKAGE_ROOT / ".tck").exists()


def _report(tck_root: Path) -> dict:
    path = tck_root / "reports" / "compatibility.json"
    assert path.exists(), f"the TCK produced no report at {path}"
    return json.loads(path.read_text())


def _failing(report: dict, level: str) -> set[str]:
    return {
        requirement_id
        for requirement_id, result in report["per_requirement"].items()
        if result.get("level") == level and result.get("status") == "FAIL"
    }


@pytest.fixture(scope="module")
def tck_report() -> dict:
    if not _available():
        pytest.skip("no TCK checkout: set A2A_TCK_PATH, or run `make tck`")

    tck_root = checkout(PACKAGE_ROOT)
    subprocess.run(
        [sys.executable, "-m", "tests.tck.run"],
        cwd=PACKAGE_ROOT,
        check=False,
        capture_output=True,
    )
    return _report(tck_root)


def test_no_unexpected_must_level_failures(tck_report: dict) -> None:
    unexpected = _failing(tck_report, "MUST") - set(KNOWN_MUST_FAILURES)

    assert not unexpected, f"MUST requirements newly failing: {sorted(unexpected)}"


def test_no_unexpected_should_level_failures(tck_report: dict) -> None:
    unexpected = _failing(tck_report, "SHOULD") - set(KNOWN_SHOULD_FAILURES)

    assert not unexpected, f"SHOULD requirements newly failing: {sorted(unexpected)}"


def test_known_failures_are_still_failing(tck_report: dict) -> None:
    """When one of them starts passing, the note explaining it is now wrong."""
    failing = _failing(tck_report, "MUST") | _failing(tck_report, "SHOULD")
    fixed = (set(KNOWN_MUST_FAILURES) | set(KNOWN_SHOULD_FAILURES)) - failing

    assert not fixed, (
        f"these now pass and should be removed from the known-failure list: {sorted(fixed)}"
    )
