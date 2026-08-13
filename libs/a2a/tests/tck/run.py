"""Run the official A2A TCK against this package's SUT.

```bash
python -m tests.tck.run                     # clone the pin into .tck/ and run
A2A_TCK_PATH=../a2a-tck python -m tests.tck.run   # use a checkout you already have
```

The commit is pinned. An unpinned conformance gate is not a gate: the suite
would change under CI and a red build would be ambiguous between "we regressed"
and "they added a requirement".
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

TCK_REPO = "https://github.com/a2aproject/a2a-tck.git"
TCK_COMMIT = "5996b79f9cefa6fc390980e383e358a66fb9e49e"
"""a2aproject/a2a-tck, 2026-06-29. Bump deliberately, with a report to match."""

SUT_HOST = "http://127.0.0.1:9999"
CARD_URL = f"{SUT_HOST}/.well-known/agent-card.json"


def checkout(root: Path) -> Path:
    """The pinned TCK, cloned on first use."""
    existing = os.environ.get("A2A_TCK_PATH")
    if existing:
        return Path(existing).resolve()

    target = root / ".tck"
    if not (target / "run_tck.py").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", TCK_REPO, str(target)], check=True)
    subprocess.run(
        ["git", "-C", str(target), "fetch", "--depth", "1", "origin", TCK_COMMIT],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(target), "checkout", "--quiet", TCK_COMMIT], check=True
    )
    if not (target / ".venv").exists():
        subprocess.run(["uv", "venv"], cwd=target, check=True)
    # `--python` explicitly: run under `uv run`, VIRTUAL_ENV points at this
    # package's own environment and the TCK would install itself there instead,
    # leaving the environment it is about to be run from empty.
    subprocess.run(
        ["uv", "pip", "install", "--python", ".venv", "-e", "."],
        cwd=target,
        check=True,
    )
    return target


def start_sut() -> subprocess.Popen[bytes]:
    process = subprocess.Popen(
        [sys.executable, "-m", "tests.tck.sut"],
        cwd=Path(__file__).resolve().parents[2],
    )
    for _ in range(100):
        try:
            if httpx.get(CARD_URL, timeout=1).status_code == 200:
                return process
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    process.terminate()
    raise RuntimeError("the SUT did not come up")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=["must", "should", "may"], default=None)
    parser.add_argument(
        "--clone-only",
        action="store_true",
        help="fetch and build the pinned TCK, then stop. For a CI step that "
        "wants the checkout in place before the tests run.",
    )
    parser.add_argument("extra", nargs="*", help="passed through to the TCK")
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[2]
    tck = checkout(package_root)
    if args.clone_only:
        return 0
    python = tck / ".venv" / "bin" / "python"

    command = [str(python), "run_tck.py", "--sut-host", SUT_HOST]
    if args.level:
        command += ["--level", args.level]
    command += args.extra

    sut = start_sut()
    try:
        return subprocess.run(command, cwd=tck, check=False).returncode
    finally:
        sut.terminate()
        sut.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
