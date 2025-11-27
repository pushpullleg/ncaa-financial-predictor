"""Quick CLI regression test for predict_trajectory.py.

Run this script from anywhere to verify that the CLI entry point
executes successfully against a known school query. The script will
exit with code 0 if the CLI works and raise an exception otherwise.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPT_PATH = BASE_DIR / "scripts" / "predict_trajectory.py"
DEFAULT_QUERY = "Alabama"


def run_cli(query: str = DEFAULT_QUERY) -> dict:
    """Execute the prediction CLI and return structured metadata."""
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Cannot locate CLI script at {SCRIPT_PATH}")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), query],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        check=True,
    )

    output = completed.stdout.strip()
    summary = {
        "query": query,
        "returncode": completed.returncode,
        "stdout": output,
    }

    # Basic assertion: ensure the CLI reported a prediction string.
    if "Predicted Trajectory:" not in output:
        raise AssertionError("CLI output did not include a prediction block")

    return summary


def main() -> None:
    summary = run_cli()
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
