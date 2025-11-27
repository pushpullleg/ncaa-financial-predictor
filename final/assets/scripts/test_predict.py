"""Quick CLI regression test for predict.py.

Run this script from anywhere to verify that the CLI entry point
executes successfully against a known school query. The script will
exit with code 0 if the CLI works and raise an exception otherwise.

Note: Run with the project's virtual environment Python:
    .venv/bin/python test_predict.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# This test file is in: final/assets/scripts/test_predict.py
# The predict.py is in:  final/assets/scripts/predict.py
SCRIPTS_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = SCRIPTS_DIR / "predict.py"
DEFAULT_QUERY = "Alabama"


def run_cli(query: str = DEFAULT_QUERY) -> dict:
    """Execute the prediction CLI and return structured metadata."""
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Cannot locate CLI script at {SCRIPT_PATH}")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), query],
        cwd=str(SCRIPTS_DIR),
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
