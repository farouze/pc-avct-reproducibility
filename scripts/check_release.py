#!/usr/bin/env python3
"""Fail closed if a proposed public release contains private data or notebook outputs."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".csv", ".tsv", ".parquet", ".h5", ".mat", ".pkl", ".joblib", ".zip"}
ALLOWED_CSV = set()
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


def tracked_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return [ROOT / line for line in completed.stdout.splitlines() if line]


def main() -> None:
    errors: list[str] = []
    files = tracked_files()
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES and relative not in ALLOWED_CSV:
            errors.append(f"tracked restricted-data file type: {relative}")
        if path.suffix == ".ipynb":
            notebook = json.loads(path.read_text())
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") == "code":
                    if cell.get("outputs"):
                        errors.append(f"saved notebook output: {relative}, cell {index}")
                    if cell.get("execution_count") is not None:
                        errors.append(f"notebook execution count: {relative}, cell {index}")
        if path.stat().st_size <= 5_000_000 and path.suffix.lower() not in {".pdf", ".png"}:
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"possible {label}: {relative}")
    if errors:
        raise SystemExit("Release check failed:\n- " + "\n- ".join(errors))
    print(f"Release check passed for {len(files)} tracked files.")


if __name__ == "__main__":
    main()
