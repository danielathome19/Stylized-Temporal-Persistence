"""Resolve immutable recorded raw paths after moving the public artifact."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def recorded_path(value):
    text = str(value).replace("\\", "/")
    marker = "/results/raw/"
    if marker in text:
        suffix = text.split(marker, 1)[1]
        if ".." in Path(suffix).parts:
            raise ValueError("Raw path escapes artifact")
        candidate = ROOT / "results/raw" / suffix
        return (
            candidate if candidate.exists() or not Path(value).exists() else Path(value)
        )
    return Path(value)
