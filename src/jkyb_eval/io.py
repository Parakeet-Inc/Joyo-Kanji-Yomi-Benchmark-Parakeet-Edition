"""JSONL input, output, hashing, and dataset resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import hf_hub_download

from .constants import DEFAULT_DATASET_FILENAME, DEFAULT_DATASET_REPO
from .models import BenchmarkRow, Prediction


@dataclass(frozen=True)
class DatasetLocation:
    path: Path
    repository: str | None

    @property
    def identifier(self) -> str:
        return self.repository or str(self.path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def resolve_dataset(
    path: Path | None,
    *,
    repository: str = DEFAULT_DATASET_REPO,
) -> DatasetLocation:
    if path is None:
        resolved = Path(
            hf_hub_download(
                repo_id=repository,
                repo_type="dataset",
                filename=DEFAULT_DATASET_FILENAME,
            )
        )
        return DatasetLocation(
            path=resolved,
            repository=repository,
        )

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Dataset JSONL not found: {resolved}")
    return DatasetLocation(
        path=resolved,
        repository=None,
    )


def load_benchmark(path: Path) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(read_jsonl(path), start=1):
        row = BenchmarkRow.from_dict(raw, location=f"{path}:{line_number}")
        if row.key in seen:
            raise ValueError(f"{path}:{line_number}: duplicate key {row.key!r}")
        seen.add(row.key)
        rows.append(row)
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def load_predictions(
    path: Path,
    *,
    value_field: str,
) -> dict[str, Prediction]:
    predictions: dict[str, Prediction] = {}
    for line_number, raw in enumerate(read_jsonl(path), start=1):
        key = raw.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{path}:{line_number}: key must be a non-empty string")
        value = raw.get(value_field)
        if not isinstance(value, str):
            raise ValueError(f"{path}:{line_number}: {value_field} must be a string")
        if key in predictions:
            raise ValueError(f"{path}:{line_number}: duplicate key {key!r}")
        predictions[key] = Prediction(key=key, value=value)
    return predictions
