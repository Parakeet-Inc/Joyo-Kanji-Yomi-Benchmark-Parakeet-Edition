"""File-oriented execution of the core evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import __version__
from .evaluate import evaluate
from .io import (
    DatasetLocation,
    load_benchmark,
    load_predictions,
    write_json,
    write_jsonl,
)
from .report import DETAIL_PATHS, detail_rows, summary_markdown


def run_evaluation(
    *,
    dataset: DatasetLocation,
    prediction_path: Path,
    output_dir: Path,
    yomi_field: str = "yomi",
    text_prediction_path: Path | None = None,
    text_field: str = "text",
    allow_missing: bool = False,
    allow_extra: bool = False,
    additional_config: dict[str, Any] | None = None,
    input_metadata: dict[str, Any] | None = None,
    input_kind: str = "prediction",
    progress: bool = False,
) -> dict[str, Any]:
    rows = load_benchmark(dataset.path)
    predictions = load_predictions(prediction_path, value_field=yomi_field)
    text_predictions = (
        load_predictions(text_prediction_path, value_field=text_field)
        if text_prediction_path is not None
        else None
    )
    evaluation = evaluate(
        rows,
        predictions,
        text_predictions=text_predictions,
        allow_missing=allow_missing,
        allow_extra=allow_extra,
        progress=progress,
    )

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = dict(input_metadata or {})
    inputs["predictions"] = str(prediction_path)
    inputs["yomi_field"] = yomi_field
    if text_prediction_path is not None:
        inputs["text_predictions"] = str(text_prediction_path)
        inputs["text_field"] = text_field

    configuration = dict(additional_config or {})
    if allow_missing:
        configuration["allow_missing"] = True
    if allow_extra:
        configuration["allow_extra"] = True

    diagnostics = {
        **evaluation.aggregate["diagnostics"],
        "extra_predictions": len(evaluation.extra_keys),
    }
    row_count = evaluation.aggregate["row_count"]
    available_inputs = row_count - len(evaluation.missing_keys)
    summary: dict[str, Any] = {
        "evaluator": {
            "name": "jkyb-eval",
            "version": __version__,
        },
        "dataset": dataset.identifier,
        "inputs": inputs,
        "configuration": configuration,
        "row_count": row_count,
        "coverage": {
            "input": input_kind,
            "expected": row_count,
            "available": available_inputs,
            "missing": len(evaluation.missing_keys),
            "rate": available_inputs / row_count,
        },
        "metrics": evaluation.aggregate["metrics"],
        "breakdowns": evaluation.aggregate["breakdowns"],
        "diagnostics": diagnostics,
    }

    details = detail_rows(evaluation.samples)
    for name, rows_for_file in details.items():
        write_jsonl(output_dir / DETAIL_PATHS[name], rows_for_file)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        summary_markdown(
            summary,
            detail_counts={
                name: len(rows_for_file) for name, rows_for_file in details.items()
            },
        ),
        encoding="utf-8",
    )
    return summary
