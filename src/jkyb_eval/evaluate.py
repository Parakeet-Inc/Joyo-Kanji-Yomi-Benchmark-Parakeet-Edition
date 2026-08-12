"""Evaluation orchestration independent of model inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .metrics import aggregate_results, score_row
from .models import BenchmarkRow, Prediction


@dataclass(frozen=True)
class EvaluationOutput:
    samples: list[dict[str, Any]]
    aggregate: dict[str, Any]
    missing_keys: tuple[str, ...]
    extra_keys: tuple[str, ...]


def evaluate(
    rows: list[BenchmarkRow],
    predictions: dict[str, Prediction],
    *,
    text_predictions: dict[str, Prediction] | None = None,
    allow_missing: bool = False,
    allow_extra: bool = False,
    progress: bool = False,
) -> EvaluationOutput:
    dataset_keys = {row.key for row in rows}
    prediction_keys = set(predictions)
    missing = tuple(sorted(dataset_keys - prediction_keys))
    extra = tuple(sorted(prediction_keys - dataset_keys))
    if missing and not allow_missing:
        raise ValueError(
            f"Missing predictions for {len(missing)} keys; examples: {missing[:10]!r}"
        )
    if extra and not allow_extra:
        raise ValueError(
            f"Predictions contain {len(extra)} unknown keys; examples: {extra[:10]!r}"
        )

    if text_predictions is not None:
        text_keys = set(text_predictions)
        missing_text = dataset_keys - text_keys
        extra_text = text_keys - dataset_keys
        if missing_text and not allow_missing:
            examples = tuple(sorted(missing_text))[:10]
            raise ValueError(
                f"Missing text predictions for {len(missing_text)} keys; "
                f"examples: {examples!r}"
            )
        if extra_text and not allow_extra:
            examples = tuple(sorted(extra_text))[:10]
            raise ValueError(
                f"Text predictions contain {len(extra_text)} unknown keys; "
                f"examples: {examples!r}"
            )

    iterable = rows
    if progress:
        from tqdm import tqdm

        iterable = tqdm(rows, desc="Scoring", unit="row", dynamic_ncols=True)

    samples = [
        score_row(
            row,
            predictions[row.key].value if row.key in predictions else None,
            text_prediction=(
                text_predictions[row.key].value
                if text_predictions is not None and row.key in text_predictions
                else None
            ),
            score_text=text_predictions is not None,
        )
        for row in iterable
    ]
    return EvaluationOutput(
        samples=samples,
        aggregate=aggregate_results(samples),
        missing_keys=missing,
        extra_keys=extra,
    )
