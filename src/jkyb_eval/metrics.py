"""Per-row and aggregate benchmark metrics."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any

from rapidfuzz.distance import Levenshtein

from .alignment import TargetAlignment, align_candidate, split_tagged
from .constants import READING_CATEGORIES
from .models import BenchmarkRow
from .normalization import canonicalize_text, canonicalize_yomi


def character_error_rate(reference: str, prediction: str) -> float:
    if not reference:
        return 0.0 if not prediction else math.inf
    return Levenshtein.distance(reference, prediction) / len(reference)


def _accepted_readings(row: BenchmarkRow) -> list[dict[str, str]]:
    accepted: list[dict[str, str]] = []
    seen: set[str] = set()
    raw_parts = split_tagged(row.tagged_yomi)
    for category, readings in (("natural", row.natural), ("marginal", row.marginal)):
        for reading in readings:
            contextual_variant = f"{raw_parts.prefix}<{reading}>{raw_parts.suffix}"
            normalized = split_tagged(
                canonicalize_yomi(contextual_variant, preserve_tags=True)
            ).target
            if not normalized:
                raise ValueError(f"{row.key}: accepted reading normalizes to empty")
            if normalized in seen:
                continue
            seen.add(normalized)
            accepted.append(
                {
                    "reading": reading,
                    "normalized": normalized,
                    "category": category,
                }
            )
    return accepted


def _alignment_fields(
    alignment: TargetAlignment,
    *,
    prefix: str = "",
) -> dict[str, Any]:
    return {
        f"{prefix}mapped_target": alignment.value,
        f"{prefix}target_start": alignment.start,
        f"{prefix}target_end": alignment.end,
        f"{prefix}alignment_status": alignment.status,
        f"{prefix}alignment_ambiguous": alignment.ambiguous,
        f"{prefix}alignment_candidate_count": alignment.candidate_count,
        f"{prefix}alignment_context_edit_distance": (alignment.context_edit_distance),
        f"{prefix}alignment_context_error_rate": alignment.context_error_rate,
    }


def score_row(
    row: BenchmarkRow,
    prediction: str | None,
    *,
    text_prediction: str | None = None,
    score_text: bool = False,
) -> dict[str, Any]:
    prediction_missing = prediction is None
    prediction_raw = prediction or ""
    prediction_normalized = canonicalize_yomi(prediction_raw)
    tagged_reference = canonicalize_yomi(row.tagged_yomi, preserve_tags=True)
    reference_parts = split_tagged(tagged_reference)
    accepted = _accepted_readings(row)

    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(accepted):
        alignment = align_candidate(
            tagged_reference,
            prediction_normalized,
            candidate=item["normalized"],
        )
        candidates.append(
            {
                "error": character_error_rate(item["normalized"], alignment.value),
                "category_priority": 0 if item["category"] == "natural" else 1,
                "index": index,
                "item": item,
                "alignment": alignment,
            }
        )

    def candidate_key(candidate: dict[str, Any]) -> tuple[float, int, int, int]:
        return (
            float(candidate["error"]),
            int(candidate["category_priority"]),
            int(candidate["alignment"].context_edit_distance),
            int(candidate["index"]),
        )

    relaxed_result = min(candidates, key=candidate_key)
    natural_results = [
        candidate
        for candidate in candidates
        if candidate["item"]["category"] == "natural"
    ]
    target_result = min(natural_results, key=candidate_key)
    relaxed_target_error = float(relaxed_result["error"])
    target_error = float(target_result["error"])
    relaxed_best = relaxed_result["item"]
    best = target_result["item"]
    alignment = target_result["alignment"]
    relaxed_alignment = relaxed_result["alignment"]

    alignment_status = "missing_prediction" if prediction_missing else alignment.status
    relaxed_alignment_status = (
        "missing_prediction" if prediction_missing else relaxed_alignment.status
    )

    if relaxed_target_error == 0.0:
        match_category = relaxed_best["category"]
    else:
        match_category = "incorrect"

    reference_sentence = canonicalize_yomi(row.yomi)
    sentence_error = character_error_rate(reference_sentence, prediction_normalized)

    result: dict[str, Any] = {
        "key": row.key,
        "text": row.text,
        "tagged_text": row.tagged_text,
        "source": row.source,
        "reading_category": row.reading_category,
        "reference_yomi": row.yomi,
        "prediction_yomi": prediction_raw,
        "reference_yomi_normalized": reference_sentence,
        "prediction_yomi_normalized": prediction_normalized,
        "reference_target": reference_parts.target,
        "accepted_readings": accepted,
        **_alignment_fields(alignment),
        **_alignment_fields(relaxed_alignment, prefix="relaxed_"),
        "alignment_status": alignment_status,
        "relaxed_alignment_status": relaxed_alignment_status,
        "best_target_reading": best["reading"],
        "best_target_reading_normalized": best["normalized"],
        "best_relaxed_target_reading": relaxed_best["reading"],
        "best_relaxed_target_reading_normalized": relaxed_best["normalized"],
        "target_match_category": match_category,
        "target_exact": target_error == 0.0,
        "relaxed_target_exact": relaxed_target_error == 0.0,
        "target_kana_cer": target_error,
        "target_kana_cer_clipped": min(1.0, target_error),
        "relaxed_target_kana_cer": relaxed_target_error,
        "relaxed_target_kana_cer_clipped": min(1.0, relaxed_target_error),
        "sentence_kana_cer": sentence_error,
        "sentence_kana_cer_clipped": min(1.0, sentence_error),
        "prediction_missing": prediction_missing,
    }

    if text_prediction is not None:
        score_text = True
    if score_text:
        text_prediction_missing = text_prediction is None
        text_prediction_raw = text_prediction or ""
        reference_text = canonicalize_text(row.text)
        normalized_text_prediction = canonicalize_text(text_prediction_raw)
        text_error = character_error_rate(reference_text, normalized_text_prediction)
        result.update(
            {
                "prediction_text": text_prediction_raw,
                "reference_text_normalized": reference_text,
                "prediction_text_normalized": normalized_text_prediction,
                "text_cer": text_error,
                "text_cer_clipped": min(1.0, text_error),
                "text_prediction_missing": text_prediction_missing,
            }
        )
    return result


def _mean(results: list[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in results]
    if not values:
        raise ValueError("Cannot summarize an empty metric")
    return statistics.fmean(values)


def _aggregate_target_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    cer_metrics = {}
    for name in (
        "target_kana_cer",
        "relaxed_target_kana_cer",
    ):
        cer_metrics[name] = {
            "raw": _mean(results, name),
            "cer_at_1": _mean(results, f"{name}_clipped"),
        }

    match_counts = Counter(row["target_match_category"] for row in results)
    total = len(results)
    natural_matches = match_counts["natural"]
    accepted_matches = natural_matches + match_counts["marginal"]
    return {
        "accuracy": {
            "rate": natural_matches / total,
            "matches": natural_matches,
        },
        "relaxed_accuracy": {
            "rate": accepted_matches / total,
            "matches": accepted_matches,
        },
        **cer_metrics,
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot aggregate an empty evaluation")
    target_metrics = _aggregate_target_metrics(results)
    metrics = {
        "accuracy": target_metrics["accuracy"],
        "relaxed_accuracy": target_metrics["relaxed_accuracy"],
        "target_kana_cer": target_metrics["target_kana_cer"],
        "relaxed_target_kana_cer": target_metrics["relaxed_target_kana_cer"],
        "sentence_kana_cer": {
            "raw": _mean(results, "sentence_kana_cer"),
            "cer_at_1": _mean(results, "sentence_kana_cer_clipped"),
        },
    }
    if all("text_cer" in row for row in results):
        metrics["text_cer"] = {
            "raw": _mean(results, "text_cer"),
            "cer_at_1": _mean(results, "text_cer_clipped"),
        }

    category_breakdown = {}
    for category in READING_CATEGORIES:
        category_results = [
            row for row in results if row["reading_category"] == category
        ]
        if category_results:
            category_breakdown[category] = {
                "row_count": len(category_results),
                "metrics": _aggregate_target_metrics(category_results),
            }

    return {
        "row_count": len(results),
        "metrics": metrics,
        "breakdowns": {"reading_category": category_breakdown},
        "diagnostics": {
            "ambiguous_target_alignments": sum(
                bool(row["alignment_ambiguous"]) for row in results
            ),
            "missing_predictions": sum(
                bool(row["prediction_missing"]) for row in results
            ),
        },
    }
