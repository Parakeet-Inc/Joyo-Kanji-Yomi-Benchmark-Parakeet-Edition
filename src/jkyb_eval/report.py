"""Human-readable summaries and row-level result views."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import READING_CATEGORIES, READING_CATEGORY_LABELS


DETAIL_PATHS = {
    "all": Path("details/all.jsonl"),
    "missing_inputs": Path("details/missing-inputs.jsonl"),
    "target_review": Path("details/target-review.jsonl"),
    "sentence_mismatches": Path("details/sentence-mismatches.jsonl"),
    "text_mismatches": Path("details/text-mismatches.jsonl"),
}


def summary_markdown(
    summary: dict[str, Any],
    *,
    detail_counts: dict[str, int] | None = None,
) -> str:
    metrics = summary["metrics"]
    category_breakdown = summary.get("breakdowns", {}).get("reading_category", {})
    accuracy = metrics["accuracy"]
    relaxed_accuracy = metrics["relaxed_accuracy"]
    diagnostics = summary["diagnostics"]
    coverage = summary["coverage"]
    coverage_label = {
        "audio": "Audio",
        "prediction": "Prediction",
    }.get(coverage["input"], str(coverage["input"]).title())
    labels = {
        "audio_dir": "Audio",
        "predictions": "Predictions",
        "text_predictions": "Text predictions",
        "text_field": "Text field",
        "yomi_field": "Reading field",
    }
    lines = [
        "# JKYB-Parakeet Results",
        "",
        "## Input",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Dataset | `{summary['dataset']}` |",
    ]
    for name, value in summary["inputs"].items():
        lines.append(f"| {labels.get(name, name)} | `{value}` |")
    lines.extend(
        [
            f"| Rows | {summary['row_count']:,} |",
            "",
            "## Coverage",
            "",
            "| Input | Available | Missing |",
            "|---|---:|---:|",
            (
                f"| {coverage_label} | {coverage['available']:,} / "
                f"{coverage['expected']:,} ({coverage['rate']:.3%}) | "
                f"{coverage['missing']:,} |"
            ),
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
            (
                "| Accuracy | "
                f"{accuracy['rate']:.3%} "
                f"({accuracy['matches']:,} / {summary['row_count']:,}) |"
            ),
            (
                "| Relaxed Accuracy | "
                f"{relaxed_accuracy['rate']:.3%} "
                f"({relaxed_accuracy['matches']:,} / "
                f"{summary['row_count']:,}) |"
            ),
            (f"| Target Kana-CER | {metrics['target_kana_cer']['raw']:.3%} |"),
            (f"| Target Kana-CER@1 | {metrics['target_kana_cer']['cer_at_1']:.3%} |"),
            (
                "| Relaxed Target Kana-CER | "
                f"{metrics['relaxed_target_kana_cer']['raw']:.3%} |"
            ),
            (
                "| Relaxed Target Kana-CER@1 | "
                f"{metrics['relaxed_target_kana_cer']['cer_at_1']:.3%} |"
            ),
            (f"| Sentence Kana-CER | {metrics['sentence_kana_cer']['raw']:.3%} |"),
            (
                "| Sentence Kana-CER@1 | "
                f"{metrics['sentence_kana_cer']['cer_at_1']:.3%} |"
            ),
        ]
    )
    if "text_cer" in metrics:
        lines.extend(
            [
                f"| Text CER | {metrics['text_cer']['raw']:.3%} |",
                f"| Text CER@1 | {metrics['text_cer']['cer_at_1']:.3%} |",
            ]
        )
    lines.extend(
        [
            "",
            "Raw CER can exceed 100%; CER@1 clips each example's CER to 100% "
            "before averaging.",
            "",
        ]
    )
    if category_breakdown:
        lines.extend(
            [
                "## Metrics by Reading Category",
                "",
                (
                    "| Category | Rows | Accuracy | Relaxed Accuracy | "
                    "Target Kana-CER | Target Kana-CER@1 | Relaxed Target "
                    "Kana-CER | Relaxed Target Kana-CER@1 |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for category in READING_CATEGORIES:
            group = category_breakdown.get(category)
            if group is None:
                continue
            group_metrics = group["metrics"]
            lines.append(
                f"| {READING_CATEGORY_LABELS[category]} | "
                f"{group['row_count']:,} | "
                f"{group_metrics['accuracy']['rate']:.3%} | "
                f"{group_metrics['relaxed_accuracy']['rate']:.3%} | "
                f"{group_metrics['target_kana_cer']['raw']:.3%} | "
                f"{group_metrics['target_kana_cer']['cer_at_1']:.3%} | "
                f"{group_metrics['relaxed_target_kana_cer']['raw']:.3%} | "
                f"{group_metrics['relaxed_target_kana_cer']['cer_at_1']:.3%} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Diagnostics",
            "",
            "| Item | Rows |",
            "|---|---:|",
            f"| Missing predictions | {diagnostics['missing_predictions']:,} |",
            f"| Extra predictions | {diagnostics['extra_predictions']:,} |",
            (
                "| Ambiguous target alignments | "
                f"{diagnostics['ambiguous_target_alignments']:,} |"
            ),
            "",
        ]
    )
    if detail_counts is not None:
        lines.extend(
            [
                "## Details",
                "",
                "| File | Rows | Contents |",
                "|---|---:|---|",
                (
                    f"| [All rows]({DETAIL_PATHS['all'].as_posix()}) | "
                    f"{detail_counts['all']:,} | Complete per-row results |"
                ),
                (
                    "| [Missing inputs]"
                    f"({DETAIL_PATHS['missing_inputs'].as_posix()})"
                    f" | {detail_counts['missing_inputs']:,} | Rows scored as "
                    "errors because no input was available |"
                ),
                (
                    "| [Target review]"
                    f"({DETAIL_PATHS['target_review'].as_posix()})"
                    f" | {detail_counts['target_review']:,} | Incorrect or marginal "
                    "target readings and alignment warnings |"
                ),
                (
                    "| [Sentence mismatches]"
                    f"({DETAIL_PATHS['sentence_mismatches'].as_posix()})"
                    f" | {detail_counts['sentence_mismatches']:,} | Rows with nonzero "
                    "Sentence Kana-CER |"
                ),
            ]
        )
        if "text_mismatches" in detail_counts:
            lines.append(
                "| [Text mismatches]"
                f"({DETAIL_PATHS['text_mismatches'].as_posix()})"
                f" | {detail_counts['text_mismatches']:,} | Rows with nonzero "
                "Text CER |"
            )
        lines.append("")
    return "\n".join(lines)


def _has_alignment_warning(row: dict[str, Any]) -> bool:
    return row["alignment_status"] != "mapped" or bool(row["alignment_ambiguous"])


def _target_review_priority(row: dict[str, Any]) -> tuple[Any, ...]:
    if row["target_match_category"] == "incorrect":
        category = 0
    elif _has_alignment_warning(row):
        category = 1
    else:
        category = 2
    return (
        category,
        -float(row["target_kana_cer"]),
        -float(row["sentence_kana_cer"]),
        row["key"],
    )


def detail_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    details = {
        "all": rows,
        "missing_inputs": [row for row in rows if row["prediction_missing"]],
        "target_review": sorted(
            (
                row
                for row in rows
                if row["target_match_category"] != "natural"
                or _has_alignment_warning(row)
            ),
            key=_target_review_priority,
        ),
        "sentence_mismatches": sorted(
            (row for row in rows if float(row["sentence_kana_cer"]) > 0),
            key=lambda row: (
                -float(row["sentence_kana_cer"]),
                -float(row["target_kana_cer"]),
                row["key"],
            ),
        ),
    }
    if rows and all("text_cer" in row for row in rows):
        details["text_mismatches"] = sorted(
            (row for row in rows if float(row["text_cer"]) > 0),
            key=lambda row: (
                -float(row["text_cer"]),
                -float(row["sentence_kana_cer"]),
                row["key"],
            ),
        )
    return details
