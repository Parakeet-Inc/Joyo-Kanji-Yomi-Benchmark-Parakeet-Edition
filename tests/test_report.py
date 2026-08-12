from jkyb_eval.report import detail_rows, summary_markdown


def test_summary_markdown_uses_public_metric_names() -> None:
    summary = {
        "dataset": "Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet",
        "inputs": {
            "predictions": "results/pyopenjtalk/predictions.jsonl",
            "yomi_field": "yomi",
        },
        "row_count": 3,
        "coverage": {
            "input": "prediction",
            "expected": 3,
            "available": 3,
            "missing": 0,
            "rate": 1.0,
        },
        "metrics": {
            "accuracy": {"rate": 1 / 3, "matches": 1},
            "relaxed_accuracy": {"rate": 2 / 3, "matches": 2},
            "target_kana_cer": {"raw": 0.25, "cer_at_1": 0.2},
            "relaxed_target_kana_cer": {"raw": 0.125, "cer_at_1": 0.1},
            "sentence_kana_cer": {"raw": 0.125, "cer_at_1": 0.125},
        },
        "breakdowns": {
            "reading_category": {
                "on_yomi": {
                    "row_count": 1,
                    "metrics": {
                        "accuracy": {"rate": 1.0, "matches": 1},
                        "relaxed_accuracy": {"rate": 1.0, "matches": 1},
                        "target_kana_cer": {"raw": 0.0, "cer_at_1": 0.0},
                        "relaxed_target_kana_cer": {
                            "raw": 0.0,
                            "cer_at_1": 0.0,
                        },
                    },
                },
                "joyo_appendix_reading": {
                    "row_count": 2,
                    "metrics": {
                        "accuracy": {"rate": 0.0, "matches": 0},
                        "relaxed_accuracy": {"rate": 0.5, "matches": 1},
                        "target_kana_cer": {"raw": 0.375, "cer_at_1": 0.3},
                        "relaxed_target_kana_cer": {
                            "raw": 0.1875,
                            "cer_at_1": 0.15,
                        },
                    },
                },
            }
        },
        "diagnostics": {
            "missing_predictions": 0,
            "extra_predictions": 0,
            "ambiguous_target_alignments": 0,
        },
    }

    rendered = summary_markdown(
        summary,
        detail_counts={
            "all": 3,
            "missing_inputs": 0,
            "target_review": 2,
            "sentence_mismatches": 1,
        },
    )

    assert "# JKYB-Parakeet Results" in rendered
    assert "| Dataset | `Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet` |" in rendered
    assert "| Predictions | `results/pyopenjtalk/predictions.jsonl` |" in rendered
    assert "| Prediction | 3 / 3 (100.000%) | 0 |" in rendered
    assert "| Target Kana-CER | 25.000% |" in rendered
    assert "| Relaxed Target Kana-CER | 12.500% |" in rendered
    assert "| Accuracy | 33.333% (1 / 3) |" in rendered
    assert "| Relaxed Accuracy | 66.667% (2 / 3) |" in rendered
    assert "## Metrics by Reading Category" in rendered
    metric_lines = (
        rendered.split("## Metrics\n", 1)[1]
        .split("Raw CER can exceed", 1)[0]
        .splitlines()
    )
    assert [line.split(" |", 1)[0] for line in metric_lines if line.startswith("| ")][
        1:
    ] == [
        "| Accuracy",
        "| Relaxed Accuracy",
        "| Target Kana-CER",
        "| Target Kana-CER@1",
        "| Relaxed Target Kana-CER",
        "| Relaxed Target Kana-CER@1",
        "| Sentence Kana-CER",
        "| Sentence Kana-CER@1",
    ]
    assert (
        "| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | "
        "Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |"
        in rendered
    )
    assert (
        "| On’yomi | 1 | 100.000% | 100.000% | 0.000% | 0.000% | 0.000% | "
        "0.000% |" in rendered
    )
    assert (
        "| Jōyō appendix readings | 2 | 0.000% | 50.000% | 37.500% | 30.000% | "
        "18.750% | 15.000% |" in rendered
    )
    assert "| [All rows](details/all.jsonl) | 3 |" in rendered
    assert "| [Target review](details/target-review.jsonl) | 2 |" in rendered
    assert "text-mismatches.jsonl" not in rendered
    assert "target_accuracy" not in rendered


def test_detail_rows_create_focused_review_views() -> None:
    rows = [
        {
            "key": "natural",
            "prediction_missing": False,
            "target_match_category": "natural",
            "alignment_status": "mapped",
            "alignment_ambiguous": False,
            "target_kana_cer": 0.0,
            "sentence_kana_cer": 0.0,
            "text_cer": 0.0,
        },
        {
            "key": "marginal",
            "prediction_missing": False,
            "target_match_category": "marginal",
            "alignment_status": "mapped",
            "alignment_ambiguous": False,
            "target_kana_cer": 0.0,
            "sentence_kana_cer": 0.1,
            "text_cer": 0.0,
        },
        {
            "key": "incorrect",
            "prediction_missing": True,
            "target_match_category": "incorrect",
            "alignment_status": "mapped",
            "alignment_ambiguous": False,
            "target_kana_cer": 1.0,
            "sentence_kana_cer": 0.2,
            "text_cer": 0.3,
        },
        {
            "key": "alignment",
            "prediction_missing": False,
            "target_match_category": "natural",
            "alignment_status": "mapped",
            "alignment_ambiguous": True,
            "target_kana_cer": 0.0,
            "sentence_kana_cer": 0.0,
            "text_cer": 0.4,
        },
    ]

    details = detail_rows(rows)

    assert details["all"] == rows
    assert [row["key"] for row in details["missing_inputs"]] == ["incorrect"]
    assert [row["key"] for row in details["target_review"]] == [
        "incorrect",
        "alignment",
        "marginal",
    ]
    assert [row["key"] for row in details["sentence_mismatches"]] == [
        "incorrect",
        "marginal",
    ]
    assert [row["key"] for row in details["text_mismatches"]] == [
        "alignment",
        "incorrect",
    ]
