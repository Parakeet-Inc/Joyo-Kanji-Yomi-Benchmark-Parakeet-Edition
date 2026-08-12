import pytest

from jkyb_eval.metrics import aggregate_results, score_row
from jkyb_eval.models import BenchmarkRow


def row(
    *,
    yomi: str = "カレワサビシイ",
    tagged_yomi: str = "カレワ<サビシ>イ",
    natural: tuple[str, ...] = ("サビシ",),
    marginal: tuple[str, ...] = ("サミシ",),
    reading_category: str = "kun_yomi",
) -> BenchmarkRow:
    return BenchmarkRow(
        key="寂_さびしい_0",
        text="彼は寂しい。",
        tagged_text="彼は<寂>しい。",
        yomi=yomi,
        tagged_yomi=tagged_yomi,
        reading_category=reading_category,
        natural=natural,
        marginal=marginal,
        source="original_alt",
    )


def test_natural_target_exact_match() -> None:
    result = score_row(row(), "カレワサビシイ")
    assert result["target_kana_cer"] == 0
    assert result["target_match_category"] == "natural"


def test_marginal_target_is_accepted_but_distinguished() -> None:
    result = score_row(row(), "カレワサミシイ")
    assert result["target_kana_cer"] == pytest.approx(1 / 3)
    assert result["relaxed_target_kana_cer"] == 0
    assert result["target_exact"] is False
    assert result["relaxed_target_exact"] is True
    assert result["target_match_category"] == "marginal"


def test_different_length_marginal_uses_its_own_alignment() -> None:
    sample = row(
        yomi="ハイエイ",
        tagged_yomi="ハイ<エイ>",
        natural=("エイ",),
        marginal=("オヨギ",),
        reading_category="on_yomi",
    )

    result = score_row(sample, "セオヨギ")

    assert result["mapped_target"] == "ヨギ"
    assert result["best_target_reading"] == "エイ"
    assert result["target_exact"] is False
    assert result["relaxed_mapped_target"] == "オヨギ"
    assert result["best_relaxed_target_reading"] == "オヨギ"
    assert result["relaxed_target_exact"] is True
    assert result["target_match_category"] == "marginal"


def test_different_length_natural_uses_its_own_alignment() -> None:
    sample = row(
        yomi="ハイエイ",
        tagged_yomi="ハイ<エイ>",
        natural=("エイ", "オヨギ"),
        marginal=(),
        reading_category="on_yomi",
    )

    result = score_row(sample, "セオヨギ")

    assert result["mapped_target"] == "オヨギ"
    assert result["best_target_reading"] == "オヨギ"
    assert result["target_exact"] is True
    assert result["relaxed_target_exact"] is True
    assert result["target_match_category"] == "natural"


def test_target_only_alternative_does_not_hide_sentence_error() -> None:
    sample = row(
        yomi="アメツチ",
        tagged_yomi="<アメ>ツチ",
        natural=("アメ",),
        marginal=("テン",),
    )

    result = score_row(sample, "テンチ")

    assert result["target_exact"] is False
    assert result["relaxed_target_exact"] is True
    assert result["relaxed_mapped_target"] == "テン"
    assert result["sentence_kana_cer"] > 0


def test_sentence_error_outside_target_does_not_change_target_score() -> None:
    result = score_row(row(), "ソレワサビシイ")
    assert result["target_kana_cer"] == 0
    assert result["sentence_kana_cer"] > 0


def test_deleted_target_counts_as_full_error() -> None:
    sample = row(
        yomi="ライゲツウミズキ",
        tagged_yomi="ライゲツ<ウ>ミズキ",
        natural=("ウ",),
        marginal=(),
    )
    result = score_row(sample, "ライゲツミズキ")
    assert result["mapped_target"] == ""
    assert result["alignment_status"] == "empty_mapped_span"
    assert result["target_kana_cer"] == 1


def test_target_normalization_uses_sentence_context() -> None:
    sample = row(
        yomi="シジヲウケタ",
        tagged_yomi="シジヲ<ウ>ケタ",
        natural=("ウ",),
        marginal=(),
    )
    result = score_row(sample, "シジヲウケタ")
    assert result["reference_target"] == "オ"
    assert result["accepted_readings"][0]["normalized"] == "オ"
    assert result["target_kana_cer"] == 0


def test_missing_prediction_counts_as_error() -> None:
    result = score_row(row(), None)
    assert result["prediction_missing"] is True
    assert result["alignment_status"] == "missing_prediction"
    assert result["target_kana_cer"] == 1
    assert result["sentence_kana_cer"] == 1


def test_raw_and_clipped_target_cer_are_both_kept() -> None:
    sample = row(
        yomi="ゴ",
        tagged_yomi="<ゴ>",
        natural=("ゴ",),
        marginal=(),
    )
    result = score_row(sample, "コオ")
    assert result["target_kana_cer"] == 2
    assert result["target_kana_cer_clipped"] == 1


def test_text_cer_is_optional() -> None:
    without_text = score_row(row(), "カレワサビシイ")
    with_text = score_row(row(), "カレワサビシイ", text_prediction="彼は寂しい")
    assert "text_cer" not in without_text
    assert with_text["text_cer"] == 0


def test_missing_text_prediction_counts_as_error_when_text_is_scored() -> None:
    result = score_row(row(), None, text_prediction=None, score_text=True)
    assert result["text_prediction_missing"] is True
    assert result["prediction_text"] == ""
    assert result["text_cer"] == 1


def test_aggregate_reports_category_rates() -> None:
    results = [
        score_row(row(), "カレワサビシイ"),
        score_row(row(), "カレワサミシイ"),
        score_row(row(), "カレワサムシイ"),
    ]
    summary = aggregate_results(results)
    assert list(summary["metrics"]) == [
        "accuracy",
        "relaxed_accuracy",
        "target_kana_cer",
        "relaxed_target_kana_cer",
        "sentence_kana_cer",
    ]
    assert summary["metrics"]["accuracy"] == {
        "rate": pytest.approx(1 / 3),
        "matches": 1,
    }
    assert summary["metrics"]["relaxed_accuracy"] == {
        "rate": pytest.approx(2 / 3),
        "matches": 2,
    }
    assert (
        summary["metrics"]["target_kana_cer"]["raw"]
        > summary["metrics"]["relaxed_target_kana_cer"]["raw"]
    )
    assert summary["diagnostics"] == {
        "ambiguous_target_alignments": 0,
        "missing_predictions": 0,
    }


def test_aggregate_reports_metrics_by_reading_category() -> None:
    results = [
        score_row(row(reading_category="on_yomi"), "カレワサビシイ"),
        score_row(row(reading_category="kun_yomi"), "カレワサミシイ"),
        score_row(row(reading_category="joyo_appendix_reading"), None),
    ]

    breakdown = aggregate_results(results)["breakdowns"]["reading_category"]

    assert list(breakdown) == [
        "on_yomi",
        "kun_yomi",
        "joyo_appendix_reading",
    ]
    assert breakdown["on_yomi"] == {
        "row_count": 1,
        "metrics": {
            "accuracy": {"rate": 1.0, "matches": 1},
            "relaxed_accuracy": {"rate": 1.0, "matches": 1},
            "target_kana_cer": {"raw": 0.0, "cer_at_1": 0.0},
            "relaxed_target_kana_cer": {"raw": 0.0, "cer_at_1": 0.0},
        },
    }
    assert list(breakdown["on_yomi"]["metrics"]) == [
        "accuracy",
        "relaxed_accuracy",
        "target_kana_cer",
        "relaxed_target_kana_cer",
    ]
    assert breakdown["kun_yomi"]["metrics"]["accuracy"] == {
        "rate": 0.0,
        "matches": 0,
    }
    assert breakdown["kun_yomi"]["metrics"]["relaxed_accuracy"] == {
        "rate": 1.0,
        "matches": 1,
    }
    assert breakdown["joyo_appendix_reading"]["metrics"]["target_kana_cer"] == {
        "raw": 1.0,
        "cer_at_1": 1.0,
    }
