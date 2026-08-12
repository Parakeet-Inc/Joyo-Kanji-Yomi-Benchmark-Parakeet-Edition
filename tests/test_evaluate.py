import pytest

from jkyb_eval.evaluate import evaluate
from jkyb_eval.models import BenchmarkRow, Prediction


def benchmark_row(key: str) -> BenchmarkRow:
    return BenchmarkRow(
        key=key,
        text="誤解した。",
        tagged_text="<誤>解した。",
        yomi="ゴカイシタ",
        tagged_yomi="<ゴ>カイシタ",
        reading_category="on_yomi",
        natural=("ゴ",),
        marginal=(),
        source="original",
    )


def test_strict_key_validation() -> None:
    rows = [benchmark_row("a"), benchmark_row("b")]
    with pytest.raises(ValueError, match="Missing predictions"):
        evaluate(rows, {"a": Prediction("a", "ゴカイシタ")})
    with pytest.raises(ValueError, match="unknown keys"):
        evaluate(
            rows,
            {
                "a": Prediction("a", "ゴカイシタ"),
                "b": Prediction("b", "ゴカイシタ"),
                "c": Prediction("c", "ゴカイシタ"),
            },
        )


def test_allowed_missing_row_is_scored() -> None:
    output = evaluate(
        [benchmark_row("a"), benchmark_row("b")],
        {"a": Prediction("a", "ゴカイシタ")},
        allow_missing=True,
    )
    assert output.missing_keys == ("b",)
    assert output.samples[1]["target_kana_cer"] == 1


def test_allowed_missing_text_prediction_is_scored() -> None:
    output = evaluate(
        [benchmark_row("a"), benchmark_row("b")],
        {"a": Prediction("a", "ゴカイシタ")},
        text_predictions={"a": Prediction("a", "誤解した。")},
        allow_missing=True,
    )
    missing = output.samples[1]
    assert missing["target_kana_cer"] == 1
    assert missing["text_cer"] == 1
    assert missing["text_prediction_missing"] is True
    assert output.aggregate["metrics"]["text_cer"]["raw"] == 0.5
