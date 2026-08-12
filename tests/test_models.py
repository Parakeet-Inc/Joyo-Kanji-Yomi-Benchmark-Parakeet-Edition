import pytest

from jkyb_eval.models import BenchmarkRow


def raw_row(reading_category: str) -> dict:
    return {
        "key": "誤_ゴ_0",
        "text": "誤解した。",
        "tagged_text": "<誤>解した。",
        "yomi": "ゴカイシタ。",
        "tagged_yomi": "<ゴ>カイシタ。",
        "reading_category": reading_category,
        "readings": {"natural": ["ゴ"], "marginal": []},
        "source": "original",
    }


@pytest.mark.parametrize(
    "reading_category",
    ["on_yomi", "kun_yomi", "joyo_appendix_reading"],
)
def test_reading_categories_are_accepted(reading_category: str) -> None:
    row = BenchmarkRow.from_dict(raw_row(reading_category), location="row 1")
    assert row.reading_category == reading_category


def test_unknown_reading_category_is_rejected() -> None:
    with pytest.raises(ValueError, match="reading_category must be one of"):
        BenchmarkRow.from_dict(raw_row("jukujikun"), location="row 1")
