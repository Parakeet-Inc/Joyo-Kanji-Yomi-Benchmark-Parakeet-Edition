import time

import pytest

from jkyb_eval.alignment import align_candidate, align_target


@pytest.mark.parametrize(
    ("reference", "prediction", "expected"),
    [
        ("イトオ<ゴ>カイ", "イトオコオカイ", "コオ"),
        ("ジブンノ<ユク>スエ", "ジブンノヨクセエ", "ヨク"),
        ("<サビ>ノキョオチ", "ジャクノキョオチ", "ジャク"),
        ("<ヌシ>トシテ", "シュトシテ", "シュ"),
        ("<トオ>デス", "ジュウデス", "ジュウ"),
        ("ウミ<サチ>ニ", "カイコオニ", "コオ"),
        ("ソノ<シッ>セキ", "ソノシカセメ", "シカ"),
        ("<コ>ガネノ", "オオゴンノ", "オオ"),
    ],
)
def test_context_alignment_extracts_replacement(
    reference: str, prediction: str, expected: str
) -> None:
    result = align_target(reference, prediction, accepted_lengths=[2])
    assert result.value == expected


def test_deleted_target_is_explicitly_empty() -> None:
    result = align_target("ライゲツ<ウ>ミズキ", "ライゲツミズキ", accepted_lengths=[1])
    assert result.value == ""
    assert result.status == "empty_mapped_span"


def test_target_at_sentence_end() -> None:
    result = align_target("キセツガ<ユク>", "キセツガイク", accepted_lengths=[2])
    assert result.value == "イク"


def test_entire_sentence_can_be_target() -> None:
    result = align_target("<トオ>", "ジュウ", accepted_lengths=[2, 3])
    assert result.value == "ジュウ"


def test_malformed_tag_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        align_target("タグナシ", "タグナシ", accepted_lengths=[1])


def test_runaway_prediction_stays_cheap() -> None:
    """A model stuck in a repetition loop must not stall the whole run.

    Scoring every boundary pair is quadratic, so a prediction three orders of
    magnitude longer than its reference is billions of candidates and enough
    scores to exhaust memory. Only boundaries near the reference context can
    win, so the cost has to track the reference rather than the prediction.
    """
    reference = "ワタシガブチョウノ<カ>ワリニ"
    prediction = "ワタシガブチョウノカ" + "ダワリニ" * 8000

    started = time.perf_counter()
    result = align_target(reference, prediction, accepted_lengths=[1])
    elapsed = time.perf_counter() - started

    assert elapsed < 5.0
    assert result.start == len("ワタシガブチョウノ")

    started = time.perf_counter()
    candidate_result = align_candidate(reference, prediction, candidate="カ")
    candidate_elapsed = time.perf_counter() - started

    assert candidate_elapsed < 5.0
    assert candidate_result.start == len("ワタシガブチョウノ")


def test_candidate_alignment_can_use_a_different_word_boundary() -> None:
    short = align_candidate("ハイ<エイ>", "セオヨギ", candidate="エイ")
    long = align_candidate("ハイ<エイ>", "セオヨギ", candidate="オヨギ")

    assert short.value == "ヨギ"
    assert (short.start, short.end) == (2, 4)
    assert long.value == "オヨギ"
    assert (long.start, long.end) == (1, 4)
    assert short.context_edit_distance == long.context_edit_distance == 2


def test_candidate_alignment_anchors_an_empty_reference_context() -> None:
    at_start = align_candidate("<ア>アト", "イアトア", candidate="ア")
    at_end = align_candidate("マエ<ア>", "アマエイ", candidate="ア")

    assert at_start.start == 0
    assert at_start.value == "イ"
    assert at_end.end == len("アマエイ")
    assert at_end.value == "イ"


def test_candidate_alignment_rejects_empty_candidate() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        align_candidate("マエ<ア>アト", "マエアアト", candidate="")
