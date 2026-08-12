from jkyb_eval.normalization import canonicalize_text, canonicalize_yomi


def test_long_vowel_conventions_compare_equal() -> None:
    assert canonicalize_yomi("ガッコウ") == canonicalize_yomi("ガッコー")
    assert canonicalize_yomi("ケイエイ") == canonicalize_yomi("ケーエー")


def test_compound_small_kana_remain_distinct() -> None:
    assert canonicalize_yomi("パーティー") == "パアティイ"
    assert canonicalize_yomi("パーティー") != canonicalize_yomi("パーテーー")
    assert canonicalize_yomi("ティ") != canonicalize_yomi("テイ")
    assert canonicalize_yomi("トゥ") != canonicalize_yomi("トウ")
    assert canonicalize_yomi("トゥー") != canonicalize_yomi("トー")


def test_equivalent_kana_compare_equal() -> None:
    assert canonicalize_yomi("ヂヅヲ") == canonicalize_yomi("ジズオ")
    assert canonicalize_yomi("そこ、うまい。") == canonicalize_yomi("ソコウマイ")


def test_distinct_particles_are_not_folded() -> None:
    assert canonicalize_yomi("ヘ") != canonicalize_yomi("エ")
    assert canonicalize_yomi("ハ") != canonicalize_yomi("ワ")


def test_text_normalization_removes_punctuation_and_width() -> None:
    assert canonicalize_text("１９２９年、世界恐慌。") == "1929年世界恐慌"


def test_tags_can_be_preserved_without_changing_vowel_context() -> None:
    assert canonicalize_yomi("ガッ<コー>", preserve_tags=True) == "ガッ<コオ>"
