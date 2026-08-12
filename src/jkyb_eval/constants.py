"""Release defaults shared by the CLI commands."""

DEFAULT_DATASET_REPO = "Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet"
DEFAULT_DATASET_FILENAME = "data/common_kanji_source.jsonl"

DEFAULT_KANA_MODEL = "sbintuitions/kana-whisper"
DEFAULT_TEXT_MODEL = "openai/whisper-large-v3-turbo"
DEFAULT_ASR_BATCH_SIZE = 16

ON_YOMI = "on_yomi"
KUN_YOMI = "kun_yomi"
JOYO_APPENDIX_READING = "joyo_appendix_reading"
READING_CATEGORIES = (ON_YOMI, KUN_YOMI, JOYO_APPENDIX_READING)
READING_CATEGORY_LABELS = {
    ON_YOMI: "On’yomi",
    KUN_YOMI: "Kun’yomi",
    JOYO_APPENDIX_READING: "Jōyō appendix readings",
}
ASR_MAX_NEW_TOKENS = 128
