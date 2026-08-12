import json
from pathlib import Path
from typing import Iterable

import pytest

import jkyb_eval.asr as asr


class FakeTranscriber:
    calls = 0

    def __init__(self, configuration: asr.AsrConfiguration) -> None:
        pass

    def transcribe(self, paths: Iterable[Path]) -> Iterable[str]:
        type(self).calls += 1
        return (f"ヨミ{path.stem}" for path in paths)


def configuration() -> asr.AsrConfiguration:
    return asr.AsrConfiguration(
        model="test-model",
        device="cpu",
        batch_size=2,
    )


def test_generation_kwargs_include_optional_token_limit() -> None:
    assert asr._generation_kwargs(configuration()) == {
        "language": "ja",
        "task": "transcribe",
    }
    limited = asr.AsrConfiguration(
        model="test-model",
        device="cpu",
        batch_size=2,
        max_new_tokens=128,
    )
    assert asr._generation_kwargs(limited) == {
        "language": "ja",
        "task": "transcribe",
        "max_new_tokens": 128,
    }


def test_parallel_audio_loading_preserves_input_order(monkeypatch) -> None:
    paths = [Path("c.wav"), Path("a.wav"), Path("b.wav")]

    def fake_load(path: Path, *, target_sample_rate: int) -> dict:
        return {"path": path.name, "sampling_rate": target_sample_rate}

    monkeypatch.setattr(asr, "_load_audio", fake_load)
    loaded = list(asr._load_audio_paths(paths, target_sample_rate=16_000, workers=3))

    assert loaded == [
        {"path": "c.wav", "sampling_rate": 16_000},
        {"path": "a.wav", "sampling_rate": 16_000},
        {"path": "b.wav", "sampling_rate": 16_000},
    ]


def test_load_audio_downmixes_and_resamples_wav(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    soundfile = pytest.importorskip("soundfile")
    pytest.importorskip("soxr")
    path = tmp_path / "stereo.wav"
    source_rate = 8_000
    samples = np.column_stack(
        (
            np.full(source_rate, 0.75, dtype=np.float32),
            np.full(source_rate, 0.25, dtype=np.float32),
        )
    )
    soundfile.write(path, samples, source_rate, subtype="FLOAT")

    loaded = asr._load_audio(path, target_sample_rate=16_000)

    assert loaded["sampling_rate"] == 16_000
    assert loaded["array"].dtype == np.float32
    assert loaded["array"].shape == (16_000,)
    assert np.mean(loaded["array"]) == pytest.approx(0.5, abs=1e-3)


def test_transcriber_passes_audio_arrays_to_pipeline(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    soundfile = pytest.importorskip("soundfile")
    pytest.importorskip("soxr")
    path = tmp_path / "audio.wav"
    soundfile.write(path, np.zeros(800, dtype=np.float32), 8_000)
    received: list[dict] = []
    pipeline_kwargs: dict = {}

    class FakePipeline:
        def __call__(self, inputs, **kwargs):
            pipeline_kwargs.update(kwargs)
            received.extend(inputs)
            return iter(({"text": " ヨミ "},))

    transcriber = object.__new__(asr.WhisperTranscriber)
    transcriber._sample_rate = 16_000
    transcriber._audio_workers = 1
    transcriber._pipeline = FakePipeline()

    results = transcriber.transcribe([path])
    assert received == []
    assert list(results) == ["ヨミ"]
    assert len(received) == 1
    assert received[0]["sampling_rate"] == 16_000
    assert received[0]["array"].shape == (1_600,)
    assert pipeline_kwargs == {"chunk_length_s": 30}


def test_transcription_is_incremental_and_reusable(tmp_path: Path, monkeypatch) -> None:
    audio = [tmp_path / "a.wav", tmp_path / "b.wav"]
    for path in audio:
        path.touch()
    output = tmp_path / "transcriptions.kana.jsonl"
    monkeypatch.setattr(asr, "WhisperTranscriber", FakeTranscriber)
    FakeTranscriber.calls = 0

    metadata = asr.transcribe_audio(
        audio_paths=audio,
        output_path=output,
        value_field="yomi",
        configuration=configuration(),
    )
    assert FakeTranscriber.calls == 1
    assert metadata == {
        "model": "test-model",
        "language": "ja",
        "row_count": 2,
        "complete": True,
    }
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == [
        {"key": "a", "yomi": "ヨミa"},
        {"key": "b", "yomi": "ヨミb"},
    ]

    asr.transcribe_audio(
        audio_paths=audio,
        output_path=output,
        value_field="yomi",
        configuration=configuration(),
    )
    assert FakeTranscriber.calls == 1


def test_partial_transcription_resumes_only_missing_rows(
    tmp_path: Path, monkeypatch
) -> None:
    audio = [tmp_path / "a.wav", tmp_path / "b.wav"]
    for path in audio:
        path.touch()
    output = tmp_path / "transcriptions.kana.jsonl"
    output.write_text('{"key":"b","yomi":"既存"}\n', encoding="utf-8")
    monkeypatch.setattr(asr, "WhisperTranscriber", FakeTranscriber)
    FakeTranscriber.calls = 0

    asr.transcribe_audio(
        audio_paths=audio,
        output_path=output,
        value_field="yomi",
        configuration=configuration(),
    )
    assert FakeTranscriber.calls == 1
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == [
        {"key": "a", "yomi": "ヨミa"},
        {"key": "b", "yomi": "既存"},
    ]


def test_completed_rows_survive_interrupted_transcription(
    tmp_path: Path, monkeypatch
) -> None:
    audio = [tmp_path / "a.wav", tmp_path / "b.wav"]
    for path in audio:
        path.touch()
    output = tmp_path / "transcriptions.kana.jsonl"

    class FailingTranscriber:
        def __init__(self, configuration: asr.AsrConfiguration) -> None:
            pass

        def transcribe(self, paths: Iterable[Path]) -> Iterable[str]:
            yield "ヨミa"
            raise RuntimeError("inference failed")

    monkeypatch.setattr(asr, "WhisperTranscriber", FailingTranscriber)
    with pytest.raises(RuntimeError, match="after transcribing 1 of 2 files"):
        asr.transcribe_audio(
            audio_paths=audio,
            output_path=output,
            value_field="yomi",
            configuration=configuration(),
        )

    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"key": "a", "yomi": "ヨミa"}
    ]
    metadata = json.loads(output.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert metadata == {
        "model": "test-model",
        "language": "ja",
        "complete": False,
    }

    monkeypatch.setattr(asr, "WhisperTranscriber", FakeTranscriber)
    FakeTranscriber.calls = 0
    asr.transcribe_audio(
        audio_paths=audio,
        output_path=output,
        value_field="yomi",
        configuration=configuration(),
    )
    assert FakeTranscriber.calls == 1
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"key": "a", "yomi": "ヨミa"},
        {"key": "b", "yomi": "ヨミb"},
    ]


def test_cache_rejects_different_model(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.touch()
    output = tmp_path / "transcriptions.kana.jsonl"
    output.write_text('{"key":"a","yomi":"ヨミ"}\n', encoding="utf-8")
    output.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "model": "different-model",
                "language": "ja",
            }
        ),
        encoding="utf-8",
    )

    try:
        asr.transcribe_audio(
            audio_paths=[audio],
            output_path=output,
            value_field="yomi",
            configuration=configuration(),
        )
    except ValueError as error:
        assert "--force-transcribe" in str(error)
    else:
        raise AssertionError("Expected mismatched ASR cache to be rejected")
