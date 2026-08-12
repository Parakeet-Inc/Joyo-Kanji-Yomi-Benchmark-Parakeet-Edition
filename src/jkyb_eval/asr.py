"""Optional Whisper transcription support for end-to-end TTS evaluation."""

from __future__ import annotations

import json
import sys
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl, write_json, write_jsonl

WHISPER_CHUNK_LENGTH_SECONDS = 30


@dataclass(frozen=True)
class AsrConfiguration:
    model: str
    device: str
    batch_size: int
    language: str = "ja"
    num_workers: int = 0 if sys.platform == "win32" else 1
    audio_workers: int = 2
    max_new_tokens: int | None = None


def _generation_kwargs(configuration: AsrConfiguration) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "language": configuration.language,
        "task": "transcribe",
    }
    if configuration.max_new_tokens is not None:
        kwargs["max_new_tokens"] = configuration.max_new_tokens
    return kwargs


def _load_audio(path: Path, *, target_sample_rate: int) -> dict[str, Any]:
    try:
        import soundfile
        import soxr
    except ImportError as error:
        raise RuntimeError(
            "TTS evaluation requires the ASR dependencies. Run `uv sync --extra asr`."
        ) from error

    try:
        samples, source_sample_rate = soundfile.read(
            path,
            dtype="float32",
            always_2d=True,
        )
    except Exception as error:
        raise RuntimeError(f"Could not read WAV file: {path}") from error
    if samples.shape[0] == 0:
        raise RuntimeError(f"WAV file is empty: {path}")

    mono = samples.mean(axis=1, dtype="float32")
    if source_sample_rate != target_sample_rate:
        mono = soxr.resample(
            mono,
            source_sample_rate,
            target_sample_rate,
            quality="HQ",
        )
    return {
        "array": mono.astype("float32", copy=False),
        "sampling_rate": target_sample_rate,
    }


def _load_audio_paths(
    paths: Iterable[Path],
    *,
    target_sample_rate: int,
    workers: int,
) -> Iterable[dict[str, Any]]:
    if workers < 1:
        raise ValueError("Audio worker count must be positive")
    if workers == 1:
        for path in paths:
            yield _load_audio(path, target_sample_rate=target_sample_rate)
        return

    load = partial(_load_audio, target_sample_rate=target_sample_rate)
    path_iterator = iter(paths)
    pending: deque[Future[dict[str, Any]]] = deque()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(workers * 2):
            try:
                pending.append(executor.submit(load, next(path_iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(load, next(path_iterator)))
            except StopIteration:
                pass


class WhisperTranscriber:
    """Lazy wrapper around a Transformers Whisper pipeline."""

    def __init__(self, configuration: AsrConfiguration) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
                pipeline,
            )
        except ImportError as error:
            raise RuntimeError(
                "TTS evaluation requires the ASR dependencies. Run "
                "`uv sync --extra asr`."
            ) from error

        device = configuration.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        is_local = Path(configuration.model).expanduser().exists()

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            configuration.model,
            dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=is_local,
        ).to(device)
        processor = AutoProcessor.from_pretrained(
            configuration.model,
            local_files_only=is_local,
        )
        self._sample_rate = int(processor.feature_extractor.sampling_rate)
        self._audio_workers = configuration.audio_workers
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=dtype,
            device=device,
            batch_size=configuration.batch_size,
            num_workers=configuration.num_workers,
            generate_kwargs=_generation_kwargs(configuration),
        )

    def transcribe(self, paths: Iterable[Path]) -> Iterable[str]:
        inputs = _load_audio_paths(
            paths,
            target_sample_rate=self._sample_rate,
            workers=self._audio_workers,
        )
        raw_results = self._pipeline(
            inputs,
            chunk_length_s=WHISPER_CHUNK_LENGTH_SECONDS,
        )
        for result in raw_results:
            if not isinstance(result, dict) or not isinstance(result.get("text"), str):
                raise RuntimeError(f"Unexpected ASR output: {result!r}")
            yield result["text"].strip()


def _existing_transcriptions(path: Path, *, value_field: str) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    existing: dict[str, str] = {}
    for line_number, row in enumerate(rows, start=1):
        key = row.get("key")
        value = row.get(value_field)
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"{path}:{line_number}: invalid cached transcription")
        if key in existing:
            raise ValueError(f"{path}:{line_number}: duplicate key {key!r}")
        existing[key] = value
    return existing


def transcribe_audio(
    *,
    audio_paths: Iterable[Path],
    output_path: Path,
    value_field: str,
    configuration: AsrConfiguration,
    force: bool = False,
) -> dict[str, Any]:
    paths = list(audio_paths)
    expected_keys = [path.stem for path in paths]
    if len(expected_keys) != len(set(expected_keys)):
        raise ValueError("Audio file stems must be unique")

    metadata_path = output_path.with_suffix(".meta.json")
    existing_metadata: dict[str, Any] = {}
    if metadata_path.exists() and not force:
        loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_metadata, dict):
            raise ValueError(f"Invalid transcription metadata: {metadata_path}")
        existing_metadata = loaded_metadata
        expected_configuration = {
            "model": configuration.model,
            "language": configuration.language,
            **(
                {"max_new_tokens": configuration.max_new_tokens}
                if configuration.max_new_tokens is not None
                else {}
            ),
        }
        mismatches = {
            key: (existing_metadata.get(key), expected)
            for key, expected in expected_configuration.items()
            if existing_metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "Cached transcription configuration differs from this run: "
                f"{mismatches!r}. Use --force-transcribe to replace it."
            )

    existing = (
        {} if force else _existing_transcriptions(output_path, value_field=value_field)
    )
    unknown_cached = set(existing) - set(expected_keys)
    if unknown_cached:
        raise ValueError(
            f"Cached transcription contains unknown keys: "
            f"{sorted(unknown_cached)[:10]!r}"
        )
    pending = [path for path in paths if path.stem not in existing]

    if pending:
        try:
            from tqdm import tqdm
        except ImportError as error:
            raise RuntimeError(
                "TTS evaluation requires the ASR dependencies. Run "
                "`uv sync --extra asr`."
            ) from error

        transcriber = WhisperTranscriber(configuration)
        metadata = {
            "model": configuration.model,
            "language": configuration.language,
            **(
                {"max_new_tokens": configuration.max_new_tokens}
                if configuration.max_new_tokens is not None
                else {}
            ),
            "complete": False,
        }
        write_json(metadata_path, metadata)
        values = transcriber.transcribe(pending)
        progress = tqdm(
            zip(pending, values, strict=True),
            total=len(paths),
            initial=len(existing),
            desc=f"Transcribing ({configuration.model})",
            unit="audio",
            dynamic_ncols=True,
        )
        mode = "w" if force else "a"
        try:
            with output_path.open(mode, encoding="utf-8") as handle, progress:
                for path, value in progress:
                    existing[path.stem] = value
                    handle.write(
                        json.dumps(
                            {"key": path.stem, value_field: value},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    handle.flush()
        except Exception as error:
            raise RuntimeError(
                f"ASR failed after transcribing {len(existing):,} of {len(paths):,} files"
            ) from error

    if set(existing) != set(expected_keys):
        missing = sorted(set(expected_keys) - set(existing))
        raise RuntimeError(f"Transcription incomplete; missing keys: {missing[:10]!r}")

    write_jsonl(
        output_path,
        ({"key": key, value_field: existing[key]} for key in expected_keys),
    )
    metadata = {
        "model": configuration.model,
        "language": configuration.language,
        **(
            {"max_new_tokens": configuration.max_new_tokens}
            if configuration.max_new_tokens is not None
            else {}
        ),
        "row_count": len(existing),
        "complete": True,
    }
    write_json(metadata_path, metadata)
    return metadata
