"""Command-line interface for G2P and TTS evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from .asr import AsrConfiguration, transcribe_audio
from .constants import (
    ASR_MAX_NEW_TOKENS,
    DEFAULT_ASR_BATCH_SIZE,
    DEFAULT_DATASET_REPO,
    DEFAULT_KANA_MODEL,
    DEFAULT_TEXT_MODEL,
)
from .io import load_benchmark, resolve_dataset
from .run import run_evaluation


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Local JKYB-Parakeet JSONL. The Hugging Face dataset is used by default.",
    )
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)


def _resolve_from_args(args: argparse.Namespace):
    return resolve_dataset(
        args.dataset,
        repository=args.dataset_repo,
    )


def _print_evaluation_summary(summary: dict[str, Any], output_dir: Path) -> None:
    metrics = summary["metrics"]
    coverage = summary["coverage"]
    coverage_label = {
        "audio": "Audio",
        "prediction": "Prediction",
    }.get(coverage["input"], str(coverage["input"]).title())
    resolved_output = output_dir.expanduser().resolve()
    print("Evaluation complete.")
    print(f"Rows: {summary['row_count']:,}")
    print(f"Summary: {resolved_output / 'summary.md'}")
    print(f"Per-row details: {resolved_output / 'details'}")
    print(
        f"{coverage_label} coverage: {coverage['available']:,} / "
        f"{coverage['expected']:,} ({coverage['rate']:.3%})"
    )
    print(f"Accuracy: {metrics['accuracy']['rate']:.3%}")
    print(f"Relaxed Accuracy: {metrics['relaxed_accuracy']['rate']:.3%}")
    print(f"Target Kana-CER: {metrics['target_kana_cer']['raw']:.3%}")
    print(f"Relaxed Target Kana-CER: {metrics['relaxed_target_kana_cer']['raw']:.3%}")
    print(f"Sentence Kana-CER: {metrics['sentence_kana_cer']['raw']:.3%}")
    if "text_cer" in metrics:
        print(f"Text CER: {metrics['text_cer']['raw']:.3%}")


def _run_g2p(args: argparse.Namespace) -> int:
    dataset = _resolve_from_args(args)
    summary = run_evaluation(
        dataset=dataset,
        prediction_path=args.predictions,
        output_dir=args.output_dir,
        yomi_field=args.yomi_field,
        allow_missing=args.allow_missing,
        allow_extra=args.allow_extra,
        additional_config={"command": "g2p"},
        progress=sys.stderr.isatty(),
    )
    _print_evaluation_summary(summary, args.output_dir)
    return 0


def _run_tts(args: argparse.Namespace) -> int:
    dataset = _resolve_from_args(args)
    rows = load_benchmark(dataset.path)
    audio_dir = args.audio_dir.expanduser().resolve()
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    audio_by_key = {
        path.stem: path for path in audio_dir.glob("*.wav") if path.is_file()
    }
    expected_keys = [row.key for row in rows]
    missing_audio = [key for key in expected_keys if key not in audio_by_key]
    extra_audio = sorted(set(audio_by_key) - set(expected_keys))
    if missing_audio:
        message = (
            f"Missing audio for {len(missing_audio)} keys; examples: "
            f"{missing_audio[:10]!r}"
        )
        if not args.allow_missing:
            raise ValueError(
                f"{message}. Rerun with --allow-missing to score these rows "
                "as empty outputs."
            )
        print(
            f"warning: {message}. These rows will be scored as empty outputs.",
            file=sys.stderr,
        )
    if extra_audio and not args.allow_extra_audio:
        raise ValueError(
            f"Audio directory contains {len(extra_audio)} unknown WAV files; "
            f"examples: {extra_audio[:10]!r}"
        )
    ordered_audio = [audio_by_key[key] for key in expected_keys if key in audio_by_key]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transcription_dir = output_dir / "transcriptions"
    kana_path = transcription_dir / "kana.jsonl"
    kana_metadata = transcribe_audio(
        audio_paths=ordered_audio,
        output_path=kana_path,
        value_field="yomi",
        configuration=AsrConfiguration(
            model=args.kana_model,
            device=args.device,
            batch_size=args.batch_size,
            max_new_tokens=ASR_MAX_NEW_TOKENS,
        ),
        force=args.force_transcribe,
    )

    text_path: Path | None = None
    text_metadata = None
    if not args.skip_text_cer:
        text_path = transcription_dir / "text.jsonl"
        text_metadata = transcribe_audio(
            audio_paths=ordered_audio,
            output_path=text_path,
            value_field="text",
            configuration=AsrConfiguration(
                model=args.text_model,
                device=args.device,
                batch_size=args.batch_size,
                max_new_tokens=ASR_MAX_NEW_TOKENS,
            ),
            force=args.force_transcribe,
        )

    summary = run_evaluation(
        dataset=dataset,
        prediction_path=kana_path,
        output_dir=output_dir,
        text_prediction_path=text_path,
        allow_missing=args.allow_missing,
        additional_config={
            "command": "tts",
            "kana_asr_model": kana_metadata["model"],
            **(
                {"text_asr_model": text_metadata["model"]}
                if text_metadata is not None
                else {}
            ),
        },
        input_metadata={"audio_dir": str(audio_dir)},
        input_kind="audio",
        progress=sys.stderr.isatty(),
    )
    _print_evaluation_summary(summary, output_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jkyb-eval",
        description="Evaluate G2P predictions and TTS audio with JKYB-Parakeet.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    g2p = subparsers.add_parser(
        "g2p", help="Evaluate full-sentence reading predictions."
    )
    g2p.add_argument("predictions", type=Path)
    g2p.add_argument("--output-dir", type=Path, required=True)
    g2p.add_argument("--yomi-field", default="yomi")
    g2p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Score missing predictions as empty outputs instead of stopping.",
    )
    g2p.add_argument("--allow-extra", action="store_true")
    _add_dataset_arguments(g2p)
    g2p.set_defaults(handler=_run_g2p)

    tts = subparsers.add_parser(
        "tts", help="Evaluate synthesized WAV files through ASR."
    )
    tts.add_argument("audio_dir", type=Path)
    tts.add_argument("--output-dir", type=Path, required=True)
    tts.add_argument("--device", default="auto")
    tts.add_argument("--batch-size", type=int, default=DEFAULT_ASR_BATCH_SIZE)
    tts.add_argument("--kana-model", default=DEFAULT_KANA_MODEL)
    tts.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    tts.add_argument("--skip-text-cer", action="store_true")
    tts.add_argument("--force-transcribe", action="store_true")
    tts.add_argument(
        "--allow-missing",
        action="store_true",
        help="Score missing audio as empty outputs instead of stopping.",
    )
    tts.add_argument("--allow-extra-audio", action="store_true")
    _add_dataset_arguments(tts)
    tts.set_defaults(handler=_run_tts)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
