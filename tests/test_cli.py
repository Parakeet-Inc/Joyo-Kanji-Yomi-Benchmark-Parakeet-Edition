import json
from pathlib import Path

import jkyb_eval.cli as cli
from jkyb_eval.cli import build_parser, main


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def dataset_row(
    key: str,
    reading: str,
    reading_category: str = "on_yomi",
) -> dict:
    return {
        "key": key,
        "text": "誤解した。",
        "tagged_text": "<誤>解した。",
        "yomi": f"{reading}カイシタ",
        "tagged_yomi": f"<{reading}>カイシタ",
        "reading_category": reading_category,
        "readings": {"natural": [reading], "marginal": []},
        "source": "original",
    }


def test_tts_defaults_to_consumer_safe_batch_size() -> None:
    args = build_parser().parse_args(
        ["tts", "audio", "--output-dir", "output", "--skip-text-cer"]
    )
    assert args.batch_size == 16


def test_g2p_command(tmp_path: Path, capsys) -> None:
    dataset = tmp_path / "dataset.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "results"
    write_jsonl(dataset, [dataset_row("誤_ゴ_0", "ゴ")])
    write_jsonl(predictions, [{"key": "誤_ゴ_0", "yomi": "ゴカイシタ"}])

    assert (
        main(
            [
                "g2p",
                str(predictions),
                "--dataset",
                str(dataset),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "summary.json").is_file()
    assert (output / "summary.md").is_file()
    assert (output / "details" / "all.jsonl").is_file()
    assert (output / "details" / "missing-inputs.jsonl").is_file()
    assert (output / "details" / "target-review.jsonl").is_file()
    assert (output / "details" / "sentence-mismatches.jsonl").is_file()
    assert not (output / "details" / "text-mismatches.jsonl").exists()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["dataset"] == str(dataset)
    assert summary["inputs"] == {
        "predictions": str(predictions),
        "yomi_field": "yomi",
    }
    assert summary["configuration"] == {"command": "g2p"}
    assert summary["coverage"] == {
        "input": "prediction",
        "expected": 1,
        "available": 1,
        "missing": 0,
        "rate": 1.0,
    }
    assert summary["metrics"]["target_kana_cer"] == {
        "raw": 0.0,
        "cer_at_1": 0.0,
    }
    assert summary["metrics"]["relaxed_target_kana_cer"] == {
        "raw": 0.0,
        "cer_at_1": 0.0,
    }
    assert summary["metrics"]["accuracy"] == {"rate": 1.0, "matches": 1}
    assert summary["metrics"]["relaxed_accuracy"] == {
        "rate": 1.0,
        "matches": 1,
    }
    assert summary["breakdowns"]["reading_category"] == {
        "on_yomi": {
            "row_count": 1,
            "metrics": {
                "target_kana_cer": {"raw": 0.0, "cer_at_1": 0.0},
                "relaxed_target_kana_cer": {"raw": 0.0, "cer_at_1": 0.0},
                "accuracy": {"rate": 1.0, "matches": 1},
                "relaxed_accuracy": {"rate": 1.0, "matches": 1},
            },
        }
    }
    assert summary["diagnostics"] == {
        "ambiguous_target_alignments": 0,
        "missing_predictions": 0,
        "extra_predictions": 0,
    }
    assert "results" not in summary
    assert "sha256" not in json.dumps(summary)
    assert "revision" not in json.dumps(summary)
    printed = capsys.readouterr().out
    assert "Evaluation complete." in printed
    assert "Rows: 1" in printed
    assert "Prediction coverage: 1 / 1 (100.000%)" in printed
    assert f"Summary: {output / 'summary.md'}" in printed
    assert f"Per-row details: {output / 'details'}" in printed
    assert "Target Kana-CER: 0.000%" in printed
    assert "Relaxed Target Kana-CER: 0.000%" in printed
    assert "Sentence Kana-CER: 0.000%" in printed
    assert "Accuracy: 100.000%" in printed
    assert "Relaxed Accuracy: 100.000%" in printed
    assert [
        line.split(":", 1)[0]
        for line in printed.splitlines()
        if line.startswith(
            (
                "Accuracy:",
                "Relaxed Accuracy:",
                "Target Kana-CER:",
                "Relaxed Target Kana-CER:",
                "Sentence Kana-CER:",
            )
        )
    ] == [
        "Accuracy",
        "Relaxed Accuracy",
        "Target Kana-CER",
        "Relaxed Target Kana-CER",
        "Sentence Kana-CER",
    ]
    details = [
        json.loads(line)
        for line in (output / "details" / "all.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert details[0]["reading_category"] == "on_yomi"
    assert "## Metrics by Reading Category" in (output / "summary.md").read_text(
        encoding="utf-8"
    )


def test_g2p_rejects_missing_prediction(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(dataset, [dataset_row("誤_ゴ_0", "ゴ")])
    write_jsonl(predictions, [])
    assert (
        main(
            [
                "g2p",
                str(predictions),
                "--dataset",
                str(dataset),
                "--output-dir",
                str(tmp_path / "results"),
            ]
        )
        == 2
    )


def test_g2p_allow_missing_scores_absent_prediction_as_error(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "results"
    write_jsonl(
        dataset,
        [
            dataset_row("誤_ゴ_0", "ゴ"),
            dataset_row("誤_ゴ_1", "ゴ"),
        ],
    )
    write_jsonl(predictions, [{"key": "誤_ゴ_0", "yomi": "ゴカイシタ"}])

    assert (
        main(
            [
                "g2p",
                str(predictions),
                "--dataset",
                str(dataset),
                "--output-dir",
                str(output),
                "--allow-missing",
            ]
        )
        == 0
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["row_count"] == 2
    assert summary["coverage"] == {
        "input": "prediction",
        "expected": 2,
        "available": 1,
        "missing": 1,
        "rate": 0.5,
    }
    assert summary["metrics"]["target_kana_cer"]["raw"] == 0.5
    assert summary["metrics"]["accuracy"] == {"rate": 0.5, "matches": 1}
    missing_rows = [
        json.loads(line)
        for line in (output / "details" / "missing-inputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["key"] for row in missing_rows] == ["誤_ゴ_1"]


def test_tts_organizes_transcriptions_and_details(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset.jsonl"
    audio = tmp_path / "audio"
    output = tmp_path / "results"
    audio.mkdir()
    (audio / "誤_ゴ_0.wav").touch()
    write_jsonl(dataset, [dataset_row("誤_ゴ_0", "ゴ")])
    transcription_paths: list[Path] = []

    def fake_transcribe_audio(
        *,
        audio_paths,
        output_path: Path,
        value_field: str,
        configuration,
        force: bool,
    ) -> dict:
        transcription_paths.append(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        value = "ゴカイシタ" if value_field == "yomi" else "誤解した。"
        write_jsonl(output_path, [{"key": "誤_ゴ_0", value_field: value}])
        return {"model": configuration.model}

    monkeypatch.setattr(cli, "transcribe_audio", fake_transcribe_audio)

    assert (
        main(
            [
                "tts",
                str(audio),
                "--dataset",
                str(dataset),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert transcription_paths == [
        output / "transcriptions" / "kana.jsonl",
        output / "transcriptions" / "text.jsonl",
    ]
    assert (output / "details" / "text-mismatches.jsonl").is_file()


def test_tts_rejects_missing_audio_by_default(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset.jsonl"
    audio = tmp_path / "audio"
    audio.mkdir()
    write_jsonl(dataset, [dataset_row("誤_ゴ_0", "ゴ")])
    transcribe_called = False

    def fake_transcribe_audio(**_kwargs) -> dict:
        nonlocal transcribe_called
        transcribe_called = True
        return {}

    monkeypatch.setattr(cli, "transcribe_audio", fake_transcribe_audio)

    assert (
        main(
            [
                "tts",
                str(audio),
                "--dataset",
                str(dataset),
                "--output-dir",
                str(tmp_path / "results"),
            ]
        )
        == 2
    )
    assert transcribe_called is False


def test_tts_allow_missing_scores_absent_audio_as_error(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    audio = tmp_path / "audio"
    output = tmp_path / "results"
    audio.mkdir()
    (audio / "誤_ゴ_0.wav").touch()
    write_jsonl(
        dataset,
        [
            dataset_row("誤_ゴ_0", "ゴ"),
            dataset_row("誤_ゴ_1", "ゴ"),
        ],
    )

    def fake_transcribe_audio(
        *,
        audio_paths,
        output_path: Path,
        value_field: str,
        configuration,
        force: bool,
    ) -> dict:
        del force
        paths = list(audio_paths)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        value = "ゴカイシタ" if value_field == "yomi" else "誤解した。"
        write_jsonl(
            output_path,
            [{"key": path.stem, value_field: value} for path in paths],
        )
        return {"model": configuration.model}

    monkeypatch.setattr(cli, "transcribe_audio", fake_transcribe_audio)

    assert (
        main(
            [
                "tts",
                str(audio),
                "--dataset",
                str(dataset),
                "--output-dir",
                str(output),
                "--allow-missing",
            ]
        )
        == 0
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["row_count"] == 2
    assert summary["coverage"] == {
        "input": "audio",
        "expected": 2,
        "available": 1,
        "missing": 1,
        "rate": 0.5,
    }
    assert summary["metrics"]["target_kana_cer"]["raw"] == 0.5
    assert summary["metrics"]["sentence_kana_cer"]["raw"] == 0.5
    assert summary["metrics"]["text_cer"]["raw"] == 0.5
    assert summary["metrics"]["accuracy"] == {"rate": 0.5, "matches": 1}
    missing_rows = [
        json.loads(line)
        for line in (output / "details" / "missing-inputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["key"] for row in missing_rows] == ["誤_ゴ_1"]
    assert missing_rows[0]["prediction_missing"] is True
    assert missing_rows[0]["text_prediction_missing"] is True
    captured = capsys.readouterr()
    assert "warning: Missing audio for 1 keys" in captured.err
    assert "Audio coverage: 1 / 2 (50.000%)" in captured.out
