#!/usr/bin/env python3
"""Evaluate ASR text and final command accuracy on project microphone samples."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.asr import normalize_command_transcript, speech_to_text  # noqa: E402
from src.parser import parse_command  # noqa: E402


def _command_signature(text: str) -> str | None:
    try:
        command = parse_command(text, allow_llm=False)
    except ValueError:
        return None
    return json.dumps(command, ensure_ascii=False, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate WAV files listed as audio,language,expected in a CSV file."
    )
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    if not rows:
        print("Manifest contains no samples.", file=sys.stderr)
        return 2

    exact = 0
    command_correct = 0
    elapsed = 0.0
    for index, row in enumerate(rows, 1):
        audio = (args.manifest.parent / row["audio"]).resolve()
        expected = row["expected"].strip()
        language = row.get("language", "zh").strip() or None
        started = time.perf_counter()
        predicted = normalize_command_transcript(
            speech_to_text(str(audio), language=language)
        )
        duration = time.perf_counter() - started
        elapsed += duration
        exact_match = predicted.casefold() == expected.casefold()
        semantic_match = (
            _command_signature(predicted) is not None
            and _command_signature(predicted) == _command_signature(expected)
        )
        exact += exact_match
        command_correct += semantic_match
        status = "PASS" if semantic_match else "FAIL"
        print(
            f"{index:02d} {status} {duration:.2f}s | "
            f"expected={expected!r} predicted={predicted!r}"
        )

    count = len(rows)
    print(f"\nExact transcript accuracy: {exact / count:.1%} ({exact}/{count})")
    print(
        f"Command semantic accuracy: {command_correct / count:.1%} "
        f"({command_correct}/{count})"
    )
    print(f"Mean inference latency: {elapsed / count:.2f}s")
    return 0 if command_correct == count else 1


if __name__ == "__main__":
    raise SystemExit(main())
