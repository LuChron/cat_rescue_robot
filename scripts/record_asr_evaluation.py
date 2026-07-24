#!/usr/bin/env python3
"""Interactively record the standard ASR command evaluation set."""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.asr import record_audio  # noqa: E402


COMMANDS = [
    ("zh_forward_10.wav", "zh", "向前走十厘米"),
    ("zh_backward_20.wav", "zh", "后退二十厘米"),
    ("zh_turn_360.wav", "zh", "旋转三百六十度"),
    ("zh_raise_arm.wav", "zh", "抬起机械臂"),
    ("zh_zone_a.wav", "zh", "去A区"),
    ("en_forward_10.wav", "en", "move forward 10 cm"),
    ("en_turn_left_90.wav", "en", "turn left 90 degrees"),
    ("en_stop.wav", "en", "stop"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "asr_evaluation",
    )
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args()

    sample_dir = args.output / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.csv"

    print("Use the same microphone and speaking distance as the robot demo.")
    print("Press Enter for each prompt, then speak once during the recording window.")
    for index, (filename, _language, expected) in enumerate(COMMANDS, 1):
        input(f"\n[{index}/{len(COMMANDS)}] Press Enter, then say: {expected}")
        destination = sample_dir / filename
        print(f"Recording for {args.duration:.1f} seconds...")
        record_audio(duration=args.duration, filename=str(destination))

    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("audio", "language", "expected"))
        for filename, language, expected in COMMANDS:
            writer.writerow((f"samples/{filename}", language, expected))

    print(f"\nSaved evaluation manifest: {manifest}")
    print(f"Run: python scripts/evaluate_asr.py {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
