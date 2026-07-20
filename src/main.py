"""
主入口 — 串联三个独立模块。
用法: python -m src.main
"""

import sys

from .parser import parse_command
from .controller import execute_mission


def run_voice_mode():
    """模块 1 → 模块 2 → 模块 3，完整语音管线。"""
    from .asr import record_audio, speech_to_text

    print("🎤 请说出指令...")
    audio_path = record_audio(duration=5)

    text = speech_to_text(audio_path)
    print(f"📝 转录: {text}")

    command = parse_command(text)
    print(f"🎯 解析: 品种={command['breed']}, 区域={command['zone']}, "
          f"动作={command['actions']}")

    execute_mission(command)


def run_text_mode(command_str: str):
    """模块 2 → 模块 3，跳过语音，直接输入文字指令。"""
    command = parse_command(command_str)
    print(f"🎯 解析: {command}")
    execute_mission(command)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 文本模式: python -m src.main "persian zoneC feed"
        run_text_mode(" ".join(sys.argv[1:]))
    else:
        # 语音模式
        run_voice_mode()
