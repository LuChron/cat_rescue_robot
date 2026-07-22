"""
互动动作控制 — 找到猫后执行。
动作：play(玩耍), feed(投喂), photo(拍照), talk(安抚)
搜猫是自动的，不在此模块。
"""

from datetime import datetime
from typing import List


def run_care_actions(breed: str, zone: str, actions: List[str]):
    """按 actions 列表依次执行互动动作。"""

    print(f"\n{'='*40}")
    print(f"[CARE] 开始互动程序 — {breed} @ {zone}")
    print(f"[CARE] 动作列表: {actions}")

    # ---- 第 1 步：语音播报 ----
    _announce(breed, zone)

    # ---- 第 2 步：靠近调整 ----
    _approach()

    # ---- 第 3 步：陪玩（如果 actions 包含 play） ----
    if "play" in actions:
        _play_with_cat()

    # ---- 第 4 步：投喂（如果 actions 包含 feed） ----
    if "feed" in actions:
        _dispense_food()

    # ---- 第 5 步：语音安抚（如果 actions 包含 talk） ----
    if "talk" in actions:
        _play_sound()

    # ---- 第 6 步：拍照（如果 actions 包含 photo） ----
    if "photo" in actions:
        _capture_photo(breed, zone)

    # ---- 第 7 步：任务报告 ----
    _report(breed, zone, actions)

    print(f"{'='*40}\n")


def _announce(breed: str, zone: str):
    """语音播报。"""
    msg = f"目标猫已找到 — {breed}，位于 {zone}，开始互动程序。"
    print(f"[CARE/播报] {msg}")
    # TODO: 树莓派扬声器 TTS 播报


def _approach():
    """根据检测框大小微调靠近猫。"""
    print("[CARE/靠近] 微调接近中...")
    # TODO: 读 YOLO bounding box，box 太小则前进一小段


def _play_with_cat():
    """逗猫棒 / 激光笔互动。"""
    print("[CARE/玩耍] 🎣 激光笔画圈、逗猫棒摆动...")
    # TODO: 舵机驱动逗猫棒 / GPIO 控制激光笔随机运动


def _dispense_food():
    """舵机投放零食。"""
    print("[CARE/投喂] 🍖 舵机开闸，投放猫零食...")
    # TODO: RPi.GPIO 控制舵机，打开 → 停留 1s → 关闭


def _play_sound():
    """播放安抚声音 / 猫叫。"""
    print("[CARE/安抚] 🔊 播放咕噜声 / 猫叫吸引注意...")
    # TODO: pygame.mixer / espeak 播放预录音频


def _capture_photo(breed: str, zone: str):
    """拍照存档。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{breed}_{zone}_{timestamp}.jpg"
    print(f"[CARE/拍照] 📸 保存照片: {filename}")
    # TODO: PiCamera / cv2 截图保存


def _report(breed: str, zone: str, actions: List[str]):
    """任务报告 → Web 控制台。"""
    action_str = "、".join(actions)
    msg = f"✅ 互动完成 — 在 {zone} 找到 {breed}，已执行: {action_str}"
    print(f"[CARE/报告] {msg}")
