"""
护理动作控制 — 找到目标猫后执行。
包括：语音播报、靠近调整、投放食物、拍照存档、任务报告。
"""

from datetime import datetime
from typing import List


def run_care_actions(breed: str, zone: str, actions: List[str]):
    """按 actions 列表依次执行护理动作。"""

    print(f"\n{'='*40}")
    print(f"[CARE] 开始护理程序 — {breed} @ {zone}")
    print(f"[CARE] 动作列表: {actions}")

    # ---- 第 1 步：语音播报 ----
    _announce(breed, zone)

    # ---- 第 2 步：靠近调整 ----
    _approach()

    # ---- 第 3 步：投放食物（如果 actions 包含 feed） ----
    if "feed" in actions:
        _dispense_food()

    # ---- 第 4 步：拍照存档（如果 actions 包含 photo） ----
    if "photo" in actions or "feed" in actions:
        _capture_photo(breed, zone)

    # ---- 第 5 步：任务报告 ----
    _report(breed, zone, actions)

    print(f"{'='*40}\n")


def _announce(breed: str, zone: str):
    """语音播报。"""
    msg = f"目标猫已找到 — {breed}，位于 {zone}，开始护理程序。"
    print(f"[CARE/播报] {msg}")
    # TODO: 通过笔记本扬声器或树莓派 GPIO 播放语音


def _approach():
    """根据检测框大小微调靠近猫。"""
    print("[CARE/靠近] 微调位置...")
    # TODO: 读 bounding box 大小，太小就发 "w" 前进一小段


def _dispense_food():
    """舵机投放食物。"""
    print("[CARE/投喂] 投放食物...")
    # TODO: RPi.GPIO 控制舵机，打开 → 停留 1s → 关闭


def _capture_photo(breed: str, zone: str):
    """拍照存档。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{breed}_{zone}_{timestamp}.jpg"
    print(f"[CARE/拍照] 保存照片: {filename}")
    # TODO: 调用 PiCamera 拍照保存


def _report(breed: str, zone: str, actions: List[str]):
    """任务报告 → Web 控制台。"""
    action_str = "、".join(actions)
    msg = f"✅ 任务完成 — 在 {zone} 找到 {breed}，已执行: {action_str}"
    print(f"[CARE/报告] {msg}")
    # TODO: 发送 report 到 Web Console
