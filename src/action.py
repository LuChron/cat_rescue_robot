"""
互动动作控制 — 找到猫后执行。
动作：play(玩耍), feed(投喂), photo(拍照), talk(安抚)
搜猫是自动的，不在此模块。
"""

from datetime import datetime
import time
from typing import List

from .motor import get_motor


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
    results = {}
    if "play" in actions:
        results["play"] = _play_with_cat()

    # ---- 第 4 步：投喂（如果 actions 包含 feed） ----
    if "feed" in actions:
        results["feed"] = _dispense_food()

    # ---- 第 5 步：语音安抚（如果 actions 包含 talk） ----
    if "talk" in actions:
        results["talk"] = _play_sound()

    # ---- 第 6 步：拍照（如果 actions 包含 photo） ----
    if "photo" in actions:
        results["photo"] = _capture_photo(breed, zone)

    # ---- 第 7 步：任务报告 ----
    _report(breed, zone, results)

    print(f"{'='*40}\n")
    return results


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
    """用机械臂小幅左右摆动陪猫玩耍，结束后回到初始姿态。"""
    motor = get_motor()
    if not motor.is_connected():
        print("[CARE/玩耍] 模拟机械臂小幅左右摆动...")
        time.sleep(0.25)
        return True

    print("[CARE/玩耍] 机械臂正在小幅左右摆动...")
    success = motor.execute_care_action("play")
    print(
        "[CARE/玩耍] 互动动作完成，机械臂已归位"
        if success
        else "[CARE/玩耍] 机械臂互动动作发送失败"
    )
    return success


def _dispense_food():
    """舵机投放零食。"""
    motor = get_motor()
    if not motor.is_connected():
        print("[CARE/投喂] 模拟开闸，投放猫零食...")
        time.sleep(0.25)
        return True

    print("[CARE/投喂] 正在放下机械臂并打开夹爪...")
    success = motor.execute_care_action("feed")
    print(
        "[CARE/投喂] 投喂动作序列完成"
        if success
        else "[CARE/投喂] 机械臂动作发送失败"
    )
    return success


def _play_sound():
    """播放安抚声音 / 猫叫。"""
    print("[CARE/安抚] 🔊 播放咕噜声 / 猫叫吸引注意...")
    # TODO: pygame.mixer / espeak 播放预录音频
    return True


def _capture_photo(breed: str, zone: str):
    """拍照存档。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{breed}_{zone}_{timestamp}.jpg"
    print(f"[CARE/拍照] 📸 保存照片: {filename}")
    # TODO: PiCamera / cv2 截图保存
    return True


def _report(breed: str, zone: str, results: dict[str, bool]):
    """任务报告 → Web 控制台。"""
    completed = [action for action, ok in results.items() if ok]
    failed = [action for action, ok in results.items() if not ok]
    action_str = "、".join(completed) or "无"
    msg = f"互动结果 — 在 {zone} 找到 {breed}，已完成: {action_str}"
    if failed:
        msg += f"，失败: {'、'.join(failed)}"
    print(f"[CARE/报告] {msg}")
