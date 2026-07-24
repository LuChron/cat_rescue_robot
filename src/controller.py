"""
模块 3: 导航控制器 — {breed, zone, actions} → 机器人执行
zone 可选：未指定时自动探索所有猫区。

通过 get_state() 暴露实时状态，供 Web 前端轮询。
"""

import math
import threading
import time
from typing import Optional

from .planner import DEFAULT_MAP_PATH, load_map, get_waypoints, get_cat_zones, plan_exploration_route
from .motor import get_motor, MotorController
from .camera import get_latest_detection, reset_detection, set_search_active

GENERIC_CAT_LABELS = {"cat", "猫"}
GENERIC_ANIMAL_LABELS = {"animal", "动物", "宠物"}
MIN_BREED_CONFIDENCE = 0.45
BREED_DISPLAY_NAMES = {
    "波斯猫": "Persian",
    "布偶猫": "Ragdoll",
    "斯芬克斯猫": "Sphynx",
    "新加坡猫": "Singapura",
    "兔狲": "Pallas cat",
    "cat": "any cat",
    "dog": "dog",
    "bird": "bird",
    "animal": "any animal",
}

# 转弯参数
HEADING_TOLERANCE = 15.0   # 航向误差容限（度）
TURN_TIMEOUT = 8.0         # 转弯超时（秒）


class NavigationController:
    """导航状态机。所有状态变更写入 self._state，可从外部轮询。"""

    def __init__(self, map_path: str = str(DEFAULT_MAP_PATH)):
        self._map_path = map_path
        self.map_data = load_map(map_path)
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()

        # ---- 共享状态（Web 前端通过 get_state() 读取） ----
        self._state: dict = {
            "mode": "idle",            # idle | running | success | failed | listening
            "nav_state": "IDLE",
            "current_node": "start",
            "transcript": "",
            "command": None,           # {breed, zone, actions}
            "route": [],
            "cat_found": False,
            "cat_breed": None,
            "cat_zone": None,
            "actions_done": [],
            "current_action": "",       # 当前正在执行的护理动作
            "explored_zones": [],
            "log": [],
        }

        # ---- 内部状态 ----
        self._waypoints: list[dict] = []
        self._current_wp_idx: int = 0
        self._command: Optional[dict] = None
        self._exploring: bool = False

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """返回当前状态的浅拷贝，供前端轮询。"""
        with self._lock:
            return {
                "mode": self._state["mode"],
                "nav_state": self._state["nav_state"],
                "current_node": self._state["current_node"],
                "transcript": self._state["transcript"],
                "command": self._state["command"],
                "route": list(self._state["route"]),
                "cat_found": self._state["cat_found"],
                "cat_breed": self._state["cat_breed"],
                "cat_zone": self._state["cat_zone"],
                "actions_done": list(self._state["actions_done"]),
                "current_action": self._state["current_action"],
                "explored_zones": list(self._state["explored_zones"]),
                "log": list(self._state["log"]),
            }

    def is_busy(self) -> bool:
        with self._lock:
            return (
                self._state["mode"] in ("running", "listening", "stopping")
                or self._state["nav_state"] != "IDLE"
            )

    def set_mode(self, mode: str):
        """供 server 设置顶层模式（如 listening）。"""
        with self._lock:
            self._state["mode"] = mode

    def log(self, msg: str):
        """供 server 写入日志。"""
        self._log(msg)

    def execute_mission(self, command: dict, transcript: str = ""):
        """模块 3 入口：接收 {breed, zone, actions}，同步执行完整任务。
        应在后台线程中调用以免阻塞调用方。"""
        with self._lock:
            current_node = self._state["current_node"]
            if current_node not in self.map_data["nodes"]:
                current_node = "start"
            # 重规划可能临时删除边；每个新任务都从原始地图开始。
            self.map_data = load_map(self._map_path)
            self._cancel_event.clear()
            self._command = command
            self._state.update(
                mode="running",
                nav_state="PLANNING",
                current_node=current_node,
                transcript=transcript,
                command=command,
                route=[],
                cat_found=False,
                cat_breed=None,
                cat_zone=None,
                actions_done=[],
                current_action="",
                explored_zones=[],
            )
            self._waypoints = []
            self._current_wp_idx = 0
            self._exploring = False

        try:
            self._run_state_machine()
        except Exception as e:
            get_motor().stop()
            self._log(f"[ERROR] Mission failed unexpectedly: {e}")
            # Keep the error visible but release the controller for a retry.
            self._set(nav_state="IDLE", mode="failed", current_action="")

    def cancel_mission(self, reason: str = "Stopped by user"):
        """立即停车并取消当前自主任务。"""
        self._cancel_event.set()
        get_motor().stop()
        with self._lock:
            mission_running = self._state["nav_state"] != "IDLE"
            was_busy = mission_running or self._state["mode"] in ("running", "listening")
            self._state.update(
                mode="stopping" if mission_running else "idle",
                nav_state="STOPPING" if mission_running else "IDLE",
                current_action="",
            )
        if was_busy:
            self._log(f"[STOP] {reason}")

    # ------------------------------------------------------------------
    # 内部：日志 & 状态写入
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        with self._lock:
            self._state["log"].append(msg)
            if len(self._state["log"]) > 50:
                self._state["log"] = self._state["log"][-50:]
        print(msg)

    def _set(self, **kwargs):
        with self._lock:
            self._state.update(kwargs)

    # ------------------------------------------------------------------
    # 状态机主循环
    # ------------------------------------------------------------------

    def _run_state_machine(self):
        while True:
            if self._cancel_event.is_set():
                get_motor().stop()
                self._set(nav_state="IDLE", mode="idle", current_action="")
                return
            with self._lock:
                st = self._state["nav_state"]

            if st == "PLANNING":
                self._do_planning()
            elif st == "TURNING":
                self._do_turning()
            elif st == "DRIVING":
                self._do_driving()
            elif st == "OBSTACLE_WAIT":
                self._do_obstacle_wait()
            elif st == "REPLANNING":
                self._do_replanning()
            elif st == "ARRIVED":
                self._do_arrived()
            elif st == "SEARCHING":
                self._do_searching()
            elif st == "SUCCESS":
                self._do_success()
                return
            elif st == "FAILED":
                self._do_failed()
                return
            elif st == "IDLE":
                return
            else:
                self._log(f"[ERROR] Unknown state: {st}")
                return

    # ------------------------------------------------------------------
    # 各状态实现
    # ------------------------------------------------------------------

    def _do_planning(self):
        breed = self._command.get("breed")
        zone = self._command.get("zone")
        distance_cm = self._command.get("distance_cm")
        turn_deg = self._command.get("turn_deg")
        manual_key = self._command.get("manual_key")
        actions = self._command.get("actions", [])
        with self._lock:
            start_node = self._state["current_node"]

        if "return" in actions:
            zone = "start"
            breed = None
            self._log(f"[PLANNER] Return route: {start_node} -> start")

        # ---- 手动驾驶指令 → 直接发到小车 ----
        if manual_key:
            action = self._command.get("manual_action", "down")
            self._log(f"[MANUAL] Key command: {action} {manual_key}")
            motor = get_motor()
            motor.send_key_event(action, manual_key)
            self._set(nav_state="IDLE", mode="idle")
            return

        # ---- 简单移动指令（不需导航） ----
        if distance_cm:
            direction = "forward" if "forward" in self._command.get("actions", []) else "backward"
            self._log(f"[PLANNER] Direct move: {direction} {distance_cm} cm")
            motor = get_motor()
            if direction == "forward":
                completed = motor.forward(
                    distance_cm, cancel_event=self._cancel_event
                )
            else:
                completed = motor.backward(
                    distance_cm, cancel_event=self._cancel_event
                )
            if not completed:
                self._log("[PLANNER] Move stopped before completion")
                self._set(nav_state="IDLE", mode="idle")
                return
            self._log("[PLANNER] Move complete")
            self._set(nav_state="IDLE", mode="idle")
            return

        if turn_deg:
            direction = "left" if "turn_left" in self._command.get("actions", []) else "right"
            self._log(f"[PLANNER] Direct turn: {direction} {turn_deg} deg")
            completed = self._execute_direct_turn(direction, turn_deg)
            if not completed:
                self._log("[PLANNER] Turn stopped before completion")
                self._set(nav_state="IDLE", mode="idle")
                return
            self._log("[PLANNER] Turn complete")
            self._set(nav_state="IDLE", mode="idle")
            return

        # ---- 导航指令 ----
        # breed=None, zone=X → 只导航不找猫
        # breed=X, zone=Y → 导航到Y找X
        # breed=X, zone=None → 探索所有猫区找X

        if zone:
            waypoints = get_waypoints(self.map_data, start_node, zone)
            if waypoints is None:
                self._log(f"[PLANNER] No route: {start_node} -> {zone}")
                self._set(nav_state="FAILED", mode="failed")
                return
            self._exploring = False
            target_name = BREED_DISPLAY_NAMES.get(breed, breed)
            self._log(f"[PLANNER] Route to {zone}: {' -> '.join(w['id'] for w in waypoints)}" +
                      (f" searching for {target_name}" if breed else " (navigation only)"))
        elif breed:
            waypoints = plan_exploration_route(self.map_data, start_node)
            if waypoints is None:
                self._log("[PLANNER] No searchable zones available")
                self._set(nav_state="FAILED", mode="failed")
                return
            self._exploring = True
            cat_ids = [
                w["id"] for w in waypoints
                if self.map_data["nodes"].get(w["id"], {}).get("type") == "cat_zone"
            ]
            target_name = BREED_DISPLAY_NAMES.get(breed, breed)
            self._log(f"[PLANNER] Search for {target_name}: {' -> '.join(w['id'] for w in waypoints)}")
            self._log(f"[PLANNER] Search zones: {cat_ids}")
        else:
            self._log("[PLANNER] No action requested")
            self._set(nav_state="IDLE", mode="idle")
            return

        self._waypoints = waypoints
        self._current_wp_idx = 1  # 跳过 start
        self._set(
            route=[w["id"] for w in waypoints],
            current_node=start_node,
            nav_state="TURNING",
        )

    def _execute_direct_turn(self, direction: str, turn_deg: float) -> bool:
        """Turn by angle using heading feedback, with a timed fallback."""
        motor = get_motor()
        if direction == "left":
            motor.start_turn_left()
        else:
            motor.start_turn_right()

        started = time.monotonic()
        deadline = started + max(TURN_TIMEOUT, turn_deg / 30.0 + 3.0)
        last_heading = motor.get_heading() % 360
        rotated = 0.0
        heading_changed = False

        while time.monotonic() < deadline:
            if self._cancel_event.wait(0.05):
                motor.stop()
                return False

            heading = motor.get_heading() % 360
            step = (heading - last_heading + 180) % 360 - 180
            if abs(step) > 1.0:
                heading_changed = True
            rotated += abs(step)
            last_heading = heading
            if rotated >= max(1.0, turn_deg - 5.0):
                motor.stop()
                return True

            elapsed = time.monotonic() - started
            if not heading_changed and elapsed >= 1.5:
                remaining = max(0.0, turn_deg / 60.0 - elapsed)
                cancelled = self._cancel_event.wait(remaining)
                motor.stop()
                return not cancelled

        motor.stop()
        self._log(
            f"[PLANNER] Turn timed out after {rotated:.0f}/{turn_deg:.0f} deg"
        )
        return False

    def _check_manual_override(self) -> bool:
        """如果手动驾驶介入过，暂停并等结束，从当前位置继续。"""
        motor = get_motor()
        if not motor.manual_active(grace=1.5):
            return False

        self._log("[PAUSE] Manual control interrupted autonomous navigation")
        self._set(nav_state="PAUSED")
        # 等手动结束
        while motor.manual_active(grace=1.5):
            time.sleep(0.2)
        self._log("[RESUME] Manual control ended; resuming route")

        # 保留剩余路径，从下一个 waypoint 继续
        if self._current_wp_idx < len(self._waypoints):
            self._set(nav_state="TURNING")
        else:
            # 路径已走完，重规划
            zone = self._command.get("zone") if self._command else None
            rest = self._waypoints[self._current_wp_idx - 1]["id"] if self._current_wp_idx > 0 else "start"
            if zone:
                waypoints = get_waypoints(self.map_data, rest, zone)
            else:
                waypoints = plan_exploration_route(self.map_data, rest)
            if waypoints:
                self._waypoints = waypoints
                self._current_wp_idx = 1
                self._set(route=[w["id"] for w in waypoints], nav_state="TURNING")
            else:
                self._set(nav_state="FAILED", mode="failed")
        return True

    def _do_turning(self):
        if self._check_manual_override():
            return
        if self._current_wp_idx >= len(self._waypoints):
            self._set(nav_state="ARRIVED")
            return

        target = self._waypoints[self._current_wp_idx]
        motor = get_motor()

        # 计算目标航向
        prev_idx = max(0, self._current_wp_idx - 1)
        prev = self._waypoints[prev_idx]
        dx = target["x"] - prev["x"]
        dy = target["y"] - prev["y"]
        target_heading = math.degrees(math.atan2(dy, dx)) % 360

        # 当前航向
        current = motor.get_heading() % 360

        # 计算需要转的角度（最短方向）
        diff = (target_heading - current) % 360
        if diff > 180:
            turn_dir = "left"
            turn_deg = 360 - diff
        else:
            turn_dir = "right"
            turn_deg = diff

        if turn_deg < HEADING_TOLERANCE:
            self._log(f"[TURNING] Aligned with {target['id']} (error {turn_deg:.1f} deg)")
            self._set(nav_state="DRIVING")
            return

        self._log(
            f"[TURNING] Aligning with {target['id']} "
            f"current={current:.1f} deg target={target_heading:.1f} deg "
            f"turn={turn_dir} {turn_deg:.1f} deg"
        )

        # 开始转弯
        start_heading = motor.get_heading()
        if turn_dir == "left":
            motor.start_turn_left()
        else:
            motor.start_turn_right()

        # 闭环等待航向对准
        deadline = time.time() + TURN_TIMEOUT
        heading_changed = False
        while time.time() < deadline:
            if self._cancel_event.is_set():
                motor.stop()
                return
            current = motor.get_heading() % 360
            diff = (target_heading - current) % 360
            error = min(diff, 360 - diff)

            if abs(current - start_heading) > 1:
                heading_changed = True

            if error < HEADING_TOLERANCE:
                motor.stop()
                time.sleep(0.1)
                break

            # heading 不更新 → 回退计时转弯（1.5s 估转 90°）
            if not heading_changed and time.time() > deadline - TURN_TIMEOUT + 1.5:
                self._log("[TURNING] Heading unavailable; using timed turn")
                elapsed = time.time() - (deadline - TURN_TIMEOUT)
                remaining = max(0.0, turn_deg / 60.0 - elapsed)
                self._cancel_event.wait(remaining)
                motor.stop()
                break

            time.sleep(0.05)

        motor.stop()
        if self._cancel_event.is_set():
            return
        final_heading = motor.get_heading() % 360
        final_error = abs(target_heading - final_heading)
        if final_error > 180:
            final_error = 360 - final_error
        self._log(f"[TURNING] Alignment complete (error {final_error:.1f} deg)")
        self._set(nav_state="DRIVING")

    def _do_driving(self):
        if self._check_manual_override():
            return
        if self._current_wp_idx >= len(self._waypoints):
            self._set(nav_state="ARRIVED")
            return

        target = self._waypoints[self._current_wp_idx]
        motor = get_motor()

        # 从地图边获取目标距离
        prev_idx = max(0, self._current_wp_idx - 1)
        prev = self._waypoints[prev_idx]
        target_id = target["id"]

        # 查邻接表获取实际边长
        adj = {}
        for e in self.map_data["edges"]:
            u, v, d = e["from"], e["to"], e["dist_cm"]
            adj.setdefault(u, {})[v] = d
            adj.setdefault(v, {})[u] = d

        dist_cm = adj.get(prev["id"], {}).get(target_id, 50)
        self._log(f"[DRIVING] Moving to {target_id} ({dist_cm} cm)")

        # 前进 + 超声波监控
        if motor.is_connected():
            completed = motor.forward(
                dist_cm, cancel_event=self._cancel_event
            )
            if self._cancel_event.is_set():
                return
            if not completed:
                self._log("[DRIVING] Motion did not complete")
                self._set(nav_state="FAILED", mode="failed")
                return
            if motor.is_blocked():
                self._log("[DRIVING] Obstacle detected")
                self._set(nav_state="OBSTACLE_WAIT")
                return
        else:
            # 模拟模式
            if not motor.forward(dist_cm, cancel_event=self._cancel_event):
                return

        self._current_wp_idx += 1
        self._set(current_node=target_id, nav_state="ARRIVED")

    def _do_obstacle_wait(self):
        """停车等 2 秒 → 重测超声波。"""
        motor = get_motor()
        motor.stop()
        self._log("[OBSTACLE_WAIT] Stopped; checking again in 2 seconds")
        if self._cancel_event.wait(2.0):
            return
        if motor.is_blocked():
            self._log("[OBSTACLE_WAIT] Obstacle remains; replanning")
            self._set(nav_state="REPLANNING")
        else:
            self._log("[OBSTACLE_WAIT] Path clear; continuing")
            self._set(nav_state="TURNING")

    def _do_replanning(self):
        """删掉当前被堵的边 → 重跑 A*。"""
        self._log("[REPLANNING] Removing blocked edge and running A*")
        prev_idx = max(0, self._current_wp_idx - 1)
        if prev_idx < len(self._waypoints) and self._current_wp_idx < len(self._waypoints):
            u = self._waypoints[prev_idx]["id"]
            v = self._waypoints[self._current_wp_idx]["id"]
            self.map_data["edges"] = [
                e for e in self.map_data["edges"]
                if not ((e["from"] == u and e["to"] == v) or
                        (e["from"] == v and e["to"] == u))
            ]
            self._log(f"[REPLANNING] Removed edge {u} <-> {v}")

        zone = self._command.get("zone")
        current_node = self._waypoints[prev_idx]["id"]
        if zone:
            waypoints = get_waypoints(self.map_data, current_node, zone)
        else:
            waypoints = get_waypoints(self.map_data, current_node,
                                      self._waypoints[-1]["id"])

        if waypoints is None:
            self._log("[REPLANNING] No alternate route")
            self._set(nav_state="FAILED", mode="failed")
            return

        self._waypoints = waypoints
        self._current_wp_idx = 1
        self._set(route=[w["id"] for w in waypoints], nav_state="TURNING")

    def _do_arrived(self):
        """到达节点。猫区→搜索，否则→继续前往下一个路径点。"""
        idx = self._current_wp_idx
        prev_idx = idx - 1

        if 0 <= prev_idx < len(self._waypoints):
            node_id = self._waypoints[prev_idx]["id"]
            node_type = self.map_data["nodes"].get(node_id, {}).get("type", "")
            if node_type == "cat_zone" and self._command.get("breed"):
                self._log(f"[ARRIVED] Reached {node_id}; starting visual search")
                self._set(nav_state="SEARCHING")
                return
            elif node_type == "cat_zone":
                # 无品种，只导航不找猫 → 到达即完成
                self._log(f"[ARRIVED] Reached {node_id}")
                self._set(nav_state="IDLE", mode="idle", current_node=node_id)
                return

        # 非猫区或无效索引
        if idx >= len(self._waypoints):
            if self._command.get("breed"):
                self._log("[ARRIVED] All zones searched")
                self._set(nav_state="FAILED", mode="failed")
            else:
                # 导航模式：到达终点
                self._log(f"[ARRIVED] Reached {self._waypoints[-1]['id']}")
                self._set(nav_state="IDLE", mode="idle", current_node=self._waypoints[-1]["id"])
        else:
            self._set(nav_state="TURNING")

    def _do_searching(self):
        """原地 360° 旋转 + 猫视觉检测。"""
        target_breed = self._command["breed"]
        prev_idx = self._current_wp_idx - 1
        if prev_idx < 0 or prev_idx >= len(self._waypoints):
            self._set(nav_state="FAILED", mode="failed")
            return

        current_node = self._waypoints[prev_idx]["id"]
        target_name = BREED_DISPLAY_NAMES.get(target_breed, target_breed)
        self._log(f"[SEARCHING] Looking for {target_name} at {current_node}; rotating 360 deg")

        reset_detection()
        set_search_active(True)
        motor = get_motor()
        found_breed = None

        # 旋转搜索（真车用电机，模拟用延时）
        if motor.is_connected():
            motor.start_turn_right()
            last_heading = motor.get_heading() % 360

        deadline = time.time() + 15.0
        rotated = 0.0

        try:
            while time.time() < deadline:
                if self._cancel_event.is_set():
                    return
                # 检查航向（真车）或模拟旋转进度
                if motor.is_connected():
                    heading = motor.get_heading() % 360
                    step = (heading - last_heading + 180) % 360 - 180
                    rotated += abs(step)
                    last_heading = heading
                    if rotated > 340:
                        break

                # 摄像头 YOLO 检测——真车和模拟都用
                det = get_latest_detection()
                if det and self._detection_matches(det, target_breed):
                    found_breed = det.get("breed") or det.get("species")
                    found_name = BREED_DISPLAY_NAMES.get(found_breed, found_breed)
                    self._log(f"[SEARCHING] Vision match: {found_name} "
                              f"(cls={det.get('classification_confidence', 0):.2f})")
                    break

                if not motor.is_connected():
                    # 模拟模式：等待摄像头检测，超时 3 秒
                    if time.time() > deadline - 12:
                        break
                time.sleep(0.1)
        finally:
            set_search_active(False)
            if motor.is_connected():
                motor.stop()

        if motor.is_connected():
            self._log(f"[SEARCHING] Rotation complete ({rotated:.0f} deg)")
        else:
            self._log("[SEARCHING] Camera search complete (3-second scan)")

        if found_breed:
            self._set(nav_state="SUCCESS")
        else:
            self._log(f"[SEARCHING] {target_name} not found at {current_node}")
            with self._lock:
                if current_node not in self._state["explored_zones"]:
                    self._state["explored_zones"].append(current_node)
            if self._current_wp_idx >= len(self._waypoints):
                self._set(nav_state="FAILED", mode="failed")
            else:
                self._set(nav_state="TURNING")

    def _detection_matches(self, det: dict, target_breed: str) -> bool:
        """判断视觉检测是否满足目标。"""
        if not det.get("stable", False):
            return False
        if target_breed in GENERIC_ANIMAL_LABELS:
            return det.get("species") in {"cat", "dog", "bird"}
        if target_breed in GENERIC_CAT_LABELS:
            return det.get("species") == "cat"
        if target_breed in {"dog", "bird"}:
            return det.get("breed") == target_breed
        return (
            det.get("breed") == target_breed
            and det.get("classification_confidence", 0) >= MIN_BREED_CONFIDENCE
        )

    def _do_success(self):
        breed = self._command.get("breed")
        prev_idx = self._current_wp_idx - 1
        found_zone = (
            self._waypoints[prev_idx]["id"]
            if 0 <= prev_idx < len(self._waypoints)
            else "unknown"
        )
        actions = self._command.get("actions", [])

        # 无品种 = 纯导航，不找猫，不执行护理动作，不自动返回
        if not breed:
            self._log(f"[ARRIVED] Reached {found_zone}")
            self._set(nav_state="IDLE", mode="idle", current_node=found_zone,
                      cat_found=False)
            return

        breed_name = BREED_DISPLAY_NAMES.get(breed, breed)
        self._log(f"[SUCCESS] Found {breed_name} at {found_zone}; actions: {actions}")

        self._set(
            nav_state="SUCCESS",
            mode="running",
            cat_found=True,
            cat_breed=breed,
            cat_zone=found_zone,
        )

        from .action import run_care_actions
        # 逐动作执行，前端可看到实时进度
        action_labels = {
            "play": "Play", "feed": "Feed",
            "photo": "Photo", "talk": "Talk",
            "return": "Return",
        }
        for action in actions:
            label = action_labels.get(action, action)
            with self._lock:
                self._state["current_action"] = label
            self._log(f"[CARE] Running: {label}")
            if self._cancel_event.wait(0.6):
                self._set(nav_state="IDLE", mode="idle", current_action="")
                return
            with self._lock:
                self._state["actions_done"].append(action)
                self._state["current_action"] = f"✓ {label}"

        try:
            run_care_actions(breed, found_zone, actions)
        except Exception as e:
            self._log(f"[CARE] Action failed: {e}")

        with self._lock:
            self._state["current_action"] = ""

        # ---- 自动返回起点 ----
        self._return_to_start()

    def _do_failed(self):
        self._log("[FAILED] Mission failed; returning to start")
        self._return_to_start()

    def _return_to_start(self):
        """规划并执行返回起点的路径。"""
        prev_idx = self._current_wp_idx - 1
        current_node = (
            self._waypoints[prev_idx]["id"]
            if 0 <= prev_idx < len(self._waypoints) and self._waypoints
            else "start"
        )

        if current_node == "start":
            self._log("[RETURN] Already at start")
            self._set(nav_state="IDLE", mode="idle", current_node="start")
            return

        waypoints = get_waypoints(self.map_data, current_node, "start")
        if waypoints is None:
            self._log("[RETURN] No return route; current position is unchanged")
            self._set(
                nav_state="IDLE",
                mode="failed",
                current_node=current_node,
                current_action="",
            )
            return

        self._log(f"[RETURN] Route to start: {' -> '.join(w['id'] for w in waypoints)}")
        self._waypoints = waypoints
        self._current_wp_idx = 1  # 跳过当前节点
        self._set(
            route=[w["id"] for w in waypoints],
            nav_state="TURNING",
        )
        # 重新进入状态机循环（TURNING → DRIVING → ... → ARRIVED → IDLE）
        self._run_return_loop()

    def _run_return_loop(self):
        """返回起点的简版导航循环。"""
        while True:
            if self._cancel_event.is_set():
                get_motor().stop()
                self._set(nav_state="IDLE", mode="idle", current_action="")
                return
            st = self._state["nav_state"]
            if st == "TURNING":
                self._do_turning()
            elif st == "DRIVING":
                self._do_driving()
            elif st == "ARRIVED":
                self._do_arrived_return()
            elif st in ("IDLE", "SUCCESS", "FAILED"):
                return
            else:
                self._do_turning()

    def _do_arrived_return(self):
        """返回途中到达节点——不搜索猫，继续到下一站或结束。"""
        idx = self._current_wp_idx
        if idx >= len(self._waypoints):
            self._log("[RETURN] Reached start")
            self._set(nav_state="IDLE", mode="idle", current_node="start")
            return
        self._set(nav_state="TURNING")


# ---- 模块级单例（兼容 CLI main.py） ----

_controller: Optional[NavigationController] = None
_lock_singleton = threading.Lock()


def _get_controller() -> NavigationController:
    global _controller
    with _lock_singleton:
        if _controller is None:
            _controller = NavigationController()
        return _controller


def execute_mission(command: dict):
    """模块 3 唯一对外入口（CLI / main.py 使用）。"""
    ctrl = _get_controller()
    ctrl.execute_mission(command)


def get_controller() -> NavigationController:
    """返回单例控制器（server.py 使用，可轮询 get_state()）。"""
    return _get_controller()
