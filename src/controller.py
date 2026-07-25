"""
模块 3: 导航控制器 — {breed, zone, actions} → 机器人执行
zone 可选：未指定时自动探索所有猫区。

通过 get_state() 暴露实时状态，供 Web 前端轮询。
"""

import math
import os
import threading
import time
from typing import Optional

from .planner import DEFAULT_MAP_PATH, load_map, get_waypoints, get_cat_zones, plan_exploration_route
from .motor import get_motor, MotorController
from .camera import (
    get_latest_beacon,
    get_latest_detection,
    reset_detection,
    set_beacon_active,
    set_search_active,
)

GENERIC_CAT_LABELS = {"cat", "猫"}
GENERIC_ANIMAL_LABELS = {"animal", "动物", "宠物"}
CARE_ACTIONS = {"play", "feed", "photo", "talk"}
MIN_BREED_CONFIDENCE = 0.45
SIMULATION_SCAN_DELAY = float(os.environ.get("SIMULATION_SCAN_DELAY", "0.30"))
SIMULATION_NODE_DWELL_SECONDS = float(
    os.environ.get("SIMULATION_NODE_DWELL_SECONDS", "0.25")
)
SEARCH_TURN_STEP_DEGREES = max(
    1, min(90, int(os.environ.get("SEARCH_TURN_STEP_DEGREES", "30")))
)
SEARCH_STEP_PAUSE_SECONDS = max(
    0.0, float(os.environ.get("SEARCH_STEP_PAUSE_SECONDS", "0.25"))
)
CENTERING_TOLERANCE = max(
    0.01, float(os.environ.get("CENTERING_TOLERANCE", "0.15"))
)
CENTERING_MAX_ITERATIONS = max(
    1, int(os.environ.get("CENTERING_MAX_ITERATIONS", "8"))
)
CENTERING_MAX_STEP_DEGREES = max(
    1, int(os.environ.get("CENTERING_MAX_STEP_DEGREES", "20"))
)
CENTERING_PAUSE_SECONDS = max(
    0.0, float(os.environ.get("CENTERING_PAUSE_SECONDS", "0.30"))
)
BEACON_ACQUIRE_SECONDS = max(
    0.0, float(os.environ.get("BEACON_ACQUIRE_SECONDS", "1.00"))
)
BEACON_CENTER_TOLERANCE = max(
    0.0, min(
        0.25,
        float(os.environ.get("BEACON_CENTER_TOLERANCE", "0.05")),
    ),
)
BEACON_MAX_CORRECTIONS = max(
    1, int(os.environ.get("BEACON_MAX_CORRECTIONS", "8"))
)
BEACON_REFRESH_SECONDS = max(
    0.1, float(os.environ.get("BEACON_REFRESH_SECONDS", "0.75"))
)
BEACON_CENTER_SETTLE_SECONDS = max(
    0.0,
    float(os.environ.get("BEACON_CENTER_SETTLE_SECONDS", "0.50")),
)
CARE_ACTION_DISPLAY_DELAY = float(os.environ.get("CARE_ACTION_DISPLAY_DELAY", "0.25"))
HOME_HEADING_DEGREES = float(os.environ.get("HOME_HEADING_DEGREES", "90"))
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
        # continue 可以从 HTTP/ASR 线程触发。这个锁保证恢复工作线程
        # 可以接管已退出的任务，同时绝不会有两个状态机重复发运动指令。
        self._state_machine_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._interaction_event = threading.Event()
        self._interaction_decision: Optional[str] = None

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
            "interaction_pending": False,
            "pending_actions": [],
            "pause_node": None,
            "explored_zones": [],
            "log": [],
        }

        # ---- 内部状态 ----
        self._waypoints: list[dict] = []
        self._current_wp_idx: int = 0
        self._command: Optional[dict] = None
        self._exploring: bool = False
        self._pause_node: Optional[str] = None
        self._resume_requested: bool = False
        self._returning_home: bool = False

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
                "interaction_pending": self._state["interaction_pending"],
                "pending_actions": list(self._state["pending_actions"]),
                "pause_node": self._state["pause_node"],
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
            self._interaction_event.clear()
            self._interaction_decision = None
            self._command = command
            self._pause_node = command.get("pause_node")
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
                interaction_pending=False,
                pending_actions=[],
                pause_node=self._pause_node,
                explored_zones=[],
            )
            self._waypoints = []
            self._current_wp_idx = 0
            self._exploring = False
            self._resume_requested = False
            self._returning_home = False

        is_map_mission = bool(command.get("zone") or command.get("breed"))
        if current_node == "start" and is_map_mission:
            motor = get_motor()
            calibrate_heading = getattr(motor, "calibrate_heading", None)
            if (
                motor.is_connected()
                and callable(calibrate_heading)
                and calibrate_heading(HOME_HEADING_DEGREES)
            ):
                self._log(
                    f"[HOME] Start heading calibrated to "
                    f"{HOME_HEADING_DEGREES % 360:.1f} deg"
                )

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
        self._interaction_event.set()
        set_beacon_active(False)
        get_motor().stop()
        with self._lock:
            self._pause_node = None
            self._resume_requested = False
            mission_running = self._state["nav_state"] != "IDLE"
            was_busy = mission_running or self._state["mode"] in ("running", "listening")
            self._state.update(
                mode="stopping" if mission_running else "idle",
                nav_state="STOPPING" if mission_running else "IDLE",
                current_action="",
                pause_node=None,
            )
        if was_busy:
            self._log(f"[STOP] {reason}")

    def request_pause_at(self, node_id: str) -> tuple[bool, str]:
        """为当前任务登记一个尚未经过的 junction 停车点。"""
        node = self.map_data.get("nodes", {}).get(node_id)
        if not node or node.get("type") != "junction":
            return False, f"{node_id} is not a junction in the active map"

        with self._lock:
            nav_state = self._state["nav_state"]
            if nav_state in {"IDLE", "STOPPING", "FAILED"}:
                return False, "No active mission to pause"
            if nav_state == "PAUSED":
                return False, "Mission is already paused"

            if self._waypoints:
                remaining_ids = {
                    waypoint["id"]
                    for waypoint in self._waypoints[self._current_wp_idx:]
                }
                if node_id not in remaining_ids:
                    return False, f"{node_id} is not ahead on the current route"

            self._pause_node = node_id
            self._state["pause_node"] = node_id

        self._log(f"[PAUSE] Will stop at {node_id}")
        return True, ""

    def continue_mission(self) -> tuple[bool, str]:
        """继续暂停任务；到达状态尚未同步时先排队，避免丢失语音。"""
        with self._lock:
            nav_state = self._state["nav_state"]
            current_node = self._state["current_node"]
            acknowledge_motion = False
            pause_node_for_ack = None
            if nav_state == "PAUSED":
                self._resume_requested = False
                self._pause_node = None
                self._state["pause_node"] = None
                self._state["mode"] = "running"
                self._state["nav_state"] = (
                    "TURNING"
                    if self._current_wp_idx < len(self._waypoints)
                    else "ARRIVED"
                )
                queued = False
            elif (
                self._pause_node
                and current_node == self._pause_node
                and self._current_wp_idx < len(self._waypoints)
                and nav_state in {"IDLE", "FAILED"}
            ):
                # 暂停点已经登记，但原任务线程异常退出：从剩余路径恢复。
                self._resume_requested = False
                self._pause_node = None
                self._state.update(
                    pause_node=None,
                    mode="running",
                    nav_state="TURNING",
                )
                queued = False
            elif (
                self._pause_node
                and nav_state in {
                    "PLANNING", "TURNING", "DRIVING",
                    "ARRIVED", "OBSTACLE_WAIT", "REPLANNING",
                }
            ):
                # 小车可能已经物理停住，但 Pi 的 idle 状态和控制器
                # ARRIVED/PAUSED 状态尚未同步。保留这次 continue，
                # 到达预定 junction 后立即消费。
                self._resume_requested = True
                queued = True
                if (
                    nav_state == "DRIVING"
                    and self._current_wp_idx < len(self._waypoints)
                    and self._waypoints[self._current_wp_idx]["id"]
                    == self._pause_node
                ):
                    acknowledge_motion = True
                    pause_node_for_ack = self._pause_node
            else:
                return False, "Mission has no scheduled junction pause"
            next_node = (
                self._waypoints[self._current_wp_idx]["id"]
                if self._current_wp_idx < len(self._waypoints)
                else None
            )
        if acknowledge_motion:
            confirm_motion = getattr(
                get_motor(), "confirm_motion_complete", None
            )
            if callable(confirm_motion) and confirm_motion():
                self._log(
                    f"[RESUME] Operator confirmed arrival at "
                    f"{pause_node_for_ack}; releasing motion wait"
                )
        self._log(
            "[RESUME] Continue queued until the scheduled junction is registered"
            if queued
            else f"[RESUME] Continuing saved route; next={next_node}"
        )
        # 正常情况下原任务线程仍在 PAUSED 循环中，会立即接着执行。
        # 若它已经退出，这个工作线程会接管后半段；单实例锁避免重复执行。
        self._ensure_state_machine_worker()
        return True, ""

    def _ensure_state_machine_worker(self):
        """启动恢复工作线程；状态机锁负责过滤重复执行。"""
        threading.Thread(
            target=self._run_state_machine,
            name="catrescue-resume",
            daemon=True,
        ).start()

    def resolve_interaction(self, execute: bool) -> bool:
        """处理网页端的交互确认；返回 False 表示当前没有待确认动作。"""
        with self._lock:
            if not self._state["interaction_pending"]:
                return False
            self._interaction_decision = "execute" if execute else "skip"
            self._state["interaction_pending"] = False
        self._interaction_event.set()
        self._log(
            "[CARE] Interaction approved"
            if execute
            else "[CARE] Interaction skipped by user"
        )
        return True

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
        with self._state_machine_lock:
            self._run_state_machine_loop()

    def _run_state_machine_loop(self):
        """在单实例锁保护下执行任务状态机。"""
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
            elif st == "PAUSED":
                self._cancel_event.wait(0.1)
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
            if self._pause_node:
                to_pause = get_waypoints(
                    self.map_data, start_node, self._pause_node
                )
                from_pause = get_waypoints(
                    self.map_data, self._pause_node, zone
                )
                waypoints = (
                    to_pause + from_pause[1:]
                    if to_pause is not None and from_pause is not None
                    else None
                )
            else:
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
            if self._pause_node:
                to_pause = get_waypoints(
                    self.map_data, start_node, self._pause_node
                )
                from_pause = plan_exploration_route(
                    self.map_data, self._pause_node
                )
                waypoints = (
                    to_pause + from_pause[1:]
                    if to_pause is not None and from_pause is not None
                    else None
                )
            else:
                waypoints = plan_exploration_route(
                    self.map_data, start_node
                )
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

        if self._pause_node:
            route_ids = [waypoint["id"] for waypoint in waypoints[1:]]
            pause_node_data = self.map_data["nodes"].get(self._pause_node)
            if (
                not pause_node_data
                or pause_node_data.get("type") != "junction"
                or self._pause_node not in route_ids
            ):
                self._log(
                    f"[PLANNER] Pause point {self._pause_node} "
                    "is not an upcoming junction on this route"
                )
                self._pause_node = None
                self._set(nav_state="FAILED", mode="failed", pause_node=None)
                return

        self._set(
            route=[w["id"] for w in waypoints],
            current_node=start_node,
            nav_state="TURNING",
        )
        first_target = waypoints[1]["id"] if len(waypoints) > 1 else zone
        self._log(
            f"[MISSION] Starting from {start_node}; first target={first_target}"
            + (f"; scheduled stop={self._pause_node}" if self._pause_node else "")
        )

    def _execute_direct_turn(self, direction: str, turn_deg: float) -> bool:
        """Send one self-contained a+angle / d+angle command."""
        return get_motor().turn_degrees(
            direction,
            turn_deg,
            cancel_event=self._cancel_event,
        )

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
        # 红点不能参与地图粗转向，也不能沿用上一次检测。
        set_beacon_active(False)
        if self._check_manual_override():
            return
        if self._current_wp_idx >= len(self._waypoints):
            self._set(nav_state="ARRIVED")
            return

        target = self._waypoints[self._current_wp_idx]
        motor = get_motor()

        # 地图使用屏幕坐标（Y 向下），小车使用数学航向：
        # 0°=向右、90°=向上，左转增加角度、右转减少角度。
        prev_idx = max(0, self._current_wp_idx - 1)
        prev = self._waypoints[prev_idx]
        dx = target["x"] - prev["x"]
        dy = target["y"] - prev["y"]
        target_heading = math.degrees(math.atan2(-dy, dx)) % 360
        reverse_into_home = self._returning_home and target["id"] == "start"
        if reverse_into_home:
            # Back into the final home edge so the robot arrives already
            # facing the canonical outbound direction.
            target_heading = HOME_HEADING_DEGREES % 360

        if not motor.is_connected():
            self._log(f"[SIMULATION] Aligning with {target['id']}")
            if motor.simulate_turn_to(target_heading, self._cancel_event):
                self._set(nav_state="DRIVING")
            return

        # 当前航向
        current = motor.get_heading() % 360

        # 按小车航向定义计算最短转向：正角度为左转，负角度为右转。
        diff = (target_heading - current) % 360
        if diff > 180:
            turn_dir = "right"
            turn_deg = 360 - diff
        else:
            turn_dir = "left"
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

        if not motor.turn_degrees(
            turn_dir,
            turn_deg,
            cancel_event=self._cancel_event,
        ):
            if not self._cancel_event.is_set():
                self._fail_in_place(
                    f"Hardware turn failed for {target['id']}"
                )
            return

        if not reverse_into_home:
            # 地图角度已经执行完毕；现在才开启红点识别做二次校准。
            set_beacon_active(True)
            try:
                beacon_ok = self._align_to_red_beacon(motor)
            finally:
                set_beacon_active(False)
            if beacon_ok is None:
                if not self._cancel_event.is_set():
                    self._fail_in_place(
                        f"Hardware beacon-correction turn failed for {target['id']}"
                    )
                return
            if beacon_ok and BEACON_CENTER_SETTLE_SECONDS > 0:
                self._log(
                    f"[TURNING] Beacon centered; holding "
                    f"{BEACON_CENTER_SETTLE_SECONDS:.2f}s before driving"
                )
                if self._cancel_event.wait(BEACON_CENTER_SETTLE_SECONDS):
                    return
            elif not beacon_ok:
                self._log(
                    f"[TURNING] Red beacon alignment was not confirmed for "
                    f"{target['id']}; continuing to the waypoint"
                )

        final_heading = motor.get_heading() % 360
        final_error = abs(target_heading - final_heading)
        if final_error > 180:
            final_error = 360 - final_error
        self._log(f"[TURNING] Alignment complete (error {final_error:.1f} deg)")
        self._set(nav_state="DRIVING")

    def _align_to_red_beacon(self, motor) -> bool | None:
        """用红点做二次校准。

        True 表示居中成功，False 表示纯视觉校准未确认（允许继续导航），
        None 表示任务取消或校准转向的硬件指令失败（禁止继续行驶）。
        """
        tolerance = BEACON_CENTER_TOLERANCE
        beacon = self._wait_for_new_beacon(
            timeout=BEACON_ACQUIRE_SECONDS
        )

        if not beacon:
            self._log(
                "[TURNING] No red beacon after coarse turn; "
                "alignment cannot be confirmed"
            )
            return False
        if abs(beacon.get("offset", 0.0)) <= tolerance:
            self._log_beacon_centered(beacon)
            return True

        self._log(
            f"[TURNING] Red beacon lock "
            f"(offset={beacon['offset']:.3f})"
        )
        for _ in range(BEACON_MAX_CORRECTIONS):
            if self._cancel_event.is_set():
                return None

            offset = float(beacon.get("offset", 0.0))
            if abs(offset) <= tolerance:
                self._log_beacon_centered(beacon)
                return True

            direction = "right" if offset > 0 else "left"
            correction_deg = max(
                2, min(12, int(round(abs(offset) * 50)))
            )
            observed_at = beacon.get("observed_at")
            if not motor.turn_degrees(
                direction,
                correction_deg,
                cancel_event=self._cancel_event,
            ):
                return None
            beacon = self._wait_for_new_beacon(
                after=observed_at,
                timeout=BEACON_REFRESH_SECONDS,
            )
            if not beacon:
                self._log(
                    "[TURNING] Red beacon lost before a fresh "
                    "post-correction frame arrived"
                )
                return False

        if beacon and abs(float(beacon.get("offset", 0.0))) <= tolerance:
            self._log_beacon_centered(beacon)
            return True
        self._log(
            "[TURNING] Red beacon correction limit reached without centering"
        )
        return False

    def _wait_for_new_beacon(
        self,
        after: float | None = None,
        timeout: float = 1.0,
    ) -> dict | None:
        """等待红点新帧，避免用转向前的旧坐标反复修正。"""
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if self._cancel_event.wait(0.05):
                return None
            beacon = get_latest_beacon()
            if not beacon:
                continue
            observed_at = beacon.get("observed_at")
            if after is None or observed_at is None or observed_at > after:
                return beacon
        return None

    def _log_beacon_centered(self, beacon: dict):
        """记录水平居中结果；上下位置不参与转向判定。"""
        frame_width = float(beacon.get("frame_width") or 0)
        center_x = beacon.get("center_x")
        if frame_width > 0 and center_x is not None:
            left = float(center_x)
            right = frame_width - left
            self._log(
                f"[TURNING] Red beacon centered "
                f"(left={left:.1f}px, right={right:.1f}px)"
            )
        else:
            self._log(
                f"[TURNING] Red beacon centered "
                f"(horizontal offset={float(beacon.get('offset', 0.0)):.3f})"
            )

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
        reverse_into_home = self._returning_home and target_id == "start"
        verb = "Backing into" if reverse_into_home else "Moving to"
        self._log(f"[DRIVING] {verb} {target_id} ({dist_cm} cm)")

        # 前进 + 超声波监控
        simulation_mode = not motor.is_connected()
        if not simulation_mode:
            motion = motor.backward if reverse_into_home else motor.forward
            completed = motion(dist_cm, cancel_event=self._cancel_event)
            if self._cancel_event.is_set():
                return
            if not completed:
                if motor.is_blocked():
                    self._fail_in_place(
                        f"Forward path to {target_id} is blocked"
                    )
                else:
                    self._fail_in_place(
                        f"Forward motion to {target_id} did not complete"
                    )
                return
            if motor.is_blocked():
                self._log("[DRIVING] Obstacle detected")
                self._set(nav_state="OBSTACLE_WAIT")
                return
        else:
            # 模拟模式
            motion = motor.backward if reverse_into_home else motor.forward
            if not motion(dist_cm, cancel_event=self._cancel_event):
                return

        self._current_wp_idx += 1
        self._set(current_node=target_id, nav_state="ARRIVED")
        if simulation_mode:
            self._cancel_event.wait(SIMULATION_NODE_DWELL_SECONDS)

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
            if node_type == "junction" and node_id == self._pause_node:
                get_motor().stop()
                with self._lock:
                    resume_now = self._resume_requested
                    self._resume_requested = False
                    if resume_now:
                        self._pause_node = None
                        self._state.update(
                            nav_state=(
                                "TURNING"
                                if self._current_wp_idx < len(self._waypoints)
                                else "ARRIVED"
                            ),
                            mode="running",
                            current_node=node_id,
                            pause_node=None,
                        )
                    else:
                        self._state.update(
                            nav_state="PAUSED",
                            mode="paused",
                            current_node=node_id,
                            pause_node=node_id,
                        )
                if resume_now:
                    self._log(
                        f"[RESUME] Reached {node_id}; consuming queued "
                        "continue and resuming the route"
                    )
                else:
                    self._log(
                        f"[PAUSE] Stopped at {node_id}; say continue to resume"
                    )
                return
            if node_type == "cat_zone" and self._command.get("breed"):
                requested_zone = self._command.get("zone")
                if requested_zone and node_id != requested_zone:
                    self._log(
                        f"[ARRIVED] Passing through {node_id}; "
                        f"search target is {requested_zone}"
                    )
                else:
                    self._log(
                        f"[ARRIVED] Reached {node_id}; starting visual search"
                    )
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
        motor = get_motor()
        found_breed = None

        # 模拟模式使用地图中预设的动物，不依赖本地摄像头画面。
        if not motor.is_connected():
            animal = self.map_data.get("simulated_animals", {}).get(current_node)
            label = animal.get("label") if animal else "empty"
            self._log(
                f"[SIMULATION] Scanning {current_node}; configured animal: {label}"
            )
            if self._cancel_event.wait(SIMULATION_SCAN_DELAY):
                return
            if animal:
                simulated_detection = {
                    **animal,
                    "stable": True,
                    "classification_confidence": 1.0,
                    "detection_confidence": 1.0,
                }
                if self._detection_matches(simulated_detection, target_breed):
                    found_breed = animal.get("breed") or animal.get("species")
                    found_name = BREED_DISPLAY_NAMES.get(found_breed, label)
                    self._log(f"[SIMULATION] Match found: {found_name} at {current_node}")

            if found_breed:
                self._set(
                    nav_state="SUCCESS",
                    cat_breed=found_breed,
                    cat_zone=current_node,
                )
            else:
                self._log(f"[SEARCHING] {target_name} not found at {current_node}")
                self._continue_after_miss(current_node)
            return

        self._log(
            f"[SEARCHING] Looking for {target_name} at {current_node}; "
            f"stepped 360 deg scan ({SEARCH_TURN_STEP_DEGREES} deg/step)"
        )
        reset_detection()
        set_search_active(True)
        rotated = 0.0

        try:
            while rotated < 360.0:
                if self._cancel_event.is_set():
                    return

                # 真车模式使用摄像头 YOLO 检测。
                det = get_latest_detection()
                if det and self._detection_matches(det, target_breed):
                    found_breed = det.get("breed") or det.get("species")
                    found_name = BREED_DISPLAY_NAMES.get(found_breed, found_breed)
                    self._log(f"[SEARCHING] Vision match: {found_name} "
                              f"(cls={det.get('classification_confidence', 0):.2f})")
                    self._center_on_target(det, target_breed)
                    break

                step_deg = min(SEARCH_TURN_STEP_DEGREES, 360.0 - rotated)
                if not motor.turn_degrees(
                    "right",
                    step_deg,
                    cancel_event=self._cancel_event,
                ):
                    if not self._cancel_event.is_set():
                        self._log("[SEARCHING] Scan turn failed; stopping search")
                        self._set(nav_state="FAILED", mode="failed")
                    return
                rotated += step_deg

                if self._cancel_event.wait(SEARCH_STEP_PAUSE_SECONDS):
                    return

            # 最后一段转完后，在停止姿态再检查一次当前画面。
            if found_breed is None:
                det = get_latest_detection()
                if det and self._detection_matches(det, target_breed):
                    found_breed = det.get("breed") or det.get("species")
                    found_name = BREED_DISPLAY_NAMES.get(found_breed, found_breed)
                    self._log(
                        f"[SEARCHING] Vision match: {found_name} "
                        f"(cls={det.get('classification_confidence', 0):.2f})"
                    )
                    self._center_on_target(det, target_breed)
        finally:
            set_search_active(False)
            motor.stop()

        self._log(f"[SEARCHING] Rotation complete ({rotated:.0f} deg)")

        if found_breed:
            self._set(
                nav_state="SUCCESS",
                cat_breed=found_breed,
                cat_zone=current_node,
            )
        else:
            self._log(f"[SEARCHING] {target_name} not found at {current_node}")
            self._continue_after_miss(current_node)

    def _center_on_target(self, det: dict, target_breed: str | None = None):
        """用有界小角度转向将检测框移到画面中央。"""
        motor = get_motor()
        if not motor.is_connected() or not det.get("box"):
            return

        current_det = det
        for iteration in range(CENTERING_MAX_ITERATIONS):
            if self._cancel_event.is_set():
                return
            if target_breed and not self._detection_matches(current_det, target_breed):
                self._log("[CENTERING] Target changed or became unstable; stopping")
                return

            frame_width = max(1, int(current_det.get("frame_width", 640)))
            x1, _, x2, _ = current_det["box"]
            offset = (((x1 + x2) / 2.0) - frame_width / 2.0) / frame_width
            if abs(offset) <= CENTERING_TOLERANCE:
                self._log(f"[CENTERING] Target centered (offset={offset:.3f})")
                return

            direction = "right" if offset > 0 else "left"
            angle = min(
                CENTERING_MAX_STEP_DEGREES,
                max(3, int(round(abs(offset) * 60))),
            )
            self._log(
                f"[CENTERING] Step {iteration + 1}: "
                f"offset={offset:.3f}, turn={direction} {angle} deg"
            )
            if not motor.turn_degrees(
                direction,
                angle,
                cancel_event=self._cancel_event,
            ):
                self._log("[CENTERING] Turn failed; stopping")
                return
            if self._cancel_event.wait(CENTERING_PAUSE_SECONDS):
                return

            current_det = get_latest_detection()
            if not current_det or not current_det.get("box"):
                self._log("[CENTERING] Target lost; stopping")
                return

        self._log("[CENTERING] Iteration limit reached; stopping")

    def _continue_after_miss(self, current_node: str):
        """记录未命中区域，并继续当前路线或扩展为全图搜索。"""
        with self._lock:
            if current_node not in self._state["explored_zones"]:
                self._state["explored_zones"].append(current_node)

        if self._current_wp_idx < len(self._waypoints):
            self._set(nav_state="TURNING")
            return

        # 指定区域未找到目标时，自动从当前位置继续搜索整张地图。
        if not self._exploring:
            waypoints = plan_exploration_route(self.map_data, current_node)
            if waypoints and len(waypoints) > 1:
                self._exploring = True
                self._waypoints = waypoints
                self._current_wp_idx = 1
                self._set(
                    route=[waypoint["id"] for waypoint in waypoints],
                    nav_state="TURNING",
                )
                self._log(
                    "[SEARCHING] Target not found in requested zone; "
                    "continuing with full-map search"
                )
                return

        self._set(nav_state="FAILED", mode="failed")

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
        with self._lock:
            breed = self._state["cat_breed"] or self._command.get("breed")
        prev_idx = self._current_wp_idx - 1
        found_zone = (
            self._waypoints[prev_idx]["id"]
            if 0 <= prev_idx < len(self._waypoints)
            else "unknown"
        )
        actions = [
            action for action in self._command.get("actions", [])
            if action in CARE_ACTIONS
        ]

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

        execute_actions = False
        if actions:
            decision = self._wait_for_interaction_decision(actions)
            if decision is None:
                return
            execute_actions = decision
        else:
            self._log("[CARE] No interaction actions requested")

        if execute_actions:
            from .action import run_care_actions
            # 逐动作执行，前端可看到实时进度
            action_labels = {
                "play": "Play", "feed": "Feed",
                "photo": "Photo", "talk": "Talk",
            }
            self._set(nav_state="SUCCESS", mode="running")
            for action in actions:
                label = action_labels.get(action, action)
                with self._lock:
                    self._state["current_action"] = label
                self._log(f"[CARE] Running: {label}")
                if self._cancel_event.wait(CARE_ACTION_DISPLAY_DELAY):
                    self._set(nav_state="IDLE", mode="idle", current_action="")
                    return
                try:
                    result = run_care_actions(breed, found_zone, [action])
                    completed = result is None or bool(result.get(action))
                except Exception as e:
                    completed = False
                    self._log(f"[CARE] Action failed: {e}")

                with self._lock:
                    if completed:
                        self._state["actions_done"].append(action)
                    self._state["current_action"] = (
                        f"✓ {label}" if completed else f"Failed: {label}"
                    )
                if completed:
                    self._log(f"[CARE] Completed: {label}")
                else:
                    self._log(f"[CARE] Hardware did not complete: {label}")

        with self._lock:
            self._state["current_action"] = ""

        # 护理任务结束后停在发现动物的位置，不再自动返航。
        # 显式的“return”命令仍可单独规划回起点。
        get_motor().stop()
        self._returning_home = False
        self._log(
            f"[MISSION] Care task finished at {found_zone}; "
            "automatic return is disabled"
        )
        self._set(
            nav_state="IDLE",
            mode="success",
            current_node=found_zone,
            current_action="",
        )

    def _wait_for_interaction_decision(self, actions: list[str]) -> Optional[bool]:
        """暂停任务，等待网页确认是否执行护理动作。"""
        self._interaction_event.clear()
        with self._lock:
            self._interaction_decision = None
            self._state.update(
                nav_state="AWAITING_CONFIRMATION",
                mode="running",
                interaction_pending=True,
                pending_actions=list(actions),
                current_action="Waiting for confirmation",
            )
        self._log(f"[CARE] Waiting for confirmation: {actions}")

        while not self._cancel_event.is_set():
            if self._interaction_event.wait(0.2):
                break

        if self._cancel_event.is_set():
            self._set(
                nav_state="IDLE",
                mode="idle",
                interaction_pending=False,
                pending_actions=[],
                current_action="",
            )
            return None

        with self._lock:
            decision = self._interaction_decision
            self._state["interaction_pending"] = False
            self._state["pending_actions"] = []
            self._state["current_action"] = ""
        return decision == "execute"

    def _do_failed(self):
        self._log("[FAILED] Mission failed; returning to start")
        self._return_to_start()

    def _fail_in_place(self, reason: str):
        """导航硬件/视觉失败时原地停车，禁止自动返航产生二次运动。"""
        set_beacon_active(False)
        get_motor().stop()
        current_node = self.get_state()["current_node"]
        self._log(
            f"[FAILED] {reason}; stopped in place at {current_node}"
        )
        self._set(
            nav_state="IDLE",
            mode="failed",
            current_action="",
        )

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
        self._returning_home = True
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
            self._set(current_node="start")
            if not self._restore_home_heading():
                self._returning_home = False
                if not self._cancel_event.is_set():
                    self._log("[HOME] Failed to restore the outbound heading")
                    self._set(nav_state="FAILED", mode="failed")
                return
            self._returning_home = False
            self._set(nav_state="IDLE", mode="idle", current_node="start")
            return
        self._set(nav_state="TURNING")

    def _restore_home_heading(self) -> bool:
        """At start, face the canonical outbound direction for the next mission."""
        motor = get_motor()
        current = motor.get_heading() % 360
        target = HOME_HEADING_DEGREES % 360
        diff = (target - current) % 360
        if diff > 180:
            direction = "right"
            angle = 360 - diff
        else:
            direction = "left"
            angle = diff

        if angle < HEADING_TOLERANCE:
            self._log(
                f"[HOME] Outbound heading ready ({current:.1f} deg)"
            )
            return True

        self._log(
            f"[HOME] Restoring outbound heading: "
            f"{direction} {angle:.1f} deg"
        )
        return motor.turn_degrees(
            direction,
            angle,
            cancel_event=self._cancel_event,
        )


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
