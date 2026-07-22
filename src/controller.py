"""
模块 3: 导航控制器 — {breed, zone, actions} → 机器人执行
zone 可选：未指定时自动探索所有猫区。

通过 get_state() 暴露实时状态，供 Web 前端轮询。
"""

import threading
import time
from typing import Optional

from .planner import DEFAULT_MAP_PATH, load_map, get_waypoints, get_cat_zones, plan_exploration_route


class NavigationController:
    """导航状态机。所有状态变更写入 self._state，可从外部轮询。"""

    def __init__(self, map_path: str = str(DEFAULT_MAP_PATH)):
        self.map_data = load_map(map_path)
        self._lock = threading.Lock()

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
                "explored_zones": list(self._state["explored_zones"]),
                "log": list(self._state["log"]),
            }

    def is_busy(self) -> bool:
        with self._lock:
            return self._state["mode"] in ("running", "listening")

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
            self._command = command
            self._state.update(
                mode="running",
                nav_state="PLANNING",
                current_node="start",
                transcript=transcript,
                command=command,
                route=[],
                cat_found=False,
                cat_breed=None,
                cat_zone=None,
                actions_done=[],
                explored_zones=[],
            )
            self._state["log"] = []
            self._waypoints = []
            self._current_wp_idx = 0
            self._exploring = False

        self._run_state_machine()

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
                self._log(f"[ERROR] 未知状态: {st}")
                return

    # ------------------------------------------------------------------
    # 各状态实现
    # ------------------------------------------------------------------

    def _do_planning(self):
        zone = self._command.get("zone")
        if zone:
            waypoints = get_waypoints(self.map_data, "start", zone)
            if waypoints is None:
                self._log(f"[PLANNER] 无可用路径: start → {zone}")
                self._set(nav_state="FAILED", mode="failed")
                return
            self._exploring = False
            self._log(f"[PLANNER] 路径: {' → '.join(w['id'] for w in waypoints)}")
        else:
            waypoints = plan_exploration_route(self.map_data, "start")
            if waypoints is None:
                self._log("[PLANNER] 无可用猫区")
                self._set(nav_state="FAILED", mode="failed")
                return
            self._exploring = True
            cat_ids = [
                w["id"] for w in waypoints
                if self.map_data["nodes"].get(w["id"], {}).get("type") == "cat_zone"
            ]
            self._log(f"[PLANNER] 探索路线: {' → '.join(w['id'] for w in waypoints)}")
            self._log(f"[PLANNER] 猫区: {cat_ids}")

        self._waypoints = waypoints
        self._current_wp_idx = 1  # 跳过 start
        self._set(
            route=[w["id"] for w in waypoints],
            current_node="start",
            nav_state="TURNING",
        )

    def _do_turning(self):
        if self._current_wp_idx >= len(self._waypoints):
            self._set(nav_state="ARRIVED")
            return

        target = self._waypoints[self._current_wp_idx]
        self._log(f"[TURNING] 转向 {target['id']}...")
        # TODO: 发送 "a"/"d" 转弯指令，读取 hall heading 做闭环
        time.sleep(0.3)   # 模拟转弯耗时
        self._set(current_node=target["id"], nav_state="DRIVING")

    def _do_driving(self):
        if self._current_wp_idx >= len(self._waypoints):
            self._set(nav_state="ARRIVED")
            return

        target = self._waypoints[self._current_wp_idx]
        self._log(f"[DRIVING] 驶向 {target['id']}...")
        # TODO: 发送 "w" 前进指令，轮询超声波，累计 hall 距离
        time.sleep(0.5)   # 模拟行驶耗时
        self._current_wp_idx += 1
        self._set(current_node=target["id"], nav_state="ARRIVED")

    def _do_obstacle_wait(self):
        self._log("[OBSTACLE_WAIT] 停车等待 2 秒...")
        # TODO: 发 "x" 停车，sleep(2)，再测超声波
        time.sleep(2.0)
        # TODO: 若仍被阻挡 → REPLANNING，否则 → DRIVING
        self._set(nav_state="REPLANNING")

    def _do_replanning(self):
        self._log("[REPLANNING] 删除当前被堵边，重跑 A* ...")
        # TODO: 从 graph 中删除当前边，重跑 A*
        #       若有新路径 → TURNING，否则 → FAILED
        time.sleep(0.2)
        self._set(nav_state="TURNING")

    def _do_arrived(self):
        """到达节点。猫区→搜索，否则→继续前往下一个路径点。"""
        idx = self._current_wp_idx
        prev_idx = idx - 1

        if 0 <= prev_idx < len(self._waypoints):
            node_id = self._waypoints[prev_idx]["id"]
            node_type = self.map_data["nodes"].get(node_id, {}).get("type", "")
            if node_type == "cat_zone":
                self._log(f"[ARRIVED] 到达猫区 {node_id}，开始搜索...")
                self._set(nav_state="SEARCHING")
                return

        # 非猫区或无效索引
        if idx >= len(self._waypoints):
            self._log("[ARRIVED] 所有区域已搜索完毕")
            self._set(nav_state="FAILED", mode="failed")
        else:
            self._set(nav_state="TURNING")

    def _do_searching(self):
        """原地旋转 + 猫视觉检测。"""
        target_breed = self._command["breed"]
        # 当前所在的猫区是上一个到达的节点
        prev_idx = self._current_wp_idx - 1
        if prev_idx < 0 or prev_idx >= len(self._waypoints):
            self._set(nav_state="FAILED", mode="failed")
            return

        current_node = self._waypoints[prev_idx]["id"]
        self._log(f"[SEARCHING] 在 {current_node} 寻找 {target_breed}...")
        # TODO: 发转向指令做 360° 旋转，持续运行 YOLO + EfficientNet
        time.sleep(0.8)

        # ---- 模拟：30% 概率找到猫 ----
        import random
        if random.random() < 0.3:
            self._set(nav_state="SUCCESS")
        else:
            self._log(f"[SEARCHING] {current_node} 未找到 {target_breed}")
            # 标记当前区已探索
            with self._lock:
                if current_node not in self._state["explored_zones"]:
                    self._state["explored_zones"].append(current_node)
            # 继续前往下一个路径点（不再多余 +1；_current_wp_idx 已指向下一节点）
            if self._current_wp_idx >= len(self._waypoints):
                self._set(nav_state="FAILED", mode="failed")
            else:
                self._set(nav_state="TURNING")

    def _do_success(self):
        breed = self._command["breed"]
        prev_idx = self._current_wp_idx - 1
        found_zone = (
            self._waypoints[prev_idx]["id"]
            if 0 <= prev_idx < len(self._waypoints)
            else "unknown"
        )
        actions = self._command["actions"]
        self._log(f"[SUCCESS] 在 {found_zone} 找到 {breed}，执行动作: {actions}")

        self._set(
            nav_state="SUCCESS",
            mode="success",
            cat_found=True,
            cat_breed=breed,
            cat_zone=found_zone,
        )

        from .action import run_care_actions
        # 把 care 动作的进度也写入状态
        for action in actions:
            time.sleep(0.4)  # 模拟动作耗时
            with self._lock:
                self._state["actions_done"].append(action)
            self._log(f"[CARE] 执行: {action}")

        run_care_actions(breed, found_zone, actions)

        self._log("[RETURN] 返回起点")
        self._set(nav_state="IDLE", mode="idle", current_node="start")

    def _do_failed(self):
        self._log("[FAILED] 任务失败，返回起点")
        self._set(nav_state="IDLE", mode="idle", current_node="start")


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
