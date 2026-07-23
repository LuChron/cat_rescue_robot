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
from .camera import get_latest_detection, reset_detection

# 转弯参数
HEADING_TOLERANCE = 15.0   # 航向误差容限（度）
TURN_TIMEOUT = 8.0         # 转弯超时（秒）


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
                current_action="",
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
        breed = self._command.get("breed")
        zone = self._command.get("zone")
        distance_cm = self._command.get("distance_cm")
        turn_deg = self._command.get("turn_deg")
        manual_key = self._command.get("manual_key")

        # ---- 手动驾驶指令 → 直接发到小车 ----
        if manual_key:
            action = self._command.get("manual_action", "down")
            self._log(f"[MANUAL] 手动指令: {action} {manual_key}")
            motor = get_motor()
            motor.send_key_event(action, manual_key)
            self._set(nav_state="IDLE", mode="idle")
            return

        # ---- 简单移动指令（不需导航） ----
        if distance_cm:
            direction = "forward" if "forward" in self._command.get("actions", []) else "backward"
            self._log(f"[PLANNER] 简单指令: {direction} {distance_cm}cm")
            motor = get_motor()
            if direction == "forward":
                motor.forward(distance_cm)
            else:
                motor.backward(distance_cm)
            self._log(f"[PLANNER] 移动完成")
            self._set(nav_state="IDLE", mode="idle")
            return

        if turn_deg:
            direction = "left" if "turn_left" in self._command.get("actions", []) else "right"
            self._log(f"[PLANNER] 简单指令: 转{direction} {turn_deg}°")
            motor = get_motor()
            if direction == "left":
                motor.start_turn_left()
            else:
                motor.start_turn_right()
            time.sleep(turn_deg / 60.0)  # ~60°/s
            motor.stop()
            self._log(f"[PLANNER] 转弯完成")
            self._set(nav_state="IDLE", mode="idle")
            return

        # ---- 导航指令 ----
        # breed=None, zone=X → 只导航不找猫
        # breed=X, zone=Y → 导航到Y找X
        # breed=X, zone=None → 探索所有猫区找X

        if zone:
            waypoints = get_waypoints(self.map_data, "start", zone)
            if waypoints is None:
                self._log(f"[PLANNER] 无可用路径: start → {zone}")
                self._set(nav_state="FAILED", mode="failed")
                return
            self._exploring = False
            self._log(f"[PLANNER] 前往 {zone}: {' → '.join(w['id'] for w in waypoints)}" +
                      (f" 找 {breed}" if breed else " (不找猫)"))
        elif breed:
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
            self._log(f"[PLANNER] 探索找 {breed}: {' → '.join(w['id'] for w in waypoints)}")
            self._log(f"[PLANNER] 猫区: {cat_ids}")
        else:
            self._log("[PLANNER] 无操作")
            self._set(nav_state="IDLE", mode="idle")
            return

        self._waypoints = waypoints
        self._current_wp_idx = 1  # 跳过 start
        self._set(
            route=[w["id"] for w in waypoints],
            current_node="start",
            nav_state="TURNING",
        )

    def _check_manual_override(self) -> bool:
        """如果手动驾驶介入过，暂停并等结束，从当前位置继续。"""
        motor = get_motor()
        if not motor.manual_active(grace=1.5):
            return False

        self._log("[PAUSE] 手动驾驶介入，暂停自主导航...")
        self._set(nav_state="PAUSED")
        # 等手动结束
        while motor.manual_active(grace=1.5):
            time.sleep(0.2)
        self._log("[RESUME] 手动结束，从当前位置继续")

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
            self._log(f"[TURNING] 已对准 {target['id']} (误差 {turn_deg:.1f}°)")
            self._set(current_node=target["id"], nav_state="DRIVING")
            return

        self._log(
            f"[TURNING] 转向 {target['id']} "
            f"当前={current:.1f}° 目标={target_heading:.1f}° "
            f"需转{turn_dir} {turn_deg:.1f}°"
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
            current = motor.get_heading() % 360
            diff = (target_heading - current) % 360
            error = min(diff, 360 - diff)

            if abs(current - start_heading) > 1:
                heading_changed = True

            self._set(current_node=target["id"])

            if error < HEADING_TOLERANCE:
                motor.stop()
                time.sleep(0.1)
                break

            # heading 不更新 → 回退计时转弯（1.5s 估转 90°）
            if not heading_changed and time.time() > deadline - TURN_TIMEOUT + 1.5:
                self._log("[TURNING] heading 无更新，用计时估转")
                time.sleep(turn_deg / 60.0)  # ~60°/s
                motor.stop()
                break

            time.sleep(0.05)

        motor.stop()
        final_heading = motor.get_heading() % 360
        final_error = abs(target_heading - final_heading)
        if final_error > 180:
            final_error = 360 - final_error
        self._log(f"[TURNING] 对准完成 (误差 {final_error:.1f}°)")
        self._set(current_node=target["id"], nav_state="DRIVING")

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
        self._log(f"[DRIVING] 驶向 {target_id} ({dist_cm}cm)...")

        # 前进 + 超声波监控
        if motor.is_connected():
            motor.forward(dist_cm)
            if motor.is_blocked():
                self._log(f"[DRIVING] 超声波检测到障碍物！")
                self._set(nav_state="OBSTACLE_WAIT")
                return
        else:
            # 模拟模式
            motor.forward(dist_cm)

        self._current_wp_idx += 1
        self._set(current_node=target_id, nav_state="ARRIVED")

    def _do_obstacle_wait(self):
        """停车等 2 秒 → 重测超声波。"""
        motor = get_motor()
        motor.stop()
        self._log("[OBSTACLE_WAIT] 停车等待 2 秒...")
        time.sleep(2.0)
        if motor.is_blocked():
            self._log("[OBSTACLE_WAIT] 障碍物仍在，触发重规划")
            self._set(nav_state="REPLANNING")
        else:
            self._log("[OBSTACLE_WAIT] 障碍物已清除，继续")
            self._set(nav_state="TURNING")

    def _do_replanning(self):
        """删掉当前被堵的边 → 重跑 A*。"""
        self._log("[REPLANNING] 删除被堵边，重跑 A* ...")
        prev_idx = max(0, self._current_wp_idx - 1)
        if prev_idx < len(self._waypoints) and self._current_wp_idx < len(self._waypoints):
            u = self._waypoints[prev_idx]["id"]
            v = self._waypoints[self._current_wp_idx]["id"]
            self.map_data["edges"] = [
                e for e in self.map_data["edges"]
                if not ((e["from"] == u and e["to"] == v) or
                        (e["from"] == v and e["to"] == u))
            ]
            self._log(f"[REPLANNING] 已删除边 {u} ↔ {v}")

        zone = self._command.get("zone")
        current_node = self._waypoints[prev_idx]["id"]
        if zone:
            waypoints = get_waypoints(self.map_data, current_node, zone)
        else:
            waypoints = get_waypoints(self.map_data, current_node,
                                      self._waypoints[-1]["id"])

        if waypoints is None:
            self._log("[REPLANNING] 无备用路径")
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
                self._log(f"[ARRIVED] 到达猫区 {node_id}，开始搜索...")
                self._set(nav_state="SEARCHING")
                return
            elif node_type == "cat_zone":
                # 无品种，只导航不找猫 → 到达即完成
                self._log(f"[ARRIVED] 已到达 {node_id}")
                self._set(nav_state="IDLE", mode="idle", current_node=node_id)
                return

        # 非猫区或无效索引
        if idx >= len(self._waypoints):
            if self._command.get("breed"):
                self._log("[ARRIVED] 所有区域已搜索完毕")
                self._set(nav_state="FAILED", mode="failed")
            else:
                # 导航模式：到达终点
                self._log(f"[ARRIVED] 已到达 {self._waypoints[-1]['id']}")
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
        self._log(f"[SEARCHING] 在 {current_node} 寻找 {target_breed}，360° 旋转中...")

        reset_detection()
        motor = get_motor()
        found_breed = None

        # 旋转搜索（真车用电机，模拟用延时）
        if motor.is_connected():
            motor.start_turn_right()
            start_heading = motor.get_heading()

        deadline = time.time() + 15.0
        rotated = 0.0

        while time.time() < deadline:
            # 检查航向（真车）或模拟旋转进度
            if motor.is_connected():
                delta = abs(motor.get_heading() - start_heading)
                if delta > 180:
                    delta = 360 - delta
                rotated = delta
                if delta > 340:
                    break

            # 摄像头 YOLO 检测——真车和模拟都用
            det = get_latest_detection()
            if det and det.get("breed") == target_breed:
                found_breed = det["breed"]
                self._log(f"[SEARCHING] 视觉检测匹配: {found_breed} "
                          f"(cls={det.get('classification_confidence', 0):.2f})")
                break

            if not motor.is_connected():
                # 模拟模式：等待摄像头检测，超时 3 秒
                if time.time() > deadline - 12:
                    break
            time.sleep(0.1)

        if motor.is_connected():
            motor.stop()
            self._log(f"[SEARCHING] 旋转完成 (转了 {rotated:.0f}°)")
        else:
            self._log(f"[SEARCHING] 搜索完成 (等待 {3:.0f}s 摄像头检测)")

        if found_breed == target_breed:
            self._set(nav_state="SUCCESS")
        else:
            self._log(f"[SEARCHING] {current_node} 未找到 {target_breed}")
            with self._lock:
                if current_node not in self._state["explored_zones"]:
                    self._state["explored_zones"].append(current_node)
            if self._current_wp_idx >= len(self._waypoints):
                self._set(nav_state="FAILED", mode="failed")
            else:
                self._set(nav_state="TURNING")

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
            self._log(f"[ARRIVED] 已到达 {found_zone}")
            self._set(nav_state="IDLE", mode="idle", current_node=found_zone,
                      cat_found=False)
            return

        self._log(f"[SUCCESS] 在 {found_zone} 找到 {breed}，执行动作: {actions}")

        self._set(
            nav_state="SUCCESS",
            mode="success",
            cat_found=True,
            cat_breed=breed,
            cat_zone=found_zone,
        )

        from .action import run_care_actions
        # 逐动作执行，前端可看到实时进度
        action_labels = {
            "play": "陪它玩耍", "feed": "投喂零食",
            "photo": "拍照留念", "talk": "语音安抚",
            "return": "返回起点",
        }
        for action in actions:
            label = action_labels.get(action, action)
            with self._lock:
                self._state["current_action"] = label
            self._log(f"[CARE] 正在执行: {label}")
            time.sleep(0.6)  # 模拟动作耗时
            with self._lock:
                self._state["actions_done"].append(action)
                self._state["current_action"] = f"✓ {label}"

        run_care_actions(breed, found_zone, actions)

        with self._lock:
            self._state["current_action"] = ""

        # ---- 自动返回起点 ----
        self._return_to_start()

    def _do_failed(self):
        self._log("[FAILED] 任务失败，返回起点")
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
            self._log("[RETURN] 已在起点")
            self._set(nav_state="IDLE", mode="idle", current_node="start")
            return

        waypoints = get_waypoints(self.map_data, current_node, "start")
        if waypoints is None:
            self._log("[RETURN] 无返回路径，直接重置")
            self._set(nav_state="IDLE", mode="idle", current_node="start")
            return

        self._log(f"[RETURN] 返回起点: {' → '.join(w['id'] for w in waypoints)}")
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
            self._log("[RETURN] 已到达起点 ✓")
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
