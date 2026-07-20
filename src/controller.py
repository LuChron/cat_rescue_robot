"""
模块 3: 导航控制器 — {breed, zone, actions} → 机器人执行
包含路径规划、导航状态机、猫视觉触发、护理动作触发。
"""

import time
from typing import Optional

from .planner import DEFAULT_MAP_PATH, load_map, get_waypoints


class NavigationController:
    """导航状态机。"""

    def __init__(self, map_path: str = str(DEFAULT_MAP_PATH)):
        self.map_data = load_map(map_path)
        self.state = "IDLE"
        self.waypoints: list[dict] = []
        self.current_waypoint_idx = 0
        self.command: Optional[dict] = None

    # ------------------------------------------------------------------
    # 状态机入口
    # ------------------------------------------------------------------

    def execute_mission(self, command: dict):
        """模块 3 入口：接收 {breed, zone, actions}，执行完整任务。"""
        self.command = command
        self.state = "PLANNING"
        self._run_state_machine()

    def _run_state_machine(self):
        """主循环，按状态转移直到终态。"""
        while self.state not in ("IDLE",):
            if self.state == "PLANNING":
                self._do_planning()
            elif self.state == "TURNING":
                self._do_turning()
            elif self.state == "DRIVING":
                self._do_driving()
            elif self.state == "OBSTACLE_WAIT":
                self._do_obstacle_wait()
            elif self.state == "REPLANNING":
                self._do_replanning()
            elif self.state == "ARRIVED":
                self._do_arrived()
            elif self.state == "SEARCHING":
                self._do_searching()
            elif self.state == "SUCCESS":
                self._do_success()
                return
            elif self.state == "FAILED":
                self._do_failed()
                return

    # ------------------------------------------------------------------
    # 各状态实现（骨架，TCP/硬件调用留好接口）
    # ------------------------------------------------------------------

    def _do_planning(self):
        """A* 搜索 → 生成路径点列表。"""
        goal = self.command["zone"]
        waypoints = get_waypoints(self.map_data, "start", goal)
        if waypoints is None:
            print(f"[PLANNER] 无可用路径: start → {goal}")
            self.state = "FAILED"
            return
        self.waypoints = waypoints
        self.current_waypoint_idx = 1  # 跳过 start
        print(f"[PLANNER] 路径: {' → '.join(w['id'] for w in waypoints)}")
        self.state = "TURNING"

    def _do_turning(self):
        """闭环转向：对准下一个路径点。"""
        if self.current_waypoint_idx >= len(self.waypoints):
            self.state = "ARRIVED"
            return
        target_id = self.waypoints[self.current_waypoint_idx]["id"]
        print(f"[TURNING] 转向 {target_id}...")
        # TODO: 发送 "a"/"d" 转弯指令，读取 hall heading 做闭环
        time.sleep(0.1)  # 占位
        self.state = "DRIVING"

    def _do_driving(self):
        """直行：发前进指令 + 轮询超声波。"""
        idx = self.current_waypoint_idx
        if idx >= len(self.waypoints):
            self.state = "ARRIVED"
            return
        target_id = self.waypoints[idx]["id"]
        print(f"[DRIVING] 驶向 {target_id}...")
        # TODO: 发送 "w" 前进指令，轮询超声波，
        #       累计 hall 距离，障碍物 → OBSTACLE_WAIT
        time.sleep(0.1)  # 占位
        self.current_waypoint_idx += 1
        if self.current_waypoint_idx >= len(self.waypoints):
            self.state = "ARRIVED"
        else:
            self.state = "TURNING"

    def _do_obstacle_wait(self):
        """停车等 2 秒 → 重测超声波。"""
        print("[OBSTACLE_WAIT] 停车等待...")
        # TODO: 发 "x" 停车，sleep(2)，再测超声波
        time.sleep(0.1)  # 占位
        self.state = "REPLANNING"

    def _do_replanning(self):
        """删掉当前被堵的边 → 重跑 A*。"""
        print("[REPLANNING] 重新规划路径...")
        # TODO: 从 graph 中删除当前边，重跑 A*
        # 若有新路径 → TURNING，否则 → FAILED
        time.sleep(0.1)  # 占位
        self.state = "TURNING"

    def _do_arrived(self):
        """到达最终节点 → 开始搜索猫。"""
        print(f"[ARRIVED] 到达 {self.command['zone']}，开始搜索...")
        self.state = "SEARCHING"

    def _do_searching(self):
        """原地旋转 + 猫视觉检测。"""
        target_breed = self.command["breed"]
        print(f"[SEARCHING] 寻找 {target_breed}...")
        # TODO: 发转向指令做 360° 旋转，
        #       持续运行 YOLO + EfficientNet，
        #       品种匹配 → SUCCESS，30s 超时 → FAILED
        time.sleep(0.1)  # 占位
        # 暂时模拟找到
        self.state = "SUCCESS"

    def _do_success(self):
        """任务成功 → 触发护理动作。"""
        breed = self.command["breed"]
        zone = self.command["zone"]
        actions = self.command["actions"]
        print(f"[SUCCESS] 在 {zone} 找到 {breed}，执行动作: {actions}")
        from .action import run_care_actions
        run_care_actions(breed, zone, actions)
        # 返回起点
        print("[RETURN] 返回起点...")
        self.state = "IDLE"

    def _do_failed(self):
        """任务失败 → 返回起点。"""
        print("[FAILED] 任务失败，返回起点...")
        self.state = "IDLE"


# 模块 3 对外接口
_controller: Optional[NavigationController] = None


def execute_mission(command: dict):
    """模块 3 唯一对外入口。"""
    global _controller
    if _controller is None:
        _controller = NavigationController()
    _controller.execute_mission(command)
