"""
小车电机控制 — 对接 MotorShield_Precision 协议。

通信链路：笔记本 → TCP:8765 → 树莓派 → 串口 /dev/ttyACM0 → Arduino

Arduino 指令：
  w <speed 1-3> <distance_cm>   — 前进精确距离（Hall 编码器）
  s <speed 1-3> <distance_cm>   — 后退精确距离
  a / d                         — 持续左转 / 右转
  x                             — 停止
  status                        — 返回 STATUS 行

反馈字段：heading_deg, hall_signed, distance_signed_cm, motion

未连接真车时自动退回模拟模式。
"""

import json
import math
import socket
import threading
import time
from dataclasses import dataclass, field


# ---- 默认连接参数 ----

PI_HOST = "100.87.177.70"
CONTROL_PORT = 8765
SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = 9600

DEFAULT_SPEED = 2         # 1=慢 2=中 3=快
TURN_PWM = 180             # 转弯 PWM（Arduino 中等速度）
HEADING_TOLERANCE = 15.0   # 航向容差（度）
OBSTACLE_DISTANCE = 15     # 障碍判定距离（cm）


@dataclass
class CarState:
    """小车实时状态。"""
    heading_deg: float = 90.0
    hall_signed: int = 0
    hall_abs: int = 0
    distance_signed_cm: float = 0.0
    distance_abs_cm: float = 0.0
    motion: str = "idle"          # idle | forward | backward | turn_left | turn_right | blocked_front
    mode: str = "step"
    speed_level: int = 2
    distance_cm: int = 10
    connected: bool = False
    logs: list = field(default_factory=list)


class MotorController:
    """小车电机控制器。自动检测连接模式。"""

    def __init__(self, host: str = PI_HOST, port: int = CONTROL_PORT):
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._running = False
        self._reader_thread: threading.Thread | None = None
        self._buffer = ""

        # 共享状态
        self.state = CarState()
        self._state_lock = threading.Lock()

        # 上次转向时的航向
        self._turn_start_heading = 0.0
        self._cumulative_turn = 0.0
        self._sim_turn_thread: threading.Thread | None = None
        self._last_manual = 0.0  # 最后一次手动驾驶时间

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """连接小车（TCP 到树莓派）。成功返回 True。"""
        try:
            self._sock = socket.create_connection(
                (self._host, self._port), timeout=5
            )
            self._sock.settimeout(0.5)
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError as e:
            print(f"[MOTOR] 无法连接小车 ({self._host}:{self._port}): {e}")
            print("[MOTOR] 退回模拟模式")
            return False

        self._running = True
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True
        )
        self._reader_thread.start()

        # 初始化：step 模式 + 速度 2
        self._send_raw("mode step")
        time.sleep(0.3)
        self._send_raw("v 2")
        time.sleep(0.1)

        with self._state_lock:
            self.state.connected = True

        print(f"[MOTOR] 已连接小车 {self._host}:{self._port}")
        return True

    def disconnect(self):
        """断开连接。"""
        self._running = False
        if self._sock:
            try:
                self._send_raw("x")
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        with self._state_lock:
            self.state.connected = False
        print("[MOTOR] 已断开")

    def is_connected(self) -> bool:
        with self._state_lock:
            return self.state.connected

    # ------------------------------------------------------------------
    # 运动控制
    # ------------------------------------------------------------------

    def forward(self, distance_cm: float, speed: int = DEFAULT_SPEED):
        """前进指定距离（cm），阻塞直到完成。

        连接真车时用 step 指令（需 Pi 补丁），
        无补丁时自动回退 hold 模式（key_sender 同款协议）。
        """
        speed = max(1, min(3, speed))
        distance_cm = max(1, int(distance_cm))

        if not self.is_connected():
            self._simulate_drive("forward", distance_cm)
            return

        self._log(f"[MOTOR] 前进 {distance_cm}cm (速度{speed})")

        # 先试 step 指令（精确，需 Pi 补丁）
        self._send_step(f"w {speed} {distance_cm}")
        time.sleep(0.5)

        # 0.5s 后检测车是否真的在动
        with self._state_lock:
            moving = self.state.motion in ("forward", "backward")

        if not moving:
            # 回退：hold 模式 + 计时（和 key_sender 一样）
            self._log("[MOTOR] step 未生效，用 hold 模式...")
            self._send_key("down", str(speed))
            time.sleep(0.05)
            self._send_raw("mode hold")
            time.sleep(0.1)
            self._send_key("down", "w")
            time.sleep(distance_cm / 45.0)   # ~45cm/s 估算
            self._send_key("down", "x")
            time.sleep(0.15)
            self._send_raw("mode step")
            time.sleep(0.1)
        else:
            self._wait_motion_idle(timeout=distance_cm * 0.15 + 5)

        self._request_status()

    def backward(self, distance_cm: float, speed: int = DEFAULT_SPEED):
        """后退指定距离（cm），阻塞直到完成。"""
        speed = max(1, min(3, speed))
        distance_cm = max(1, int(distance_cm))

        if not self.is_connected():
            self._simulate_drive("backward", distance_cm)
            return

        self._log(f"[MOTOR] 后退 {distance_cm}cm (速度{speed})")
        self._send_step(f"s {speed} {distance_cm}")
        time.sleep(0.5)

        with self._state_lock:
            moving = self.state.motion in ("forward", "backward")
        if not moving:
            self._log("[MOTOR] step 未生效，用 hold 模式...")
            self._send_key("down", str(speed))
            time.sleep(0.05)
            self._send_raw("mode hold")
            time.sleep(0.1)
            self._send_key("down", "s")
            time.sleep(distance_cm / 45.0)
            self._send_key("down", "x")
            time.sleep(0.15)
            self._send_raw("mode step")
            time.sleep(0.1)
        else:
            self._wait_motion_idle(timeout=distance_cm * 0.15 + 5)

        self._request_status()

    def start_turn_left(self):
        """开始左转（非阻塞）。"""
        if not self.is_connected():
            self._simulate_turn("turn_left")
            return
        self._log("[MOTOR] 左转")
        self._send_key("down", "a")
        with self._state_lock:
            self._turn_start_heading = self.state.heading_deg
            self._cumulative_turn = 0.0
            self.state.motion = "turn_left"

    def start_turn_right(self):
        """开始右转（非阻塞）。"""
        if not self.is_connected():
            self._simulate_turn("turn_right")
            return
        self._log("[MOTOR] 右转")
        self._send_key("down", "d")
        with self._state_lock:
            self._turn_start_heading = self.state.heading_deg
            self._cumulative_turn = 0.0
            self.state.motion = "turn_right"

    def stop_turn(self):
        """停止转弯。"""
        with self._state_lock:
            was_turning = self.state.motion in ("turn_left", "turn_right")
            turn_key = "a" if self.state.motion == "turn_left" else "d"
            self.state.motion = "idle"

        if self.is_connected() and was_turning:
            self._send_key("up", turn_key)
            self._send_key("down", "x")

    # ------------------------------------------------------------------
    # 传感器查询
    # ------------------------------------------------------------------

    def get_heading(self) -> float:
        """当前航向角（度）。"""
        with self._state_lock:
            return self.state.heading_deg

    def send_key_event(self, action: str, key: str):
        """手动控制：发送键盘事件到小车。action=down/up, key=w/a/s/d/x/space。"""
        if not self.is_connected():
            return
        self._send_key(action, key)
        if action == "down" and key in ("w", "a", "s", "d"):
            self._last_manual = time.time()

    def manual_active(self, grace: float = 2.0) -> bool:
        """最近 grace 秒内有手动驾驶操作？"""
        return time.time() - self._last_manual < grace

    def get_turn_delta(self) -> float:
        """从上次 start_turn 以来的累积转角（度）。"""
        with self._state_lock:
            current = self.state.heading_deg
        delta = current - self._turn_start_heading
        # 标准化到 [-180, 180]
        if delta > 180:
            delta -= 360
        elif delta < -180:
            delta += 360
        return abs(delta)

    def is_moving(self) -> bool:
        with self._state_lock:
            return self.state.motion not in ("idle", "blocked_front")

    def is_blocked(self) -> bool:
        with self._state_lock:
            return self.state.motion == "blocked_front"

    # ------------------------------------------------------------------
    # 内部：TCP 通信
    # ------------------------------------------------------------------

    def _send_raw(self, text: str):
        """发送原始文本行。"""
        if not self._sock:
            return
        try:
            self._sock.sendall(f"{text}\n".encode("utf-8"))
        except OSError:
            pass

    def _send_key(self, action: str, key: str):
        """发送键盘事件（标准协议）。"""
        self._send_raw(f"{action} {key}")

    def _send_step(self, cmd: str):
        """发送 step 指令（扩展协议：step <arduino_cmd>）。

        需要在树莓派 car_control.py 的 serve() 中添加：
            if parts[0] == "step" and len(parts) > 1:
                controller.send_line(parts[1])
                continue
        """
        self._send_raw(f"step {cmd}")

    def _request_status(self):
        self._send_step("status")

    def _reader_loop(self):
        """后台读取树莓派发来的 JSON 状态。"""
        while self._running and self._sock:
            try:
                data = self._sock.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            if not data:
                break

            self._buffer += data.decode("utf-8", errors="ignore")
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._handle_payload(payload)

        with self._state_lock:
            self.state.connected = False

    def _handle_payload(self, p: dict):
        ptype = p.get("type", "")
        if ptype == "status":
            with self._state_lock:
                self.state.mode = p.get("mode", self.state.mode)
                self.state.speed_level = int(p.get("speed_level", self.state.speed_level))
                self.state.distance_cm = int(p.get("distance_cm", self.state.distance_cm))
                self.state.motion = p.get("motion", self.state.motion)
                self.state.hall_signed = int(p.get("hall_signed", self.state.hall_signed))
                self.state.hall_abs = int(p.get("hall_abs", self.state.hall_abs))
                self.state.distance_signed_cm = float(
                    p.get("distance_signed_cm", self.state.distance_signed_cm)
                )
                self.state.distance_abs_cm = float(
                    p.get("travel_cm", self.state.distance_abs_cm)
                )
                self.state.heading_deg = float(
                    p.get("heading_deg", self.state.heading_deg)
                )
        elif ptype == "log":
            self._log(f"[PI] {p.get('message', '')}")

    def _wait_motion_idle(self, timeout: float = 10.0):
        """等待 Arduino 完成 step 运动。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._state_lock:
                if self.state.motion in ("idle", "blocked_front"):
                    return
            time.sleep(0.05)

    def _log(self, msg: str):
        with self._state_lock:
            self.state.logs.append(msg)
            if len(self.state.logs) > 100:
                self.state.logs = self.state.logs[-100:]
        print(msg)

    # ------------------------------------------------------------------
    # 模拟模式
    # ------------------------------------------------------------------

    def _simulate_drive(self, direction: str, distance_cm: float):
        """模拟行驶。"""
        secs = distance_cm / 50.0  # 假设 50cm/s
        self._log(f"[SIM] {direction} {distance_cm:.0f}cm ({secs:.1f}s)")
        time.sleep(max(0.1, secs))

    def _simulate_turn(self, direction: str):
        """模拟转弯——持续更新航向直到外部 stop()。"""
        self._log(f"[SIM] {direction}")
        # 启动后台线程持续更新航向
        sign = 1 if direction == "turn_right" else -1
        with self._state_lock:
            self.state.motion = direction

        def _turn_loop():
            while True:
                with self._state_lock:
                    if self.state.motion not in ("turn_left", "turn_right"):
                        break
                    self.state.heading_deg = (self.state.heading_deg + sign * 30) % 360
                time.sleep(0.15)

        t = threading.Thread(target=_turn_loop, daemon=True)
        t.start()
        self._sim_turn_thread = t

    def stop(self):
        """停止所有运动。"""
        if not self.is_connected():
            with self._state_lock:
                self.state.motion = "idle"
            return
        self._log("[MOTOR] 停止")
        self._send_key("down", "x")
        with self._state_lock:
            self.state.motion = "idle"


# ---- 模块级单例 ----

_motor: MotorController | None = None
_lock = threading.Lock()


def get_motor() -> MotorController:
    global _motor
    with _lock:
        if _motor is None:
            _motor = MotorController()
        return _motor
