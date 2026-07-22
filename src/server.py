"""
Web 前端服务器 — 用于演示展示。
后台线程运行 NavigationController，前端通过 /api/state 轮询实时状态。
"""

import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .parser import parse_command
from .planner import DEFAULT_MAP_PATH, load_map
from .controller import NavigationController

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

# ---- 全局组件 ----

_map_data: dict | None = None
_controller: NavigationController | None = None
_asr_available: bool | None = None   # None = 未检测


def _get_map() -> dict:
    global _map_data
    if _map_data is None:
        _map_data = load_map()
    return _map_data


def _get_ctrl() -> NavigationController:
    global _controller
    if _controller is None:
        _controller = NavigationController()
    return _controller


def _check_asr() -> bool:
    """检测 ASR 依赖是否可用（sounddevice + PortAudio + faster-whisper）。"""
    global _asr_available
    if _asr_available is None:
        try:
            import sounddevice as sd  # noqa: F401
            from faster_whisper import WhisperModel  # noqa: F401
            _asr_available = True
        except Exception:
            _asr_available = False
    return _asr_available


# ---- 确保地图已加载 ----

@app.before_request
def _ensure_map():
    _get_map()


# ---- 路由 ----

@app.get("/")
def index():
    return send_from_directory(str(TEMPLATE_DIR), "index.html")


@app.get("/api/state")
def api_state():
    """返回导航控制器的实时状态。"""
    ctrl = _get_ctrl()
    state = ctrl.get_state()
    return jsonify(state)


@app.get("/api/map")
def api_map():
    return jsonify(_get_map())


@app.post("/api/command")
def api_command():
    """文本指令：解析后交给 NavigationController 后台执行。"""
    ctrl = _get_ctrl()
    if ctrl.is_busy():
        return jsonify({"ok": False, "error": "任务进行中，请等待完成"}), 409

    payload = request.get_json(force=True)
    text = str(payload.get("text", "")).strip()
    if not text:
        return jsonify({"ok": False, "error": "empty command"}), 400

    try:
        cmd = parse_command(text)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    thread = threading.Thread(
        target=ctrl.execute_mission, args=(cmd, text), daemon=True
    )
    thread.start()

    return jsonify({"ok": True, "command": cmd})


@app.get("/api/health")
def api_health():
    """报告服务能力：语音是否可用。"""
    return jsonify({
        "asr_available": _check_asr(),
        "map_loaded": _map_data is not None,
    })


@app.post("/api/voice/start")
def api_voice_start():
    """开始录音（非阻塞）。"""
    ctrl = _get_ctrl()
    if ctrl.is_busy():
        return jsonify({"ok": False, "error": "任务进行中，请等待完成"}), 409

    if not _check_asr():
        return jsonify({
            "ok": False,
            "error": "语音功能不可用：缺少 PortAudio 系统库或 faster-whisper。"
                     "请运行: sudo apt install libportaudio2 && pip install faster-whisper sounddevice",
        }), 503

    from .asr import start_recording, is_recording

    if is_recording():
        return jsonify({"ok": False, "error": "已在录音中"}), 409

    start_recording()
    ctrl.set_mode("listening")
    ctrl.log("🎤 录音中... 再次点击停止")

    return jsonify({"ok": True})


@app.post("/api/voice/stop")
def api_voice_stop():
    """停止录音 → 转录 → 解析 → 后台执行。"""
    ctrl = _get_ctrl()

    from .asr import stop_recording, speech_to_text, is_recording

    if not is_recording():
        # 可能已经停止了，尝试直接停止（清理残留）
        pass

    audio_path = stop_recording()
    ctrl.set_mode("idle")

    if audio_path is None:
        ctrl.log("录音太短，请重试")
        return jsonify({"ok": False, "error": "录音太短（<0.3 秒），请长按录音按钮说完再停"}), 400

    try:
        ctrl.log("📝 正在转录...")
        text = speech_to_text(audio_path)
    except Exception as e:
        ctrl.log(f"转录失败: {e}")
        return jsonify({"ok": False, "error": f"转录失败: {e}"}), 500

    ctrl.log(f"📝 转录: {text}")

    try:
        cmd = parse_command(text)
    except ValueError as e:
        ctrl.log(f"解析失败: {e}")
        return jsonify({"ok": False, "error": str(e), "transcript": text}), 400

    thread = threading.Thread(
        target=ctrl.execute_mission, args=(cmd, text), daemon=True
    )
    thread.start()

    return jsonify({"ok": True, "transcript": text, "command": cmd})


# ---- 启动 ----

def main():
    ctrl = _get_ctrl()
    ctrl.log("Web 前端已启动: http://127.0.0.1:8080")
    app.run(host="127.0.0.1", port=8080, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
