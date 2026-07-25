"""
Web 前端服务器 — 用于演示展示。
后台线程运行 NavigationController，前端通过 /api/state 轮询实时状态。
"""

import threading
import importlib.util
import os
import socket
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from .parser import is_safe_voice_command, parse_command
from .planner import DEFAULT_MAP_PATH, load_map
from .controller import NavigationController
from .camera import (
    start_camera,
    stop_camera,
    mjpeg_generator,
    get_cat_status,
    set_inference_paused,
)
from .motor import get_motor

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

# ---- 全局组件 ----

_map_data: dict | None = None
_controller: NavigationController | None = None
_asr_available: bool | None = None   # None = 未检测
_mission_submit_lock = threading.Lock()
_voice_processing_lock = threading.Lock()
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8090"))


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
    """检测 ASR 依赖和输入设备，不加载 Whisper 模型。"""
    global _asr_available
    # Retry a previous failure so a microphone connected after startup works.
    if _asr_available is not True:
        dependencies_available = (
            importlib.util.find_spec("sounddevice") is not None
            and importlib.util.find_spec("faster_whisper") is not None
        )
        if not dependencies_available:
            _asr_available = False
        else:
            try:
                import sounddevice as sd
                device = sd.query_devices(kind="input")
                _asr_available = int(device.get("max_input_channels", 0)) > 0
            except Exception:
                _asr_available = False
    return _asr_available


def _port_is_available(host: str, port: int) -> bool:
    """Check the HTTP port before connecting hardware or loading models."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
    except OSError:
        return False
    return True


def _submit_mission(ctrl: NavigationController, cmd: dict, text: str) -> bool:
    """原子地预留控制器并启动任务，防止两个请求同时通过 busy 检查。"""
    with _mission_submit_lock:
        if ctrl.is_busy():
            return False
        target = cmd.get("zone") or cmd.get("breed") or "direct command"
        pause_node = cmd.get("pause_node")
        ctrl.log(
            f"[MISSION] Accepted target={target}"
            + (f", stop_at={pause_node}" if pause_node else "")
        )
        ctrl.set_mode("running")
        thread = threading.Thread(
            target=ctrl.execute_mission, args=(cmd, text), daemon=True
        )
        try:
            thread.start()
        except RuntimeError:
            ctrl.set_mode("idle")
            raise
    return True


def _apply_navigation_control(
    ctrl: NavigationController,
    cmd: dict,
) -> tuple[bool, str]:
    """Apply pause-at/continue without replacing the active mission."""
    action = cmd.get("control_action")
    if action == "pause_at":
        node_id = cmd.get("pause_node")
        if not node_id:
            return False, "No junction was specified"
        return ctrl.request_pause_at(node_id)
    if action == "continue":
        return ctrl.continue_mission()
    return False, "Unsupported navigation control"


# ---- 确保地图已加载 ----

@app.before_request
def _ensure_map():
    _get_map()


@app.after_request
def _disable_browser_cache(response):
    """The dashboard is edited frequently during hardware integration."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ---- 路由 ----

@app.get("/")
def index():
    return send_from_directory(str(TEMPLATE_DIR), "index.html")


@app.get("/api/state")
def api_state():
    """返回导航控制器 + 猫检测的实时状态。"""
    ctrl = _get_ctrl()
    state = ctrl.get_state()
    state["cat"] = get_cat_status()
    state["robot_connected"] = get_motor().is_connected()
    return jsonify(state)


@app.get("/api/map")
def api_map():
    return jsonify(_get_map())


@app.post("/api/command")
def api_command():
    """文本指令：解析后交给 NavigationController 后台执行。"""
    ctrl = _get_ctrl()

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "invalid JSON payload"}), 400
    text = str(payload.get("text", "")).strip()
    if not text:
        return jsonify({"ok": False, "error": "empty command"}), 400

    try:
        cmd = parse_command(text)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    if cmd.get("control_action"):
        ok, error = _apply_navigation_control(ctrl, cmd)
        if not ok:
            return jsonify({"ok": False, "error": error, "command": cmd}), 409
        return jsonify({"ok": True, "command": cmd})

    # 手动驾驶指令（停/向前/左转等）无论是否 busy 都立即执行
    if cmd.get("manual_key"):
        ctrl.log(f"[MANUAL] Command: {text}")
        ctrl.cancel_mission(
            "Stop command received"
            if cmd["manual_key"] == "x"
            else "Manual command override"
        )
        motor = get_motor()
        sent = motor.send_key_event(cmd["manual_action"], cmd["manual_key"])
        if not sent and cmd["manual_key"] != "x":
            return jsonify({"ok": False, "error": "Robot is offline", "command": cmd}), 503
        return jsonify({"ok": True, "sent": sent, "command": cmd})

    if not _submit_mission(ctrl, cmd, text):
        return jsonify({"ok": False, "error": "A mission is already running"}), 409

    return jsonify({"ok": True, "command": cmd})


@app.post("/api/interaction")
def api_interaction():
    """确认或跳过找到动物后的护理动作。"""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "invalid JSON payload"}), 400

    decision = str(payload.get("decision", "")).strip().lower()
    if decision not in {"execute", "skip"}:
        return jsonify({
            "ok": False,
            "error": "decision must be 'execute' or 'skip'",
        }), 400

    accepted = _get_ctrl().resolve_interaction(decision == "execute")
    if not accepted:
        return jsonify({
            "ok": False,
            "error": "No interaction is waiting for confirmation",
        }), 409
    return jsonify({"ok": True, "decision": decision})


@app.get("/api/camera/stream")
def api_camera_stream():
    """MJPEG 摄像头流（带猫检测标注）。"""
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/health")
def api_health():
    """报告服务能力，便于前端区分断流、模型缺失和机器人离线。"""
    vision = get_cat_status()
    asr_available = _check_asr()
    asr_config = None
    if asr_available:
        from .asr import get_asr_config
        asr_config = get_asr_config()
    return jsonify({
        "asr_available": asr_available,
        "asr_config": asr_config,
        "map_loaded": _map_data is not None,
        "vision_status": vision["status"],
        "vision_message": vision["message"],
        "robot_connected": get_motor().is_connected(),
    })


@app.post("/api/voice/start")
def api_voice_start():
    """开始录音（非阻塞）。"""
    ctrl = _get_ctrl()

    if not _check_asr():
        return jsonify({
            "ok": False,
            "error": "Voice input is unavailable. PortAudio or faster-whisper is missing.",
        }), 503
    if _voice_processing_lock.locked():
        return jsonify({
            "ok": False,
            "error": "The previous voice command is still being processed.",
        }), 409

    from .asr import start_recording, is_recording

    if is_recording():
        return jsonify({"ok": False, "error": "Recording is already active"}), 409

    try:
        start_recording()
    except Exception as e:
        ctrl.log(f"Recording failed to start: {e}")
        return jsonify({"ok": False, "error": f"Recording failed to start: {e}"}), 500
    if not ctrl.is_busy():
        ctrl.set_mode("listening")
    ctrl.log("[VOICE] Recording. Press Stop when finished.")

    return jsonify({"ok": True})


@app.post("/api/voice/stop")
def api_voice_stop():
    """停止录音 → 转录 → 解析 → 后台执行。"""
    ctrl = _get_ctrl()

    from .asr import (
        stop_recording,
        speech_to_text_result,
        is_recording,
        is_chinese_or_english,
        is_repetitive_transcript,
        normalize_command_transcript,
        transcript_preview,
        ASR_MIN_LOG_PROB,
        voice_command_min_log_prob,
    )

    if not is_recording():
        return jsonify({"ok": False, "error": "No recording is active"}), 409

    if not _voice_processing_lock.acquire(blocking=False):
        return jsonify({
            "ok": False,
            "error": "A voice command is already being processed.",
        }), 409

    try:
        audio_path = stop_recording()
    except Exception as e:
        _voice_processing_lock.release()
        ctrl.set_mode("idle")
        ctrl.log(f"[VOICE] Recording failed to stop: {e}")
        return jsonify({
            "ok": False,
            "error": f"Recording failed to stop: {e}",
        }), 500
    if ctrl.get_state().get("mode") == "listening":
        ctrl.set_mode("idle")

    if audio_path is None:
        _voice_processing_lock.release()
        ctrl.log("[VOICE] Recording was too short")
        return jsonify({"ok": False, "error": "Recording is too short. Speak for at least 0.3 seconds."}), 400

    candidates: list[dict] = []
    cmd = None
    text = ""
    last_transcribe_error = None
    repeated_preview = ""
    try:
        ctrl.log("[VOICE] Processing speech...")
        set_inference_paused(True)

        # Compare automatic, Mandarin, and English decoding. A forced-language
        # result is never accepted merely because it happens to contain a keyword.
        for language in (None, "zh", "en"):
            label = language or "auto"
            try:
                result = speech_to_text_result(audio_path, language=language)
            except Exception as e:
                last_transcribe_error = e
                continue
            candidate = result["text"]
            if not candidate:
                continue
            if is_repetitive_transcript(candidate):
                repeated_preview = transcript_preview(candidate)
                print(
                    f"[VOICE DEBUG] Rejected repetitive transcript ({label}): "
                    f"{repeated_preview}"
                )
                continue
            normalized = normalize_command_transcript(candidate)
            if normalized != candidate:
                print(f"[VOICE DEBUG] Normalized ({label}): {candidate} -> {normalized}")
            candidate = normalized
            if any(candidate == item["text"] for item in candidates):
                continue
            if not is_chinese_or_english(candidate):
                print(f"[VOICE DEBUG] Ignored non-Chinese/English result ({label}): {candidate}")
                continue
            try:
                parsed = parse_command(candidate, allow_llm=False)
            except ValueError:
                parsed = None
            if parsed is not None and not is_safe_voice_command(candidate, parsed):
                print(f"[VOICE DEBUG] Rejected partial movement command ({label}): {candidate}")
                parsed = None
            candidate_result = {
                "language": label,
                "text": candidate,
                "avg_logprob": result["avg_logprob"],
                "detected_language": result["language"],
                "language_probability": result["language_probability"],
                "command": parsed,
            }
            candidates.append(candidate_result)
            print(
                f"[VOICE DEBUG] Transcript ({label}, confidence="
                f"{result['avg_logprob']:.2f}): {candidate}"
            )

        valid_candidates = [
            item for item in candidates
            if (
                item["command"] is not None
                and item["avg_logprob"] >= voice_command_min_log_prob(item["command"])
            )
        ]
        if valid_candidates:
            best = max(
                valid_candidates,
                key=lambda item: (
                    item["avg_logprob"]
                    + (0.15 * item["language_probability"] if item["language"] == "auto" else 0)
                    + (0.25 if item["command"].get("pause_node") else 0)
                    + (0.15 if item["command"].get("control_action") else 0)
                ),
            )
            cmd = best["command"]
            text = best["text"]

        # 规则都无法识别时，最后才让 LLM 处理候选文本。
        if cmd is None:
            eligible = [
                item for item in candidates
                if item["avg_logprob"] >= ASR_MIN_LOG_PROB
            ]
            if eligible:
                item = max(eligible, key=lambda value: value["avg_logprob"])
                try:
                    cmd = parse_command(item["text"])
                    if not is_safe_voice_command(item["text"], cmd):
                        cmd = None
                    else:
                        text = item["text"]
                except ValueError:
                    cmd = None
    finally:
        set_inference_paused(False)
        try:
            Path(audio_path).unlink(missing_ok=True)
        except OSError:
            pass
        _voice_processing_lock.release()

    if not candidates:
        if last_transcribe_error:
            ctrl.log(f"[VOICE] Transcription failed: {last_transcribe_error}")
            return jsonify({
                "ok": False,
                "error": f"Transcription failed: {last_transcribe_error}",
            }), 500
        if repeated_preview:
            error = (
                "Unreliable repeated speech output was detected. "
                "Record one short command and stop immediately after speaking."
            )
            ctrl.log(f"[VOICE] Command rejected: {error}")
            return jsonify({
                "ok": False,
                "error": error,
                "transcript": repeated_preview,
            }), 400
        ctrl.log("[VOICE] No speech detected")
        return jsonify({"ok": False, "error": "No speech detected. Move closer to the microphone and retry."}), 400

    if cmd is None:
        best_candidate = max(candidates, key=lambda item: item["avg_logprob"])
        confidence = best_candidate["avg_logprob"]
        confidence_threshold = voice_command_min_log_prob(best_candidate["command"])
        if confidence < confidence_threshold:
            error = (
                f"Low speech confidence ({confidence:.2f}). "
                "Move closer to the microphone and speak once."
            )
        else:
            error = "The speech does not match a supported robot command."
        ctrl.log(f"[VOICE] Command rejected: {error}")
        return jsonify({
            "ok": False,
            "error": error,
            "transcript": best_candidate["text"],
            "confidence": confidence,
            "candidates": candidates,
        }), 400

    ctrl.log(f"[VOICE] Heard: {text}")
    if cmd.get("control_action"):
        ok, error = _apply_navigation_control(ctrl, cmd)
        if not ok:
            ctrl.log(f"[VOICE] Navigation control rejected: {error}")
            return jsonify({
                "ok": False,
                "error": error,
                "transcript": text,
                "command": cmd,
            }), 409
        return jsonify({
            "ok": True,
            "transcript": text,
            "command": cmd,
        })

    if cmd.get("manual_key"):
        ctrl.log(f"[MANUAL] Voice command: {text}")
        ctrl.cancel_mission(
            "Voice stop command received"
            if cmd["manual_key"] == "x"
            else "Voice manual command override"
        )
        motor = get_motor()
        sent = motor.send_key_event(cmd["manual_action"], cmd["manual_key"])
        if not sent and cmd["manual_key"] != "x":
            return jsonify({
                "ok": False,
                "error": "Robot is offline",
                "transcript": text,
                "command": cmd,
            }), 503
        return jsonify({"ok": True, "sent": sent, "transcript": text, "command": cmd})

    if not _submit_mission(ctrl, cmd, text):
        return jsonify({
            "ok": False,
            "error": "A mission is already running",
            "transcript": text,
            "command": cmd,
        }), 409

    return jsonify({"ok": True, "transcript": text, "command": cmd})


@app.post("/api/manual")
def api_manual():
    """WASD 手动驾驶。"""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "invalid JSON payload"}), 400
    action = str(payload.get("action", "")).lower()
    key = str(payload.get("key", "")).lower()

    if action not in ("down", "up") or key not in (
        "w", "a", "s", "d", "x",             # 移动（注意：space 现在是机械爪，不再停车）
        "m", "1", "2", "3", "[", "]",        # 模式/速度/步长
        "c",                                   # 拍照
        "q", "e", "r", "f", "t", "g",        # 机械臂 (基座/大臂/小臂)
        "space", "z", "p",                    # 机械爪 / HOME / DEMO
    ):
        return jsonify({"ok": False, "error": "invalid"}), 400

    motor = get_motor()
    if action == "down" and key in {
        "w", "a", "s", "d", "x", "q", "e", "r", "f", "t", "g",
        "space", "z", "p",
    }:
        _get_ctrl().cancel_mission(
            "Manual emergency stop" if key == "x" else "Keyboard manual override"
        )
    sent = motor.send_key_event(action, key)
    if not sent and key != "x":
        return jsonify({"ok": False, "error": "Robot is offline"}), 503
    return jsonify({"ok": True, "sent": sent})


# ---- 启动 ----

def main():
    if not _port_is_available(SERVER_HOST, SERVER_PORT):
        print(
            f"[STARTUP] Port {SERVER_PORT} is already in use. "
            f"Open http://127.0.0.1:{SERVER_PORT} if CatRescue is already running, "
            "or stop the existing process before restarting."
        )
        return 1

    ctrl = _get_ctrl()
    ctrl.log(f"Web console started: http://127.0.0.1:{SERVER_PORT}")

    # 连接小车
    motor = get_motor()
    if motor.connect():
        ctrl.log("[ROBOT] Connected")
    else:
        ctrl.log("[ROBOT] Offline. Simulation mode enabled.")

    start_camera()  # 自动：优先 PiCamera 流，失败回退本地摄像头
    if _check_asr():
        def _warm_asr():
            try:
                from .asr import warmup_model
                warmup_model()
                ctrl.log("[VOICE] ASR model ready")
            except Exception as e:
                ctrl.log(f"[VOICE] ASR preload failed: {e}")

        threading.Thread(target=_warm_asr, daemon=True).start()
    try:
        app.run(
            host=SERVER_HOST,
            port=SERVER_PORT,
            threaded=True,
            use_reloader=False,
        )
    finally:
        motor.disconnect()
        stop_camera()


if __name__ == "__main__":
    raise SystemExit(main())
