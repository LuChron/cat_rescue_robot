"""
摄像头 MJPEG 流 — 可选叠加猫检测框。
如果 cat_vision_pipeline 可用则自动启用 YOLO 检测标注。
"""

import threading
import time

import cv2

# ---- 猫检测（可选） ----

_detector_available = False
_detectors = None
_classifier = None
_classes = None
_transform = None
_device = None


def _init_detector():
    """尝试加载 cat_vision_pipeline 的 YOLO + EfficientNet。"""
    global _detector_available, _detectors, _classifier, _classes, _transform, _device
    global _cat_status, _cat_status_msg
    if _detector_available:
        return
    try:
        import sys
        sys.path.insert(
            0, "/media/zhao/Data1/task/NUS/cat_vision_pipeline"
        )
        from catvision.detector import load_detectors
        from catvision.runtime import load_classifier, SmoothedClassifier, predict_frame
        import torch
        from pathlib import Path
        from torchvision.models import EfficientNet_B2_Weights

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _cat_status_msg = str(_device)
        _detectors = load_detectors()
        model_path = Path(
            "/media/zhao/Data1/task/NUS/cat_vision_pipeline/models/classifier/"
            "best_effnet_b2_cat_breeds.pth"
        )
        class_map = Path(
            "/media/zhao/Data1/task/NUS/cat_vision_pipeline/config/class_to_idx.json"
        )
        _classifier, _classes = load_classifier(model_path, class_map, _device)
        _transform = EfficientNet_B2_Weights.DEFAULT.transforms()
        _detector_available = True
        _cat_status = "ready"
        print(f"[CAMERA] 猫检测模型已加载 (device={_device})")
    except Exception as e:
        _cat_status = "error"
        _cat_status_msg = str(e)
        print(f"[CAMERA] 猫检测不可用 ({e})，仅显示原始画面")
        _detector_available = False


# ---- 帧缓冲 & 检测结果 ----

_frame = None
_frame_lock = threading.Lock()
_latest_detection: dict | None = None  # {breed, classification_confidence, detection_confidence, detector, box}
_detection_lock = threading.Lock()
_cat_status: str = "loading"  # loading | ready | detected | no_cat | error | stream_error
_cat_status_msg: str = ""
_cap: cv2.VideoCapture | None = None
_running = False
_source_url: str | None = None


def _open_capture(source) -> cv2.VideoCapture | None:
    """打开视频源：URL 优先（树莓派 PiCamera），失败回退本地摄像头。"""
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if cap.isOpened():
        if isinstance(source, str) and source.startswith("http"):
            print(f"[CAMERA] 已连接树莓派 PiCamera: {source}")
        else:
            print(f"[CAMERA] 已连接本地摄像头")
        return cap

    # URL 失败，回退本地摄像头
    if isinstance(source, str) and source.startswith("http"):
        print(f"[CAMERA] 树莓派流不可用 ({source})，回退本地摄像头...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if cap.isOpened():
            print("[CAMERA] 已连接本地摄像头（回退模式）")
            return cap

    return None


def _capture_loop(source):
    global _frame, _running
    cap = _open_capture(source)
    if cap is None:
        print(f"[CAMERA] 无可用摄像头")
        _running = False
        return

    smoother = None
    if _detector_available:
        from catvision.runtime import SmoothedClassifier, predict_frame
        smoother = SmoothedClassifier(window=15)

    while _running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue

        # 猫检测标注
        if _detector_available:
            from catvision.runtime import predict_frame
            try:
                frame, pred = predict_frame(
                    frame, _detectors, _classifier, _classes,
                    _transform, _device, 0.25, smoother=smoother,
                )
                with _detection_lock:
                    global _latest_detection, _cat_status, _cat_status_msg
                    _latest_detection = pred
                    if pred:
                        _cat_status = "detected"
                        _cat_status_msg = f"{pred['breed']} cls={pred['classification_confidence']:.2f}"
                    else:
                        _cat_status = "no_cat"
                        _cat_status_msg = "No cat detected"
            except Exception as e:
                with _detection_lock:
                    _cat_status = "error"
                    _cat_status_msg = str(e)

        # 状态条
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
        status = "CAT DETECTION ACTIVE" if _detector_available else "CAMERA ONLY"
        cv2.putText(frame, f" {status} | {time.strftime('%H:%M:%S')}",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with _frame_lock:
            _frame = jpg.tobytes()

        time.sleep(0.03)  # ~30 fps cap

    cap.release()


def start_camera(source=None):
    """启动后台摄像头采集。

    source 可以是：
    - None：自动模式（先试 Pi 流，失败则本地摄像头）
    - str：URL（如 "http://100.87.177.70:5000/video_feed"）
    - int：本地摄像头索引（如 0）
    """
    global _running, _source_url
    if _running:
        return

    # 自动模式：优先用树莓派 PiCamera 流
    if source is None:
        source = "http://100.87.177.70:5000/video_feed"

    _init_detector()
    _running = True
    t = threading.Thread(target=_capture_loop, args=(source,), daemon=True)
    t.start()


def stop_camera():
    global _running
    _running = False


def get_frame() -> bytes | None:
    """返回最新 JPEG 帧，无数据返回 None。"""
    with _frame_lock:
        return _frame


def get_latest_detection() -> dict | None:
    """返回最新猫检测结果，供导航控制器查询。"""
    with _detection_lock:
        return _latest_detection.copy() if _latest_detection else None


def get_cat_status() -> dict:
    """返回猫检测状态，供前端展示。"""
    with _detection_lock:
        det = _latest_detection.copy() if _latest_detection else None
    return {
        "status": _cat_status,
        "message": _cat_status_msg,
        "detection": det,
    }


def reset_detection():
    """清空检测结果（每次搜索前调用）。"""
    global _latest_detection, _cat_status
    with _detection_lock:
        _latest_detection = None
        _cat_status = "no_cat"
        _cat_status_msg = ""


def mjpeg_generator():
    """MJPEG 流生成器，供 Flask Response 使用。"""
    while True:
        frame = get_frame()
        if frame is None:
            # 无摄像头时返回占位图
            placeholder = _placeholder_frame()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   placeholder + b"\r\n")
        else:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   frame + b"\r\n")
        time.sleep(0.05)


def _placeholder_frame() -> bytes:
    """摄像头不可用时的占位图。"""
    import numpy as np
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    img[:] = (22, 27, 34)  # dark bg
    cv2.putText(img, "NO CAMERA", (90, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (139, 148, 158), 2)
    cv2.putText(img, "Connect webcam or PiCamera", (60, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (139, 148, 158), 1)
    _, jpg = cv2.imencode(".jpg", img)
    return jpg.tobytes()
