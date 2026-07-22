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
        print("[CAMERA] 猫检测模型已加载")
    except Exception as e:
        print(f"[CAMERA] 猫检测不可用 ({e})，仅显示原始画面")
        _detector_available = False


# ---- 帧缓冲 ----

_frame = None
_frame_lock = threading.Lock()
_cap: cv2.VideoCapture | None = None
_running = False


def _capture_loop(camera_id: int):
    global _frame, _running
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"[CAMERA] 无法打开摄像头 {camera_id}")
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
                frame, _ = predict_frame(
                    frame, _detectors, _classifier, _classes,
                    _transform, _device, 0.25, smoother=smoother,
                )
            except Exception:
                pass  # 检测失败不阻塞流

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


def start_camera(camera_id: int = 0):
    """启动后台摄像头采集。"""
    global _running
    if _running:
        return
    _init_detector()
    _running = True
    t = threading.Thread(target=_capture_loop, args=(camera_id,), daemon=True)
    t.start()


def stop_camera():
    global _running
    _running = False


def get_frame() -> bytes | None:
    """返回最新 JPEG 帧，无数据返回 None。"""
    with _frame_lock:
        return _frame


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
