"""
摄像头 MJPEG 流 — YOLO 多物种检测 + 猫品种分类。
检测流程: YOLO26n 多物种检测 → YOLO11m 猫检测兜底 → EfficientNet 猫品种分类。
"""

import os
import threading
import time
from pathlib import Path

import cv2

CATVISION_ROOT = os.environ.get(
    "CATVISION_ROOT",
    str(Path(__file__).resolve().parents[2] / "cat_vision_pipeline"),
)
CAMERA_STREAM_URL = os.environ.get(
    "CAMERA_STREAM_URL", "http://100.87.177.70:5000/video_feed"
)
PIPELINE_CONFIDENCE = float(os.environ.get("VISION_CONFIDENCE", "0.25"))
DETECTION_INTERVAL = float(os.environ.get("VISION_INTERVAL", "0.25"))
DETECTION_TTL = float(os.environ.get("VISION_RESULT_TTL", "1.5"))
STABLE_FRAMES = int(os.environ.get("VISION_STABLE_FRAMES", "2"))

# COCO class IDs: 14=bird, 15=cat, 16=dog. COCO 只能识别 bird，不能确认一定是鸡。
ANIMAL_CLASSES = {15: "cat", 16: "dog", 14: "bird"}
BREED_LABELS = {
    "persian": "波斯猫", "ragdoll": "布偶猫", "sphynx": "斯芬克斯猫",
    "singapura": "新加坡猫", "pallas": "兔狲",
}

# ---- 检测模型 ----

_detector_available = False
_detectors = None
_yolo_animal = None
_classifier = None
_classes = None
_transform = None
_device = None


def _init_detector():
    """加载 YOLO 多物种检测 + EfficientNet 猫品种分类。"""
    global _detector_available, _detectors, _yolo_animal
    global _classifier, _classes, _transform, _device
    global _cat_status, _cat_status_msg
    if _detector_available:
        return
    try:
        import sys
        if CATVISION_ROOT not in sys.path:
            sys.path.insert(0, CATVISION_ROOT)
        from catvision.detector import load_detectors
        from catvision.runtime import load_classifier
        import torch
        from torchvision.models import EfficientNet_B2_Weights

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _cat_status_msg = str(_device)

        model_paths = (
            str(Path(CATVISION_ROOT) / "models/detector/yolo26n.pt"),
            str(Path(CATVISION_ROOT) / "models/detector/yolo11m.pt"),
        )
        _detectors = load_detectors(model_paths)
        _yolo_animal = _detectors[0][0]
        print("[CAMERA] YOLO26n -> YOLO11m 猫检测级联已加载")

        # EfficientNet 猫品种分类
        model_path = Path(CATVISION_ROOT) / "models/classifier/best_effnet_b2_cat_breeds.pth"
        class_map = Path(CATVISION_ROOT) / "config/class_to_idx.json"
        _classifier, _classes = load_classifier(model_path, class_map, _device)
        _transform = EfficientNet_B2_Weights.DEFAULT.transforms()
        _detector_available = True
        _cat_status = "ready"
        print(f"[CAMERA] 检测模型全部加载 (device={_device})")
    except Exception as e:
        _cat_status = "error"
        _cat_status_msg = str(e)
        print(f"[CAMERA] 检测不可用 ({e})，仅显示原始画面")
        _detector_available = False


# ---- 帧缓冲 & 检测结果 ----

_frame = None
_frame_lock = threading.Lock()
_latest_detection: dict | None = None  # {breed, classification_confidence, detection_confidence, detector, box}
_latest_detection_at: float = 0.0
_detection_signature: str | None = None
_detection_streak: int = 0
_detection_lock = threading.Lock()
_cat_status: str = "loading"  # loading | ready | detected | no_cat | error | stream_error
_cat_status_msg: str = ""
_running = False
_camera_thread: threading.Thread | None = None
_camera_stop_event = threading.Event()
_camera_lifecycle_lock = threading.Lock()
_search_active = False
_inference_paused = False


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


def _capture_loop(source, stop_event: threading.Event):
    global _frame, _running, _cat_status, _cat_status_msg
    cap = _open_capture(source)
    if cap is None:
        print(f"[CAMERA] 无可用摄像头")
        with _detection_lock:
            _cat_status = "stream_error"
            _cat_status_msg = "No camera available"
            _clear_detection_locked()
        _running = False
        return

    smoother = None
    if _detector_available:
        from catvision.runtime import SmoothedClassifier
        smoother = SmoothedClassifier(window=15)

    last_inference = 0.0
    read_failed_at = None
    while not stop_event.is_set():
        if cap is None:
            stop_event.wait(0.5)
            if stop_event.is_set():
                break
            cap = _open_capture(source)
            continue
        ok, frame = cap.read()
        if not ok:
            if read_failed_at is None:
                read_failed_at = time.monotonic()
            elif time.monotonic() - read_failed_at > 1.0:
                with _detection_lock:
                    _cat_status = "stream_error"
                    _cat_status_msg = "Camera stream interrupted"
                    _clear_detection_locked()
                cap.release()
                cap = _open_capture(source)
                read_failed_at = None
                if cap is None:
                    stop_event.wait(0.5)
            stop_event.wait(0.1)
            continue
        read_failed_at = None

        now = time.monotonic()
        if (
            _detector_available
            and not _inference_paused
            and now - last_inference >= DETECTION_INTERVAL
        ):
            last_inference = now
            try:
                pred = _detect_and_classify(frame, smoother)
                with _detection_lock:
                    _update_detection_locked(pred)
            except Exception as e:
                with _detection_lock:
                    _cat_status = "error"
                    _cat_status_msg = str(e)
                    _clear_detection_locked()

        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with _frame_lock:
            _frame = jpg.tobytes()

        stop_event.wait(0.03)  # ~30 fps cap

    if cap is not None:
        cap.release()


def _detect_and_classify(frame, smoother):
    """优先检测猫；主模型漏检猫时用第二个 YOLO 兜底。"""
    from PIL import Image
    import torch

    primary = _yolo_animal.predict(
        frame, classes=[14, 15, 16], conf=PIPELINE_CONFIDENCE,
        imgsz=640, verbose=False,
    )[0]

    result = primary
    detector_name = _detectors[0][1]
    cat_indices = (
        (primary.boxes.cls == 15).nonzero(as_tuple=False).flatten()
        if primary is not None and len(primary.boxes) else []
    )

    if len(cat_indices):
        best_idx = int(cat_indices[primary.boxes.conf[cat_indices].argmax()].item())
    elif _search_active:
        fallback_model, detector_name = _detectors[1]
        fallback = fallback_model.predict(
            frame, classes=[15], conf=PIPELINE_CONFIDENCE,
            imgsz=640, verbose=False,
        )[0]
        if fallback is not None and len(fallback.boxes):
            result = fallback
            best_idx = int(fallback.boxes.conf.argmax().item())
        elif primary is not None and len(primary.boxes):
            result = primary
            detector_name = _detectors[0][1]
            non_cat = (primary.boxes.cls != 15).nonzero(as_tuple=False).flatten()
            if not len(non_cat):
                return None
            best_idx = int(non_cat[primary.boxes.conf[non_cat].argmax()].item())
        else:
            return None
    elif primary is not None and len(primary.boxes):
        non_cat = (primary.boxes.cls != 15).nonzero(as_tuple=False).flatten()
        if not len(non_cat):
            return None
        best_idx = int(non_cat[primary.boxes.conf[non_cat].argmax()].item())
    else:
        return None

    cls_id = int(result.boxes.cls[best_idx].item())
    det_conf = float(result.boxes.conf[best_idx].item())
    box = result.boxes.xyxy[best_idx].cpu().numpy().astype(int).tolist()
    species = ANIMAL_CLASSES.get(cls_id, "unknown")
    x1, y1, x2, y2 = box

    # 猫才进入 EfficientNet 品种分类。视频框统一由前端绘制，避免双层框残留。
    pred = {"species": species, "detection_confidence": det_conf,
            "detector": Path(detector_name).name, "box": box,
            "frame_width": frame.shape[1], "frame_height": frame.shape[0]}

    if species == "cat":
        crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
        if crop.size > 0:
            try:
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                tensor = _transform(pil).unsqueeze(0).to(_device)
                with torch.no_grad():
                    probs = torch.softmax(_classifier(tensor), dim=1)[0]
                if smoother is not None:
                    probs = smoother.update(probs)
                idx = int(probs.argmax().item())
                cls_conf = float(probs[idx].item())
                raw_breed = _classes[idx]
                breed = BREED_LABELS.get(raw_breed, raw_breed)
                pred.update({"breed": breed, "raw_breed": raw_breed,
                             "classification_confidence": cls_conf})
            except Exception as e:
                pred["classification_error"] = str(e)
    elif species == "dog":
        pred["breed"] = "dog"
    elif species == "bird":
        pred["breed"] = "bird"
    else:
        pred["breed"] = species

    return pred


def _clear_detection_locked():
    global _latest_detection, _latest_detection_at
    global _detection_signature, _detection_streak
    _latest_detection = None
    _latest_detection_at = 0.0
    _detection_signature = None
    _detection_streak = 0


def _update_detection_locked(pred: dict | None):
    global _latest_detection, _latest_detection_at
    global _detection_signature, _detection_streak
    global _cat_status, _cat_status_msg

    if not pred:
        _clear_detection_locked()
        _cat_status = "no_cat"
        _cat_status_msg = "No animal detected"
        return

    signature = str(pred.get("breed") or pred.get("species"))
    _detection_streak = _detection_streak + 1 if signature == _detection_signature else 1
    _detection_signature = signature
    pred = pred.copy()
    pred["stable_count"] = _detection_streak
    pred["stable"] = _detection_streak >= STABLE_FRAMES
    pred["observed_at"] = time.time()
    _latest_detection = pred
    _latest_detection_at = time.monotonic()
    _cat_status = "detected"
    if pred.get("species") == "cat" and pred.get("breed"):
        _cat_status_msg = f"{pred['breed']} cls={pred.get('classification_confidence', 0):.2f}"
    else:
        _cat_status_msg = (
            f"{pred.get('species', 'animal')} "
            f"det={pred.get('detection_confidence', 0):.2f}"
        )


def _detection_is_fresh_locked() -> bool:
    return (
        _latest_detection is not None
        and time.monotonic() - _latest_detection_at <= DETECTION_TTL
    )


def _normalize_prediction(pred: dict | None, frame) -> dict | None:
    """兼容旧接口。"""
    if not pred:
        return None
    pred = pred.copy()
    pred["frame_width"] = frame.shape[1]
    pred["frame_height"] = frame.shape[0]
    return pred


def start_camera(source=None):
    """启动后台摄像头采集。

    source 可以是：
    - None：自动模式（先试 Pi 流，失败则本地摄像头）
    - str：URL（如 "http://100.87.177.70:5000/video_feed"）
    - int：本地摄像头索引（如 0）
    """
    global _running, _camera_thread, _camera_stop_event
    with _camera_lifecycle_lock:
        if _camera_thread is not None and _camera_thread.is_alive():
            return

        # 自动模式：优先用树莓派 PiCamera 流
        if source is None:
            source = CAMERA_STREAM_URL

        _init_detector()
        _camera_stop_event = threading.Event()
        _running = True
        _camera_thread = threading.Thread(
            target=_capture_loop,
            args=(source, _camera_stop_event),
            daemon=True,
        )
        _camera_thread.start()


def stop_camera():
    global _running, _camera_thread
    with _camera_lifecycle_lock:
        _running = False
        _camera_stop_event.set()
        thread = _camera_thread
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=2.0)
    with _camera_lifecycle_lock:
        if _camera_thread is thread and (thread is None or not thread.is_alive()):
            _camera_thread = None


def set_search_active(active: bool):
    """搜索任务期间启用经过验证的第二级 YOLO 猫检测。"""
    global _search_active
    _search_active = bool(active)


def set_inference_paused(paused: bool):
    """ASR 转录等 CPU 敏感阶段暂停视觉推理，视频采集仍继续。"""
    global _inference_paused
    _inference_paused = bool(paused)


def get_frame() -> bytes | None:
    """返回最新 JPEG 帧，无数据返回 None。"""
    with _frame_lock:
        return _frame


def get_latest_detection() -> dict | None:
    """返回最新猫检测结果，供导航控制器查询。"""
    with _detection_lock:
        if not _detection_is_fresh_locked():
            return None
        return _latest_detection.copy()


def get_cat_status() -> dict:
    """返回猫检测状态，供前端展示。"""
    with _detection_lock:
        fresh = _detection_is_fresh_locked()
        det = _latest_detection.copy() if fresh else None
        status = _cat_status
        message = _cat_status_msg
        if not fresh and status == "detected":
            status = "no_cat"
            message = "No animal detected"
        return {"status": status, "message": message, "detection": det}


def reset_detection():
    """清空检测结果（每次搜索前调用）。"""
    global _cat_status, _cat_status_msg
    with _detection_lock:
        _clear_detection_locked()
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
