import pytest
import threading
import time

pytest.importorskip("cv2")

from src import camera
from src.controller import NavigationController
from src.motor import MotorController


def test_detection_requires_confirmation_and_clears():
    camera.reset_detection()
    prediction = {"species": "dog", "breed": "dog", "box": [1, 2, 30, 40]}

    with camera._detection_lock:
        camera._update_detection_locked(prediction)
    assert camera.get_latest_detection()["stable"] is False

    with camera._detection_lock:
        camera._update_detection_locked(prediction)
    assert camera.get_latest_detection()["stable"] is True

    with camera._detection_lock:
        camera._update_detection_locked(None)
    assert camera.get_latest_detection() is None
    assert camera.get_cat_status()["status"] == "no_cat"


def test_camera_stop_waits_for_capture_thread(monkeypatch):
    camera.stop_camera()
    started = threading.Event()

    def capture_loop(_source, stop_event):
        started.set()
        stop_event.wait()

    monkeypatch.setattr(camera, "_init_detector", lambda: None)
    monkeypatch.setattr(camera, "_capture_loop", capture_loop)

    camera.start_camera(source=0)
    assert started.wait(0.5)
    thread = camera._camera_thread
    camera.stop_camera()

    assert thread is not None
    assert not thread.is_alive()
    assert camera._camera_thread is None


def test_controller_rejects_unstable_or_low_confidence_detection():
    controller = NavigationController()
    assert not controller._detection_matches(
        {"stable": False, "species": "dog", "breed": "dog"}, "dog"
    )
    assert not controller._detection_matches(
        {
            "stable": True,
            "species": "cat",
            "breed": "波斯猫",
            "classification_confidence": 0.2,
        },
        "波斯猫",
    )
    assert controller._detection_matches(
        {
            "stable": True,
            "species": "cat",
            "breed": "波斯猫",
            "classification_confidence": 0.8,
        },
        "波斯猫",
    )


def test_return_route_starts_at_current_node():
    controller = NavigationController()
    controller._state["current_node"] = "zoneC"
    controller._command = {
        "breed": None,
        "zone": None,
        "actions": ["return"],
        "distance_cm": None,
        "turn_deg": None,
        "manual_key": None,
    }
    controller._do_planning()
    assert controller.get_state()["route"] == ["zoneC", "junc2", "junc3", "start"]


def test_unexpected_mission_error_does_not_leave_controller_busy(monkeypatch):
    controller = NavigationController()

    def fail():
        raise RuntimeError("test failure")

    monkeypatch.setattr(controller, "_run_state_machine", fail)
    controller.execute_mission(
        {"breed": None, "zone": "zoneA", "actions": []},
        "go to zone A",
    )

    assert controller.get_state()["mode"] == "failed"
    assert controller.get_state()["nav_state"] == "IDLE"
    assert not controller.is_busy()


def test_direct_turn_can_be_cancelled_without_waiting_full_angle(monkeypatch):
    controller = NavigationController()
    motor = MotorController(host="unused", port=0)
    monkeypatch.setattr("src.controller.get_motor", lambda: motor)
    command = {
        "breed": None,
        "zone": None,
        "actions": ["turn_right"],
        "turn_deg": 360,
    }
    thread = threading.Thread(
        target=controller.execute_mission,
        args=(command, "turn 360 degrees"),
    )

    thread.start()
    time.sleep(0.1)
    controller.cancel_mission("test stop")
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert controller.get_state()["nav_state"] == "IDLE"


def test_missing_return_route_does_not_fake_start_position():
    controller = NavigationController()
    controller._state["current_node"] = "zoneA"
    controller._waypoints = [{"id": "zoneA", "x": 0, "y": 0}]
    controller._current_wp_idx = 1
    controller.map_data["edges"] = []

    controller._return_to_start()

    state = controller.get_state()
    assert state["current_node"] == "zoneA"
    assert state["mode"] == "failed"
