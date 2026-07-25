import pytest
import json
import threading
import time
from pathlib import Path

pytest.importorskip("cv2")

from src import camera
from src.controller import NavigationController
from src.motor import MotorController


def test_full_map_has_one_simulated_animal_per_search_zone():
    map_path = Path(__file__).parents[1] / "config" / "map_full.json"
    map_data = json.loads(map_path.read_text(encoding="utf-8"))
    zone_ids = {
        node_id
        for node_id, node in map_data["nodes"].items()
        if node.get("type") == "cat_zone"
    }

    assert set(map_data["simulated_animals"]) == zone_ids
    assert all(
        animal.get("species") and animal.get("breed") and animal.get("label")
        for animal in map_data["simulated_animals"].values()
    )


def test_full_map_places_chicken_southwest_of_start():
    map_path = Path(__file__).parents[1] / "config" / "map_full.json"
    map_data = json.loads(map_path.read_text(encoding="utf-8"))
    start = map_data["nodes"]["start"]
    chicken_point = map_data["nodes"]["zoneG"]
    edge = next(
        item
        for item in map_data["edges"]
        if {item["from"], item["to"]} == {"start", "zoneG"}
    )

    assert chicken_point["x"] - start["x"] == -40
    assert chicken_point["y"] - start["y"] == 40
    assert edge["dist_cm"] == pytest.approx(56.57, abs=0.01)
    assert map_data["simulated_animals"]["zoneG"] == {
        "species": "bird",
        "breed": "bird",
        "label": "Chicken",
    }


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


def test_simulation_uses_configured_animal_instead_of_camera(monkeypatch):
    map_path = Path(__file__).parents[1] / "config" / "map_full.json"
    controller = NavigationController(str(map_path))
    motor = MotorController(host="unused", port=0)
    monkeypatch.setattr("src.controller.get_motor", lambda: motor)
    monkeypatch.setattr("src.controller.SIMULATION_SCAN_DELAY", 0)
    controller._command = {"breed": "波斯猫", "actions": []}
    controller._waypoints = [{"id": "zoneA", "x": 0, "y": 0}]
    controller._current_wp_idx = 1

    controller._do_searching()

    state = controller.get_state()
    assert state["nav_state"] == "SUCCESS"
    assert state["cat_breed"] == "波斯猫"
    assert state["cat_zone"] == "zoneA"


def test_requested_zone_miss_expands_to_full_map_search():
    map_path = Path(__file__).parents[1] / "config" / "map_full.json"
    controller = NavigationController(str(map_path))
    controller._command = {"breed": "波斯猫", "zone": "zoneE", "actions": []}
    controller._waypoints = [{"id": "zoneE", "x": 0, "y": 200}]
    controller._current_wp_idx = 1

    controller._continue_after_miss("zoneE")

    state = controller.get_state()
    assert controller._exploring
    assert state["nav_state"] == "TURNING"
    assert state["route"][0] == "zoneE"
    assert len(state["route"]) > 1
    assert "zoneE" in state["explored_zones"]


def test_simulation_turn_jumps_to_target_heading(monkeypatch):
    motor = MotorController(host="unused", port=0)
    monkeypatch.setattr("src.motor.SIMULATION_TURN_SECONDS", 0)

    assert motor.simulate_turn_to(225)
    assert motor.get_heading() == 225


def test_simple_map_route_uses_latest_car_heading_convention(monkeypatch):
    class FakeMotor:
        def __init__(self):
            self.heading = 90.0
            self.turns = []

        def is_connected(self):
            return True

        def manual_active(self, grace=0):
            return False

        def get_heading(self):
            return self.heading

        def turn_degrees(self, direction, angle_deg, cancel_event=None):
            angle_deg = round(angle_deg)
            self.turns.append((direction, angle_deg))
            if direction == "left":
                self.heading = (self.heading + angle_deg) % 360
            else:
                self.heading = (self.heading - angle_deg) % 360
            return True

    map_path = Path(__file__).parents[1] / "config" / "map_simple.json"
    controller = NavigationController(str(map_path))
    motor = FakeMotor()
    monkeypatch.setattr("src.controller.get_motor", lambda: motor)
    controller._waypoints = [
        {"id": "start", "x": 20, "y": 50},
        {"id": "junc2", "x": 20, "y": 30},
        {"id": "junc1", "x": 0, "y": 30},
        {"id": "zoneA", "x": 0, "y": 0},
    ]

    for waypoint_index in range(1, 4):
        controller._current_wp_idx = waypoint_index
        controller._do_turning()

    assert motor.turns == [("left", 90), ("right", 90)]
    assert motor.heading == 90


def test_feed_translates_to_latest_pi_servo_protocol(monkeypatch):
    monkeypatch.setattr(
        "src.motor.FEED_LOWER_SEQUENCE",
        (("F30", 0), ("S170", 0), ("G60", 0)),
    )
    monkeypatch.setattr(
        "src.motor.FEED_RETURN_SEQUENCE",
        (("B70", 0), ("F45", 0), ("S150", 0), ("G0", 0)),
    )
    monkeypatch.setattr("src.motor.FEED_RELEASE_SECONDS", 0)

    class FakeSocket:
        def __init__(self):
            self.sent = []

        def sendall(self, data):
            self.sent.append(data.decode("utf-8"))

    motor = MotorController(host="unused", port=0)
    motor._sock = FakeSocket()
    motor.state.connected = True

    assert motor.execute_care_action("feed", timeout=1)
    assert motor._sock.sent == [
        "raw F30\n",
        "raw S170\n",
        "raw G60\n",
        "raw B70\n",
        "raw F45\n",
        "raw S150\n",
        "raw G0\n",
    ]


def test_play_translates_to_gentle_arm_motion_and_returns_home(monkeypatch):
    monkeypatch.setattr(
        "src.motor.PLAY_ARM_SEQUENCE",
        (
            ("B70", 0),
            ("F45", 0),
            ("S150", 0),
            ("G0", 0),
            ("B62", 0),
            ("F42", 0),
            ("S146", 0),
            ("B78", 0),
            ("F48", 0),
            ("S154", 0),
            ("B70", 0),
            ("F45", 0),
            ("S150", 0),
            ("G0", 0),
        ),
    )

    class FakeSocket:
        def __init__(self):
            self.sent = []

        def sendall(self, data):
            self.sent.append(data.decode("utf-8"))

    motor = MotorController(host="unused", port=0)
    motor._sock = FakeSocket()
    motor.state.connected = True

    assert motor.execute_care_action("play", timeout=1)
    assert motor._sock.sent[:4] == [
        "raw B70\n", "raw F45\n", "raw S150\n", "raw G0\n",
    ]
    assert motor._sock.sent[4:10] == [
        "raw B62\n", "raw F42\n", "raw S146\n",
        "raw B78\n", "raw F48\n", "raw S154\n",
    ]
    assert motor._sock.sent[-4:] == [
        "raw B70\n", "raw F45\n", "raw S150\n", "raw G0\n",
    ]


def test_hardware_search_stops_after_one_circle_and_continues(monkeypatch):
    class FakeMotor:
        def __init__(self):
            self.turns = []
            self.stop_count = 0

        def is_connected(self):
            return True

        def turn_degrees(self, direction, angle_deg, cancel_event=None):
            self.turns.append((direction, angle_deg))
            return True

        def stop(self):
            self.stop_count += 1

    controller = NavigationController()
    controller._command = {"breed": "波斯猫", "zone": None, "actions": []}
    controller._waypoints = [
        {"id": "zoneA", "x": 0, "y": 0},
        {"id": "zoneB", "x": 50, "y": 0},
    ]
    controller._current_wp_idx = 1
    controller._exploring = True
    motor = FakeMotor()

    monkeypatch.setattr("src.controller.get_motor", lambda: motor)
    monkeypatch.setattr("src.controller.get_latest_detection", lambda: None)
    monkeypatch.setattr("src.controller.reset_detection", lambda: None)
    monkeypatch.setattr("src.controller.set_search_active", lambda _active: None)
    monkeypatch.setattr("src.controller.SEARCH_TURN_STEP_DEGREES", 30)
    monkeypatch.setattr("src.controller.SEARCH_STEP_PAUSE_SECONDS", 0)

    controller._do_searching()

    assert motor.turns == [("right", 30)] * 12
    assert sum(angle for _, angle in motor.turns) == 360
    assert motor.stop_count == 1
    assert controller.get_state()["nav_state"] == "TURNING"
    assert "zoneA" in controller.get_state()["explored_zones"]


def test_target_centering_uses_bounded_angle_commands(monkeypatch):
    class FakeMotor:
        def __init__(self):
            self.turns = []

        def is_connected(self):
            return True

        def turn_degrees(self, direction, angle_deg, cancel_event=None):
            self.turns.append((direction, angle_deg))
            return True

    controller = NavigationController()
    motor = FakeMotor()
    centered_detection = {
        "stable": True,
        "species": "cat",
        "breed": "波斯猫",
        "classification_confidence": 0.9,
        "frame_width": 640,
        "box": [280, 100, 360, 300],
    }
    monkeypatch.setattr("src.controller.get_motor", lambda: motor)
    monkeypatch.setattr(
        "src.controller.get_latest_detection",
        lambda: centered_detection,
    )
    monkeypatch.setattr("src.controller.CENTERING_PAUSE_SECONDS", 0)

    controller._center_on_target(
        {
            **centered_detection,
            "box": [500, 100, 600, 300],
        },
        "波斯猫",
    )

    assert motor.turns == [("right", 20)]


def test_visual_beacon_failure_continues_to_the_waypoint(monkeypatch):
    class FakeMotor:
        def __init__(self):
            self.heading = 90.0
            self.turns = []

        def is_connected(self):
            return True

        def manual_active(self, grace=1.5):
            return False

        def get_heading(self):
            return self.heading

        def turn_degrees(self, direction, angle_deg, cancel_event=None):
            self.turns.append((direction, angle_deg))
            self.heading = 0.0
            return True

    controller = NavigationController()
    motor = FakeMotor()
    controller._waypoints = [
        {"id": "start", "x": 0, "y": 0},
        {"id": "junc2", "x": 10, "y": 0},
    ]
    controller._current_wp_idx = 1
    controller._state.update(mode="running", nav_state="TURNING")

    monkeypatch.setattr("src.controller.get_motor", lambda: motor)
    monkeypatch.setattr("src.controller.set_beacon_active", lambda _active: None)
    monkeypatch.setattr(
        controller,
        "_align_to_red_beacon",
        lambda _motor: False,
    )

    controller._do_turning()

    state = controller.get_state()
    assert state["mode"] == "running"
    assert state["nav_state"] == "DRIVING"
    assert any(
        "continuing to the waypoint" in line
        for line in state["log"]
    )


def test_hardware_failure_during_beacon_correction_still_stops(monkeypatch):
    class FakeMotor:
        def __init__(self):
            self.heading = 90.0
            self.stop_count = 0

        def is_connected(self):
            return True

        def manual_active(self, grace=1.5):
            return False

        def get_heading(self):
            return self.heading

        def turn_degrees(self, direction, angle_deg, cancel_event=None):
            self.heading = 0.0
            return True

        def stop(self):
            self.stop_count += 1

    controller = NavigationController()
    motor = FakeMotor()
    controller._waypoints = [
        {"id": "start", "x": 0, "y": 0},
        {"id": "junc2", "x": 10, "y": 0},
    ]
    controller._current_wp_idx = 1
    controller._state.update(mode="running", nav_state="TURNING")

    monkeypatch.setattr("src.controller.get_motor", lambda: motor)
    monkeypatch.setattr("src.controller.set_beacon_active", lambda _active: None)
    monkeypatch.setattr(
        controller,
        "_align_to_red_beacon",
        lambda _motor: None,
    )

    controller._do_turning()

    state = controller.get_state()
    assert state["mode"] == "failed"
    assert state["nav_state"] == "IDLE"
    assert motor.stop_count == 1


def test_mission_pauses_at_preselected_junction_and_resumes(monkeypatch):
    class FakeMotor:
        def __init__(self):
            self.stop_count = 0

        def stop(self):
            self.stop_count += 1

    controller = NavigationController()
    motor = FakeMotor()
    monkeypatch.setattr("src.controller.get_motor", lambda: motor)
    monkeypatch.setattr(
        controller, "_ensure_state_machine_worker", lambda: None
    )
    controller._command = {
        "breed": None,
        "zone": "zoneA",
        "actions": [],
        "pause_node": "junc1",
    }
    controller._pause_node = "junc1"
    controller._waypoints = [
        {"id": "start", "x": 20, "y": 50},
        {"id": "junc2", "x": 20, "y": 30},
        {"id": "junc1", "x": 0, "y": 30},
        {"id": "zoneA", "x": 0, "y": 0},
    ]
    # junc1 has just been reached; index points to the next waypoint.
    controller._current_wp_idx = 3
    controller._state.update(
        mode="running",
        nav_state="ARRIVED",
        current_node="junc1",
        pause_node="junc1",
    )

    controller._do_arrived()

    paused = controller.get_state()
    assert paused["mode"] == "paused"
    assert paused["nav_state"] == "PAUSED"
    assert paused["current_node"] == "junc1"
    assert paused["pause_node"] == "junc1"
    assert motor.stop_count == 1

    ok, error = controller.continue_mission()

    assert ok
    assert error == ""
    resumed = controller.get_state()
    assert resumed["mode"] == "running"
    assert resumed["nav_state"] == "TURNING"
    assert resumed["pause_node"] is None
    assert controller._current_wp_idx == 3


def test_continue_is_queued_until_junction_pause_state_is_registered(monkeypatch):
    class FakeMotor:
        def __init__(self):
            self.stop_count = 0

        def stop(self):
            self.stop_count += 1

    controller = NavigationController()
    motor = FakeMotor()
    monkeypatch.setattr("src.controller.get_motor", lambda: motor)
    monkeypatch.setattr(
        controller, "_ensure_state_machine_worker", lambda: None
    )
    controller._command = {
        "breed": "新加坡猫",
        "zone": "zoneD",
        "actions": ["feed"],
        "pause_node": "junc2",
    }
    controller._pause_node = "junc2"
    controller._waypoints = [
        {"id": "start", "x": 40, "y": 80},
        {"id": "junc2", "x": 40, "y": 40},
        {"id": "zoneD", "x": 80, "y": 40},
    ]
    # Physical motion has ended, but the worker has not run _do_arrived yet.
    controller._current_wp_idx = 2
    controller._state.update(
        mode="running",
        nav_state="ARRIVED",
        current_node="junc2",
        pause_node="junc2",
    )

    ok, error = controller.continue_mission()

    assert ok
    assert error == ""
    assert controller._resume_requested

    controller._do_arrived()

    state = controller.get_state()
    assert state["mode"] == "running"
    assert state["nav_state"] == "TURNING"
    assert state["current_node"] == "junc2"
    assert state["pause_node"] is None
    assert controller._pause_node is None
    assert not controller._resume_requested
    assert motor.stop_count == 1


def test_continue_releases_missing_idle_ack_at_scheduled_junction(monkeypatch):
    class FakeMotor:
        def __init__(self):
            self.confirm_count = 0

        def confirm_motion_complete(self):
            self.confirm_count += 1
            return True

    controller = NavigationController()
    motor = FakeMotor()
    monkeypatch.setattr("src.controller.get_motor", lambda: motor)
    monkeypatch.setattr(
        controller, "_ensure_state_machine_worker", lambda: None
    )
    controller._pause_node = "junc2"
    controller._waypoints = [
        {"id": "start", "x": 40, "y": 80},
        {"id": "junc2", "x": 40, "y": 40},
        {"id": "zoneD", "x": 80, "y": 40},
    ]
    controller._current_wp_idx = 1
    controller._state.update(
        mode="running",
        nav_state="DRIVING",
        current_node="start",
        pause_node="junc2",
    )

    ok, error = controller.continue_mission()

    assert ok
    assert error == ""
    assert controller._resume_requested
    assert motor.confirm_count == 1


def test_operator_can_confirm_motion_without_pi_idle_status(monkeypatch):
    monkeypatch.setattr("src.motor.MOTION_SETTLE_SECONDS", 0)

    class FakeSocket:
        def __init__(self):
            self.sent = threading.Event()

        def sendall(self, _data):
            self.sent.set()

    motor = MotorController(host="unused", port=0)
    motor._sock = FakeSocket()
    motor.state.connected = True
    result = {}
    worker = threading.Thread(
        target=lambda: result.update(ok=motor.forward(20))
    )

    worker.start()
    assert motor._sock.sent.wait(0.5)
    assert motor.confirm_motion_complete()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert result["ok"] is True


def test_compound_navigation_leaves_start_before_pausing(monkeypatch):
    map_path = Path(__file__).parents[1] / "config" / "map_simple.json"
    controller = NavigationController(str(map_path))
    motor = MotorController(host="unused", port=0)
    monkeypatch.setattr("src.controller.get_motor", lambda: motor)
    monkeypatch.setattr("src.motor.SIMULATION_TURN_SECONDS", 0)
    monkeypatch.setattr("src.motor.SIMULATION_MIN_DRIVE_SECONDS", 0)
    monkeypatch.setattr("src.motor.SIMULATION_DRIVE_CM_PER_SECOND", 100000)
    monkeypatch.setattr("src.controller.SIMULATION_NODE_DWELL_SECONDS", 0)
    command = {
        "manual_key": None,
        "manual_action": None,
        "control_action": None,
        "pause_node": "junc1",
        "breed": None,
        "zone": "zoneA",
        "actions": [],
        "distance_cm": None,
        "turn_deg": None,
    }
    worker = threading.Thread(
        target=controller.execute_mission,
        args=(command, "Go to point A and stop at junction 1."),
        daemon=True,
    )

    worker.start()
    deadline = time.time() + 1
    while time.time() < deadline:
        if controller.get_state()["nav_state"] == "PAUSED":
            break
        time.sleep(0.01)

    state = controller.get_state()
    assert state["route"] == ["start", "junc2", "junc1", "zoneA"]
    assert state["current_node"] == "junc1"
    assert state["nav_state"] == "PAUSED"
    assert state["pause_node"] == "junc1"
    controller.cancel_mission("test cleanup")
    worker.join(timeout=1)


def test_return_restores_home_heading_for_the_next_mission(monkeypatch):
    class FakeMotor:
        def __init__(self):
            self.heading = 270.0
            self.turns = []

        def get_heading(self):
            return self.heading

        def turn_degrees(self, direction, angle_deg, cancel_event=None):
            angle_deg = round(angle_deg)
            self.turns.append((direction, angle_deg))
            if direction == "left":
                self.heading = (self.heading + angle_deg) % 360
            else:
                self.heading = (self.heading - angle_deg) % 360
            return True

    controller = NavigationController()
    motor = FakeMotor()
    monkeypatch.setattr("src.controller.get_motor", lambda: motor)
    controller._waypoints = [
        {"id": "junc2", "x": 20, "y": 30},
        {"id": "start", "x": 20, "y": 50},
    ]
    controller._current_wp_idx = 2
    controller._state.update(nav_state="ARRIVED", mode="running")

    controller._do_arrived_return()

    state = controller.get_state()
    assert motor.turns == [("left", 180)]
    assert motor.heading == 90
    assert state["current_node"] == "start"
    assert state["nav_state"] == "IDLE"
    assert state["mode"] == "idle"


def test_final_return_edge_backs_into_start_without_home_uturn(monkeypatch):
    class FakeMotor:
        def __init__(self):
            self.heading = 0.0
            self.turns = []
            self.drives = []

        def is_connected(self):
            return True

        def manual_active(self, grace=0):
            return False

        def get_heading(self):
            return self.heading

        def turn_degrees(self, direction, angle_deg, cancel_event=None):
            angle_deg = round(angle_deg)
            self.turns.append((direction, angle_deg))
            if direction == "left":
                self.heading = (self.heading + angle_deg) % 360
            else:
                self.heading = (self.heading - angle_deg) % 360
            return True

        def forward(self, distance_cm, cancel_event=None):
            self.drives.append(("forward", distance_cm))
            return True

        def backward(self, distance_cm, cancel_event=None):
            self.drives.append(("backward", distance_cm))
            return True

        def is_blocked(self):
            return False

    controller = NavigationController()
    motor = FakeMotor()
    monkeypatch.setattr("src.controller.get_motor", lambda: motor)
    controller._returning_home = True
    controller._waypoints = [
        {"id": "junc2", "x": 20, "y": 30},
        {"id": "start", "x": 20, "y": 50},
    ]
    controller._current_wp_idx = 1

    controller._do_turning()
    controller._do_driving()

    assert motor.turns == [("left", 90)]
    assert motor.drives == [("backward", 20)]
    assert motor.heading == 90
    assert controller.get_state()["current_node"] == "start"


@pytest.mark.parametrize(
    ("method", "value", "expected_line", "active_motion"),
    [
        ("forward", 20, "raw w 2 16\n", "forward"),
        ("backward", 35, "raw s 2 35\n", "backward"),
        ("turn_left", 90, "a+90\n", "turn_left"),
        ("turn_right", 45, "d+45\n", "turn_right"),
    ],
)
def test_latest_pi_motion_protocol(
    method,
    value,
    expected_line,
    active_motion,
    monkeypatch,
):
    monkeypatch.setattr("src.motor.MOTION_SETTLE_SECONDS", 0)
    monkeypatch.setattr("src.motor.FORWARD_DISTANCE_SCALE", 0.8)

    class FakeSocket:
        def __init__(self):
            self.sent = []

        def sendall(self, data):
            self.sent.append(data.decode("utf-8"))

    motor = MotorController(host="unused", port=0)
    motor._sock = FakeSocket()
    motor.state.connected = True
    result = {}

    def execute():
        if method == "forward":
            ok = motor.forward(value)
        elif method == "backward":
            ok = motor.backward(value)
        elif method == "turn_left":
            ok = motor.turn_degrees("left", value)
        else:
            ok = motor.turn_degrees("right", value)
        result["ok"] = ok

    worker = threading.Thread(target=execute)
    worker.start()
    deadline = time.time() + 0.5
    while time.time() < deadline and not motor._sock.sent:
        time.sleep(0.01)

    assert motor._sock.sent == [expected_line]
    motor._handle_payload({"type": "status", "motion": active_motion})
    motor._handle_payload({"type": "status", "motion": "idle"})
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert result["ok"] is True
    assert motor._sock.sent == [expected_line]


def test_hardware_motion_waits_for_chassis_to_settle(monkeypatch):
    settle_seconds = 0.05
    monkeypatch.setattr("src.motor.MOTION_SETTLE_SECONDS", settle_seconds)

    class FakeSocket:
        def __init__(self):
            self.sent = threading.Event()

        def sendall(self, _data):
            self.sent.set()

    motor = MotorController(host="unused", port=0)
    motor._sock = FakeSocket()
    motor.state.connected = True
    result = {}
    worker = threading.Thread(
        target=lambda: result.update(ok=motor.forward(20))
    )

    worker.start()
    assert motor._sock.sent.wait(0.5)
    motor._handle_payload({"type": "status", "motion": "forward"})
    settled_at = time.monotonic()
    motor._handle_payload({"type": "status", "motion": "idle"})
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert result["ok"] is True
    assert time.monotonic() - settled_at >= settle_seconds * 0.8


def test_interaction_waits_for_web_confirmation(monkeypatch):
    controller = NavigationController()
    controller._command = {"breed": "波斯猫", "actions": ["feed", "photo"]}
    controller._waypoints = [{"id": "zoneA", "x": 0, "y": 0}]
    controller._current_wp_idx = 1
    controller._state["cat_breed"] = "波斯猫"

    class FakeMotor:
        def __init__(self):
            self.stop_count = 0

        def stop(self):
            self.stop_count += 1

    motor = FakeMotor()
    monkeypatch.setattr("src.controller.get_motor", lambda: motor)
    monkeypatch.setattr(
        controller,
        "_return_to_start",
        lambda: pytest.fail("care completion must not auto-return"),
    )
    monkeypatch.setattr(
        "src.action.run_care_actions",
        lambda _breed, _zone, actions: {action: True for action in actions},
    )
    monkeypatch.setattr("src.controller.SIMULATION_SCAN_DELAY", 0)
    worker = threading.Thread(target=controller._do_success)

    worker.start()
    deadline = time.time() + 1
    while time.time() < deadline and not controller.get_state()["interaction_pending"]:
        time.sleep(0.01)

    pending = controller.get_state()
    assert pending["nav_state"] == "AWAITING_CONFIRMATION"
    assert pending["pending_actions"] == ["feed", "photo"]
    assert controller.resolve_interaction(True)
    worker.join(timeout=3)

    assert not worker.is_alive()
    state = controller.get_state()
    assert state["actions_done"] == ["feed", "photo"]
    assert state["current_node"] == "zoneA"
    assert state["nav_state"] == "IDLE"
    assert state["mode"] == "success"
    assert motor.stop_count == 1


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
