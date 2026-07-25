import pytest

from src.parser import is_safe_voice_command, parse_command


@pytest.mark.parametrize(
    ("text", "field", "expected"),
    [
        ("向前", "manual_key", "w"),
        ("please stop", "manual_key", "x"),
        ("向前大约 200cm", "distance_cm", 200),
        ("向前走十厘米", "distance_cm", 10),
        ("直行十厘米", "distance_cm", 10),
        ("直走十公分", "distance_cm", 10),
        ("go straight 10 cm", "distance_cm", 10),
        ("向前移动二十五厘米", "distance_cm", 25),
        ("后退两米", "distance_cm", 200),
        ("turn left 90 degrees", "turn_deg", 90),
        ("旋转360度", "turn_deg", 360),
        ("旋转三百六十度", "turn_deg", 360),
        ("转一圈", "turn_deg", 360),
        ("悬正三百六十分。", "turn_deg", 360),
        ("Narcois 360度", "turn_deg", 360),
        ("抬起机械臂", "manual_key", "f"),
        ("把机器人的手臂往上抬", "manual_key", "f"),
        ("please raise arm", "manual_key", "f"),
        ("抓", "manual_key", "space"),
        ("抓猫", "breed", "cat"),
        ("去茶水间", "zone", "zoneH"),
        ("Go to point C.", "zone", "zoneC"),
        ("please go to point H", "zone", "zoneH"),
        ("找狗", "breed", "dog"),
        ("找鸡", "breed", "bird"),
        ("找动物", "breed", "animal"),
    ],
)
def test_command_forms(text, field, expected):
    assert parse_command(text)[field] == expected


def test_combined_mission_keeps_action_order():
    command = parse_command("去B区找斯芬克斯猫喂食拍照")
    assert command["breed"] == "斯芬克斯猫"
    assert command["zone"] == "zoneB"
    assert command["actions"] == ["feed", "photo"]


@pytest.mark.parametrize(
    ("text", "control_action", "pause_node"),
    [
        ("stop at junction 1", "pause_at", "junc1"),
        ("stop at juction1", "pause_at", "junc1"),
        ("在路口一停下", "pause_at", "junc1"),
        ("continue", "continue", None),
        ("继续任务", "continue", None),
    ],
)
def test_junction_pause_controls(text, control_action, pause_node):
    command = parse_command(text, allow_llm=False)
    assert command["control_action"] == control_action
    assert command["pause_node"] == pause_node
    assert command["manual_key"] is None


def test_navigation_can_include_a_preplanned_pause_node():
    command = parse_command(
        "go to point A and stop at junction 1",
        allow_llm=False,
    )

    assert command["zone"] == "zoneA"
    assert command["pause_node"] == "junc1"
    assert command["control_action"] is None


def test_compound_pause_search_and_feed_stays_one_mission():
    command = parse_command(
        "Go to point A, stop at junction one, find the cat and feed it",
        allow_llm=False,
    )

    assert command["zone"] == "zoneA"
    assert command["pause_node"] == "junc1"
    assert command["breed"] == "cat"
    assert command["actions"] == ["feed"]
    assert command["control_action"] is None
    assert command["manual_key"] is None


def test_mixed_language_singapura_pause_and_feed_mission():
    singapura = "\u65b0\u52a0\u5761"
    command = parse_command(
        f"Go to point d, stop at junction two find the {singapura} cat and feed it",
        allow_llm=False,
    )

    assert command["zone"] == "zoneD"
    assert command["pause_node"] == "junc2"
    assert command["breed"] == f"{singapura}\u732b"
    assert command["actions"] == ["feed"]
    assert command["control_action"] is None
    assert command["manual_key"] is None


def test_singpo_cat_interact_phrase_requests_play():
    command = parse_command(
        "Find the singpo cat and interact with it",
        allow_llm=False,
    )

    assert command["breed"] == "\u65b0\u52a0\u5761\u732b"
    assert command["actions"] == ["play"]
    assert command["zone"] is None


@pytest.mark.parametrize(
    "text",
    [
        "go to zone A and stop at junction one",
        "go to zone A and stop at juction one",
        "go to zone A and stop at junction won",
    ],
)
def test_spoken_junction_number_keeps_the_full_mission(text):
    command = parse_command(text, allow_llm=False)

    assert command["zone"] == "zoneA"
    assert command["pause_node"] == "junc1"
    assert command["control_action"] is None
    assert command["manual_key"] is None


def test_search_pause_junction_is_not_mistaken_for_search_destination():
    command = parse_command(
        "find Ragdoll and stop at junction 1",
        allow_llm=False,
    )

    assert command["breed"] == "布偶猫"
    assert command["zone"] is None
    assert command["pause_node"] == "junc1"


def test_unsupported_visual_breed_is_rejected():
    with pytest.raises(ValueError, match="does not support 暹罗猫"):
        parse_command("去A区找暹罗猫")


def test_voice_continuous_movement_requires_a_complete_phrase():
    assert is_safe_voice_command("请向前走一下", parse_command("请向前走一下"))
    assert not is_safe_voice_command("向前走无法解释的尾词", parse_command("向前走无法解释的尾词"))


@pytest.mark.parametrize("text", ["向前走1001厘米", "右转361度"])
def test_rejects_unbounded_motion_commands(text):
    with pytest.raises(ValueError):
        parse_command(text)
