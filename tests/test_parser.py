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
