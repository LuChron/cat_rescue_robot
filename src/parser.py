"""
模块 2: 指令解析 — 文字 → {breed, zone, actions}
只管文本解析，不关心文字是语音来的还是键盘输入来的。
"""

from typing import List

# ---- 三个维度的关键词库 ----

BREED_KEYWORDS: dict[str, str] = {
    # 正确拼写
    "persian":    "波斯猫",
    "波斯猫":      "波斯猫",
    "siamese":    "暹罗猫",
    "暹罗猫":      "暹罗猫",
    "maine coon": "缅因猫",
    "缅因猫":      "缅因猫",
    "bengal":     "孟加拉猫",
    "孟加拉猫":    "孟加拉猫",
    "ragdoll":    "布偶猫",
    "布偶猫":      "布偶猫",
    # 常见 ASR 听错映射（Whisper 容易把某些品种听成别的词）
    "prison":     "波斯猫",   # Persian → prison
    "version":    "波斯猫",   # Persian → version
    "pursian":    "波斯猫",
    "side me's":  "暹罗猫",   # Siamese
    "siam knees": "暹罗猫",
    "main coon":  "缅因猫",   # Maine Coon
    "mancoon":    "缅因猫",
    "main cool":  "缅因猫",
    "ben gal":    "孟加拉猫", # Bengal
    "bengle":     "孟加拉猫",
    "rag doll":   "布偶猫",   # Ragdoll
}

ZONE_KEYWORDS: dict[str, str] = {
    "zone a":     "zoneA",
    "zonea":      "zoneA",
    "a区":        "zoneA",
    "a zone":     "zoneA",
    "zone b":     "zoneB",
    "zoneb":      "zoneB",
    "b区":        "zoneB",
    "b zone":     "zoneB",
    "zone c":     "zoneC",
    "zonec":      "zoneC",
    "c区":        "zoneC",
    "c zone":     "zoneC",
    "start":      "start",
    "起点":       "start",
    "junction 1": "junc1",
    "junction 2": "junc2",
}

ACTION_KEYWORDS: dict[str, List[str]] = {
    "feed":      ["feed"],
    "喂":        ["feed"],
    "投喂":      ["feed"],
    "rescue":    ["rescue"],
    "救":        ["rescue"],
    "救助":      ["rescue"],
    "find":      ["rescue"],
    "找":        ["rescue"],
    "search":    ["rescue"],
    "take care": ["feed", "photo"],
    "照顾":      ["feed", "photo"],
    "return":    ["return"],
    "回去":      ["return"],
    "回来":      ["return"],
}


def parse_command(text: str) -> dict:
    """将文字解析为 {breed, zone, actions}。"""
    text_lower = text.lower()

    # 匹配品种
    breed = None
    for keyword, name in BREED_KEYWORDS.items():
        if keyword in text_lower:
            breed = name
            break

    # 匹配区域
    zone = None
    for keyword, node_id in ZONE_KEYWORDS.items():
        if keyword in text_lower:
            zone = node_id
            break

    # 匹配动作（可叠加）
    actions: List[str] = []
    for keyword, action_list in ACTION_KEYWORDS.items():
        if keyword in text_lower:
            for a in action_list:
                if a not in actions:
                    actions.append(a)

    if not actions:
        actions = ["rescue"]

    if breed is None or zone is None:
        found = []
        if breed: found.append(f"breed={breed}")
        if zone:  found.append(f"zone={zone}")
        raise ValueError(
            f"无法解析指令: \"{text}\"\n"
            f"  已识别: {', '.join(found) if found else '无'}\n"
            f"  原始转录可能不准，请重说一遍或换种说法"
        )

    return {"breed": breed, "zone": zone, "actions": actions}
