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
    "zone d":     "zoneD",
    "zoned":      "zoneD",
    "d区":        "zoneD",
    "d zone":     "zoneD",
    "zone e":     "zoneE",
    "zonee":      "zoneE",
    "e区":        "zoneE",
    "e zone":     "zoneE",
    "zone f":     "zoneF",
    "zonef":      "zoneF",
    "f区":        "zoneF",
    "f zone":     "zoneF",
    "zone g":     "zoneG",
    "zoneg":      "zoneG",
    "g区":        "zoneG",
    "g zone":     "zoneG",
    "zone h":     "zoneH",
    "zoneh":      "zoneH",
    "h区":        "zoneH",
    "h zone":     "zoneH",
    "茶水间":     "zoneH",
    "start":      "start",
    "起点":       "start",
    "junction 1": "junc1",
    "junction 2": "junc2",
}

# ---- 手动驾驶指令 → 映射到 keyboard key ----
MANUAL_KEYWORDS: dict[str, tuple[str, str]] = {
    # (action, key)
    "向前":     ("down", "w"),  "前进": ("down", "w"),  "直走": ("down", "w"),
    "后退":     ("down", "s"),  "倒车": ("down", "s"),
    "左转":     ("down", "a"),
    "右转":     ("down", "d"),
    "停":       ("down", "x"),  "停止": ("down", "x"),  "停下": ("down", "x"),
    "加速":     ("down", "3"),  "快点": ("down", "3"),  "快一点": ("down", "3"),
    "减速":     ("down", "1"),  "慢点": ("down", "1"),  "慢一点": ("down", "1"),
    "中速":     ("down", "2"),  "正常": ("down", "2"),
}

ACTION_KEYWORDS: dict[str, List[str]] = {
    # 陪玩
    "play":      ["play"],
    "玩":        ["play"],
    "逗":        ["play"],
    # 投喂
    "feed":      ["feed"],
    "喂":        ["feed"],
    "投喂":      ["feed"],
    "零食":      ["feed"],
    "吃":        ["feed"],
    # 拍照
    "photo":     ["photo"],
    "拍照":      ["photo"],
    "拍":        ["photo"],
    # 安抚
    "talk":      ["talk"],
    "安抚":      ["talk"],
    "声音":      ["talk"],
    "喵":        ["talk"],
    # 组合
    "照顾":      ["play", "feed", "talk"],
    "互动":      ["play", "talk", "photo"],
    "一条龙":    ["play", "feed", "photo", "talk"],
    # 返回
    "return":    ["return"],
    "回去":      ["return"],
    "回来":      ["return"],
}


def _parse_keyword(text: str) -> dict:
    """关键词匹配解析。返回字段: breed, zone, actions, distance_cm, turn_deg, manual_key, manual_action。"""
    import re
    text_lower = text.lower()

    # ---- 移动距离：向前/前进/后退 + 数字 + cm/米（数字优先于手动） ----
    distance_cm = None
    move_dir = None
    m = re.search(r'(向前|前进|往前|forward|后退|往后|backward)\D*(\d+)\s*(cm|厘米|米|m)?', text_lower)
    if m:
        dir_word, num_str, unit = m.group(1), m.group(2), m.group(3) or "cm"
        num = int(num_str) if num_str else 0
        if unit in ("米", "m"):
            num *= 100
        distance_cm = num if num > 0 else 30  # 默认 30cm
        move_dir = "forward" if dir_word in ("向前", "前进", "往前", "forward") else "backward"

    # ---- 转弯：左转/右转 + 数字 + 度 ----
    turn_deg = None
    turn_dir = None
    m = re.search(r'(左转|右转|turn left|turn right|left|right)\s*(\d+)\s*度?', text_lower)
    if m:
        dir_word, num = m.group(1), int(m.group(2))
        turn_deg = num
        turn_dir = "left" if dir_word in ("左转", "turn left", "left") else "right"

    # ---- 手动驾驶指令（纯关键词，不带数字） ----
    for keyword, (action, key) in MANUAL_KEYWORDS.items():
        # 只匹配纯指令，不匹配"向前200cm"这种带数字的
        if re.search(re.escape(keyword) + r'(?!\s*\d)', text_lower):
            return {"breed": None, "zone": None, "actions": [],
                    "distance_cm": None, "turn_deg": None,
                    "manual_key": key, "manual_action": action}

    # ---- 匹配品种 ----
    breed = None
    for keyword, name in BREED_KEYWORDS.items():
        if keyword in text_lower:
            breed = name
            break

    # ---- 匹配区域 ----
    zone = None
    for keyword, node_id in ZONE_KEYWORDS.items():
        if keyword in text_lower:
            zone = node_id
            break

    # 去/到/前往 + 区域 → 只导航不找猫
    go_only = bool(re.search(r'(去|到|前往|go to|go\s+)\s*(' + '|'.join(re.escape(k) for k in ZONE_KEYWORDS) + r')', text_lower))

    # ---- 匹配互动动作 ----
    actions: List[str] = []
    for keyword, action_list in ACTION_KEYWORDS.items():
        if keyword in text_lower:
            for a in action_list:
                if a not in actions:
                    actions.append(a)

    # ---- 判定指令类型 ----
    if "return" in actions:
        return {"breed": None, "zone": None, "actions": ["return"],
                "distance_cm": None, "turn_deg": None}

    if distance_cm is not None:
        return {"breed": breed, "zone": zone, "actions": [move_dir],
                "distance_cm": distance_cm, "turn_deg": None}

    if turn_deg is not None:
        return {"breed": breed, "zone": zone, "actions": [f"turn_{turn_dir}"],
                "distance_cm": None, "turn_deg": turn_deg}

    if go_only and zone:
        # "去A区" → 导航到A区，不找猫
        return {"breed": None, "zone": zone, "actions": [],
                "distance_cm": None, "turn_deg": None}

    # 默认：找猫模式（breed/zone/actions 可为空）
    return {"breed": breed, "zone": zone, "actions": actions,
            "distance_cm": None, "turn_deg": None}


def parse_command(text: str) -> dict:
    """将自然语言文字解析为结构化指令。
    LLM 优先（如果已配置），关键词兜底。"""
    kw = _parse_keyword(text)  # 关键词总是先算好

    try:
        from .llm_parser import parse_with_llm
        llm = parse_with_llm(text)
        if llm:
            # 关键词有手动指令 → 优先（LLM 可能猜错"左转"为 turn_deg）
            if kw.get("manual_key"):
                return kw
            # LLM 有手动指令 → 直接用
            if llm.get("manual_key"):
                return llm
            # 合并：LLM 为主，关键词补漏
            for field in ("breed", "zone", "distance_cm", "turn_deg"):
                if not llm.get(field) and kw.get(field):
                    llm[field] = kw[field]
            for a in kw.get("actions", []):
                if a not in llm.get("actions", []):
                    llm["actions"].append(a)
            return llm
    except Exception:
        pass

    # 有效性检查
    valid = (
        kw.get("breed") or
        "return" in kw.get("actions", []) or
        kw.get("distance_cm") or
        kw.get("turn_deg") or
        kw.get("manual_key") or
        (kw.get("zone") and not kw.get("breed"))
    )
    if not valid:
        raise ValueError(
            f"无法解析指令: \"{text}\"\n"
            f"  试试: 去A区 / 向前 / 向前200cm / 左转90度 / 找波斯猫 / 停"
        )

    return kw
