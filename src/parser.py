"""
模块 2: 指令解析 — 文字 → {breed, zone, actions}
只管文本解析，不关心文字是语音来的还是键盘输入来的。
"""

import re
from typing import List

# ---- 三个维度的关键词库 ----

BREED_KEYWORDS: dict[str, str] = {
    # 与当前 EfficientNet 五分类模型保持一致
    "persian":    "波斯猫",
    "波斯猫":      "波斯猫",
    "ragdoll":    "布偶猫",
    "布偶猫":      "布偶猫",
    "sphynx":     "斯芬克斯猫",
    "sphinx":     "斯芬克斯猫",
    "斯芬克斯猫":  "斯芬克斯猫",
    "无毛猫":      "斯芬克斯猫",
    "singapore cat": "新加坡猫",
    "singaporean cat": "新加坡猫",
    "singpo cat":  "新加坡猫",
    "singapo cat": "新加坡猫",
    "singapore":  "新加坡猫",
    "singapura":  "新加坡猫",
    "新加坡猫":    "新加坡猫",
    "pallas":     "兔狲",
    "pallas cat": "兔狲",
    "兔狲":        "兔狲",
    # 其他动物
    "dog":        "dog",
    "狗":          "dog",
    "小狗":        "dog",
    "bird":       "bird",
    "鸟":          "bird",
    "chicken":    "bird",  # COCO 只能确认 bird，不能做鸡的细分类
    "鸡":          "bird",
    "小鸡":        "bird",
    "cat":        "cat",      # 通用猫，不指定品种
    "猫":          "cat",
    "animal":     "animal",
    "动物":        "animal",
    "宠物":        "animal",
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

UNSUPPORTED_BREED_KEYWORDS = {
    "siamese": "暹罗猫",
    "暹罗猫": "暹罗猫",
    "side me's": "暹罗猫",
    "siam knees": "暹罗猫",
    "maine coon": "缅因猫",
    "main coon": "缅因猫",
    "mancoon": "缅因猫",
    "main cool": "缅因猫",
    "缅因猫": "缅因猫",
    "bengal": "孟加拉猫",
    "ben gal": "孟加拉猫",
    "bengle": "孟加拉猫",
    "孟加拉猫": "孟加拉猫",
}

ZONE_KEYWORDS: dict[str, str] = {
    "zone a":     "zoneA",
    "area a":     "zoneA",
    "point a":    "zoneA",
    "zonea":      "zoneA",
    "a区":        "zoneA",
    "a zone":     "zoneA",
    "zone b":     "zoneB",
    "area b":     "zoneB",
    "point b":    "zoneB",
    "zoneb":      "zoneB",
    "b区":        "zoneB",
    "b zone":     "zoneB",
    "zone c":     "zoneC",
    "area c":     "zoneC",
    "point c":    "zoneC",
    "zonec":      "zoneC",
    "c区":        "zoneC",
    "c zone":     "zoneC",
    "zone d":     "zoneD",
    "area d":     "zoneD",
    "point d":    "zoneD",
    "zoned":      "zoneD",
    "d区":        "zoneD",
    "d zone":     "zoneD",
    "zone e":     "zoneE",
    "area e":     "zoneE",
    "point e":    "zoneE",
    "zonee":      "zoneE",
    "e区":        "zoneE",
    "e zone":     "zoneE",
    "zone f":     "zoneF",
    "area f":     "zoneF",
    "point f":    "zoneF",
    "zonef":      "zoneF",
    "f区":        "zoneF",
    "f zone":     "zoneF",
    "zone g":     "zoneG",
    "area g":     "zoneG",
    "point g":    "zoneG",
    "zoneg":      "zoneG",
    "g区":        "zoneG",
    "g zone":     "zoneG",
    "zone h":     "zoneH",
    "area h":     "zoneH",
    "point h":    "zoneH",
    "zoneh":      "zoneH",
    "h区":        "zoneH",
    "h zone":     "zoneH",
    "茶水间":     "zoneH",
    "tea room":   "zoneH",
    "猫爬架":     "zoneA",  "纸箱": "zoneB",  "窗台": "zoneC",
    "书架":       "zoneD",  "沙发": "zoneE",  "桌底": "zoneF",
    "盆栽":       "zoneG",
    "cat tree":   "zoneA",  "cardboard box": "zoneB",
    "windowsill": "zoneC",  "window": "zoneC",
    "bookshelf":  "zoneD",  "sofa": "zoneE", "couch": "zoneE",
    "under the table": "zoneF", "potted plant": "zoneG",
    "start":      "start",
    "起点":       "start",
    "junction 1": "junc1",
    "junction 2": "junc2",
}

# ---- 手动驾驶指令 → 映射到 keyboard key ----
MANUAL_KEYWORDS: dict[str, tuple[str, str]] = {
    # (action, key) — 长关键词在前，避免"左转座"被"左转"抢先
    # 机械臂（长关键词优先）
    "左转座":   ("down", "q"),  "右转座": ("down", "e"),
    "基座左转": ("down", "q"),  "基座右转": ("down", "e"),
    "turn base left": ("down", "q"), "turn base right": ("down", "e"),
    "放下机械臂": ("down", "r"), "降低机械臂": ("down", "r"),
    "抬起机械臂": ("down", "f"), "升起机械臂": ("down", "f"),
    "机械臂往下放": ("down", "r"), "手臂往下放": ("down", "r"),
    "机械臂往上抬": ("down", "f"), "手臂往上抬": ("down", "f"),
    "降臂":     ("down", "r"),  "落臂": ("down", "r"),
    "抬臂":     ("down", "f"),  "升臂": ("down", "f"),
    "lower arm": ("down", "r"), "raise arm": ("down", "f"),
    "伸出机械臂": ("down", "t"), "收回机械臂": ("down", "g"),
    "伸臂":     ("down", "t"),  "收臂": ("down", "g"),
    "extend arm": ("down", "t"), "retract arm": ("down", "g"),
    "切换夹爪": ("down", "space"), "打开夹爪": ("down", "space"),
    "闭合夹爪": ("down", "space"), "松爪": ("down", "space"),
    "张开爪子": ("down", "space"), "打开爪子": ("down", "space"),
    "闭合爪子": ("down", "space"), "抓紧爪子": ("down", "space"),
    "张开": ("down", "space"),
    "open gripper": ("down", "space"), "close gripper": ("down", "space"),
    "机械臂归位": ("down", "z"), "arm home": ("down", "z"),
    "归位":     ("down", "z"),  "回正": ("down", "z"),
    "机械臂演示": ("down", "p"), "arm demo": ("down", "p"),
    "演示":     ("down", "p"),
    # 移动
    "向前":     ("down", "w"),  "前进": ("down", "w"),  "直走": ("down", "w"),
    "直行":     ("down", "w"),
    "后退":     ("down", "s"),  "倒车": ("down", "s"),
    "左转":     ("down", "a"),  "右转": ("down", "d"),
    "停止":     ("down", "x"),  "停下": ("down", "x"),  "停": ("down", "x"),
    "move forward": ("down", "w"), "forward": ("down", "w"),
    "move backward": ("down", "s"), "backward": ("down", "s"),
    "turn left": ("down", "a"), "turn right": ("down", "d"),
    "stop": ("down", "x"),
    # 速度
    "快一点":   ("down", "3"),  "慢一点": ("down", "1"),
    "加速":     ("down", "3"),  "快点": ("down", "3"),
    "减速":     ("down", "1"),  "慢点": ("down", "1"),
    "中速":     ("down", "2"),  "正常": ("down", "2"),
    # 短词
    "抓":       ("down", "space"),
}

EXACT_MANUAL_KEYWORDS = {"抓", "张开", "归位", "回正", "演示"}
MAX_DISTANCE_CM = 1000
MAX_TURN_DEG = 360

ACTION_KEYWORDS: dict[str, List[str]] = {
    # 陪玩
    "interact with it":      ["play"],
    "interact with the cat": ["play"],
    "play with it":          ["play"],
    "play with the cat":     ["play"],
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
    "回到起点":  ["return"],
    "返回起点":  ["return"],
}


def _chinese_number(value: str) -> int:
    """Convert common command-sized Chinese numerals (0-9999) to an integer."""
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    units = {"十": 10, "百": 100, "千": 1000}
    if all(char in digits for char in value):
        return int("".join(str(digits[char]) for char in value))
    total = 0
    current = 0
    for char in value:
        if char in digits:
            current = digits[char]
        elif char in units:
            total += (current or 1) * units[char]
            current = 0
    return total + current


def is_safe_voice_command(text: str, command: dict) -> bool:
    """Reject partially matched continuous-motion voice transcripts."""
    key = command.get("manual_key")
    if key not in {"w", "a", "s", "d"}:
        return True

    normalized = text.lower().strip("，。！？,.!? ")
    normalized = re.sub(
        r"^(?:请|请你|麻烦|机器人|小车|please|could you|can you)\s*",
        "",
        normalized,
    )
    normalized = re.sub(r"\s*(?:一下|一点|吧|please)\s*$", "", normalized)
    safe_forms = {
        "w": {"向前", "向前走", "前进", "往前", "往前走", "直走", "直行",
              "move forward", "forward", "go straight"},
        "s": {"后退", "往后", "往后走", "倒车", "move backward", "backward"},
        "a": {"左转", "turn left"},
        "d": {"右转", "turn right"},
    }
    return normalized in safe_forms[key]


def _parse_keyword(text: str) -> dict:
    """关键词匹配解析，包括导航、手动控制和预设节点暂停。"""
    text_lower = re.sub(r"\s+", " ", text.lower()).strip()
    # Accept mixed Chinese/English breed names produced by bilingual speech,
    # e.g. "新加坡 cat", before the generic "cat" keyword is matched.
    text_lower = re.sub(r"新加坡\s*(?:cat|猫)", "新加坡猫", text_lower)
    junction_number_words = {
        "one": "1", "won": "1",
        "two": "2", "to": "2", "too": "2",
        "three": "3", "tree": "3",
        "four": "4", "for": "4",
        "five": "5",
    }
    text_lower = re.sub(
        r"\bju(?:n)?ction\s*(one|won|two|to|too|three|tree|four|for|five)\b",
        lambda match: f"junction {junction_number_words[match.group(1)]}",
        text_lower,
    )
    text_lower = re.sub(r"\bju(?:n)?ction\s*([1-9])\b", r"junction \1", text_lower)
    text_lower = re.sub(r"([a-h])\s*区", r"\1区", text_lower)
    for wrong, correct in {
        "悬正": "旋转",
        "旋正": "旋转",
        "悬转": "旋转",
        "三百六十": "360",
        "一百八十": "180",
        "九十": "90",
        "四十五": "45",
    }.items():
        text_lower = text_lower.replace(wrong, correct)
    text_lower = re.sub(r"(360|180|90|45)\s*分\b", r"\1度", text_lower)

    # ---- 持久暂停控制 ----
    continue_text = text_lower.strip("，。！？,.!? ")
    if continue_text in {
        "continue", "continue mission", "resume", "resume mission",
        "继续", "继续任务", "继续走", "恢复任务",
    }:
        return {
            "breed": None, "zone": None, "actions": [],
            "distance_cm": None, "turn_deg": None,
            "manual_key": None, "manual_action": None,
            "control_action": "continue", "pause_node": None,
        }

    pause_node = None
    english_pause = re.search(
        r"\b(?:stop|pause)\s+(?:at\s+)?junction\s+([1-9])\b",
        text_lower,
    )
    chinese_pause = re.search(
        r"(?:停在|暂停在|到|在)\s*(?:路口|交叉点)\s*([一二三四五六七八九1-9])"
        r"(?:\s*(?:停下|停止|暂停))?",
        text_lower,
    )
    if english_pause:
        pause_node = f"junc{english_pause.group(1)}"
    elif chinese_pause and re.search(r"停|暂停", text_lower):
        number = chinese_pause.group(1)
        chinese_digits = {
            "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9,
        }
        pause_node = f"junc{chinese_digits.get(number, int(number) if number.isdigit() else 0)}"

    for keyword, name in UNSUPPORTED_BREED_KEYWORDS.items():
        if keyword in text_lower:
            raise ValueError(
                f"The current vision model does not support {name}. "
                "Use Persian, Ragdoll, Sphynx, Singapura, Pallas cat, or any cat."
            )

    # ---- 移动距离：向前/前进/后退 + 数字 + cm/米（数字优先于手动） ----
    distance_cm = None
    move_dir = None
    number_pattern = r"(\d+|[零〇一二两三四五六七八九十百千]+)"
    m = re.search(
        rf'(向前|前进|往前|直行|直走|forward|go straight|后退|往后|backward)'
        rf'(?:走|移动|开|\s|大约|约)*{number_pattern}\s*(cm|厘米|公分|米|m)?',
        text_lower,
    )
    if m:
        dir_word, num_str, unit = m.group(1), m.group(2), m.group(3) or "cm"
        num = int(num_str) if num_str.isdigit() else _chinese_number(num_str)
        if unit in ("米", "m"):
            num *= 100
        distance_cm = num if num > 0 else 30  # 默认 30cm
        if distance_cm > MAX_DISTANCE_CM:
            raise ValueError(
                f"Movement distance must be between 1 and {MAX_DISTANCE_CM} cm."
            )
        move_dir = "forward" if dir_word in (
            "向前", "前进", "往前", "直行", "直走", "forward", "go straight",
        ) else "backward"

    # ---- 转弯：左转/右转 + 数字 + 度 ----
    turn_deg = None
    turn_dir = None
    m = re.search(
        r'(左转|右转|turn left|turn right|left|right)\s*(\d+)\s*(?:度|degrees?|deg)?',
        text_lower,
    )
    if m:
        dir_word, num = m.group(1), int(m.group(2))
        turn_deg = num
        if turn_deg > MAX_TURN_DEG:
            raise ValueError(
                f"Turn angle must be between 1 and {MAX_TURN_DEG} degrees."
            )
        turn_dir = "left" if dir_word in ("左转", "turn left", "left") else "right"
    else:
        # 未指定方向的“旋转/转一圈”统一按右转执行。
        m = re.search(
            r"(?:旋转|rotate(?:\s+around)?)(?:\D*?)(\d+)\s*(?:度|degrees?|deg)?",
            text_lower,
        )
        if m:
            turn_deg = int(m.group(1))
            if turn_deg > MAX_TURN_DEG:
                raise ValueError(
                    f"Turn angle must be between 1 and {MAX_TURN_DEG} degrees."
                )
            turn_dir = "right"
        elif re.search(r"(?:转|旋转)\s*(?:一圈|1圈)", text_lower):
            turn_deg = 360
            turn_dir = "right"
        elif re.search(r"\b360\s*度", text_lower):
            # 360° 不依赖左右方向；即使动词被 ASR 听错，也可安全归一为转一圈。
            turn_deg = 360
            turn_dir = "right"

    # ---- 手动驾驶指令（纯关键词，不带数字） ----
    for keyword, (action, key) in MANUAL_KEYWORDS.items():
        if distance_cm is not None or turn_deg is not None:
            break
        if pause_node and key == "x":
            continue
        if keyword in EXACT_MANUAL_KEYWORDS:
            matched = text_lower.strip("，。！？,.!? ") == keyword
        elif keyword.isascii():
            matched = bool(re.search(rf"\b{re.escape(keyword)}\b", text_lower))
        else:
            matched = keyword in text_lower
        if matched:
            return {"breed": None, "zone": None, "actions": [],
                    "distance_cm": None, "turn_deg": None,
                    "manual_key": key, "manual_action": action,
                    "control_action": None, "pause_node": pause_node}

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
    # A junction named only by the pause clause is not the mission destination.
    if pause_node and zone == pause_node:
        zone = None

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
    _base = {
        "manual_key": None,
        "manual_action": None,
        "control_action": None,
        "pause_node": pause_node,
    }

    # “stop at junction 1”只登记当前任务的预设停车点，不创建新任务。
    if (
        pause_node
        and breed is None
        and not actions
        and distance_cm is None
        and turn_deg is None
        and zone in {None, pause_node}
    ):
        return {
            **_base,
            "breed": None,
            "zone": None,
            "actions": [],
            "distance_cm": None,
            "turn_deg": None,
            "control_action": "pause_at",
        }
    if "return" in actions:
        return {**_base, "breed": None, "zone": None, "actions": ["return"],
                "distance_cm": None, "turn_deg": None}

    if distance_cm is not None:
        return {**_base, "breed": breed, "zone": zone, "actions": [move_dir],
                "distance_cm": distance_cm, "turn_deg": None}

    if turn_deg is not None:
        return {**_base, "breed": breed, "zone": zone, "actions": [f"turn_{turn_dir}"],
                "distance_cm": None, "turn_deg": turn_deg}

    if go_only and zone and breed is None:
        return {**_base, "breed": None, "zone": zone, "actions": [],
                "distance_cm": None, "turn_deg": None}

    return {**_base, "breed": breed, "zone": zone, "actions": actions,
            "distance_cm": None, "turn_deg": None}


def parse_command(text: str, allow_llm: bool = True) -> dict:
    """关键词精确匹配优先；复杂自然语言用 LLM 兜底。"""
    kw = _parse_keyword(text)

    # 关键词已经有明确结果 → 直接返回
    if (
        kw.get("control_action")
        or kw.get("pause_node")
        or kw.get("manual_key")
        or kw.get("distance_cm")
        or kw.get("turn_deg")
        or kw.get("breed")
    ):
        return {**kw, "parser_source": "rules"}
    if "return" in kw.get("actions", []):
        return {**kw, "parser_source": "rules"}
    if kw.get("zone") and not kw.get("breed"):
        # "去A区" → 只有 zone 没 breed，关键词已够
        return {**kw, "parser_source": "rules"}

    # 关键词没招了 → 试 LLM。ASR 候选筛选阶段会显式关闭 LLM，避免错误外语拖慢解析。
    if allow_llm:
        try:
            from .llm_parser import parse_with_llm
            llm = parse_with_llm(text)
            if llm:
                result = {
                    "breed": llm.get("breed") or kw.get("breed"),
                    "zone": llm.get("zone") or kw.get("zone"),
                    "actions": list(dict.fromkeys(llm.get("actions", []) + kw.get("actions", []))),
                    "distance_cm": llm.get("distance_cm") or kw.get("distance_cm"),
                    "turn_deg": llm.get("turn_deg") or kw.get("turn_deg"),
                    "manual_key": llm.get("manual_key"),
                    "manual_action": llm.get("manual_action"),
                    "control_action": kw.get("control_action"),
                    "pause_node": kw.get("pause_node"),
                    "parser_source": "llm",
                }
                # 清洗：有 breed 时去掉 return
                if result["breed"] and "return" in result["actions"]:
                    result["actions"] = [a for a in result["actions"] if a != "return"]
                if any((
                    result["breed"],
                    result["zone"],
                    result["actions"],
                    result["distance_cm"],
                    result["turn_deg"],
                    result["manual_key"],
                )):
                    return result
        except Exception:
            pass

    # 什么都没识别
    raise ValueError(
        f"Unable to parse \"{text}\". Try: forward / stop / go to zone A / "
        "find Persian / forward 200 cm"
    )
