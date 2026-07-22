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
    """关键词匹配解析。"""
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

    # 没指定互动动作 → 只找猫，不互动（breed 非空就会自动搜索）

    # return 不需要 breed
    if "return" in actions:
        return {"breed": None, "zone": None, "actions": ["return"]}

    # zone 可选——不说区域就自己探索所有猫区
    # breed 非空 = 隐式搜索猫；actions 只放互动动作
    return {"breed": breed, "zone": zone, "actions": actions}


def parse_command(text: str) -> dict:
    """将自然语言文字解析为 {breed, zone, actions}。
    LLM 优先（如果已配置），关键词兜底。"""
    # 尝试 LLM 解析
    try:
        from .llm_parser import parse_with_llm
        result = parse_with_llm(text)
        if result is not None and result.get("breed"):
            # LLM 成功解析了品种，但区域/动作可能漏了 → 用关键词补充
            kw = _parse_keyword(text)
            if result.get("zone") is None and kw.get("zone"):
                result["zone"] = kw["zone"]
            # 合并动作（去重）
            for a in kw.get("actions", []):
                if a not in result["actions"]:
                    result["actions"].append(a)
            return result
    except Exception:
        pass  # LLM 不可用，静默退回

    # 退回关键词匹配
    result = _parse_keyword(text)

    if result["breed"] is None and "return" not in result["actions"]:
        raise ValueError(
            f"无法解析指令: \"{text}\"\n"
            f"  已识别: zone={result['zone']}\n"
            f"  缺少猫品种，请重说"
        )

    return result
