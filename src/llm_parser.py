"""
LLM 指令解析 — 用大模型理解自然语言，输出结构化指令。
配置: 环境变量 LLM_BASE_URL / LLM_MODEL。
默认走本地 Ollama (qwen2.5:1.5b)。
"""

import json
import os
import re
import urllib.request
import urllib.error

# ---- 配置 ----
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:1.5b")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "6"))

SYSTEM_PROMPT = """\
Extract one robot command from Chinese or English speech. Output ONLY JSON.

Fields:
- breed: string or null
- zone: string or null
- actions: string array
- distance_cm: integer or null
- turn_deg: integer or null
- manual_key: string or null
- manual_action: "down" or null

Breeds: cat(any cat), 波斯猫(Persian), 布偶猫(Ragdoll), 斯芬克斯猫(Sphynx), 新加坡猫(Singapura), 兔狲(Pallas cat), dog, bird, animal
Zones: zoneA(A区/猫爬架), zoneB(B区/纸箱), zoneC(C区/窗台), zoneD(D区/书架), zoneE(E区/沙发), zoneF(F区/桌底), zoneG(G区/盆栽), zoneH(H区/茶水间)
Actions: play, feed, photo, talk, return, forward, backward, turn_left, turn_right
Manual keys:
w=forward, s=backward, a=left, d=right, x=stop,
q=base left, e=base right, r=lower arm, f=raise arm,
t=extend arm, g=retract arm, space=gripper, z=arm home, p=arm demo

Use distance_cm with forward/backward for bounded movement.
Use turn_deg with turn_left/turn_right for bounded rotation.
Use manual_key only for continuous movement, stop, arm, or gripper commands.

Examples:
"直行十厘米" → {"breed":null,"zone":null,"actions":["forward"],"distance_cm":10,"turn_deg":null,"manual_key":null,"manual_action":null}
"往左转九十度" → {"breed":null,"zone":null,"actions":["turn_left"],"distance_cm":null,"turn_deg":90,"manual_key":null,"manual_action":null}
"raise the robot arm" → {"breed":null,"zone":null,"actions":[],"distance_cm":null,"turn_deg":null,"manual_key":"f","manual_action":"down"}
"把机器人的手臂往上抬" → {"breed":null,"zone":null,"actions":[],"distance_cm":null,"turn_deg":null,"manual_key":"f","manual_action":"down"}
"放下机械臂" → {"breed":null,"zone":null,"actions":[],"distance_cm":null,"turn_deg":null,"manual_key":"r","manual_action":"down"}
"停止" → {"breed":null,"zone":null,"actions":[],"distance_cm":null,"turn_deg":null,"manual_key":"x","manual_action":"down"}
"find Persian in zone C and feed" → {"breed":"波斯猫","zone":"zoneC","actions":["feed"],"distance_cm":null,"turn_deg":null,"manual_key":null,"manual_action":null}
"去茶水间" → {"breed":null,"zone":"zoneH","actions":[],"distance_cm":null,"turn_deg":null,"manual_key":null,"manual_action":null}
"""


def parse_with_llm(text: str) -> dict | None:
    """用 LLM 解析自然语言指令。失败或未配置返回 None。"""
    if not LLM_API_KEY:
        return None

    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
    }).encode("utf-8")

    url = f"{LLM_BASE_URL}/chat/completions"
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    })

    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        print(f"[LLM] 请求失败: {e}")
        return None

    try:
        content = body["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            content = content.rsplit("```", 1)[0]
        result = json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"[LLM] 解析响应失败: {e}")
        return None

    if not isinstance(result, dict):
        return None

    # 规范化所有字段
    breed = result.get("breed")
    zone = result.get("zone")
    actions = result.get("actions", [])
    distance_cm = result.get("distance_cm")
    turn_deg = result.get("turn_deg")
    manual_key = result.get("manual_key")
    manual_action = result.get("manual_action")

    if not isinstance(actions, list):
        actions = [actions] if actions else []

    # 校验
    VB = {
        "cat", "波斯猫", "布偶猫", "斯芬克斯猫", "新加坡猫", "兔狲",
        "dog", "bird", "animal",
    }
    VZ = {"zoneA", "zoneB", "zoneC", "zoneD", "zoneE", "zoneF", "zoneG", "zoneH"}
    VA = {"play", "feed", "photo", "talk", "return"}
    VK = {"w", "a", "s", "d", "x", "1", "2", "3", "[", "]", "c",
          "q", "e", "r", "f", "t", "g", "space", "z", "p"}

    breed = breed if breed in VB else None
    zone = zone if zone in VZ else None
    manual_key = manual_key if manual_key in VK else None
    actions = [a for a in actions if a in VA or a in ("forward", "backward", "turn_left", "turn_right")]

    # Small local models occasionally invert paired arm keys. Explicit words
    # are authoritative because executing the opposite arm direction is unsafe.
    normalized_text = text.casefold()
    manual_overrides = (
        (r"(?:抬起|升起|往上抬|向上抬|raise).*(?:机械臂|手臂|arm)|"
         r"(?:机械臂|手臂|arm).*(?:抬起|升起|往上抬|向上抬|raise)", "f"),
        (r"(?:放下|降低|往下放|向下放|lower).*(?:机械臂|手臂|arm)|"
         r"(?:机械臂|手臂|arm).*(?:放下|降低|往下放|向下放|lower)", "r"),
        (r"(?:伸出|伸展|extend).*(?:机械臂|手臂|arm)|"
         r"(?:机械臂|手臂|arm).*(?:伸出|伸展|extend)", "t"),
        (r"(?:收回|缩回|retract).*(?:机械臂|手臂|arm)|"
         r"(?:机械臂|手臂|arm).*(?:收回|缩回|retract)", "g"),
    )
    for pattern, key in manual_overrides:
        if re.search(pattern, normalized_text):
            manual_key = key
            break

    def bounded_int(value, minimum, maximum):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if minimum <= parsed <= maximum else None

    distance_cm = bounded_int(distance_cm, 1, 1000)
    turn_deg = bounded_int(turn_deg, 1, 360)
    if distance_cm and not any(a in actions for a in ("forward", "backward")):
        distance_cm = None
    if turn_deg and not any(a in actions for a in ("turn_left", "turn_right")):
        turn_deg = None
    manual_action = "down" if manual_key else None

    parsed = {
        "breed": breed, "zone": zone, "actions": actions,
        "distance_cm": distance_cm,
        "turn_deg": turn_deg,
        "manual_key": manual_key,
        "manual_action": manual_action,
    }
    return parsed if any((
        breed, zone, actions, distance_cm, turn_deg, manual_key,
    )) else None
