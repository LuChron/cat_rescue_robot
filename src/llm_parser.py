"""
LLM 指令解析 — 用大模型理解自然语言，输出结构化指令。
配置: 环境变量 LLM_BASE_URL / LLM_MODEL。
默认走本地 Ollama (qwen2.5:1.5b)。
"""

import json
import os
import urllib.request
import urllib.error

# ---- 配置 ----
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:1.5b")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")

SYSTEM_PROMPT = """\
You are a command parser for a voice-controlled cat rescue robot.
Extract the user's intent and output ONLY a JSON object.

## Output fields (set null for unused):
{"breed":null,"zone":null,"actions":[],"distance_cm":null,"turn_deg":null,"manual_key":null,"manual_action":null}

## Command types (pick ONE per request):

### MANUAL DRIVE — direct car control (set manual_key + manual_action="down"):
w=forward, s=backward, a=turn_left, d=turn_right, x=stop, 1=slow, 2=medium, 3=fast
"向前"→w, "后退"→s, "左转"→a, "右转"→d, "停"→x, "加速"→3, "减速"→1

### DISTANCE MOVE (set distance_cm + actions):
"向前200cm"→distance_cm=200,actions=["forward"]
"后退50"→distance_cm=50,actions=["backward"]
No number given → use manual drive instead.

### TURN (set turn_deg + actions):
"左转90度"→turn_deg=90,actions=["turn_left"]

### GO TO ZONE (breed=null, zone=zoneId):
"去A区"→breed=null,zone="zoneA",actions=[]
"去茶水间"→breed=null,zone="zoneH",actions=[]

### FIND CAT (breed set, zone optional):
"找波斯猫"→breed="波斯猫",zone=null,actions=[]
"去B区找暹罗猫"→breed="暹罗猫",zone="zoneB",actions=[]

### FIND + INTERACT (breed + cat actions):
"找波斯猫喂食"→breed="波斯猫",zone=null,actions=["feed"]
"去C区找暹罗猫拍照"→breed="暹罗猫",zone="zoneC",actions=["photo"]

## Known values:
breeds: 波斯猫, 暹罗猫, 缅因猫, 孟加拉猫, 布偶猫
zones: zoneA(猫爬架/A区), zoneB(纸箱/B区), zoneC(窗台/C区), zoneD(书架/D区), zoneE(沙发/E区), zoneF(桌底/F区), zoneG(盆栽/G区), zoneH(茶水间/H区)
cat_actions: play, feed, photo, talk, return
manual_keys: w, a, s, d, x, 1, 2, 3

## Examples:
"向前" → {"manual_key":"w","manual_action":"down"}
"停" → {"manual_key":"x","manual_action":"down"}
"向前200cm" → {"distance_cm":200,"actions":["forward"]}
"左转90度" → {"turn_deg":90,"actions":["turn_left"]}
"去茶水间" → {"breed":null,"zone":"zoneH","actions":[]}
"找波斯猫" → {"breed":"波斯猫","zone":null,"actions":[]}
"去B区找暹罗猫喂食拍照" → {"breed":"暹罗猫","zone":"zoneB","actions":["feed","photo"]}
"回去" → {"actions":["return"]}
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
        with urllib.request.urlopen(req, timeout=30) as resp:
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
    VB = {"波斯猫", "暹罗猫", "缅因猫", "孟加拉猫", "布偶猫"}
    VZ = {"zoneA", "zoneB", "zoneC", "zoneD", "zoneE", "zoneF", "zoneG", "zoneH"}
    VA = {"play", "feed", "photo", "talk", "return"}
    VK = {"w", "a", "s", "d", "x", "1", "2", "3", "[", "]", "c"}

    breed = breed if breed in VB else None
    zone = zone if zone in VZ else None
    manual_key = manual_key if manual_key in VK else None
    actions = [a for a in actions if a in VA or a in ("forward", "backward", "turn_left", "turn_right")]

    return {
        "breed": breed, "zone": zone, "actions": actions,
        "distance_cm": int(distance_cm) if distance_cm else None,
        "turn_deg": int(turn_deg) if turn_deg else None,
        "manual_key": manual_key, "manual_action": manual_action if manual_key else None,
    }
