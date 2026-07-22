"""
LLM 指令解析 — 用大模型理解自然语言，输出结构化指令。
配置: 环境变量 OPENAI_API_KEY / LLM_BASE_URL / LLM_MODEL。
未配置时自动退回关键词匹配。
"""

import json
import os
import urllib.request
import urllib.error

# ---- 配置（环境变量，默认走本地 Ollama） ----

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:1.5b")
# Ollama 不需要 key，但某些兼容端点要；给个占位值
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")

# ---- 领域定义（传给 LLM 作为 schema） ----

SYSTEM_PROMPT = """\
You are a command parser for a voice-controlled cat rescue robot.
Extract the user's intent from their spoken command and output ONLY a JSON object.

## Output format (strict JSON, no extra text):
{"breed": "<breed_name>", "zone": "<zone_id or null>", "actions": ["<action>", ...]}

## Known breeds (pick exact name):
- 波斯猫 (Persian)
- 暹罗猫 (Siamese)
- 缅因猫 (Maine Coon)
- 孟加拉猫 (Bengal)
- 布偶猫 (Ragdoll)

## Known zones (pick exact ID, or null if not specified):
- zoneA (A区, 猫爬架区, cat climbing area)
- zoneB (B区, 纸箱区, cardboard box area)
- zoneC (C区, 窗台区, windowsill area)
- zoneD (D区, 书架顶, bookshelf top)
- zoneE (E区, 沙发区, sofa area)
- zoneF (F区, 桌底区, under table area)
- zoneG (G区, 盆栽区, potted plant area)
- zoneH (H区, 茶水间, pantry, tea room, kitchenette)

## Known actions — only include what the user wants to DO with the cat:
- "play" — play with the cat (toys, laser pointer, feather wand)
- "feed" — give treats / food
- "photo" — take a picture
- "talk" — make soothing sounds / meow
- "return" — go back to base (only if no cat involved)

Finding/searching for the cat is automatic whenever a breed is specified — do NOT list it as an action.

## Examples:
User: "find the persian cat in zone C and feed it"
→ {"breed": "波斯猫", "zone": "zoneC", "actions": ["feed"]}

User: "go to B区 play with the siamese and take a picture"
→ {"breed": "暹罗猫", "zone": "zoneB", "actions": ["play", "photo"]}

User: "找一下缅因猫，陪它玩然后喂零食"
→ {"breed": "缅因猫", "zone": null, "actions": ["play", "feed"]}

User: "去茶水间看看波斯猫，安抚一下拍个照"
→ {"breed": "波斯猫", "zone": "zoneH", "actions": ["talk", "photo"]}

User: "在沙发区有只布偶猫，陪玩喂食拍照一条龙"
→ {"breed": "布偶猫", "zone": "zoneE", "actions": ["play", "feed", "photo"]}

User: "help the maine coon cat" / "find siamese"
→ {"breed": "缅因猫", "zone": null, "actions": []}

User: "回去" / "go back"
→ {"breed": null, "zone": null, "actions": ["return"]}
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
        # 清理可能的 markdown 包裹
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            content = content.rsplit("```", 1)[0]
        result = json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"[LLM] 解析响应失败: {e}\n  内容: {content[:200]}")
        return None

    # 验证必填字段
    if not isinstance(result, dict):
        return None
    if "breed" not in result or "actions" not in result:
        return None

    # 规范化
    breed = result.get("breed")
    zone = result.get("zone")
    actions = result.get("actions", [])

    if not isinstance(actions, list):
        actions = [actions] if actions else []

    # 去除非标准值
    valid_breeds = {"波斯猫", "暹罗猫", "缅因猫", "孟加拉猫", "布偶猫"}
    valid_zones = {"zoneA", "zoneB", "zoneC", "zoneD", "zoneE", "zoneF", "zoneG", "zoneH"}
    valid_actions = {"play", "feed", "photo", "talk", "return"}

    if breed and breed not in valid_breeds:
        breed = None  # LLM 编造了品种名，退回关键词
    if zone and zone not in valid_zones:
        zone = None

    actions = [a for a in actions if a in valid_actions]

    return {"breed": breed, "zone": zone, "actions": actions}
