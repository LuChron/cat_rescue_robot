# 语音引导猫 rescue 机器人 — 技术方案

> **Team 12 — 你说的队**
> SWS3009 Advance Proposal | 2026-07-18 ~ 2026-07-27

---

## 1. 项目概述

### 1.1 做什么

一台语音控制的自主机器人，在室内环境中找到走失的猫并进行救助。用户说出猫的品种和目标区域，机器人加载地图、规划路径、自主导航到达目的地，途中遇到障碍物会动态重规划，到达后用视觉模型检测并确认目标猫，最后执行投喂等护理动作。

### 1.2 完整流程

```text
用户语音指令（"去 C 区找波斯猫然后喂它"）
        │
        ▼
   [模块 1] ASR: 语音 → 文字
   faster-whisper small 本地转录
        │ "go to zone C find persian and feed"
        ▼
   [模块 2] 指令解析: 文字 → {breed, zone, actions}
   关键词匹配 → {品种: "波斯猫", 区域: "zoneC", 动作: ["rescue", "feed"]}
        │
        ▼
   [模块 3] A* 路径规划 → 路径点序列 [start → junc1 → zoneC]
        │
        ▼
   导航状态机 → 运动指令 (TCP) → 树莓派 → Arduino → 电机
        │                    │
        │          超声波障碍检测 + Hall传感器反馈
        ▼
   到达目标区域 → 猫视觉识别 (YOLO + EfficientNet)
        │
        ├── 找到目标猫，品种匹配 ──→ 护理动作
        │       │
        │       ├── 舵机投喂食物
        │       ├── 语音播报
        │       ├── 拍照存档
        │       └── 回报任务完成 → 返回起点
        │
        └── 30 秒未找到 ──→ 报告失败 → 返回起点
```

### 1.3 预计工作量

| 模块 | 新增代码 | 难度 |
|------|---------|------|
| 模块1 ASR（语音→文字） | ~60 行 | 低 — faster-whisper 调用 |
| 模块2 指令解析（文字→指令） | ~80 行 | 低 — 关键词规则匹配 |
| 地图 + A* 规划 | ~120 行 | 低 — 标准算法 |
| 导航控制器（状态机） | ~300 行 | 高 — 闭环控制 + 传感器融合 |
| 护理动作控制 | ~100 行 | 低 — 舵机 + 音频 |
| 集成 + Web UI 更新 | ~100 行 | 中 |
| **合计** | **~760 行** | |

---

## 2. 系统架构

```text
┌──────────────────────────────────────────────────────────────┐
│                         笔记本电脑                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ asr.py   │ │parser.py │ │planner.py│ │controller│ │action.py │  │
│  │ 语音→文字│ │文字→指令 │ │ map.json │ │  .py     │ │舵机/喇叭 │  │
│  │ faster-  │ │关键词匹配│ │ + A*     │ │ 状态机   │ │控制      │  │
│  │ whisper  │ │          │ │          │ │          │ │          │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│                                  │               │          │
│                    TCP 通信（复用现有）            │          │
│                    运动指令 + 状态回传            │          │
│                    PiCamera 视频流 ←──────────────┘          │
└─────────────────────────────────┼────────────────────────────┘
                                  │
┌─────────────────────────────────┼────────────────────────────┐
│                        树莓派                                │
│  TCP Server ↔ 串口 ↔ Arduino                                │
│  PiCamera2 → MJPEG 视频流 → 笔记本                           │
│  GPIO → 舵机（食物投放器）                                    │
│  GPIO → 喇叭（语音反馈）                                      │
└─────────────────────────────────┼────────────────────────────┘
                                  │ 串口
┌─────────────────────────────────┼────────────────────────────┐
│                        Arduino                               │
│  电机驱动, Hall 编码器, 超声波传感器                          │
│  指令协议: "w 2 100", "a", "d", "x", "status"               │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 各模块设计

### 3.1 语音指令管线（三个解耦模块）

语音指令的处理拆成三个独立模块，各自只负责一件事，接口清晰：

```text
模块 1: ASR (asr.py)          模块 2: 指令解析 (parser.py)      模块 3: 导航执行 (controller.py)
─────────────────────         ───────────────────────         ──────────────────────────
输入: 音频文件 (.wav)          输入: 文字串                    输入: {breed, zone, actions}
输出: 文字串                   输出: {breed, zone, actions}    输出: 机器人动作 + 护理动作

只做语音→文字                 只做文字→结构化指令               只做指令→执行
不知道猫品种有哪些             不知道文字怎么来的               不知道语音怎么来的
不知道区域叫什么               不知道音频长什么样               不知道文字怎么来的
```

#### 模块 1: ASR — 语音 → 文字

**方案**：本地部署 faster-whisper small 模型，离线运行。API 作为备选。

##### 为什么选本地部署

| 对比维度 | 本地 faster-whisper | Whisper API |
|---------|-------------------|-------------|
| 网络依赖 | 无，完全离线 | 必须联网 |
| 每次成本 | 免费 | ~$0.006/分钟 |
| 延迟 | 2-3 秒（CPU），< 1 秒（GPU） | 1-2 秒（网络往返） |
| 模型大小 | small ~500MB，medium ~1.5GB | 无本地文件 |
| 精度 | small 对清晰语音足够 | 略高 |
| 部署 | `pip install faster-whisper` | `pip install openai` |

演示环境网络不一定稳定，本地跑更可靠。

#### 实现

```python
# asr.py — 模块 1: 只管语音→文字，其他一概不管
import sounddevice as sd
import wave
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu", compute_type="int8")

def record_audio(duration=5, samplerate=16000, filename="command.wav") -> str:
    """录音，返回文件路径"""
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate,
                   channels=1, dtype='int16')
    sd.wait()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    return filename

def speech_to_text(audio_path: str) -> str:
    """语音 → 文字，不关心内容是什么"""
    segments, _ = model.transcribe(audio_path, language="en")
    return " ".join(seg.text for seg in segments)
```

##### 备选：切到 API

```python
# 只换这个函数，其他不变
import openai
def speech_to_text(audio_path):
    with open(audio_path, "rb") as f:
        return openai.Audio.transcribe(model="whisper-1", file=f)["text"]
```

#### 模块 2: 指令解析 — 文字 → {breed, zone, actions}

这个模块**不知道文字是怎么来的**——ASR 来的、键盘打进来的、测试字符串，对它来说都一样。

```python
# parser.py — 模块 2: 只管文字→结构化指令，其他一概不管

BREED_KEYWORDS = {
    "persian":    "波斯猫",  "siamese": "暹罗猫",
    "maine coon": "缅因猫",  "bengal":  "孟加拉猫",  "ragdoll": "布偶猫",
}

ZONE_KEYWORDS = {
    "zone a": "zoneA", "a区": "zoneA", "a zone": "zoneA",
    "zone b": "zoneB", "b区": "zoneB", "b zone": "zoneB",
    "zone c": "zoneC", "c区": "zoneC", "c zone": "zoneC",
    "start":  "start",  "起点":  "start",
    "junction 1": "junc1", "junction 2": "junc2",
}

ACTION_KEYWORDS = {
    "feed":      ["feed"],    "喂":   ["feed"],
    "rescue":    ["rescue"],  "救":   ["rescue"],
    "find":      ["rescue"],  "找":   ["rescue"],  "search": ["rescue"],
    "take care": ["feed", "photo"],  "照顾": ["feed", "photo"],
    "return":    ["return"],  "回去": ["return"],
}

def parse_command(text: str) -> dict:
    """文字 → {breed, zone, actions}，三个维度独立匹配"""
    text_lower = text.lower()

    breed = None
    for keyword, name in BREED_KEYWORDS.items():
        if keyword in text_lower:
            breed = name
            break

    zone = None
    for keyword, node_id in ZONE_KEYWORDS.items():
        if keyword in text_lower:
            zone = node_id
            break

    actions = []
    for keyword, action_list in ACTION_KEYWORDS.items():
        if keyword in text_lower:
            for a in action_list:
                if a not in actions:
                    actions.append(a)

    if not actions:
        actions = ["rescue"]

    if breed is None or zone is None:
        raise ValueError(f"无法解析: {text} (breed={breed}, zone={zone})")

    return {"breed": breed, "zone": zone, "actions": actions}
```

#### 管线组装

三个模块在 `main.py` 里串起来，接口只有函数调用：

```python
# main.py — 模块 1, 2, 3 的胶水层
from asr import record_audio, speech_to_text
from parser import parse_command
from controller import execute_mission  # 模块 3（见 3.4 节）

def run():
    # 模块 1: 语音 → 文字（可替换为文本输入，直接跳到模块 2）
    audio_path = record_audio(duration=5)
    text = speech_to_text(audio_path)
    print(f"转录: {text}")

    # 模块 2: 文字 → 结构化指令
    command = parse_command(text)
    print(f"解析: {command}")

    # 模块 3: 指令 → 导航执行 + 护理动作
    execute_mission(command)
```

#### 兜底方案

Web 控制台文本输入——绕过模块 1，直接把输入文字传给模块 2：

```python
text = web_input_box.get()  # 替代 speech_to_text()
command = parse_command(text)  # 模块 2 不变
execute_mission(command)       # 模块 3 不变
```

### 3.2 地图与路径规划

**核心决策**：手动定义拓扑图，不做 SLAM。

**为什么不做 SLAM**：
- SLAM 需要深度相机或激光雷达做度量级建图，我们只有单目 RGB 摄像头和超声波，硬件不匹配
- 实验室是已知的结构化室内环境，不是未知野外——不需要从零探索
- SLAM 从搭环境到调参至少 1-2 周，我们只有 10 天

**拓扑图是什么**：不关心真实环境的精确尺寸，自己设计一个有节点有边的图就行。每个节点代表猫可能出现的位置（A 区、B 区、走廊、茶水间等），边代表两个节点之间的路径，距离自己设定。本质上就是一个带权重的无向图。

**地图格式** (`config/map.json`)：

```json
{
  "nodes": {
    "start":  {"label": "起点",   "x":   0, "y":   0},
    "junc1":  {"label": "路口1",  "x": 200, "y":   0},
    "junc2":  {"label": "路口2",  "x": 200, "y": 180},
    "zoneA":  {"label": "A区",    "x": 350, "y":   0},
    "zoneB":  {"label": "B区",    "x": 200, "y": 350},
    "zoneC":  {"label": "C区",    "x":   0, "y": 180}
  },
  "edges": [
    {"from": "start", "to": "junc1", "dist_cm": 200},
    {"from": "junc1", "to": "zoneA", "dist_cm": 150},
    {"from": "junc1", "to": "junc2", "dist_cm": 180},
    {"from": "junc2", "to": "zoneB", "dist_cm": 170},
    {"from": "junc2", "to": "zoneC", "dist_cm": 200},
    {"from": "start", "to": "zoneC", "dist_cm": 180}
  ]
}
```

**建图方式**：自己设计节点和边，距离自己定。写成 JSON 就是地图，之后随便改。

**A\* 路径规划**：标准实现，20 个节点以内的图一次搜索不到 1 毫秒。动态重规划就是删掉被堵的边，重跑一次 A\*。

### 3.3 导航控制器

导航控制器是一个有限状态机，把路径点序列转换成闭环电机控制指令，同时持续监控传感器。

```text
                      ┌──────────┐
                      │   IDLE   │ 等待语音指令
                      └────┬─────┘
                           │ 收到指令
                           ▼
                      ┌──────────┐
                      │ PLANNING │ A* 搜索 → 路径点列表
                      └────┬─────┘
                           │
                           ▼
              ┌─────────────────────┐
              │      TURNING        │◄─── 计算目标航向 = atan2(dy, dx)
              │  闭环控制: 持续读    │     从 hall sensor 读当前航向
              │  hall heading，误差  │     误差 = 目标 - 当前
              │  < 15° 时停止转弯   │     持续发 "a"/"d" 直到对准
              └────────┬────────────┘
                       │ 航向对准
                       ▼
              ┌─────────────────────┐
              │      DRIVING        │◄─── 发前进指令，目标距离 = 边长
              │  每 200ms 轮询:      │     累计 hall 编码器距离
              │  超声波障碍检测      │
              └───┬─────────┬───────┘
                  │         │ 超声波 < 15cm
                  │         ▼
                  │  ┌──────────────┐
                  │  │OBSTACLE_WAIT │ 停车 → 等 2 秒 → 重新检测
                  │  └──────┬───────┘
                  │         │ 仍被阻挡（排除行人走过）
                  │         ▼
                  │  ┌──────────────┐
                  │  │ REPLANNING   │ 从图中删除当前边
                  │  │              │ 重跑 A* → 新路径 → 回到 TURNING
                  │  │              │ 无路径 → 报告失败 → 返回起点
                  │  └──────────────┘
                  │
                  │ hall 累计距离 ≥ 目标距离
                  ▼
          ┌──────────────┐
          │   ARRIVED    │── 还有路径点? ──→ TURNING（下一个）
          └──────┬───────┘
                 │ 已是最终节点
                 ▼
          ┌──────────────┐
          │  SEARCHING   │ 原地 360° 旋转 + 持续猫检测
          └──────┬───────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
   ┌─────────┐      ┌─────────┐
   │ SUCCESS │      │ FAILED  │
   │ → 护理  │      │ → 返回  │
   │   动作  │      │   起点  │
   └─────────┘      └─────────┘
```

**实现要点**：

- **闭环转弯**：持续读 hall sensor 航向，和计算出的目标方位角比较，误差小于 15° 才停止。不是"转大概 1 秒"的开环操作。
- **超声波滤波**：每次读 3 次取中值，排除偶发噪声误触发。
- **障碍物确认**：检测到障碍物后先停 2 秒再测一次——如果是人走过，第二次就通过了；只有持续阻挡才判定为 blocked 并触发重规划。
- **位置追踪**：hall 编码器航位推算。每条边距离已知，累计编码器计数换算距离。短边（< 3m）下累积误差可控，到达每个节点后重新校准。

### 3.4 猫视觉识别

YOLO 检测 + EfficientNet-B2 五分类品种识别的 pipeline 已经训练好并验证过。Advance 项目只需要做集成：

- **触发时机**：只在到达目的地后的 SEARCHING 状态（原地旋转）运行，不在行驶过程中持续推理
- **品种匹配**：把检测到的品种和 ASR 提取的目标品种做比对，只有匹配才报成功
- **多帧投票**：现有 pipeline 已有平滑机制，减少单帧误判

### 3.5 护理动作

找到目标猫之后，机器人不只是"发现"就完了——要做一系列护理动作，形成完整的 rescue+care 闭环。

#### 需要的硬件

| 部件 | 用途 | 控制方式 | 成本 |
|------|------|---------|------|
| 微型舵机 (SG90) | 食物投放器开关 | 树莓派 GPIO PWM | ~15 元 |
| 小容器 | 装猫粮/零食 | 3D 打印或纸盒 | 0 |
| 喇叭/蜂鸣器 | 语音播报反馈 | GPIO 或笔记本扬声器 | 0 |
| LED 灯带（可选） | 状态指示灯 | GPIO | ~10 元 |

#### 动作序列

```text
猫检测 + 品种确认
        │
        ▼
   第 1 步：语音播报
   "目标猫已找到，开始护理程序。"
   （笔记本扬声器或树莓派蜂鸣器）
        │
        ▼
   第 2 步：靠近调整
   根据检测框大小判断距离，微调靠近猫
   （bounding box 越大说明越近，太小就往前挪一点）
        │
        ▼
   第 3 步：投放食物
   舵机旋转 → 打开食物容器闸门
   → 猫粮/零食掉落在猫附近
   → 舵机归位关闭
        │
        ▼
   第 4 步：拍照存档
   拍摄高清照片
   保存为 "波斯猫_zoneC_20260720_153000.jpg"
        │
        ▼
   第 5 步：任务报告
   Web 控制台显示：
   "✅ 任务完成 — 在 C 区找到波斯猫，已投喂，照片已保存。"
        │
        ▼
   第 6 步：返回起点
   重新规划路径回到 start，等待下一条指令
```

#### 舵机控制代码（树莓派 GPIO）

```python
import RPi.GPIO as GPIO
import time

class FoodDispenser:
    def __init__(self, servo_pin=18):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(servo_pin, GPIO.OUT)
        self.servo = GPIO.PWM(servo_pin, 50)  # 50Hz
        self.servo.start(0)

    def dispense(self):
        """舵机打开闸门 → 停留 1 秒 → 关闭"""
        self.servo.ChangeDutyCycle(7.5)   # 90° 开
        time.sleep(1.0)
        self.servo.ChangeDutyCycle(2.5)   # 0° 关
        time.sleep(0.5)
        self.servo.ChangeDutyCycle(0)     # 停止信号

    def cleanup(self):
        self.servo.stop()
        GPIO.cleanup()
```

**兜底方案**：如果舵机结构来不及做或者现场出问题，退一步用喇叭播放猫叫/安抚声音 + LED 闪烁 + 拍照，同样能展示"找到猫后有实质性动作"这个设计意图。

---

## 4. 开发计划

| 天数 | 内容 | 产出 |
|------|------|------|
| 1–2 | 地图 + A* | `config/map.json` 实测并编写；`planner.py` 完成，mock 路径和断边重规划测试通过 |
| 3–4 | 导航控制器 | `controller.py` 状态机在真车上跑通路径点跟踪；障碍检测 + 动态重规划联调 |
| 5 | ASR + 指令解析 | `asr.py` 本地 faster-whisper 录音转录；`parser.py` 关键词匹配；Web 控制台文本输入兜底 |
| 6 | 护理动作 | 舵机投食器装车；动作序列调试；猫检测触发拍照 |
| 7 | 系统集成 | 端到端跑通：语音 → 规划 → 导航 → 检测 → 投喂 → 报告 |
| 8–9 | 实验室测试 | 真实环境调试；阈值调整；边界情况覆盖（无路径、找不到猫、API 断网） |
| 10 | 演示准备 | UI 完善、演示脚本、备用视频录制 |

---

## 5. 风险管理

| 风险 | 应对 |
|------|------|
| 实际行驶距离和地图设定有偏差 | 到达判定设宽松阈值（±20cm）；距离不对直接改 JSON 重试 |
| Hall 编码器漂移 | 单边距离不超过 3m；每个节点重新校准；演示距离内可接受 |
| 超声波误触发 | 中值滤波（3 次取中值）+ 2 秒等待二次确认 |
| faster-whisper 转录不准（噪声大） | 切到 Whisper API（只换一个函数）；或 Web 控制台文本输入兜底 |
| 猫不在目标区域 | 30 秒搜索旋转后报告"未找到"，这也是合法的任务结果 |
| 舵机投食器故障 | 退一步用音频播报 + 拍照 + LED 闪烁 |
| 导航控制器没调完 | A* 规划器本身可独立演示（路径可视化）；状态机可以逐步加 |

---

## 6. 关键决策总结

| 决策 | 选择 | 原因 |
|------|------|------|
| 地图 | 手动拓扑图 (JSON)，不做 SLAM | 结构化室内环境不需要度量级 SLAM；拓扑图是标准方法 |
| 路径规划 | A* 图搜索 + 动态删边重规划 | 轻量、可靠、天然支持动态环境 |
| 定位 | Hall 编码器航位推算 | 现有硬件；短距离累计误差可接受 |
| 语音识别 | faster-whisper small 本地部署 + 规则匹配 | 完全离线、免费；指令空间有限用规则匹配即可 |
| 障碍检测 | 超声波 + 中值滤波 | 现有硬件；软件滤波即可处理噪声 |
| 猫视觉 | YOLO + EfficientNet-B2（复用） | 已训练验证；只需集成触发逻辑 |
| 护理动作 | 舵机投食 + 语音播报 + 拍照存档 | 硬件成本极低；展示完整的 rescue+care 闭环 |

---

*SWS3009 Advance Proposal — Group 12 — 你说的队*
