# Voice-Guided Cat Rescue Robot

> **Team 12 — 你说的队** | SWS3009 Advance Project | 2026-07

语音 + 文本控制的自主猫 rescue 机器人。说出指令，机器人自动导航、搜索、互动。

## Quick Start

```bash
# 树莓派（小车端）
cd MotorShield_Precision && python car_control.py

# 笔记本（控制端）
conda activate catvision
cd final_project
python -m src.server
# → 浏览器 http://127.0.0.1:8090
```

树莓派 `car_control.py` 需加一行补丁（支持精确定距）：
```python
# 在 TCP 读取循环中，parts = line.split(maxsplit=1) 之后：
if parts[0] == "step" and len(parts) > 1:
    controller.send_line(parts[1])
    continue
```

## 指令系统

完整中英文指令、机械臂说法和当前模型限制见 [docs/command_reference.md](docs/command_reference.md)。

### 手动驾驶（语音 / 键盘 / 文本均可）

| 说法 | 效果 |
|------|------|
| `向前` / `前进` / `直走` | 持续前进（说 `停` 停止） |
| `后退` / `倒车` | 持续后退 |
| `左转` / `右转` | 持续转弯 |
| `停` / `停止` / `停下` | 停止 |
| `加速` / `快点` / `减速` / `慢点` | 速度 3/2/1 |
| `向前200cm` / `后退50厘米` | 定距移动 |
| `左转90度` / `右转45度` | 精确转向 |

键盘控制：W/A/S/D 移动，X 停，M 切换模式，1/2/3 调速，[/] 步长，C 拍照。Q/E 基座旋转，R/F 大臂升降，T/G 小臂伸缩，Space 爪子开合，Z 归位，P 演示。键盘优先级高于自主导航，松手自动恢复。

### 导航与找猫

| 说法 | 效果 |
|------|------|
| `去A区` / `去茶水间` | 导航到指定区域（不找猫） |
| `找波斯猫` | 探索全部猫区搜索波斯猫 |
| `去C区找斯芬克斯猫` | 导航到 C 区搜索斯芬克斯猫 |
| `去B区找布偶猫喂食拍照` | 完整任务：导航→搜索→互动 |
| `找狗` / `找鸡` / `找猫` / `找动物` | 通用搜索（不指定品种） |
| `回去` / `回到起点` | 返回起点 |

视觉模型：YOLO26n 多物种检测（猫/狗/鸟） + YOLO11m 猫兜底 + EfficientNet-B2 五分类（波斯猫/布偶猫/斯芬克斯猫/新加坡猫/兔狲）。通用 "猫"/"狗"/"鸟"/"动物" 也可识别，不挑品种。不支持暹罗猫/缅因猫/孟加拉猫（模型未训练）。

### 互动动作

| 动作 | 触发词 | 效果 |
|------|--------|------|
| 🎣 play | play, 玩, 逗 | 激光笔/逗猫棒互动 |
| 🍖 feed | feed, 喂, 投喂, 零食 | 舵机投食 |
| 📸 photo | photo, 拍照, 拍 | PiCamera 拍照存档 |
| 🔊 talk | talk, 安抚, 声音, 喵 | 播放安抚声音 |

搜猫是自动的（指定品种即搜索），动作只放互动内容。

### 机械臂控制（语音 / 键盘均可）

| 说法 | 键盘 | 效果 |
|------|------|------|
| `左转座` / `右转座` | Q / E | 基座左右旋转 |
| `降臂` / `抬臂` | R / F | 大臂升降 |
| `伸臂` / `收臂` | T / G | 小臂伸缩 |
| `切换夹爪` / `张开` / `抓` | Space | 切换爪子开合状态 |
| `归位` / `回正` | Z | 所有舵机归中 |
| `演示` | P | 舵机演示序列 |

## 操作指南

### 启动

```bash
# 1. 树莓派（小车端）
ssh team12@100.87.177.70
cd MotorShield_Precision
python car_control.py

# 2. 笔记本（控制端）
conda activate catvision
cd final_project
python -m src.server
# → 浏览器 http://127.0.0.1:8090
```

### 三种控制方式

| 方式 | 怎么用 | 适合 |
|------|--------|------|
| **键盘** | 页面聚焦后直接按键 | 精确手动操控、紧急停车 |
| **文本** | 输入框键入 → 回车或点发送 | 复杂指令、调试 |
| **语音** | 点 🎤 → 说话 → 再点停止 | 解放双手 |

### 典型操作

```
"去茶水间找波斯猫喂食拍照"  → 全自动任务
"向前" / "停"               → 手动驾驶
"张开" / "抓"               → 机械爪控制
WASD 开车 + 空格抓猫        → 键盘精细操作
```

### 树莓派补丁

`car_control.py` TCP 读取循环中需加（支持精确定距）：
```python
parts = line.split(maxsplit=1)
if parts[0] == "step" and len(parts) > 1:  # ← 加这 3 行
    controller.send_line(parts[1])
    continue
```

## 系统架构

```
笔记本 (final_project)
  ├── ASR: faster-whisper large-v3-turbo (CUDA) / small (CPU) → 中英文语音→文字
  ├── Parser: 本地 LLM (Ollama qwen2.5:1.5b) + 关键词兜底
  ├── Planner: A* 拓扑图搜索 + 贪心探索
  ├── Controller: 状态机 (IDLE→PLANNING→TURNING→DRIVING→SEARCHING→SUCCESS/FAILED)
  ├── Motor: TCP→树莓派→串口→Arduino 电机控制
  ├── Camera: 树莓派 PiCamera MJPEG 拉流 + YOLO+EfficientNet 猫检测
  └── Server: Flask :8090 + 前端 Dashboard

树莓派 (car_control.py)
  ├── TCP Server :8765 — 接收控制指令
  ├── Serial → Arduino — 电机/传感器
  └── PiCamera MJPEG :5000 — 视频流

Arduino (MotorShield_Precision.ino)
  ├── AFMotor 四轮驱动 + Hall 编码器定距
  ├── MPU6500 IMU 航向
  └── 超声波障碍检测
```

## 文件结构

```
final_project/
├── README.md
├── requirements.txt
├── config/
│   ├── map.json              # 当前使用的地图
│   ├── map_simple.json       # 5 节点简单测试版
│   └── map_full.json         # 14 节点完整版
├── src/
│   ├── asr.py                # 语音→文字 (faster-whisper)
│   ├── parser.py             # 指令解析 (LLM + 关键词)
│   ├── llm_parser.py         # LLM 解析 (Ollama)
│   ├── planner.py            # A* 路径规划
│   ├── controller.py         # 导航状态机
│   ├── motor.py              # 小车电机控制 (TCP→Arduino)
│   ├── camera.py             # 摄像头 MJPEG + YOLO 检测
│   ├── action.py             # 互动动作 (喂食/拍照/玩耍)
│   ├── main.py               # CLI 入口
│   ├── server.py             # Flask Web 服务
│   └── templates/
│       └── index.html        # 前端 Dashboard
└── docs/
    └── advance_proposal_technical_plan.md
```

## 配置

| 环境变量 | 默认值 | 说明 |
|------|------|------|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | LLM 端点 |
| `LLM_MODEL` | `qwen2.5:1.5b` | 模型名 |
| `OPENAI_API_KEY` | `ollama` | API Key（Ollama 不需要） |
| `ASR_MODEL` | CUDA: `large-v3-turbo`；CPU: `small` | Whisper 模型 |
| `ASR_DEVICE` | `auto` | 自动优先 CUDA，也可设为 `cpu` / `cuda` |
| `ASR_COMPUTE_TYPE` | CUDA: `int8_float16`；CPU: `int8` | 推理精度 |
| `VISION_INTERVAL` | `0.25` | 视觉推理最小间隔（秒） |
| `VISION_RESULT_TTL` | `1.5` | 检测结果有效期（秒） |
| `VISION_STABLE_FRAMES` | `2` | 连续稳定帧数阈值 |
| `MAP_NAME` | `map_simple` | 地图名（config/ 下的 json 文件名） |
| `CATVISION_ROOT` | 同级 `cat_vision_pipeline` | 已验证视觉 pipeline 路径 |
| `CAMERA_STREAM_URL` | `http://100.87.177.70:5000/video_feed` | PiCamera MJPEG 地址 |
| `ROBOT_HOST` | `100.87.177.70` | 树莓派控制地址 |
| `ROBOT_CONTROL_PORT` | `8765` | 树莓派 TCP 控制端口 |

地图默认使用 `config/map_simple.json`（5 节点测试版），可通过环境变量切换：

```bash
MAP_NAME=map_full python -m src.server    # 完整版 14 节点
MAP_NAME=map_simple python -m src.server  # 简单版 5 节点（默认）
```

地图文件 `config/map.json` 可自定义节点和边，格式见 `map_simple.json`。

## 依赖

```
faster-whisper  sounddevice  numpy  flask  opencv-python
```

## ASR accuracy evaluation

Do not infer project accuracy from the public Whisper benchmark. Record the
standard samples with the microphone used by the robot, then evaluate them:

```bash
python scripts/record_asr_evaluation.py
python scripts/evaluate_asr.py asr_evaluation/manifest.csv
```

The report includes exact transcript accuracy, final command semantic accuracy,
and mean inference latency.

猫检测需要 `catvision` conda 环境（含 ultralytics, torch, torchvision）。
