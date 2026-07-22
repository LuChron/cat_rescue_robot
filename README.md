# Voice-Guided Cat Rescue Robot — Advance Project

> **Team 12 — 你说的队**
> SWS3009 Advance Proposal | 2026-07-18 ~ 2026-07-27

语音控制的自主猫 rescue 机器人。只需说出猫品种和护理动作，机器人自动探索所有猫区、规划路径、导航搜索、找到后执行投喂/拍照等护理动作。

## Quick start

```bash
conda activate catvision
cd final_project

# 启动 Web 演示界面
python -m src.server
# → 浏览器打开 http://127.0.0.1:8080

# 或命令行文本模式
python -m src.main "find persian and feed it"
# 或语音模式
python -m src.main
```

## 架构：三个解耦模块

```text
模块 1: ASR (asr.py)          模块 2: 指令解析 (parser.py)      模块 3: 导航执行 (controller.py)
─────────────────────         ───────────────────────         ──────────────────────────
输入: 音频文件 (.wav)          输入: 文字串                    输入: {breed, zone?, actions}
输出: 文字串                   输出: {breed, zone?, actions}   输出: 机器人动作 + 护理动作

faster-whisper medium         关键词规则匹配                    A* 路径规划 + 状态机
只做语音→文字                 只做文字→结构化指令                zone 为空 → 自动探索所有猫区
```

## Repository layout

```text
docs/            Design documents and technical plans
src/             Source code
  asr.py           Module 1: speech → text
  parser.py        Module 2: text → structured command
  planner.py       A* path planning + map loading
  controller.py    Module 3: navigation state machine
  action.py        Post-rescue care actions (feed, photo, etc.)
  main.py          CLI glue for all three modules
  server.py        Flask web dashboard
  templates/
    index.html     Demo frontend UI
config/           Map JSON and configuration
```

## Documentation

- [Advance Proposal Technical Plan](docs/advance_proposal_technical_plan.md)
