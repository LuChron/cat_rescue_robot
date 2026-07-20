# Voice-Guided Cat Rescue Robot — Advance Project

> **Team 12 — 你说的队**
> SWS3009 Advance Proposal | 2026-07-18 ~ 2026-07-27

Autonomous robot that understands voice commands, plans routes on an offline map, navigates with dynamic obstacle replanning, and verifies target cats using deep learning vision.

## Quick overview

```text
Voice command ("Find Persian in Zone C")
  → Whisper API + GPT-4o-mini extracts {breed, zone}
  → A* path planner on predefined topological graph map
  → Navigation state machine drives robot autonomously
  → Ultrasonic obstacle detection → dynamic replanning
  → YOLO + EfficientNet-B2 verifies target cat at destination
```

## Repository layout

```text
docs/            Design documents and technical plans
src/             Source code (ASR pipeline, navigation, integration)
config/          Map JSON and system configuration
```

## Documentation

- [Advance Proposal Technical Plan](docs/advance_proposal_technical_plan.md) — architecture, feasibility analysis, ASR API approach, map design, navigation state machine, and development schedule.

## Current status

| Module | Status |
|--------|--------|
| Motor control (Arduino + Pi) | Done |
| Camera + video stream | Done |
| Cat detection + classification | Done |
| ASR pipeline | Planned |
| Navigation (map + A* + state machine) | Planned |
