---
title: Karin AI
emoji: 🧠
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Karin AI — Trinity Architecture

> Research project by **Phonex** (72phonex) · MAIT Rohini, Delhi · 2026

A deployable AI agent with persistent per-user episodic memory, a VAD-based emotional state, and autonomous reasoning goals.

## Architecture

| Module | Description |
|--------|-------------|
| **[M] EpisodicMemoryBank** | Per-user, SQLite-backed, sentence-transformer indexed |
| **[E] EmotionalState** | VAD (valence/arousal/dominance) + 8 traits, persists across sessions |
| **[R] AutonomousReasoning** | Goal spawning, post-response reflection, quality tracking |

## Stack
`Python · Flask · SQLite · sentence-transformers · Docker · HuggingFace Spaces`
