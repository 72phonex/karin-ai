# Karin AI — Trinity Architecture

> Research project by **Phonex** (72phonex) · MAIT Rohini, Delhi · 2026

A deployable AI agent with persistent per-user episodic memory, a VAD-based emotional state, and autonomous reasoning goals — all running without a GPU.

---

## Architecture (Trinity)

| Module | Status | Description |
|--------|--------|-------------|
| **[M] EpisodicMemoryBank** | ✅ Phase 0 | SQLite-backed, per-user, sentence-transformer indexed |
| **[E] EmotionalState** | ✅ Phase 0 | VAD (valence/arousal/dominance) + 8 derived traits, persists across sessions |
| **[R] AutonomousReasoning** | ✅ Phase 0 | Goal spawning, post-response reflection, quality tracking |
| **LLM Backend** | 🔄 Phase 1 | Mistral 7B via Colab / local GPU (3-line swap in `karin_core.py`) |
| **Fine-tuning** | 📋 Phase 2 | LoRA on accumulated conversation data |

---

## Features

- **Per-user persistent memory** — everything a user says is stored, semantically indexed, recalled on relevance
- **Emotional continuity** — Karin's mood carries across sessions and influences responses
- **Auth + security** — username/password login, 30-day sessions, IP + user-agent logging
- **Owner dashboard** — secret key unlocks view of all users' memories, login logs, stats
- **Clean dark UI** — chat interface, memory panel, goals panel, live mood bars

---

## Quickstart (Local)

```bash
git clone https://github.com/72phonex/karin-ai
cd karin-ai

pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

---

## Deploy to Render (Free, Persistent Disk)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click **Deploy**
5. In Render dashboard → Environment → add:
   - `KARIN_OWNER_KEY` = your secret owner password (keep this private)
6. Share the live URL with friends

> **Persistent disk** is configured in `render.yaml` — all memory survives restarts.

---

## Owner Dashboard

- In the chat UI → sidebar → **Owner View**
- Enter your `KARIN_OWNER_KEY`
- See: all registered users, their memory episode counts, login history with IPs

---

## Run Tests

```bash
python -m pytest tests/ -v
```

32 tests covering: embedder, emotional state, memory bank, reasoning loop, context builder.

---

## Upgrading to Phase 1 (LLM)

In `karin_core.py`, `KarinResponseEngine.generate()`, replace the template return with:

```python
# Phase 1 swap — 3 lines
from transformers import pipeline
pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")
return pipe(f"[INST]{context}[/INST]", max_new_tokens=256)[0]["generated_text"]
```

---

## Research Notes

See `docs/research_notes.md` for per-session observations feeding into the eventual paper.

---

## Stack

`Python · Flask · SQLite · sentence-transformers · Render`
