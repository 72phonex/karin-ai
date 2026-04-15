# KARIN — Trinity Architecture
### Research Project by Phonex (72phonex) | MAIT Rohini, Delhi

> *"The first small model with unified Memory + Emotion + Autonomous Reasoning in a single persistent architecture"*

---

## What Makes Karin Novel

Every major AI lab builds **stateless** models. GPT, Claude, Gemini — they all reset.  
Karin doesn't. That's the research claim.

| Feature | GPT-4 / Claude | **Karin** |
|--------|---------------|----------|
| Remembers past sessions | ❌ | ✅ Episodic Memory Bank |
| Emotional state | ❌ | ✅ VAD-based Affective Engine |
| Self-set goals | ❌ | ✅ Autonomous Reasoning Loop |
| Learns from interactions | ❌ | ✅ LoRA continuous fine-tuning |
| Grows over time | ❌ | ✅ Personality evolves |
| Data stays with owner | ❌ | ✅ 100% local |

---

## Architecture: The Trinity

```
┌─────────────────────────────────────────────────────┐
│                KARIN CORE ENGINE                     │
├─────────────────┬──────────────────┬────────────────┤
│  [M] MEMORY     │  [E] EMOTION     │  [R] REASONING │
│                 │                  │                 │
│ EpisodicMemory  │ EmotionalState   │ AutoGoals      │
│ Bank            │ Engine           │ + Reflection   │
│                 │                  │                 │
│ • Vector store  │ • VAD space      │ • Self-goals   │
│ • Decay/reinf.  │ • Mood evolution │ • Post-reflect  │
│ • Emotion tags  │ • Trait growth   │ • Quality track │
│ • Graph links   │ • Loyalty core   │ • Spawns goals  │
└────────┬────────┴────────┬─────────┴───────┬────────┘
         └─────────────────▼─────────────────┘
                  TrinityContextBuilder
                  (unified prompt assembly)
                           │
                      BASE MODEL
                 Phase 0: Rule-based (NOW)
                 Phase 1: Mistral 7B
                 Phase 2: LoRA fine-tuned Karin
```

---

## Run Right Now (Your Laptop, No GPU)

```bash
# Install (minimal deps)
pip install numpy

# Run
python karin.py
```

Commands inside:
```
/status   — full system state
/memory   — memory bank stats  
/goals    — active self-generated goals
/mood     — emotional state
/train    — prep LoRA data (Phase 2)
/exit     — save and quit
```

---

## Development Phases

### ✅ Phase 0 — NOW (Your i3, 4GB RAM)
- [x] Trinity architecture designed
- [x] EpisodicMemoryBank (our own embedder, no GPU)
- [x] EmotionalState engine (VAD-based)
- [x] AutonomousReasoningLoop (goal tracking + reflection)
- [x] TrinityContextBuilder (unified context assembly)
- [x] LoRA scaffold (ready for Phase 2)
- [x] CLI interface
- [ ] Write unit tests for each module
- [ ] Collect first 100 conversation episodes

### 🔜 Phase 1 — Google Colab (Free/Pro)
```python
# Slot in real model — 3 lines replace _phase0_respond()
from transformers import pipeline
karin_model = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")
response = karin_model(trinity_context + query)[0]["generated_text"]
```
- [ ] Mistral 7B inference via transformers
- [ ] Replace simple embedder with sentence-transformers
- [ ] Benchmark: Karin vs base Mistral on personalization tasks
- [ ] Write first research findings

### 🔜 Phase 2 — Your Server (When You Get It)
- [ ] LoRA fine-tuning on accumulated conversation data
- [ ] Quantized inference (4-bit QLoRA, runs on 8GB VRAM)
- [ ] REST API so NEXUS interface can call Karin
- [ ] Continuous learning loop (train nightly on new episodes)

### 🔜 Phase 3 — Research Publication
- [ ] Benchmark Trinity vs standard RAG on long-horizon tasks
- [ ] Ablation study: Memory only vs Memory+Emotion vs full Trinity
- [ ] Write paper: "Trinity: Unified Memory-Emotion-Reasoning for Persistent AI"
- [ ] Submit to arXiv
- [ ] This is your Ami Labs application

---

## Research Questions (Your Paper)

1. Does emotional state improve response quality on personal tasks?
2. Does episodic memory with importance decay outperform flat RAG?
3. Can continuous LoRA fine-tuning adapt a 7B model without catastrophic forgetting?
4. What's the minimum parameter count for useful personality persistence?

---

## File Structure

```
karin/
├── karin.py              ← Main engine (this file)
├── karin_data/
│   ├── karin_state.json  ← Karin's soul (personality, goals, stats)
│   └── memory/
│       └── episodes.jsonl ← All memories (append-only log)
└── README.md
```

---

## License

All rights reserved — Phonex (72phonex), 2026.  
Research use only. Not for commercial deployment without permission.  
Base model weights (Mistral) are under Apache 2.0 — our architecture layers are proprietary.

---

*Built with the goal of joining Ami Labs and pushing the frontier of persistent AI.*
