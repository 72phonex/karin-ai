# KARIN — Research Questions
> by Phonex (72phonex)

These are the core questions this project investigates.
Each one will be answered experimentally across phases.

---

## Primary Questions

### Q1 — Memory
> Does importance-weighted episodic memory with emotional
> tagging outperform flat vector retrieval (standard RAG)
> on long-horizon personal tasks?

**Hypothesis:** Yes. Emotional salience and importance decay
mirror how human memory works. Flat RAG treats a casual
remark the same as a life goal. Karin doesn't.

**How to test:** Compare retrieval relevance scores on
a personal task dataset — Karin vs ChromaDB flat retrieval.

---

### Q2 — Emotion
> Does emotional state as an internal variable (feeding
> INTO generation) improve response quality on personal
> queries vs models with no emotional state?

**Hypothesis:** Yes. A model that "knows" it's curious
will ask better follow-up questions. A model that "knows"
it's confident will give more direct answers.

**How to test:** Blind evaluation — same queries, Karin
with emotion module ON vs OFF. Human raters score quality.

---

### Q3 — Reasoning
> Do self-generated goals reduce hallucination on
> multi-step personal tasks?

**Hypothesis:** Yes. An explicit internal agenda gives
the model a target to stay coherent toward. Without goals,
long responses drift.

**How to test:** Multi-step task completion rate —
Karin with goals vs without.

---

### Q4 — Continuous Learning
> Can LoRA fine-tuning on personal interaction data
> adapt model behavior without catastrophic forgetting
> of general capabilities?

**Hypothesis:** Yes, with small LoRA rank (r=16) and
careful learning rate. General capability preserved,
personal style strengthened.

**How to test:** MMLU benchmark before and after
LoRA fine-tuning. Personal task quality measured separately.

---

### Q5 — Scale
> What is the minimum parameter count for useful
> personality persistence?

**Hypothesis:** 3B-7B range is sufficient. Personality
is more about architecture than raw scale.

**How to test:** Same Trinity architecture on
GPT-2 (124M) vs Mistral 7B. Compare personality
consistency scores over 100 conversations.

---

## Observations Log

| Date | Observation | Phase |
|------|-------------|-------|
| 2026 | Trinity architecture validated on CPU | 0 |

*Updated as research progresses.*

