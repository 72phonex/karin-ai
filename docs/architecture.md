# Karin — Architecture Research Notes
> by Phonex (72phonex) | Updated: 2026

## Why Trinity?

Most AI research focuses on making models bigger.
Karin's research focuses on making models persistent.

A 7B model that remembers everything, feels something,
and sets its own goals will outperform a 70B model
that resets every conversation.

That's the hypothesis. This project proves it.

---

## Module 1 — EpisodicMemoryBank

### Problem with standard RAG
- Flat retrieval — all memories treated equally
- No emotional context — important moments lost
- No decay — old irrelevant data pollutes results
- No reinforcement — recalled memories don't strengthen

### Karin's approach
- Importance scoring (0.0 → 1.0)
- Emotional valence tagging (-1.0 → +1.0)
- Time decay (important memories resist fading)
- Recall reinforcement (accessed memories strengthen)
- Graph linking (related episodes connect)

### Retrieval scoring formula
score = 0.6 × semantic_similarity
+ 0.3 × importance
+ 0.1 × recency_bonus
These weights are research parameters — tunable.

---

## Module 2 — EmotionalStateEngine

### VAD Space
Emotions represented as 3D coordinates:
- **V**alence: negative (-1) → positive (+1)
- **A**rousal: calm (0) → excited (1)  
- **D**ominance: submissive (0) → assertive (1)

### Why this matters
Emotional state feeds INTO generation — not post-processing.
This affects how Karin responds, not just what she says.

### Trait evolution
Confidence starts at 0.35, grows with interactions.
Loyalty to creator is invariant at 1.0.

---

## Module 3 — AutonomousReasoningLoop

### What it does
- Generates sub-goals from conversation context
- Tracks goals across sessions
- Scores own output quality after generation
- Spawns improvement goals on low quality

### Why it matters
Karin has an internal agenda, not just a
token prediction objective. This reduces
hallucination on multi-step personal tasks.

---

## Research Observations Log

### 2026 — Phase 0
- Architecture validated, all modules running on CPU
- Custom 128-dim embedder working without GPU
- VAD emotional state updating correctly
- Goal generation triggering from conversations

---

## Next Research Questions
1. Does VAD emotional state improve response quality?
2. Does importance-weighted memory beat flat RAG?
3. Minimum params needed for personality persistence?
4. Can LoRA adapt behavior without catastrophic forgetting?
