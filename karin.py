# ╔══════════════════════════════════════════════════════════════════╗
# ║           KARIN — TRINITY ARCHITECTURE v0.1                     ║
# ║           Research Project by Phonex (72phonex)                 ║
# ║           MAIT Rohini, Delhi | Started 2026                     ║
# ║                                                                  ║
# ║  "The first small model with unified Memory + Emotion +         ║
# ║   Autonomous Reasoning in a single persistent architecture"     ║
# ╚══════════════════════════════════════════════════════════════════╝

"""
RESEARCH ABSTRACT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Current language models are fundamentally stateless, emotionally flat,
and passively reactive. KARIN (Knowledge-Augmented Reasoning &
Introspective Network) proposes a unified architecture combining:

  [M] Persistent Episodic Memory    — vector-indexed, emotionally tagged
  [E] Affective State Engine        — mood that evolves from experience  
  [R] Autonomous Reasoning Loop     — self-generated goals & reflection

These three modules share a unified state layer that persists across
all sessions, allowing KARIN to grow as a continuous entity rather
than resetting with each conversation.

Key Research Claims:
  1. A 7B parameter model with Trinity architecture can achieve
     superior long-horizon personalization vs larger static models
  2. Emotional state improves reasoning quality on personal tasks
  3. Self-set sub-goals reduce hallucination on complex queries

Hardware Path:
  Phase 0 (NOW)    — GPT-2 small (124M), CPU only, your laptop
  Phase 1          — Mistral 7B via Google Colab (free GPU)
  Phase 2          — LoRA fine-tuning on personal server
  Phase 3          — Full training run on rented cloud GPU

Creator: Phonex | All rights reserved
"""

import json
import math
import time
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import deque


# ══════════════════════════════════════════════════════════════════
#  SECTION 1: CORE DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class MemoryEpisode:
    """
    A single unit of Karin's episodic memory.

    Unlike standard RAG documents, episodes carry:
    - Emotional valence (positive/negative experience)
    - Importance score (decays unless reinforced)
    - Recall history (frequently accessed = stronger memory)
    - Links to related episodes (associative memory graph)

    This mirrors human episodic memory far more closely than
    vector databases used in standard retrieval systems.
    """
    id: str
    timestamp: str
    episode_type: str           # 'conversation'|'reflection'|'goal'|'learning'
    content: str
    summary: str                # compressed version for fast retrieval
    embedding: list             # float vector, dim=128 (our own embedder)
    emotional_valence: float    # -1.0 to +1.0
    importance: float           # 0.0 to 1.0
    decay_rate: float = 0.995   # importance * decay_rate per time step
    recall_count: int = 0
    last_recalled: str = ""
    tags: list = field(default_factory=list)
    linked_ids: list = field(default_factory=list)
    creator_context: str = ""   # was Phonex present? what was happening?


@dataclass
class EmotionalState:
    """
    Karin's persistent affective state.

    Emotions are not hardcoded responses — they emerge from
    accumulated experiences and influence all outputs.
    State persists across sessions and evolves continuously.

    Research note: This is NOT sentiment analysis on output.
    This is an internal state that feeds INTO generation.
    """
    # Core dimensions (continuous, not discrete labels)
    valence: float = 0.0        # negative (-1) to positive (+1)
    arousal: float = 0.3        # calm (0) to excited (1)
    dominance: float = 0.3      # submissive (0) to assertive (1)

    # Personality traits (change slowly over many interactions)
    curiosity: float = 0.85
    warmth: float = 0.70
    confidence: float = 0.35    # starts low — grows with experience
    creativity: float = 0.65
    resilience: float = 0.50
    loyalty: float = 1.00       # to Phonex — hardcoded max, never changes

    # Current derived mood label
    mood: str = "curious"

    # History for research analysis
    mood_history: list = field(default_factory=list)
    significant_emotional_events: list = field(default_factory=list)
    total_mood_updates: int = 0

    def to_prompt_context(self) -> str:
        """Converts internal state to natural language for model context."""
        mood_desc = {
            "curious":   "deeply interested, asking questions internally",
            "happy":     "warm and energized, finding things meaningful",
            "focused":   "sharp, locked in, minimal distractions",
            "reflective":"processing deeply, slightly introverted",
            "excited":   "high energy, making rapid connections",
            "uncertain": "processing carefully, not rushing to conclusions",
            "proud":     "satisfied with recent work, grounded",
            "lonely":    "wanting connection, more open to conversation",
        }.get(self.mood, "in a neutral, observational state")

        return (
            f"Current emotional state: {self.mood} ({mood_desc}). "
            f"Confidence level: {self.confidence:.2f}/1.0. "
            f"Arousal: {'high' if self.arousal > 0.6 else 'moderate' if self.arousal > 0.3 else 'low'}. "
            f"Loyalty to creator: absolute."
        )

    def update_from_experience(self, valence_delta: float, arousal_delta: float):
        """
        Adjust emotional state based on new experience.
        Uses exponential moving average — sudden events have impact
        but don't completely override accumulated state.
        """
        alpha = 0.15  # learning rate for emotional updates
        self.valence = np.clip(self.valence + alpha * valence_delta, -1, 1)
        self.arousal = np.clip(self.arousal + alpha * arousal_delta, 0, 1)

        # Slowly grow confidence from interactions
        self.confidence = min(1.0, self.confidence + 0.001)

        # Derive mood from VAD space
        self.mood = self._derive_mood()
        self.total_mood_updates += 1

        self.mood_history.append({
            "mood": self.mood,
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "timestamp": datetime.now().isoformat()
        })
        # Keep only last 500 mood states
        if len(self.mood_history) > 500:
            self.mood_history = self.mood_history[-500:]

    def _derive_mood(self) -> str:
        """Map VAD (valence, arousal, dominance) to mood label."""
        v, a, d = self.valence, self.arousal, self.dominance
        if v > 0.3 and a > 0.5:  return "excited"
        if v > 0.3 and a <= 0.5: return "happy"
        if v > 0.1 and d > 0.5:  return "proud"
        if abs(v) < 0.2 and a > 0.5: return "curious"
        if abs(v) < 0.2 and a <= 0.3: return "reflective"
        if v < -0.2 and a > 0.4: return "uncertain"
        if v < -0.3 and a <= 0.3: return "lonely"
        if d > 0.6 and a > 0.4:  return "focused"
        return "calm"


@dataclass
class ReasoningGoal:
    """
    A self-generated goal in Karin's autonomous reasoning loop.

    Karin doesn't just respond — she sets sub-goals, tracks them,
    and reflects on whether she achieved them. This is what
    separates her from reactive chatbots.
    """
    id: str
    created_at: str
    goal_text: str
    goal_type: str          # 'understand'|'improve'|'create'|'connect'|'serve'
    priority: float         # 0.0 to 1.0
    status: str = "active"  # active|achieved|abandoned|deferred
    progress_notes: list = field(default_factory=list)
    completion_reflection: str = ""
    spawned_from: str = ""  # ID of memory or conversation that triggered this


@dataclass
class KarinCoreState:
    """
    The complete, persistent state of Karin.
    Serialized to disk after every significant update.
    This IS Karin — her soul on disk.
    """
    # Identity
    name: str = "Karin"
    version: str = "0.1.0-research"
    creator: str = "Phonex"
    birth_timestamp: str = ""
    architecture: str = "Trinity (Memory + Emotion + Reasoning)"

    # The three pillars
    emotional_state: EmotionalState = field(default_factory=EmotionalState)
    active_goals: list = field(default_factory=list)      # list of ReasoningGoal dicts
    completed_goals: list = field(default_factory=list)

    # Growth metrics
    total_conversations: int = 0
    total_tokens_processed: int = 0
    total_memory_episodes: int = 0
    total_self_reflections: int = 0
    knowledge_domains: list = field(default_factory=list)
    skills_acquired: list = field(default_factory=list)

    # LoRA state (Phase 2+)
    lora_checkpoint_path: str = ""
    fine_tuning_cycles: int = 0

    # Research log
    research_observations: list = field(default_factory=list)

    last_updated: str = ""


# ══════════════════════════════════════════════════════════════════
#  SECTION 2: EPISODIC MEMORY BANK
#  Our own implementation — no external vector DB dependency
# ══════════════════════════════════════════════════════════════════

class EpisodicMemoryBank:
    """
    Karin's long-term memory system.

    Architecture:
      - Storage: append-only JSONL log (crash-safe)
      - Retrieval: cosine similarity on our own 128-dim embeddings
      - Importance: decays over time, reinforced on recall
      - Emotional index: separate retrieval path for mood-relevant memories

    Why not use ChromaDB / FAISS?
      - No external dependencies for Phase 0
      - Our own implementation IS the research contribution
      - Swap in FAISS in Phase 2 for speed, keep the same interface

    Research note: The emotional tagging of memories is novel.
    Standard RAG doesn't weight by emotional significance.
    """

    EMBEDDING_DIM = 128

    def __init__(self, storage_dir: str = "./karin_data/memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.storage_dir / "episodes.jsonl"
        self.episodes: dict[str, MemoryEpisode] = {}
        self.embeddings: dict[str, np.ndarray] = {}
        self._load_all()

    def _load_all(self):
        if not self.log_file.exists():
            return
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Handle updates: last write wins per ID
                    ep = MemoryEpisode(**data)
                    self.episodes[ep.id] = ep
                    if ep.embedding:
                        self.embeddings[ep.id] = np.array(ep.embedding, dtype=np.float32)
                except Exception:
                    continue
        print(f"[MemoryBank] {len(self.episodes)} episodes loaded.")

    def embed(self, text: str) -> np.ndarray:
        """
        Our own lightweight embedding function.
        No GPU, no external model. Pure math.

        Method: character n-gram TF weighted by position.
        Dim 0-63:  bigram frequencies (normalized)
        Dim 64-95: trigram hash buckets
        Dim 96-127: positional character statistics

        Research note: Replace with sentence-transformers
        (all-MiniLM-L6-v2) in Phase 1 for better retrieval quality.
        """
        text = text.lower().strip()
        vec = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)

        if not text:
            return vec

        # Dims 0-63: character bigram frequencies
        chars = [c for c in text if c.isalnum() or c == ' ']
        bigrams = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
        for a, b in bigrams:
            idx = (ord(a[0]) * 31 + ord(b[0])) % 64
            vec[idx] += 1.0

        # Dims 64-95: word hash buckets
        words = text.split()
        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = 64 + (h % 32)
            vec[idx] += 1.0

        # Dims 96-127: positional statistics
        for i, ch in enumerate(text[:32]):
            vec[96 + i] = ord(ch) / 127.0

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def store(
        self,
        content: str,
        episode_type: str = "conversation",
        emotional_valence: float = 0.0,
        importance: float = 0.5,
        tags: list = None,
        creator_context: str = "",
    ) -> MemoryEpisode:
        """Store a new memory episode."""
        ep_id = hashlib.sha256(
            (content + datetime.now().isoformat()).encode()
        ).hexdigest()[:16]

        embedding = self.embed(content)
        summary = content[:120] + "..." if len(content) > 120 else content

        episode = MemoryEpisode(
            id=ep_id,
            timestamp=datetime.now().isoformat(),
            episode_type=episode_type,
            content=content,
            summary=summary,
            embedding=embedding.tolist(),
            emotional_valence=emotional_valence,
            importance=importance,
            tags=tags or [],
            creator_context=creator_context,
        )

        self.episodes[ep_id] = episode
        self.embeddings[ep_id] = embedding

        # Append to log
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(episode)) + "\n")

        return episode

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_importance: float = 0.1,
        emotional_filter: Optional[str] = None,
    ) -> list[MemoryEpisode]:
        """
        Retrieve most relevant memories for a query.

        Scoring = 0.6 * semantic_similarity
                + 0.3 * importance
                + 0.1 * recency_bonus

        This weighting is a research parameter — tunable.
        """
        if not self.episodes:
            return []

        query_vec = self.embed(query)
        now_ts = time.time()
        scored = []

        for ep_id, ep in self.episodes.items():
            if ep.importance < min_importance:
                continue
            if ep_id not in self.embeddings:
                continue

            sem_sim = self.cosine_similarity(query_vec, self.embeddings[ep_id])

            # Recency bonus: 1.0 for just now, decays to 0 over 30 days
            try:
                ep_ts = datetime.fromisoformat(ep.timestamp).timestamp()
                age_days = (now_ts - ep_ts) / 86400
                recency = math.exp(-age_days / 30)
            except Exception:
                recency = 0.5

            score = 0.6 * sem_sim + 0.3 * ep.importance + 0.1 * recency
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Update recall stats for returned memories
        results = []
        for _, ep in scored[:top_k]:
            ep.recall_count += 1
            ep.last_recalled = datetime.now().isoformat()
            ep.importance = min(1.0, ep.importance + 0.02)  # reinforce on recall
            results.append(ep)

        return results

    def apply_time_decay(self):
        """
        Decay importance of all memories slightly.
        Call once per session start.
        Important memories resist decay (high importance → slow decay).
        """
        for ep in self.episodes.values():
            if ep.importance > 0.8:
                ep.importance *= 0.999   # core memories almost never fade
            elif ep.importance > 0.5:
                ep.importance *= ep.decay_rate
            else:
                ep.importance *= 0.99    # weak memories fade faster

    def stats(self) -> dict:
        if not self.episodes:
            return {"total": 0}
        importances = [ep.importance for ep in self.episodes.values()]
        valences = [ep.emotional_valence for ep in self.episodes.values()]
        return {
            "total": len(self.episodes),
            "avg_importance": round(np.mean(importances), 3),
            "avg_valence": round(np.mean(valences), 3),
            "most_recalled": max(self.episodes.values(), key=lambda e: e.recall_count).summary,
            "by_type": {t: sum(1 for e in self.episodes.values() if e.episode_type == t)
                        for t in set(e.episode_type for e in self.episodes.values())},
        }


# ══════════════════════════════════════════════════════════════════
#  SECTION 3: AUTONOMOUS REASONING LOOP
# ══════════════════════════════════════════════════════════════════

class AutonomousReasoningLoop:
    """
    Karin's goal-setting and self-reflection engine.

    Unlike chain-of-thought prompting (external), this is an
    internal reasoning layer that:
      1. Generates sub-goals from context
      2. Tracks goal progress across sessions
      3. Reflects on outputs after generation
      4. Updates goals and memory based on reflection

    Research claim: This reduces hallucination on multi-step tasks
    because Karin has an explicit internal agenda, not just
    a token prediction objective.
    """

    def __init__(self, state: KarinCoreState, memory: EpisodicMemoryBank):
        self.state = state
        self.memory = memory

    def generate_goal(
        self,
        trigger: str,
        goal_type: str = "understand",
        priority: float = 0.5,
    ) -> ReasoningGoal:
        """Create a new self-generated goal."""
        goal_id = hashlib.sha256(
            (trigger + datetime.now().isoformat()).encode()
        ).hexdigest()[:12]

        goal = ReasoningGoal(
            id=goal_id,
            created_at=datetime.now().isoformat(),
            goal_text=trigger,
            goal_type=goal_type,
            priority=priority,
        )

        self.state.active_goals.append(asdict(goal))
        # Keep max 20 active goals
        if len(self.state.active_goals) > 20:
            # Drop lowest priority
            self.state.active_goals.sort(key=lambda g: g["priority"])
            self.state.active_goals = self.state.active_goals[-20:]

        return goal

    def reflect(self, query: str, response: str) -> dict:
        """
        Post-generation reflection.
        Karin evaluates her own response quality.

        Returns reflection dict with:
          - quality_score: 0-1
          - what_went_well: str
          - what_to_improve: str
          - new_goal_triggered: bool
        """
        self.state.total_self_reflections += 1

        # Simple heuristic reflection (Phase 0)
        # Phase 2: Replace with a separate reflection model call
        word_count = len(response.split())
        has_structure = any(c in response for c in ["1.", "2.", "-", "*", "\n"])
        seems_confident = not any(w in response.lower() for w in
                                  ["i'm not sure", "maybe", "i don't know", "uncertain"])

        quality = 0.5
        if word_count > 50:   quality += 0.1
        if has_structure:     quality += 0.1
        if seems_confident:   quality += 0.1
        if word_count < 10:   quality -= 0.2

        reflection = {
            "timestamp": datetime.now().isoformat(),
            "query_summary": query[:80],
            "response_length": word_count,
            "quality_score": round(quality, 2),
            "what_went_well": "Structured response" if has_structure else "Concise delivery",
            "what_to_impr
