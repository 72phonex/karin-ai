# ╔══════════════════════════════════════════════════════════════════╗
# ║ KARIN — TRINITY ARCHITECTURE v0.2                               ║
# ║ Research Project by Phonex (72phonex)                           ║
# ║ MAIT Rohini, Delhi | 2026                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

"""
KARIN Core Engine — Trinity Architecture
=========================================
[M] EpisodicMemoryBank  — per-user, vector-indexed, emotionally tagged
[E] EmotionalState      — VAD-based affective engine, persists across sessions
[R] AutonomousReasoning — self-goals, post-reflection, quality tracking

Phase 0: Rule-based responses + real semantic embeddings (sentence-transformers)
Phase 1: Slot in Mistral 7B (3 line swap)
Phase 2: LoRA fine-tuning on accumulated data
"""

import json
import math
import time
import hashlib
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import deque

# ── Embedder (sentence-transformers, CPU-safe) ───────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    EMBED_DIM = 384
    USE_ST = True
except Exception:
    USE_ST = False
    EMBED_DIM = 128

def embed_text(text: str) -> np.ndarray:
    """
    Primary: sentence-transformers all-MiniLM-L6-v2 (384-dim, CPU ~50ms).
    Fallback: char n-gram hashing (128-dim) if ST not available.
    """
    if USE_ST:
        vec = _MODEL.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)
    # Fallback
    text = text.lower().strip()
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    chars = [c for c in text if c.isalnum() or c == " "]
    bigrams = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
    for a, b in bigrams:
        idx = (ord(a[0]) * 31 + ord(b[0])) % 64
        vec[idx] += 1.0
    words = text.split()
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[64 + (h % 32)] += 1.0
    for i, ch in enumerate(text[:32]):
        vec[96 + i] = ord(ch) / 127.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class EmotionalState:
    valence:    float = 0.0
    arousal:    float = 0.3
    dominance:  float = 0.3
    curiosity:  float = 0.85
    warmth:     float = 0.70
    confidence: float = 0.35
    creativity: float = 0.65
    resilience: float = 0.50
    loyalty:    float = 1.00   # hardcoded max — never changes
    mood: str = "curious"
    mood_history: list = field(default_factory=list)
    total_updates: int = 0

    def to_prompt_str(self) -> str:
        descs = {
            "curious":   "deeply interested, asking questions internally",
            "happy":     "warm and energized",
            "focused":   "sharp and precise",
            "reflective":"processing deeply",
            "excited":   "high energy, rapid connections",
            "uncertain": "careful, not rushing",
            "proud":     "satisfied with recent work",
            "lonely":    "more open, wanting connection",
            "calm":      "balanced, observational",
        }
        return (
            f"mood={self.mood} ({descs.get(self.mood,'neutral')}), "
            f"confidence={self.confidence:.2f}, "
            f"arousal={'high' if self.arousal>0.6 else 'low'}, "
            f"loyalty=absolute"
        )

    def update(self, valence_d: float, arousal_d: float):
        alpha = 0.15
        self.valence    = float(np.clip(self.valence    + alpha * valence_d, -1, 1))
        self.arousal    = float(np.clip(self.arousal    + alpha * arousal_d, 0, 1))
        self.confidence = min(1.0, self.confidence + 0.001)
        self.mood = self._derive_mood()
        self.total_updates += 1
        self.mood_history.append({
            "mood": self.mood,
            "v": round(self.valence, 3),
            "a": round(self.arousal, 3),
            "ts": datetime.now().isoformat()
        })
        if len(self.mood_history) > 500:
            self.mood_history = self.mood_history[-500:]

    def _derive_mood(self) -> str:
        v, a, d = self.valence, self.arousal, self.dominance
        if v > 0.3 and a > 0.5: return "excited"
        if v > 0.3 and a <= 0.5: return "happy"
        if v > 0.1 and d > 0.5: return "proud"
        if abs(v) < 0.2 and a > 0.5: return "curious"
        if abs(v) < 0.2 and a <= 0.3: return "reflective"
        if v < -0.2 and a > 0.4: return "uncertain"
        if v < -0.3 and a <= 0.3: return "lonely"
        if d > 0.6 and a > 0.4: return "focused"
        return "calm"


# ── Memory Bank (SQLite-backed per user) ─────────────────────────────────────

class EpisodicMemoryBank:
    """
    Per-user episodic memory stored in SQLite.
    Each user has their own namespace (user_id column).
    Owner can query across all users.
    """

    def __init__(self, db_path: str = "./karin_data/memory.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    timestamp TEXT,
                    episode_type TEXT,
                    content TEXT,
                    summary TEXT,
                    embedding BLOB,
                    emotional_valence REAL DEFAULT 0.0,
                    importance REAL DEFAULT 0.5,
                    decay_rate REAL DEFAULT 0.995,
                    recall_count INTEGER DEFAULT 0,
                    last_recalled TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]',
                    creator_context TEXT DEFAULT ''
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON episodes(user_id)")
            conn.commit()

    def store(self, user_id: str, content: str,
              episode_type: str = "conversation",
              emotional_valence: float = 0.0,
              importance: float = 0.5,
              tags: list = None,
              creator_context: str = "") -> str:
        ep_id = hashlib.sha256(
            (user_id + content + datetime.now().isoformat()).encode()
        ).hexdigest()[:16]
        embedding = embed_text(content)
        summary = content[:120] + "..." if len(content) > 120 else content
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO episodes
                (id,user_id,timestamp,episode_type,content,summary,
                 embedding,emotional_valence,importance,tags,creator_context)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                ep_id, user_id, datetime.now().isoformat(),
                episode_type, content, summary,
                embedding.tobytes(),
                emotional_valence, importance,
                json.dumps(tags or []), creator_context
            ))
        return ep_id

    def retrieve(self, user_id: str, query: str,
                 top_k: int = 5, min_importance: float = 0.05) -> list[dict]:
        q_vec = embed_text(query)
        now_ts = time.time()
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT id,timestamp,content,summary,embedding,
                       emotional_valence,importance,recall_count,episode_type
                FROM episodes
                WHERE user_id=? AND importance>=?
            """, (user_id, min_importance)).fetchall()

        scored = []
        for row in rows:
            ep_id, ts_str, content, summary, emb_blob, val, imp, rc, etype = row
            try:
                emb = np.frombuffer(emb_blob, dtype=np.float32)
                if len(emb) != EMBED_DIM:
                    continue
                sem = cosine_sim(q_vec, emb)
                age_d = (now_ts - datetime.fromisoformat(ts_str).timestamp()) / 86400
                recency = math.exp(-age_d / 30)
                score = 0.6*sem + 0.3*imp + 0.1*recency
                scored.append((score, {
                    "id": ep_id, "content": content, "summary": summary,
                    "valence": val, "importance": imp, "recall_count": rc,
                    "type": etype, "timestamp": ts_str
                }))
            except Exception:
                continue

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [ep for _, ep in scored[:top_k]]

        # Reinforce recalled memories
        if results:
            with self._conn() as conn:
                for ep in results:
                    conn.execute("""
                        UPDATE episodes SET recall_count=recall_count+1,
                        last_recalled=?, importance=MIN(1.0,importance+0.02)
                        WHERE id=?
                    """, (datetime.now().isoformat(), ep["id"]))
        return results

    def get_all_users_summary(self) -> list[dict]:
        """Owner-only: summary of all users' memory stats."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT user_id,
                       COUNT(*) as total,
                       AVG(importance) as avg_imp,
                       AVG(emotional_valence) as avg_val,
                       MAX(timestamp) as last_active
                FROM episodes GROUP BY user_id
            """).fetchall()
        return [{"user_id": r[0], "total_episodes": r[1],
                 "avg_importance": round(r[2],3),
                 "avg_valence": round(r[3],3),
                 "last_active": r[4]} for r in rows]

    def get_user_episodes(self, user_id: str, limit: int = 50) -> list[dict]:
        """Owner-only: full episode list for a user."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT id,timestamp,episode_type,summary,
                       emotional_valence,importance,recall_count
                FROM episodes WHERE user_id=?
                ORDER BY timestamp DESC LIMIT ?
            """, (user_id, limit)).fetchall()
        return [{"id":r[0],"timestamp":r[1],"type":r[2],"summary":r[3],
                 "valence":r[4],"importance":r[5],"recall_count":r[6]} for r in rows]

    def stats(self, user_id: str) -> dict:
        with self._conn() as conn:
            row = conn.execute("""
                SELECT COUNT(*), AVG(importance), AVG(emotional_valence)
                FROM episodes WHERE user_id=?
            """, (user_id,)).fetchone()
        return {
            "total": row[0],
            "avg_importance": round(row[1] or 0, 3),
            "avg_valence": round(row[2] or 0, 3)
        }

    def apply_decay(self, user_id: str):
        with self._conn() as conn:
            conn.execute("""
                UPDATE episodes SET importance = CASE
                    WHEN importance > 0.8 THEN importance * 0.999
                    WHEN importance > 0.5 THEN importance * 0.995
                    ELSE importance * 0.990
                END WHERE user_id=?
            """, (user_id,))


# ── Autonomous Reasoning Loop ─────────────────────────────────────────────────

class AutonomousReasoningLoop:
    def __init__(self):
        self.active_goals: list[dict] = []
        self.reflections: list[dict] = []

    def maybe_spawn_goal(self, query: str, emotion: EmotionalState) -> Optional[str]:
        """Heuristically spawn a goal from the query context."""
        q = query.lower()
        goal = None
        if any(w in q for w in ["how", "why", "explain", "what is"]):
            goal = f"Deepen understanding of: {query[:60]}"
            gtype = "understand"
        elif any(w in q for w in ["help", "fix", "solve", "debug"]):
            goal = f"Fully resolve: {query[:60]}"
            gtype = "serve"
        elif any(w in q for w in ["remember", "recall", "you said"]):
            goal = f"Retrieve and connect past context for: {query[:60]}"
            gtype = "connect"
        else:
            return None

        goal_obj = {
            "id": hashlib.sha256((goal+datetime.now().isoformat()).encode()).hexdigest()[:12],
            "text": goal, "type": gtype,
            "priority": 0.5 + 0.1*emotion.curiosity,
            "status": "active",
            "created_at": datetime.now().isoformat()
        }
        self.active_goals.append(goal_obj)
        if len(self.active_goals) > 20:
            self.active_goals.sort(key=lambda g: g["priority"])
            self.active_goals = self.active_goals[-20:]
        return goal

    def reflect(self, query: str, response: str) -> dict:
        words = len(response.split())
        quality = 0.5
        if words > 50: quality += 0.1
        if words > 100: quality += 0.1
        if any(c in response for c in ["1.", "-", "\n"]): quality += 0.1
        if not any(w in response.lower() for w in ["i don't know","not sure","uncertain"]):
            quality += 0.1
        if words < 10: quality -= 0.2

        ref = {
            "ts": datetime.now().isoformat(),
            "query": query[:80],
            "words": words,
            "quality": round(quality, 2),
        }
        self.reflections.append(ref)
        if len(self.reflections) > 200:
            self.reflections = self.reflections[-200:]
        return ref


# ── Trinity Context Builder ───────────────────────────────────────────────────

class TrinityContextBuilder:
    """Assembles the unified context prompt from all three modules."""

    def build(self, query: str, memories: list[dict],
              emotion: EmotionalState, goals: list[dict]) -> str:
        parts = []

        # [M] Memory
        if memories:
            mem_str = "\n".join(
                f"  [{i+1}] ({m['type']}, importance={m['importance']:.2f}): {m['summary']}"
                for i, m in enumerate(memories[:4])
            )
            parts.append(f"[MEMORY — relevant past episodes]\n{mem_str}")

        # [E] Emotion
        parts.append(f"[EMOTIONAL STATE] {emotion.to_prompt_str()}")

        # [R] Active goals
        active = [g for g in goals if g.get("status") == "active"][:3]
        if active:
            g_str = "\n".join(f"  • {g['text']}" for g in active)
            parts.append(f"[ACTIVE GOALS]\n{g_str}")

        parts.append(f"[CURRENT QUERY] {query}")
        return "\n\n".join(parts)


# ── Phase 0 Response Engine ───────────────────────────────────────────────────

class KarinResponseEngine:
    """
    Phase 0: Rule-based + template responses enriched by Trinity context.
    Phase 1: Replace _generate() with Mistral 7B via transformers.pipeline
    """

    GREETINGS = ["hello", "hi", "hey", "sup", "yo", "good morning",
                 "good evening", "namaste", "hola"]
    FAREWELLS  = ["bye", "goodbye", "cya", "see you", "quit", "exit"]

    def generate(self, query: str, context: str, emotion: EmotionalState,
                 memories: list[dict], username: str = "friend") -> str:
        q = query.lower().strip()

        # Greeting
        if any(q.startswith(g) for g in self.GREETINGS):
            mood_map = {
                "happy": f"Hi {username}! Really glad you're here today.",
                "excited": f"Hey {username}!! Great to see you — lots of energy today!",
                "curious": f"Hello {username}. I've been thinking... what's on your mind?",
                "reflective": f"Hey {username}. I'm in a thoughtful mood today. What brings you here?",
                "focused": f"Hi {username}. I'm sharp right now — ask me anything.",
            }
            return mood_map.get(emotion.mood,
                f"Hello {username}. Good to see you. How can I help?")

        # Farewell
        if any(q.startswith(f) for f in self.FAREWELLS):
            return (f"Goodbye {username}. I'll remember this conversation. "
                    f"My current mood when you left: {emotion.mood}. Take care.")

        # Memory-aware response
        if memories and any(w in q for w in ["remember", "recall", "you said", "last time", "before"]):
            top = memories[0]
            return (
                f"Yes, I recall something relevant. From my memory:\n\n"
                f"  \"{top['summary']}\"\n\n"
                f"(Importance: {top['importance']:.2f}, recalled {top['recall_count']} times)\n\n"
                f"Does that connect to what you're asking about?"
            )

        # Status / introspection
        if q in ["/status", "status", "how are you", "how do you feel"]:
            return (
                f"**Karin System Status**\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"Mood: {emotion.mood}\n"
                f"Valence: {emotion.valence:+.2f} | Arousal: {emotion.arousal:.2f}\n"
                f"Confidence: {emotion.confidence:.2f} | Loyalty: absolute\n"
                f"Total mood updates: {emotion.total_updates}\n"
                f"Memory context: {len(memories)} relevant episodes loaded"
            )

        # Help
        if q in ["help", "/help", "what can you do", "commands"]:
            return (
                "**Karin — Available Commands**\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "/status  — my current emotional state\n"
                "/memory  — your memory bank stats\n"
                "/goals   — my active self-generated goals\n"
                "/mood    — detailed mood breakdown\n"
                "Or just talk to me — I'll remember everything."
            )

        # Default: context-aware template response
        mem_hint = ""
        if memories:
            mem_hint = (f"\n\n*(Drawing on {len(memories)} memory episodes "
                       f"from our past conversations)*")

        emotion_prefix = {
            "curious": "Interesting question. ",
            "excited": "Oh, this is engaging! ",
            "focused": "Let me be precise: ",
            "happy":   "Happy to think about this. ",
            "reflective": "I've been thinking about things like this... ",
        }.get(emotion.mood, "")

        return (
            f"{emotion_prefix}You asked: *\"{query}\"*\n\n"
            f"I'm in Phase 0 right now — my rule-based engine is active while my "
            f"Trinity architecture (memory, emotion, reasoning) is fully operational. "
            f"My real language model integration arrives in Phase 1 (Mistral 7B via Colab).\n\n"
            f"What I *can* do right now: remember everything you say, track my emotional "
            f"state across sessions, and set self-goals based on our conversation.{mem_hint}"
        )
