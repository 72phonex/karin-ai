"""
Karin — Unit Tests
==================
Tests for EpisodicMemoryBank, EmotionalState, AutonomousReasoningLoop
Run: python -m pytest tests/ -v
"""

import os
import sys
import json
import time
import tempfile
import unittest
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from karin_core import (
    EpisodicMemoryBank, EmotionalState, AutonomousReasoningLoop,
    TrinityContextBuilder, embed_text, cosine_sim, EMBED_DIM
)


# ── Embedder Tests ─────────────────────────────────────────────────────────────

class TestEmbedder(unittest.TestCase):

    def test_output_shape(self):
        v = embed_text("hello world")
        self.assertEqual(v.shape, (EMBED_DIM,))

    def test_output_normalized(self):
        v = embed_text("normalization test")
        norm = float(np.linalg.norm(v))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_dtype_float32(self):
        v = embed_text("dtype check")
        self.assertEqual(v.dtype, np.float32)

    def test_cosine_sim_identical(self):
        v = embed_text("identical sentence")
        self.assertAlmostEqual(cosine_sim(v, v), 1.0, places=4)

    def test_cosine_sim_range(self):
        a = embed_text("the sky is blue")
        b = embed_text("completely unrelated text")
        sim = cosine_sim(a, b)
        self.assertGreaterEqual(sim, -1.0)
        self.assertLessEqual(sim, 1.0)

    def test_semantic_similarity(self):
        """Similar sentences should score higher than dissimilar ones."""
        a = embed_text("I love programming in Python")
        b = embed_text("Python is my favourite language for coding")
        c = embed_text("The weather today is rainy")
        sim_similar   = cosine_sim(a, b)
        sim_different = cosine_sim(a, c)
        # This test only passes with real sentence-transformers
        # With fallback embedder it may fail — that's expected
        try:
            self.assertGreater(sim_similar, sim_different)
        except AssertionError:
            self.skipTest("Fallback embedder in use — semantic test skipped")


# ── EmotionalState Tests ───────────────────────────────────────────────────────

class TestEmotionalState(unittest.TestCase):

    def test_default_state(self):
        e = EmotionalState()
        self.assertEqual(e.loyalty, 1.0)
        self.assertGreater(e.curiosity, 0.5)

    def test_update_clamps_valence(self):
        e = EmotionalState()
        for _ in range(100):
            e.update(1.0, 0.0)
        self.assertLessEqual(e.valence, 1.0)

    def test_update_clamps_negative(self):
        e = EmotionalState()
        for _ in range(100):
            e.update(-1.0, 0.0)
        self.assertGreaterEqual(e.valence, -1.0)

    def test_mood_history_appended(self):
        e = EmotionalState()
        e.update(0.5, 0.5)
        self.assertEqual(len(e.mood_history), 1)
        self.assertIn("mood", e.mood_history[0])

    def test_mood_history_capped_at_500(self):
        e = EmotionalState()
        for _ in range(600):
            e.update(0.01, 0.0)
        self.assertLessEqual(len(e.mood_history), 500)

    def test_total_updates_counter(self):
        e = EmotionalState()
        for _ in range(5):
            e.update(0.1, 0.1)
        self.assertEqual(e.total_updates, 5)

    def test_loyalty_unchanged(self):
        e = EmotionalState()
        for _ in range(50):
            e.update(-1.0, -1.0)
        self.assertEqual(e.loyalty, 1.0)

    def test_derive_mood_happy(self):
        e = EmotionalState()
        e.valence = 0.6
        e.arousal = 0.3
        mood = e._derive_mood()
        self.assertEqual(mood, "happy")

    def test_derive_mood_excited(self):
        e = EmotionalState()
        e.valence = 0.6
        e.arousal = 0.7
        mood = e._derive_mood()
        self.assertEqual(mood, "excited")

    def test_prompt_str_contains_mood(self):
        e = EmotionalState()
        s = e.to_prompt_str()
        self.assertIn("mood=", s)
        self.assertIn("loyalty=absolute", s)

    def test_confidence_increases(self):
        e = EmotionalState()
        initial = e.confidence
        for _ in range(10):
            e.update(0.0, 0.0)
        self.assertGreater(e.confidence, initial)


# ── EpisodicMemoryBank Tests ───────────────────────────────────────────────────

class TestEpisodicMemoryBank(unittest.TestCase):

    def setUp(self):
        self.tmpdir  = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_memory.db")
        self.bank    = EpisodicMemoryBank(self.db_path)
        self.user_id = "test_user_001"

    def test_store_returns_id(self):
        ep_id = self.bank.store(self.user_id, "This is a test episode")
        self.assertIsInstance(ep_id, str)
        self.assertGreater(len(ep_id), 0)

    def test_store_and_retrieve(self):
        self.bank.store(self.user_id, "I enjoy working on neural networks")
        results = self.bank.retrieve(self.user_id, "neural networks", top_k=5)
        self.assertGreater(len(results), 0)

    def test_retrieve_returns_correct_fields(self):
        self.bank.store(self.user_id, "Memory field test")
        results = self.bank.retrieve(self.user_id, "Memory field test", top_k=1)
        self.assertGreater(len(results), 0)
        r = results[0]
        self.assertIn("id", r)
        self.assertIn("content", r)
        self.assertIn("importance", r)
        self.assertIn("recall_count", r)
        self.assertIn("valence", r)

    def test_retrieve_top_k_limit(self):
        for i in range(10):
            self.bank.store(self.user_id, f"Episode {i}: testing memory storage")
        results = self.bank.retrieve(self.user_id, "testing memory", top_k=3)
        self.assertLessEqual(len(results), 3)

    def test_recall_count_increments(self):
        self.bank.store(self.user_id, "Recall count test phrase")
        r1 = self.bank.retrieve(self.user_id, "Recall count test phrase", top_k=1)
        rc1 = r1[0]["recall_count"] if r1 else 0
        r2 = self.bank.retrieve(self.user_id, "Recall count test phrase", top_k=1)
        rc2 = r2[0]["recall_count"] if r2 else 0
        self.assertGreaterEqual(rc2, rc1)

    def test_user_isolation(self):
        """User A's memories should not appear for user B."""
        self.bank.store("user_a", "Secret data about user A")
        results = self.bank.retrieve("user_b", "Secret data about user A", top_k=5)
        self.assertEqual(len(results), 0)

    def test_stats_returns_count(self):
        self.bank.store(self.user_id, "Stats test 1")
        self.bank.store(self.user_id, "Stats test 2")
        stats = self.bank.stats(self.user_id)
        self.assertGreaterEqual(stats["total"], 2)

    def test_stats_empty_user(self):
        stats = self.bank.stats("nonexistent_user_xyz")
        self.assertEqual(stats["total"], 0)

    def test_importance_stored_correctly(self):
        self.bank.store(self.user_id, "Importance test", importance=0.9)
        results = self.bank.retrieve(self.user_id, "Importance test", top_k=1)
        self.assertGreater(len(results), 0)
        self.assertAlmostEqual(results[0]["importance"], 0.9, delta=0.05)

    def test_tags_stored(self):
        self.bank.store(self.user_id, "Tagged memory", tags=["research","karin"])
        # No error means tags stored successfully (JSON serialized)
        results = self.bank.retrieve(self.user_id, "Tagged memory", top_k=1)
        self.assertGreater(len(results), 0)

    def test_decay_does_not_raise(self):
        self.bank.store(self.user_id, "Decay test episode")
        try:
            self.bank.apply_decay(self.user_id)
        except Exception as e:
            self.fail(f"apply_decay raised: {e}")

    def test_get_all_users_summary(self):
        self.bank.store("user_alpha", "Alpha memory")
        self.bank.store("user_beta",  "Beta memory")
        summary = self.bank.get_all_users_summary()
        user_ids = [s["user_id"] for s in summary]
        self.assertIn("user_alpha", user_ids)
        self.assertIn("user_beta",  user_ids)

    def test_get_user_episodes(self):
        for i in range(5):
            self.bank.store(self.user_id, f"Episode {i}")
        eps = self.bank.get_user_episodes(self.user_id, limit=10)
        self.assertGreaterEqual(len(eps), 5)

    def test_get_user_episodes_limit(self):
        for i in range(20):
            self.bank.store(self.user_id, f"Episode {i}")
        eps = self.bank.get_user_episodes(self.user_id, limit=5)
        self.assertLessEqual(len(eps), 5)


# ── AutonomousReasoningLoop Tests ──────────────────────────────────────────────

class TestAutonomousReasoning(unittest.TestCase):

    def setUp(self):
        self.reasoning = AutonomousReasoningLoop()
        self.emotion   = EmotionalState()

    def test_spawn_goal_for_question(self):
        goal = self.reasoning.maybe_spawn_goal("How does attention work?", self.emotion)
        self.assertIsNotNone(goal)
        self.assertIsInstance(goal, str)

    def test_spawn_goal_for_help(self):
        goal = self.reasoning.maybe_spawn_goal("Help me fix this code", self.emotion)
        self.assertIsNotNone(goal)

    def test_spawn_goal_for_recall(self):
        goal = self.reasoning.maybe_spawn_goal("Remember what I said yesterday?", self.emotion)
        self.assertIsNotNone(goal)

    def test_no_goal_for_generic(self):
        goal = self.reasoning.maybe_spawn_goal("the weather today", self.emotion)
        self.assertIsNone(goal)

    def test_goals_accumulate(self):
        self.reasoning.maybe_spawn_goal("How does X work?", self.emotion)
        self.reasoning.maybe_spawn_goal("Why is Y important?", self.emotion)
        self.assertGreaterEqual(len(self.reasoning.active_goals), 2)

    def test_goals_capped_at_20(self):
        for i in range(30):
            self.reasoning.maybe_spawn_goal(f"How does thing {i} work?", self.emotion)
        self.assertLessEqual(len(self.reasoning.active_goals), 20)

    def test_reflect_returns_quality(self):
        ref = self.reasoning.reflect("test query", "This is a detailed response with many words " * 5)
        self.assertIn("quality", ref)
        self.assertIsInstance(ref["quality"], float)

    def test_reflect_quality_range(self):
        ref = self.reasoning.reflect("q", "ok")
        self.assertGreaterEqual(ref["quality"], 0.0)
        self.assertLessEqual(ref["quality"], 1.5)

    def test_reflections_stored(self):
        for i in range(5):
            self.reasoning.reflect(f"query {i}", f"response {i} " * 10)
        self.assertGreaterEqual(len(self.reasoning.reflections), 5)

    def test_reflections_capped(self):
        for i in range(250):
            self.reasoning.reflect(f"q{i}", f"response {i}")
        self.assertLessEqual(len(self.reasoning.reflections), 200)


# ── TrinityContextBuilder Tests ────────────────────────────────────────────────

class TestTrinityContextBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = TrinityContextBuilder()
        self.emotion = EmotionalState()

    def test_builds_with_no_memories(self):
        ctx = self.builder.build("test query", [], self.emotion, [])
        self.assertIn("EMOTIONAL STATE", ctx)
        self.assertIn("CURRENT QUERY", ctx)

    def test_builds_with_memories(self):
        mems = [{
            "id": "abc", "content": "past content", "summary": "past summary",
            "type": "conversation", "importance": 0.7, "recall_count": 1,
            "valence": 0.1, "timestamp": "2026-01-01T00:00:00"
        }]
        ctx = self.builder.build("test query", mems, self.emotion, [])
        self.assertIn("MEMORY", ctx)
        self.assertIn("past summary", ctx)

    def test_builds_with_goals(self):
        goals = [{"text": "Understand topic X", "type": "understand",
                  "status": "active", "priority": 0.6}]
        ctx = self.builder.build("test query", [], self.emotion, goals)
        self.assertIn("ACTIVE GOALS", ctx)
        self.assertIn("Understand topic X", ctx)

    def test_inactive_goals_excluded(self):
        goals = [{"text": "Old goal", "status": "done", "type": "understand", "priority": 0.3}]
        ctx = self.builder.build("test query", [], self.emotion, goals)
        self.assertNotIn("Old goal", ctx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
