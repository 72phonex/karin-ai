# KARIN — Unit Tests
# EpisodicMemoryBank Tests
# by Phonex (72phonex)

import sys
sys.path.append("..")

from karin import EpisodicMemoryBank, MemoryEpisode

def test_store_and_retrieve():
    bank = EpisodicMemoryBank("./test_memory")
    ep = bank.store(
        content="Phonex wants to build Jarvis-level AI",
        episode_type="conversation",
        emotional_valence=0.8,
        importance=0.9,
        tags=["goal", "phonex"]
    )
    assert ep.id is not None
    assert ep.importance == 0.9
    results = bank.retrieve("Jarvis AI", top_k=1)
    assert len(results) > 0
    print("✅ store_and_retrieve passed")

def test_embedding_shape():
    bank = EpisodicMemoryBank("./test_memory")
    vec = bank.embed("test sentence for karin")
    assert vec.shape[0] == 128
    print("✅ embedding_shape passed")

def test_importance_decay():
    bank = EpisodicMemoryBank("./test_memory")
    bank.store("test memory", importance=0.6)
    before = list(bank.episodes.values())[0].importance
    bank.apply_time_decay()
    after = list(bank.episodes.values())[0].importance
    assert after <= before
    print("✅ importance_decay passed")

if __name__ == "__main__":
    test_store_and_retrieve()
    test_embedding_shape()
    test_importance_decay()
    print("\n✅ All tests passed")
