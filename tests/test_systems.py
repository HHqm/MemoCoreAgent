import unittest

from memocore.datasets.locomo import load_locomo
from memocore.embeddings import HashingEmbedder
from memocore.systems import FullContextSystem, NaiveRAGSystem, build_system
from tests.helpers import FIXTURE


class SystemsTest(unittest.TestCase):
    def setUp(self):
        self.conversation = load_locomo(FIXTURE)[0].conversation

    def test_full_context_returns_every_turn(self):
        system = FullContextSystem()
        system.ingest(self.conversation)
        context = system.retrieve("anything")
        self.assertEqual(context.turn_ids, self.conversation.turn_ids)
        self.assertIn("20 June, 2023", context.text)

    def test_naive_rag_finds_the_matching_turn(self):
        system = NaiveRAGSystem(HashingEmbedder(), top_k=1)
        system.ingest(self.conversation)
        context = system.retrieve("Who is learning the violin?")
        self.assertEqual(context.turn_ids, ("D1:2",))
        self.assertIn("8 May, 2023", context.text)  # the hit keeps its session date

    def test_chunks_never_cross_sessions_and_come_back_in_order(self):
        system = NaiveRAGSystem(HashingEmbedder(), top_k=10, chunk_turns=2)
        system.ingest(self.conversation)
        context = system.retrieve("Max")
        # Session 1 has 3 turns -> [D1:1, D1:2] and [D1:3]; session 2 -> [D2:1, D2:2].
        self.assertEqual(context.turn_ids, ("D1:1", "D1:2", "D1:3", "D2:1", "D2:2"))
        # Adjacent hits from one session share a single date header.
        self.assertEqual(context.text.count("--- Conversation at"), 2)

    def test_ranked_chunks_are_best_first_per_question(self):
        system = NaiveRAGSystem(HashingEmbedder(), top_k=1)
        system.ingest(self.conversation)
        ranked = system.ranked_chunks(["Who is learning the violin?", "Boston Seattle move"], 2)
        self.assertEqual(ranked[0][0], ("D1:2",))
        self.assertEqual(ranked[1][0], ("D2:1",))
        self.assertEqual(len(ranked[0]), 2)

    def test_invalid_settings_and_names(self):
        with self.assertRaises(ValueError):
            NaiveRAGSystem(HashingEmbedder(), top_k=0)
        with self.assertRaises(ValueError):
            build_system("naive_rag")
        with self.assertRaises(ValueError):
            build_system("nope")


if __name__ == "__main__":
    unittest.main()
