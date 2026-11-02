import unittest

import numpy as np

from memocore.datasets.locomo import load_locomo
from memocore.embeddings import HashingEmbedder
from memocore.retrieval.bm25 import BM25, tokenize
from memocore.retrieval.fusion import rrf, top_indices
from memocore.retrieval.rerank import LexicalReranker
from memocore.systems import HybridOptions, HybridRAGSystem, NoMemorySystem, build_system
from tests.helpers import FIXTURE


class LexicalPiecesTest(unittest.TestCase):
    def test_tokenize_drops_stopwords_and_light_stems(self):
        self.assertEqual(tokenize("What instruments is Bob learning?"), ["instrument", "bob", "learn"])

    def test_tokenize_splits_chinese_into_bigrams(self):
        self.assertEqual(tokenize("我在学小提琴 violin"), ["violin", "我在", "在学", "学小", "小提", "提琴"])
        bm25 = BM25(["我上周开始学小提琴", "今天去公园散步", "周末看电影"])
        self.assertEqual(int(np.argmax(bm25.scores("谁在学小提琴？"))), 0)

    def test_bm25_prefers_rare_matching_terms(self):
        bm25 = BM25(["violin lessons today", "lessons about cooking", "nothing related"])
        scores = bm25.scores("violin")
        self.assertEqual(int(np.argmax(scores)), 0)
        self.assertEqual(scores[2], 0.0)

    def test_top_indices_and_rrf(self):
        scores = np.array([0.1, 0.0, 0.9, 0.5], dtype=np.float32)
        self.assertEqual(top_indices(scores, 2), [2, 3])
        self.assertEqual(top_indices(scores, 10, positive_only=True), [2, 3, 0])
        # Item 1 is second in both lists and so beats items that top only one list.
        self.assertEqual(rrf([[0, 1, 2], [3, 1, 4]])[0], 1)

    def test_lexical_reranker_counts_shared_words(self):
        scores = LexicalReranker().score("violin lessons", ["violin lessons", "violin", "cooking"])
        self.assertEqual(list(scores), [2.0, 1.0, 0.0])


class HybridSystemTest(unittest.TestCase):
    def setUp(self):
        self.conversation = load_locomo(FIXTURE)[0].conversation

    def _system(self, **options):
        system = HybridRAGSystem(HashingEmbedder(), LexicalReranker(), top_k=1, options=HybridOptions(**options))
        system.ingest(self.conversation)
        return system

    def test_every_stage_finds_the_violin_turn(self):
        for options in ({}, {"use_bm25": False}, {"use_dense": False}, {"rerank_candidates": 5}):
            with self.subTest(**options):
                context = self._system(**options).retrieve("Who is learning the violin?")
                self.assertEqual(context.turn_ids, ("D1:2",))

    def test_neighbours_stay_inside_the_session(self):
        system = self._system(neighbors=1)
        ranked = system.rank(["Look at this painting"])[0]
        self.assertEqual(ranked[0], 2)  # D1:3, the last turn of session 1
        chosen = system.select(ranked, 3)
        self.assertEqual(system.turn_ids(chosen[:2]), ("D1:3", "D1:2"))
        # D2:1 directly follows D1:3 in the flat list but belongs to another session.
        self.assertNotEqual(system.turn_ids(chosen)[1], "D2:1")

    def test_session_cap_spreads_the_budget(self):
        uncapped = self._system()
        capped = self._system(max_per_session=1)
        question = ["Alice Bob Max violin painting moved park"]
        self.assertEqual(len(uncapped.select(uncapped.rank(question)[0], 2)), 2)
        sessions = {capped._pairs[i][0].index for i in capped.select(capped.rank(question)[0], 2)}
        self.assertEqual(sessions, {1, 2})

    def test_pool_fusion_adds_bm25_hits_without_displacing_dense_ones(self):
        system = self._system(fusion="pool", rerank_candidates=2, bm25_pool=2)
        question = "Seattle Boston"
        dense_head = HybridRAGSystem(HashingEmbedder(), options=HybridOptions(use_bm25=False))
        dense_head.ingest(self.conversation)
        head = set(dense_head.rank([question])[0][:2])
        ranked = system.rank([question])[0]
        self.assertTrue(head <= set(ranked[:4]))
        self.assertEqual(sorted(ranked), list(range(len(self.conversation.turn_ids))))  # nothing lost or duplicated

    def test_feedback_hop_adds_reranked_candidates_without_losing_turns(self):
        question = ["Who is learning the violin?"]
        plain = self._system(use_bm25=False, rerank_candidates=2).rank(question)[0]
        system = self._system(use_bm25=False, rerank_candidates=2, feedback_turns=1, feedback_candidates=2)
        ranked = system.rank(question)[0]
        self.assertEqual(ranked[0], plain[0])  # the best hit is reranked against the question either way
        self.assertEqual(sorted(ranked), list(range(len(self.conversation.turn_ids))))  # nothing lost or duplicated
        with self.assertRaises(ValueError):
            HybridRAGSystem(HashingEmbedder(), options=HybridOptions(feedback_turns=2))

    def test_time_anchors_reach_the_context(self):
        system = self._system(time_anchors=True)
        context = system.retrieve("When did Alice adopt Max?")
        self.assertIn("yesterday = 7 May 2023", context.text)

    def test_invalid_combinations(self):
        with self.assertRaises(ValueError):
            HybridRAGSystem(HashingEmbedder(), options=HybridOptions(fusion="pool"))
        with self.assertRaises(ValueError):
            HybridRAGSystem(HashingEmbedder(), options=HybridOptions(fusion="max"))
        with self.assertRaises(ValueError):
            HybridRAGSystem(HashingEmbedder(), options=HybridOptions(use_dense=False, use_bm25=False))
        with self.assertRaises(ValueError):
            HybridRAGSystem(None)
        with self.assertRaises(ValueError):
            HybridRAGSystem(HashingEmbedder(), options=HybridOptions(rerank_candidates=5))

    def test_no_memory_and_factory(self):
        system = build_system("no_memory")
        self.assertIsInstance(system, NoMemorySystem)
        system.ingest(self.conversation)
        self.assertEqual(system.retrieve("anything").turn_ids, ())
        self.assertIsInstance(build_system("hybrid_rag", embedder=HashingEmbedder()), HybridRAGSystem)


if __name__ == "__main__":
    unittest.main()
