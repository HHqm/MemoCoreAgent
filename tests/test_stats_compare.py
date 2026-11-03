import unittest

from memocore.compare import compare_runs, render_comparison
from memocore.embeddings import HashingEmbedder
from memocore.llm import MockChat
from memocore.retrieval.rerank import LexicalReranker
from memocore.runner import RunConfig, run_judge, run_predictions, summarize
from memocore.stats import bootstrap_ci, cohen_kappa, paired_bootstrap, wilson_ci
from tests.helpers import temp_settings


class StatsTest(unittest.TestCase):
    def test_bootstrap_ci_brackets_the_mean_and_is_reproducible(self):
        values = [1.0] * 60 + [0.0] * 40
        low, high = bootstrap_ci(values)
        self.assertLess(low, 0.6)
        self.assertGreater(high, 0.6)
        self.assertEqual(bootstrap_ci(values), (low, high))
        self.assertIsNone(bootstrap_ci([1.0]))

    def test_wilson_and_kappa(self):
        low, high = wilson_ci(90, 100)
        self.assertLess(low, 0.9)
        self.assertLess(0.9, high)
        self.assertLessEqual(wilson_ci(10, 10)[1], 1.0)
        self.assertIsNone(wilson_ci(0, 0))
        # 90% raw agreement, but on a skewed label mix most of it is chance.
        pairs = [("C", "C")] * 85 + [("C", "W")] * 5 + [("W", "C")] * 5 + [("W", "W")] * 5
        self.assertAlmostEqual(cohen_kappa(pairs), (0.9 - (0.9 * 0.9 + 0.1 * 0.1)) / (1 - 0.82))
        self.assertIsNone(cohen_kappa([("C", "C")] * 3))

    def test_paired_bootstrap_detects_a_clear_gain(self):
        baseline = [0.0] * 50 + [1.0] * 50
        candidate = [1.0] * 90 + [0.0] * 10
        result = paired_bootstrap(baseline, candidate)
        self.assertAlmostEqual(result["diff"], 0.4)
        self.assertGreater(result["ci"][0], 0)
        self.assertLess(result["p_value"], 0.01)
        with self.assertRaises(ValueError):
            paired_bootstrap([1.0], [1.0, 0.0])


class CompareRunsTest(unittest.TestCase):
    def test_identical_answers_give_zero_difference(self):
        settings = temp_settings(self)
        for system in ("naive_rag", "hybrid_rag"):
            cfg = RunConfig(system=system, run_name=system, categories=(1, 2, 3, 4, 5), mock=True)
            run_predictions(
                cfg, settings, chat=MockChat(), embedder=HashingEmbedder(), reranker=LexicalReranker(), progress=False
            )
            run_judge(system, "strict", settings, progress=False)
        result = compare_runs(settings, "naive_rag", "hybrid_rag")
        overall = result["groups"]["overall"]
        self.assertEqual(overall["n"], 7)
        self.assertEqual(overall["diff"], 0.0)
        self.assertEqual(overall["candidate_only"] + overall["baseline_only"], 0)
        self.assertIn("配对对比", render_comparison(result))
        self.assertIsNotNone(summarize("hybrid_rag", settings)["groups"]["overall"]["judge_strict_ci"])
        with self.assertRaises(ValueError):
            compare_runs(settings, "naive_rag", "hybrid_rag", metric="judge_lenient")


if __name__ == "__main__":
    unittest.main()
