import unittest

from memocore.embeddings import HashingEmbedder
from memocore.retrieval_eval import SWEEP_MD, retrieval_sweep
from tests.helpers import temp_settings


class RetrievalSweepTest(unittest.TestCase):
    def test_recall_grows_with_k_and_reaches_full_coverage(self):
        settings = temp_settings(self)
        result = retrieval_sweep(
            settings, top_ks=(1, 10), chunk_turns=(1, 2), embedder=HashingEmbedder(), progress=False
        )
        self.assertEqual(result["questions"], 6)  # adversarial excluded by default
        self.assertEqual(result["evidence_ids_missing"], 0)
        self.assertEqual(len(result["rows"]), 4)
        by_setting = {(row["chunk_turns"], row["top_k"]): row for row in result["rows"]}
        for size in (1, 2):
            self.assertLessEqual(by_setting[(size, 1)]["recall"], by_setting[(size, 10)]["recall"])
            # k=10 covers every chunk of these tiny conversations.
            self.assertEqual(by_setting[(size, 10)]["recall"], 1.0)
            self.assertEqual(by_setting[(size, 10)]["all_evidence_rate"], 1.0)
        self.assertTrue((settings.runs_dir / SWEEP_MD).exists())


if __name__ == "__main__":
    unittest.main()
