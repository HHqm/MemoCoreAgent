import unittest

from memocore.metrics import answer_visible, evidence_recall, locomo_f1, normalize_answer, token_f1


class MetricsTest(unittest.TestCase):
    def test_normalize_drops_case_punctuation_and_articles(self):
        self.assertEqual(normalize_answer("The Violin, and a Piano!"), "violin piano")

    def test_token_f1_uses_stemming(self):
        self.assertAlmostEqual(token_f1("playing violins", "play violin"), 1.0)

    def test_token_f1_partial_overlap(self):
        self.assertAlmostEqual(token_f1("violin piano", "violin"), 2 / 3)
        self.assertEqual(token_f1("", "violin"), 0.0)

    def test_multi_hop_scores_each_gold_item(self):
        self.assertAlmostEqual(locomo_f1("Boston, Seattle", "Boston, Seattle", 1), 1.0)
        self.assertAlmostEqual(locomo_f1("Seattle", "Boston, Seattle", 1), 0.5)

    def test_open_domain_ignores_rationale_after_semicolon(self):
        self.assertAlmostEqual(locomo_f1("Likely yes", "Likely yes; she paints", 3), 1.0)

    def test_adversarial_rewards_abstention_only(self):
        self.assertEqual(locomo_f1("Not mentioned in the conversation.", "anything", 5), 1.0)
        self.assertEqual(locomo_f1("golden retriever", "anything", 5), 0.0)

    def test_answer_visible(self):
        context = "Alice: I adopted Max yesterday [time: yesterday = 7 May 2023]"
        self.assertTrue(answer_visible("7 May 2023", context, 2))
        self.assertTrue(answer_visible("May 7, 2023", context, 2))  # word order and commas do not matter
        self.assertFalse(answer_visible("8 May 2023", context, 2))
        self.assertTrue(answer_visible("Likely yes; Alice adopted", "yes likely", 3))
        self.assertIsNone(answer_visible("anything", context, 5))

    def test_evidence_recall(self):
        self.assertAlmostEqual(evidence_recall(["D1:1", "D2:1"], ["D1:1", "D1:2"]), 0.5)
        self.assertIsNone(evidence_recall([], ["D1:1"]))


if __name__ == "__main__":
    unittest.main()
