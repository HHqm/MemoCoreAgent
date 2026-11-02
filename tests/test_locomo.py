import unittest

from memocore.datasets.locomo import ABSTAIN_ANSWER, load_locomo
from tests.helpers import FIXTURE


class LocomoLoaderTest(unittest.TestCase):
    def setUp(self):
        self.samples = load_locomo(FIXTURE)
        self.conversation = self.samples[0].conversation

    def test_sessions_are_ordered_by_index_not_file_order(self):
        self.assertEqual([s.index for s in self.conversation.sessions], [1, 2])
        self.assertEqual(self.conversation.turn_ids, ("D1:1", "D1:2", "D1:3", "D2:1", "D2:2"))

    def test_render_keeps_dates_and_image_captions(self):
        text = self.conversation.render()
        self.assertIn("--- Conversation at 1:00 pm on 8 May, 2023 ---", text)
        self.assertIn("[shares an image: a painting of a sunset over a lake]", text)
        self.assertLess(text.index("8 May, 2023"), text.index("20 June, 2023"))

    def test_qa_parsing(self):
        qa = self.samples[0].qa
        self.assertEqual(qa[0].qa_id, "conv-test#q0000")
        self.assertEqual(qa[0].category_name, "temporal")
        self.assertEqual(qa[5].answer, "2023")  # numeric answers become strings

    def test_adversarial_gold_is_abstention_and_evidence_is_split(self):
        adversarial = self.samples[0].qa[4]
        self.assertEqual(adversarial.gold, ABSTAIN_ANSWER)
        self.assertEqual(adversarial.adversarial_answer, "golden retriever")
        self.assertEqual(adversarial.evidence, ("D1:1", "D2:2"))


if __name__ == "__main__":
    unittest.main()
