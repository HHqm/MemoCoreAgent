import unittest

from memocore.datasets.locomo import Conversation
from memocore.prompts import ANSWER_TEMPLATES, answer_messages, prompt_version


class PromptTests(unittest.TestCase):
    def test_baseline_prompt_is_frozen(self):
        # Published results (docs/results/) were answered with these; editing one would orphan its runs.
        published = {"v1": "c43f5fea8ca2", "v2": "352b03225f74", "v3": "e521df09020c"}
        self.assertEqual({name: prompt_version(name) for name in published}, published)
        self.assertEqual(len({prompt_version(name) for name in ANSWER_TEMPLATES}), len(ANSWER_TEMPLATES))

    def test_memory_comes_before_the_question(self):
        conversation = Conversation("c", "Ann", "Bob", ())
        for name in ANSWER_TEMPLATES:
            user = answer_messages(conversation, "MEMORY", "QUESTION?", name)[1]["content"]
            self.assertLess(user.index("MEMORY"), user.index("QUESTION?"))
        with self.assertRaises(ValueError):
            answer_messages(conversation, "m", "q", "v0")

    def test_split_citations(self):
        from memocore.prompts import split_citations

        self.assertEqual(split_citations("An electric guitar [D4:7]"), ("An electric guitar", ("D4:7",)))
        self.assertEqual(
            split_citations("Rock climbing, baking [D2:3, D9:14] [D2:3]"), ("Rock climbing, baking", ("D2:3", "D9:14"))
        )
        self.assertEqual(split_citations("Likely yes (D3:8; D7:1)."), ("Likely yes.", ("D3:8", "D7:1")))
        self.assertEqual(split_citations("Not mentioned in the conversation."), ("Not mentioned in the conversation.", ()))
