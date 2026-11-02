import unittest

from memocore.judge import judge_messages, mock_judge_responder, parse_verdict, rule_verdict


class JudgeTest(unittest.TestCase):
    def test_parses_json(self):
        verdict = parse_verdict('{"reasoning": "same date", "label": "CORRECT"}')
        self.assertEqual(verdict.label, "CORRECT")
        self.assertEqual(verdict.reasoning, "same date")
        self.assertTrue(verdict.correct)

    def test_parses_fenced_json_and_free_text(self):
        self.assertEqual(parse_verdict('```json\n{"reasoning": "x", "label": "wrong"}\n```').label, "WRONG")
        self.assertEqual(parse_verdict("The answer is WRONG.").label, "WRONG")

    def test_incorrect_is_not_read_as_correct(self):
        verdict = parse_verdict("This is incorrect")
        self.assertIsNone(verdict.label)
        self.assertFalse(verdict.correct)

    def test_strict_rubric_has_granularity_rule_and_lenient_does_not(self):
        strict = judge_messages("strict", "q", "7 May 2023", "May 2023")[0]["content"]
        lenient = judge_messages("lenient", "q", "7 May 2023", "May 2023")[0]["content"]
        self.assertIn("Time granularity", strict)
        self.assertNotIn("Time granularity", lenient)
        with self.assertRaises(ValueError):
            judge_messages("other", "q", "g", "p")

    def test_rule_verdict_settles_only_clear_cases(self):
        abstain = "Not mentioned in the conversation."
        self.assertEqual(rule_verdict("violin", abstain).label, "WRONG")
        self.assertEqual(rule_verdict("Not mentioned in the conversation", abstain).label, "CORRECT")
        self.assertEqual(rule_verdict("Not mentioned in the conversation", "golden retriever").label, "WRONG")
        self.assertEqual(rule_verdict("The violin", "violin!").label, "CORRECT")
        self.assertIsNone(rule_verdict("violin", "She plays the violin"))
        # A long reply that declines and then answers anyway still goes to the judge model.
        self.assertIsNone(rule_verdict("violin", "Not mentioned directly, but she seems to play the violin a lot"))

    def test_mock_judge_matches_substring(self):
        messages = judge_messages("strict", "q", "violin", "She plays the violin")
        self.assertEqual(parse_verdict(mock_judge_responder(messages)).label, "CORRECT")
        messages = judge_messages("strict", "q", "violin", "piano")
        self.assertEqual(parse_verdict(mock_judge_responder(messages)).label, "WRONG")


if __name__ == "__main__":
    unittest.main()
