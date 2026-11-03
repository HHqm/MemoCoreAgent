import json
import unittest
from datetime import datetime, timezone

from memocore.datasets.locomo import Conversation
from memocore.llm import Completion, OpenAICompatibleChat, ResponseCache, Usage
from memocore.pricing import call_cost, estimate_line, is_peak, next_off_peak
from memocore.prompts import ANSWER_TEMPLATES, answer_messages
from memocore.runner import _complete
from memocore.spend import render_spend, spend_report
from tests.helpers import temp_settings

TUESDAY_PEAK = datetime(2026, 11, 3, 6, 9, tzinfo=timezone.utc)  # 14:09 in Beijing


class PricingTest(unittest.TestCase):
    def test_peak_hours(self):
        self.assertTrue(is_peak(TUESDAY_PEAK))
        self.assertFalse(is_peak(datetime(2026, 11, 3, 10, 0, tzinfo=timezone.utc)))  # 18:00 Beijing
        self.assertFalse(is_peak(datetime(2026, 11, 3, 4, 30, tzinfo=timezone.utc)))  # lunch break
        self.assertFalse(is_peak(datetime(2026, 11, 7, 2, 0, tzinfo=timezone.utc)))  # Saturday
        self.assertEqual(next_off_peak(TUESDAY_PEAK), datetime(2026, 11, 3, 10, 0, tzinfo=timezone.utc))

    def test_call_cost(self):
        # 1M uncached input + 1M output on deepseek-flash at peak: 2 + 8 yuan.
        self.assertAlmostEqual(call_cost("deepseek-flash", 1_000_000, 0, 1_000_000, TUESDAY_PEAK), 10.0)
        self.assertAlmostEqual(call_cost("deepseek-flash", 1_000_000, 1_000_000, 0), 0.04)  # all cached
        off_peak = datetime(2026, 11, 7, 2, 0, tzinfo=timezone.utc)
        self.assertAlmostEqual(call_cost("deepseek-v4-pro", 0, 0, 1_000_000, off_peak), 13.5)
        self.assertIsNone(call_cost("mock", 10, 0, 10))
        self.assertIn("11-03 18:00", estimate_line("deepseek-flash", 300, "answer", TUESDAY_PEAK))
        self.assertEqual(estimate_line("mock", 300, "answer"), "")


class ApiUsageTest(unittest.TestCase):
    def test_cache_key_is_unchanged_without_a_reasoning_level(self):
        messages = [{"role": "user", "content": "hi"}]
        before = json.dumps(
            {"model": "m", "messages": messages, "max_tokens": 5, "temperature": 0.0}, ensure_ascii=False, sort_keys=True
        )
        import hashlib

        self.assertEqual(ResponseCache.key("m", messages, 5, 0.0), hashlib.sha256(before.encode("utf-8")).hexdigest())
        self.assertNotEqual(ResponseCache.key("m", messages, 5, 0.0, "none"), ResponseCache.key("m", messages, 5, 0.0))

    def test_reasoning_levels_map_to_request_arguments(self):
        def args(level):
            return OpenAICompatibleChat("deepseek-flash", api_key="unused", base_url="http://unused.invalid",
                                        reasoning=level)._reasoning_args()

        self.assertEqual(args(None), {})
        self.assertEqual(args("none"), {"extra_body": {"thinking": {"type": "disabled"}}})
        self.assertEqual(args("low"), {"reasoning_effort": "low"})
        with self.assertRaises(ValueError):
            args("medium")

    def test_a_response_without_choices_is_asked_for_again(self):
        from types import SimpleNamespace
        from unittest import mock

        reply = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" Paris "), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, prompt_cache_hit_tokens=0),
        )
        empty = SimpleNamespace(choices=None, usage=None)
        chat = OpenAICompatibleChat("deepseek-flash", api_key="unused", base_url="http://unused.invalid")
        create = mock.Mock(side_effect=[empty, reply])
        chat._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with mock.patch("memocore.llm.time.sleep"):
            self.assertEqual(chat.complete([], max_tokens=5).text, "Paris")
            self.assertEqual(create.call_count, 2)
            create.side_effect = [empty] * 3
            with self.assertRaises(RuntimeError):
                chat.complete([], max_tokens=5)

    def test_a_truncated_call_and_its_retry_are_both_counted(self):
        class Scripted:
            model = "deepseek-flash"

            def __init__(self, replies):
                self.replies = list(replies)

            def complete(self, messages, *, max_tokens, temperature=0.0):
                return self.replies.pop(0)

        at = TUESDAY_PEAK.isoformat()
        cut = Completion("", Usage(1_000, 2_048, 0, 2_048), 1.0, finish_reason="length", called_at=at)
        done = Completion("Paris", Usage(1_000, 3_000, 400, 2_900), 1.0, finish_reason="stop", called_at=at)
        completion, spend = _complete(Scripted([cut, done]), [], 2_048)
        self.assertEqual(completion.text, "Paris")
        self.assertEqual((spend["prompt_tokens"], spend["completion_tokens"], spend["reasoning_tokens"]), (2_000, 5_048, 4_948))
        expected = call_cost("deepseek-flash", 1_000, 0, 2_048, TUESDAY_PEAK) + call_cost("deepseek-flash", 1_000, 400, 3_000, TUESDAY_PEAK)
        self.assertAlmostEqual(spend["cost_cny"], expected, places=6)
        cached = Completion("Paris", Usage(1_000, 30), 0.0, from_cache=True, finish_reason="stop")
        self.assertEqual(_complete(Scripted([cached]), [], 2_048)[1]["cost_cny"], 0)  # paid by an earlier run

    def test_v4_puts_the_fixed_text_first(self):
        one = answer_messages(Conversation("a", "Alice", "Bob", ()), "[D1:1] Alice: hi", "Who?", "v4")
        two = answer_messages(Conversation("b", "Eve", "Sam", ()), "[D9:9] Sam: yo", "When?", "v4")
        shared = one[1]["content"].split("Below is memory")[0]
        self.assertTrue(two[1]["content"].startswith(shared))
        self.assertEqual(one[0], two[0])  # the system message too
        rules = ANSWER_TEMPLATES["v3"].split("Instructions:\n")[1].split("\nQuestion:")[0]
        self.assertIn(rules.replace("the memory above", "the memory below"), ANSWER_TEMPLATES["v4"])


class SpendReportTest(unittest.TestCase):
    def test_runs_are_charged_once_per_paid_call(self):
        settings = temp_settings(self)
        prediction = {"qa_id": "q1", "question": "Q", "gold": "G", "prediction": "P"}
        for name, judge_cached in (("first", False), ("second", True)):
            run = settings.runs_dir / name
            run.mkdir(parents=True)
            (run / "run.json").write_text(json.dumps({"answer_model": "deepseek-flash"}), encoding="utf-8")
            rows = [{**prediction, "prompt_tokens": 1_000_000, "completion_tokens": 0, "from_cache": judge_cached}]
            (run / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            verdict = {"qa_id": "q1", "label": "CORRECT", "judge_model": "deepseek-v4-pro", "prompt_tokens": 0,
                       "completion_tokens": 1_000_000}
            (run / "judge_strict.jsonl").write_text(json.dumps(verdict) + "\n", encoding="utf-8")
        reports = {r["run"]: r for r in spend_report(settings)}
        self.assertEqual((reports["first"]["low"], reports["first"]["high"]), (14.5, 29.0))  # 2 + 27 at peak
        self.assertEqual(reports["second"]["high"], 0)  # same answer from the cache, same judge request
        self.assertIn("¥14.50～29.00", render_spend(list(reports.values())))
