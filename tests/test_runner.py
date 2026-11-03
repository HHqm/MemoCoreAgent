import json
import unittest

from memocore.embeddings import HashingEmbedder
from memocore.jsonl import read_jsonl
from memocore.llm import Completion, MockChat, ResponseCache, Usage
from memocore.runner import (
    PREDICTIONS,
    RunConfig,
    RunMismatchError,
    judge_path,
    run_judge,
    run_predictions,
    select_questions,
    summarize,
)
from memocore.datasets.locomo import load_locomo
from tests.helpers import temp_settings

ALL_CATEGORIES = (1, 2, 3, 4, 5)


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.settings = temp_settings(self)
        self.cfg = RunConfig(system="naive_rag", run_name="t", categories=ALL_CATEGORIES, top_k=2, mock=True)

    def _run(self, cfg=None, chat=None):
        return run_predictions(
            cfg or self.cfg, self.settings, chat=chat or MockChat(), embedder=HashingEmbedder(), progress=False
        )

    def test_predictions_have_scores_and_retrieval_diagnostics(self):
        rows = {row["qa_id"]: row for row in read_jsonl(self._run())}
        self.assertEqual(len(rows), 7)
        adversarial = rows["conv-test#q0004"]
        self.assertEqual(adversarial["f1"], 1.0)  # the mock always abstains
        violin = rows["conv-test#q0001"]
        self.assertEqual(violin["f1"], 0.0)
        self.assertEqual(violin["context_turns"], 2)
        self.assertIsNotNone(violin["evidence_recall"])

    def test_citing_prompt_strips_and_scores_citations(self):
        def responder(messages):
            # The citing prompt must show turn ids in the context, or the model has nothing to cite.
            return "The violin [D1:2]" if "[D1:2] Bob:" in messages[-1]["content"] else "no ids"

        cfg = RunConfig(system="full_context", run_name="cite", prompt="v3", mock=True)
        rows = {row["qa_id"]: row for row in read_jsonl(self._run(cfg=cfg, chat=MockChat(responder=responder)))}
        violin = rows["conv-test#q0001"]
        self.assertEqual(violin["prediction"], "The violin")
        self.assertEqual(violin["raw_prediction"], "The violin [D1:2]")
        self.assertEqual(violin["cited_ids"], ["D1:2"])
        self.assertEqual((violin["citation_precision"], violin["citation_in_context"]), (1.0, 1.0))
        uncited = [row for row in rows.values() if not row["cited_ids"]]
        self.assertEqual({row["sample_id"] for row in uncited}, {"conv-two"})  # its context has no D1:2
        self.assertIsNone(uncited[0]["citation_precision"])
        cited = summarize("cite", self.settings)["groups"]["overall"]["cited"]
        self.assertAlmostEqual(cited, 1 - len(uncited) / len(rows))

    def test_resume_skips_answered_questions(self):
        self._run()
        second = MockChat()
        self._run(chat=second)
        self.assertEqual(second.calls, 0)
        self.assertEqual(len(list(read_jsonl(self.settings.runs_dir / "t" / PREDICTIONS))), 7)

    def test_resume_with_different_settings_is_refused(self):
        self._run()
        with self.assertRaises(RunMismatchError):
            self._run(cfg=RunConfig(system="naive_rag", run_name="t", categories=ALL_CATEGORIES, top_k=3, mock=True))

    def test_category_filter_and_seeded_sampling(self):
        samples = load_locomo(self.settings.locomo_path)
        default = select_questions(samples, RunConfig(system="full_context", run_name="x"))
        self.assertNotIn(5, {qa.category for _, questions in default for qa in questions})
        cfg = RunConfig(system="full_context", run_name="x", categories=ALL_CATEGORIES, max_questions=3, seed=1)
        first = [qa.qa_id for _, qs in select_questions(samples, cfg) for qa in qs]
        again = [qa.qa_id for _, qs in select_questions(samples, cfg) for qa in qs]
        self.assertEqual(first, again)
        self.assertEqual(len(first), 4)  # 3 from conv-test + the single question of conv-two
        with self.assertRaises(ValueError):
            select_questions(samples, RunConfig(system="full_context", run_name="x", conversations=("conv-missing",)))
        picked = select_questions(samples, RunConfig(system="full_context", run_name="x", qa_ids=(first[0],), max_questions=3))
        self.assertEqual([qa.qa_id for _, qs in picked for qa in qs], [first[0]])
        with self.assertRaises(ValueError):
            select_questions(samples, RunConfig(system="full_context", run_name="x", qa_ids=("conv-test#q9999",)))

    def test_judge_at_another_thinking_level_writes_its_own_verdicts(self):
        from memocore.compare import compare_runs

        self._run()
        run_judge("t", "strict", self.settings, progress=False)
        variant = run_judge("t", "strict", self.settings, progress=False, reasoning="none")
        self.assertEqual(variant.name, "judge_strict-none.jsonl")
        self.assertEqual(len(list(read_jsonl(variant))), 7)
        self.assertEqual(len(list(read_jsonl(judge_path(self.settings.runs_dir / "t", "strict")))), 7)
        self.assertEqual(compare_runs(self.settings, "t", "t", "judge_strict-none")["groups"]["overall"]["n"], 7)
        with self.assertRaises(ValueError):
            compare_runs(self.settings, "t", "t", "judge_bogus-none")

    def test_a_failed_call_does_not_lose_the_calls_that_finished(self):
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace

        from memocore.runner import _write_all

        written = []

        def call(i):
            if i == 0:
                raise RuntimeError("no choices")
            return i

        writer, bar = SimpleNamespace(write=written.append), SimpleNamespace(update=lambda n: None)
        with ThreadPoolExecutor(max_workers=4) as pool, self.assertRaises(RuntimeError):
            _write_all([pool.submit(call, i) for i in range(6)], writer, bar)
        # Everything but the failed call is recorded, so a rerun repeats only that one.
        self.assertEqual(sorted(written), [1, 2, 3, 4, 5])

    def test_judge_summary(self):
        self._run()
        run_judge("t", "strict", self.settings, progress=False)
        run_judge("t", "strict", self.settings, progress=False)  # resumes: no duplicates
        self.assertEqual(len(list(read_jsonl(judge_path(self.settings.runs_dir / "t", "strict")))), 7)

        summary = summarize("t", self.settings)
        overall = summary["groups"]["overall"]
        self.assertEqual(overall["n"], 7)
        self.assertEqual(overall["judge_strict_n"], 7)
        self.assertIn("adversarial", summary["groups"])
        markdown = (self.settings.runs_dir / "t" / "summary.md").read_text(encoding="utf-8")
        self.assertIn("--mock", markdown)

    def test_full_context_run_records_whole_conversation(self):
        cfg = RunConfig(system="full_context", run_name="fc", categories=(4,), mock=True)
        rows = list(read_jsonl(run_predictions(cfg, self.settings, chat=MockChat(), progress=False)))
        self.assertEqual({row["context_turns"] for row in rows}, {5, 1})
        self.assertTrue(all(row["evidence_recall"] == 1.0 for row in rows))
        info = json.loads((self.settings.runs_dir / "fc" / "run.json").read_text(encoding="utf-8"))
        self.assertIsNone(info["embedding_model"])


class ResponseCacheTest(unittest.TestCase):
    def test_round_trip_marks_hits(self):
        settings = temp_settings(self)
        cache = ResponseCache(settings.cache_path)
        key = ResponseCache.key("m", [{"role": "user", "content": "hi"}], 10, 0.0)
        self.assertIsNone(cache.get(key))
        cache.put(key, Completion("hello", Usage(5, 1, 3), latency_s=0.8))
        hit = cache.get(key)
        self.assertEqual(hit.text, "hello")
        self.assertEqual(hit.usage.cached_prompt_tokens, 3)
        self.assertEqual(hit.latency_s, 0.8)
        self.assertTrue(hit.from_cache)


if __name__ == "__main__":
    unittest.main()
