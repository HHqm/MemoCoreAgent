import json
import unittest

from memocore.audit import CORRECT, UNSURE, WRONG, export_audit, label_audit, read_audit, score_audit
from memocore.embeddings import HashingEmbedder
from memocore.jsonl import read_jsonl
from memocore.llm import MockChat
from memocore.runner import PREDICTIONS, RunConfig, judge_path, run_predictions
from tests.helpers import temp_settings

ALL_CATEGORIES = (1, 2, 3, 4, 5)


def _script(*answers):
    """Stand-in for input(): replays the answers, then behaves like a closed terminal."""
    pending = list(answers)

    def ask(_prompt):
        if not pending:
            raise EOFError
        return pending.pop(0)

    return ask


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.settings = temp_settings(self)
        for name in ("a", "b"):
            cfg = RunConfig(system="naive_rag", run_name=name, categories=ALL_CATEGORIES, top_k=2, mock=True)
            run_predictions(cfg, self.settings, chat=MockChat(), embedder=HashingEmbedder(), progress=False)
            qa_ids = [row["qa_id"] for row in read_jsonl(self.settings.runs_dir / name / PREDICTIONS)]
            verdicts = [{"qa_id": qa_id, "label": WRONG, "judge_model": "m", "reasoning": "stale"} for qa_id in qa_ids]
            verdicts[-1]["judge_model"] = "rule"
            # The first question was judged again; the later line must win.
            verdicts.append({"qa_id": qa_ids[0], "label": CORRECT, "judge_model": "m", "reasoning": "re-judged"})
            judge_path(self.settings.runs_dir / name, "strict").write_text(
                "".join(json.dumps(verdict) + "\n" for verdict in verdicts), encoding="utf-8"
            )
        self.qa_ids = qa_ids
        self.path = export_audit(["a", "b"], "strict", self.settings, n=10)

    def test_export_is_blind_and_spreads_questions_across_runs(self):
        rows = read_audit(self.path)
        self.assertEqual(len(rows), len(self.qa_ids) - 1)  # the rule-decided question is left out
        self.assertNotIn(self.qa_ids[-1], {row["qa_id"] for row in rows})
        self.assertEqual(len({row["qa_id"] for row in rows}), len(rows))
        self.assertEqual({row["run"] for row in rows}, {"a", "b"})
        self.assertTrue(all(CORRECT not in row.values() and WRONG not in row.values() for row in rows))
        violin = next(row for row in rows if row["question"] == "What instrument is Bob learning?")
        self.assertIn("[D1:2] Bob:", violin["evidence"])
        with self.assertRaises(FileExistsError):
            export_audit(["a", "b"], "strict", self.settings, n=10)
        everything = export_audit(["a"], "strict", self.settings, include_rule=True, out=self.path.with_name("all.csv"))
        self.assertEqual(len(read_audit(everything)), len(self.qa_ids))

    def test_label_resumes_and_score_joins_the_verdicts(self):
        shown = []
        counts = label_audit(self.path, ask=_script("1", "x", "0", "?", "gold looks off", "b", "", "q"), show=shown.append)
        self.assertEqual((counts[CORRECT], counts[WRONG], counts[UNSURE]), (1, 1, 1))
        self.assertIn("请输入 1、0、?、b 或 q", shown)
        shown.clear()
        label_audit(self.path, ask=_script(), show=shown.append)  # picks up at the first unlabeled question
        self.assertTrue(any("第 4/" in line for line in shown))

        rows = read_audit(self.path)
        result = score_audit(self.path, self.settings)
        self.assertEqual((result["labeled"], result["unsure"]), (2, 1))
        self.assertEqual(result["unsure_items"][0]["note"], "gold looks off")
        judge = [CORRECT if row["qa_id"] == self.qa_ids[0] else WRONG for row in rows[:2]]
        agree = sum(j == h for j, h in zip(judge, (CORRECT, WRONG)))
        self.assertAlmostEqual(result["agreement"], agree / 2)
        self.assertEqual(result["judge_too_lenient"] + result["judge_too_strict"], 2 - agree)
        self.assertEqual(len(result["disagreements"]), 2 - agree)
        self.assertEqual(sum(group["n"] for group in result["by_run"].values()), 2)
