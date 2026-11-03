"""Blind human audit of the judge: sample judged answers, label them, and measure agreement.

The sheet never shows the judge's verdict or which system wrote the answer, so the human label is
independent of both; the verdicts are joined back in only when scoring. Verdicts decided by rule
(exact matches and abstentions) are left out by default: they are right by construction, and
auditing them would only inflate the agreement rate.
"""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from textwrap import indent
from typing import Any, Callable, Sequence

from memocore.config import Settings
from memocore.datasets.locomo import Conversation, load_locomo
from memocore.jsonl import read_jsonl
from memocore.judge import JUDGE_KINDS
from memocore.runner import PREDICTIONS, existing_run, judge_path
from memocore.stats import cohen_kappa, wilson_ci

CORRECT, WRONG, UNSURE = "CORRECT", "WRONG", "UNSURE"
AUDIT_FIELDS = [
    "item", "question", "gold", "prediction", "evidence", "human_label", "note", "category", "run", "qa_id", "judge",
]
_HUMAN_LABELS = {
    "correct": CORRECT, "c": CORRECT, "1": CORRECT, "y": CORRECT, "yes": CORRECT, "对": CORRECT,
    "wrong": WRONG, "w": WRONG, "0": WRONG, "n": WRONG, "no": WRONG, "错": WRONG,
    "unsure": UNSURE, "u": UNSURE, "?": UNSURE, "？": UNSURE, "存疑": UNSURE,
}

RUBRIC = """判定标准和严格裁判相同，下面几条都满足才算"对"：
  1. 覆盖：包含标准答案要求的全部信息，列表题要列全
  2. 不矛盾：没有和标准答案冲突的内容
  3. 不越界：没有标准答案不支持的具体内容（多出的列表项、人名、日期、原因）；只是换个说法没关系
  4. 时间粒度：标准答案精确到某天时，只给月份或年份算错；同一天的其他说法算对
以标准答案为准。原文证据只用来帮你看懂题目；觉得标准答案本身有误、或者没法判断，就选 ? 并写一句备注。"""


def human_label(text: str) -> str | None:
    return _HUMAN_LABELS.get(text.strip().lower())


def latest_verdicts(run_dir: Path, kind: str) -> dict[str, dict[str, Any]]:
    # A re-judged question has several lines; the last one wins, as in the run summary.
    return {row["qa_id"]: row for row in read_jsonl(judge_path(run_dir, kind))}


def evidence_text(conversation: Conversation, ids: Sequence[str]) -> str:
    where = {turn.dia_id: (session.date_time, turn) for session in conversation.sessions for turn in session.turns}
    lines = []
    for dia_id in ids:
        if dia_id in where:
            date, turn = where[dia_id]
            lines.append(f"({date}) {turn.render(with_id=True)}")
        else:
            lines.append(f"[{dia_id}] (not in the conversation)")
    return "\n".join(lines)


def export_audit(
    run_names: Sequence[str],
    kind: str,
    settings: Settings,
    *,
    n: int = 100,
    seed: int = 0,
    include_rule: bool = False,
    out: Path | None = None,
) -> Path:
    """Sample n distinct questions across the runs, taking each run in turn, and write a sheet to label."""
    if kind not in JUDGE_KINDS:
        raise ValueError(f"unknown judge {kind!r}; choose from {', '.join(JUDGE_KINDS)}")
    eligible: dict[str, dict[str, dict[str, Any]]] = {}
    for run in run_names:
        run_dir = existing_run(settings, run)
        verdicts = latest_verdicts(run_dir, kind)
        if not verdicts:
            raise FileNotFoundError(f"运行 {run!r} 还没有 {kind} 裁判结果，请先执行 memocore judge。")
        predictions = {row["qa_id"]: row for row in read_jsonl(run_dir / PREDICTIONS)}
        eligible[run] = {
            qa_id: predictions[qa_id]
            for qa_id, verdict in verdicts.items()
            if verdict["label"] is not None and (include_rule or verdict.get("judge_model") != "rule")
        }
    path = Path(out) if out else settings.runs_dir / f"audit_{kind}.csv"
    if path.exists():
        # Never overwrite a file that may already hold hand-written labels.
        raise FileExistsError(f"{path} 已存在；如需重新抽样请先手动改名或删除。")

    rng = random.Random(seed)
    qa_ids = sorted({qa_id for rows in eligible.values() for qa_id in rows})
    rng.shuffle(qa_ids)
    chosen: list[tuple[str, dict[str, Any]]] = []
    for qa_id in qa_ids:
        if len(chosen) >= n:
            break
        # Each question appears once; rotating the runs gives every system about the same share.
        start = len(chosen) % len(run_names)
        for offset in range(len(run_names)):
            run = run_names[(start + offset) % len(run_names)]
            if qa_id in eligible[run]:
                chosen.append((run, eligible[run][qa_id]))
                break

    conversations = {sample.conversation.sample_id: sample.conversation for sample in load_locomo(settings.locomo_path)}
    rows = [
        {
            "item": index,
            "question": row["question"],
            "gold": row["gold"],
            "prediction": row["prediction"],
            "evidence": evidence_text(conversations[row["sample_id"]], row["evidence"]),
            "human_label": "",
            "note": "",
            "category": row["category_name"],
            "run": run,
            "qa_id": row["qa_id"],
            "judge": kind,
        }
        for index, (run, row) in enumerate(chosen, start=1)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_audit(path, rows)
    return path


def read_audit(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_audit(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    # utf-8-sig so Excel opens the sheet with the right encoding.
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def label_audit(path: Path, *, ask: Callable[[str], str] = input, show: Callable[[str], None] = print) -> Counter:
    """Label the sheet one question at a time in the terminal, saving after every answer."""
    rows = read_audit(path)
    position = next((i for i, row in enumerate(rows) if not human_label(row["human_label"])), len(rows))
    show(RUBRIC)
    while position < len(rows):
        row = rows[position]
        show(f"\n—— 第 {position + 1}/{len(rows)} 题（{row['category']}）——")
        show(f"问题：{row['question']}")
        show(f"标准答案：{row['gold']}")
        show(f"待评回答：{row['prediction'] or '（空）'}")
        if row["evidence"]:
            show("原文证据：\n" + indent(row["evidence"], "  "))
        current = human_label(row["human_label"])
        hint = f"（已标 {current}，回车保留）" if current else ""
        try:
            key = ask(f"1 对 / 0 错 / ? 存疑 / b 上一题 / q 退出{hint} > ").strip().lower()
            if key == "q":
                break
            if key == "b":
                position = max(position - 1, 0)
                continue
            if key == "" and current:
                position += 1
                continue
            label = human_label(key)
            if label is None:
                show("请输入 1、0、?、b 或 q")
                continue
            row["human_label"] = label
            row["note"] = ask("备注（可留空）> ").strip() if label == UNSURE else ""
        except (EOFError, KeyboardInterrupt):
            break
        write_audit(path, rows)
        position += 1
    counts = Counter(human_label(row["human_label"]) for row in rows)
    done = len(rows) - counts[None]
    show(
        f"\n已标 {done}/{len(rows)}（对 {counts[CORRECT]}，错 {counts[WRONG]}，存疑 {counts[UNSURE]}），已保存到 {path}。"
        f"\n查看一致率：memocore audit-score --csv {path}"
    )
    return counts


def _agreement(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    agree = sum(judge == human for judge, human in pairs)
    return {"n": len(pairs), "agreement": agree / len(pairs) if pairs else None}


def score_audit(path: Path, settings: Settings) -> dict[str, Any]:
    """Join the human labels with the judge's verdicts and report agreement and which way the judge errs."""
    rows = read_audit(path)
    verdicts: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    labeled, unsure, disagreements = [], [], []
    by_run: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_category: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        human = human_label(row["human_label"])
        if human is None:
            continue
        if human == UNSURE:
            unsure.append({"item": row["item"], "qa_id": row["qa_id"], "note": row["note"]})
            continue
        key = (row["run"], row["judge"])
        if key not in verdicts:
            verdicts[key] = latest_verdicts(existing_run(settings, row["run"]), row["judge"])
        verdict = verdicts[key][row["qa_id"]]
        pair = (verdict["label"], human)
        labeled.append(pair)
        by_run[row["run"]].append(pair)
        by_category[row["category"]].append(pair)
        if pair[0] != human:
            disagreements.append(
                {
                    "item": row["item"],
                    "run": row["run"],
                    "qa_id": row["qa_id"],
                    "question": row["question"],
                    "gold": row["gold"],
                    "prediction": row["prediction"],
                    "judge": verdict["label"],
                    "human": human,
                    "judge_reasoning": verdict.get("reasoning", ""),
                }
            )
    agree = sum(judge == human for judge, human in labeled)
    judge_accuracy = fmean(judge == CORRECT for judge, _ in labeled) if labeled else None
    human_accuracy = fmean(human == CORRECT for _, human in labeled) if labeled else None
    return {
        "total": len(rows),
        "labeled": len(labeled),
        "unsure": len(unsure),
        "agreement": agree / len(labeled) if labeled else None,
        "agreement_ci": wilson_ci(agree, len(labeled)),
        "kappa": cohen_kappa(labeled),
        # Accuracy on the same answers by each rater; the gap is how far the judge moves the headline number.
        "judge_accuracy": judge_accuracy,
        "human_accuracy": human_accuracy,
        "judge_too_lenient": sum(judge == CORRECT and human == WRONG for judge, human in labeled),
        "judge_too_strict": sum(judge == WRONG and human == CORRECT for judge, human in labeled),
        "by_run": {run: _agreement(pairs) for run, pairs in sorted(by_run.items())},
        "by_category": {category: _agreement(pairs) for category, pairs in sorted(by_category.items())},
        "disagreements": disagreements,
        "unsure_items": unsure,
    }
