"""Paired comparison of two runs on the questions both answered."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from memocore.config import Settings
from memocore.jsonl import read_jsonl
from memocore.judge import JUDGE_KINDS
from memocore.runner import PREDICTIONS, existing_run, judge_path
from memocore.stats import paired_bootstrap

METRICS = ("f1", *(f"judge_{kind}" for kind in JUDGE_KINDS))


def _per_question(run_dir: Path, metric: str) -> dict[str, tuple[str, float]]:
    predictions = {row["qa_id"]: row for row in read_jsonl(run_dir / PREDICTIONS)}
    if metric == "f1":
        return {qa_id: (row["category_name"], float(row["f1"])) for qa_id, row in predictions.items()}
    verdicts = read_jsonl(judge_path(run_dir, metric.removeprefix("judge_")))
    return {
        row["qa_id"]: (predictions[row["qa_id"]]["category_name"], 1.0 if row["correct"] else 0.0)
        for row in verdicts
        if row["qa_id"] in predictions
    }


def compare_runs(settings: Settings, baseline: str, candidate: str, metric: str = "judge_strict") -> dict[str, Any]:
    # judge_<kind>-<reasoning> compares verdicts of a judge run at another thinking level.
    if metric not in METRICS and not (metric.startswith("judge_") and metric.removeprefix("judge_").split("-")[0] in JUDGE_KINDS):
        raise ValueError(f"unknown metric {metric!r}; choose from {', '.join(METRICS)}")
    a = _per_question(existing_run(settings, baseline), metric)
    b = _per_question(existing_run(settings, candidate), metric)
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError(f"运行 {baseline!r} 和 {candidate!r} 没有共同的已评测题目（指标 {metric}）")
    groups: dict[str, list[str]] = defaultdict(list)
    for qa_id in common:
        groups["overall"].append(qa_id)
        groups[a[qa_id][0]].append(qa_id)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "metric": metric,
        "groups": {
            name: {
                **paired_bootstrap([a[q][1] for q in ids], [b[q][1] for q in ids]),
                # For 0/1 metrics: questions only one side got right.
                "candidate_only": sum(b[q][1] > a[q][1] for q in ids),
                "baseline_only": sum(a[q][1] > b[q][1] for q in ids),
            }
            for name, ids in groups.items()
        },
    }


def render_comparison(result: dict[str, Any]) -> str:
    def pct(value: float | None) -> str:
        return "—" if value is None else f"{value * 100:.1f}%"

    lines = [
        f"# 配对对比：{result['candidate']} vs {result['baseline']}（{result['metric']}）",
        "",
        "差值 = 候选 − 基线，只比较两边都评测过的题；置信区间与 p 值来自按题配对的 bootstrap（2000 次）。",
        "",
        "| 题型 | 题数 | 基线 | 候选 | 差值 | 95% 置信区间 | p 值 | 仅候选答对 | 仅基线答对 |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for name, g in result["groups"].items():
        ci = "—" if g["ci"] is None else f"[{g['ci'][0] * 100:+.1f}, {g['ci'][1] * 100:+.1f}]"
        diff = "—" if g["diff"] is None else f"{g['diff'] * 100:+.1f}"
        p_value = "—" if g["p_value"] is None else f"{g['p_value']:.3f}"
        lines.append(
            f"| {name} | {g['n']} | {pct(g['baseline'])} | {pct(g['candidate'])} | {diff} | {ci} | {p_value} | "
            f"{g['candidate_only']} | {g['baseline_only']} |"
        )
    return "\n".join(lines) + "\n"
