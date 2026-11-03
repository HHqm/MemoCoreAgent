"""How much the evaluations have cost so far, per run and in total, from the tokens each row recorded.

Rows written since per-call accounting was added carry their cost in yuan. Older rows only have token
counts and no call time, so their cost is given as a range: all off-peak to all at peak. A response
served from the local cache was paid for once, by whichever run made the call, so it is counted there:
answer rows say whether they came from the cache, and judge requests are counted once per distinct
(judge, question, gold, prediction), which is what the cache key covers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memocore.config import Settings
from memocore.jsonl import read_jsonl
from memocore.pricing import OFF_PEAK_FACTOR, call_cost

PREDICTIONS = "predictions.jsonl"
RUN_INFO = "run.json"


def _row_cost(model: str, row: dict[str, Any]) -> tuple[float, float]:
    """(lowest, highest) yuan for one row: exact when it was recorded, otherwise off-peak to peak."""
    if row.get("cost_cny") is not None:
        return row["cost_cny"], row["cost_cny"]
    peak = call_cost(model, row.get("prompt_tokens") or 0, row.get("cached_prompt_tokens") or 0, row.get("completion_tokens") or 0)
    if peak is None:
        return 0.0, 0.0
    return peak * OFF_PEAK_FACTOR, peak


def run_spend(run_dir: Path, seen_judgements: set[tuple[str, ...]]) -> dict[str, Any] | None:
    info = json.loads((run_dir / RUN_INFO).read_text(encoding="utf-8"))
    if info.get("mock"):
        return None
    stages: dict[str, dict[str, Any]] = {}
    answers = [row for row in read_jsonl(run_dir / PREDICTIONS) if not row.get("from_cache")]
    stages["answer"] = _stage(info["answer_model"], answers)
    questions = {row["qa_id"]: row for row in read_jsonl(run_dir / PREDICTIONS)}
    for path in sorted(run_dir.glob("judge_*.jsonl")):
        kind = path.stem.removeprefix("judge_")
        paid = []
        for row in read_jsonl(path):
            if row.get("judge_model") in (None, "rule") or row.get("from_cache"):
                continue
            asked = questions.get(row["qa_id"], {})
            key = (row["judge_model"], kind, asked.get("question", ""), asked.get("gold", ""), asked.get("prediction", ""))
            if key in seen_judgements and row.get("cost_cny") is None:
                continue  # an older row whose response came from the cache: paid by an earlier run
            seen_judgements.add(key)
            paid.append(row)
        models = {row["judge_model"] for row in paid}
        stages[f"judge[{kind}]"] = _stage(models.pop() if len(models) == 1 else "deepseek-v4-pro", paid)
    return {"run": run_dir.name, "stages": stages,
            "low": sum(s["low"] for s in stages.values()), "high": sum(s["high"] for s in stages.values())}


def _stage(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    low = high = 0.0
    for row in rows:
        row_low, row_high = _row_cost(model, row)
        low, high = low + row_low, high + row_high
    return {
        "model": model,
        "calls": len(rows),
        "prompt_tokens": sum(row.get("prompt_tokens") or 0 for row in rows),
        "cached_prompt_tokens": sum(row.get("cached_prompt_tokens") or 0 for row in rows),
        "completion_tokens": sum(row.get("completion_tokens") or 0 for row in rows),
        "low": low,
        "high": high,
    }


def spend_report(settings: Settings) -> list[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    # Oldest first, so a judge request repeated by a later run is charged to the run that made it.
    run_dirs = sorted(
        (path.parent for path in settings.runs_dir.glob(f"*/{RUN_INFO}")), key=lambda path: ((path / RUN_INFO).stat().st_mtime, path.name)
    )
    return [report for run_dir in run_dirs if (report := run_spend(run_dir, seen)) is not None]


def _yuan(low: float, high: float) -> str:
    return f"¥{high:.2f}" if abs(high - low) < 0.005 else f"¥{low:.2f}～{high:.2f}"


def render_spend(reports: list[dict[str, Any]]) -> str:
    lines = [
        "| 运行 | 作答调用 | 裁判调用 | 输入 token（命中缓存） | 输出 token | 花费 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        stages = report["stages"].values()
        judge_calls = sum(s["calls"] for name, s in report["stages"].items() if name != "answer")
        prompt = sum(s["prompt_tokens"] for s in stages)
        cached = sum(s["cached_prompt_tokens"] for s in stages)
        output = sum(s["completion_tokens"] for s in stages)
        lines.append(
            f"| {report['run']} | {report['stages']['answer']['calls']} | {judge_calls} | "
            f"{prompt:,}（{cached:,}） | {output:,} | {_yuan(report['low'], report['high'])} |"
        )
    low, high = sum(r["low"] for r in reports), sum(r["high"] for r in reports)
    lines += ["", f"合计 {_yuan(low, high)}。没有记录调用时间的旧记录给出区间：全部按低谷价到全部按高峰价。"]
    return "\n".join(lines) + "\n"
