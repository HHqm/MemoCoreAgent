"""Retrieval-only evaluation: how much gold evidence naive RAG puts in context, with no LLM calls.

It costs nothing to run, so retrieval settings can be tuned before any API money is spent, and it
later separates "the memory never surfaced the fact" from "the model failed to use it".
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from statistics import fmean
from typing import Any, Sequence

from tqdm import tqdm

from memocore.config import Settings
from memocore.datasets.locomo import CATEGORY_NAMES, DEFAULT_CATEGORIES, load_locomo
from memocore.embeddings import Embedder
from memocore.metrics import evidence_recall
from memocore.runner import RunConfig, build_embedder, select_questions
from memocore.systems import NaiveRAGSystem

SWEEP_JSON = "retrieval_sweep.json"
SWEEP_MD = "retrieval_sweep.md"


def retrieval_sweep(
    settings: Settings,
    *,
    top_ks: Sequence[int],
    chunk_turns: Sequence[int],
    categories: Sequence[int] = DEFAULT_CATEGORIES,
    embedder: Embedder | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    selected = select_questions(
        load_locomo(settings.locomo_path),
        RunConfig(system="naive_rag", run_name="retrieval-sweep", categories=tuple(categories)),
    )
    embedder = embedder or build_embedder(settings)
    ks = sorted(set(top_ks))
    evidence_total = evidence_missing = 0
    for sample, questions in selected:
        known = set(sample.conversation.turn_ids)
        for qa in questions:
            evidence_total += len(qa.evidence)
            evidence_missing += sum(turn_id not in known for turn_id in qa.evidence)

    rows, timing = [], []
    for size in chunk_turns:
        recalls: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        context_turns: dict[int, list[int]] = defaultdict(list)
        ingest_s = query_s = 0.0
        queries = 0
        for sample, questions in tqdm(selected, desc=f"chunk_turns={size}", disable=not progress):
            system = NaiveRAGSystem(embedder, top_k=ks[-1], chunk_turns=size)
            start = time.perf_counter()
            system.ingest(sample.conversation)
            ingest_s += time.perf_counter() - start
            start = time.perf_counter()
            rankings = system.ranked_chunks([qa.question for qa in questions], ks[-1])
            query_s += time.perf_counter() - start
            queries += len(questions)
            for qa, ranked in zip(questions, rankings):
                for k in ks:
                    ids = [turn_id for chunk in ranked[:k] for turn_id in chunk]
                    context_turns[k].append(len(ids))
                    recall = evidence_recall(qa.evidence, ids)
                    if recall is not None:
                        recalls[k][qa.category].append(recall)
        for k in ks:
            everything = [value for values in recalls[k].values() for value in values]
            rows.append(
                {
                    "chunk_turns": size,
                    "top_k": k,
                    "context_turns": fmean(context_turns[k]) if context_turns[k] else 0.0,
                    "recall": fmean(everything) if everything else None,
                    "all_evidence_rate": fmean(value == 1.0 for value in everything) if everything else None,
                    "by_category": {
                        CATEGORY_NAMES.get(category, str(category)): fmean(values)
                        for category, values in sorted(recalls[k].items())
                    },
                }
            )
        timing.append(
            {"chunk_turns": size, "ingest_s": round(ingest_s, 2), "query_ms": round(1000 * query_s / max(queries, 1), 2)}
        )

    result = {
        "embedding_model": embedder.name,
        "categories": list(categories),
        "questions": sum(len(questions) for _, questions in selected),
        "evidence_ids": evidence_total,
        "evidence_ids_missing": evidence_missing,
        "rows": rows,
        "timing": timing,
    }
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    (settings.runs_dir / SWEEP_JSON).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (settings.runs_dir / SWEEP_MD).write_text(render_sweep(result), encoding="utf-8")
    return result


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def render_sweep(result: dict[str, Any]) -> str:
    category_names = list(result["rows"][0]["by_category"]) if result["rows"] else []
    lines = [
        "# 朴素 RAG 检索召回（不调用 LLM）",
        "",
        f"向量模型 `{result['embedding_model'].replace(chr(92), '/').rsplit('/', 1)[-1]}` · 题型 "
        f"{','.join(map(str, result['categories']))} · 共 {result['questions']} 题 · 证据 {result['evidence_ids']} 条，"
        f"其中 {result['evidence_ids_missing']} 条在对话里找不到（标注噪声，召回上限因此低于 100%）",
        "",
        "| 每块轮数 | top_k | 平均上下文轮数 | 证据召回 | 证据全部召回的题占比 | " + " | ".join(category_names) + " |",
        "|---:|---:|---:|---:|---:|" + "---:|" * len(category_names),
    ]
    for row in result["rows"]:
        cells = " | ".join(_pct(row["by_category"].get(name)) for name in category_names)
        lines.append(
            f"| {row['chunk_turns']} | {row['top_k']} | {row['context_turns']:.1f} | {_pct(row['recall'])} | "
            f"{_pct(row['all_evidence_rate'])} | {cells} |"
        )
    lines += ["", "耗时：" + "；".join(
        f"每块 {t['chunk_turns']} 轮时建索引共 {t['ingest_s']} s、单次检索 {t['query_ms']} ms" for t in result["timing"]
    )]
    return "\n".join(lines) + "\n"
