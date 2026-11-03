"""Retrieval-only evaluation: how much gold evidence a read path puts in context, with no LLM calls.

It costs nothing to run, so retrieval settings can be tuned before any API money is spent, and it
later separates "the memory never surfaced the fact" from "the model failed to use it".
"""

from __future__ import annotations

import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from tqdm import tqdm

from memocore.config import Settings
from memocore.datasets.locomo import CATEGORY_NAMES, DEFAULT_CATEGORIES, load_locomo
from memocore.embeddings import Embedder
from memocore.metrics import answer_visible, evidence_recall
from memocore.retrieval.rerank import CrossEncoderReranker, Reranker
from memocore.runner import RunConfig, build_embedder, build_reranker, select_questions
from memocore.systems import HybridOptions, HybridRAGSystem, NaiveRAGSystem

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
            rows.append(
                {
                    "chunk_turns": size,
                    "top_k": k,
                    "context_turns": fmean(context_turns[k]) if context_turns[k] else 0.0,
                    **_recall_stats(recalls[k]),
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


def _recall_stats(by_category: dict[int, list[float]]) -> dict[str, Any]:
    everything = [value for values in by_category.values() for value in values]
    return {
        "recall": fmean(everything) if everything else None,
        "all_evidence_rate": fmean(value == 1.0 for value in everything) if everything else None,
        "by_category": {
            CATEGORY_NAMES.get(category, str(category)): fmean(values) for category, values in sorted(by_category.items())
        },
    }


# Each configuration adds or swaps one stage, so adjacent rows isolate a single change at the same budget.
# A first round also tried BM25 alone, neighbour expansion and a per-session cap (docs/results).
ABLATION_CONFIGS: dict[str, HybridOptions] = {
    "dense": HybridOptions(use_bm25=False),
    "dense+bm25(rrf)": HybridOptions(),
    "dense+rerank20": HybridOptions(use_bm25=False, rerank_candidates=20),
    "dense+rerank50": HybridOptions(use_bm25=False, rerank_candidates=50),
    "dense+bm25(rrf)+rerank50": HybridOptions(rerank_candidates=50),
    "dense+bm25(pool)+rerank50": HybridOptions(fusion="pool", rerank_candidates=50),
    "dense+rerank50+time": HybridOptions(use_bm25=False, rerank_candidates=50, time_anchors=True),
    "dense+bm25(pool)+rerank50+time": HybridOptions(fusion="pool", rerank_candidates=50, time_anchors=True),
    "dense+bm25(pool)+rerank50+time+feedback": HybridOptions(
        fusion="pool", rerank_candidates=50, time_anchors=True, feedback_turns=2
    ),
    # Rounds 2 and 3 of docs/results/optimization_rounds.md; the last row is the default read path.
    "...+feedback+time-query": HybridOptions(
        fusion="pool", rerank_candidates=50, time_anchors=True, feedback_turns=2, time_candidates=20
    ),
    "...+feedback+time-query+speaker": HybridOptions(
        fusion="pool", rerank_candidates=50, time_anchors=True, feedback_turns=2, time_candidates=20, speaker_boost=0.1
    ),
}


def retrieval_ablation(
    settings: Settings,
    *,
    budgets: Sequence[int] = (5, 10, 20),
    configs: dict[str, HybridOptions] | None = None,
    categories: Sequence[int] = DEFAULT_CATEGORIES,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    progress: bool = True,
    out_name: str = "retrieval_ablation",
) -> dict[str, Any]:
    """Evidence recall of each read-path configuration at the same turn budgets. No LLM calls."""
    configs = configs or ABLATION_CONFIGS
    budgets = sorted(set(budgets))
    selected = select_questions(
        load_locomo(settings.locomo_path),
        RunConfig(system="hybrid_rag", run_name="retrieval-ablation", categories=tuple(categories)),
    )
    embedder = embedder or build_embedder(settings)
    if reranker is None and any(options.rerank_candidates for options in configs.values()):
        reranker = build_reranker(settings)

    rows, rank_ms = [], {}
    for name, options in configs.items():
        recalls: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        visible: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        chars: dict[int, list[int]] = defaultdict(list)
        rank_s, queries = 0.0, 0
        for sample, questions in tqdm(selected, desc=name, disable=not progress):
            system = HybridRAGSystem(
                embedder if options.use_dense else None,
                reranker if options.rerank_candidates else None,
                top_k=budgets[-1],
                options=options,
            )
            system.ingest(sample.conversation)
            start = time.perf_counter()
            rankings = system.rank([qa.question for qa in questions])
            rank_s += time.perf_counter() - start
            queries += len(questions)
            for qa, ranked in zip(questions, rankings):
                for budget in budgets:
                    context = system.render(system.select(ranked, budget))
                    chars[budget].append(len(context.text))
                    recall = evidence_recall(qa.evidence, context.turn_ids)
                    if recall is not None:
                        recalls[budget][qa.category].append(recall)
                    seen = answer_visible(qa.gold, context.text, qa.category)
                    if seen is not None:
                        visible[budget][qa.category].append(1.0 if seen else 0.0)
        for budget in budgets:
            shown = _recall_stats(visible[budget])
            rows.append(
                {
                    "config": name,
                    "budget": budget,
                    **_recall_stats(recalls[budget]),
                    "answer_visible": shown["recall"],
                    "answer_visible_by_category": shown["by_category"],
                    "context_chars": fmean(chars[budget]) if chars[budget] else 0.0,
                }
            )
        rank_ms[name] = round(1000 * rank_s / max(queries, 1), 2)

    result: dict[str, Any] = {
        "embedding_model": embedder.name,
        "reranker_model": reranker.name if reranker else None,
        "categories": list(categories),
        "questions": sum(len(questions) for _, questions in selected),
        "budgets": budgets,
        "rows": rows,
        "rank_ms_per_question": rank_ms,
    }
    if isinstance(reranker, CrossEncoderReranker) and reranker.pairs_scored:
        result["reranker_pairs_per_second"] = round(reranker.pairs_scored / reranker.model_seconds, 1)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    (settings.runs_dir / f"{out_name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (settings.runs_dir / f"{out_name}.md").write_text(render_ablation(result), encoding="utf-8")
    return result


def _short(model: str | None) -> str:
    return "—" if not model else model.replace("\\", "/").rsplit("/", 1)[-1]


def render_ablation(result: dict[str, Any]) -> str:
    rows = result["rows"]
    category_names = list(rows[0]["by_category"]) if rows else []
    lines = [
        "# 读取路径消融：同一上下文预算下的证据召回（不调用 LLM）",
        "",
        f"向量模型 `{_short(result['embedding_model'])}` · 重排模型 `{_short(result['reranker_model'])}` · "
        f"共 {result['questions']} 题 · 预算 = 放进上下文的对话轮数",
        "",
        "答案可见 = 标准答案的每个词都出现在上下文里（不调用 LLM 的代理指标，主要用来衡量时间锚定）。",
    ]
    for budget in result["budgets"]:
        at_budget = [row for row in rows if row["budget"] == budget]
        base = next((row["recall"] for row in at_budget if row["config"] == "dense"), None)
        lines += [
            "",
            f"## 预算 {budget} 轮",
            "",
            "| 配置 | 证据召回 | 比 dense | 证据全部召回的题占比 | "
            + " | ".join(category_names)
            + " | 答案可见 | temporal 答案可见 | 平均上下文字符 |",
            "|---|---:|---:|---:|" + "---:|" * len(category_names) + "---:|---:|---:|",
        ]
        for row in at_budget:
            delta = "—" if base is None or row["recall"] is None else f"{(row['recall'] - base) * 100:+.1f}"
            cells = " | ".join(_pct(row["by_category"].get(name)) for name in category_names)
            temporal_visible = row.get("answer_visible_by_category", {}).get("temporal")
            lines.append(
                f"| {row['config']} | {_pct(row['recall'])} | {delta} | {_pct(row['all_evidence_rate'])} | {cells} | "
                f"{_pct(row.get('answer_visible'))} | {_pct(temporal_visible)} | {row.get('context_chars', 0):,.0f} |"
            )
    lines += ["", "单题排序耗时（毫秒，向量与重排分数命中磁盘缓存时会更快）："]
    lines += [f"- {name}：{ms}" for name, ms in result["rank_ms_per_question"].items()]
    if result.get("reranker_pairs_per_second"):
        per_question = 50 / result["reranker_pairs_per_second"] * 1000
        lines.append(
            f"- 重排模型实际吞吐 {result['reranker_pairs_per_second']} 对/秒，"
            f"即 50 个候选约 {per_question:.0f} ms/题（不含缓存）"
        )
    return "\n".join(lines) + "\n"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def render_sweep(result: dict[str, Any]) -> str:
    category_names = list(result["rows"][0]["by_category"]) if result["rows"] else []
    lines = [
        "# 朴素 RAG 检索召回（不调用 LLM）",
        "",
        f"向量模型 `{_short(result['embedding_model'])}` · 题型 "
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


REPLAY_JSON = "memory_replay.json"


def memory_replay(
    settings: Settings,
    *,
    budget: int = 20,
    categories: Sequence[int] = DEFAULT_CATEGORIES,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Write LoCoMo into a MemoryStore one message at a time, recalling after every session as an
    agent would, then check every question retrieves exactly the turns the evaluated HybridRAGSystem
    picks. The recalls between sessions mean the questions run on an index grown write by write
    (MemoryStore extends it rather than rebuilding). No LLM calls."""
    from memocore.store import DEFAULT_OPTIONS, MemoryStore, parse_session_time

    selected = select_questions(
        load_locomo(settings.locomo_path),
        RunConfig(system="hybrid_rag", run_name="memory-replay", categories=tuple(categories)),
    )
    embedder = embedder or build_embedder(settings)
    reranker = reranker or build_reranker(settings)
    recalls: dict[int, list[float]] = defaultdict(list)
    identical = turns = sessions = 0
    add_s = search_s = 0.0
    mismatched: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "replay.sqlite"
        store = MemoryStore(path, embedder, reranker, budget=budget)
        for sample, questions in tqdm(selected, desc="memory-replay", disable=not progress):
            conversation = sample.conversation
            for session in conversation.sessions:
                at = parse_session_time(session.date_time)
                start = time.perf_counter()
                for turn in session.turns:
                    memory = store.add(
                        conversation.sample_id, turn.text, speaker=turn.speaker, at=at,
                        session=f"locomo-{session.index}", image=turn.image_caption,
                    )
                    if memory.id != turn.dia_id:
                        raise AssertionError(f"store numbered {turn.dia_id} as {memory.id}")
                    turns += 1
                add_s += time.perf_counter() - start
                store.search(conversation.sample_id, session.turns[-1].text)
                sessions += 1

            reference = HybridRAGSystem(embedder, reranker, top_k=budget, options=DEFAULT_OPTIONS)
            reference.ingest(conversation)
            rankings = reference.rank([qa.question for qa in questions])
            for qa, ranked in zip(questions, rankings):
                expected = set(reference.turn_ids(reference.select(ranked, budget)))
                start = time.perf_counter()
                got = [memory.id for memory in store.search(conversation.sample_id, qa.question).memories]
                search_s += time.perf_counter() - start
                if set(got) == expected:
                    identical += 1
                else:
                    mismatched.append(qa.qa_id)
                recall = evidence_recall(qa.evidence, got)
                if recall is not None:
                    recalls[qa.category].append(recall)
        db_bytes = path.stat().st_size
        store.close()

    questions = sum(len(qs) for _, qs in selected)
    result = {
        "embedding_model": embedder.name,
        "reranker_model": reranker.name,
        "budget": budget,
        "turns": turns,
        "sessions_recalled_after": sessions,
        "questions": questions,
        "identical_to_evaluated_system": identical,
        "mismatched": mismatched[:20],
        **_recall_stats(recalls),
        "add_ms_per_turn": round(1000 * add_s / max(turns, 1), 2),
        "search_ms_per_question": round(1000 * search_s / max(questions, 1), 2),
        "db_bytes": db_bytes,
    }
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    (settings.runs_dir / REPLAY_JSON).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
