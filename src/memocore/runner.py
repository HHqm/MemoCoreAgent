"""Run a memory system over LoCoMo, grade its answers and summarise the scores.

Every stage appends to JSONL and skips work already on disk, and every LLM call goes through the
response cache, so an interrupted run resumes where it stopped without paying twice.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

from memocore.config import Settings
from memocore.datasets.locomo import CATEGORY_NAMES, DEFAULT_CATEGORIES, Conversation, LocomoSample, QAItem, load_locomo
from memocore.embeddings import CachedEmbedder, Embedder, HashingEmbedder, SentenceTransformerEmbedder
from memocore.jsonl import JsonlWriter, read_jsonl
from memocore.judge import (
    JUDGE_KINDS,
    JUDGE_MAX_TOKENS,
    judge_messages,
    mock_judge_responder,
    parse_verdict,
    rule_verdict,
)
from memocore.llm import ChatModel, Completion, Message, MockChat, OpenAICompatibleChat, ResponseCache, Usage
from memocore.metrics import answer_visible, citation_scores, evidence_recall, locomo_f1
from memocore.pricing import call_cost, estimate_line
from memocore.prompts import (
    ANSWER_MAX_TOKENS,
    CITING_PROMPTS,
    DEFAULT_PROMPT,
    answer_messages,
    prompt_version,
    split_citations,
)
from memocore.retrieval.rerank import CrossEncoderReranker, LexicalReranker, Reranker
from memocore.stats import bootstrap_ci
from memocore.systems import NEEDS_EMBEDDER, HybridOptions, RetrievedContext, build_system

PREDICTIONS = "predictions.jsonl"
INGEST_LOG = "ingest.jsonl"
RUN_INFO = "run.json"
SUMMARY_JSON = "summary.json"
SUMMARY_MD = "summary.md"
# Full-context runs would otherwise store every turn id of the conversation for every question.
MAX_STORED_IDS = 100


class RunMismatchError(RuntimeError):
    """Resuming a run with settings that would make its rows incomparable."""


@dataclass(frozen=True)
class RunConfig:
    system: str
    run_name: str
    categories: tuple[int, ...] = DEFAULT_CATEGORIES
    conversations: tuple[str, ...] = ()
    max_conversations: int | None = None
    max_questions: int | None = None  # per conversation, drawn with a fixed seed
    seed: int = 0
    qa_ids: tuple[str, ...] = ()  # spot checks: only these questions, ignoring max_questions
    # Chosen from `memocore retrieval-sweep`: at a fixed context budget, one-turn chunks recall the
    # most evidence, and 20 of them (about 3% of a conversation) recall 76% of it with bge-m3.
    top_k: int = 20
    chunk_turns: int = 1  # naive_rag only
    # hybrid_rag only; see `memocore retrieval-ablation` for how each one moves evidence recall.
    fusion: str = "pool"
    rerank_candidates: int = 50
    neighbors: int = 0
    max_per_session: int | None = None
    time_anchors: bool = True
    # Chosen with `memocore retrieval-ablation`: +1.2 points evidence recall at 20 turns, +4.2 on multi-hop.
    feedback_turns: int = 2
    # Chosen in the five optimisation rounds (docs/results/optimization_rounds.md).
    context_before: int = 0
    time_candidates: int = 20
    speaker_boost: float = 0.1
    prompt: str = DEFAULT_PROMPT
    # Answer model's thinking: None keeps the API default (thinking, effort "high"); "none" turns it off.
    reasoning: str | None = None
    mock: bool = False

    @property
    def hybrid(self) -> HybridOptions:
        return HybridOptions(
            fusion=self.fusion,
            rerank_candidates=self.rerank_candidates,
            neighbors=self.neighbors,
            max_per_session=self.max_per_session,
            time_anchors=self.time_anchors,
            feedback_turns=self.feedback_turns,
            context_before=self.context_before,
            time_candidates=self.time_candidates,
            speaker_boost=self.speaker_boost,
        )


def judge_path(run_dir: Path, kind: str) -> Path:
    return run_dir / f"judge_{kind}.jsonl"


def build_embedder(settings: Settings) -> Embedder:
    inner = SentenceTransformerEmbedder(settings.embedding_model, device=settings.embedding_device)
    return CachedEmbedder(inner, settings.embedding_cache_path)


def build_reranker(settings: Settings) -> Reranker:
    return CrossEncoderReranker(
        settings.reranker_model, device=settings.embedding_device, cache_path=settings.rerank_cache_path
    )


def build_chat(
    settings: Settings, model: str, *, mock: bool, responder=None, reasoning: str | None = None
) -> ChatModel:
    if mock:
        return MockChat(responder=responder)
    return OpenAICompatibleChat(
        model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        cache=ResponseCache(settings.cache_path),
        reasoning=reasoning,
    )


def select_questions(samples: list[LocomoSample], cfg: RunConfig) -> list[tuple[LocomoSample, list[QAItem]]]:
    if cfg.conversations:
        unknown = sorted(set(cfg.conversations) - {s.conversation.sample_id for s in samples})
        if unknown:
            raise ValueError(f"unknown conversation ids: {', '.join(unknown)}")
        samples = [s for s in samples if s.conversation.sample_id in cfg.conversations]
    if cfg.max_conversations is not None:
        samples = samples[: cfg.max_conversations]
    selected = []
    for sample in samples:
        questions = [qa for qa in sample.qa if qa.category in cfg.categories]
        if cfg.qa_ids:
            questions = [qa for qa in questions if qa.qa_id in cfg.qa_ids]
        elif cfg.max_questions is not None and len(questions) > cfg.max_questions:
            # A seeded draw rather than the first N: LoCoMo lists its questions roughly grouped by type.
            rng = random.Random(f"{cfg.seed}:{sample.conversation.sample_id}")
            questions = sorted(rng.sample(questions, cfg.max_questions), key=lambda qa: qa.index)
        selected.append((sample, questions))
    missing = set(cfg.qa_ids) - {qa.qa_id for _, questions in selected for qa in questions}
    if missing:
        raise ValueError(f"unknown question ids (or outside --categories): {', '.join(sorted(missing))}")
    return selected


def run_predictions(
    cfg: RunConfig,
    settings: Settings,
    *,
    chat: ChatModel | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    workers: int = 8,
    progress: bool = True,
) -> Path:
    chat = chat or build_chat(settings, settings.answer_model, mock=cfg.mock, reasoning=cfg.reasoning)
    run_dir = settings.runs_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if cfg.system not in NEEDS_EMBEDDER:
        embedder = None
    elif embedder is None:
        embedder = HashingEmbedder() if cfg.mock else build_embedder(settings)
    if cfg.system != "hybrid_rag" or not cfg.rerank_candidates:
        reranker = None
    elif reranker is None:
        reranker = LexicalReranker() if cfg.mock else build_reranker(settings)
    _check_identity(run_dir, _identity(cfg, chat, embedder, reranker))

    selected = select_questions(load_locomo(settings.locomo_path), cfg)
    pred_path = run_dir / PREDICTIONS
    done = {row["qa_id"] for row in read_jsonl(pred_path)}
    total = sum(qa.qa_id not in done for _, questions in selected for qa in questions)
    if progress:
        print(f"回答[{cfg.system}]：已完成 {len(done)} 题，本次调用回答模型 {total} 次", file=sys.stderr)
        if not cfg.mock:
            kind = "answer_full_context" if cfg.system == "full_context" else "answer"
            if line := estimate_line(chat.model, total, kind):
                print(line, file=sys.stderr)

    with (
        JsonlWriter(pred_path) as predictions,
        JsonlWriter(run_dir / INGEST_LOG) as ingest_log,
        tqdm(total=total, desc=f"answer[{cfg.system}]", disable=not progress) as bar,
    ):
        for sample, questions in selected:
            todo = [qa for qa in questions if qa.qa_id not in done]
            if not todo:
                continue
            conversation = sample.conversation
            system = build_system(
                cfg.system,
                embedder=embedder,
                reranker=reranker,
                top_k=cfg.top_k,
                chunk_turns=cfg.chunk_turns,
                hybrid=cfg.hybrid,
                cite=cfg.prompt in CITING_PROMPTS,
            )
            start = time.perf_counter()
            system.ingest(conversation)
            ingest_log.write(
                {
                    "sample_id": conversation.sample_id,
                    "system": cfg.system,
                    "turns": len(conversation.turn_ids),
                    "ingest_s": round(time.perf_counter() - start, 4),
                }
            )
            # Retrieval runs serially (local models are not shared across threads); only LLM calls fan out.
            jobs = []
            for qa in todo:
                start = time.perf_counter()
                context = system.retrieve(qa.question)
                jobs.append((qa, context, time.perf_counter() - start))
            # Answer one question alone first: with a shared prompt prefix (full_context) this lets the
            # provider build its prefix cache before the rest arrive in parallel and would all miss it.
            predictions.write(_answer_one(chat, conversation, cfg.prompt, *jobs[0]))
            bar.update(1)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(_answer_one, chat, conversation, cfg.prompt, qa, context, retrieve_s)
                    for qa, context, retrieve_s in jobs[1:]
                ]
                for future in as_completed(futures):
                    predictions.write(future.result())
                    bar.update(1)
    return pred_path


def _identity(
    cfg: RunConfig, chat: ChatModel, embedder: Embedder | None, reranker: Reranker | None
) -> dict[str, Any]:
    retrieval = cfg.system in NEEDS_EMBEDDER
    hybrid = cfg.system == "hybrid_rag"
    return {
        "system": cfg.system,
        "answer_model": chat.model,
        "prompt_version": prompt_version(cfg.prompt),
        "reasoning": cfg.reasoning,
        "mock": cfg.mock,
        "top_k": cfg.top_k if retrieval else None,
        "chunk_turns": cfg.chunk_turns if cfg.system == "naive_rag" else None,
        "embedding_model": embedder.name if embedder else None,
        "fusion": cfg.fusion if hybrid else None,
        "rerank_candidates": cfg.rerank_candidates if hybrid else None,
        "neighbors": cfg.neighbors if hybrid else None,
        "max_per_session": cfg.max_per_session if hybrid else None,
        "time_anchors": cfg.time_anchors if hybrid else None,
        "feedback_turns": cfg.feedback_turns if hybrid else None,
        "context_before": cfg.context_before if hybrid else None,
        "time_candidates": cfg.time_candidates if hybrid else None,
        "speaker_boost": cfg.speaker_boost if hybrid else None,
        "reranker_model": reranker.name if reranker else None,
    }


def _check_identity(run_dir: Path, identity: dict[str, Any]) -> None:
    path = run_dir / RUN_INFO
    if not path.exists():
        path.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    stored = json.loads(path.read_text(encoding="utf-8"))
    diff = {key: {"stored": stored.get(key), "now": value} for key, value in identity.items() if stored.get(key) != value}
    if diff:
        raise RunMismatchError(f"运行 {run_dir.name!r} 是用不同设置创建的：{diff}。请换一个 --run-name。")


def _complete(chat: ChatModel, messages: list[Message], max_tokens: int) -> tuple[Completion, dict[str, Any]]:
    """One call, retried once with twice the cap if a thinking model spent it all; usage covers both calls."""
    attempts = [chat.complete(messages, max_tokens=max_tokens, temperature=0.0)]
    if attempts[0].finish_reason == "length":
        attempts.append(chat.complete(messages, max_tokens=max_tokens * 2, temperature=0.0))
    usage = sum((attempt.usage for attempt in attempts), Usage())
    live = [attempt for attempt in attempts if not attempt.from_cache]
    costs = [
        call_cost(chat.model, a.usage.prompt_tokens, a.usage.cached_prompt_tokens, a.usage.completion_tokens,
                  datetime.fromisoformat(a.called_at) if a.called_at else None)
        for a in live
    ]
    return attempts[-1], {
        "prompt_tokens": usage.prompt_tokens,
        "cached_prompt_tokens": usage.cached_prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        # What this run paid; a response served from the local cache costs nothing.
        "cost_cny": None if any(cost is None for cost in costs) else round(sum(costs), 6),
        "called_at": live[-1].called_at if live else None,
    }


def _answer_one(
    chat: ChatModel,
    conversation: Conversation,
    prompt: str,
    qa: QAItem,
    context: RetrievedContext,
    retrieve_s: float,
) -> dict[str, Any]:
    messages = answer_messages(conversation, context.text, qa.question, prompt)
    completion, spend = _complete(chat, messages, ANSWER_MAX_TOKENS)
    answer, cited = completion.text, None
    if prompt in CITING_PROMPTS:
        answer, cited = split_citations(completion.text)
    row = {
        "qa_id": qa.qa_id,
        "sample_id": qa.sample_id,
        "category": qa.category,
        "category_name": qa.category_name,
        "question": qa.question,
        "gold": qa.gold,
        "prediction": answer,
        "f1": locomo_f1(answer, qa.gold, qa.category),
        "evidence": list(qa.evidence),
        "evidence_recall": evidence_recall(qa.evidence, context.turn_ids),
        "answer_visible": answer_visible(qa.gold, context.text, qa.category),
        "context_turns": len(context.turn_ids),
        "retrieved_ids": list(context.turn_ids) if len(context.turn_ids) <= MAX_STORED_IDS else None,
        **spend,
        "answer_latency_s": round(completion.latency_s, 4),
        "retrieve_latency_s": round(retrieve_s, 4),
        "from_cache": completion.from_cache,
        "finish_reason": completion.finish_reason,
    }
    if cited is not None:
        row["raw_prediction"] = completion.text
        row.update(citation_scores(cited, qa.evidence, context.turn_ids))
    return row


def run_judge(
    run_name: str,
    kind: str,
    settings: Settings,
    *,
    chat: ChatModel | None = None,
    workers: int = 8,
    progress: bool = True,
) -> Path:
    if kind not in JUDGE_KINDS:
        raise ValueError(f"unknown judge {kind!r}; choose from {', '.join(JUDGE_KINDS)}")
    run_dir = existing_run(settings, run_name)
    info = json.loads((run_dir / RUN_INFO).read_text(encoding="utf-8"))
    out = judge_path(run_dir, kind)
    # Unparsed verdicts are redone; summaries keep the last line written for each question.
    done = {row["qa_id"] for row in read_jsonl(out) if row["label"] is not None}
    todo = [row for row in read_jsonl(run_dir / PREDICTIONS) if row["qa_id"] not in done]
    if progress:
        by_rule = sum(rule_verdict(row["gold"], row["prediction"]) is not None for row in todo)
        print(f"裁判[{kind}]：待判 {len(todo)} 题，规则直接判定 {by_rule} 题，调用裁判模型 {len(todo) - by_rule} 次",
              file=sys.stderr)
        if not info.get("mock"):
            if line := estimate_line(settings.judge_model, len(todo) - by_rule, "judge"):
                print(line, file=sys.stderr)
    # A mock run is graded by the mock judge, so its placeholder answers never reach a paid API.
    chat = chat or build_chat(settings, settings.judge_model, mock=info.get("mock", False), responder=mock_judge_responder)
    with (
        JsonlWriter(out) as writer,
        ThreadPoolExecutor(max_workers=workers) as pool,
        tqdm(total=len(todo), desc=f"judge[{kind}]", disable=not progress) as bar,
    ):
        futures = [pool.submit(_judge_one, chat, kind, row) for row in todo]
        for future in as_completed(futures):
            writer.write(future.result())
            bar.update(1)
    return out


def _judge_one(chat: ChatModel, kind: str, prediction: dict[str, Any]) -> dict[str, Any]:
    verdict = rule_verdict(prediction["gold"], prediction["prediction"])
    if verdict is not None:
        return {
            "qa_id": prediction["qa_id"],
            "label": verdict.label,
            "correct": verdict.correct,
            "reasoning": verdict.reasoning,
            "judge_model": "rule",
            "finish_reason": None,
        }
    messages: list[Message] = judge_messages(kind, prediction["question"], prediction["gold"], prediction["prediction"])
    completion, spend = _complete(chat, messages, JUDGE_MAX_TOKENS)
    verdict = parse_verdict(completion.text)
    return {
        "qa_id": prediction["qa_id"],
        "label": verdict.label,
        "correct": verdict.correct,
        "reasoning": verdict.reasoning,
        "judge_model": chat.model,
        "finish_reason": completion.finish_reason,
        "from_cache": completion.from_cache,
        **spend,
    }


def summarize(run_name: str, settings: Settings) -> dict[str, Any]:
    run_dir = existing_run(settings, run_name)
    info = json.loads((run_dir / RUN_INFO).read_text(encoding="utf-8"))
    predictions = list(read_jsonl(run_dir / PREDICTIONS))
    verdicts = {
        kind: {row["qa_id"]: row for row in read_jsonl(judge_path(run_dir, kind))}
        for kind in JUDGE_KINDS
        if judge_path(run_dir, kind).exists()
    }
    groups: dict[str, list[dict[str, Any]]] = {"overall": predictions}
    for category in sorted({row["category"] for row in predictions}):
        groups[CATEGORY_NAMES.get(category, f"category-{category}")] = [
            row for row in predictions if row["category"] == category
        ]
    summary = {
        "run": run_name,
        "info": info,
        "groups": {name: _group_stats(rows, verdicts) for name, rows in groups.items()},
        "tokens": {
            "answer": _token_totals(predictions),
            **{f"judge_{kind}": _token_totals(by_id.values()) for kind, by_id in verdicts.items()},
        },
    }
    (run_dir / SUMMARY_JSON).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / SUMMARY_MD).write_text(render_markdown(summary), encoding="utf-8")
    return summary


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return fmean(values) if values else None


def _group_stats(rows: list[dict[str, Any]], verdicts: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    f1 = [row["f1"] for row in rows]
    stats: dict[str, Any] = {"n": len(rows), "f1": _mean(f1), "f1_ci": bootstrap_ci(f1)}
    for kind, by_id in verdicts.items():
        judged = [by_id[row["qa_id"]] for row in rows if row["qa_id"] in by_id]
        correct = [1.0 if verdict["correct"] else 0.0 for verdict in judged]
        stats[f"judge_{kind}"] = _mean(correct)
        stats[f"judge_{kind}_ci"] = bootstrap_ci(correct)
        stats[f"judge_{kind}_n"] = len(judged)
        # Error attribution: a wrong answer either lacked evidence in its context or had it all and still failed.
        wrong = [row for row in rows if row["qa_id"] in by_id and not by_id[row["qa_id"]]["correct"]]
        stats[f"judge_{kind}_wrong_missing_evidence"] = sum(
            row["evidence_recall"] is not None and row["evidence_recall"] < 1.0 for row in wrong
        )
        stats[f"judge_{kind}_wrong_with_evidence"] = sum(row["evidence_recall"] == 1.0 for row in wrong)
        stats[f"judge_{kind}_unparsed"] = sum(verdict["label"] is None for verdict in judged)
    stats["evidence_recall"] = _mean(row["evidence_recall"] for row in rows if row["evidence_recall"] is not None)
    stats["answer_visible"] = _mean(
        float(row["answer_visible"]) for row in rows if row.get("answer_visible") is not None
    )
    if any("cited_ids" in row for row in rows):
        stats["cited"] = _mean(float(bool(row.get("cited_ids"))) for row in rows)
        for key in ("citation_precision", "citation_recall", "citation_in_context"):
            stats[key] = _mean(row[key] for row in rows if row.get(key) is not None)
    stats["truncated"] = sum(row.get("finish_reason") == "length" for row in rows)
    stats["context_turns"] = _mean(row["context_turns"] for row in rows)
    stats["prompt_tokens"] = _mean(row["prompt_tokens"] for row in rows)
    latencies = [row["answer_latency_s"] for row in rows]
    stats["latency_p50_s"] = float(np.percentile(latencies, 50)) if latencies else None
    stats["latency_p95_s"] = float(np.percentile(latencies, 95)) if latencies else None
    return stats


def _token_totals(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    totals: Counter = Counter()
    for row in rows:
        for key in ("prompt_tokens", "cached_prompt_tokens", "completion_tokens", "reasoning_tokens", "cost_cny"):
            totals[key] += row.get(key) or 0
    return {key: round(value, 4) if key == "cost_cny" else value for key, value in totals.items()}


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 0) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def render_markdown(summary: dict[str, Any]) -> str:
    info = summary["info"]
    meta = [f"系统 `{info['system']}`", f"回答模型 `{info['answer_model']}`", f"Prompt `{info['prompt_version']}`"]
    if info.get("chunk_turns") is not None:
        meta.append(f"top_k={info['top_k']}，每块 {info['chunk_turns']} 轮，向量模型 `{Path(info['embedding_model']).name}`")
    if info.get("rerank_candidates") is not None:
        reranker = Path(info["reranker_model"]).name if info.get("reranker_model") else "无"
        meta.append(
            f"预算 {info['top_k']} 轮，BM25 {info.get('fusion')}，重排候选 {info['rerank_candidates']}（`{reranker}`），"
            f"邻近 ±{info['neighbors']} 轮，每会话上限 {info['max_per_session'] or '不限'}，"
            f"时间锚定 {'开' if info.get('time_anchors') else '关'}，"
            f"伪相关反馈 {info.get('feedback_turns') or '关'}，"
            f"上文索引 {info.get('context_before') or 0} 轮，时间候选 {info.get('time_candidates') or 0}，"
            f"说话人加分 {info.get('speaker_boost') or 0}"
        )
    lines = [f"# 评测结果：{summary['run']}", "", " · ".join(meta), ""]
    if info.get("mock"):
        lines += ["> **这是 --mock 离线运行：回答和裁判都是占位，数字只用来检查流程，没有任何意义。**", ""]
    lines += [
        "| 题型 | 题数 | F1 | 严格裁判 | 宽松裁判 | 证据召回 | 答案可见 | 平均上下文轮数 | 平均 prompt token | 延迟 P50 / P95 (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in summary["groups"].items():
        lines.append(
            f"| {name} | {stats['n']} | {_pct(stats['f1'])} | {_pct(stats.get('judge_strict'))} | "
            f"{_pct(stats.get('judge_lenient'))} | {_pct(stats['evidence_recall'])} | "
            f"{_pct(stats.get('answer_visible'))} | {_num(stats['context_turns'], 1)} | "
            f"{_num(stats['prompt_tokens'])} | {_num(stats['latency_p50_s'], 2)} / {_num(stats['latency_p95_s'], 2)} |"
        )
    overall = summary["groups"]["overall"]
    intervals = [("F1", overall.get("f1_ci"))] + [
        (label, overall.get(f"judge_{kind}_ci")) for kind, label in (("strict", "严格裁判"), ("lenient", "宽松裁判"))
    ]
    shown = [f"{label} [{ci[0] * 100:.1f}%, {ci[1] * 100:.1f}%]" for label, ci in intervals if ci]
    if shown:
        lines += ["", "整体 95% 置信区间（按题 bootstrap 2000 次）：" + "；".join(shown)]
    if "judge_strict_n" in overall:
        lines += [
            "",
            "严格裁判下的错误归因：",
            "",
            "| 题型 | 答错 | 证据没召回全 | 证据齐全却答错 |",
            "|---|---:|---:|---:|",
        ]
        for name, stats in summary["groups"].items():
            missing = stats.get("judge_strict_wrong_missing_evidence", 0)
            complete = stats.get("judge_strict_wrong_with_evidence", 0)
            wrong = round(stats["judge_strict_n"] * (1 - (stats.get("judge_strict") or 0)))
            lines.append(f"| {name} | {wrong} | {missing} | {complete} |")
    if "cited" in overall:
        lines += [
            "",
            f"出处引用（对照 LoCoMo 标注证据，标注不全时精确率偏低）：带引用的回答 {_pct(overall['cited'])}，"
            f"引用精确率 {_pct(overall.get('citation_precision'))}，标注证据被引用 {_pct(overall.get('citation_recall'))}，"
            f"引用落在上下文内 {_pct(overall.get('citation_in_context'))}",
        ]
    lines += ["", "Token 合计（prompt / 其中命中前缀缓存 / completion / 其中思考）与本次实际花费：", ""]
    for stage, totals in summary["tokens"].items():
        lines.append(
            f"- {stage}：{_num(totals.get('prompt_tokens', 0))} / {_num(totals.get('cached_prompt_tokens', 0))} / "
            f"{_num(totals.get('completion_tokens', 0))} / {_num(totals.get('reasoning_tokens', 0))}，"
            f"¥{totals.get('cost_cny', 0):.2f}"
        )
    unparsed = {kind: summary["groups"]["overall"].get(f"judge_{kind}_unparsed", 0) for kind in JUDGE_KINDS}
    if any(unparsed.values()):
        lines += ["", f"裁判输出无法解析的条数（计为错误）：{unparsed}"]
    if summary["groups"]["overall"].get("truncated"):
        lines += ["", f"回答被输出上限截断的条数：{summary['groups']['overall']['truncated']}"]
    return "\n".join(lines) + "\n"


def existing_run(settings: Settings, run_name: str) -> Path:
    run_dir = settings.runs_dir / run_name
    if not (run_dir / RUN_INFO).exists():
        raise FileNotFoundError(f"找不到运行 {run_name!r}（{run_dir}），请先执行 memocore run。")
    return run_dir
