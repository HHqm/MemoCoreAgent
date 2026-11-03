from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from memocore.audit import export_audit, label_audit, score_audit
from memocore.config import Settings
from memocore.datasets.locomo import CATEGORY_NAMES, DEFAULT_CATEGORIES, LOCOMO_URL, load_locomo
from memocore.judge import JUDGE_KINDS
from memocore.llm import REASONING_LEVELS
from memocore.prompts import ANSWER_TEMPLATES, DEFAULT_PROMPT
from memocore.runner import (
    SUMMARY_MD,
    RunConfig,
    RunMismatchError,
    run_judge,
    run_predictions,
    summarize,
)
from memocore.systems import SYSTEM_NAMES


def _int_list(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def _str_list(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _judge_kinds(choice: str) -> tuple[str, ...]:
    if choice == "none":
        return ()
    return JUDGE_KINDS if choice == "both" else (choice,)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memocore", description="MemoCore：Agent 记忆系统的 LoCoMo 评测工具")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download-locomo", help="下载 LoCoMo 数据集（locomo10.json）")

    run = sub.add_parser("run", help="让一个记忆系统回答 LoCoMo 问题，然后裁判并汇总")
    run.add_argument("--system", choices=SYSTEM_NAMES, required=True)
    run.add_argument("--run-name", help="结果目录名，默认等于 --system；同名会续跑")
    run.add_argument(
        "--categories",
        type=_int_list,
        default=DEFAULT_CATEGORIES,
        help="题型编号，逗号分隔：" + "，".join(f"{k}={v}" for k, v in CATEGORY_NAMES.items()) + "（默认 1,2,3,4）",
    )
    run.add_argument("--conversations", type=_str_list, default=(), help="只跑这些对话，如 conv-26,conv-30")
    run.add_argument("--max-conversations", type=int)
    run.add_argument("--max-questions", type=int, help="每段对话按固定种子抽多少题")
    run.add_argument("--prompt", choices=tuple(ANSWER_TEMPLATES), default=DEFAULT_PROMPT, help="回答 prompt 版本，用于 A/B")
    run.add_argument("--qa-ids", type=_str_list, default=(), help="只跑这些题，如 conv-26#q0019，用于抽查（忽略 --max-questions）")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--top-k", type=int, default=20, help="检索类系统放进上下文的块数（hybrid_rag 中为轮数预算）")
    run.add_argument("--chunk-turns", type=int, default=1, help="naive_rag 每块包含的对话轮数")
    run.add_argument(
        "--fusion", choices=("pool", "rrf"), default="pool", help="hybrid_rag 中 BM25 的用法：补充重排候选（pool）或排名融合（rrf）"
    )
    run.add_argument("--rerank-candidates", type=int, default=50, help="hybrid_rag 交给重排模型的候选数，0 关闭重排")
    run.add_argument("--no-time-anchors", action="store_true", help="hybrid_rag 关闭写入时的时间锚定")
    run.add_argument("--feedback-turns", type=int, default=2, help="hybrid_rag 伪相关反馈取前几轮做第二跳，0 关闭")
    run.add_argument("--context-before", type=int, default=RunConfig.context_before,
                     help="hybrid_rag 编码每轮时带上同一会话里的前几轮（只用于检索，不进上下文）")
    run.add_argument("--time-candidates", type=int, default=RunConfig.time_candidates,
                     help="hybrid_rag 问题点名时间时，补进重排候选的该时段轮数，0 关闭")
    run.add_argument("--speaker-boost", type=float, default=RunConfig.speaker_boost,
                     help="hybrid_rag 问题只点名一位说话人时，给其发言的向量相似度加分，0 关闭")
    run.add_argument("--neighbors", type=int, default=0, help="hybrid_rag 每个命中额外带上前后几轮")
    run.add_argument("--max-per-session", type=int, help="hybrid_rag 每个会话最多命中几次")
    run.add_argument(
        "--judge", choices=(*JUDGE_KINDS, "both", "none"), default="strict", help="默认只跑严格裁判以节省额度"
    )
    run.add_argument(
        "--reasoning", choices=REASONING_LEVELS, help="回答模型的思考程度；默认沿用 API 的默认值（思考，high），none 为关闭思考"
    )
    run.add_argument("--wait-off-peak", action="store_true", help="等到低谷时段（半价）再开始调用 API")
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--mock", action="store_true", help="离线占位模型，只检查流程，不调用 API")

    judge = sub.add_parser("judge", help="对已有运行补做裁判")
    judge.add_argument("--run-name", required=True)
    judge.add_argument("--kind", choices=(*JUDGE_KINDS, "both"), default="strict")
    judge.add_argument(
        "--reasoning", choices=REASONING_LEVELS, help="裁判的思考程度；设了以后结果写到 judge_<kind>-<程度>.jsonl，不与默认裁判混在一起"
    )
    judge.add_argument("--wait-off-peak", action="store_true", help="等到低谷时段（半价）再开始调用 API")
    judge.add_argument("--workers", type=int, default=8)

    report = sub.add_parser("report", help="重新生成汇总表")
    report.add_argument("--run-name", required=True)

    audit = sub.add_parser("audit-export", help="盲抽样导出裁判过的回答供人工标注，表里不显示裁判结论和系统名")
    audit.add_argument("--run-name", nargs="+", required=True, help="可给多个运行：题目不重复，各运行轮流分配")
    audit.add_argument("--kind", choices=JUDGE_KINDS, default="strict")
    audit.add_argument("--n", type=int, default=100)
    audit.add_argument("--seed", type=int, default=0)
    audit.add_argument("--include-rule", action="store_true", help="也抽规则直接判定的题（默认跳过：它们按构造就是对的）")
    audit.add_argument("--out", type=Path, help="默认 runs/audit_<kind>.csv")

    label = sub.add_parser("audit-label", help="在终端里逐题标注，每标一题就保存，可随时退出再续")
    label.add_argument("--csv", type=Path, required=True)

    agreement = sub.add_parser("audit-score", help="把人工标注和裁判结论对上，计算一致率、kappa 和裁判偏严还是偏松")
    agreement.add_argument("--csv", type=Path, required=True)

    sweep = sub.add_parser("retrieval-sweep", help="只量朴素 RAG 的证据召回，不调用 LLM、不花钱")
    sweep.add_argument("--top-k", type=_int_list, default=(1, 3, 5, 10, 20))
    sweep.add_argument("--chunk-turns", type=_int_list, default=(1, 2, 4))
    sweep.add_argument("--categories", type=_int_list, default=DEFAULT_CATEGORIES)

    compare = sub.add_parser("compare", help="两个运行按题配对比较，给出差值的置信区间和 p 值")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--metric", default="judge_strict", help="judge_strict / judge_lenient / f1")

    ablation = sub.add_parser("retrieval-ablation", help="同一预算下逐个组件比较读取路径的证据召回，不调用 LLM")
    ablation.add_argument("--budgets", type=_int_list, default=(5, 10, 20))
    ablation.add_argument("--categories", type=_int_list, default=DEFAULT_CATEGORIES)

    replay = sub.add_parser(
        "memory-replay", help="把 LoCoMo 逐条写入 MemoryStore 再检索，核对与评测过的检索路径逐题一致，不调用 LLM"
    )
    replay.add_argument("--budget", type=int, default=20)

    sub.add_parser("cost", help="统计各次运行实际花了多少钱（只读本地记录，不调用 API）")

    serve = sub.add_parser("mcp", help="启动 MCP 服务（stdio），让 Claude Code 等 Agent 读写长期记忆")
    serve.add_argument("--user", default="default", help="记忆归属的用户；不同用户的记忆互相隔离")
    serve.add_argument("--db", type=Path, help="记忆库文件，默认 data/memory.sqlite")
    serve.add_argument("--offline", action="store_true", help="用哈希向量和字面重排代替模型，只用来试流程")

    http = sub.add_parser("serve", help="启动 REST 接口（FastAPI），并发检索会合批使用 GPU；接口文档在 /docs")
    http.add_argument("--host", default="127.0.0.1")
    http.add_argument("--port", type=int, default=8000)
    http.add_argument("--db", type=Path, help="记忆库文件，默认 data/memory.sqlite")
    http.add_argument("--offline", action="store_true", help="用哈希向量和字面重排代替模型，只用来试流程")
    return parser


def _download(settings: Settings) -> None:
    import httpx

    target = settings.locomo_path
    target.parent.mkdir(parents=True, exist_ok=True)
    response = httpx.get(LOCOMO_URL, follow_redirects=True, timeout=120)
    response.raise_for_status()
    target.write_bytes(response.content)
    samples = load_locomo(target)
    print(f"已下载 {target}：{len(samples)} 段对话，{sum(len(s.qa) for s in samples)} 道题")


def _report(settings: Settings, run_name: str) -> None:
    summarize(run_name, settings)
    print((settings.runs_dir / run_name / SUMMARY_MD).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to GBK; never crash on a character it cannot show.
    sys.stdout.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args, Settings.from_env())
    except (ValueError, FileNotFoundError, FileExistsError, RunMismatchError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "download-locomo":
        _download(settings)
    elif args.command == "run":
        if not settings.locomo_path.exists():
            print(f"缺少数据集 {settings.locomo_path}，请先执行 memocore download-locomo", file=sys.stderr)
            return 1
        cfg = RunConfig(
            system=args.system,
            run_name=args.run_name or args.system,
            categories=args.categories,
            conversations=args.conversations,
            max_conversations=args.max_conversations,
            qa_ids=args.qa_ids,
            prompt=args.prompt,
            max_questions=args.max_questions,
            seed=args.seed,
            top_k=args.top_k,
            chunk_turns=args.chunk_turns,
            fusion=args.fusion,
            rerank_candidates=args.rerank_candidates,
            neighbors=args.neighbors,
            max_per_session=args.max_per_session,
            time_anchors=not args.no_time_anchors,
            feedback_turns=args.feedback_turns,
            context_before=args.context_before,
            time_candidates=args.time_candidates,
            speaker_boost=args.speaker_boost,
            reasoning=args.reasoning,
            mock=args.mock,
        )
        if args.wait_off_peak and not args.mock:
            _wait_off_peak()
        run_predictions(cfg, settings, workers=args.workers)
        for kind in _judge_kinds(args.judge):
            run_judge(cfg.run_name, kind, settings, workers=args.workers)
        _report(settings, cfg.run_name)
    elif args.command == "judge":
        if args.wait_off_peak:
            _wait_off_peak()
        for kind in _judge_kinds(args.kind):
            run_judge(args.run_name, kind, settings, workers=args.workers, reasoning=args.reasoning)
        _report(settings, args.run_name)
    elif args.command == "report":
        _report(settings, args.run_name)
    elif args.command == "audit-export":
        path = export_audit(
            args.run_name, args.kind, settings, n=args.n, seed=args.seed, include_rule=args.include_rule, out=args.out
        )
        print(f"已导出 {path}。逐题标注：memocore audit-label --csv {path}"
              f"（也可以直接在 human_label 列填 对 / 错 / ?），标完运行 memocore audit-score --csv {path}")
    elif args.command == "audit-label":
        label_audit(args.csv)
    elif args.command == "audit-score":
        print(json.dumps(score_audit(args.csv, settings), ensure_ascii=False, indent=2))
    elif args.command == "retrieval-sweep":
        from memocore.retrieval_eval import render_sweep, retrieval_sweep

        result = retrieval_sweep(
            settings, top_ks=args.top_k, chunk_turns=args.chunk_turns, categories=args.categories
        )
        print(render_sweep(result))
    elif args.command == "compare":
        from memocore.compare import compare_runs, render_comparison

        print(render_comparison(compare_runs(settings, args.baseline, args.candidate, args.metric)))
    elif args.command == "retrieval-ablation":
        from memocore.retrieval_eval import render_ablation, retrieval_ablation

        print(render_ablation(retrieval_ablation(settings, budgets=args.budgets, categories=args.categories)))
    elif args.command == "memory-replay":
        from memocore.retrieval_eval import memory_replay

        print(json.dumps(memory_replay(settings, budget=args.budget), ensure_ascii=False, indent=2))
    elif args.command == "mcp":
        _serve(settings, args)
    elif args.command == "serve":
        _serve_http(settings, args)
    elif args.command == "cost":
        from memocore.spend import render_spend, spend_report

        print(render_spend(spend_report(settings)))
    return 0


def _wait_off_peak() -> None:
    from datetime import datetime, timezone

    from memocore.pricing import BEIJING, next_off_peak

    now = datetime.now(timezone.utc)
    start = next_off_peak(now)
    if start > now:
        print(f"等到低谷时段再调用 API：北京时间 {start.astimezone(BEIJING):%m-%d %H:%M} 开始", file=sys.stderr, flush=True)
        time.sleep((start - now).total_seconds() + 5)


def _open_store(settings: Settings, args: argparse.Namespace, *, batching: bool = False):
    from memocore.store import MemoryStore

    if args.offline:
        from memocore.embeddings import HashingEmbedder
        from memocore.retrieval.rerank import LexicalReranker

        embedder, reranker = HashingEmbedder(), LexicalReranker()
    else:
        from memocore.embeddings import SentenceTransformerEmbedder
        from memocore.retrieval.rerank import BatchingReranker, CrossEncoderReranker

        # No disk caches here: the store keeps its own vectors, so forgetting a message leaves nothing behind.
        embedder = SentenceTransformerEmbedder(settings.embedding_model, device=settings.embedding_device)
        reranker = CrossEncoderReranker(settings.reranker_model, device=settings.embedding_device)
        if batching:  # concurrent searches share GPU calls
            reranker = BatchingReranker(reranker)
    return MemoryStore(args.db or settings.data_dir / "memory.sqlite", embedder, reranker)


def _serve(settings: Settings, args: argparse.Namespace) -> None:
    from memocore.mcp_server import build_server

    build_server(_open_store(settings, args), args.user).run("stdio")


def _serve_http(settings: Settings, args: argparse.Namespace) -> None:
    import uvicorn

    from memocore.http_server import build_app

    token = os.environ.get("MEMOCORE_SERVER_TOKEN")
    if args.host not in ("127.0.0.1", "localhost") and not token:
        raise ValueError("监听非本机地址时必须设置 MEMOCORE_SERVER_TOKEN，否则任何人都能读写记忆")
    uvicorn.run(build_app(_open_store(settings, args, batching=True), token=token), host=args.host, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())
