# MemoCore

[![CI](https://github.com/HHqm/MemoCoreAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/HHqm/MemoCoreAgent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Agent 长期记忆引擎，目前处于第一阶段：先把**可复现的评测框架**和**两个基线**搭好，后续每加一个记忆模块，都在同一把尺子上量出它到底带来了多少提升。

## 现在有什么

| 模块 | 内容 |
|---|---|
| 数据 | LoCoMo：10 段长对话，1,986 道题。默认评测 1～4 类共 1,540 题；第 5 类对抗题（正确做法是拒答）可选 |
| 基线 | `full_context`：整段对话放进 prompt，代表准确率和成本的上限。`naive_rag`：用 bge-m3 对原始对话轮次做向量检索 |
| 指标 | 与官方脚本口径一致的 token F1；严格、宽松两套 LLM 裁判；证据召回（检索有没有把标注的证据轮次放进上下文）；token 用量、前缀缓存命中、延迟 |
| 不花钱的检索评测 | `retrieval-sweep`：不调用 LLM，直接量不同 top_k 和分块大小下的证据召回 |
| 工程 | 断点续跑（JSONL 追加写 + SQLite 响应缓存，重复的调用不再花钱）；向量按内容寻址缓存到磁盘；设置变了会拒绝续跑，避免把不同口径的结果混在一起；`--mock` 离线跑通全流程；导出样本做人工标注，计算裁判与人工的一致率 |

## 设计原则

1. **控制变量**：所有系统共用同一个回答模型和 prompt，系统只负责提供上下文。分数的差异只可能来自记忆本身。
2. **先把评测搭好，再写记忆引擎**。
3. **裁判要透明**：同时报严格和宽松两种裁判，并抽样人工标注计算一致率。据 [LoCoMo-Refined](https://github.com/mem-eval-suite/LoCoMo_refined) 的报告，原始 LoCoMo 裁判与人工的一致率只有 43.67%，宽松裁判会让分数虚高 15～22 个点。
4. **省钱的 prompt 结构**：记忆放在前面，问题放在最后。`full_context` 下同一段对话的所有问题共享一个很长的前缀，可以命中服务商的前缀缓存。
5. **能分清错在哪**：证据召回把"检索没找到"和"找到了但模型没答对"这两类错误分开。

## 快速开始

```bash
conda activate timeseries
```

```bash
cd D:\prru\memocore
```

```bash
pip install -e . --no-deps
```

```bash
python -m unittest discover -s tests -t .
```

数据集和 bge-m3 已经放在 `data/` 和 `models/` 下（这两个目录都不提交）。换一台机器时，先执行 `memocore download-locomo`。

### 1. 不花钱：先量检索

```bash
memocore retrieval-sweep --top-k 1,3,5,10,20 --chunk-turns 1,2,4
```

### 2. 离线检查流程

```bash
memocore run --system full_context --run-name fc-mock --max-conversations 1 --max-questions 20 --mock
```

### 3. 真实评测

把 `.env.example` 复制为 `.env`，填入 `DEEPSEEK_API_KEY`，然后先抽样跑一小轮：

```bash
memocore run --system naive_rag --run-name rag-k20-smoke --max-conversations 2 --max-questions 25
```

```bash
memocore run --system full_context --run-name fc-smoke --max-conversations 2 --max-questions 25
```

确认没问题后再跑全量（去掉两个 `--max-*` 参数，并换一个 `--run-name`）。结果写在 `runs/<run-name>/summary.md`。中途中断的话，用同样的命令重跑即可续上。

### 4. 抽检裁判

```bash
memocore audit-export --run-name rag-k20 --kind strict --n 100
```

在导出的 CSV 的 `human_label` 列填 `CORRECT` / `WRONG`（也可以填 对 / 错），然后：

```bash
memocore audit-score --csv runs/rag-k20/audit_strict.csv
```

## 成本估算

- `full_context`：每题约 1.6 万～1.8 万 prompt token（按字符数估算），全量 1,540 题约 2,500 万 token；其中大部分能命中前缀缓存。
- `naive_rag`（默认 top_k=20）：每题约 1 千 token。
- 每次裁判调用约 300～400 token；默认会跑严格和宽松两套。

具体费用以 DeepSeek 当前价格为准。每次运行的实际 token 和缓存命中数都会写进 `summary.md`。

## 检索召回实测（bge-m3，不调用 LLM）

评测 1～4 类共 1,540 题，标注证据共 2,356 条，其中 8 条在对话里找不到（属于标注噪声）。一段对话平均约 590 轮。下表是节选，完整结果见 [docs/results/retrieval_sweep.md](docs/results/retrieval_sweep.md)。

| 每块轮数 | top_k | 平均上下文轮数 | 证据召回 | multi-hop | temporal | open-domain | single-hop |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 10.0 | 68.6% | 44.8% | 77.5% | 38.4% | 76.5% |
| **1** | **20** | **20.0** | **76.4%** | 57.3% | 83.3% | 46.5% | 83.4% |
| 2 | 10 | 19.6 | 75.8% | 52.5% | 80.2% | 46.8% | 85.2% |
| 4 | 5 | 19.2 | 73.2% | 48.0% | 76.2% | 47.9% | 83.3% |
| 4 | 20 | 76.2 | 87.9% | 75.7% | 89.1% | 65.5% | 93.9% |

从表中可以看出三点：

1. **上下文预算相同时，按单轮切块的召回最高**：都取约 20 轮时，三种切法分别是 76.4%、75.8%、73.2%。因此 `naive_rag` 默认按单轮切块、取 top 20：只用约 3% 的对话，就能召回 76% 的证据。
2. **多跳题和开放推理题最难**：它们的证据分散在不同会话里，单纯按相似度检索很难一次召回齐。这是下一阶段记忆引擎（跨会话整合事实、按时间建索引）要重点解决的问题。
3. **证据召回不等于答对**：证据进了上下文，模型也不一定能答对，最终要看 LLM 回答加裁判的结果。

### 性能优化

在同一台 RTX 5060 笔记本上测得：

| 环节 | 优化前 | 优化后 | 做法 |
|---|---:|---:|---|
| 建索引（三种切块合计约 1 万段文本） | 143.4 s | 51.3 s | GPU 上用半精度（召回变化不超过 0.4 个百分点） |
| 单次检索（单轮切块） | 16.8 ms | 0.8 ms | 同一段对话的所有问题一次性批量编码 |
| 第二次完整扫描（15 组参数） | — | 3.5 s | 按内容寻址的向量磁盘缓存全部命中 |

## 路线图

1. **写入路径**：从对话中抽取候选记忆，与已有记忆比对后决定新增、更新、删除或不变；事件时间与记录时间分开存储。
2. **读取路径**：混合检索（向量 + BM25 + 时间过滤）、重排、在 token 预算内压缩上下文；对每个组件做消融实验。
3. **服务化**：REST 和 MCP 接口、多用户隔离、异步写入、k6 压测；接入 RepoPilot。
4. **更多基准**：LongMemEval 子集、用 LoCoMo-Refined 交叉评测；以 Mem0 作为对照。

可借鉴的开源项目见 [docs/REFERENCES.md](docs/REFERENCES.md)。

## 目录

```text
src/memocore/
  datasets/locomo.py   数据加载，题型编号与官方一致
  systems/             记忆系统：full_context、naive_rag，后续的记忆引擎也放这里
  prompts.py           所有系统共用的回答 prompt
  metrics.py           F1（复刻官方口径）与证据召回
  judge.py             严格 / 宽松裁判与输出解析
  runner.py            回答、裁判、汇总、人工抽检
  retrieval_eval.py    不调用 LLM 的检索召回评测
  llm.py               OpenAI 兼容客户端、响应缓存、mock
  embeddings.py        bge-m3 编码器（GPU 半精度）、向量磁盘缓存
tests/                 离线单元测试（mock 模型 + 哈希向量，不需要 API Key）
```
