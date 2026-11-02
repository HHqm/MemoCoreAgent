# MemoCore

[![CI](https://github.com/HHqm/MemoCoreAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/HHqm/MemoCoreAgent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**原文为本、零 LLM 写入、评测可信**的 Agent 长期记忆引擎。

我调研了 Mem0、Graphiti 等高星记忆项目讨论最多的 issue，社区反复提到的问题有：垃圾记忆、不会处理时间、每次写入都调 LLM、评测复现不出来、结果查不到出处。它们有一个共同根源：**写入时用 LLM 把对话加工成另一种表示，而加工是有损的，加工结果又取代了原文。** MemoCore 反过来做：

- **原文永远不改写**：派生信息（时间注释、索引）只挂在原文旁边，每条上下文都能回溯到原始对话轮次。
- **写入不调用 LLM**：向量编码 + BM25 索引 + 规则时间锚定。
- **读取靠精排**：bge-m3 召回，BM25 补充候选，bge-reranker 重排。
- **每个改动都要在同一预算下用数字证明**。

详见 [与高星项目的对比](docs/COMPARISON.md) 和 [评测方法](docs/EVALUATION.md)。

## 端到端结果（LLM 作答 + 严格裁判）

在 LoCoMo 上按固定种子抽了 300 题（每段对话 30 题）。回答模型是 `deepseek-flash`，严格裁判是 `deepseek-v4-pro`，四个系统用同一个回答模型和同一份 prompt：

| 系统 | 严格裁判准确率 | 95% 置信区间 | 证据召回 | 每题 prompt token |
|---|---:|---|---:|---:|
| `no_memory` | 0.0% | — | 0% | 256 |
| `naive_rag` | 49.7% | [44.0, 55.3] | 74.5% | 1,236 |
| **`hybrid_rag`** | **55.0%** | [49.3, 60.7] | 87.4% | 1,381 |
| `full_context` | 60.3% | [54.3, 66.0] | 99.7% | 21,310 |

- **hybrid_rag 对 naive_rag：+5.3 个点**，按题配对 bootstrap 95% CI [+1.3, +9.3]，p = 0.017。其中时间类题目 +6.9（p = 0.035）。
- **full_context 对 hybrid_rag：+5.3 个点**（p = 0.029），代价是每题 15 倍的 prompt token。也就是说，hybrid_rag 只用 **6.5% 的 token** 就拿到了全量上下文 **91% 的准确率**。
- **错误归因**：hybrid_rag 答错的 135 题里，只有 43 题是证据没召回全，另外 **92 题证据齐全却答错**。再细分：漏答要点 41%，多答了细节 33%，答错事实 11%，有证据仍拒答 8%，时间粒度和标准答案不一致 8%。现在的瓶颈在作答，不在检索。

完整数据见 [docs/results/end_to_end_s300.md](docs/results/end_to_end_s300.md)。

## 检索诊断（LoCoMo 全部 1,540 题，不调用 LLM）

| 配置 | 预算 | 证据召回 | multi-hop | temporal 答案可见 | 平均上下文字符 |
|---|---:|---:|---:|---:|---:|
| 朴素向量检索 `naive_rag` | 20 轮 | 76.4% | 57.3% | 37.1% | 3,831 |
| + BM25 RRF 融合 + 重排 | 20 轮 | 84.0% | 64.6% | 37.1% | 4,146 |
| + BM25 补充候选 + 重排 | 20 轮 | 86.4% | 71.6% | 36.1% | 4,283 |
| **`hybrid_rag` 默认（再加时间锚定）** | 20 轮 | **86.4%** | 69.8% | **70.4%** | 4,421 |
| **`hybrid_rag` 默认** | **5 轮** | 77.5% | 53.6% | 64.8% | 1,255 |

- 同样 20 轮，证据召回 **+10.0** 个点，多跳题 **+12.5** 个点。
- 只用 **1/4 的上下文**（5 轮），召回就超过了朴素检索用 20 轮的结果。
- 时间锚定让时间类问题的答案可见率从 36.1% 提到 **70.4%**，上下文只多 3.2%。
- 两个反直觉的发现：BM25 与向量做 RRF 融合**反而更差**（多跳题 -5.2 个点）；邻近轮扩展在同预算下也是负收益。

完整消融见 [docs/results/](docs/results/)。这些是作答之前的诊断指标，免费，用来先调好检索，再花钱跑端到端。

## 有什么

| 模块 | 内容 |
|---|---|
| 系统 | `no_memory`（下限）、`full_context`（上限）、`naive_rag`（朴素检索）、`hybrid_rag`（MemoCore 读取路径 + 时间锚定） |
| 写入 | 规则时间锚定（20 余条规则，"昨天 / 上周 / 去年夏天"换算成具体日期），每轮约 77 µs，0 次 LLM 调用 |
| 读取 | bge-m3 向量召回、BM25 补充候选（支持中文二元分词）、bge-reranker-v2-m3 精排、按预算选取，按时间顺序渲染并带日期 |
| 指标 | 与官方口径一致的 token F1；严格、宽松两套 LLM 裁判；证据召回；答案可见率；token、前缀缓存命中、延迟 |
| 统计 | bootstrap 95% 置信区间；`compare` 按题配对比较两个运行，给出差值区间和 p 值；人工抽检裁判一致率 |
| 工程 | 断点续跑；LLM 响应、向量、重排分数全部按内容寻址缓存；设置不一致拒绝续跑；`--mock` 离线全流程；GPU 半精度与批量编码；GitHub Actions 跑 56 个离线测试 |

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

数据集、bge-m3 和 bge-reranker-v2-m3 放在 `data/` 和 `models/` 下（这两个目录都不提交）。换一台机器时，先执行 `memocore download-locomo`；模型默认从 `models/` 读取，找不到时从 HuggingFace 下载。

### 1. 不花钱：先量检索

```bash
memocore retrieval-ablation --budgets 5,10,20
```

```bash
memocore retrieval-sweep --top-k 1,3,5,10,20 --chunk-turns 1,2,4
```

### 2. 离线检查流程

```bash
memocore run --system hybrid_rag --run-name hybrid-mock --max-conversations 1 --max-questions 20 --mock
```

### 3. 端到端评测（省额度的做法）

把 `.env.example` 复制为 `.env`，填入 `DEEPSEEK_API_KEY`。然后在固定种子的 300 题子集上跑：

```bash
memocore run --system hybrid_rag --run-name s300-hybrid_rag --max-questions 30
```

```bash
memocore run --system naive_rag --run-name s300-naive_rag --max-questions 30
```

```bash
memocore compare --baseline s300-naive_rag --candidate s300-hybrid_rag
```

结果写在 `runs/<run-name>/summary.md`；中途中断的话，用同样的命令重跑即可续上，已完成的题不会再调用。命令开始前会打印本次要调用多少次模型。

为了省额度，默认做了这些：

- **规则预判**：拒答、精确匹配这类一眼能判的情况不调用裁判模型。在 1,502 条已有判定上，规则结论和 LLM 裁判 100% 一致。像 `no_memory` 这种几乎总是拒答的系统，裁判调用基本可以全部省掉。
- **默认只跑严格裁判**：宽松裁判用 `--judge both` 再开。
- **前缀缓存预热**：每段对话先单独答一题，`full_context` 的缓存命中率从 36% 提到 96%。
- **用 300 题子集做配对对比**：差值显著性靠配对检验，不靠堆题量。
- **只重试失败项**：被截断、无法解析的回答和判定不进缓存，续跑时只重做这些。

### 4. 抽检裁判

```bash
memocore audit-export --run-name hybrid --kind strict --n 100
```

在导出的 CSV 的 `human_label` 列填 `CORRECT` / `WRONG`（也可以填 对 / 错），然后：

```bash
memocore audit-score --csv runs/hybrid/audit_strict.csv
```

## 实际花费（300 题子集，三个系统）

| 环节 | prompt token（其中命中缓存） | 输出 token |
|---|---:|---:|
| naive_rag 作答 | 37 万（4.6 万） | 4.2 万 |
| hybrid_rag 作答 | 41 万（5.9 万） | 3.7 万 |
| full_context 作答 | 639 万（615 万） | 4.2 万 |
| 三个系统的严格裁判 | 21 万 | 20 万 |

重排在笔记本 GPU 上每题约 200～300 ms，分数有缓存。

## 性能优化

在同一台 RTX 5060 笔记本上测得：

| 环节 | 优化前 | 优化后 | 做法 |
|---|---:|---:|---|
| 建索引（约 1 万段文本） | 143.4 s | 51.3 s | GPU 半精度（召回变化不超过 0.4 个点） |
| 单次检索 | 16.8 ms | 0.8 ms | 同一段对话的问题批量编码 |
| 重复完整扫描 | — | 3.5 s | 向量磁盘缓存全部命中 |
| 第二轮消融（8 种配置 × 3 档预算） | 516 s（第一轮） | 110 s | 重排分数内容寻址缓存 |

## 路线图

1. **冲突感知读取**：同一属性多次变化时，按时间排序返回并标注哪条最新（用 LongMemEval knowledge-update 题验证）。
2. **带质量门的派生索引**：LLM 抽取的实体卡片、事实，每条都必须能回链到原文轮次并通过忠实性校验。
3. **重排降本**：只在向量检索不确定时才重排。
4. **服务化**：REST 和 MCP 接口、多用户隔离、异步写入、k6 压测；接入 RepoPilot。
5. **更多基准**：LongMemEval 子集、LoCoMo-Refined 交叉评测；以 Mem0 作为外部对照。

## 目录

```text
src/memocore/
  datasets/locomo.py   数据加载，题型编号与官方一致
  temporal.py          写入时时间锚定（规则，不调用 LLM）
  retrieval/           BM25（含中文二元分词）、RRF 与候选池、交叉编码器重排
  systems/             no_memory、full_context、naive_rag、hybrid_rag
  prompts.py           所有系统共用的回答 prompt
  metrics.py           F1（复刻官方口径）、证据召回、答案可见率
  judge.py             严格 / 宽松裁判与输出解析
  runner.py            回答、裁判、汇总、人工抽检
  compare.py, stats.py 配对对比、bootstrap 置信区间
  retrieval_eval.py    不调用 LLM 的检索扫描与消融
  llm.py, embeddings.py, kvcache.py  模型客户端与各类缓存
docs/                  对比、评测方法、可借鉴项目、实测结果
tests/                 离线单元测试（mock 模型 + 哈希向量，不需要 API Key）
```
