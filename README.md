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

## 实测结果（LoCoMo 1,540 题，不调用 LLM 的检索诊断）

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

完整消融见 [docs/results/](docs/results/)。这些是回答之前的诊断指标，最终准确率要跑 LLM 回答加裁判，见下文的"真实评测"。

## 有什么

| 模块 | 内容 |
|---|---|
| 系统 | `no_memory`（下限）、`full_context`（上限）、`naive_rag`（朴素检索）、`hybrid_rag`（MemoCore 读取路径 + 时间锚定） |
| 写入 | 规则时间锚定（20 余条规则，"昨天 / 上周 / 去年夏天"换算成具体日期），每轮约 77 µs，0 次 LLM 调用 |
| 读取 | bge-m3 向量召回、BM25 补充候选（支持中文二元分词）、bge-reranker-v2-m3 精排、按预算选取，按时间顺序渲染并带日期 |
| 指标 | 与官方口径一致的 token F1；严格、宽松两套 LLM 裁判；证据召回；答案可见率；token、前缀缓存命中、延迟 |
| 统计 | bootstrap 95% 置信区间；`compare` 按题配对比较两个运行，给出差值区间和 p 值；人工抽检裁判一致率 |
| 工程 | 断点续跑；LLM 响应、向量、重排分数全部按内容寻址缓存；设置不一致拒绝续跑；`--mock` 离线全流程；GPU 半精度与批量编码；GitHub Actions 跑 55 个离线测试 |

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

### 3. 真实评测

把 `.env.example` 复制为 `.env`，填入 `DEEPSEEK_API_KEY`，然后先抽样跑一小轮：

```bash
memocore run --system hybrid_rag --run-name hybrid-smoke --max-conversations 2 --max-questions 25
```

```bash
memocore run --system naive_rag --run-name rag-smoke --max-conversations 2 --max-questions 25
```

```bash
memocore compare --baseline rag-smoke --candidate hybrid-smoke
```

确认没问题后再跑全量：去掉 `--max-*` 参数，换一个 `--run-name`，四个系统都跑一遍。结果写在 `runs/<run-name>/summary.md`；中途中断的话，用同样的命令重跑即可续上。

### 4. 抽检裁判

```bash
memocore audit-export --run-name hybrid --kind strict --n 100
```

在导出的 CSV 的 `human_label` 列填 `CORRECT` / `WRONG`（也可以填 对 / 错），然后：

```bash
memocore audit-score --csv runs/hybrid/audit_strict.csv
```

## 成本估算

- `full_context`：每题约 1.6 万～1.8 万 prompt token，全量约 2,500 万 token，其中大部分能命中前缀缓存。
- `naive_rag` / `hybrid_rag`（20 轮预算）：每题约 1 千 token。
- 每次裁判调用约 300～400 token，默认严格和宽松两套都跑。
- 重排：在笔记本 GPU 上每题约 200～300 ms，分数有缓存。

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
