# 与高星开源记忆项目对比：特点、痛点与创新

调研时间：2026-11-02。星数和 issue 数据来自 GitHub API；痛点取自各项目按评论数排序的 issue。

## 1. 主流方案怎么做

| 项目 | 星数 | 写入时做什么 | 存储 | 读取 |
|---|---:|---|---|---|
| [Mem0](https://github.com/mem0ai/mem0) | 68k | 调用 LLM 从对话中抽取事实，再与相似的旧记忆比对，决定新增、更新还是删除（论文中的流程） | 向量库（默认 Qdrant），可选图 | 向量检索 |
| [Graphiti（Zep）](https://github.com/getzep/graphiti) | 32k | 对每个 episode 调用 LLM 抽取实体和关系、去重；关系边带有效期，新事实出现时旧边失效 | 图数据库（Neo4j / FalkorDB） | 向量 + BM25 + 图遍历 |
| [Letta（MemGPT）](https://github.com/letta-ai/letta) | 25k | Agent 自己调用工具编辑记忆块；上下文满了就做压缩摘要 | 核心记忆 + 归档记忆 | Agent 主动检索 |
| [Cognee](https://github.com/topoteretes/cognee) | 33k | `cognify` 流水线用 LLM 构建知识图谱 | 图 + 向量 | 图检索 |
| [MemOS](https://github.com/MemTensor/MemOS) | 12k | 抽象成 MemCube，支持多种记忆类型 | 多种后端 | 混合检索 |
| [Memobase](https://github.com/memodb-io/memobase) | 2.9k | 按用户画像组织记忆 | 关系库 | 画像 + 事件 |

这些方案有一个共同点：**写入阶段用 LLM 把原始对话加工成另一种表示（事实、图、画像、摘要），之后检索的是加工结果，而不是原话。**

## 2. 社区反馈最多的痛点

| 痛点 | 证据（括号内为评论数） |
|---|---|
| 写入质量差：垃圾记忆、幻觉写入 | [mem0#4573](https://github.com/mem0ai/mem0/issues/4573) 审计 10,134 条记忆，称 97.8% 是垃圾（25）；[mem0#7283](https://github.com/mem0ai/mem0/issues/7283) 请求写入前做忠实性检查（34）；[graphiti#760](https://github.com/getzep/graphiti/issues/760) Hallucinations（10） |
| 不会处理时间，新旧事实冲突 | [mem0#5352](https://github.com/mem0ai/mem0/issues/5352) time-blindness、语义冲突（54，该仓库评论最多的 issue）；[mem0#4956](https://github.com/mem0ai/mem0/issues/4956) 只新增的抽取方式会给出过时或矛盾的事实（23） |
| 每次写入都调 LLM，成本高、速度慢 | [graphiti#290](https://github.com/getzep/graphiti/issues/290) 调用次数与速率限制（13）；[graphiti#450](https://github.com/getzep/graphiti/issues/450) 处理更正时 CPU 跑满（13）；[cognee#3516](https://github.com/topoteretes/cognee/issues/3516) 优化 cognify 的 LLM 调用（26） |
| 静默丢失、删不干净 | [mem0#5245](https://github.com/mem0ai/mem0/issues/5245) 批量编码部分失败时静默丢记忆（22）；[mem0#7452](https://github.com/mem0ai/mem0/issues/7452) `delete_all` 之后，旧消息仍会被发给 LLM（17）；[memobase#109](https://github.com/memodb-io/memobase/issues/109) 随机记不住新信息（12） |
| 摘要压缩丢信息 | [letta#3270](https://github.com/letta-ai/letta/issues/3270) 滑动窗口压缩把上下文整个清空（14） |
| 评测无法复现 | [mem0#2800](https://github.com/mem0ai/mem0/issues/2800) 本地复现不出 LoCoMo 分数（25）；[memobase#79](https://github.com/memodb-io/memobase/issues/79) LoCoMo 结果不一致（6）；[letta#3115](https://github.com/letta-ai/letta/issues/3115) 请求接入标准记忆基准（14） |
| 结果无法追溯出处 | [cognee#633](https://github.com/topoteretes/cognee/issues/633) 要求查询结果附带出处（24） |
| 中文支持差 | [MemOS#130](https://github.com/MemTensor/MemOS/issues/130) 中文输入处理不正确（6） |

前五类痛点有同一个根源：**LLM 加工是有损的，而加工结果又取代了原文。** 抽取错了、时间丢了、压缩过头了，原话都已经找不回来。

## 3. MemoCore 的定位：原文为本、零 LLM 写入、可信评测

| 设计 | 针对的痛点 | 做法 |
|---|---|---|
| **原文是唯一事实来源** | 写入质量、静默丢失、压缩丢信息、无法追溯 | 永远不改写、不删除原话。时间注释、索引这些派生信息挂在原文旁边，每条上下文都带对话轮次编号和日期，可以直接引用出处 |
| **零 LLM 写入** | 写入成本高 | 写入 = 向量编码 + BM25 索引 + 规则时间锚定，不调用 LLM |
| **写入时时间锚定** | 不会处理时间 | 把"昨天、上周、去年夏天"按会话日期换算成具体日期，作为注释放在原文后面。只用规则，不会编出原文没有暗示的日期 |
| **精排优先的读取** | 检索不准 | 交叉编码器重排带来的收益最大；BM25 只补充重排候选，不参与排名融合（实测融合会把语义命中挤掉） |
| **可信评测** | 评测无法复现 | 固定上下文预算；严格、宽松两套裁判；人工抽检；bootstrap 置信区间与配对检验；设置不一致拒绝续跑 |
| **中英文都能用** | 中文支持差 | bge-m3 多语言向量 + BM25 中文二元分词 |

### 诚实的边界

- 以原文为本，检索单元就会多。对话长到几十万轮时，需要再加分层索引（会话摘要、实体卡片），这在路线图里。
- 不做 LLM 抽取，就没有"用户画像"这类聚合视图。多跳题只能靠检索把分散的证据都找回来，目前 multi-hop 仍是短板。
- 以后加入的 LLM 加工（实体聚合、冲突消解）只作为派生索引，不改写原文，并且要用同一套评测证明它值得。

## 4. 实测（LoCoMo 1,540 题，不调用 LLM）

完整数据见 [results/retrieval_ablation.md](results/retrieval_ablation.md) 和 [第一轮](results/retrieval_ablation_round1.md)。下表比较同一上下文预算下的结果：

| 配置 | 预算 | 证据召回 | multi-hop | temporal 答案可见 | 平均上下文字符 |
|---|---:|---:|---:|---:|---:|
| 朴素向量检索（dense） | 20 轮 | 76.4% | 57.3% | 37.1% | 3,831 |
| dense + BM25（RRF 融合）+ 重排 | 20 轮 | 84.0% | 64.6% | 37.1% | 4,146 |
| dense + BM25（补充候选）+ 重排 | 20 轮 | 86.4% | 71.6% | 36.1% | 4,283 |
| **MemoCore 默认：再加时间锚定** | 20 轮 | **86.4%** | 69.8% | **70.4%** | 4,421 |
| **MemoCore 默认** | **5 轮** | 77.5% | 53.6% | 64.8% | 1,255 |

- **同样 20 轮，证据召回 +10.0 个点，多跳题 +12.5 个点。**
- **只用 1/4 的上下文**（5 轮，约 1/3 的字符）就超过了朴素检索用 20 轮的召回（77.5% 对 76.4%）。
- **时间锚定**：时间类问题的答案可见率从 36.1% 提到 70.4%，上下文只多 3.2%，证据召回不变。
- **写入成本**：时间锚定加 BM25 建索引每轮约 77 µs，0 次 LLM 调用；全部 5,882 轮里有 491 轮（8.3%）被加上时间注释。向量编码在 RTX 5060 半精度下约 5.6 ms/轮。
- **代价**：交叉编码器重排在笔记本 GPU 上约 165～250 对/秒，即每题 50 个候选要 200～300 ms（结果有缓存）。降本方案见下一节。

两个反直觉的发现：

1. **BM25 和向量做 RRF 融合反而更差**：同为 20 轮，比单用向量低 1.6 个点，多跳题低 5.2 个点。原因是字面匹配会把语义相关的轮次挤出前列。改成"BM25 只往重排候选池里补 10 个"后，多跳题比 RRF 融合高 7.0 个点。
2. **邻近轮扩展在同预算下是负收益**（第一轮消融：-0.7～-5.9 个点）。给每个命中带上前后轮会挤占预算，少放进来的独立命中比多出来的上下文更有价值。

注意：证据召回和答案可见率都只是 LLM 回答前的诊断指标。端到端的结果如下：300 题、deepseek-flash 作答、deepseek-v4-pro 严格裁判，详见 [results/end_to_end_s300.md](results/end_to_end_s300.md)。

| 系统 | 严格裁判准确率 | 每题 prompt token |
|---|---:|---:|
| 朴素向量检索 | 49.7% | 1,236 |
| **MemoCore（hybrid_rag）** | **55.0%**（比朴素检索 +5.3，p = 0.017） | 1,381 |
| 全量上下文 | 60.3% | 21,310 |

## 5. 下一步创新（按痛点优先级）

| 方向 | 针对的痛点 | 怎么验证 |
|---|---|---|
| 冲突感知读取：同一属性多次变化时（搬家、换工作），按时间排序返回并标注哪条最新 | 过时事实（mem0#4956、#5352） | LongMemEval 的 knowledge-update 题 |
| 写入质量门：LLM 抽取出的每条记忆都必须能回链到原文轮次，并通过忠实性校验，否则不写入 | 幻觉写入（mem0#7283、graphiti#760） | HaluMem 的按操作评测 |
| 实体卡片索引：跨会话聚合同一实体出现过的轮次 | multi-hop 证据分散 | multi-hop 题的证据全部召回占比 |
| 带引用的回答：答案附带轮次编号 | 无法追溯（cognee#633） | 人工抽检时核对引用是否正确 |
| 重排降本：只在向量检索结果不确定时才重排 | 重排约 200 ms/题 | 召回与延迟的取舍曲线 |
