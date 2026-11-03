# 可借鉴的开源项目

数据来自 GitHub，2026-11-02 查询，星数会变。“借鉴什么”是针对本项目的建议，项目自身的说法以其 README 为准。

## 先说结论

- 记忆赛道已经很拥挤：不少项目在 README 里自称在 LoCoMo / LongMemEval 上达到 90% 以上。但它们用的裁判、题目子集和题型取舍各不相同，数字不能直接比较。
- LoCoMo-Refined 用同一批系统输出重新打分，严格裁判下各家掉了 15～22 个点。
- 所以本项目不以刷榜为目标，而是做到：**同一口径、可复现、同时报两种裁判，并公开裁判抽检的一致率和全部分歧题**。这是面试时最站得住的部分。

## 评测基准（直接使用）

| 项目 | 星数 | 用途 |
|---|---:|---|
| [snap-research/locomo](https://github.com/snap-research/locomo) | 1.3k | LoCoMo 数据集和官方评分脚本。本项目的题型编号和 F1 计算照搬官方脚本。 |
| [mem-eval-suite/LoCoMo_refined](https://github.com/mem-eval-suite/LoCoMo_refined) | 677 | 修订了 337 道有问题的题，并换成更严格的裁判。据其报告，原始裁判与人工标注的一致率只有 43.67%，新裁判为 86.33%（官方裁判模型是 Qwen3-14B）。本项目的 strict 裁判按它的原则编写，`qa_id` 格式也与它兼容，以后可以交叉评测。数据许可证是 CC BY-NC 4.0。 |
| [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval) | 1.2k | ICLR 2025。考五类能力：信息抽取、跨会话推理、时间推理、知识更新、拒答。下一阶段接入。另有 LongMemEval-V2（201★）。 |
| [MemTensor/HaluMem](https://github.com/MemTensor/HaluMem) | 179 | 按操作分别评测记忆幻觉：抽取、更新、问答各自打分。适合直接评测"写入路径"，而不只看最终答题。 |
| [Fengzdadi/memory-bench-registry](https://github.com/Fengzdadi/memory-bench-registry) | 210 | 记忆类基准的索引，每个字段都标了出处。 |

## 主流记忆系统（读设计、做对照）

| 项目 | 星数 | 借鉴什么 |
|---|---:|---|
| [mem0ai/mem0](https://github.com/mem0ai/mem0) | 68k | 最主流的方案，写入流程是：LLM 抽取事实，与相似的旧记忆比对，再决定新增、更新、删除或不变（ADD / UPDATE / DELETE / NOOP）。本项目的写入路径以此为起点，Mem0 本身也作为对照基线。 |
| [getzep/graphiti](https://github.com/getzep/graphiti) | 32k | 双时间轴的时序知识图谱：同时记录事件时间和写入时间。新事实出现时，旧边被标记为失效而不是删除。时间建模和冲突处理都可以参考它。 |
| [letta-ai/letta](https://github.com/letta-ai/letta) | 25k | 前身是 MemGPT。记忆分层：核心记忆常驻上下文，归档记忆按需检索；Agent 通过调用工具自己编辑记忆。 |
| [topoteretes/cognee](https://github.com/topoteretes/cognee) | 33k | 把文档和对话加工成知识图谱的流水线。 |
| [MemTensor/MemOS](https://github.com/MemTensor/MemOS) | 12k | "记忆操作系统"：混合检索、跨任务复用技能。 |
| [memodb-io/memobase](https://github.com/memodb-io/memobase) | 2.9k | 以用户画像为中心组织记忆，适合偏好类信息。 |
| [langchain-ai/langmem](https://github.com/langchain-ai/langmem) | 1.7k | LangGraph 生态的记忆工具，和 RepoPilot 技术栈一致，接入时可参考它的接口。 |

## 研究原型（读思路）

| 项目 | 星数 | 思路 |
|---|---:|---|
| [agiresearch/A-mem](https://github.com/agiresearch/A-mem) | 1.2k | 卡片盒笔记法（Zettelkasten）式的记忆：记忆之间互相链接，并随新信息演化。 |
| [BAI-LAB/MemoryOS](https://github.com/BAI-LAB/MemoryOS) | 1.6k | EMNLP 2025 Oral。短期、中期、长期三层存储，以及各层之间的更新策略。 |
| [nemori-ai/nemori](https://github.com/nemori-ai/nemori) | 217 | 按人类"情节记忆"的粒度切分对话，而不是按固定轮数切。 |
| [RinDig/cost-of-remembering](https://github.com/RinDig/cost-of-remembering) | 40 | 据其报告，文件系统式记忆在 LongMemEval 上和长上下文准确率相当，读取的 token 少 97%。它提醒我们：简单方案也要当基线，别默认非用向量库不可。 |

## 入门与索引

- [NirDiamant/Agent_Memory_Techniques](https://github.com/NirDiamant/Agent_Memory_Techniques)（1.1k★）：30 个可运行的 notebook，覆盖缓冲区、向量库、知识图谱、MemGPT、Mem0、Zep 等，入门最快。
- [IAAR-Shanghai/Awesome-AI-Memory](https://github.com/IAAR-Shanghai/Awesome-AI-Memory)（1.3k★）、[TeleAI-UAGI/Awesome-Agent-Memory](https://github.com/TeleAI-UAGI/Awesome-Agent-Memory)（700★）：论文、系统、基准的汇总。
