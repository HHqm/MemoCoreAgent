# 裁判抽检：跨模型盲审

**先说清楚：这次的参照标签不是人标的，是 Claude（Opus 5.5）标的。** 它和裁判 deepseek-v4-pro 不是同一家的模型，逐题标注时看不到裁判结论，也看不到答案来自哪个系统，所以能独立检验裁判；但两个 LLM 的一致率不等于人工一致率。下面 20 道分歧题全部列出，人工复核只需要看这 20 题。

逐题标签见 [judge_audit_claude.csv](judge_audit_claude.csv)（`labeler` 列为 `claude-opus-5.5`），汇总见 [judge_audit_claude.json](judge_audit_claude.json)。人工标注用的原表 `runs/audit_strict.csv` 保持空白，有人愿意标时可以直接用 `memocore audit-label`。

## 做法

```bash
memocore audit-export --run-name s300-full_context s300-naive_rag s300-hybrid_rag s300-hybrid_rag-v3 --n 100
memocore audit-score --csv runs/audit_strict_claude.csv
```

- **样本**：四个 s300 运行轮流分配，抽 100 道不重复的题（full_context 26、naive_rag 26、hybrid_rag 24、hybrid_rag v3 24）；单跳 54、时间 20、多跳 19、开放域 7。
- **只抽 LLM 裁判判定的题**：规则判定的题（精确匹配、拒答）占全部判定的 33.5%，按构造就是对的，不抽。
- **盲标**：标注表里没有裁判结论和系统名。标注前只看过样本里裁判判对 / 判错的总数，没看过逐题结论。
- **标准**：和严格裁判同一套规则（覆盖、不矛盾、不越界、时间粒度）。几条解释口径：多出一个独立的列表项或实体（多答一个州、多答车型）算越界；只是对同一事物的补充描述（"since age ten"、"on an awesome beach"）不算；事物答对但少了修饰语（"once a month"、"Sennheiser"）算对，并标为边界；比标准答案更精确、且落在其范围内的时间算对，只给截止时间（"by 3 April"）算错。

## 结果

| 指标 | 结果 |
|---|---|
| 一致率 | **80%**（95% Wilson 区间 71.1%～86.7%） |
| Cohen's kappa | 0.61 |
| 裁判偏松（裁判判对、盲审判错） | **0 题** |
| 裁判偏严（裁判判错、盲审判对） | **20 题** |
| 同一批 100 题的准确率 | 裁判 42%，盲审 62% |
| 含规则判定在内的整体一致率（估计） | 约 87%（规则判定占 33.5%，按全部正确计） |

按系统看（每个约 25 题，区间很宽，只能看方向）：

| 系统 | 题数 | 裁判准确率 | 盲审准确率 | 差 |
|---|---:|---:|---:|---:|
| naive_rag | 26 | 42% | 50% | +8 |
| full_context | 26 | 58% | 77% | +19 |
| hybrid_rag（v1） | 24 | 29% | 58% | +29 |
| hybrid_rag（v3） | 24 | 38% | 62% | +25 |

按题型的一致率：多跳 84%，开放域 86%，单跳 78%，时间 80%。

## 20 道分歧题

| 类型 | 题数 | 例子（标准答案 → 回答 → 裁判理由） |
|---|---:|---|
| 加了与题目相符的补充细节，被判"越界" | 11 | Prius → "A new Prius" → "加了没被支持的 new"；Miami → "Miami (on an awesome beach)"；July 11, 2022 → "Evening of 11 July 2022"；June 2023 → "The week before 21 June 2023" |
| 事物答对，少了修饰语 | 5 | "…dark chocolate with high cocoa content" → "…dark chocolate"；"Sennheiser headphones, …" → "New headphones, …" |
| 合并或日期换算有出入 | 2 | "first week of April 2022" → "The week before 12 April 2022"；"hanging out at the park, having picnics" → "park picnics" |
| 既少了一点又多了一点 | 2 | "Appearance of a woman he saw during a walk" → "A stranger he saw while walking his dogs" |

逐题清单（题号对应 csv 的 `item` 列）：9、14、15、27、28、33、35、43、45、56、60、63、76、79、81、86、91、92、93、99。

后三类在字面规则下裁判也说得通；第一类（11 题）是裁判把"不越界"读得过死。

## 结论

1. **严格裁判不放水，报告的准确率是保守下限。** 100 题里没有一题是裁判判对、盲审判错。hybrid_rag 的 55.0% 不是裁判宽松给出来的。
2. **偏严集中在"不越界"这条规则。** 回答多带一个与题目相符的细节（"new"、"evening"、"on a beach"）也会被判错。这也意味着之前错误归因里"多答了细节 33%"有一部分其实是裁判过严，而不全是作答的问题；v2 prompt 里"不要多写"的规则，某种程度上是在迎合裁判。
3. **系统间的比较没有被裁判放大。** 偏严程度在上下文更丰富的系统上更明显（hybrid +29、full_context +19，naive 只有 +8），时间锚定给出的更精确日期也会被判"越界"（4 道时间类分歧题里 3 道来自 hybrid）。所以严格裁判下 hybrid 对 naive 的 +5.3 个点，更可能是低估而不是高估。每个系统只有约 25 题，这一条只能看方向。

## 还没做

- **人工复核这 20 道分歧题**：约 3 分钟，就能把"跨模型一致率"升级为人工结论。
- **宽松裁判对照**：对同样 100 题跑宽松裁判（约 100 次 deepseek-v4-pro 调用），看它和盲审是否更一致。
- **收紧"越界"的定义**：只把多出的独立列表项、实体和可能错误的断言算越界。改裁判规则后，所有系统都要重判才能比较，按规则预判算约 800 次调用，先不花这笔额度。
