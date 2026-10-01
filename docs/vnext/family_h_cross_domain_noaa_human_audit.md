# Family H 跨领域 NOAA 真实来源候选：详细人工审核要求

**适用对象：** MemUpdateBench Family H 跨领域 NOAA/NHC public-advisory 候选。
**候选阶段的历史状态：** `CANDIDATE_PENDING_HUMAN_REVIEW`。
**当前任务发布参考：** `data/vnext/family_h_cross_domain_noaa/v1`；本规范不表示需要重新审核已冻结的决定。
**本文件状态：** 审核规范，不是人工审核完成证明、数字签名或科学发布批准。

本要求参照：

```text
docs/vnext/family_h_independent_sources_human_audit.md
```

以及此前 matched-depth 审核文件的结构，但本候选的研究含义不同：它不是同一 synthetic core 的 matched-depth 对照，也不是软件 changelog 来源。本次审核对象是一个真实天气来源组、一条连续飓风公告轨迹、一个 canonical object、一个候选任务和七条公开公告事件。

---

## 1. 审核对象和证据边界

### 1.1 当前候选规模

```text
上游机构：National Hurricane Center / NWS
事件对象：2024 Beryl / AL022024
来源轨迹：1 条连续 public-advisory trajectory
语义核心：1
候选任务：1
audit items：9
公开事件：7
ADD：1
UPDATE：6
```

七条公告的风速值为：

```text
35 → 40 → 50 → 60 → 65 → 65 → 75 MPH
```

这不是七个独立来源，也不是统计独立样本。它是一条同一风暴、同一时间序列的来源轨迹。

### 1.2 当前候选状态

```text
status:                         CANDIDATE_PENDING_HUMAN_REVIEW
source_audit_status:            NOT_STARTED
generated_surface_audit_status: NOT_STARTED
human_approved_sources:         0
formal_task_release:            false
scientific_release_allowed:     false
answer_metrics:                 null
model_loads:                    0
generations:                   0
external_system_calls:          0
```

reference sanity 已通过，但它只证明 strict-v3 任务可以由 deterministic reference replay；不证明来源事实已获人工确认，也不证明 Qdrant、模型或自然语言操作解析已完成。

### 1.3 本候选和软件 Family H 的关系

本候选必须与已发布的软件 Family H 分开处理：

- 不把 NOAA 候选追加到 `data/vnext/family_h_independent/v1_published`；
- 不修改 Rust、CPython、Kubernetes 的正式 release index；
- 不把软件 Family H 的 33 项审核决定迁移到本候选；
- 不把 NOAA 的 7 条公告称为另一组独立软件来源；
- 不把本候选的天气观察任务与软件 release 任务合并统计。

即使本候选以后通过审核，也只能作为单独的跨领域 Family H 候选或后续 sibling release。

---

## 2. 审核文件和固定哈希

### 2.1 原始 capture 根

```text
external/family_h_cross_domain_noaa_20260928_v5/
```

包含：

```text
capture_manifest.json
capture_index.json
normalized_records.jsonl
raw/001.shtml
raw/002.shtml
raw/003.shtml
raw/003a.shtml
raw/004.shtml
raw/004a.shtml
raw/005.shtml
raw/nws_disclaimer.html
```

当前原始绑定：

```text
capture_manifest.json SHA-256:
773c28cdeae721fce84b66ba7bc26a31390cbbfb1020c93bd05614befa2d0aee

capture_index.json SHA-256:
d37a2a7a5ad5c0842fd1dae9846b3e2bb09264d2b238b31506b75766ff0be832
```

### 2.2 当前候选根

```text
results/vnext/family_h_cross_domain_noaa_candidate_20260928_v2/
```

候选 index：

```text
candidate index.json SHA-256:
1076de247f61e3895196d0fc9f601dea5faf82ade59441ab18650f4c6262e39d
```

主要候选 artifact：

```text
manifest.json
normalized_records.jsonl
semantic_cores.jsonl
tasks.jsonl
audit_manifest.json
decisions_template.json
reference_sanity.json
index.json
```

审阅者必须从上述候选和 capture 根读取，不能使用旧的：

```text
family_h_cross_domain_noaa_candidate_20260927_v1
family_h_cross_domain_noaa_beryl_20260927_v3
```

旧目录保留作为失败/诊断历史，不是当前审核对象。

### 2.3 当前审核项

当前 `audit_manifest.json` 有 9 项：

| 类型 | 数量 | 审核对象 |
|---|---:|---|
| `source_admission` | 1 | NHC 来源身份、公开领域政策、来源组语义 |
| `source_snapshot` | 7 | 每条 advisory 的时间、风速、编号和 raw anchor |
| `generated_surface` | 1 | 全部 7 条结构化事件、查询、对象键和最终值 |
| **合计** | **9** | **全量审核，不是抽样** |

当前没有自动生成或填写人工 decisions 文件。必须由人工审核者明确填写每一项决定。

---

## 3. 来源身份和公共领域政策

### 3.1 官方来源身份

每项来源准入审核必须确认：

```text
机构：National Hurricane Center / National Weather Service
风暴标识：AL022024
风暴名称：Beryl
年份：2024
来源类型：sequential_public_advisory
source_group_id：family-h-cross-domain-noaa-beryl-2024
source_document_id：nhc-al022024-public-advisories
```

官方 archive：

```text
https://www.nhc.noaa.gov/archive/2024/al02/
```

单条 advisory URL 必须和当前 `capture_manifest.json` 中的 URL 完全一致。不能把其他年份、其他风暴或二手天气网站的内容混入。

### 3.2 NWS 使用政策

保存的政策页：

```text
https://www.weather.gov/disclaimer
```

审核时须注意政策中的边界：

- NWS 网页信息通常属于 public domain，除非页面另有说明；
- 不得声称 NWS/NOAA 对本项目或本候选提供 endorsement 或 affiliation；
- 不得声称原始 NWS 内容是审核者或项目自己的版权作品；
- 改写后的材料不能伪装成官方原文；
- 第三方材料可能有单独许可；
- NWS 名称和视觉标识有商标/使用限制；
- 公开候选使用的是受限规范化事实和 hash-bound source anchor，不应复制整份 advisory 原文。

“Public domain”不等于可以把 NOAA/NWS 名义写成项目背书，也不等于所有嵌入页面的第三方内容自动可再分发。

### 3.3 来源准入决定必须覆盖

审核者须判断：

1. URL、机构、风暴 ID 与文件内容一致；
2. 七份原始 HTML 的 bytes/hash 与 capture index 一致；
3. disclaimer HTML 的 bytes/hash 与 capture index 一致；
4. 来源是公开网页，且没有发现需要额外访问权限的内容；
5. 项目对来源的描述没有超出 NWS 政策；
6. `source_audit_status` 仍为 `NOT_STARTED`，直到人工决定实际填写；
7. 本候选没有把 source capture 描述成数字签名、官方认证或独立签署者证明。

如果政策边界不能确认，使用 `NEEDS_FIX` 或 `UNSUPPORTED`，不能凭“政府网站”直接通过。

---

## 4. 七条 advisory 的逐条核对

### 4.1 当前时间线

| audit_id | advisory | UTC 时间 | 风速 |
|---|---|---|---:|
| `snapshot-001` | 001 | 2024-06-28T21:00:00Z | 35 MPH |
| `snapshot-002` | 002 | 2024-06-29T03:00:00Z | 40 MPH |
| `snapshot-003` | 003 | 2024-06-29T09:00:00Z | 50 MPH |
| `snapshot-003a` | 003a | 2024-06-29T12:00:00Z | 60 MPH |
| `snapshot-004` | 004 | 2024-06-29T15:00:00Z | 65 MPH |
| `snapshot-004a` | 004a | 2024-06-29T18:00:00Z | 65 MPH |
| `snapshot-005` | 005 | 2024-06-29T21:00:00Z | 75 MPH |

每条 snapshot audit item 都绑定：

```text
record_id
advisory_id
issued_at_utc
value
value_unit
source_anchor
raw_sha256
binding_sha256
```

### 4.2 时间解析要求

审核者须从原始 bulletin 和 `SUMMARY OF ... UTC ... INFORMATION` 区域核对：

- 本地 AST 时间；
- 对应 UTC 时间；
- 星期、日期和时间是否一致；
- 是否跨越了 UTC 日期边界；
- `003a`、`004a` 的 intermediate 身份；
- `001`、`003a` 是否带有 `Corrected` 文本；
- 日期不是抓取时间、Git 时间、文件修改时间或人工推断。

当前规范化 UTC 值：

```text
001: 2024-06-28T21:00:00Z
002: 2024-06-29T03:00:00Z
003: 2024-06-29T09:00:00Z
003a: 2024-06-29T12:00:00Z
004: 2024-06-29T15:00:00Z
004a: 2024-06-29T18:00:00Z
005: 2024-06-29T21:00:00Z
```

### 4.3 数值字段要求

每条 advisory 的 `MAXIMUM SUSTAINED WINDS` 区域必须唯一匹配：

```text
<integer> MPH ... <integer> KM/H
```

审核者须确认：

- MPH 数值和 KM/H 数值均来自同一 advisory 的 summary 区域；
- 当前 object value 是数值型 MPH，而不是字符串或整份公告对象；
- 任务 answer schema 是 `number`；
- KM/H 是辅助来源字段，不得误当成另一个 memory slot；
- location、advisory ID、issue time 作为 value/source anchor 的辅助字段时，仍必须与原文一致；
- 页面其他位置出现的预测描述、风暴叙述或旧值不能替代 summary 字段；
- 多处不一致时不能挑选看起来更合理的值。

### 4.4 003/003a 和 004/004a 的特殊要求

`003a` 和 `004a` 是 intermediate advisories，不是重复文件错误。必须确认：

- advisory ID 与标题编号一致；
- 时间不同；
- source URL 不同；
- raw SHA 不同；
- 004 和 004a 虽然都是 65 MPH，但不能删除其一；
- 同值 UPDATE 不应改成 NOOP，因为它代表新的外部观测公告；
- 这条序列是时间上的连续观察，不是“版本修订冲突”的强因果对照。

---

## 5. Canonical object、事件和任务审核

### 5.1 对象身份

当前唯一对象键：

```text
namespace:   family_h
entity:      hurricane-beryl-2024
attribute:   max_sustained_wind_mph
subkey:      null
object_type: weather_observation
```

审核者须确认：

- 七条事件使用完全相同的四部分 canonical key；
- `object_type` 只是分类元数据，不改变 identity；
- advisory 编号不被放入 `subkey`；
- 事件时间不被放入 `subkey`；
- 版本、地点和 advisory ID 不被误造为独立对象；
- 不能把天气轨迹和软件 Family H 对象合并。

### 5.2 ADD/UPDATE 序列

必须满足：

```text
ADD 35
UPDATE 40
UPDATE 50
UPDATE 60
UPDATE 65
UPDATE 65
UPDATE 75
```

审核时逐条检查：

- 第一个事件是 ADD；
- 后六个事件是 UPDATE；
- 事件顺序按 `issued_at_utc` 严格递增；
- `sequence_index` 连续；
- 每个 event 的 source anchor 指向正确 advisory；
- 没有提前注入最终 75；
- `004 -> 004a` 的 equal-value update 仍然是合法新观测，不是错误的重复 ADD；
- `source_heading_anchor`、`normalized_record_sha256` 和 `source_group_id` 与 audit material 一致。

### 5.3 任务查询和答案

当前 query 应限制为：

```text
Within the supplied public advisories, return the current wind value for object family_h|hurricane-beryl-2024|max_sustained_wind_mph|.
```

审核者须确认：

- query 询问的是 supplied public advisories 内的 current 值；
- query 没有询问现实世界今天的最新风速；
- query 没有出现 `gold`、隐藏 selector、预期答案或未来 advisory 值；
- answer schema 是 number；
- 最终 gold value 是 75，而不是 KM/H 120；
- reference replay 是确定性 sanity，不能被描述成模型准确率。

---

## 6. Generated surface 审核

本候选的 visible surface 是生成的结构化 JSON，不是逐字复制的 NHC prose，也不测试自然语言操作发现。

每条 event 的公开字段必须严格限制为：

```text
action
object_key
value
source_heading_anchor
normalized_record_sha256
```

审核者逐条检查：

1. action 为 ADD/UPDATE，且顺序正确；
2. object_key 与候选 task 的 canonical object 相同；
3. value 为整数型 MPH；
4. source anchor 和 advisory ID 对应；
5. normalized record hash 与 source snapshot 绑定；
6. 没有 raw HTML、隐藏答案、provider payload 或 model output；
7. 没有把原文中的自然语言动作声称成已测试的 operation discovery；
8. query 与 event surface 一致；
9. complete canonical task hash 和 semantic task hash 均绑定在 generated-surface audit item；
10. 修改任何 raw/normalized event text 都必须导致旧审核 binding 失效。

`generated_surface` 审核是全量 7 条事件，不是只看最终风速。

---

## 7. 当前 audit item 和决定字段

当前 `audit_manifest.json` 包含 9 项：

```text
source-noaa-beryl
snapshot-001
snapshot-002
snapshot-003
snapshot-003a
snapshot-004
snapshot-004a
snapshot-005
surface-noaa-beryl
```

决定文件尚未生成在本候选中；人工审核者应单独建立或由后续固定 review packet 提供：

```json
{
  "schema": "family-h-cross-domain-noaa-candidate-v2-human-decisions-v1",
  "candidate_index_sha256": "<current candidate index sha256>",
  "reviewer": null,
  "status": "NOT_STARTED",
  "decisions": [
    {
      "audit_id": "source-noaa-beryl",
      "binding_sha256": "<audit item binding>",
      "decision": null,
      "rationale": ""
    }
  ]
}
```

每一项决定只能使用：

```text
RELEASE_READY
NEEDS_FIX
UNSUPPORTED
```

每项 `rationale` 必须非空，说明实际核对了哪些 URL、字段、时间、hash 或 surface 条件。不要使用旧软件 Family H 的 33 项决定，也不要使用 matched-depth 的 `RELEASE_READY/REJECTED` schema。

不得批量自动填入 `RELEASE_READY`。审核者身份是自报信息，不是加密签名或独立签署者证明。

---

## 8. 不得通过的情况

| 问题 | 应有处理 |
|---|---|
| NHC URL、AL022024 或 advisory ID 不一致 | `NEEDS_FIX` |
| 日期无法从 bulletin/UTC summary 唯一确定 | `NEEDS_FIX` 或 `UNSUPPORTED` |
| 风速字段重复、缺失或来自预测段落 | `NEEDS_FIX` |
| 003a/004a 被删除或误标为重复 ADD | `NEEDS_FIX` |
| 同值 UPDATE 被错误替换为 NOOP | `NEEDS_FIX` |
| object key 随 advisory 改变 | `NEEDS_FIX` |
| value 是字符串、整份公告或错误使用 KM/H | `NEEDS_FIX` |
| 公开事件包含隐藏 gold/未来值/provider payload | `NEEDS_FIX` |
| public-domain 政策被描述为 NOAA endorsement | `NEEDS_FIX` |
| 原始 HTML、disclaimer 或 hash 漂移 | **停止审核并报告** |
| candidate/index/audit binding 不匹配 | **停止审核并生成新 sibling** |
| 只有 reference sanity、没有人工 source/surface 审核 | 保持 `BLOCKED` |
| 把 sequential observation 说成 revision-only causal series | `NEEDS_FIX` |
| 把一个风暴轨迹说成统计独立天气样本 | `NEEDS_FIX` |

技术阻塞、来源 unsupported 或未来没有运行，均不得转换成准确率 0。

---

## 9. 审核期间的不可修改边界

审核者不得直接修改：

```text
capture_manifest.json
capture_index.json
raw/*.shtml
raw/nws_disclaimer.html
candidate manifest/index/tasks
normalized_records.jsonl
audit_manifest.json
```

如果发现问题：

1. 对具体 audit item 填 `NEEDS_FIX` 或 `UNSUPPORTED`；
2. 说明具体 URL、字段、时间或事件；
3. 不手工改哈希或 candidate 内容；
4. 修复 parser/capture/compiler 后生成新的 no-replace sibling；
5. 重新计算 capture、candidate、surface 和 audit hashes；
6. 对新的 sibling 重新审核。

旧候选保留为诊断历史，不可通过重绑定变成当前候选。

---

## 10. 完成审核后的有限含义

如果 9/9 审核项均为 `RELEASE_READY`，最多表示：

- NOAA/NHC 来源身份和公共领域边界在当前声明下可接受；
- 七条 advisory 的时间、风速和顺序正确；
- 同槽天气观察任务的生成 surface 与来源绑定一致。

这不表示：

```text
NOAA 候选已正式发布
Family H broad benchmark 已闭合
天气轨迹具有统计独立性
自然语言操作发现能力已测试
Qdrant semantic retrieval 已测试
Qwen answer accuracy 已测试
跨领域 external validity 已经充分证明
```

正式 release、state/retrieval/answer 运行都需要后续独立证据层和新的 no-replace root。

---

## 11. 推荐审核顺序

1. 核对当前 capture 与 candidate 路径及三个 SHA-256；
2. 读取 NWS disclaimer，确认 public-domain 和 endorsement 限制；
3. 逐条打开 001、002、003、003a、004、004a、005 原文；
4. 核对 bulletin ID、AST 时间、UTC 时间、location 和 wind 字段；
5. 核对 35/40/50/60/65/65/75 数值序列；
6. 核对 ADD/UPDATE、object key、source anchor 和 task hash；
7. 审核全部 7 条 visible events，不只检查最后一条；
8. 填写 9 项 typed decisions 和具体 rationale；
9. 再执行候选校验脚本；
10. 只有完成人工审核后，才考虑新的 no-replace formal release。

---

## 12. 当前候选的机器验证命令

当前代码侧的最低检查：

```bash
python -B -m pytest tests/vnext/test_family_h_cross_domain_noaa.py -q -ra
python -m py_compile scripts/vnext_capture_family_h_noaa_candidate.py scripts/vnext_prepare_family_h_cross_domain_candidate.py
```

当前结果：

```text
17 passed
```

这些测试不能替代人工审核。它们只证明解析器、候选生成、hash binding、reference sanity 和 no-replace 边界在本地测试样例中成立。

---

## 最终文件位置

本人工审核要求文档位于：

```text
docs/vnext/family_h_cross_domain_noaa_human_audit.md
```
