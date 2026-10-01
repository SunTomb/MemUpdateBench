# Family H Cross-Domain BEA Candidate：详细人工审核要求

**适用对象：** MemUpdateBench Family H BEA 2024 Q4 release-vintage candidate。
**文档状态：** 历史审核规范，不是科学发布批准。
**当前任务发布参考：** `data/vnext/family_h_cross_domain_bea_gdp/v1`；本规范不表示需要重新审核已冻结的决定。
**当前 candidate：** `family_h_cross_domain_bea_gdp_20260930_v4_policy_verified`。

本文件参照 `family_h_cross_domain_noaa_human_audit.md` 和 `family_h_independent_sources_human_audit.md` 的结构，但本候选的科学含义不同：它不是天气观测序列，也不是普通的连续经济时间序列，而是**同一季度、同一指标在三个官方 BEA release vintages 中的 successive estimate/revision trajectory**。

核心限制必须贯穿审核：

```text
这是一个 bounded same-quarter revision-vintage candidate。
不是三个统计独立样本。
不是 2024 Q4、2025 Q1、2025 Q2 的连续经济时间序列。
不是宏观经济预测准确率实验。
不是模型、Qdrant 或回答层结果。
```

---

## 1. Candidate 范围和当前状态

### 1.1 固定轨迹

| 项目 | 固定值 |
|---|---|
| source group | `family-h-cross-domain-bea-gdp-2024q4` |
| source document | `bea-gdp-2024q4-revision-trajectory` |
| domain | `macroeconomics` |
| language | `en` |
| source kind | `sequential_release_vintage` |
| source type | `other` |
| reference period | `2024-Q4` |
| trajectory semantics | same-quarter estimate revisions |
| semantic cores | 1 |
| tasks | 1 |
| snapshots/events | 3 |
| operations | `ADD + UPDATE + UPDATE` |
| independent samples | `false` |
| revision-only | `true` |

### 1.2 三个 release vintages

| Operation | Estimate stage | Release date | Real GDP value |
|---|---|---:|---:|
| `ADD` | Advance estimate | 2025-01-30 | 2.3 |
| `UPDATE` | Second estimate | 2025-02-27 | 2.3 |
| `UPDATE` | Third estimate | 2025-03-27 | 2.4 |

`2.3 → 2.3` 必须保留为一次 `UPDATE`。第二次发布是新的官方 estimate vintage；数值四舍五入后相同，不代表 `NOOP`，也不代表没有来源变化。

### 1.3 Candidate flags

```text
status:                         CANDIDATE_PENDING_HUMAN_REVIEW
source_audit_status:            NOT_STARTED
generated_surface_audit_status: NOT_STARTED
policy_status:                  POLICY_VERIFIED_PUBLIC_DOMAIN_WITH_ATTRIBUTION_BOUNDARY
formal_task_release:            false
scientific_release_allowed:     false
answer_metrics:                 null
model_loads:                    0
generations:                    0
provider_calls:                 0
```

已完成的 deterministic reference sanity 只说明任务合同、strict-v3 replay 和 reference adapter 在当前输入上成立。它不替代来源事实、表格字段、发布语义或许可政策的人审。

### 1.3 Policy evidence update

BEA 的官方 Linking Policy 已单独捕获并绑定：

```text
external/family_h_cross_domain_bea_policy_20260930_v1/linking.html
SHA-256: e49a712b7e0ec2e8946e256315f4af675af72c29c8d9e019c46626fa56b5a79e
```

该政策说明 BEA 网站信息除非另有说明属于 public domain，可以无需特别许可使用或复制，并建议引用 `Source: U.S. Bureau of Economic Analysis`。同时保留 non-endorsement、BEA logo 和第三方材料限制。因此当前候选的政策状态更新为 `POLICY_VERIFIED_PUBLIC_DOMAIN_WITH_ATTRIBUTION_BOUNDARY`，而不是无限制再发布许可。

由于 capture、manifest 和 candidate bytes 随政策证据变化，之前针对旧 candidate 的决定不能直接重新绑定。新的审核工作区为：

```text
results/vnext/family_h_cross_domain_bea_review_20260930_v3_policy_verified/
```

需要对新的 candidate index 和新的 5 项 audit bindings 重新填写决定；在此之前不得正式 promotion。


### 1.4 与其他 Family H 结果的边界

本候选不得并入：

```text
data/vnext/family_h_independent/v1_published
 data/vnext/family_h_cross_domain_noaa/v1
```

Rust/CPython/Kubernetes 软件来源和 NOAA 天气来源已有各自独立 release boundary。BEA 是新的宏观经济来源族，必须建立自己的 candidate、review、release 和后续 runtime roots。

---

## 2. 审核材料和哈希锚点

### 2.1 Current roots

Capture root：

```text
external/family_h_cross_domain_bea_gdp_20260930_v6_policy_verified/
```

Candidate root：

```text
results/vnext/family_h_cross_domain_bea_gdp_20260930_v4_policy_verified/
```

Candidate audit instructions：

```text
docs/vnext/family_h_cross_domain_bea_human_audit.md
```

### 2.2 Hash bindings

```text
capture_manifest.json:
9395b50167b4b47f64e19b9ebcbbdafe6c3ba70e1084ebb9a273bc3ba179eb1b

capture_index.json:
17aa9bd3a576effa7b30b79f87e6090e07edf58839594518f808049ac11e6313

candidate index.json:
2a12b35dd2972f98d6b90f8cb9bfea46a60924e910ac94cab37c79756a6f86cd
```

候选主要文件：

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

这些是内容绑定，不是数字签名或人工身份认证。GitHub/BEA URL、download URL、文件 SHA-256、record hash、semantic task hash、canonical task hash 和 audit binding hash 各自属于不同层，不得互相代替。

### 2.3 正确的审核工作区

当前模板位于：

```text
results/vnext/family_h_cross_domain_bea_gdp_20260930_v4_policy_verified/decisions_template.json
```

审核者不应直接覆盖 candidate 内的 `decisions_template.json`。应创建单独 review root，例如：

```text
results/vnext/family_h_cross_domain_bea_review_20260930_v3_policy_verified/decisions.json
```

单独 review 文件必须保留：

- 正确的 candidate index SHA-256；
- 顶层 reviewer 标识；
- 5 个原始 audit ID；
- 原始 `binding_sha256`；
- 3 种允许的 typed decision；
- 每项非空 rationale。

不要把审核决定写回 candidate 的任务、manifest、audit manifest 或 source capture。

---

## 3. BEA 来源准入审核

### 3.1 来源身份

本 candidate 只接受以下官方 BEA 直接材料：

#### Advance estimate

- Release date：2025-01-30
- Tables XLSX：<https://www.bea.gov/sites/default/files/2025-01/gdp4q24-adv.xlsx>
- Release page：<https://www.bea.gov/news/2025/gross-domestic-product-fourth-quarter-and-year-2024-advance-estimate>

#### Second estimate

- Release date：2025-02-27
- Tables XLSX：<https://www.bea.gov/sites/default/files/2025-02/gdp4q24-2nd.xlsx>
- Release page：<https://www.bea.gov/news/2025/gross-domestic-product-4th-quarter-and-year-2024-second-estimate>

#### Third estimate

- Release date：2025-03-27
- Tables XLSX：<https://www.bea.gov/sites/default/files/2025-03/gdp4q24-3rd.xlsx>
- Official release document：<https://www.bea.gov/sites/default/files/2025-03/gdp4q24-3rd.pdf>

官方 release schedule：

<https://www.bea.gov/news/schedule/full-2025>

### 3.2 Source of record

审核时以日期绑定的 BEA Tables XLSX 为主要 source of record，release page/PDF 作为发布身份、日期、估计阶段和 revision language 的辅助证据。

不能使用：

```text
新闻媒体摘要
第三方经济网站
搜索结果摘要
仅 release title，不看表格
没有保存 vintage 的当前 BEA API 查询
年度 2024 GDP 数值
current-dollar GDP
GDI
GDP-by-industry
```

BEA API 不是本候选 v1 的 source of record。当前 API 可能反映后续修订，不能自动代表历史 release vintage。

### 3.3 许可和再分发政策

当前 policy evidence 已绑定于：

<https://www.bea.gov/about/policies-and-information/linking>

本地 capture：

```text
external/family_h_cross_domain_bea_policy_20260930_v1/linking.html
SHA-256: e49a712b7e0ec2e8946e256315f4af675af72c29c8d9e019c46626fa56b5a79e
```

该官方 Linking Policy 明确说明：

- BEA 网站信息除非另有说明属于 public domain；
- 可以使用或复制，无需特别许可；
- `Source: U.S. Bureau of Economic Analysis` 的 citation would be appreciated；
- 不得暗示 BEA 对项目、组织或商业产品提供 endorsement/affiliation；
- BEA logo 不得用于链接识别之外的其他目的；
- BEA 不能授权外部链接中的 copyrighted materials。

候选因此使用：

```text
policy_status = POLICY_VERIFIED_PUBLIC_DOMAIN_WITH_ATTRIBUTION_BOUNDARY
```

这不是无限制再发布许可。公开 release 只复制规范化事实、source anchors、哈希绑定元数据和 `SOURCE_ATTRIBUTION.txt`，不复制原始 XLSX/PDF。派生结构化记录必须说明不是官方 BEA 产品，并保留来源 attribution 和 non-endorsement 边界。

由于 policy evidence 改变了 capture、manifest、task provenance 和 audit bindings，旧的 `NEEDS_FIX`/`RELEASE_READY` 决定不能重绑定。当前需要对新的 candidate v4 重新完成 5 项人工审核。
---

## 4. 每个 snapshot 的表格审核

当前 audit manifest 中的三个 snapshot 项为：

```text
snapshot-advance
snapshot-second
snapshot-third
```

每一项都必须回到对应的 XLSX 文件和 release document 检查。审核者至少核对以下表格条件：

```text
worksheet: Table 1
reference period: Q4 2024
row/measure: Real GDP
measure meaning: percent change from Q3 to Q4
rate basis: seasonally adjusted at annual rates
unit: percent
value precision: one decimal place
```

候选保存的表格 anchor 当前包含：

```text
worksheet = Table 1
row = 5
column = 21
```

审核者必须确认这两个坐标在对应 XLSX 中确实指向 `Real GDP`、`Q4 2024` 和正确的 value，而不能只相信坐标字段本身。

### 4.1 Advance

必须核对：

```text
stage: advance
release date: 2025-01-30
value: 2.3
```

### 4.2 Second

必须核对：

```text
stage: second
release date: 2025-02-27
value: 2.3
```

特别注意：BEA 叙述中的修订可能是“小于 0.1 percentage point”，而表格保留一位小数后仍为 2.3。不得为了匹配叙述而发明未公开的精度；也不得把四舍五入相同误判为没有新的 release observation。

### 4.3 Third

必须核对：

```text
stage: third
release date: 2025-03-27
value: 2.4
```

第三期材料使用官方 XLSX，并以官方 PDF 作为对应 release document。不要因为原先某个 HTML URL 返回 404，就使用非官方二手页面替代；候选 capture 已绑定实际可获取的官方文件。

### 4.4 Snapshot rejection conditions

以下任一情况都不能通过对应 snapshot：

- worksheet 不是 `Table 1`；
- 找不到唯一 `Real GDP` measure row；
- 找不到唯一 `Q4 2024` period column；
- 表格含义不是从 Q3 到 Q4 的 annualized percent change；
- 值来自 annual 2024 GDP 或 current-dollar GDP；
- 值为字符串、缺失、suppressed、NaN 或无法确认精度；
- 值不是该 release 的一位小数报告值；
- release stage、日期、URL 或源文件 hash 不匹配；
- 发生同一 snapshot 的重复或冲突行；
- 人工通过当前 BEA API 值覆盖了已捕获的 dated XLSX。

---

## 5. Release-vintage 语义审核

### 5.1 这不是现实经济时间序列

三条记录的 reference period 都是：

```text
2024-Q4
```

变化的是：

```text
官方发布时间 / estimate vintage / release stage
```

因此不能写成：

```text
Q4 2024 → Q1 2025 → Q2 2025 GDP trajectory
```

正确描述是：

```text
three official estimate vintages for the same 2024-Q4 Real GDP measure
```

### 5.2 ADD/UPDATE 规则

候选必须满足：

```text
ADD 2.3  (advance, 2025-01-30)
UPDATE 2.3 (second, 2025-02-27)
UPDATE 2.4 (third, 2025-03-27)
```

审核者必须确认：

- 三个事件都使用同一个 canonical object key；
- estimate stage 和 release date 留在 source/metadata/anchor 中；
- `advance`、`second`、`third` 不进入 `subkey`；
- 第二个 2.3 是新 vintage 的 UPDATE，不是 NOOP；
- 2.4 只在 third release 中出现，不提前泄漏到前两个事件；
- final value 是 2.4；
- 任务没有把三个 release 当成统计独立样本。

### 5.3 Object identity

精确对象键：

```text
namespace:   family_h
entity:      us-gdp-2024q4
attribute:   real_gdp_growth_annual_rate
subkey:      null
object_type: macroeconomic_estimate
```

`object_type` 是分类元数据，不参与四部分 identity。period、estimate stage、release date、unit 和 rate basis 不能通过改变 object key 来规避同槽更新。

---

## 6. Generated surface、查询和答案边界

### 6.1 Public event surface

公开事件应是结构化事件，不是 BEA 原始表格全文或 release prose。审核者需核对：

- 第一条事件为 `ADD`，后两条为 `UPDATE`；
- value 是 scalar number `2.3 / 2.3 / 2.4`；
- unit/rate/reference-period/release-stage 留在 source-normalized metadata 和 source anchor；
- 没有把未来 third value 提前写入 advance/second event；
- 没有复制受限 raw XLSX、PDF 或不必要的完整表格内容；
- 没有 API key、credential-bearing URL、hidden gold 或 provider payload；
- source record hash 和表格坐标与 snapshot audit 项一致。

### 6.2 Query

当前查询语义应限制在三个 supplied BEA release vintages：

```text
Within the supplied BEA release vintages for 2024-Q4, return the current value for object family_h|us-gdp-2024q4|real_gdp_growth_annual_rate|.
```

必须确认：

- query 没有询问现实世界当前 GDP；
- query 没有写入正确答案 2.4；
- answer schema 为 `number`；
- query 对象与事件 object key 一致；
- task 没有声称测试自然语言操作发现。

### 6.3 完整 surface binding

Generated-surface audit 必须绑定：

```text
canonical_task_sha256
semantic_task_sha256
event_count
query text
```

如果只修改事件 raw/normalized text 而 semantic hash 未变，完整 canonical task binding 仍应失效。审核者不能只核对 task ID 或 semantic hash。

---

## 7. 5 项 audit item 和决定字段

当前 audit manifest 正好包含 5 项：

```text
source-bea-gdp-2024q4
snapshot-advance
snapshot-second
snapshot-third
surface-bea-gdp-2024q4
```

| 类型 | 数量 | 审核内容 |
|---|---:|---|
| `source_admission` | 1 | BEA 身份、官方文件、reuse policy、vintage 语义 |
| `source_snapshot` | 3 | 每个 XLSX/PDF 的表格 row/column/value/date/stage/hash |
| `generated_surface` | 1 | ADD/UPDATE/UPDATE、scalar value、key/query、完整 task hash |
| **合计** | **5** | **全量审核** |

当前 `decisions_template.json` 必须保留：

```text
reviewer: null
status: NOT_STARTED
decision: null
rationale: ""
```

允许的 decision：

```text
RELEASE_READY
NEEDS_FIX
UNSUPPORTED
```

本次不能使用 matched-depth 的 `REJECTED` 字段，也不能把软件 Family H 的 33 项决定复制过来。

每个 rationale 必须说明实际核对的来源、表格字段、hash、release stage、值、时间线或 surface；空泛的“已检查”不足以作为完整审核理由。

---

## 8. 不得通过的情况

| 问题 | 正确处理 |
|---|---|
| BEA 文件、release stage 或日期不匹配 | `NEEDS_FIX` |
| 政策/再分发边界未确认 | `NEEDS_FIX` 或 `UNSUPPORTED` |
| 表格 row/column/period/unit/annualization 不明确 | `NEEDS_FIX` |
| 使用年度 GDP、current-dollar GDP、GDI 或其他指标 | `NEEDS_FIX` |
| 把 2.3→2.3 合并为 NOOP | `NEEDS_FIX` |
| 把同季度三个 vintage 描述成三个独立样本 | `NEEDS_FIX` |
| 把发布阶段放进 subkey 形成不同对象 | `NEEDS_FIX` |
| 2.4 被提前泄漏到前两个事件 | `NEEDS_FIX` |
| raw XLSX/PDF、API key 或 hidden gold 进入公开 task | `NEEDS_FIX` |
| capture/candidate/index/audit hash 不符 | 停止审核，生成新的 sibling |
| 只有 reference sanity、没有人工来源检查 | 保持 `BLOCKED` |
| 候选被描述成 GDP 预测准确率或 broad economic benchmark | `NEEDS_FIX` |

Unsupported、技术失败、来源政策阻塞和模型未运行不得转为准确率 0。

---

## 9. 审核期间不可修改边界

不得直接修改：

```text
external/family_h_cross_domain_bea_gdp_20260930_v6_policy_verified/
results/vnext/family_h_cross_domain_bea_gdp_20260930_v4_policy_verified/
capture_manifest.json
capture_index.json
normalized_records.jsonl
tasks.jsonl
audit_manifest.json
index.json
```

如发现问题：

1. 在独立 review 文件中写 `NEEDS_FIX` 或 `UNSUPPORTED`；
2. 写明 BEA URL、文件 hash、worksheet、row/column、release stage 或 policy 问题；
3. 不手工修改 candidate 或 hash；
4. 修复 capture/compiler 后创建新的 no-replace sibling；
5. 重新计算全部 hashes 并重新审核。

---

## 10. 审核通过的有限含义

即使 5/5 全部 `RELEASE_READY`，最多说明：

- 三个官方 BEA release vintage 已被正确绑定；
- 2024 Q4 Real GDP 指标的 2.3→2.3→2.4 revision trajectory 规范化正确；
- 同槽对象、事件、查询和生成 surface 可追溯。

不能说明：

```text
BEA 数据已获得无限制再分发许可
宏观经济预测能力已测试
经济数据 forecast accuracy 已测试
三个 vintage 是统计独立样本
跨领域 benchmark 已完成
模型、Qdrant 或 answer accuracy 已完成
Family H broad main-track 已闭合
```

正式 release、状态/检索运行、回答运行和统计汇总都是独立后续关口。

---

## 11. 推荐审核顺序

1. 核对 capture/candidate 路径和三个主要 SHA-256；
2. 逐一打开 Advance/Second/Third 的官方 XLSX；
3. 核对 `Table 1`、`Real GDP`、`Q4 2024`、Q3→Q4、annualized rate 和数值；
4. 核对 release date、estimate stage 和 official schedule；
5. 核对官方 Linking Policy、署名、non-endorsement、logo 和第三方材料边界，并确认 candidate 的 `POLICY_VERIFIED_PUBLIC_DOMAIN_WITH_ATTRIBUTION_BOUNDARY` evidence binding；
6. 检查 2.3→2.3 不被折叠，2.4 不提前泄漏；
7. 检查 ADD/UPDATE/UPDATE、对象身份、scalar answer schema 和 query；
8. 检查完整 task canonical hash 与 audit binding；
9. 填写 5 项决定及具体 rationale；
10. 保存独立 review root，完成机器一致性校验后，才考虑正式 promotion。

---

## 12. 当前机器验证

```bash
python -B -m pytest tests/vnext/test_family_h_cross_domain_bea.py -q -ra
python -m py_compile scripts/vnext_capture_family_h_bea_candidate.py scripts/vnext_prepare_family_h_bea_candidate.py
```

当前本地候选验证结果：

```text
15 passed
```

这只是 parser、capture、candidate、reference 和 boundary 的自动化检查，不替代人工审核。

## 最终文件位置

```text
docs/vnext/family_h_cross_domain_bea_human_audit.md
```
