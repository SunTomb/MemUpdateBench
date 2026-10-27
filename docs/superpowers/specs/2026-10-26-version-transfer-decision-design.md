# 重复更新与版本传递：继续／转向判定计划

## Context

用户已决定暂缓广义 benchmark 扩张，先做有明确继续／转向门槛的原生系统实验，再决定是否投入聚焦诊断／实证论文。Graphiti/Zep 已有时间图、矛盾失效和历史保留；LongMemEval、STALE、MINTEval、LAPSE 等已覆盖知识更新、干扰与部分跨层分析，不能仅以“我们是 benchmark”声称新颖性。

本计划的贡献候选是：在信息充分、原生系统配置公平的前提下，隔离重复更新与时间顺序对“事件理解 → 有效状态 → 检索证据 → 回答”的作用，并识别版本语义在哪一层丢失。Go/no-go 是研究投资决策，不是论文录用预测或预先承诺发现。

### 已核实边界

- 当前本地 HEAD `a80e75f`，仅 `docs/WORKSPACE_LAYOUT.md` 有用户未提交改动，必须排除。
- 原集群 5,222-test v2 回归已有完整 summary：5,024 passed、155 failed、43 skipped；8 shards 均声明完整执行覆盖，但尚未独立审计全部事件。保持原样，不把修完所有历史工程测试作为新研究的前置条件。
- Graphiti 当前核查 revision 为 `5c27edfa8d3e0f49b2571bcebb4261bcc89ee037`。`add_episode` 支持公开事件正文与 `reference_time`，内部设置摄入时间；原生抽取、消歧、失效和排序不能由 gold CRUD 或本地最新值选择器替代。
- Tang-2 的只读检查未发现可直接使用的 Graphiti/Neo4j/FalkorDB 启动环境；这不是模型能力 blocker，后续必须做受限环境资格验证。不要借用别人的本地服务。
- 重计算只在实验室集群；集群工作目录、环境、临时文件、日志、socket 均在 `/NAS/yesh`。不读取/修改其他用户工作，不使用未获准付费 API，不沿用已耗尽的单次 Qwen 试验授权。
- Core/Pilot/Phase 0/Family-H 等冻结数据、结果、hash 和证据类别不变；新任务与结果采用 sibling 根。

## 推荐路径

只推进“原生 Graphiti 为必测对象、确定性时间账本为 sanity reference、一个资格合格的非图原生系统为受限对照”的小试验。先离线准备与单元测试，再少量能力 smoke，最后一个有硬上限的判定样本；不增加模型面板、语言面板、排行榜或训练任务。

不使用纯文献差异直接宣布 GO；也不一开始扩大数据/系统笛卡尔积。

## 实验原则

1. 当前与历史查询分开计分；正确保留已失效历史不是错误。
2. 目标对象、时间、事实均来自系统可见事件或问题；隐藏 gold key、gold 状态、最新标签不得进入原生写入或检索决策。
3. 主条件保留原生有效时间/失效字段，使用合法的 query-time 时间约束；默认不加过滤的接口只能作为另行标记的条件。
4. “同一检索证据删去时间字段”与 oracle 正确证据只做诊断对照，不当作部署方案或系统本体失败。
5. 不把缺失时间信息导致的不可判定问题算作模型错误；也不把适配器自行丢失字段归因给 Graphiti。
6. 维护状态、历史保真、检索、回答、资源开销分开报告；UNSUPPORTED/BLOCKED/技术错误不得改成准确率零或静默删行。
7. 生成器使所有版本使用平衡措辞；不沿用“只有最新版本带 now/relocated”等表面线索来支撑新结论。

## 判定试验设计

### 样本与变量

- 建立一个小型 sibling diagnostic slice：16 个独立语义轨迹，8 个开发、8 个冻结验证；在单值的用户状态和项目配置两种受控场景中均衡分配。改名/改写不计为新的独立轨迹；它们是受控合成样本，不冒充真实来源数据或大规模外部效度证据。
- 每条轨迹构造 `u ∈ {1,4,16}` 次目标替换，`u` 不含初始写入。每个版本总共 17 个公开事件：初始目标值、`u` 个更新、`16-u` 个等长度非目标事件；目标最终更新时间固定。匹配并报告实际 tokenizer 长度，保持可见时间精度、最终值位置和措辞规则一致。主要效应因此是固定事件预算内目标更新密度，不宣称已经排除了所有干扰机制。
- 固定语义轨迹作成对比较；新增命名空间不等于新增独立样本。验证集不得因结果不好而换样本、改提示或调参数。
- 每条轨迹有 current 和 as-of 历史两类 query，正确答案均由公开时间规则和确定性 replay 推出。历史问题不得仅采用“事件第几条”隐式规则，必须对系统可见。
- 主试验按公开有效时间正序摄入，公开 `timestamp` 与 replay 的 `effective_at` 相同。4 个资格案例只覆盖正常替换、重复同值、历史时点查询、相似对象不应误失效。已经核实当前 v3 要求同对象 logical-time anchors 严格增加，不能忠实表示追溯更正；因此晚到/双时间线明确不属于本轮，不修改冻结契约或用 metadata 伪装支持。

### 系统与读出

- Graphiti：固定上游 commit，真实图数据库，原生 `add_episode` 抽取/消歧/失效；主路径使用问题公开时间生成的 native 时间过滤，保留原始证据顺序、分数、实体/关系 ID 和四类时间字段。原生默认无时间过滤只作单列对照，不以较差默认配置代替合理配置。
- 固定一个已存在且可验证的本地底层模型作为两系统的抽取/推理基础，并固定一个共同 readout 模型、提示、temperature、上下文/输出预算和序列化契约；执行清单明确各角色，即使是同一模型也分别记账。嵌入/重排模型尽量匹配，不可匹配的架构差异明确列出。现有 worker 若硬绑定其他模型，不直接冒充 matched configuration；资格阶段解决或报 BLOCKED，不偷偷换模型。
- 确定性时间账本：离线 state sanity 与 oracle-evidence reference，不作为真实外部系统胜利，也不预设回答性能上界。
- 一个非图系统：复用 `OfficialMem0BackendV1` 的真实 `Memory.add` 推断路径，新增薄的 episode 边界并如实声明支持项，不复用其当前限制 UPDATE/history 的 adapter 并伪造能力。Letta block-profile 和 LangGraph store 是自定义管理，不作为本轮自主 native 对照；direct-Qdrant CRUD 与 gold 驱动 adapter 同样不算。若 Mem0 历史能力不满足，保留 typed UNSUPPORTED，不能公平比较的历史指标只报告 Graphiti 及 oracle；Mem0 整体未资格合格时不升级跨系统主张。
- 对每次实际 native retrieval，保存两种**时间信息都完整**的读出：原生排序（主结果）和逆转原生排序（成对诊断）。后者不是按 gold 最新值重排；证据集合、字段、token 内容和问题均保持不变，检验充分信息条件下的呈现顺序敏感性。主结果仍使用原生顺序。
- 另预先固定开发集 4 个 query 做时间字段删减诊断，共最多 8 次回答；不扩到验证集，不把删去必要信息后的不可识别问题当模型错误，也不据此单独判定 GO。
- 确定性正确证据的读出为独立 oracle 诊断，对同一个 task/query/模型可复用；不进入系统 write/retrieval 或排名指标。显式 latest/outdated gold 标签不作为主要干预。

### 分层指标

- 写入/状态：新事实是否被保留；应失效关系是否仍被标为有效；当前有效区间冲突；身份无法对齐时 null，而非强制映射。
- 历史：as-of 可恢复性；保留正确历史仅计成本，不判为错误。
- 检索：query-time 正确证据可见性、无有效性说明的旧证据暴露；原生时间字段在 adapter/serialization 中是否丢失。
- 回答：typed current/history accuracy；只在正确且充分证据已可见的条件下归因回答层错误。
- 对照 full-time 原生条件，以每个语义轨迹的 `u=16 − u=1` 当前回答差异作为主深度对比；历史准确率单列。呈现顺序效应采用 difference-in-differences：高深度与低深度的“原生顺序 − 逆序”差异之差，不把所有普通位置偏差称作重复更新机制。
- 按 semantic core 配对；展示 8 个验证轨迹的全部结果与分母，置信区间仅作小样本描述，不作总体保证。预先固定缺失/UNSUPPORTED/对齐 null 的处理，至少有 6 个完整验证配对才作筛选判断；否则 INCONCLUSIVE。缺失随深度变化本身单列，不能只报完成样本。
- 条件化“正确证据已可见”的回答错误用于机制诊断，不把不同深度选出的不同案例子集差异称为独立因果效应。Oracle 对比的长度/格式/干扰变化必须报告；正确状态不保证 oracle prompt 回答准确，故它是参考条件，不是经验性能上界。
- 运行：原生内部调用、失败、解析错误、重试、token、延迟、数据库空间及清理；技术失败与 unsupported 全部单列。

### 执行上限

- 先 4-case capability smoke；不合格时不启动主矩阵。
- 主矩阵最多 16 cores × 3 depths × 2 native systems = 96 次独立系统轨迹、1,632 个 episode 写入请求；两类 query 共 192 次主检索。
- 两种 native evidence 读出最多 384 次回答生成；oracle evidence 读出最多 96 次，加开发集 8 次信息损失诊断，主矩阵与该诊断共 488 次回答生成。历史不支持的行保留但不生成，不能补成零。
- 原生 SDK 的内部 LLM/embedding/reranker 调用不能伪装成一次 episode 调用。smoke 最多 8 次系统轨迹（4 cases × 2 systems，每条最多 5 个公开事件），它检验可执行、可观测、忠实集成和清理，**不要求模型答对所有挑战**，否则会筛掉研究对象。
- 预留最多 4 个验证 core 的复核（按预定 hash 顺序选至多 3 个失败 core 和 1 个成功 core）：u=1/16、两系统、最多 16 次 fresh-ingestion 轨迹/272 个事件、64 次 native readout。不能只复读保存的证据就声称系统可复现。
- 本工作包全部 LLM 尝试硬上限 10,000（smoke、主矩阵、复核、重试全算），输入/输出 token 上限分别 24M/2M；单响应 max-output 2,048；model inference concurrency=1，GPU 总占用上限 12 小时、总墙钟 24 小时，新增图数据库/运行日志上限 10 GiB（不含已存在权重）。embedding/reranker 次数与时间独立计量但纳入 GPU/墙钟预算。smoke 后估算主矩阵是否能在预算内完成，不能通过则 BLOCKED，不静默删任务。
- 每次完整 paired block 包含同 core 的所有深度与系统，跨 core 平衡轮换执行顺序；不得先跑完所有低深度再耗尽高深度预算。保持调用/请求账本，达到任何上限即写 `BUDGET_BLOCKED`，不自动扩充或隐藏重试。
- 零付费 API。仅在执行清单确认具体本地模型 revision/路径、原生数据库与 GPU 资格后运行；不自动沿用旧单次 load/generation 授权。若模型输入无法在已授权 `/NAS/yesh` 边界内合法访问，报告资源 blocker，不自行下载或搬运大权重绕过。

## 预先写定的决策规则

- **GO_TO_CONFIRMATION（继续，但还不是论文结论）**：资格与公平性 gate 通过；完整时间信息 native 条件的冻结验证配对中，至少 3 个独立 core 在低深度正确而高深度错误，并且减去相反变化后净退化至少 2 个 core；或按同一标准出现随深度增强的充分信息呈现顺序效应。须覆盖两种受控场景、由预先冻结的分类定位环节，并在选定 fresh-ingestion 复核上重现。它是小样本投资筛选阈值，不声称统计显著或跨系统普遍性。普通低深度错误、单纯剥离必要信息，以及未预先定义的“当前/历史 tradeoff”不能单独触发 GO；后者只作后续假设。
- **PIVOT（调整主张）**：观察到的下降仅来自去掉时间、oracle 标签、适配器丢字段、错误配置或资源失败；修正后不再有新的 native 机制证据。转为范围明确的接口/时间语义传递研究，或停止重复更新主张；不继续堆任务数量来追求失败。
- **INCONCLUSIVE（证据不足）**：合格试验中没有清晰效应或只出现少量不稳定错误。小样本零失败不是“Graphiti 已解决问题”；最多提出一次另行冻结的确认，不能自动追加。本工作包停止投入；一次确认仍无有用信号则停止该科研主张，不循环加难度/换模型直到失败。
- **BLOCKED（技术上未测成）**：数据库、模型解析、预算、权限或可观测性不合格。只记录阻塞，不能给出科学 no-go 或系统失败。
- Graphiti 表现良好是有效对照，不要求它必须失败；如果所有差异只是已知设计原则的重复，不能以此支撑宽泛主会新颖性。

GO 后才扩展确认样本、增加一项真实来源检验和必要的第二答案模型；具体规模由 pilot 效应量与预算决定，另行冻结，禁止在看验证结果后无记录扩样。

## 实施步骤与复用位置

### 1. 冻结研究问题，不再扩工程范围

- 新建 `docs/superpowers/specs/2026-10-26-version-transfer-decision-design.md` 和 `configs/vnext/version_transfer_decision_v1.json`，记录上述分组、时间语义、估计量、排除条件、预算、继续/转向规则及 Graphiti/STALE/MINTEval/LAPSE 的差异矩阵。
- 每项 claim 标为既有事实、待检验假设或不作主张；旧 P6/P8 只是提出假设的依据，不能替代新 native 实验。
- 旧全量回归和冻结结果保持原证据类别；只处理阻挡本试验的依赖或正确性问题。

### 2. 离线任务准备与 deterministic sanity

- 新增 `scripts/vnext_prepare_version_transfer_decision.py`，输出 sibling roots，直接构造/序列化既有 `MemUpdateTaskV3`、gold version ledger、queries 和 hashes，不新增平行 task 字典或修改 Core 生成计划。
- 复用 `mub/vnext/contracts/v3/task.py` 的事件与 selectors，`mub/vnext/validation/replay_v3.py` 的 `replay_task_v3`/`resolve_query_v3`，`mub/vnext/generation/family_f.py` 的历史 selector 绑定方法。定义本 slice 为单时间线 ISO-UTC，事件公开时间=gold effective time；不借用隐藏 action 时间给 provider。
- `ReferenceAdapterV3` 仅做 privileged sanity；`RawAppendAdapterV3`、`ExactCrudAdapterV3` 可做免费离线控制，后者不支持历史时明确记录。
- 通过后冻结开发/验证 trajectory IDs、任务哈希、配置哈希与独立性分组；无 provider/model execution。

### 3. 薄的原生 Graphiti/对照适配边界

- 新增 `mub/vnext/external/providers/graphiti_native_episode.py`；公开边界复用 `visible_event_input`/`visible_query_input`（`mub/vnext/external/visibility.py`）。Graphiti 收到 raw event、公开时间和隔离 namespace，不收到 normalized gold CRUD、目标 slot 列表、selector、gold answer 或 version ledger。
- 对照复用 `mub/vnext/external/workers/mem0_worker.py` 的 `OfficialMem0BackendV1.ingest_event(..., infer=True)`；不能把 `infer=False` 或手工更新等价成原生记忆管理。
- 保持 `MemoryAdapterV3`、`TaskRunRecordV3` 和 `ScoreRecordV3` 的既有结构。`execute_task_v3` 的可见动作 admission 必须审查：若原生 episode 不能诚实对应 observer 字段，则以 nullable observation/typed unsupported 接入，不能伪造能力或让 gold action 驱动原生 writes。仅在本 slice 接入的必要边界扩展，不做 runtime 全局重写。
- Graphiti 原始 edge/episode ID、时间字段与顺序保留；评分所需 key 对齐在 evaluator 侧进行，歧义输出 null，保持与 write/read visibility 隔离。
- 使用独立依赖环境，不把第三方 package 安装进源码根；隔离数据库/loopback端口与清理仅作用于自身资源。关闭并验证遥测、自动下载、外部 provider fallback；Graphiti 及模型/embedding/reranker/DB 的实际版本与配置必须记录。

### 4. 资格 gate → 受限主矩阵

- 新增 `scripts/vnext_run_version_transfer_decision.py`；复用 `engine_v3.py`、`PromptedAnswerModelV3` 的可注入边界，参考 cross-domain readout 的 prepare/load/generate/close 生命周期，但不扩写其冻结四读出脚本。
- 4-case smoke 要核查 JSON/结构响应、公开时间解析、原生属性保留、namespace 隔离、正确清理及完整调用计数；失败停在资格阶段。内部 retry 必须有明确配置和预算计数。
- 首先执行开发半集；诊断不成立时不消耗验证集。开发结果不能改变已冻结验证样本；如必须改设计，创建新研究版本并明确旧验证集是否已被查看。
- 在准备清单完成后才启动主矩阵；串行运行、实时逐请求事件、阶段超时、明确终态，按 manifest 对实际执行逐项核验。不能把 echo 请求数量当执行覆盖。
- 模型、检索、引用数据与文本记录保存在受控私有根；公开摘要不含凭据、隐私路径、raw provider 内容或隐式推理。

### 5. 成对分析与决策输出

- 新增 `scripts/vnext_analyze_version_transfer_decision.py`，复用 `scorer_v3.py::score_task_v3`、`VerifiedScoringContextV3` 和 `lifecycle_v3.py`，只增加 study-level paired contrasts，不替换规范评分。
- 按 semantic core 配对比较 u=1/4/16，分别报告 current/history 与 native evidence/order 对照；记录“低更新已错”的案例，不误归因为重复更新。原生当前/历史元数据是否正确要单列核验。
- 输出 `decision.json`、逐案例机制归因、缺失/unsupported ledger、成本记录和中文判定说明，结论只可为 GO_TO_CONFIRMATION/PIVOT/INCONCLUSIVE/BLOCKED。
- 更新 `WORKFLOW.md` 与 `paper/manuscript_sections/related_work_positioning_draft.md`，写出当前支持/不支持的最小论文主张；未有新实验前不写虚构 Results。

## 验证要求

- 新测试：`tests/vnext/test_version_transfer_decision.py`、`tests/vnext/test_graphiti_native_episode.py`、`tests/vnext/test_version_transfer_analysis.py`。重点是：可见输入不含 gold；raw/normalized 替换不会改变 provider payload；语义独立配对；时间公开且严格有序；历史保留不作错误；supported/null；预算耗尽；no-replace；清理只触及自身 namespace；事件/结果集合严格匹配；新旧实验不混用。
- 复用 focused gate：`test_v3_replay_scoring.py`、`test_lifecycle_v3.py`、`test_core_runtime_v3.py`、相关 visibility 测试；实际测试名在实施时用 collection 核验。
- fake/native-client 单元测试只证明边界和协议；只有真实 Graphiti DB/推断链的 smoke 可作为 runtime 资格。两者均不是科学准确率。
- 主试验前 deterministic replay 必须全通过；native lifecycle 数据、映射歧义和历史查询 capability 不合格即停，不降低标准补零。
- 模型/较重验证全部集群执行，源 revision 与输入 hash 固定，原始失败保留。避免全局 `GIT_DIR/GIT_WORK_TREE`；临时 HOME、pytest basetemp、缓存、tmux socket 均固定到 `/NAS/yesh`。
- 实施批次只在少数里程碑复核，不为每个文件启动子代理/审查，也不再触发 5,222-test 全量回归来证明研究创新。

## 完成定义

本工作包完成不是“Graphiti 被证明失败”，而是得到一份可追溯、方法公平、有成本上限的继续／转向判定。若资格 gate blocked，也必须交付完整准备、已证实 blocker 和没有测到的范围，不能把技术未执行解释成科研否定。正式论文只在 GO 后按缩窄主张推进；不保证主会可发表。
