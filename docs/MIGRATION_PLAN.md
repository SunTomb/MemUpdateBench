# 项目迁移计划：G-MSRA → MemUpdateBench

> 创建时间：2026-05-01  
> 目的：将 update-frequency stress test 方向从 G-MSRA 中独立出来，建立干净的新项目

---

## 一、为什么要迁移

| 问题 | 说明 |
|------|------|
| 名称不符 | G-MSRA = "Grounded Memory Self-Rewarding Agent"，但现在做的是 benchmark/diagnostic |
| 历史包袱 | 54 个 WORKFLOW 文件、Phase 1-5 废弃的训练脚本和数据 |
| 代码冗余 | 21 个脚本中只有 6 个是 P6.x 需要的 |
| 开源/投稿不友好 | 审稿人/用户看到的 repo 应该是干净、聚焦的 |

---

## 二、新项目结构

```
d:\USTC\2026Winter\MemUpdateBench\
├── README.md                     # 新项目说明
├── setup.py                      # 包名改为 mem_update_bench
├── requirements.txt              # 精简后的依赖
├── activate.sh                   # 集群环境激活（路径更新）
│
├── mub/                          # 核心包（原 gmsra，重命名）
│   ├── __init__.py
│   ├── config.py                 # GMSRAConfig → MUBConfig
│   ├── utils.py
│   ├── memory/
│   │   ├── entry.py
│   │   └── store.py
│   └── manager/
│       └── memory_manager.py
│
├── scripts/
│   ├── prepare_data.py           # benchmark 数据生成器
│   ├── eval_evomemory.py         # 主评测脚本
│   ├── analyze_ood_errors.py     # 错误分析
│   ├── summarize_update_frequency.py  # 自动化汇总 + 画图
│   ├── smoke_test.py             # 回归测试
│   ├── train_constrained_sft.py  # 训练脚本（如需 repair）
│   └── generate_constrained_sft.py   # SFT 数据生成
│
├── baselines/                    # 外部 baseline 集成
│   ├── README.md
│   ├── base_agent.py
│   └── memory_r1_agent.py        # project-local approximation
│
├── data/                         # 只保留 P6.3 hard splits
├── results/                      # 只保留 P6.3/P6.4 结果
├── paper/                        # 论文 draft（全新开始）
│
├── WORKFLOW.md                   # 精简版工作流（从 P6.0.1 思路开始）
└── docs/
    └── HISTORY.md                # 简要记录 G-MSRA 的历史演变
```

---

## 三、文件迁移清单

### 3.1 必须复制 ✅

#### 核心包 `gmsra/` → `mub/`

| 源文件 | 目标 | 说明 |
|-------|------|------|
| `gmsra/__init__.py` | `mub/__init__.py` | 修改包名 |
| `gmsra/config.py` | `mub/config.py` | 可选：重命名 class |
| `gmsra/utils.py` | `mub/utils.py` | `compute_f1`, `compute_exact_match`, `load_model_and_tokenizer`, `generate_text` 等 |
| `gmsra/memory/__init__.py` | `mub/memory/__init__.py` | |
| `gmsra/memory/entry.py` | `mub/memory/entry.py` | MemoryEntry |
| `gmsra/memory/store.py` | `mub/memory/store.py` | MemoryStore |
| `gmsra/manager/__init__.py` | `mub/manager/__init__.py` | |
| `gmsra/manager/memory_manager.py` | `mub/manager/memory_manager.py` | MemoryManager |

#### 脚本 `scripts/`

| 源文件 | 必要性 | 说明 |
|-------|:------:|------|
| `scripts/prepare_data.py` | ✅ **必须** | update_frequency + update_frequency_hard 生成器 |
| `scripts/eval_evomemory.py` | ✅ **必须** | 核心评测（slot_direct, slot_prompt, rag, rich diagnostics）|
| `scripts/analyze_ood_errors.py` | ✅ **必须** | by_k_updates / by_stress_type 分组分析 |
| `scripts/summarize_update_frequency.py` | ✅ **必须** | P6.4 自动化汇总 + 画图 |
| `scripts/smoke_test.py` | ✅ **必须** | 31 项回归测试 |
| `scripts/train_constrained_sft.py` | ✅ 需要 | 如果做 targeted repair 需要 |
| `scripts/generate_constrained_sft.py` | ✅ 需要 | SFT 训练数据生成 |

#### 数据 `data/`

| 文件模式 | 必要性 | 数量 | 说明 |
|---------|:------:|:----:|------|
| `evomemory_update_frequency_hard_k*_p63_{dev,test}.json` | ✅ **必须** | 10 个 | P6.3 hard splits — 论文的核心数据 |
| `evomemory_update_frequency_hard_p63_{train,dev,test}.json` | ✅ **必须** | 3 个 | 合并后的 train/dev/test |

#### 结果 `results/`

| 目录 | 必要性 | 说明 |
|------|:------:|------|
| `results/update_frequency_p63/` | ✅ **必须** | 45 个实验结果 — 论文主表数据 |
| `results/update_frequency_p63_summary/` | ✅ **必须** | 汇总 CSV/JSON + 4 张图表 |

#### 其他

| 文件 | 必要性 | 说明 |
|------|:------:|------|
| `requirements.txt` | ✅ 复制后精简 | 去掉 trl、wandb 等 P6.x 不需要的依赖 |
| `setup.py` | ✅ 新建 | 包名改为 `mem_update_bench` |
| `activate.sh` | ✅ 新建 | 路径改为新项目 |
| `.gitignore` | ✅ 复制 | |

#### Baselines（按需）

| 文件 | 必要性 | 说明 |
|------|:------:|------|
| `baselines/base_agent.py` | 🟡 可选 | 如果后续跑外部 baseline |
| `baselines/memory_r1_agent.py` | 🟡 可选 | project-local approximation |
| `gmsra/baselines.py` | 🟡 可选 | 内部 baseline 注册表 |

---

### 3.2 可选复制（参考用）🟡

| 文件 | 说明 |
|------|------|
| `results/update_frequency_pilot2_rich/` | P6.2 pilot 数据，可作为补充参考 |
| `data/evomemory_update_frequency_k*_pilot2_*.json` | pilot 数据，可保留用于 ablation |
| `paper/references.bib` | 已有的参考文献（需大幅更新） |
| `PROJECT_WORKFLOW6.0.0.md` ~ `6.4.0.md` | 历史工作流记录，迁移到 `docs/` |
| `LITERATURE_REVIEW_*.md` | 调研报告，迁移到 `docs/` |

---

### 3.3 不需要复制 ❌

| 文件/目录 | 原因 |
|----------|------|
| `PROJECT_WORKFLOW.md` ~ `5.3.1.md` (47 个文件) | Phase 1-5 历史，与新方向无关 |
| `scripts/train_phase0_sft.py` | Phase 0 SFT 训练 — 废弃 |
| `scripts/train_phase1_rl.py` | Phase 1 RL 训练 — 废弃 |
| `scripts/train_phase2_transition.py` | Phase 2 过渡训练 — 废弃 |
| `scripts/train_phase3_full.py` | Phase 3 全量训练 — 废弃 |
| `scripts/train_phase_v2.py` | Phase v2 训练 — 废弃 |
| `scripts/eval_locomo.py` | LoCoMo 评测 — P6.x 未使用 |
| `scripts/eval_agent_tasks.py` | Agent task 评测 — 废弃 |
| `scripts/eval_lora_merge.py` | LoRA 合并 — 废弃 |
| `scripts/run_ablations.py` | 旧消融实验 — 废弃 |
| `scripts/run_baselines.py` | 旧 baseline 运行器 — 废弃 |
| `scripts/run_diag_eval.py` | 旧诊断评测 — 废弃 |
| `scripts/check_data_leakage.py` | 数据泄漏检查 — 一次性使用 |
| `scripts/eval_ablations_benchmarks.sh` | 旧 shell 脚本 — 废弃 |
| `gmsra/agent.py` | GMSRAAgent — P6.x 不使用 |
| `gmsra/reward/` | 奖励函数 — RL 阶段专用 |
| `gmsra/consolidation/` | 记忆压缩 — P6.x 不使用 |
| `baselines/evolver_agent.py` | Evolver baseline — 废弃 |
| `baselines/reflexion_agent.py` | Reflexion baseline — 废弃 |
| `baselines/self_consolidation_agent.py` | Self-consolidation — 废弃 |
| `baselines/mem0_memoryr1_agent.py` | Mem0 hybrid — 废弃 |
| `baselines/eval_baselines.py` | 旧 baseline 评测 — 废弃 |
| `baselines/train_and_eval_rl_baselines.py` | RL baseline 训练 — 废弃 |
| `baselines/run_rl_baselines.sh` | RL baseline shell — 废弃 |
| `data/evomemory_{basic/advanced/hard/ood/schema}*` | 旧版 EvoMemory splits |
| `data/constrained_sft_*` | 旧 SFT 训练数据 |
| `data/locomo_*` | LoCoMo 数据 |
| `data/alfworld_tasks.json` | AlfWorld 数据 |
| `data/longmemeval_*` | LongMemEval 数据（277MB） |
| `outputs/` | 所有旧训练 checkpoint |
| `outputs_v1/` | v1 输出 — 废弃 |
| `results_v1/` | v1 结果 — 废弃 |
| `logs/` | 训练日志 — 废弃 |
| `logs_v1/` | v1 日志 — 废弃 |
| `parse_old.py`, `parse_results.py`, `parse_results_diag.py` | 旧解析脚本 |
| `test_mock.py`, `test_peft_crash.py` | 一次性测试 |
| `paper/main.tex` | 旧论文 draft — 需要全新撰写 |

---

## 四、Checkpoint 处理

`outputs/constrained_sft_curriculum_long25/best` 是 P6.x 的核心 learned baseline。

> ⚠️ **这个 checkpoint 只存在于集群 `/NAS/yesh/G-MSRA/outputs/` 上，本地没有副本。**  
> 如果要删除集群上的 G-MSRA，必须**先把它移到新目录**。

**操作方案**：用 `mv`（移动而非复制）将 checkpoint 移到新项目目录，避免占用额外空间：

```bash
# 在集群上：移动 checkpoint 到新项目
ssh Tang-2-Wu 'mkdir -p /NAS/yesh/MemUpdateBench/checkpoints && \
  mv /NAS/yesh/G-MSRA/outputs/constrained_sft_curriculum_long25 \
     /NAS/yesh/MemUpdateBench/checkpoints/long25'
```

评测脚本通过 `--lora_checkpoint checkpoints/long25/best` 参数指定，路径更新即可。

---

## 五、集群上不可删除的文件清单 🔴

在删除 `/NAS/yesh/G-MSRA` 之前，以下内容**必须先迁移或确认已保留**：

| 内容 | 位置 | 大小估算 | 处理方式 |
|------|------|---------|---------|
| **long25 checkpoint** | `outputs/constrained_sft_curriculum_long25/` | ~500MB-2GB（LoRA 权重） | ✅ `mv` 到新项目 |
| **P6.3 远程结果**（如有未下载的） | `results/update_frequency_p63/` | ~数 MB | ✅ 确认本地已有完整副本 |

以下内容**不需要保留**，可以安全删除：

| 内容 | 原因 |
|------|------|
| `outputs/v2_base/`, `v2_compact/`, `v2.1_fixed/`, `v2_high_penalty/` | Phase 1-3 旧 checkpoint，本地有副本 |
| `data/` 下所有文件 | 可由 `prepare_data.py` 重新生成，本地也有 |
| `scripts/` | 本地有完整副本 |
| `gmsra/` 包 | 本地有完整副本 |
| `baselines/` | 本地有完整副本 |
| `results/` 下旧结果 | P6.3 结果已下载到本地 |
| `logs/`, `logs_v1/` | 旧训练日志 |
| `.git/` | 本地 Git 是完整的 |

以下是**不属于 G-MSRA 但在同一 NAS 上**，绝对不能误删：

| 路径 | 说明 |
|------|------|
| `/NAS/yesh/miniconda3/` | conda 安装，所有项目共用 |
| `/NAS/yesh/hf_cache/` | HuggingFace 模型缓存，所有项目共用 |
| `/NAS/yesh/` 下其他项目目录 | 如有其他项目 |

---

## 六、集群同步与删除流程

**严格按顺序执行**，每一步确认后再进行下一步：

```bash
# ─── Step 1: 创建新项目目录 ───
ssh Tang-2-Wu 'mkdir -p /NAS/yesh/MemUpdateBench/checkpoints'

# ─── Step 2: 移动 checkpoint（最关键的一步）───
ssh Tang-2-Wu 'mv /NAS/yesh/G-MSRA/outputs/constrained_sft_curriculum_long25 \
                  /NAS/yesh/MemUpdateBench/checkpoints/long25'

# ─── Step 3: 验证 checkpoint 移动成功 ───
ssh Tang-2-Wu 'ls -la /NAS/yesh/MemUpdateBench/checkpoints/long25/best/'
# 应该看到 adapter_config.json, adapter_model.safetensors 等文件

# ─── Step 4: 确认本地 P6.3 结果完整 ───
# 本地运行：
# ls results/update_frequency_p63/ | wc -l
# 应该是 45 个目录

# ─── Step 5: 同步新项目代码和数据到集群 ───
rsync -avz --exclude='__pycache__' --exclude='.git' \
  d:/USTC/2026Winter/MemUpdateBench/ \
  Tang-2-Wu:/NAS/yesh/MemUpdateBench/

# ─── Step 6: 创建新的 activate.sh ───
ssh Tang-2-Wu 'cat > /NAS/yesh/MemUpdateBench/activate.sh << "EOF"
cd /NAS/yesh/MemUpdateBench
eval "$(/NAS/yesh/miniconda3/bin/conda shell.bash hook)"
conda activate gmsra
export HF_HUB_CACHE=/NAS/yesh/hf_cache/hub
export HF_HUB_OFFLINE=1
export PYTHONPATH=/NAS/yesh/MemUpdateBench
echo "MemUpdateBench environment ready ✅"
EOF'

# ─── Step 7: 在新项目目录下验证 ───
ssh Tang-2-Wu 'source /NAS/yesh/MemUpdateBench/activate.sh && \
  python -m py_compile scripts/eval_evomemory.py && \
  echo "Compile OK"'

# ─── Step 8: 确认一切正常后，删除旧目录 ───
# ⚠️ 最后才执行！确认 Step 2-7 全部通过！
ssh Tang-2-Wu 'rm -rf /NAS/yesh/G-MSRA'
```

> **注意**：继续使用 `gmsra` conda 环境，不需要新建 conda env。只改 `PYTHONPATH`。

### 预计释放空间

| 目录 | 估算大小 | 说明 |
|------|---------|------|
| `outputs/` 下其他 checkpoint | 2-8 GB | Phase 1-3 旧 checkpoint |
| `data/longmemeval_*` | ~277 MB | LongMemEval 原始数据 |
| `data/` 其他旧数据 | ~50 MB | 旧版 EvoMemory splits |
| `logs/` + `logs_v1/` | ~数 MB | 训练日志 |
| `.git/` | ~数十 MB | Git 历史 |
| **合计** | **约 3-9 GB** | 视旧 checkpoint 大小而定 |

---

## 七、代码修改清单

迁移后需要做的代码修改（**批量 find-replace**）：

| 修改项 | 范围 | 说明 |
|--------|------|------|
| `from gmsra.` → `from mub.` | 所有脚本 | 包名更新 |
| `import gmsra` → `import mub` | 所有脚本 | 包名更新 |
| `name="gmsra"` → `name="mem_update_bench"` | `setup.py` | 包注册名 |
| `G-MSRA` → `MemUpdateBench` | `activate.sh`, `README.md` | 项目名 |
| PYTHONPATH 路径 | `activate.sh` | 集群路径 |
| `outputs/constrained_sft_curriculum_long25/best` | 评测脚本/命令 | 改为 `checkpoints/long25/best` |

---

## 八、迁移后 G-MSRA 的处理

### 本地 `d:\USTC\2026Winter\G-MSRA\`

| 操作 | 说明 |
|------|------|
| **保留不删** | 作为历史 archive，本地磁盘空间相对充裕 |
| **不再开发** | 所有新开发在 MemUpdateBench 中进行 |
| **Git archive** | 可选：打一个 `v1.0-archive` tag，然后不再 commit |

### 集群 `/NAS/yesh/G-MSRA/`

| 操作 | 说明 |
|------|------|
| **迁移后删除** | 按 Step 1-8 流程执行，释放 3-9 GB 空间 |
| **删除前必须确认** | ① long25 checkpoint 已移到新目录 ② P6.3 结果本地有完整副本 |
| **不可恢复** | 删除后无法恢复，务必逐步验证 |

---

## 八、注意事项

### 8.1 迁移前

- [ ] 确认 `smoke_test.py` 在当前 G-MSRA 下仍通过 31/31
- [ ] 确认 P6.3 results 完整（45 个目录 + 7 个 summary 文件）
- [ ] 如果有未 commit 的改动，先在 G-MSRA 里 commit

### 8.2 迁移时

- [ ] 复制文件后，**先做包名批量替换**（`gmsra` → `mub`）
- [ ] 运行 `python -m py_compile` 确认所有脚本无语法错误
- [ ] 运行 `smoke_test.py` 确认在新项目下仍通过
- [ ] 确认 `summarize_update_frequency.py` 能正确读取迁移后的 results

### 8.3 迁移后

- [ ] 新项目 `git init` + 首次 commit
- [ ] 更新 `.claude/` 或 `CLAUDE.md` 指向新路径
- [ ] 集群 rsync + 验证 `activate.sh`
- [ ] P6.5 开始在新项目中开发

### 8.4 关于项目名

建议名称（可选）：

| 名称 | 优势 | 劣势 |
|------|------|------|
| `MemUpdateBench` | 直接明确 | 偏 benchmark 命名 |
| `UpdateStress` | 简短 | 不够正式 |
| `SlotUpdateDiag` | 体现 slot + diagnostic | 稍长 |
| `EvoMemBench` | 延续 EvoMemory 品牌 | 可能与 MemEvoBench 混淆 |

**推荐 `MemUpdateBench`**，简洁且不与现有论文名冲突。

### 8.5 requirements.txt 精简

新项目可以去掉以下不再需要的依赖：

```diff
 # Core
 torch>=2.4.0
 transformers>=4.46.0,<4.49.0
 accelerate>=0.34.0,<1.0
 datasets>=2.18.0
 
-# RL Training
-trl==0.15.2
 
 # LoRA / PEFT
 peft>=0.10.0
 bitsandbytes>=0.43.0
 
 # Retrieval
 sentence-transformers>=2.6.0
 
 # Evaluation
 rouge-score>=0.1.2
 nltk>=3.8.1
 
 # Utilities
 numpy>=1.26.0
 pandas>=2.2.0
 scipy>=1.12.0
 scikit-learn>=1.4.0
 pyyaml>=6.0.1
 tqdm>=4.66.0
-wandb>=0.16.0
 loguru>=0.7.2
 jsonlines>=4.0.0
+matplotlib>=3.8.0        # summarize_update_frequency.py 需要
```

> `trl` 和 `wandb` 是 RL 训练专用，新项目不需要。  
> `matplotlib` 是画图需要的，应该显式列出。
