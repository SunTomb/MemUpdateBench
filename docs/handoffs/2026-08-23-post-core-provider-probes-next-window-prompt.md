# MemUpdateBench 新窗口继续提示词（Post-Core Phase 1–2）

请在 `D:\USTC\2026Winter\MemUpdateBench` 目录继续 MemUpdateBench，并使用中文回复。

开始时先只读检查，不要立即修改或运行模型：

1. 完整阅读并严格遵守根目录 `CLAUDE.md`。
2. 阅读 `WORKFLOW.md` 最后一节“Post-Core closed-provider local and Tang-2 capability preflight (2026-08-23)”。
3. 阅读持久记忆：
   - `memupdatebench_post_core_closed_provider_preflight.md`
   - `memupdatebench_post_core_gpu6_qwen_preflight.md`
   - `memupdatebench_post_core_public_snapshots.md`
   - `memupdatebench_post_core_identity_preflight.md`
   - `memupdatebench_benchmark_engineering_direction.md`
4. 运行只读 Git 检查：`git status --short --branch`、`git rev-parse HEAD`、`git fetch origin`、`git rev-parse origin/master`。本地主目录在上一窗口结束时有大量用户自己的未提交改动，且可能仍停留在旧的本地 `master`；即使 GitHub `master` 已更新，也不得 pull、reset、checkout、stash、clean、覆盖或自动解决冲突。先保护并汇报这些本地改动，再决定如何建立新的干净 worktree/分支。

当前已完成的新增能力证据：

- 开源快照已认证并共享：Qwen3.5-9B、Muse Glimmer 30B GGUF、Muse Glimmer 30B BF16。
- Qwen3.5-9B 已在 Tang-3 GPU6 完成严格 offline load/unload，零生成。
- API-Transfer-Station 的五个精确请求名已在本地和 Tang-2 做过最小能力检查：
  - `claude-sonnet-4-6`
  - `claude-opus-4-8`
  - `Gemini 3.6 Flash (Low)`
  - `grok-4.5`
  - `gpt-5.5`
- 共 12 次真实 provider 调用，零重试、零 benchmark generation；所有路由均返回 HTTP 200、匹配的 provider `model` 和精确 `OK`。
- GPT-5.5 最初忽略 `stream=false` 并返回 SSE；中转站修复后，Tang-2 复测已返回标准 `application/json` Anthropic Message。
- Gemini 的规范身份是 `gemini-3.6-flash`，实际中转站请求名是 `Gemini 3.6 Flash (Low)`，reasoning tier 为 Low；三个字段都必须保留。

证据边界必须严格保持：

- 这些只是 provider connectivity/interface capability evidence，不是 MemUpdateBench accuracy、prompted-answer 或科学证据。
- Claude Sonnet 4.6、Opus 4.8 和 Gemini 3.6 Flash 有官方文档身份依据；中转站请求/响应名一致不能把 Grok 4.5 或 GPT-5.5 自动升级为不可变官方上游身份。
- 不得修改、重新生成、覆盖或重绑定 `data/vnext/core/v3`、Task 9–14 根、Task 14 final root/index、Post-Core Phase 0 root/index 或 `configs/vnext/post_core/official_identity_evidence_v1.json`。
- 不得把 unsupported/null 静默记为 0，不得把能力检查当作 benchmark passing。
- 不得读取、打印、写入、提交或持久化任何 API key/token/Authorization 值。

下一项推荐工作是独立的 **Post-Core Phase 1–2 qualification release**，不是直接跑完整 benchmark：

1. 设计并发布 no-replace、脱敏、source-bound 的闭源 provider qualification receipts，绑定五个请求名、响应模型、调用计数、Gemini Low 映射和 GPT 格式修复；
2. 为 Muse GGUF 冻结可复现的 llama.cpp commit/build/runtime/device/context 边界，speculative decoding 先关闭；
3. 将 Qwen 从 load-only 推进到小规模短生成、chat template、parser、determinism 和 unload gate；
4. 对 Muse GGUF 做同等 load/short-generation gate；
5. 在开源/闭源面板上运行统一的 8–16 请求 capability smoke，输出明确的 `READY`、`BLOCKED` 或 `UNSUPPORTED`；
6. 只有上述 gate 通过并再次获得执行授权后，才启动计划中的 320-generation canary；之后才是三个 k=16 confirmatory hard 条件，最后才考虑完整矩阵。

请先汇报 Git/工作树真实状态、当前证据边界和 Phase 1–2 的实现设计，再进行任何实质写入。不要因本地 `master` 与 `origin/master` 不一致而破坏用户现有未提交改动。
