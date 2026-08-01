# CLAUDE.md — 活书 huoshu 项目

> 本文件由 OpenClaw 生成，Claude Code 启动时自动读取。
> 共享上下文和任务看板在 workspace 的 .shared 下（绝对路径），不在本项目内。

## 🚀 启动流程（第一条消息后执行）

1. 读 `/root/.openclaw/workspace/.shared/CONTEXT.md` — 当前活跃任务
2. 读 `/root/.openclaw/workspace/.shared/task-board/active/` — 找第一个 `planned` 状态任务
3. 读 `/root/.openclaw/workspace/.shared/task-board/decisions.md` — 已确认决策（不可推翻）
4. 报告状态 → 有 planned 任务直接开始实现

## 📐 项目定位

- 活书 huoshu：知识图谱 × 自适应学习 × RAG × 复习闭环的独立应用（开源，MIT）
- 完整任务规格：`/root/.openclaw/workspace/.shared/task-board/active/2026-08-01-活书huoshu.md`
- 框架参考（只读）：`/root/projects/learning-agent/`（bookmap schema + skills 协议）
- 设计文档：`/root/projects/learning-agent/docs/完整思路-v3.md`

## ⚠️ 硬约束

- Python ≥ 3.12；Streamlit UI；ChromaDB；Ollama embedding（本地默认）；DeepSeek LLM（可配置）
- 不 fork benkyo，仅借鉴；不随仓库分发教材 PDF（版权）
- 用户数据（projects/、SQLite）必须 gitignore
- 每完成一个里程碑：任务文件 STATUS 改 `ready-for-review`，等 OpenClaw 审查

## 🔄 完成通知

实现完成后通知 OpenClaw 审查（改 STATUS 即可）。
