# CLAUDE.md — 活书 huoshu 项目说明

> 活书 huoshu：知识图谱 × 自适应学习 × RAG × 间隔复习的开源自主学习工具。
> 开发环境专用说明，不随开源分发敏感信息。

## 开发协作（双 Agent 协议）

- 完整任务规格与决策见共享任务看板（开发机本地路径，不在此文件公开）
- 编码约定：Python ≥ 3.12、类型注解必填、PEP 8、`src/` + `tests/` 结构
- 质量门：`ruff check src/ tests/` + `mypy src/learning_agent/` + `pytest tests/` 全绿才能报 ready-for-review
- 新依赖必须写入 pyproject.toml；用户数据目录（output/、projects/、*.db）禁止 commit
- 里程碑完成：git commit → 任务文件 STATUS 改 ready-for-review → 等 OpenClaw 审查

## 开源说明

- 本仓库面向开源分发：不包含任何个人学习数据、密钥、内部路径
- 教材 PDF 不入库（版权）；示例用 examples/demo-math.json
