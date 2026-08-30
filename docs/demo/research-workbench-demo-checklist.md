# 科研工作台本地演示清单

这份清单用于在本地验证工作台，不把未实际运行的联网结果写成项目指标。

## 运行前

- [ ] 在本机 `.env` 配置已轮换的 provider 和模型凭据，不在截图或终端输出中展示文件内容。
- [ ] 运行离线测试：`.venv\Scripts\python.exe -m pytest -q`。
- [ ] 运行 `..\.venv\Scripts\python.exe app.py provider-health`，只记录标准化状态。
- [ ] 确认 `data/research.db`、`data/papers/`、`data/vector_store/` 和 `data/reports/` 不进入 Git。

## 知识库准备

- [ ] 运行 `.venv\Scripts\python.exe app.py seed --config config/seed_queries.yml`。
- [ ] 再次运行相同 seed，确认不会重复新增 DOI、规范化标题或向量。
- [ ] 运行 `.venv\Scripts\python.exe app.py stats`，查看 metadata、PDF、解析、画像、索引、仅摘要和失败数量。
- [ ] 只有在本地 seed 完成后，才把评估集中的占位符替换为真实 `paper_id/chunk_id`。

## 页面演示顺序

1. `科研助手`：先展示“叶绿素是什么”这类证据不足问题，确认页面给出带边界的通用解释；再展示一次 evidence QA，说明 Agent 只路由到受控 Skill。
2. 在 `科研助手` 对话消息中展开 `运行诊断`：查看本轮 route、skill、检索候选数、证据数、引用数、fallback 和耗时。
3. `知识库维护`：展示元数据、全文证据和向量一致性审计结果。
4. `文献库`：按标题、DOI 和状态筛选 indexed、abstract-only 或 failed 论文，说明证据层级。
5. 普通问候走 `general_chat`，科研问题走受控技能并展示引用；两条路径都在聊天区即时显示用户消息。
6. `任务与日志`、`数据源设置`：展示本地同步记录和点击触发的标准化健康状态。

## 截图要求

- [ ] 截图只包含页面、脱敏标题和稳定状态码。
- [ ] 遮挡 API key、邮箱、绝对路径、完整本地文件名和未公开全文。
- [ ] 不使用伪造的 450–500 条统计数字、召回率或问答质量数字。
- [ ] 空数据库截图可以作为安装后的初始状态，真实数据截图必须来自本机实际运行。
