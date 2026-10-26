# ResearchRadar 桌面便携版

2026-10-22。面向 Windows x64 的本地科研工作台，保留文献发现、证据问答、研究路线和维护入口。

## 启动与配置

解压完整便携包后双击 `Start.cmd`。完整包内置 `runtime/`，无需预装 Python。独立窗口使用 pywebview 与系统 WebView2；`Start.cmd --browser` 使用系统浏览器。退出窗口会结束启动器自己的本地服务；浏览器模式按 Ctrl+C 退出。

源码环境可运行 `.venv/Scripts/python.exe desktop.py`。仅有源码的 ZIP 不包含 Python。

复制 `.env.example` 为 `.env`，配置对话模型。轻量便携运行时包含远程 Embedding 所需依赖；请将 `EMBEDDINGS_PROVIDER=openai` 并配置可用的 Embedding 模型和服务地址。若选择本地 BGE，需要另行安装 torch、sentence-transformers、langchain-huggingface 并下载模型，不在轻量包内。

## 本轮业务改进

- 多查询候选先统一融合，给不同查询保留代表证据，再截断上下文；单分支失败保留其他结果。
- 两阶段检索保留全范围证据分支，降低论文初筛漏召回；显式论文范围仍然生效。
- 复杂比较问题增加独立证据审阅调用，提出缺失方面，最多补检两个查询，然后生成引用答案。该模型判断用于提出检索需求，不能当作结论正确性的证明。
- 路由模型规划的子查询会继续传入 Skill。常规事实问答不额外调用审阅模型。
- 请求共享模型／工具／检索预算与截止时间。默认最多 4 次模型、1 次业务 Skill、6 次查询级检索调用；一次混合检索内部仍可能调用多个索引。截止时间只在阶段边界检查，已开始的提供方请求仍受各自网络超时控制。
- Skill 模型预算实际执行；工具白名单、数据范围和论文 ID 由可信调用方提供。模型参数不能扩大权限。政策事件只含固定状态码，不记录原始提示词或凭据。

工作流采用“路由规划 → 检索与证据审阅 → 综合回答”的角色分工；检索和权限检查仍是确定性代码，并非每个节点都是自主 Agent。

## 数据与分发

应用目录 data 下保存 research.db、papers、vector_store 等。停止后备份完整 data 目录。初次分发包不包含本机语料、数据库、下载论文、API Key 或日志；空知识库需先摄入论文。迁移数据时保持 SQLite、PDF 和向量索引一致，不能混用不同 Embedding 生成的索引。

只监听 127.0.0.1，保持 Streamlit 原有访问防护。此版为个人本地应用，尚无服务端多租户登录系统。

## 验证与打包

```powershell
python -m pytest -q
python desktop.py --check
python desktop.py --no-ui --shutdown-after 1
python scripts/build_portable.py --output dist/ResearchRadar-portable.zip --runtime PATH_TO_PREPARED_EMBEDDED_PYTHON
```

打包器只选源码和公开资源，可选加入已准备的嵌入式 Python；manifest.json 记录文件哈希。健康检查不等于模型联调。固定回归测试验证覆盖与边界；真实召回率仍需在冻结语料与保留测试集上重新测量。
