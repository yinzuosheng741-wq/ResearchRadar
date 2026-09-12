# 水色遥感科研助手：工程复核与优化方案

## 1. 逐点判断

这是一个比普通 RAG Demo 更完整的科研工作台：OpenAlex 发现、Crossref/Unpaywall/CORE 合法全文解析、PDF 分页切分、SQLite/FTS5、FAISS、关键词+向量混合检索、引用校验、LangGraph 受控路由、研究路线技能、项目记忆、Streamlit 工作台和离线评估均已覆盖。

## 2. 主要漏洞和边界

- 外部 provider 数量多，网络限流、服务变更和 PDF 下载失败会让全文覆盖率波动。
- PDF 解析对扫描版、复杂表格、公式和多栏排版有限，容易形成“有文本但证据缺失”。
- 本地 FAISS/SQLite 是单机存储，没有多用户权限、租户隔离和备份策略。
- 向量索引与数据库存在一致性风险，需要定期 audit 和 guarded rebuild。
- 模型路由有 fallback，但在线模型、Embedding 和 provider 的依赖安装较重，首次启动体验需要更明确。
- 研究记忆适合“项目级已确认事实”，不应扩展为未经确认的长期用户画像。

## 3. 大厂面试官评价

亮点是把数据源合法性、证据引用、模型路由和评测边界放在了同一条链路里。面试时应重点说明：为什么模型不能直接决定版权边界；引用如何回查 chunk、paper、页码和 quote；证据不足时如何阻止模型伪造答案；项目记忆如何通过确认后写入；以及如何用 Evidence-group Recall、MRR 和 Paper Recall 区分检索质量与生成质量。

## 4. 优化设计

推荐将系统拆成五个可观测阶段：`discover -> lawful_ingest -> parse/profile -> index -> cited_answer`。每阶段输出稳定状态码、耗时、计数和输入 fingerprint；provider 采用统一 adapter + 限速/重试；PDF 解析增加 OCR/版面解析可选层；索引维护使用数据库版本号和原子替换；UI 只呈现脱敏诊断信息。

## 5. 本次实施

- 新增 `scripts/healthcheck.py`，离线检查配置、评测数据和 Web 入口，可用于 CI 或部署后监测。
- 现有 Web 工作台已具备运行诊断、任务日志、数据源健康检查、引用展示和敏感信息脱敏。
- 保留项目当前的“证据不足 fallback”与引用回查边界，不把通用模型回答包装成论文事实。

## 6. 回归监测

```powershell
cd document/intel_agent
python scripts/healthcheck.py
python -m compileall -q agent domain evaluation ingestion model providers rag retrieval storage web workflows
python -m pytest -q
```

当前系统 Python 缺少 `langchain_core`、`langchain_openai` 和 `pypdf`，完整 pytest 需要先执行 `pip install -r requirements.txt`。静态编译和离线 healthcheck 不依赖外部 API。
