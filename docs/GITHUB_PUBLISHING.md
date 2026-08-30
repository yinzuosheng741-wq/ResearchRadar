# GitHub 发布检查

这个仓库只提交可复现的源码、配置模板、Prompt、测试、评估输入和使用文档。运行时数据留在本机。

## 不提交

- `.env` 和任何 API key、邮箱或 webhook；
- SQLite 数据库、PDF、FAISS 索引和生成报告；
- 虚拟环境、缓存、日志和临时测试目录。

这些路径已在 `.gitignore` 中排除。发布前仍要检查暂存区，不要只依赖忽略规则。

## 发布前命令

```powershell
git status --short --ignored
git diff --check
python -m pytest -q
python app.py --help
```

检查结果中不得出现 `.env`、`research.db`、PDF、`index.faiss`、`index.pkl` 或凭据文件。`knowledge-audit` 需要本地数据库和向量索引，适合在已配置运行环境中执行，不应作为空克隆的安装步骤。

## 公开仓库结构

```text
agent/ domain/ evaluation/ ingestion/
providers/ rag/ retrieval/ storage/ workflows/
web/ utils/ model/ prompts/ config/
tests/ data/evaluation/ docs/
app.py  README.md  requirements.txt  .env.example
```

项目只使用用户自行配置的开放数据源和本地运行数据，不在仓库中上传受版权保护的全文或个人凭据。
