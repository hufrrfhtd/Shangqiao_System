# 耗材工作台

耗材工作台是面向打印机耗材的本地业务系统，提供批量询价报价、耗材选型与售后问答、商品分类。系统内置示例商品、询价单和技术文档，价格与库存来自本地模拟 ERP。

## 快速开始

1. [下载 Windows x64 便携版](https://github.com/hufrrfhtd/Shangqiao_System/releases/latest/download/AI-Interview-Demo-windows-x64.zip)。
2. 将 ZIP 完整解压到有写入权限的目录。
3. 双击 `Run-Demo.cmd`，在浏览器中打开工作台。若浏览器没有自动打开，可访问终端显示的本地地址。

便携版内置 Python 和运行依赖，解压后即可离线使用。按 `Ctrl+C` 可停止本地服务。

## 功能与用法

### 询价报价

1. 在「业务助手」点击「进入询价」，上传 Excel 或 CSV，或直接使用内置的示例询价单。表格应包含商品描述或型号、数量，可补充品牌与兼容要求。
2. 点击「开始询价分析」。系统匹配 SKU，查询商品单价和库存，并标出需要人工核对的记录。
3. 有待审核记录时，在「人工审核」中逐行选择「确认报价」或「跳过」。确认的记录需要核对 SKU 和数量，同一 SKU 的合计数量会参与库存校验。审核完毕后点击「确认并生成报价」。
4. 报价生成后，点击「下载报价 Excel」。

### 业务对话

在「业务助手」点击「开始对话」，在右侧底部输入问题。左侧可新建对话、搜索历史会话，或打开旧会话继续提问。

| 可查询的内容 | 示例问题 |
| --- | --- |
| 耗材选型 | HP M404 可以用哪些硒鼓？ |
| 价格与库存 | CF258A 的价格和库存是多少？ |
| 故障排查与售后政策 | HP M404 卡纸怎么处理？如何申请退货？ |

选型与售后问题使用同一知识库。询问价格或库存时，系统会按需查询只读 ERP。回复下方可展开「查看检索排名与原文」。对话记录保存在本机的 `.runtime/chat_history.sqlite3`。

需要添加资料时，打开「检索设置」中的「管理知识库」，选择文档类型，上传可提取文字的 PDF、DOCX 或 TXT，再点击「更新知识库」。

### 商品智能分类

1. 在左侧选择「商品智能分类」，上传 Excel 或使用内置示例。标准列为「商品名称」「品牌」「商品规格」「供应商原始型号」。
2. 点击「清洗并分类」，查看预测类目、置信度和对应的 ERP 编码。
3. 在「人工复核」中确认或修正待复核商品的类目，补齐缺失的商品名称，再点击「下载 ERP 标准 Excel」。

如需评估分类效果，可在「标注数据评测」中使用内置测试集，或上传含「标注二级类目」列的测试表。

## 从源码运行与开启可选功能

下载源码后也可以双击 `Run-Demo.cmd`。首次启动会联网准备 Python 和运行依赖。此方式和便携版均默认使用本地 BM25 检索与规则分类。

需要启用 DeepSeek、BGE 或 MacBERT 时，先安装 uv，再在源码目录创建运行环境并按本节末尾的命令启动。 [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/)

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
if (-not (Test-Path '.env')) { Copy-Item '.env.example' '.env' }
```

- **DeepSeek** 支持询价信息抽取和带引用的问答。在 `.env` 填写 `DEEPSEEK_API_KEY`。启用后，查询或抽取所需的内容会发送到配置的 API。
- **BGE** 支持向量检索和重排序。安装下方模型依赖后，可在 `.env` 设置 `RAG_ALLOW_MODEL_DOWNLOAD=1`，或将 `BGE_MODEL` 指向本地权重目录。可选的精排模型通过 `BGE_RERANKER_MODEL` 设置。
- **MacBERT** 支持查看 18 个二级类目的模型概率。安装模型依赖后运行 `train_macbert.py` 生成分类权重。首次训练会下载基础模型，也可用 `--base-model` 指定本地模型目录。

```powershell
# 使用 BGE 或 MacBERT 时安装
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3"
uv pip install --python .venv/Scripts/python.exe -r requirements-models.txt

# 使用 MacBERT 时训练
& .\.venv\Scripts\python.exe train_macbert.py
```

配置完成后，在项目目录运行以下命令。

```powershell
& .\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```
