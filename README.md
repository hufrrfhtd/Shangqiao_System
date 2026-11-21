# AI-Interview-Demo

一个 Windows 上运行的项目面试 Demo。首页的左侧菜单可切换「业务助手」和「商品智能分类」。业务助手提供两个入口，分别用于批量询价报价，以及查询耗材选型、售后问题和 ERP 价格库存。价格、库存、知识文档与训练语料均为虚构演示数据。

## Windows x64 便携版

1. 下载 Windows x64 便携版。 [AI-Interview-Demo-windows-x64.zip](../../releases/latest/download/AI-Interview-Demo-windows-x64.zip)
2. 将 ZIP 完整解压到可写目录。
3. 双击 `Run-Demo.cmd`。启动后在浏览器打开工作台。终端会显示本地访问地址，按 `Ctrl+C` 停止服务。

便携版内置 Python、运行依赖和演示数据，无需安装 Python、申请 API Key 或联网启动。支持 FAQ、BM25 知识库检索、只读 ERP 查询、批量报价审核与 Excel 导出，以及商品分类的规则基线。DeepSeek、BGE 和 MacBERT 可按下方的开发者说明启用。

## 面试演示指南

### 1. 业务对话

在「业务助手」首页点击「开始对话」，在底部输入问题并发送。左侧可新建对话，也可按会话标题或消息内容搜索，再点选历史继续提问。会话保存在项目的 `.runtime/chat_history.sqlite3`。

- 「HP M404 可以选哪些硒鼓？」查询设备与耗材的兼容依据。
- 「如何申请退货？」展示 FAQ 标准答案。输入「打印时卡纸怎么办？」后补充「HP M404」，演示售后追问和设备型号过滤。
- 「CF258A 的价格和库存是多少？」演示只读 ERP 查询。SKU 及其价格和库存取自 `data/sku.csv`。可在问同一机型的兼容耗材时一并询问价格和库存。

选型和售后文档来自 `data/selection/` 与 `data/after_sales/`，由同一知识库检索。展开检索明细可查看来源、原文、BM25 和可用的向量召回、RRF 融合及精排结果。配置 DeepSeek 后可生成带引用的回答，本地模式可查看引用原文。

选好文档类型后，可将文字 PDF、`.docx` 和 TXT 上传到统一知识库。TXT 可用 `设备型号：HP M404、HP M428` 标注适用设备，用 Markdown 标题标记章节。PDF 扫描件需要先做 OCR，旧版 `.doc` 请转存为 `.docx`。重复上传同一份文件会按 Hash 去重，上传已修改的同名文件可演示增量更新。

### 2. 询价报价

1. 在「业务助手」首页的「询价报价」卡片点击「进入询价」，使用默认询价单，或上传 `data/quotation.xlsx` 或 `data/quotation.csv`。8 行分别涵盖标准型号、简称、自定义别名、打印机型号推选耗材、中文数量、范围数量、超库存和未知型号。
2. 点击「开始询价分析」，对照规则与 DeepSeek 的抽取结果，按原始行号核对型号、数量及类别。DeepSeek 关闭或请求失败时，流程按规则提取并交人工确认。 [LangGraph 中断与恢复](https://docs.langchain.com/oss/python/langgraph/interrupts)
3. 在审核表选择有效 SKU，填写明确数量，或跳过本次无法报价的行。例如审批范围数量时可选 `TON-003` 并填 6。同一 SKU 的多行数量合计也参与库存校验。确认后以同一 `thread_id` 恢复工作流，下载报价 Excel。

### 3. 商品智能分类

1. 加载 `data/products_demo.xlsx` 或上传同结构 Excel，观察缺失字段处理、文本标准化和重复记录识别。输入字段为商品名称、品牌、商品规格、供应商原始型号。
2. 执行分类，检查是否显示 MacBERT CPU 推理或规则基线。结果表展示二级类目、对应的一级类目和固定 ERP 编码。MacBERT 模式下可查看各类的概率。
3. 在低置信度表中确认或修正二级类目，并补齐缺失的商品名称。再下载 ERP 标准 Excel。可提高复核阈值以演示更多待处理项。
4. 在评测入口使用内置标注测试数据，或上传包含「标注二级类目」的自有测试集。Accuracy、Macro-F1 和混淆矩阵由当次预测和实际标签计算。样例评测只用于演示计算链路。

## 开发者运行

### 从源码启动

Windows x64 下载源码 ZIP 后，解压并双击 `Run-Demo.cmd` 也可运行。首次启动时会联网下载 Python 和基础依赖。

安装 uv 后，在项目目录打开 PowerShell，创建 Python 3.11 环境并安装依赖。

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File .\start-demo.ps1
```

终端会显示本地访问地址。Streamlit 通常使用 8501，端口被占用时会尝试下一个端口。需要固定端口时，在命令末尾加 `-Port 8503`。只启动服务时可加 `-Headless`。

### DeepSeek 和本地模型设置

需要扩展功能时，可将 `.env.example` 复制为 `.env`。进程环境变量优先于 `.env`。

- 在 `.env` 填写 `DEEPSEEK_API_KEY`，保留默认 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL=deepseek-flash`，然后重启应用。`DEEPSEEK_ENABLED=0` 可强制走本地演示流程。启用服务时，询价行及生成回答所需的知识片段会发送至配置的 API。 [DeepSeek JSON 输出文档](https://api-docs.deepseek.com/guides/json_mode/)
- BGE 默认用 `BAAI/bge-small-zh-v1.5`。联网时可在 `.env` 设置 `RAG_ALLOW_MODEL_DOWNLOAD=1`，然后发起一次检索下载并缓存权重。离线时将 `BGE_MODEL` 指向包含权重和分词器的本地模型目录。可用 `BAAI/bge-m3` 切换 Embedding，可用 `BGE_RERANKER_MODEL=BAAI/bge-reranker-base` 启用精排，随后在页面勾选。 [BGE 中文小模型](https://huggingface.co/BAAI/bge-small-zh-v1.5) [BGE-M3](https://huggingface.co/BAAI/bge-m3) [BGE Reranker](https://huggingface.co/BAAI/bge-reranker-base)
- MacBERT 需先用训练脚本生成 18 类分类头和本地权重。规则基线的分数用于安排复核，MacBERT 模式显示模型输出的最大类别概率。 [Chinese MacBERT-base](https://huggingface.co/hfl/chinese-macbert-base)

BGE 向量检索、重排序与 MacBERT 使用 CPU 版 PyTorch 和本地模型依赖。

```powershell
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3"
uv pip install --python .venv/Scripts/python.exe -r requirements-models.txt
```

### 训练 MacBERT

```powershell
& .\.venv\Scripts\python.exe train_macbert.py --help
& .\.venv\Scripts\python.exe train_macbert.py
```

脚本默认从 Hugging Face 加载基础模型，也可用 `--base-model` 指定本地权重目录。脚本以 CPU 训练 54 条虚构样例，再用 36 条虚构样例评测。训练完毕后重启应用。若要重新训练，可以用 `--output .runtime/macbert-next` 保存新的检查点，再将 `.env` 中的 `MACBERT_MODEL_DIR` 指向该目录。

## Demo 与生产项目的技术差异

| 环节 | Demo |
| --- | --- |
| ERP / MySQL / PostgreSQL | 虚构 CSV SKU 库与 JSON 配置。问答和报价共用只读价格与库存数据，批量报价审核后导出 Excel。 |
| Redis / 工作流状态 | LangGraph 进程内 Checkpointer 与 Streamlit Session State。报价待审核流程在服务重启后需要重新发起，业务对话历史存于本地 SQLite。 |
| Milvus / 知识库 | 统一知识库使用本地 Chroma 存向量，配合 Python BM25 和文档 Hash 更新。 |
| 分类部署 | CPU 本地批量推理，使用独立脚本训练，固定表配置 ERP 类目映射。 |
| 服务入口 | 一个 Streamlit 进程，提供业务助手和商品分类两个入口。 |

报价、RAG 和分类的示例数据位于 `data/`。会话历史、上传文档、索引和本地权重留在各自的 `.runtime/` 目录，由运行环境生成。

## 自动测试

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Windows Python 3.11 下通过自动测试。实机验证覆盖报价人工审核和 Excel 导出、PDF/Word/TXT 解析、BGE 与 BM25 混合检索、Chroma 增量更新、DeepSeek 原文引用、分类人工复核与 ERP 导出，以及基于官方 MacBERT 的 CPU 训练、评测和 18 类概率推理。

BGE-M3 和 BGE-Reranker 的配置路径已提供，本次未下载并运行这两组可选权重。
