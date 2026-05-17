# 🍳 RAG 菜谱助手

基于 **RAG（检索增强生成）** 的智能菜谱问答系统。将你的私人菜谱知识库（Markdown 文档）通过向量索引 + LLM，变成一个可以自由提问、实时流式回答、附带来源引用的 AI 烹饪顾问。

## ✨ 特性

- **📂 本地知识库** — 数百篇菜谱 Markdown 文档，覆盖水产、肉菜、素菜、汤羹、主食、早餐、甜品、饮品、酱料、半成品等十多个品类
- **🔍 混合检索** — 语义向量检索（FAISS）+ 关键词检索（BM25），RRF 倒数排名融合，互补召回
- **🧠 LLM 生成** — 兼容所有 OpenAI 格式 API（DeepSeek / 硅基流动 / Ollama 等），支持流式打字机输出
- **🖥️ 双模式交互**
  - 命令行版 — Rich 美化 TUI，流式输出
  - Web 版 — FastAPI + 原生 HTML/CSS/JS，一键提问
- **🐳 Docker 部署** — 开箱即用的 Dockerfile + docker-compose，挂载本地数据与模型
- **⚙️ 灵活配置** — `.env` 集中管理所有参数（模型、检索 top-k、端口等）

## 🏗️ 架构

```
用户提问
    │
    ▼
┌─────────────────────────────┐
│   RetrieverGenerationMoudle  │  ← 混合检索
│   FAISS 语义向量 + BM25 关键词  │
│          RRF 重排             │
└──────────────┬──────────────┘
               │ Top-K 子块
               ▼
┌─────────────────────────────┐
│   DataProcessingModule      │  ← 父子文档追溯
│   子块 → 父文档（去重排序）     │
└──────────────┬──────────────┘
               │ Top-K 父文档全文
               ▼
┌─────────────────────────────┐
│   GenerationModule          │  ← LLM 生成
│   构建 prompt → OpenAI API  │
│   流式输出 + 来源引用         │
└─────────────────────────────┘
```

## 📦 环境要求

- Python 3.10+
- （可选）Docker & Docker Compose

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd my_rag
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env`，填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env`：

```env
LLM_API_KEY=sk-your-api-key-here
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
```

### 4. 准备嵌入模型

确保 `local_models/bge-small-zh-v1.5/` 目录下有 BGE 模型文件。项目已自带该模型，无需额外下载。

### 5. 运行

**命令行版：**

```bash
python main.py
```

**Web 版：**

```bash
python server.py
# 或
uvicorn server:app --host 0.0.0.0 --port 8000
```

启动后访问 `http://127.0.0.1:8000`

## 🐳 Docker 部署

```bash
docker compose up --build
```

启动后访问 `http://127.0.0.1:8000`

> 数据目录 (`data/`)、向量索引 (`faiss_index/`)、嵌入模型 (`local_models/`) 通过 volume 挂载，无需重新构建镜像。
>
> 详细说明见 [DOCKER.md](DOCKER.md)。

## 📁 项目结构

```
my_rag/
├── main.py                      # 命令行入口（Rich TUI）
├── server.py                    # FastAPI Web 服务入口
├── rag_service.py               # RAG 服务封装（单例，支持 CLI/Web 复用）
├── config.py                    # 集中配置（环境变量 + 默认值）
│
├── DataProcessingModule.py      # 数据处理：加载 Markdown、切分、父子映射
├── IndexConstructionModule.py   # 索引构建：FAISS 向量库、本地加载/保存
├── RetrieverOptimizationMoudle.py # 混合检索：FAISS + BM25 + RRF 重排
├── GenerationModule.py          # LLM 生成：OpenAI 兼容 API、流式输出
│
├── data/                        # 📂 菜谱知识库（Markdown + 图片）
│   ├── aquatic/                 #   水产类
│   ├── meat_dish/               #   肉菜类
│   ├── vegetable_dish/          #   素菜类
│   ├── soup/                    #   汤羹类
│   ├── staple/                  #   主食类
│   ├── breakfast/               #   早餐类
│   ├── dessert/                 #   甜品类
│   ├── drink/                   #   饮品类
│   ├── condiment/               #   酱料类
│   └── semi-finished/           #   半成品类
│
├── local_models/                # 本地嵌入模型 (bge-small-zh-v1.5)
├── faiss_index/                 # 向量索引持久化目录
│
├── templates/index.html         # Web 前端页面
├── static/
│   ├── app.js                   #   前端交互逻辑
│   └── style.css                #   前端样式
│
├── Dockerfile
├── docker-compose.yml
├── DOCKER.md                    # Docker 部署详细说明
├── .env.example                 # 环境变量模板
└── requirements.txt             # Python 依赖
```

## ⚙️ 配置说明

所有配置项在 `.env` 文件中设置：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | LLM API Key（必填） | — |
| `LLM_BASE_URL` | API 地址（OpenAI 兼容） | `https://api.deepseek.com` |
| `LLM_MODEL` | 模型名称 | `deepseek-v4-flash` |
| `LLM_MAX_TOKENS` | 最大生成 token 数 | `1024` |
| `LLM_TEMPERATURE` | 生成温度 | `0.3` |
| `DATA_DIR` | 菜谱数据目录 | `./data` |
| `INDEX_PATH` | 向量索引存储路径 | `./faiss_index` |
| `EMBEDDING_MODEL` | 嵌入模型路径 | `./local_models/bge-small-zh-v1.5` |
| `HYBRID_TOP_K` | 混合检索返回数量 | `5` |
| `PARENT_TOP_K` | 最终送入 LLM 的文档数 | `3` |
| `APP_HOST` | Web 服务监听地址 | `0.0.0.0` |
| `APP_PORT` | Web 服务端口 | `8000` |

## 🧪 添加新菜谱

在 `data/` 对应品类目录下创建新的 `.md` 文件即可。文件格式参考已有菜谱（自由 Markdown 即可，系统会自动解析标题层级进行切分）。

首次运行或索引目录为空时，系统会自动构建 FAISS 索引；添加新菜谱后需删除 `faiss_index/` 目录重新构建。

## 📄 License

MIT
