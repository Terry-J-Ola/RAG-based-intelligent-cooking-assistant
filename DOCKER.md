# Docker 部署说明

## 1. 准备 `.env`

至少需要配置：

```env
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
APP_HOST=0.0.0.0
APP_PORT=8000
```

如果你不想用默认目录，也可以继续加：

```env
DATA_DIR=/app/data
INDEX_PATH=/app/faiss_index
EMBEDDING_MODEL=/app/local_models/bge-small-zh-v1.5
```

## 2. 推荐方式：用挂载复用本地数据

这样不会把 `data`、`faiss_index`、`local_models` 烧进镜像，重建更快。

```powershell
docker compose up --build
```

启动后访问：

```text
http://127.0.0.1:8000
```

## 3. 仅用 Dockerfile 启动

```powershell
docker build -t my-rag-web .
docker run --rm -p 8000:8000 --env-file .env ^
  -v ${PWD}/data:/app/data ^
  -v ${PWD}/faiss_index:/app/faiss_index ^
  -v ${PWD}/local_models:/app/local_models ^
  my-rag-web
```

## 4. 目录建议

- `data`：知识库原始 Markdown
- `faiss_index`：已构建好的向量索引
- `local_models`：本地嵌入模型

如果容器里没有索引，但有 `data` 和 `local_models`，应用首次启动时会自动构建索引。

## 5. 部署到服务器时要注意

- 服务器要有足够磁盘空间存模型和索引。
- 第一次构建索引会比较慢，建议本地构建好 `faiss_index` 后再挂载到服务器。
- 如果云服务器访问外部模型源受限，尽量走“本地模型目录挂载”的方式。
