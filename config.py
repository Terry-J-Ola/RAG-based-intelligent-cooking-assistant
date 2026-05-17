"""
RAG 菜谱助手 - 集中配置文件
修改环境变量或此文件中的默认值即可调整系统行为
"""

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


def _resolve_path(env_name: str, default_relative: str) -> str:
    raw_value = os.getenv(env_name, default_relative)
    path = Path(raw_value)
    if not path.is_absolute():
        path = BASE_DIR / raw_value
    return str(path)


# ---- 数据 & 索引 ----
DATA_DIR = _resolve_path("DATA_DIR", "data")
INDEX_PATH = _resolve_path("INDEX_PATH", "faiss_index")
EMBEDDING_MODEL = _resolve_path("EMBEDDING_MODEL", "local_models/bge-small-zh-v1.5")

# ---- 检索参数 ----
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
HYBRID_TOP_K = int(os.getenv("HYBRID_TOP_K", "5"))
PARENT_TOP_K = int(os.getenv("PARENT_TOP_K", "3"))

# ---- LLM 设置（OpenAI 兼容 API）----
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash").strip()
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# ---- Web 服务 ----
APP_HOST = os.getenv("APP_HOST", "0.0.0.0").strip()
APP_PORT = int(os.getenv("APP_PORT", "8000"))
