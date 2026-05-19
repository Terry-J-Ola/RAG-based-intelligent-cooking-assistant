"""
FAISS index build/load/save helpers.
"""

import os
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

MODELSCOPE_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class IndexConstructionModule:
    def __init__(self, model_name: str, index_save_path: str):
        self.index_save_path = index_save_path
        self.model_name = self._resolve_model(model_name)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = None

    @staticmethod
    def _resolve_model(model_name: str) -> str:
        """Resolve either a local model path or a remote ModelScope model name."""
        if Path(model_name).exists():
            return model_name

        if "/" not in model_name:
            return model_name

        local_dir = os.path.join(MODELSCOPE_CACHE, model_name.replace("/", "_"))
        if Path(local_dir).exists():
            print(f"模型已缓存: {local_dir}")
            return local_dir

        print(f"从 ModelScope 下载模型 {model_name} ...")
        try:
            from modelscope import snapshot_download

            os.makedirs(MODELSCOPE_CACHE, exist_ok=True)
            downloaded = snapshot_download(model_name, cache_dir=MODELSCOPE_CACHE)
            print(f"模型已下载到: {downloaded}")
            return downloaded
        except Exception:
            print("ModelScope 下载失败，回退到 HuggingFace 直接加载")
            return model_name

    def build_vector_index(self, chunks: List[Document]):
        if not chunks:
            raise ValueError("文档块列表不能为空")

        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )
        return self.vectorstore

    def add_documents(self, new_chunks: List[Document]):
        if not self.vectorstore:
            raise ValueError("请先构建索引")
        self.vectorstore.add_documents(new_chunks)

    def save_index(self):
        if not self.vectorstore:
            raise ValueError("请先构建索引")

        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(self.index_save_path)

    def load_index(self):
        if not Path(self.index_save_path).exists():
            raise ValueError("请先构建并保存索引")

        try:
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print("索引已经加载成功")
            return self.vectorstore
        except Exception as error:
            print(f"加载向量索引失败: {error}")
            return None

    def is_index_compatible(self, chunks: List[Document]) -> bool:
        """
        Validate whether the loaded FAISS store matches the current chunk ids.
        """
        if not self.vectorstore:
            return False

        stored_docs = getattr(getattr(self.vectorstore, "docstore", None), "_dict", None)
        if not isinstance(stored_docs, dict):
            return False

        stored_chunk_ids = {
            doc.metadata.get("chunk_id")
            for doc in stored_docs.values()
            if doc.metadata.get("chunk_id")
        }
        current_chunk_ids = {
            chunk.metadata.get("chunk_id")
            for chunk in chunks
            if chunk.metadata.get("chunk_id")
        }

        return bool(current_chunk_ids) and stored_chunk_ids == current_chunk_ids

    def similarity_search(self, query: str, k: int = 5):
        if not self.vectorstore:
            raise ValueError("请先构建或加载向量索引")

        return self.vectorstore.similarity_search(query, k=k)
