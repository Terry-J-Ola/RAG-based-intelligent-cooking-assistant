"""
索引构建模块
向量索引构建 -> 保存索引到本地 -> 从本地加载索引
"""
import os
from typing import List
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

MODELSCOPE_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


class IndexConstructionModule:

    def __init__(self, model_name: str, index_save_path: str):
        self.index_save_path = index_save_path
        self.model_name = self._resolve_model(model_name)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None

    @staticmethod
    def _resolve_model(model_name: str) -> str:
        """如果是远程模型名，从 ModelScope 下载到本地，返回本地路径。"""
        # 已经是本地路径则直接返回
        if Path(model_name).exists():
            return model_name

        # 不含 / 的短名称也视为本地模型名
        if '/' not in model_name:
            return model_name

        local_dir = os.path.join(MODELSCOPE_CACHE, model_name.replace('/', '_'))

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
        """
        构建向量索引

        Arg:
            chunks
        returns:
            FAISS
        """
        if not chunks:
            raise ValueError("文档列表不能为空")
        
        # 构建FAISS向量库
        self.vectorstore = FAISS.from_documents(
            documents = chunks,
            embedding = self.embeddings
            )
        
        return self.vectorstore

    def add_documents(self, new_chunks: List[Document]):
        """向现有索引批量添加新子文档块。"""
        if not self.vectorstore:
            raise ValueError("请先构建索引")
        self.vectorstore.add_documents(new_chunks) # 这里的add_documents()是FAISS实例化对象vectorstor自己的方法。

    def save_index(self):
        """
        保存索引到配置的路径
        """
        if not self.vectorstore:
            raise ValueError("请先构建索引")
        # 确保保存目录存在
        Path(self.index_save_path).mkdir(parents=True,exist_ok=True)
        # 这里的.save_local()方法是是它自带的
        self.vectorstore.save_local(self.index_save_path) 

    def load_index(self):
        """
        从配置路径下加载索引
        """
        if not Path(self.index_save_path).exists():
            raise ValueError("请先加载并保存索引")
        try:
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("索引已经加载成功")
            return self.vectorstore
        except Exception as e:
            print(f"加载向量索引失败: {e}")
            return None
        
    def similarity_search(self, query: str, k: int=5):
        """
        相似度搜索
        Arg: 查询文本, 返回数量
        return: 与query相似的文档列表
        """
        if not self.vectorstore:
            raise ValueError("请先构建或加载向量索引")
        
        return self.vectorstore.similarity_search(query, k = k)
        
