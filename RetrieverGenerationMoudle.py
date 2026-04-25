"""
索引生成模块
"""

from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

class RetrieverGenerationMoudle:
    """
    初始化
    """
    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.setup_vectorstore()

    

    """
    设置检索器, 分别是向量检索器和bm25检索器
    """
    def setup_vectorstore(self):
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type = "similarity",
            search_kwargs = {'k', 5}
        )
        self.bm25_retriever = BM25Retriever.from_focuments(
            self.chunks,
            k = 5
        )
    """
    设置完检索器开始启动这两个检索器, 并且开始重排
    """
    def hybrid_search(self, query: str, top_k: int=3) -> List[Document]:
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        reranked_docs = self.rrf_reank(vector_docs, bm25_docs)
        return reranked_docs[:top_k]
    """
    设置一个过滤器, 用元数据过滤
    """
    def metadata_filtered_search(self, query: str, fliters: Dict[str,any], top_k: int = 5):
        docs = self.hybrid_search(query, top_k * 3)

        filtered_doc = []
        for doc in docs:
            for key, value in fliters.items():


