"""
检索生成模块：混合向量+BM25检索, 支持RRF重排与元数据过滤
"""

from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

class RetrieverGenerationMoudle:
    """混合检索器：语义向量检索(FAISS) + 关键词检索(BM25)，RRF融合排序"""

    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        """
        Args:
            vectorstore: 已构建的FAISS向量库
            chunks: 原始文档切片列表, 用于初始化BM25检索器
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.setup_vectorstore()

    def setup_vectorstore(self):
        """初始化向量检索器(similarity)和BM25关键词检索器, 设置默认召回数k=5"""
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type = "similarity",
            search_kwargs = {'k': 5}
        )
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k = 5
            )

    def hybrid_search(self, query: str, top_k: int=3) -> List[Document]:
        """混合检索：对两路召回结果进行RRF重排，返回top_k条"""
        # 并行调用两路检索器：语义向量 + 关键词BM25，互补召回
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        # RRF倒数排名融合，无需模型即可合并排序
        reranked_docs = self._rrf_rerank_1(vector_docs, bm25_docs)
        return reranked_docs[:top_k]

    def _rrf_rerank_1(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        """
        RRF重排: 对两路检索结果进行倒数排名融合
        score = 1/(k+rank)，两路都命中的文档分数累加
        """

        doc_map: Dict[str, tuple] = {}

        # 第一路：向量检索结果，rank越小表示排名越靠前
        for rank, doc in enumerate(vector_docs):
            doc_id = doc.metadata.get("id", doc.page_content)
            # RRF公式: score = 1 / (k + rank)，排名越靠前分数越高
            score = 1.0 / (k + rank + 1)
            doc_map[doc_id] = (doc, score)

        # 第二路：BM25关键词检索，重复文档分数累加
        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.metadata.get("id", doc.page_content)
            score = 1.0 / (k + rank + 1)
            if doc_id in doc_map:
                # 同一文档在两路都出现，累加分数（增强置信度）
                existing_doc, existing_score = doc_map[doc_id]
                doc_map[doc_id] = (existing_doc, existing_score + score)
            else:
                doc_map[doc_id] = (doc, score)

        # 按融合分数由高到低排列
        sorted_docs = sorted(doc_map.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs]
    
    def rrf_rerank_2(self, vector_docs: List[Document], bm25_docs: List[Document], k: int=60):
        docs_scores = {} # 存 哈希值：rrf分数
        docs_objects = {} # 存 哈希值：doc
        # 计算向量检索结果的rrf分数
        for rank, doc in enumerate(vector_docs):
            # 使用文档内容的哈希值来做唯一标识
            doc_id = hash(doc.page_content)
            docs_objects[doc_id] = doc
            # rrf公式: 1/(k+rank)
            rrf_score = 1.0 / (k + rank + 1)
            # 如果有内容重复的文档, 那么哈希值一定是一样的, 那么就会先获取它此前的分数再加上现在的分数, 没有就返回0
            docs_scores[doc_id] = docs_scores.get(doc_id, 0) + rrf_score

        # 计算BM25检索结果的rrf分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            docs_objects[doc_id] = doc
            rrf_score = 1.0 / (k + rank + 1)
            docs_scores[doc_id] = docs_scores.get(doc_id, 0) + rrf_score

        # 按照rrf分数排序
        sorted_docs = sorted(docs_scores.items(), key=lambda x: x[1], reverse=True)
        # 构建最终结果
        rerank_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in docs_objects:
                doc = docs_objects[doc_id]
                # 将rrf分数加入背景信息
                doc.metadata['rrf_score'] = final_score
                rerank_docs.append(doc)
        return rerank_docs

    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        元数据过滤检索：先检索再按字段过滤
        先放大召回量(top_k*3)，过滤后再截断，避免过滤后数量不足
        """
        # 放大召回，防止元数据过滤后结果不足
        docs = self.hybrid_search(query, top_k * 3)

        filtered_docs = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered_docs.append(doc)

        return filtered_docs[:top_k]
