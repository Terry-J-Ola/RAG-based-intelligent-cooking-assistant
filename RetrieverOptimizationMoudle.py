"""
Hybrid retrieval: FAISS semantic search + BM25 keyword search + RRF fusion.
"""

import re
from typing import Any, Dict, List

import config
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class RetrieverGenerationMoudle:
    """Hybrid retriever with semantic search and keyword search."""

    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.setup_vectorstore()

    @staticmethod
    def _tokenize_for_bm25(text: str) -> List[str]:
        """Tokenize mixed Chinese/English text for BM25."""
        if not text:
            return []

        lowered = text.lower()
        tokens: List[str] = []
        tokens.extend(re.findall(r"[a-z0-9]+", lowered))
        tokens.extend(re.findall(r"[\u4e00-\u9fff]", lowered))
        return tokens

    def setup_vectorstore(self):
        """Initialize vector and BM25 retrievers."""
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.RETRIEVER_K},
        )
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=config.RETRIEVER_K,
            preprocess_func=self._tokenize_for_bm25,
        )

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """Run semantic + keyword retrieval, then fuse with RRF."""
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        reranked_docs = self._rrf_rerank_1(vector_docs, bm25_docs)
        return reranked_docs[:top_k]

    def _rrf_rerank_1(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document],
        k: int = 60,
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion.
        score = 1 / (k + rank + 1)
        """
        doc_map: Dict[str, tuple[Document, float]] = {}

        for rank, doc in enumerate(vector_docs):
            doc_id = doc.metadata.get("chunk_id", doc.page_content)
            score = 1.0 / (k + rank + 1)
            doc_map[doc_id] = (doc, score)

        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.metadata.get("chunk_id", doc.page_content)
            score = 1.0 / (k + rank + 1)
            if doc_id in doc_map:
                existing_doc, existing_score = doc_map[doc_id]
                doc_map[doc_id] = (existing_doc, existing_score + score)
            else:
                doc_map[doc_id] = (doc, score)

        sorted_docs = sorted(doc_map.values(), key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in sorted_docs]

    def rrf_rerank_2(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document],
        k: int = 60,
    ):
        docs_scores = {}
        docs_objects = {}

        for rank, doc in enumerate(vector_docs):
            doc_id = hash(doc.page_content)
            docs_objects[doc_id] = doc
            docs_scores[doc_id] = docs_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            docs_objects[doc_id] = doc
            docs_scores[doc_id] = docs_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        sorted_docs = sorted(docs_scores.items(), key=lambda item: item[1], reverse=True)
        rerank_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in docs_objects:
                doc = docs_objects[doc_id]
                doc.metadata["rrf_score"] = final_score
                rerank_docs.append(doc)
        return rerank_docs

    def metadata_filtered_search(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int = 5,
    ) -> List[Document]:
        """Retrieve first, then filter by metadata fields."""
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
