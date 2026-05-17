from __future__ import annotations

import os
from threading import Lock
from typing import Any

import config
from DataProcessingModule import DataProcessingModule
from GenerationModule import GenerationModule
from IndexConstructionModule import IndexConstructionModule
from RetrieverOptimizationMoudle import RetrieverGenerationMoudle


class RAGService:
    def __init__(self) -> None:
        self._lock = Lock()
        self._initialized = False
        self.processor: DataProcessingModule | None = None
        self.retriever: RetrieverGenerationMoudle | None = None
        self.generator: GenerationModule | None = None
        self.summary: dict[str, Any] = {}

    def initialize(self) -> None:
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            processor = DataProcessingModule(config.DATA_DIR)
            processor.loader_files()
            chunks = processor.split_document()

            index_builder = IndexConstructionModule(
                config.EMBEDDING_MODEL,
                config.INDEX_PATH,
            )

            if os.path.exists(config.INDEX_PATH) and os.listdir(config.INDEX_PATH):
                vectorstore = index_builder.load_index()
            else:
                vectorstore = index_builder.build_vector_index(chunks)
                index_builder.save_index()

            if vectorstore is None:
                raise RuntimeError("向量索引加载失败")

            self.processor = processor
            self.retriever = RetrieverGenerationMoudle(vectorstore, chunks)
            self.generator = GenerationModule()
            self.summary = processor.summary()
            self._initialized = True

    def answer(self, query: str) -> dict[str, Any]:
        self.initialize()

        if not self.processor or not self.retriever or not self.generator:
            raise RuntimeError("RAG 服务尚未完成初始化")

        child_chunks = self.retriever.hybrid_search(query, top_k=config.HYBRID_TOP_K)
        parent_docs = self.processor.get_parent_by_child(child_chunks)
        top_docs = parent_docs[: config.PARENT_TOP_K]
        result = self.generator.generate(query, top_docs)

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_chunks": len(child_chunks),
            "retrieved_docs": len(top_docs),
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "initialized": self._initialized,
            "summary": self.summary,
            "model": config.LLM_MODEL,
        }


rag_service = RAGService()
