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
    """RAG 服务核心编排层

    负责串联整个 RAG 管道的各个模块：
    1. DataProcessingModule   — 文档加载与分块
    2. IndexConstructionModule — 向量索引构建与持久化
    3. RetrieverGenerationMoudle — 混合检索（向量 + BM25）
    4. GenerationModule       — LLM 生成回答

    采用懒初始化 + 双重检查锁定的单例模式，确保索引只构建一次。
    """

    def __init__(self) -> None:
        """初始化 RAG 服务实例，但不执行实际的数据加载和索引构建。

        实际初始化在 initialize() 中懒加载执行，避免启动时就占用大量资源。
        """
        # 线程锁，保证 initialize() 在并发场景下只执行一次
        self._lock = Lock()
        # 初始化标志位，双重检查锁定（DCL）用
        self._initialized = False
        # 数据处理模块：负责从目录加载文件、拆分文档
        self.processor: DataProcessingModule | None = None
        # 检索增强模块：封装向量检索与 BM25 混合检索
        self.retriever: RetrieverGenerationMoudle | None = None
        # 生成模块：调用 LLM 生成最终答案
        self.generator: GenerationModule | None = None
        # 文档摘要统计信息（文件数、总字符数等）
        self.summary: dict[str, Any] = {}

    def initialize(self) -> None:
        """执行完整的 RAG 管道初始化：数据加载 → 分块 → 索引构建 → 检索器/生成器装配。

        采用双重检查锁定（Double-Checked Locking）：
        - 第一重检查避免已初始化时仍获取锁的开销
        - 第二重检查防止多个线程同时通过第一重检查后重复初始化

        初始化流程：
        1. DataProcessingModule 从 DATA_DIR 加载所有文件
        2. 按配置的分块策略将文档拆分为子块（chunks）
        3. IndexConstructionModule 构建向量索引，首次运行时新建，后续直接加载已有索引
        4. 用向量存储和原始分块构造 RetrieverGenerationMoudle
        5. 装配 GenerationModule，并缓存文档摘要
        """
        # 第一重检查：无锁快速路径，已初始化则直接返回
        if self._initialized:
            return

        with self._lock:
            # 第二重检查：获取锁后再次确认，防止重复初始化
            if self._initialized:
                return

            # --- 阶段一：数据处理 ---
            # 从配置的文档目录加载所有文件
            processor = DataProcessingModule(config.DATA_DIR)
            processor.loader_files()
            # 按配置的分块策略（chunk_size、chunk_overlap）拆分文档
            chunks = processor.split_document()

            # --- 阶段二：向量索引构建 ---
            # 使用配置的 Embedding 模型初始化索引构建器
            index_builder = IndexConstructionModule(
                config.EMBEDDING_MODEL,
                config.INDEX_PATH,
            )

            # 若索引目录已存在且非空，直接加载已有索引；否则从 chunks 重新构建
            if os.path.exists(config.INDEX_PATH) and os.listdir(config.INDEX_PATH):
                vectorstore = index_builder.load_index()
            else:
                vectorstore = index_builder.build_vector_index(chunks)
                # 将构建好的向量索引持久化到磁盘，下次启动可复用
                index_builder.save_index()

            # 索引加载/构建失败时，无法继续后续流程，直接抛出异常
            if vectorstore is None:
                raise RuntimeError("向量索引加载失败")

            # --- 阶段三：装配模块 ---
            self.processor = processor
            # 将向量存储和原始分块传入检索器，用于后续混合检索
            self.retriever = RetrieverGenerationMoudle(vectorstore, chunks)
            # 初始化 LLM 生成模块（模型、温度等参数来自 config）
            self.generator = GenerationModule()
            # 缓存文档统计摘要，供 health() 等接口使用
            self.summary = processor.summary()
            self._initialized = True

    def answer(self, query: str) -> dict[str, Any]:
        """对用户查询执行完整的 RAG 问答流程，返回答案及溯源信息。

        RAG 管道流程：
        1. 混合检索 —— 在向量索引中检索与 query 最相似的 top_k 个子块
        2. 子块到父文档映射 —— 将检索到的子块映射回其所属的原始父文档
        3. 取 top-K 父文档 —— 限制传入 LLM 的上下文数量
        4. LLM 生成 —— 基于检索到的文档生成最终答案

        Args:
            query: 用户的自然语言查询字符串

        Returns:
            dict 包含以下字段：
            - answer:            LLM 生成的最终答案文本
            - sources:           答案所引用的来源文档列表
            - retrieved_chunks:  检索阶段返回的子块数量
            - retrieved_docs:    最终传入 LLM 的父文档数量
        """
        # 确保服务已初始化（若已初始化则为空操作）
        self.initialize()

        # 防御性检查：理论上 initialize() 成功后会设置这些字段，但多线程环境下仍需验证
        if not self.processor or not self.retriever or not self.generator:
            raise RuntimeError("RAG 服务尚未完成初始化")

        # 步骤 1：混合检索 —— 从向量索引中召回 top_k 个最相关子块
        child_chunks = self.retriever.hybrid_search(query, top_k=config.HYBRID_TOP_K)
        # 步骤 2：子块 → 父文档映射 —— 将检索到的子块还原为其所属的原始文档
        parent_docs = self.processor.get_parent_by_child(child_chunks)
        # 步骤 3：取前 K 个父文档作为 LLM 的检索上下文
        top_docs = parent_docs[: config.PARENT_TOP_K]
        # 步骤 4：LLM 生成 —— 将 query 和检索到的文档一起送入 LLM，生成带溯源的答案
        result = self.generator.generate(query, top_docs)

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_chunks": len(child_chunks),
            "retrieved_docs": len(top_docs),
        }

    def health(self) -> dict[str, Any]:
        """返回服务健康状态及基本信息。

        Returns:
            dict 包含：
            - status:      固定为 "ok"，表示服务正常运行
            - initialized: 服务是否已完成初始化
            - summary:     已加载文档的统计摘要
            - model:       当前使用的 LLM 模型名称
        """
        return {
            "status": "ok",
            "initialized": self._initialized,
            "summary": self.summary,
            "model": config.LLM_MODEL,
        }


# 模块级 RAG 服务单例，供 FastAPI 路由等模块直接导入使用
rag_service = RAGService()
