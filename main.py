"""
RAG 菜谱助手 — 端到端命令行入口

用法：
    python main.py

交互：
    问菜谱> 水煮鱼怎么做？
    # 输出 LLM 基于菜谱生成的回答 + 引用来源
    问菜谱> exit   # 退出
"""

import os
from DataProcessingModule import DataProcessingModule
from IndexConstructionModule import IndexConstructionModule
from RetrieverOptimizationMoudle import RetrieverGenerationMoudle
from GenerationModule import GenerationModule
import config


def init_system():
    """加载数据、构建/加载索引、初始化检索器和生成器，只执行一次"""
    print("初始化 RAG 系统中...")

    # ---- 数据处理 ----
    processor = DataProcessingModule(config.DATA_DIR)
    documents = processor.loader_files()
    chunks = processor.split_document()

    # ---- 向量索引 ----
    index_builder = IndexConstructionModule(config.EMBEDDING_MODEL, config.INDEX_PATH)
    if os.path.exists(config.INDEX_PATH) and os.listdir(config.INDEX_PATH):
        vectorstore = index_builder.load_index()
    else:
        print("  首次运行，构建索引...")
        vectorstore = index_builder.build_vector_index(chunks)
        index_builder.save_index()

    if vectorstore is None:
        raise RuntimeError("向量索引加载失败")

    # ---- 检索器 ----
    retriever = RetrieverGenerationMoudle(vectorstore, chunks)

    # ---- 生成器 ----
    generator = GenerationModule()

    print("初始化完成！输入问题开始查询（输入 exit 退出）\n")
    return processor, retriever, generator


def main():
    processor, retriever, generator = init_system()

    while True:
        try:
            query = input("菜谱助手> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("再见！")
            break

        # 检索
        child_chunks = retriever.hybrid_search(query, top_k=config.HYBRID_TOP_K)
        parent_docs = processor.get_parent_by_child(child_chunks)
        top_docs = parent_docs[: config.PARENT_TOP_K]

        # 流式生成
        print()
        for token in generator.generate_stream(query, top_docs):
            print(token, end="", flush=True)
        print("\n")

        # 引用来源
        sources = [
            doc.metadata.get("source_file", "").replace(".md", "")
            for doc in top_docs
            if doc.metadata.get("source_file")
        ]
        if sources:
            print(f"参考菜谱：{'、'.join(sources)}")
        print("-" * 60)


if __name__ == "__main__":
    main()
