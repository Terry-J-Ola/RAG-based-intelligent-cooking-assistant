"""
RAG 菜谱助手 — 端到端命令行入口（Rich 美化版）

用法：
    python main.py
"""

import os

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.prompt import Prompt
from rich.text import Text
from rich.rule import Rule

from DataProcessingModule import DataProcessingModule
from IndexConstructionModule import IndexConstructionModule
from RetrieverOptimizationMoudle import RetrieverGenerationMoudle
from GenerationModule import GenerationModule
import config

console = Console()


def init_system():
    """加载数据、构建/加载索引、初始化检索器和生成器，只执行一次"""

    # ---- 启动横幅 ----
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🍳 菜谱 RAG 助手[/bold cyan]\n[dim]你的私人 AI 烹饪顾问[/dim]",
        border_style="cyan",
    ))
    console.print()

    # ---- 数据处理 ----
    console.print("[bold]📂 加载菜谱数据...[/bold]")
    processor = DataProcessingModule(config.DATA_DIR)
    documents = processor.loader_files()
    chunks = processor.split_document()
    summary = processor.summary()
    console.print(
        f"   [green]✓[/green] 加载 [bold]{summary['parent_docs']}[/bold] 篇菜谱，"
        f"切分为 [bold]{summary['child_chunks']}[/bold] 个子块 "
        f"([dim]平均 {summary['avg_chunks_per_doc']} 块/篇[/dim])"
    )

    # ---- 向量索引 ----
    console.print("[bold]🔧 加载向量索引...[/bold]")
    index_builder = IndexConstructionModule(config.EMBEDDING_MODEL, config.INDEX_PATH)
    if os.path.exists(config.INDEX_PATH) and os.listdir(config.INDEX_PATH):
        vectorstore = index_builder.load_index()
        console.print("   [green]✓[/green] 索引已就绪")
    else:
        console.print("   [yellow]⏳ 首次运行，构建索引...[/yellow]")
        vectorstore = index_builder.build_vector_index(chunks)
        index_builder.save_index()
        console.print("   [green]✓[/green] 索引构建完成")

    if vectorstore is None:
        raise RuntimeError("向量索引加载失败")

    # ---- 检索器 & 生成器 ----
    retriever = RetrieverGenerationMoudle(vectorstore, chunks)
    generator = GenerationModule()

    # ---- 完成提示 ----
    console.print()
    console.print(Rule(style="dim"))
    console.print(Panel.fit(
        "[bold green]✅ 初始化完成！[/bold green]\n"
        "输入菜谱相关问题开始查询  [dim]·[/dim]  输入 [bold]exit[/bold] 退出",
        border_style="green"
    ))
    console.print()

    return processor, retriever, generator


def main():
    processor, retriever, generator = init_system()

    while True:
        # ---- 输入 ----
        try:
            query = Prompt.ask("[bold cyan]🥘 菜谱助手[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold yellow]👋 再见！[/bold yellow]")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            console.print("[bold yellow]👋 再见！[/bold yellow]")
            break

        # ---- 检索 ----
        with console.status("[bold green]🔍 正在检索菜谱...[/bold green]"):
            child_chunks = retriever.hybrid_search(query, top_k=config.HYBRID_TOP_K)
            parent_docs = processor.get_parent_by_child(child_chunks)
            top_docs = parent_docs[: config.PARENT_TOP_K]

        # ---- 流式生成（Live 逐 token 展示）----
        console.print()
        content = ""
        try:
            with Live(
                Panel("", border_style="blue", title="[bold]回答[/bold]"),
                refresh_per_second=10,
                vertical_overflow="visible",
            ) as live:
                for token in generator.generate_stream(query, top_docs):
                    content += token
                    live.update(
                        Panel(
                            Markdown(content),
                            border_style="blue",
                            title="[bold]回答[/bold]",
                        )
                    )
        except Exception:
            console.print_exception()
            continue

        # ---- 引用来源 ----
        sources = [
            doc.metadata.get("source_file", "").replace(".md", "")
            for doc in top_docs
            if doc.metadata.get("source_file")
        ]
        if sources:
            source_text = Text("📖 参考菜谱：", style="bold yellow")
            source_text.append("、".join(sources), style="italic")
            console.print()
            console.print(source_text)
        else:
            console.print("\n[dim]（未找到相关菜谱）[/dim]")

        console.print(Rule(style="dim"))
        console.print()


if __name__ == "__main__":
    main()
