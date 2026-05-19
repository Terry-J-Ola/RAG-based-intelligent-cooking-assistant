"""
CLI entrypoint for the recipe RAG assistant.

Usage:
    python main.py
"""

import os

import config
from DataProcessingModule import DataProcessingModule
from GenerationModule import GenerationModule
from IndexConstructionModule import IndexConstructionModule
from RetrieverOptimizationMoudle import RetrieverGenerationMoudle
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text

console = Console()


def init_system():
    """Load data, validate or rebuild index, then initialize retriever and generator."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Recipe RAG Assistant[/bold cyan]\n[dim]Your private AI cooking helper[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    console.print("[bold]Loading recipe documents...[/bold]")
    processor = DataProcessingModule(config.DATA_DIR)
    processor.loader_files()
    chunks = processor.split_document()
    summary = processor.summary()
    console.print(
        f"   [green]OK[/green] Loaded [bold]{summary['parent_docs']}[/bold] recipes, "
        f"split into [bold]{summary['child_chunks']}[/bold] chunks "
        f"([dim]avg {summary['avg_chunks_per_doc']} chunks/doc[/dim])"
    )

    console.print("[bold]Loading vector index...[/bold]")
    index_builder = IndexConstructionModule(config.EMBEDDING_MODEL, config.INDEX_PATH)
    if os.path.exists(config.INDEX_PATH) and os.listdir(config.INDEX_PATH):
        vectorstore = index_builder.load_index()
        if vectorstore is not None and not index_builder.is_index_compatible(chunks):
            console.print("   [yellow]Index is stale. Rebuilding to match current documents...[/yellow]")
            vectorstore = index_builder.build_vector_index(chunks)
            index_builder.save_index()
        console.print("   [green]OK[/green] Index ready")
    else:
        console.print("   [yellow]No local index found. Building a new one...[/yellow]")
        vectorstore = index_builder.build_vector_index(chunks)
        index_builder.save_index()
        console.print("   [green]OK[/green] Index built")

    if vectorstore is None:
        raise RuntimeError("Failed to load vector index")

    retriever = RetrieverGenerationMoudle(vectorstore, chunks)
    generator = GenerationModule()

    console.print()
    console.print(Rule(style="dim"))
    console.print(
        Panel.fit(
            "[bold green]Initialization complete[/bold green]\n"
            "Ask a recipe question to start. Type [bold]exit[/bold] to quit.",
            border_style="green",
        )
    )
    console.print()

    return processor, retriever, generator


def main():
    processor, retriever, generator = init_system()

    while True:
        try:
            query = Prompt.ask("[bold cyan]Recipe Assistant[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold yellow]Bye[/bold yellow]")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            console.print("[bold yellow]Bye[/bold yellow]")
            break

        with console.status("[bold green]Retrieving relevant recipes...[/bold green]"):
            child_chunks = retriever.hybrid_search(query, top_k=config.HYBRID_TOP_K)
            parent_docs = processor.get_parent_by_child(child_chunks)
            top_docs = parent_docs[: config.PARENT_TOP_K]

        console.print()
        content = ""
        try:
            with Live(
                Panel("", border_style="blue", title="[bold]Answer[/bold]"),
                refresh_per_second=10,
                vertical_overflow="visible",
            ) as live:
                for token in generator.generate_stream(query, top_docs):
                    content += token
                    live.update(
                        Panel(
                            Markdown(content),
                            border_style="blue",
                            title="[bold]Answer[/bold]",
                        )
                    )
        except Exception:
            console.print_exception()
            continue

        sources = [
            doc.metadata.get("source_file", "").replace(".md", "")
            for doc in top_docs
            if doc.metadata.get("source_file")
        ]
        if sources:
            source_text = Text("Sources: ", style="bold yellow")
            source_text.append(" / ".join(sources), style="italic")
            console.print()
            console.print(source_text)
        else:
            console.print("\n[dim](No relevant recipe found)[/dim]")

        console.print(Rule(style="dim"))
        console.print()


if __name__ == "__main__":
    main()
