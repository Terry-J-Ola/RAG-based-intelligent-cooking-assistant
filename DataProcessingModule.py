"""
Data processing for the RAG pipeline.
"""

import json
import os
import sys
import uuid
from typing import Dict, List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


class DataProcessingModule:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.parent_child_map: Dict[str, str] = {}
        self.parent_child_map_reverse: Dict[str, List[str]] = {}

    def loader_files(self):
        documents = []
        if not os.path.exists(self.directory_path):
            print(f"错误：目录 {self.directory_path} 不存在！")
            sys.exit(1)

        loader = DirectoryLoader(
            path=self.directory_path,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )

        md_documents = loader.load()
        print(
            f"成功加载 {self.directory_path} 下的所有 .md 文件, 一共有 {len(md_documents)} 份 .md 文档"
        )

        for doc in md_documents:
            source = doc.metadata.get("source", "")
            doc.metadata.update(
                {
                    "parent_id": str(uuid.uuid5(uuid.NAMESPACE_URL, source)),
                    "doc_type": "parent_doc",
                    "source_file": os.path.basename(source),
                }
            )
            documents.append(doc)

        self.documents = documents
        return documents

    def split_document(self):
        self.chunks = []
        self.parent_child_map.clear()
        self.parent_child_map_reverse.clear()

        if not self.documents:
            raise RuntimeError("请先调用 loader_files() 加载文档，再执行切块操作")

        headers_to_split_on = [
            ("#", "主标题"),
            ("##", "副标题"),
            ("###", "三级标题"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

        all_chunks = []
        for doc in self.documents:
            md_chunks = markdown_splitter.split_text(doc.page_content)
            parent_id = doc.metadata["parent_id"]

            for i, chunk in enumerate(md_chunks):
                child_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{parent_id}:{i}"))
                chunk.metadata.update(doc.metadata)
                chunk.metadata.update(
                    {
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "doc_type": "child",
                        "chunk_index": i,
                    }
                )
                self.parent_child_map[child_id] = parent_id
                self.parent_child_map_reverse.setdefault(parent_id, []).append(child_id)

            all_chunks.extend(md_chunks)

        self.chunks = all_chunks
        return all_chunks

    def export_metadata(self, output_path: str):
        metadata_list = []
        for doc in self.documents:
            metadata_list.append(
                {
                    "source": doc.metadata.get("source"),
                    "content_length": len(doc.page_content),
                    "parent_id": doc.metadata.get("parent_id"),
                }
            )

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(metadata_list, file, ensure_ascii=False, indent=2)

        print(f"元数据已导出到：{output_path}")

    def get_parent_by_child(self, child_chunks: List[Document]):
        """
        Map retrieved child chunks back to parent docs.
        Ranking favors earlier ranks and repeated hits, and falls back to
        source_file if an old FAISS index carries stale parent_id metadata.
        """
        parent_docs_by_id = {
            doc.metadata.get("parent_id"): doc
            for doc in self.documents
            if doc.metadata.get("parent_id")
        }
        parent_docs_by_source = {
            doc.metadata.get("source_file"): doc
            for doc in self.documents
            if doc.metadata.get("source_file")
        }

        parent_scores: Dict[str, Dict[str, float | int]] = {}
        parent_docs_map: Dict[str, Document] = {}

        for rank, chunk in enumerate(child_chunks):
            parent_id = chunk.metadata.get("parent_id")
            source_file = chunk.metadata.get("source_file")

            parent_doc = parent_docs_by_id.get(parent_id)
            if parent_doc is None and source_file:
                parent_doc = parent_docs_by_source.get(source_file)
                if parent_doc is not None:
                    parent_id = parent_doc.metadata.get("parent_id")

            if parent_doc is None or not parent_id:
                continue

            stats = parent_scores.setdefault(
                parent_id,
                {"score": 0.0, "hits": 0, "best_rank": rank},
            )
            stats["score"] += 1.0 / (rank + 1)
            stats["hits"] += 1
            stats["best_rank"] = min(stats["best_rank"], rank)
            parent_docs_map[parent_id] = parent_doc

        sorted_parent_ids = sorted(
            parent_scores.keys(),
            key=lambda pid: (
                parent_scores[pid]["score"],
                parent_scores[pid]["hits"],
                -parent_scores[pid]["best_rank"],
            ),
            reverse=True,
        )

        return [
            parent_docs_map[parent_id]
            for parent_id in sorted_parent_ids
            if parent_id in parent_docs_map
        ]

    def summary(self) -> dict:
        parent_count = len(self.documents)
        chunk_count = len(self.chunks)
        return {
            "parent_docs": parent_count,
            "child_chunks": chunk_count,
            "avg_chunks_per_doc": round(chunk_count / parent_count, 1) if parent_count else 0,
        }

    def get_chunks_by_parent(self, parent_id: str) -> List[Document]:
        child_ids = self.parent_child_map_reverse.get(parent_id, [])
        return [chunk for chunk in self.chunks if chunk.metadata["chunk_id"] in child_ids]
