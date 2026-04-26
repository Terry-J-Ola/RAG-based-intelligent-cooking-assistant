"""
练习重写rag的数据处理模块
内容包括: 数据加载、给父文档添加唯一id、生成映射字典、切割文档、构建父子文档的映射关系
"""

import os
import sys
from langchain_community.document_loaders import TextLoader, DirectoryLoader
import uuid
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from typing import List, Dict

class DataProcessingMoudle:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.parent_child_map: Dict[str, str] = {}
        self.parent_child_map_reverse: Dict[str, List[str]] = {}  # parent_id → [child_id, ...]
    # 加载文件
    def loader_files(self):
        documents = []
        if not os.path.exists(self.directory_path):
            print(f"错误：目录 {self.directory_path} 不存在！")
            sys.exit(1)

        loader = DirectoryLoader(
            path=self.directory_path,
            glob="*.md",
            loader_cls=TextLoader,
            loader_kwargs={'encoding':'utf-8'},
            show_progress=True
        )

        md_documents = loader.load() # 被加载的.md文档
        print(f"成功加载 {self.directory_path} 下的所有.md文件, 一共有 {len(md_documents)} 份.md文档")
        # 增强父文档元数据：为每份父文档添加唯一ID、文档类型及原始路径信息，
        # 方便后续切割后通过映射字典追溯子文档与父文档的归属关系。
        for doc in md_documents:
            # .update()方法会覆盖已有的键值,原数据中没有则添加进去
            # 原数据有的，但new数据没有则不变
            doc.metadata.update({
                'parent_id': str(uuid.uuid4()),          # 全局唯一标识符
                'doc_type': 'parent_doc',   # 标记为父文档，区别于后续切割生成的子文档
                'source_file': os.path.basename(doc.metadata.get('source', '')),  # 保留原始文件名
            })
            documents.append(doc)
        self.documents = documents
        return documents
    # 切割文件
    def split_document(self):
        # 防呆：确保先加载文档再切割
        if not self.documents:
            raise RuntimeError("请先调用 loader_files() 加载文档，再执行切割操作")

        # 定义要分割的标题层级
        head_to_split_on = [
            ("#","主标题"),
            ("##","副标题"),
            ("###","三级标题")
        ]
        # 使用markdown分割器
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=head_to_split_on,
            strip_headers=False #保留标题，便于理解上下文
        )
        all_chunks = []

        for doc in self.documents:
            # 对每个文档进行Markdown分割
            md_chunks = markdown_splitter.split_documents(doc)
            parent_id = doc.metadata["parent_id"]
            # 对每个chunk进行元数据增强并建立映射
            for i, chunk in enumerate(md_chunks):
                # 为每一个子块分配唯一id
                child_id = str(uuid.uuid4())
                # 合并原文档的元数据和新的标题元数据
                chunk.metadata.update(doc.metadata)
                chunk.metadata.update({
                    "chunk_id":child_id,
                    "parent_id":parent_id,
                    "doc_type":"child",
                    "chunk_index":i
                    })
                # 建立父子映射关系字典
                self.parent_child_map[child_id] = parent_id
                # 同时维护反向映射：parent_id → [child_id, ...]，方便按父文档查找所有子块
                self.parent_child_map_reverse.setdefault(parent_id, []).append(child_id)
                #举个例子：
                #如果 child_id 是 "小明",parent_id 是 "大明"。
                #执行后，字典里就会存入：{"小明": "大明"}。
                #这样当你拿着 "小明" 去查这个字典时，立刻就能知道他的父亲是 "大明"。
            # 再对每一个父文档的所有chunks增强后,添加回去
            all_chunks.extend(md_chunks)
        self.chunks = all_chunks
        return all_chunks

    def summary(self) -> dict:
        """快速查看数据处理概况，方便调试。"""
        parent_count = len(self.documents)
        chunk_count = len(self.chunks)
        return {
            "parent_docs": parent_count,
            "child_chunks": chunk_count,
            "avg_chunks_per_doc": round(chunk_count / parent_count, 1) if parent_count else 0,
        }

    def get_chunks_by_parent(self, parent_id: str) -> List[Document]:
        """根据父文档ID检索其所有子块。"""
        child_ids = self.parent_child_map_reverse.get(parent_id, [])
        return [c for c in self.chunks if c.metadata["chunk_id"] in child_ids]


        