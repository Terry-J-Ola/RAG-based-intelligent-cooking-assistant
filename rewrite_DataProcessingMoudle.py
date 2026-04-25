"""
练习重写rag的数据处理模块
内容包括: 数据加载、给父文档添加唯一id、生成映射字典、切割文档、构建父子文档的映射关系
"""

import os
import sys
from langchain_community.document_loaders import TextLoader, DirectoryLoader
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

class DataProcessingMoudle:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path
    # 加载文件
    def Loader_files(self, directory_path: str):
        documents = []
        if not os.path.exists(directory_path):
            print(f"错误：目录 {directory_path} 不存在！")
            sys.exit(1)

        loader = DirectoryLoader(
            directory_path,
            glob="*.md",
            loader_cls=TextLoader,
            loader_kwargs={'encoding':'utf-8'},
            show_progress=True
        )

        md_documents = loader.load() # 被加载的.md文档
        print(f"成功加载 {directory_path} 下的所有.md文件, 一共有 {len(md_documents)} 份.md文档")
        # 增强元数据
        id = 0
        for doc in md_documents:
            id += 1
            doc.metadata.update({'parent_id' : id, 'doc_type' : 'parent_doc'})
            documents.append(doc)
        print(f"成功加载并且增强了 {directory_path} 下的所有.md文件, 一共有 {len(documents)} 份.md文档")
        return documents
    # 切割文件
    def splitter_document(self, documents: List[Document]):
        Splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 100,
            separators=['#', '##', '']
        )
        chunks = Splitter.split_documents(documents)

        


        