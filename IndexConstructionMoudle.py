from typing import List
from pathlib import Path

from idna import encode
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class IndexConstructionMoudle:

    def __init__(self, model_name: str, index_save_path: str):
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.setup_embedings()

    def setup_embedings(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name = self.model_name,
            model_kwarg = {'device', 'cpu'},
            encode_kwarg = {'normalize', True}
        )
    
    def build_vector_index(self, chunks: List[Document]):
        """
        构建向量索引

        Arg:
            chunks
        returns:
            FAISS
        """
        if not chunks:
            raise ValueError("文档列表不能为空")
        
        self.vectorstore = FAISS.from_documents(
            documents = chunks,
            embeddings = self.embeddings
            )
        
        return self.vectorstore
    def add_document(self, new_chunks: List[Document]):
        """
        向现有的索引添加新的文档
        Arg:
            new_chunks:新的文档
        """
        if not self.vectorstore:
            raise ValueError("请先构建索引")
        self.vectorstore.add_documents(new_chunks) # 这里的add_documents()是FAISS实例化对象vectorstor自己的方法。

    def save_index(self):
        """
        保存索引到配置的路径
        """
        if not self.vectorstore:
            raise ValueError("请先构建索引")
        # 确保保存目录存在
        Path(self.index_save_path).mkdir(parent=True,exist_ok=True)

        self.vectorstore.save_local(self.index_save_path)


    


