"""
生成模块：接收检索到的菜谱父文档，构建 prompt 送入 LLM，返回带来源引用的回答
支持所有 OpenAI 兼容 API（DeepSeek、硅基流动、Ollama 等）
"""

from typing import List, Dict, Generator
from openai import OpenAI
from langchain_core.documents import Document
import config


class GenerationModule:
    """基于检索到的菜谱文档，用 LLM 生成烹饪回答"""

    def __init__(
        self,
        api_key: str = config.LLM_API_KEY,
        base_url: str = config.LLM_BASE_URL,
        model: str = config.LLM_MODEL,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    @staticmethod
    def _build_prompt(query: str, parent_docs: List[Document]) -> str:
        """将菜谱全文拼接为 prompt context，每个菜谱包裹在 --- 分隔线中"""
        context_blocks = []
        for i, doc in enumerate(parent_docs):
            source = doc.metadata.get("source_file", f"菜谱{i+1}")
            context_blocks.append(f"[{source}]\n{doc.page_content}\n---\n")

        context_text = "\n".join(context_blocks)
        return f"""你是一个专业的厨师助手。请严格基于以下菜谱内容回答用户的问题。
如果菜谱中没有相关信息，请如实说"当前菜谱中没有提到"。
回答时请引用具体的菜谱名称（如「水煮鱼」）。

参考菜谱：
---
{context_text}
用户问题：{query}"""

    def generate_stream(self, query: str, parent_docs: List[Document]) -> Generator[str, None, None]:
        """流式生成：逐 token 产出文本，适合实时打字机效果"""
        if not parent_docs:
            yield "没有找到相关菜谱，请换个问法试试。"
            return

        messages = [
            {"role": "user", "content": self._build_prompt(query, parent_docs)},
        ]

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def generate(self, query: str, parent_docs: List[Document]) -> Dict:
        """
        Args:
            query: 用户问题
            parent_docs: 检索到的父文档列表（已排序）

        Returns:
            {"answer": str, "sources": [文件名, ...]}
        """
        if not parent_docs:
            return {
                "answer": "没有找到相关菜谱，请换个问法试试。",
                "sources": [],
            }

        messages = [
            {"role": "user", "content": self._build_prompt(query, parent_docs)},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
        )

        answer = response.choices[0].message.content

        sources = []
        for doc in parent_docs:
            src = doc.metadata.get("source_file", "")
            if src:
                sources.append(src.replace(".md", ""))

        return {"answer": answer, "sources": sources}
