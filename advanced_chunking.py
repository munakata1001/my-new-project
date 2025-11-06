from langchain.text_splitter import RecursiveCharacterTextSplitter  # pyright: ignore[reportMissingImports]
from langchain.docstore.document import Document  # pyright: ignore[reportMissingImports]

def semantic_chunking(text, chunk_size=500, overlap=50):
    """意味的なまとまりを考慮したチャンキング"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[
            "\n\n",  # 段落
            "\n",    # 改行
            "。",    # 日本語の文末
            ".",     # 英語の文末
            "、",    # 読点
            " ",     # スペース
            ""       # 文字単位
        ],
        length_function=len,
    )
    
    docs = [Document(page_content=text)]
    return splitter.split_documents(docs)


def context_aware_chunking(text, chunk_size=500, overlap=100):
    """コンテキストを保持したチャンキング"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", " ", ""],
    )
    
    docs = [Document(page_content=text)]
    chunks = splitter.split_documents(docs)
    
    # メタデータに前後のチャンク情報を追加
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)
        if i > 0:
            chunk.metadata["previous_chunk"] = chunks[i-1].page_content[:100]
        if i < len(chunks) - 1:
            chunk.metadata["next_chunk"] = chunks[i+1].page_content[:100]
    
    return chunks