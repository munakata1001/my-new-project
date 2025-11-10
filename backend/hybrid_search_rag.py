from langchain_community.vectorstores import Chroma  # pyright: ignore[reportMissingImports]
from langchain_community.embeddings import HuggingFaceEmbeddings  # pyright: ignore[reportMissingImports]
from langchain.retrievers import BM25Retriever, EnsembleRetriever  # pyright: ignore[reportMissingImports]
from langchain.text_splitter import RecursiveCharacterTextSplitter  # pyright: ignore[reportMissingImports]
from langchain.docstore.document import Document  # pyright: ignore[reportMissingImports]

class HybridSearchRAG:
    def __init__(self, alpha=0.5):
        """
        alpha: セマンティック検索とキーワード検索の重み
               0.5 = 両方を均等に使用(推奨)
        """
        self.alpha = alpha
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.vectorstore = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.documents = []
    
    def load_documents(self, texts):
        """文書を読み込み、ハイブリッド検索の準備"""
        documents = [Document(page_content=text) for text in texts]
        
        # チャンキング
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        self.documents = text_splitter.split_documents(documents)
        
        # セマンティック検索用のベクトルDB
        self.vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings
        )
        
        # キーワード検索用のBM25
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 4
        
        # ハイブリッド検索用のEnsemble Retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.vectorstore.as_retriever(search_kwargs={"k": 4}),
                self.bm25_retriever
            ],
            weights=[self.alpha, 1 - self.alpha]
        )
    
    def search(self, query, k=4):
        """ハイブリッド検索を実行"""
        return self.ensemble_retriever.get_relevant_documents(query)[:k]


# 使用例
rag = HybridSearchRAG(alpha=0.5)
rag.load_documents(["文書1", "文書2", "文書3"])
results = rag.search("質問")

"""
reranking_rag.py
Re-rankingを実装したRAGシステム
"""

from sentence_transformers import CrossEncoder
from typing import List, Tuple

class ReRankingRAG:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """CrossEncoderでRe-rankingを実行"""
        self.cross_encoder = CrossEncoder(model_name)
        self.vectorstore = None
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List,
        top_k: int = 4
    ) -> List[Tuple]:
        """文書を再ランキング"""
        # 質問と各文書のペアを作成
        pairs = [[query, doc.page_content] for doc in documents]
        
        # CrossEncoderでスコアリング
        scores = self.cross_encoder.predict(pairs)
        
        # スコアと文書を紐付け
        doc_score_pairs = list(zip(documents, scores))
        
        # スコアの降順でソート
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # 上位top_kを返す
        return doc_score_pairs[:top_k]
    
    def query(
        self, 
        question: str, 
        initial_k: int = 20,
        final_k: int = 4
    ):
        """Re-rankingを使った質問応答"""
        # 初期検索(多めに取得)
        initial_docs = self.vectorstore.similarity_search(
            question, 
            k=initial_k
        )
        
        # Re-ranking実行
        reranked = self.rerank_documents(question, initial_docs, final_k)
        final_docs = [doc for doc, score in reranked]
        
        # LLMで回答生成
        # ... (省略)
        
        return final_docs
