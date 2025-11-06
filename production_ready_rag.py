import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # pyright: ignore[reportMissingImports]
from langchain.chains import RetrievalQA  # pyright: ignore[reportMissingImports]
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate  # pyright: ignore[reportMissingImports]
from langchain.retrievers import BM25Retriever, EnsembleRetriever  # pyright: ignore[reportMissingImports]
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
import time

load_dotenv()

class ProductionRAG:
    """本番環境向け高精度RAGシステム"""
    
    def __init__(
        self,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        temperature=0.2,
        verbose=True
    ):
        self.verbose = verbose
        self._log("Production RAGシステム初期化中...")
        
        # Embedding
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Re-ranker
        self.cross_encoder = CrossEncoder(reranker_model)
        
        # LLM
        self.llm = OpenAI(temperature=temperature)
        
        # データストア
        self.vectorstore = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.documents = []
        
        # 最適化されたプロンプト
        self.prompt = self._create_optimized_prompt()
        
        self._log("初期化完了!")
    
    def _log(self, message: str):
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def _create_optimized_prompt(self) -> PromptTemplate:
        """本番環境向けの最適化されたプロンプト"""
        template = """あなたは正確で信頼性の高い情報を提供するアシスタントです。

【重要な指示】
1. 文脈に明確に記載されている情報のみを使用してください
2. 推測や一般知識を混ぜないでください
3. 答えが文脈にない場合は、「提供された情報からは回答できません」と正直に伝えてください
4. 可能な限り具体的で、簡潔な回答を心がけてください

【文脈情報】
{context}

【質問】
{question}

【回答】
"""
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def load_documents(
        self,
        texts: List[str],
        chunk_size: int = 800,
        chunk_overlap: int = 150
    ):
        """文書を読み込み、最適化されたチャンキングを実行"""
        self._log(f"{len(texts)}件の文書を処理中...")
        
        from langchain.docstore.document import Document  # pyright: ignore[reportMissingImports]
        documents = [Document(page_content=text) for text in texts]
        
        # チャンキング
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "、", " ", ""],
        )
        self.documents = text_splitter.split_documents(documents)
        
        # ベクトルDB
        self.vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings
        )
        
        # BM25
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        
        # Ensemble Retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.vectorstore.as_retriever(search_kwargs={"k": 10}),
                self.bm25_retriever
            ],
            weights=[0.5, 0.5]
        )
        
        self._log("文書読み込み完了!")
    
    def _rerank(self, query: str, documents: List, top_k: int) -> List:
        """Re-ranking実行"""
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in doc_score_pairs[:top_k]]
    
    def query(
        self,
        question: str,
        initial_k: int = 20,
        final_k: int = 4,
        use_reranking: bool = True
    ) -> Dict[str, Any]:
        """高精度質問応答"""
        if not self.ensemble_retriever:
            return {"error": "文書が読み込まれていません"}
        
        start_time = time.time()
        
        # ハイブリッド検索
        initial_docs = self.ensemble_retriever.get_relevant_documents(question)
        initial_docs = initial_docs[:initial_k]
        
        # Re-ranking
        if use_reranking:
            final_docs = self._rerank(question, initial_docs, final_k)
        else:
            final_docs = initial_docs[:final_k]
        
        # 回答生成
        from langchain.chains.question_answering import load_qa_chain  # pyright: ignore[reportMissingImports]
        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=self.prompt)
        result = chain(
            {"input_documents": final_docs, "question": question},
            return_only_outputs=True
        )
        
        elapsed = time.time() - start_time
        
        return {
            "answer": result["output_text"],
            "sources": final_docs,
            "num_sources": len(final_docs),
            "processing_time": elapsed,
            "used_reranking": use_reranking
        }