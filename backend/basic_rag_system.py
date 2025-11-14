import logging
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma  # pyright: ignore[reportMissingImports]
from langchain_community.embeddings import HuggingFaceEmbeddings  # pyright: ignore[reportMissingImports]
from langchain.chains.retrieval_qa.base import RetrievalQA  # pyright: ignore[reportMissingImports]
from langchain_google_genai import ChatGoogleGenerativeAI  # pyright: ignore[reportMissingImports]
from langchain.docstore.document import Document    # pyright: ignore[reportMissingImports]
from prompt_templates import get_prompt  # pyright: ignore[reportMissingImports]
load_dotenv()

logger = logging.getLogger(__name__)

class SimpleRAG:
    def __init__(self):
        """RAGシステムの初期化"""
        logger.info("Initializing SimpleRAG")
        
        # Embeddingモデルの設定(無料・ローカル)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Vector Storeの設定
        self.vectorstore = None
        
        # LLMの設定（Gemini）
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  # 安定版のGemini 2.5 Flash
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        logger.info("SimpleRAG initialization complete")
    
    def load_documents(self, texts):
        """テキストドキュメントを読み込み、ベクトル化して保存"""
        logger.info("Processing %d documents for ingestion", len(texts))
        
        # ドキュメントオブジェクトに変換
        documents = [Document(page_content=text) for text in texts]
        
        # テキストを分割(チャンク化)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)
        logger.info("Chunked into %d pieces", len(splits))
        
        # ベクトルDBに保存
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        logger.info("Vector store persisted to ./chroma_db")
    
    def query(self, question, k=3):
        """質問に対して回答を生成"""
        if self.vectorstore is None:
            logger.warning("Query attempted before documents were loaded")
            return {"error": "ドキュメントが読み込まれていません"}
        
        logger.info("New query received: %s", question)
        logger.debug("Running similarity search with k=%d", k)
        
        # RetrievalQAチェーンの作成
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": get_prompt("basic")}
        )
        
        # 質問を実行
        result = qa_chain({"query": question})
        sources = result["source_documents"]
        logger.info(
            "Generated answer (retrieved %d source docs)",
            len(sources),
        )
        if sources:
            logger.debug("Top source preview: %s", sources[0].page_content[:200])
        
        return {
            "answer": result["result"],
            "sources": sources
        }

load_dotenv()

class PDFRagSystem:
    def __init__(self, persist_directory="./chroma_pdf_db"):
        """PDFベースのRAGシステム初期化"""
        self.persist_directory = persist_directory
        
        # 日本語対応Embeddingモデル
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base"
        )
        
        self.vectorstore = None
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  # 安定版のGemini 2.5 Flash
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
    def load_pdf(self, pdf_path):
        """PDFファイルを読み込み、ベクトル化"""
        logger.info("Loading PDF: %s", pdf_path)
        
        # PDFの読み込み
        loader = PyPDFLoader(pdf_path)  # pyright: ignore[reportUndefinedVariable]
        pages = loader.load()
        logger.info("Loaded %d pages from PDF", len(pages))
        
        # テキスト分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "、", " ", ""]
        )
        splits = text_splitter.split_documents(pages)
        logger.info("Chunked PDF into %d pieces", len(splits))
        
        # ベクトルDB作成
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        logger.info("Persisted PDF embeddings to %s", self.persist_directory)
        
    def query(self, question, k=4):
        """質問応答"""
        if not self.vectorstore:
            logger.warning("PDF query called before vector store initialization")
            return {"error": "PDFが読み込まれていません"}
        
        logger.info("PDF query received: %s (k=%d)", question, k)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": get_prompt("pdf")}
        )
        
        result = qa_chain({"query": question})
        sources = result["source_documents"]
        logger.info("Generated PDF answer (retrieved %d docs)", len(sources))
        if sources:
            logger.debug("PDF top source preview: %s", sources[0].page_content[:200])
        
        return {
            "answer": result["result"],
            "sources": sources
        }

        