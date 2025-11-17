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
        
        # 埋め込みモデルを日本語対応の高性能なものに変更
        model_name = "intfloat/multilingual-e5-base"
        logger.info("Using embedding model: %s", model_name)
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
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
        
        # 既存のベクトルストアがある場合は読み込む、なければ新規作成
        persist_dir = "./chroma_db"
        if os.path.exists(persist_dir) and os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
            try:
                logger.info("Loading existing vector store from %s", persist_dir)
                self.vectorstore = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embeddings
                )
                # 既存のストアにドキュメントを追加
                if splits:
                    self.vectorstore.add_documents(splits)
                    logger.info("Added %d new documents to existing vector store", len(splits))
            except Exception as e:
                logger.warning("Failed to load existing vector store, creating new one: %s", e)
                self.vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    persist_directory=persist_dir
                )
        else:
            # 新規作成
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=persist_dir
            )
            logger.info("Created new vector store at %s", persist_dir)
        
        logger.info("Vector store ready at ./chroma_db")
        
        # ベクトルストアの内容を確認（デバッグ用）
        if self.vectorstore is not None:
            try:
                # ベクトルストア内のドキュメント数を取得
                collection = self.vectorstore._collection
                if collection:
                    count = collection.count()
                    logger.info("Vector store contains %d documents", count)
            except Exception as e:
                logger.debug("Could not get vector store count: %s", e)
    
    def add_documents(self, documents):
        """既存のベクトルストアにドキュメントを追加"""
        if self.vectorstore is None:
            logger.warning("Vector store not initialized, creating new one")
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
        else:
            logger.info("Adding %d documents to existing vector store", len(documents))
            # テキストを分割(チャンク化)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
            )
            splits = text_splitter.split_documents(documents)
            logger.info("Chunked into %d pieces", len(splits))
            
            # 既存のベクトルストアに追加
            self.vectorstore.add_documents(splits)
            # 永続化を明示的に実行（persist_directoryが設定されている場合は自動永続化される）
            try:
                if hasattr(self.vectorstore, 'persist'):
                    self.vectorstore.persist()
                    logger.debug("Vector store explicitly persisted")
            except Exception as e:
                logger.debug("Persist method not available or failed: %s", e)
            
            # 追加後のドキュメント数を確認
            try:
                collection = self.vectorstore._collection
                if collection:
                    count = collection.count()
                    logger.info("Documents added to vector store. Total documents: %d", count)
            except Exception as e:
                logger.debug("Could not get vector store count after adding: %s", e)
                logger.info("Documents added to vector store and persisted")
    
    def query(self, question, k=5):
        """質問に対して回答を生成"""
        if self.vectorstore is None:
            logger.warning("Query attempted before documents were loaded")
            return {"error": "ドキュメントが読み込まれていません"}
        
        logger.info("New query received: %s", question)
        logger.info("Running similarity search with k=%d", k)
        
        # まず、直接検索を実行して結果を確認
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
            retrieved_docs = retriever.get_relevant_documents(question)
            logger.info("=" * 80)
            logger.info("DIRECT SEARCH RESULTS (before LLM processing)")
            logger.info("=" * 80)
            logger.info("Direct search retrieved %d documents for query: '%s'", len(retrieved_docs), question)
            
            # 検索結果の詳細をログに記録
            for i, doc in enumerate(retrieved_docs):
                content_preview = doc.page_content[:800].replace('\n', ' ')
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                logger.info("-" * 80)
                logger.info("Retrieved doc %d/%d (length: %d chars, metadata: %s):", 
                           i + 1, len(retrieved_docs), len(doc.page_content), metadata)
                logger.info("Content preview: %s", content_preview)
                logger.info("-" * 80)
            
            # 質問に関連するキーワードが含まれているか確認（日本語の単語分割を改善）
            import re
            # 日本語の文字列を単語に分割（より適切な方法）
            question_words = re.findall(r'[\w]+|[\u3040-\u309F]+|[\u30A0-\u30FF]+', question.lower())
            question_keywords = set(question_words)
            logger.info("Checking if retrieved documents contain question keywords: %s", question_keywords)
            for i, doc in enumerate(retrieved_docs):
                doc_lower = doc.page_content.lower()
                matching_keywords = [kw for kw in question_keywords if kw in doc_lower]
                logger.info("Doc %d: Contains %d/%d keywords: %s", 
                           i + 1, len(matching_keywords), len(question_keywords), matching_keywords)
            
            # 検索結果に質問に関連する内容が含まれていない場合の警告
            if not any(len([kw for kw in question_keywords if kw in doc.page_content.lower()]) > 0 for doc in retrieved_docs):
                logger.warning("WARNING: None of the retrieved documents contain question keywords!")
                logger.warning("This suggests the search may not be finding relevant documents.")
                logger.warning("Consider: 1) Checking if the data is in the vector store")
                logger.warning("          2) Using different search parameters")
                logger.warning("          3) Trying hybrid search (keyword + vector)")
            logger.info("=" * 80)
        except Exception as e:
            logger.exception("Error in direct search: %s", e)
        
        # RetrievalQAチェーンの作成
        # MMR (Maximum Marginal Relevance) 検索を試す（多様性を確保）
        try:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': 0.7, 'k': k}
            )
            logger.info("Using similarity_score_threshold search with threshold=0.7")
        except Exception as e:
            logger.warning("Similarity search with threshold failed, falling back to default similarity search: %s", e)
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
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
            # ソースドキュメントの詳細をログに記録
            for i, source in enumerate(sources[:3]):  # 上位3件をログに記録
                source_preview = source.page_content[:500].replace('\n', ' ')
                source_metadata = source.metadata if hasattr(source, 'metadata') else {}
                logger.info(
                    "Source %d (length: %d, metadata: %s):\n%s...",
                    i + 1, len(source.page_content), source_metadata, source_preview
                )
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

        