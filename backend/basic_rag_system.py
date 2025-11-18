import logging
import os
import re
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma  # pyright: ignore[reportMissingImports]
from langchain_community.embeddings import HuggingFaceEmbeddings  # pyright: ignore[reportMissingImports]
from langchain.chains.retrieval_qa.base import RetrievalQA  # pyright: ignore[reportMissingImports]
from langchain_google_genai import ChatGoogleGenerativeAI  # pyright: ignore[reportMissingImports]
from langchain.docstore.document import Document    # pyright: ignore[reportMissingImports]
from langchain_core.retrievers import BaseRetriever  # pyright: ignore[reportMissingImports]

from prompt_templates import get_prompt  # pyright: ignore[reportMissingImports]
load_dotenv()

logger = logging.getLogger(__name__)


class StaticRetriever(BaseRetriever):
    """Returns a precomputed set of documents for a given query."""

    documents: List[Document]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.documents

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self.documents


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
            chunk_size=300,
            chunk_overlap=80,
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
                chunk_size=300,
                chunk_overlap=80,
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
        
        # 多様な検索手法を組み合わせて文脈を収集
        combined_docs: List[Document] = []
        seen_keys = set()
        use_threshold = False
        query_variants = [question]

        tokens = re.findall(r"[一-龯ぁ-んァ-ヶA-Za-z0-9]+", question)
        for token in tokens:
            if len(token) >= 2:
                variant = f"{token} 術式 技 能力 詳細"
                if variant not in query_variants:
                    query_variants.append(variant)
        expanded_query = f"{question} 術式 技 能力 詳解"
        if expanded_query not in query_variants:
            query_variants.append(expanded_query)

        def dedup_and_append(docs: List[Document], method: str):
            for doc in docs:
                content = (doc.page_content or "").strip()
                if not content:
                    continue
                metadata = getattr(doc, "metadata", {}) or {}
                doc.metadata = metadata
                metadata.setdefault("retrieval_method", method)
                key = (
                    metadata.get("source"),
                    metadata.get("file_name"),
                    metadata.get("page"),
                    content[:160],
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                combined_docs.append(doc)

        threshold_docs: List[Document] = []
        for variant in query_variants:
            try:
                threshold_retriever = self.vectorstore.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={'score_threshold': 0.5, 'k': k}
                )
                docs = threshold_retriever.get_relevant_documents(variant)
                if docs:
                    use_threshold = True
                    logger.info(
                        "Threshold search (0.5) retrieved %d docs for variant '%s'",
                        len(docs), variant
                    )
                    dedup_and_append(docs, f"threshold:{variant}")
                    threshold_docs.extend(docs)
                if len(combined_docs) >= max(k * 2, 10):
                    break
            except Exception as e:
                logger.warning("Threshold search failed for variant '%s': %s", variant, e)

        if not threshold_docs:
            for variant in query_variants:
                try:
                    similarity_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
                    similarity_docs = similarity_retriever.get_relevant_documents(variant)
                    logger.info("Standard similarity search retrieved %d docs for variant '%s'", len(similarity_docs), variant)
                    dedup_and_append(similarity_docs, f"similarity:{variant}")
                    if len(combined_docs) >= max(k * 2, 10):
                        break
                except Exception as e:
                    logger.warning("Standard similarity search failed for variant '%s': %s", variant, e)

        # MMRを用いて多様性の高い文脈を追加
        try:
            mmr_retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": max(k, 5), "fetch_k": max(10, k * 2), "lambda_mult": 0.5}
            )
            for variant in query_variants[:3]:
                mmr_docs = mmr_retriever.get_relevant_documents(variant)
                logger.info("MMR search retrieved %d docs for variant '%s'", len(mmr_docs), variant)
                dedup_and_append(mmr_docs, f"mmr:{variant}")
                if len(combined_docs) >= max(k * 2, 10):
                    break
        except Exception as e:
            logger.warning("MMR search failed: %s", e)

        # 追加の類似度検索で不足分を補完
        if len(combined_docs) < k:
            for variant in query_variants:
                try:
                    extra_docs = self.vectorstore.similarity_search(variant, k=k)
                    logger.info("Extra similarity search added %d docs for variant '%s'", len(extra_docs), variant)
                    dedup_and_append(extra_docs, f"similarity_extra:{variant}")
                except Exception as e:
                    logger.warning("Extra similarity search failed for variant '%s': %s", variant, e)
                if len(combined_docs) >= max(k * 2, 10):
                    break

        max_docs = max(k, 6)
        selected_docs = combined_docs[:max_docs]

        if not selected_docs:
            logger.warning("No documents available after combined retrieval. Returning empty context.")

        static_retriever = StaticRetriever(documents=selected_docs)
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=static_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": get_prompt("basic")}
        )
        
        # 質問を実行
        result = qa_chain({"query": question})
        sources = result.get("source_documents", selected_docs)
        
        logger.info(
            "Generated answer (retrieved %d source docs, threshold used: %s)",
            len(sources),
            use_threshold,
        )
        if sources:
            logger.debug("Top source preview: %s", sources[0].page_content[:200])
        
        final_answer = result["result"].strip() if isinstance(result, dict) else str(result).strip()
        if sources:
            reference_lines = []
            seen_keys = set()
            for idx, doc in enumerate(sources, 1):
                metadata = getattr(doc, "metadata", {}) or {}
                file_name = metadata.get("file_name")
                source_path = metadata.get("source")
                page_info = metadata.get("page")
                if not file_name and source_path:
                    file_name = Path(source_path).name
                source_label = file_name or (source_path or f"Document {idx}")
                if page_info is not None:
                    source_label += f" (page {page_info})"
                snippet = (doc.page_content or "").replace("\n", " ").strip()
                snippet = snippet[:160] + ("..." if len(snippet) > 160 else "")
                key = (source_label, snippet)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                reference_lines.append(f"- {source_label}: {snippet}")
            if reference_lines:
                final_answer = f"{final_answer}\n\n参考情報:\n" + "\n".join(reference_lines)
        
        return {
            "answer": final_answer,
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

        