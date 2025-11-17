from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from basic_rag_system import SimpleRAG  # 既存のクラスを利用
from image_processor import find_images_in_directory, process_image_file
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import logging
import os
import traceback

LOG_LEVEL = os.getenv("RAG_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_RAG_OVERVIEW = """
このRAG（Retrieval Augmented Generation）システムは、次の4層で構成されています。

1. フロントエンド: Next.js (App Router) + Tailwind。ユーザーがブラウザ上で質問を送信し、Geminiベースの回答を受け取ります。
2. APIレイヤー: FastAPI。`/ask` エンドポイントが質問を受け取り、RAGエンジンを呼び出します。CORS設定済みなので `localhost:3000` からアクセスできます。
3. RAGエンジン: `basic_rag_system.SimpleRAG`。HuggingFaceの `all-MiniLM-L6-v2` で埋め込みを生成し、ChromaベクトルDBに保存、LangChainの `RetrievalQA` を通じて Gemini 2.5 Flash モデルに文脈付きで質問します。
4. ドキュメント前処理: `advanced_chunking.py` を使ったチャンク分割や Seed 文書のロード機能。アプリ起動時に概要テキストを登録し、質問が来たら関連チャンクを再検索します。

ユーザーが「このRAGシステムについて教えて」と尋ねた場合、上記の構成を説明する回答が返るよう、概要テキストが初期ロードされています。
"""

# FastAPIアプリケーションのインスタンスを作成
app = FastAPI()

# CORSヘッダーをすべてのレスポンスに追加するカスタムミドルウェア
class CORSHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

# カスタムミドルウェアを追加（CORSミドルウェアより先に実行）
app.add_middleware(CORSHeaderMiddleware)

# CORS (Cross-Origin Resource Sharing) の設定
# フロントエンド (localhost:3000) からのアクセスを許可するために必要
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開発環境では全てのオリジンを許可
    allow_credentials=False,  # allow_origins=["*"] の場合は False にする必要がある
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAGエンジンを保持するグローバル変数
rag_engine: SimpleRAG | None = None

@app.on_event("startup")
def startup_event():
    """
    アプリケーション起動時に一度だけ実行される関数。
    ここでRAGシステムのセットアップを行います。
    """
    global rag_engine
    logger.info("Starting backend startup workflow")

    # SimpleRAG を使った最小セットアップ（テキストを少量投入）
    rag_engine = SimpleRAG()

    # 簡易なシード文書（最低限ベクトルDBを初期化して動作確認できるように）
    seed_text = os.getenv(
        "RAG_SEED_TEXT",
        DEFAULT_RAG_OVERVIEW.strip()
    )
    rag_engine.load_documents([seed_text])
    
    # Process files from a dedicated 'source_documents' directory
    source_documents_path = "./source_documents"
    if not os.path.exists(source_documents_path):
        logger.warning("Source documents directory not found, creating one: %s", source_documents_path)
        os.makedirs(source_documents_path)
        logger.info("Please add your PDF, text, or image files to the '%s' directory.", source_documents_path)

    logger.info("Scanning for files in %s", source_documents_path)
    
    # 画像ファイルを処理
    image_files = find_images_in_directory(source_documents_path)
    if image_files:
        logger.info("Found %d image files, processing...", len(image_files))
        for image_path in image_files:
            try:
                logger.info("Processing image: %s", image_path)
                doc = process_image_file(image_path)
                rag_engine.add_documents([doc])
                logger.info("Successfully added image content to RAG system")
            except Exception as e:
                logger.exception("Failed to process image %s: %s", image_path, e)
    else:
        logger.info("No image files found in %s", source_documents_path)
    
    # PDFとテキストファイルを処理
    pdf_files = []
    text_files = []
    # Chroma DBの内部ファイルを除外するリスト
    chroma_internal_files = {
        'data_level0.bin', 'header.bin', 'length.bin', 'link_lists.bin',
        'chroma.sqlite3', 'chroma.sqlite3-wal', 'chroma.sqlite3-shm',
        'index_metadata.pickle'  # Chroma DBのメタデータファイルも除外
    }
    
    for root, dirs, files in os.walk(source_documents_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = Path(file).suffix.lower()
            file_name = file.lower()
            
            # Chroma DBの内部ファイルはスキップ (source_documentsには無いはずだが念のため)
            if file_name in chroma_internal_files:
                logger.debug("Skipping Chroma DB internal file: %s", file_path)
                continue
            
            # サブディレクトリ内のChroma DB内部ファイルもスキップ
            if any(internal_file in file_name for internal_file in ['data_level', 'header', 'link_list', 'index_metadata']):
                logger.debug("Skipping Chroma DB internal file: %s", file_path)
                continue
            
            if file_ext == '.pdf':
                pdf_files.append(file_path)
                logger.info("Found PDF file: %s", file_path)
            elif file_ext in ['.txt', '.bin', '.md']:
               # テキストファイル（.txt, .bin, .md）も処理対象に
                text_files.append(file_path)
                logger.info("Found text file: %s", file_path)
    
    if pdf_files:
        logger.info("Found %d PDF files, processing...", len(pdf_files))
        for pdf_path in pdf_files:
            try:
                logger.info("Processing PDF: %s", pdf_path)
                
                # PDFファイルの存在確認
                if not os.path.exists(pdf_path):
                    logger.warning("PDF file not found, skipping: %s", pdf_path)
                    continue
                    
                # PDFの読み込み
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                logger.info("Loaded %d pages from PDF: %s", len(pages), pdf_path)
                
                if not pages:
                    logger.warning("PDF contains no pages: %s", pdf_path)
                    continue
                
                # PDFの各ページをドキュメントとして追加
                logger.info("Adding %d pages from PDF to RAG system...", len(pages))
                
                # PDFの内容をターミナルに表示
                logger.info("=" * 80)
                logger.info("PDF Content from: %s", pdf_path)
                logger.info("=" * 80)
                for i, page in enumerate(pages, 1):
                    if page.page_content:
                        logger.info("-" * 80)
                        logger.info("Page %d/%d (length: %d characters):", i, len(pages), len(page.page_content))
                        logger.info("-" * 80)
                        # ページの内容を表示（長い場合は最初の1000文字）
                        content = page.page_content
                        if len(content) > 1000:
                            logger.info("%s\n... (truncated, total: %d characters)", content[:1000], len(content))
                        else:
                            logger.info("%s", content)
                        logger.info("")
                
                rag_engine.add_documents(pages)
                total_chars = sum(len(page.page_content) for page in pages)
                logger.info("=" * 80)
                logger.info(
                    "Successfully added PDF content to RAG system: %s (%d pages, %d characters)",
                    pdf_path, len(pages), total_chars
                )
                logger.info("=" * 80)
            except Exception as e:
                logger.exception("Failed to process PDF %s: %s", pdf_path, e)
    else:
        logger.info("No PDF files found in %s", source_documents_path)
    
    # テキストファイルを処理
    if text_files:
        logger.info("Found %d text files, processing...", len(text_files))
        for text_path in text_files:
            try:
                logger.info("Processing text file: %s", text_path)
                # テキストファイルを読み込む（バイナリファイルの場合はスキップ）
                try:
                    with open(text_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    logger.warning("File appears to be binary, skipping: %s", text_path)
                    continue
                
                if content.strip():
                    # テキストファイルの内容をターミナルに表示
                    logger.info("=" * 80)
                    logger.info("Text File Content from: %s", text_path)
                    logger.info("=" * 80)
                    # 内容を表示（長い場合は最初の2000文字）
                    if len(content) > 2000:
                        logger.info("%s\n... (truncated, total: %d characters)", content[:2000], len(content))
                    else:
                        logger.info("%s", content)
                    logger.info("=" * 80)
                    
                    # メタデータにファイル情報を追加
                    from langchain.docstore.document import Document
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": text_path,
                            "file_type": "text",
                            "file_name": os.path.basename(text_path)
                        }
                    )
                    rag_engine.add_documents([doc])
                    logger.info("Successfully added text file content to RAG system (%d characters)", len(content))
                else:
                    logger.warning("Text file is empty: %s", text_path)
            except Exception as e:
                logger.exception("Failed to process text file %s: %s", text_path, e)
    else:
        logger.info("No text files found in %s", source_documents_path)

    # ベクトルストアの内容を確認
    if rag_engine and rag_engine.vectorstore:
        try:
            collection = rag_engine.vectorstore._collection
            if collection:
                count = collection.count()
                logger.info("=" * 80)
                logger.info("VECTOR STORE SUMMARY")
                logger.info("=" * 80)
                logger.info("Total documents in vector store: %d", count)
                
                # サンプルドキュメントを取得して内容を確認
                try:
                    # 最初の10件のドキュメントを取得
                    results = collection.get(limit=10)
                    if results and 'documents' in results:
                        logger.info("Sample documents in vector store:")
                        for i, doc in enumerate(results['documents'][:5], 1):
                            preview = doc[:200].replace('\n', ' ') if doc else "[empty]"
                            logger.info("  Doc %d: %s...", i, preview)
                except Exception as e:
                    logger.debug("Could not get sample documents: %s", e)
                logger.info("=" * 80)
        except Exception as e:
            logger.debug("Could not get vector store summary: %s", e)
    
    logger.info("RAG engine setup complete")

# リクエストボディの型定義
class QueryRequest(BaseModel):
    query: str

# レスポンスボディの型定義
class AnswerResponse(BaseModel):
    answer: str

@app.options("/ask")
async def ask_options():
    """
    /ask エンドポイントに対するCORSプリフライトリクエスト（OPTIONS）を処理
    """
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "3600",
        }
    )

@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """
    すべてのパスに対するCORSプリフライトリクエスト（OPTIONS）を処理
    """
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "3600",
        }
    )

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QueryRequest):
    """
    フロントエンドから質問を受け取り、RAGシステムで回答を生成して返すAPIエンドポイント
    """
    try:
        if rag_engine is None:
            logger.warning("Received query before RAG engine was ready")
            return AnswerResponse(answer="RAG engine is not ready yet.")
        
        logger.info("Received query: %s", request.query)
        
        # RAGエンジンを使って回答を生成
        result = rag_engine.query(request.query)
        
        # エラーチェック
        if isinstance(result, dict) and "error" in result:
            logger.warning("RAG engine returned error: %s", result["error"])
            response_text = result["error"]
        else:
            response_text = result.get("answer") if isinstance(result, dict) else str(result)
            
            # ソースドキュメントの情報をログに記録
            if isinstance(result, dict) and "sources" in result:
                sources = result["sources"]
                logger.info("Retrieved %d source documents for query", len(sources))
                if sources:
                    # ソースのメタデータを確認
                    for i, source in enumerate(sources[:2]):  # 上位2件を確認
                        if hasattr(source, 'metadata'):
                            logger.debug("Source %d metadata: %s", i + 1, source.metadata)
        
        # 回答が空の場合の処理
        if not response_text or not response_text.strip():
            logger.warning("Generated answer is empty")
            response_text = "申し訳ございませんが、回答を生成できませんでした。別の質問を試してください。"
        
        logger.info("Generated answer (length: %d, first 200 chars): %s", len(response_text), response_text[:200])
        
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content={"answer": response_text},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.exception("Error in /ask handler: %s", e)
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={"answer": f"エラーが発生しました: {str(e)}"},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    すべての例外をキャッチして、CORSヘッダーを含むエラーレスポンスを返す
    """
    error_details = traceback.format_exc()
    logger.exception("Global error handler triggered: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.get("/")
def read_root():
    return {"message": "RAG Backend is running"}