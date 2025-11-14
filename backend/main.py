from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from basic_rag_system import SimpleRAG  # 既存のクラスを利用
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
        response_text = result.get("answer") if isinstance(result, dict) else str(result)
        
        logger.info("Generated answer (first 200 chars): %s", response_text[:200])
        
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