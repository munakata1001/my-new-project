from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from basic_rag_system import SimpleRAG  # 既存のクラスを利用
import os
import traceback

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
    print("Loading documents and setting up RAG engine...")

    # SimpleRAG を使った最小セットアップ（テキストを少量投入）
    rag_engine = SimpleRAG()

    # 簡易なシード文書（最低限ベクトルDBを初期化して動作確認できるように）
    seed_text = os.getenv(
        "RAG_SEED_TEXT",
        "これはRAGシステムの接続テスト用のサンプル文書です。GeminiベースのRAGが正しく応答できるかを検証するための短いテキストです。"
    )
    rag_engine.load_documents([seed_text])

    print("RAG engine setup complete.")

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
            return AnswerResponse(answer="RAG engine is not ready yet.")
        
        print(f"Received query: {request.query}")
        
        # RAGエンジンを使って回答を生成
        result = rag_engine.query(request.query)
        response_text = result.get("answer") if isinstance(result, dict) else str(result)
        
        print(f"Generated answer: {response_text}")
        
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
        print(f"Error in ask_question: {error_details}")
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
    print(f"Global error handler: {error_details}")
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