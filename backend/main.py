from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from basic_rag_system import SimpleRAG  # 既存のクラスを利用
import os

# FastAPIアプリケーションのインスタンスを作成
app = FastAPI()

# CORS (Cross-Origin Resource Sharing) の設定
# フロントエンド (localhost:3000) からのアクセスを許可するために必要
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
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

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QueryRequest):
    """
    フロントエンドから質問を受け取り、RAGシステムで回答を生成して返すAPIエンドポイント
    """
    if rag_engine is None:
        return AnswerResponse(answer="RAG engine is not ready yet.")
    
    print(f"Received query: {request.query}")
    
    # RAGエンジンを使って回答を生成
    result = rag_engine.query(request.query)
    response_text = result.get("answer") if isinstance(result, dict) else str(result)
    
    print(f"Generated answer: {response_text}")
    
    return AnswerResponse(answer=response_text)

@app.get("/")
def read_root():
    return {"message": "RAG Backend is running"}