import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter  # pyright: ignore[reportMissingImports]
from langchain_community.vectorstores import Chroma  # pyright: ignore[reportMissingImports]
from langchain_community.embeddings import HuggingFaceEmbeddings  # pyright: ignore[reportMissingImports]
from langchain.chains import RetrievalQA  # pyright: ignore[reportMissingImports]
from langchain_community.llms import OpenAI    # pyright: ignore[reportMissingImports]
from langchain.docstore.document import Document    # pyright: ignore[reportMissingImports]
load_dotenv()

class SimpleRAG:
    def __init__(self):
        """RAGシステムの初期化"""
        print("RAGシステムを初期化しています...")
        
        # Embeddingモデルの設定(無料・ローカル)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Vector Storeの設定
        self.vectorstore = None
        
        # LLMの設定
        self.llm = OpenAI(
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        print("初期化完了!")
    
    def load_documents(self, texts):
        """テキストドキュメントを読み込み、ベクトル化して保存"""
        print(f"\n{len(texts)}件のドキュメントを処理中...")
        
        # ドキュメントオブジェクトに変換
        documents = [Document(page_content=text) for text in texts]
        
        # テキストを分割(チャンク化)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)
        print(f"→ {len(splits)}個のチャンクに分割しました")
        
        # ベクトルDBに保存
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        print("ベクトルDBへの保存完了!")
    
    def query(self, question, k=3):
        """質問に対して回答を生成"""
        if self.vectorstore is None:
            return {"error": "ドキュメントが読み込まれていません"}
        
        print(f"\n質問: {question}")
        print("関連文書を検索中...")
        
        # RetrievalQAチェーンの作成
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            ),
            return_source_documents=True
        )
        
        # 質問を実行
        result = qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": result["source_documents"]
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
        self.llm = OpenAI(temperature=0.3)
        
    def load_pdf(self, pdf_path):
        """PDFファイルを読み込み、ベクトル化"""
        print(f"PDFを読み込み中: {pdf_path}")
        
        # PDFの読み込み
        loader = PyPDFLoader(pdf_path)  # pyright: ignore[reportUndefinedVariable]
        pages = loader.load()
        print(f"→ {len(pages)}ページ読み込みました")
        
        # テキスト分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "、", " ", ""]
        )
        splits = text_splitter.split_documents(pages)
        print(f"→ {len(splits)}個のチャンクに分割")
        
        # ベクトルDB作成
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("ベクトル化完了!")
        
    def query(self, question, k=4):
        """質問応答"""
        if not self.vectorstore:
            return {"error": "PDFが読み込まれていません"}
        
        # カスタムプロンプト
        template = """以下の文脈を使用して、質問に日本語で答えてください。
        
        文脈:
        {context}
        
        質問: {question}
        
        回答:"""
        
        prompt = PromptTemplate(  # pyright: ignore[reportUndefinedVariable]
            template=template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        result = qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }

        