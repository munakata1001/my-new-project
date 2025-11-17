"""画像ファイルを処理してRAGシステムで利用できるテキストに変換するモジュール"""
import os
import logging
from pathlib import Path
import google.generativeai as genai
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

def extract_text_from_image(image_path: str) -> str:
    """
    画像ファイルからテキストを抽出（OCR + 画像説明）
    
    Args:
        image_path: 画像ファイルのパス
        
    Returns:
        画像から抽出されたテキスト
    """
    try:
        logger.info("Processing image: %s", image_path)
        
        # Google Gemini APIを設定
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        genai.configure(api_key=api_key)
        
        # Gemini 2.5 Flashモデルを使用（画像理解対応）
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # 画像を読み込む
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        import PIL.Image
        image = PIL.Image.open(image_path)
        
        # 画像の内容を説明してもらう（OCR + 構造理解）
        prompt = """この画像の内容を詳しく説明してください。
以下の点を含めてください：
1. 画像に含まれるテキスト（メニュー、価格、商品名など）
2. 画像の構造やレイアウト
3. 重要な情報や特徴

日本語で回答してください。"""
        
        response = model.generate_content([prompt, image])
        extracted_text = response.text
        
        logger.info("Extracted %d characters from image", len(extracted_text))
        logger.debug("Extracted text preview: %s", extracted_text[:200])
        
        return extracted_text
        
    except Exception as e:
        logger.exception("Error extracting text from image: %s", e)
        raise

def process_image_file(image_path: str) -> Document:
    """
    画像ファイルを処理してDocumentオブジェクトに変換
    
    Args:
        image_path: 画像ファイルのパス
        
    Returns:
        画像から抽出されたテキストを含むDocumentオブジェクト
    """
    extracted_text = extract_text_from_image(image_path)
    
    # メタデータに画像ファイル情報を追加
    metadata = {
        "source": image_path,
        "file_type": "image",
        "file_name": os.path.basename(image_path)
    }
    
    return Document(page_content=extracted_text, metadata=metadata)

def find_images_in_directory(directory: str) -> list[str]:
    """
    ディレクトリ内の画像ファイルを検索
    
    Args:
        directory: 検索するディレクトリのパス
        
    Returns:
        見つかった画像ファイルのパスのリスト
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    logger.info("Found %d image files in %s", len(image_files), directory)
    return image_files

