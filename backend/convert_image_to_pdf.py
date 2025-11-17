"""画像ファイルをPDFに変換するスクリプト"""
import os
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_image_to_pdf(image_path: str, output_path: str = None) -> str:
    """
    画像ファイルをPDFに変換
    
    Args:
        image_path: 変換する画像ファイルのパス
        output_path: 出力PDFファイルのパス（指定しない場合は画像と同じディレクトリに保存）
        
    Returns:
        生成されたPDFファイルのパス
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        logger.info("Converting image to PDF: %s", image_path)
        
        # 画像を開く
        image = Image.open(image_path)
        
        # RGBモードに変換（PDF変換のため）
        if image.mode != 'RGB':
            logger.info("Converting image from %s to RGB mode", image.mode)
            image = image.convert('RGB')
        
        # 出力パスを決定
        if output_path is None:
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}.pdf"
        
        # PDFとして保存
        image.save(output_path, "PDF", resolution=100.0)
        logger.info("PDF saved to: %s", output_path)
        
        return output_path
        
    except Exception as e:
        logger.exception("Error converting image to PDF: %s", e)
        raise

if __name__ == "__main__":
    # menu1205.jpgをPDFに変換
    image_path = "./chroma_db/34d232a2-f223-4a94-84a4-78cd4721d5e7/menu1205.jpg"
    output_path = "./chroma_db/34d232a2-f223-4a94-84a4-78cd4721d5e7/menu1205.pdf"
    
    try:
        pdf_path = convert_image_to_pdf(image_path, output_path)
        print(f"Successfully converted to PDF: {pdf_path}")
    except Exception as e:
        print(f"Error: {e}")

