import base64
import os
from typing import Optional

def encode_image_to_base64(image_path: str) -> Optional[str]:
        """将图像文件编码为 Base64 字符串。"""
        if not image_path or not os.path.exists(image_path):
            print(f"错误：图像文件不存在 {image_path}")
            return None
        
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                # 根据扩展名确定 MIME 类型
                ext = os.path.splitext(image_path)[1].lower()
                mime_type = f"image/{ext[1:]}" if ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp'] else "image/jpeg" # 默认为 jpeg
                return f"data:{mime_type};base64,{encoded_string}"
        except FileNotFoundError:
            print(f"错误：编码图像时未找到文件 {image_path}")
            return None
        except Exception as e:
            print(f"错误：编码图像 {image_path} 时出错: {e}")
            return None