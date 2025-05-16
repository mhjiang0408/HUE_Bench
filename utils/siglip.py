import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import requests
from typing import Union, List
import numpy as np
from transformers.image_utils import load_image

class SigLIP2Encoder:
    def __init__(self, model_name: str = "google/siglip2-so400m-patch16-naflex"):
        """
        初始化SigLIP2编码器
        
        Args:
            model_name: 模型名称，默认使用base版本
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        
    def save_image_embeddings(self, images: Union[str, List[str], Image.Image, List[Image.Image]], save_path: str) -> np.ndarray:
        """
        编码图像
        
        Args:
            images: 可以是图像路径、PIL图像对象或它们的列表
            
        Returns:
            图像特征向量
        """
        # 转换输入为列表格式
        try:
            if not isinstance(images, list):
                images = [images]
                
            # 加载图像
            pil_images = []
            for image in images:
                pil_images.append(load_image(image))
                    
            # 处理图像
            inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
            
            # 获取图像特征
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs).cpu()
                
            # 归一化特征
            torch.save(image_features.cpu(), save_path)
            return True
        except Exception as e:
            print(f"Error saving image embeddings: {e}")
            return False
    
    def save_text_embeddings(self, texts: Union[str, List[str]], save_path: str) -> np.ndarray:
        """
        编码文本
        
        Args:
            texts: 文本或文本列表
            
        Returns:
            文本特征向量
        """
        # 转换输入为列表格式
        try:
            if isinstance(texts, str):
                texts = [texts]
                
            # 处理文本
            inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            
            # 获取文本特征
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            torch.save(text_features.cpu(), save_path)
            return True
        except Exception as e:
            print(f"Error saving text embeddings: {e}")
            return False
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # return text_features.cpu().numpy()
    
    def eval_embedding_similarity(self, embeddings1, embeddings2):
        """
        加载已保存的embeddings并评估两个embedding列表之间的相似度
        
        Args:
            image_embeddings_paths: 图像embedding文件路径列表，与image_embeddings二选一
            text_embeddings_paths: 文本embedding文件路径列表，与text_embeddings二选一
            image_embeddings: 已加载的图像embedding列表
            text_embeddings: 已加载的文本embedding列表
            normalize: 是否对embeddings进行归一化
            return_probs: 是否返回概率分布（使用softmax）而不是原始相似度
            
        Returns:
            numpy.ndarray: 相似度矩阵，形状为[num_images, num_texts]
            如果return_probs=True，则返回概率分布
        """
        # # 加载图像embeddings
        # if image_embeddings is None and image_embeddings_paths is not None:
        #     image_embeddings = []
        #     for path in image_embeddings_paths:
        #         try:
        #             emb = torch.load(path)
        #             image_embeddings.append(emb)
        #         except Exception as e:
        #             print(f"Error loading image embedding from {path}: {e}")
        #             continue
            
        #     if not image_embeddings:
        #         raise ValueError("No valid image embeddings could be loaded")
            
        #     # 将列表转换为张量
        #     image_embeddings = torch.cat(image_embeddings, dim=0)
        
        # # 加载文本embeddings
        # if text_embeddings is None and text_embeddings_paths is not None:
        #     text_embeddings = []
        #     for path in text_embeddings_paths:
        #         try:
        #             emb = torch.load(path)
        #             text_embeddings.append(emb)
        #         except Exception as e:
        #             print(f"Error loading text embedding from {path}: {e}")
        #             continue
            
        #     if not text_embeddings:
        #         raise ValueError("No valid text embeddings could be loaded")
            
        #     # 将列表转换为张量
        #     text_embeddings = torch.cat(text_embeddings, dim=0)
        
        # 确保embeddings已加载
        if embeddings1 is None or embeddings2 is None:
            raise ValueError("Either provide embedding paths or pre-loaded embeddings")
        if isinstance(embeddings1, list):
            embeddings1 = torch.cat(embeddings1, dim=0)
        if isinstance(embeddings2, list):
            embeddings2 = torch.cat(embeddings2, dim=0)
        
        # 确保是torch张量
        if not isinstance(embeddings1, torch.Tensor):
            embeddings1 = torch.tensor(embeddings1)
        if not isinstance(embeddings2, torch.Tensor):
            embeddings2 = torch.tensor(embeddings2)
        
        # 归一化embeddings
        embeddings1 = embeddings1 / embeddings1.norm(dim=-1, keepdim=True)
        embeddings2 = embeddings2 / embeddings2.norm(dim=-1, keepdim=True)
        
        # 计算相似度
        with torch.no_grad():
            similarity = (embeddings1 @ embeddings2.T).cpu()
            
            # # 如果需要返回概率分布  
            # if return_probs:
            #     similarity = (100.0 * similarity).softmax(dim=-1).numpy()
            # else:
            similarity = similarity.numpy()
        
        return similarity