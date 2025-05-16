import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import os
import json
from PIL import Image
import pandas as pd

class CLIPEmbedding:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        初始化CLIP模型用于文本和图像嵌入
        
        Args:
            model_name: CLIP模型名称或路径
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # 加载模型和处理器
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # 设置为评估模式
        self.model.eval()
        
    def get_text_embeddings(self, texts, batch_size=32):
        """
        获取文本的嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            numpy数组，形状为 (len(texts), embedding_dim)
        """
        all_embeddings = []
        
        # 分批处理文本
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 处理文本输入
            with torch.no_grad():
                inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 获取文本特征
                text_features = self.model.get_text_features(**inputs)
                
                # 归一化特征
                text_embeddings = text_features / text_features.norm(dim=1, keepdim=True)
                
                # 转移到CPU并转换为numpy
                all_embeddings.append(text_embeddings.cpu().numpy())
        
        # 合并所有批次的嵌入
        return np.vstack(all_embeddings)
    
    def get_and_save_text_embedding(self, text, output_path):
        """
        将单个文本通过CLIP模型转换为embedding并保存
        
        Args:
            text: 文本内容
            output_path: 保存embedding的文件路径
            
        Returns:
            bool: 处理成功返回True，失败返回False
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 处理文本输入
            with torch.no_grad():
                inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 获取文本特征
                text_features = self.model.get_text_features(**inputs)
                
                # 归一化特征
                text_embedding = text_features / text_features.norm(dim=1, keepdim=True)
                
                # 保存embedding
                torch.save(text_embedding[0].cpu().float(), output_path)
                print(f"文本embedding已保存到: {output_path}")
                
                return True
        except Exception as e:
            print(f"处理文本时出错: {str(e)}")
            return False

    def get_and_save_image_embedding(self, image_path, output_path):
        """
        将单个图像通过CLIP模型转换为embedding并保存
        
        Args:
            image_path: 图像文件路径或PIL.Image对象
            output_path: 保存embedding的文件路径
            
        Returns:
            bool: 处理成功返回True，失败返回False
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 加载图像
            if isinstance(image_path, str):
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception as e:
                    print(f"加载图像时出错 {image_path}: {str(e)}")
                    return False
            else:
                # 假设已经是PIL.Image对象
                image = image_path
                
            # 处理图像输入
            with torch.no_grad():
                inputs = self.processor(images=[image], return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 获取图像特征
                image_features = self.model.get_image_features(**inputs)
                
                # 归一化特征
                image_embedding = image_features / image_features.norm(dim=1, keepdim=True)
                
                # 保存embedding
                torch.save(image_embedding[0].cpu(), output_path)
                print(f"图像embedding已保存到: {output_path}")
                
                return True
        except Exception as e:
            print(f"处理图像时出错: {str(e)}")
            return False
    
    def get_image_embeddings(self, images, batch_size=16):
        """
        获取图像的嵌入向量
        
        Args:
            images: 图像路径列表或PIL.Image对象列表
            batch_size: 批处理大小
            
        Returns:
            numpy数组，形状为 (len(images), embedding_dim)
        """
        all_embeddings = []
        
        # 分批处理图像
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            processed_images = []
            
            # 处理每个图像
            for img in batch_images:
                if isinstance(img, str):  # 如果是路径
                    try:
                        img = Image.open(img).convert('RGB')
                    except Exception as e:
                        print(f"Error loading image {img}: {e}")
                        continue
                processed_images.append(img)
            
            if not processed_images:
                continue
                
            # 处理图像输入
            with torch.no_grad():
                inputs = self.processor(images=processed_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 获取图像特征
                image_features = self.model.get_image_features(**inputs)
                
                # 归一化特征
                image_embeddings = image_features / image_features.norm(dim=1, keepdim=True)
                
                # 转移到CPU并转换为numpy
                all_embeddings.append(image_embeddings.cpu().numpy())
        
        if not all_embeddings:
            return np.array([])
            
        # 合并所有批次的嵌入
        return np.vstack(all_embeddings)
    
    def eval_similarity(self, text1, text2=None, images=None):
        """
        计算文本与文本或文本与图像的相似度
        
        Args:
            text1: 文本1或文本列表1
            text2: 文本2或文本列表2，如果与图像比较则为None
            images: 图像路径或PIL.Image对象列表，如果与文本比较则为None
            
        Returns:
            相似度分数(0-1之间)，如果输入是列表则返回相似度矩阵
        """
        # 处理输入为单个文本的情况
        if isinstance(text1, str):
            text1 = [text1]
        
        # 获取文本1的嵌入
        embeddings1 = self.get_text_embeddings(text1)
        
        # 文本-文本相似度
        if text2 is not None:
            if isinstance(text2, str):
                text2 = [text2]
            embeddings2 = self.get_text_embeddings(text2)
        # 文本-图像相似度
        elif images is not None:
            if isinstance(images, str) or isinstance(images, Image.Image):
                images = [images]
            embeddings2 = self.get_image_embeddings(images)
        else:
            raise ValueError("Either text2 or images must be provided")
        
        # 计算余弦相似度
        similarity_matrix = np.matmul(embeddings1, embeddings2.T)
        
        # 如果只有一对输入，返回单个相似度值
        if len(embeddings1) == 1 and len(embeddings2) == 1:
            return float(similarity_matrix[0, 0])
        
        return similarity_matrix

    def eval_embedding_similarity(self, embeddings1, embeddings2):
        """
        计算两组embeddings之间的相似度，支持embedding列表作为输入
        
        Args:
            embeddings1: 第一组embeddings，可以是:
                        - 单个numpy数组或torch张量
                        - numpy数组或torch张量的列表
            embeddings2: 第二组embeddings，格式同embeddings1
                
        Returns:
            相似度矩阵或单个相似度值
        """
        # 处理embeddings1列表
        if isinstance(embeddings1, (list, tuple)):
            # 检查列表中的元素类型
            if all(isinstance(e, torch.Tensor) for e in embeddings1):
                # 确保所有张量都是float类型并且归一化
                normalized_embeddings = []
                for emb in embeddings1:
                    emb = emb.float()
                    # 确保是2D张量
                    if len(emb.shape) == 1:
                        emb = emb.unsqueeze(0)
                    # 归一化
                    emb = emb / emb.norm(dim=1, keepdim=True)
                    normalized_embeddings.append(emb)
                
                # 合并所有embeddings
                embeddings1 = torch.cat(normalized_embeddings, dim=0)
            elif all(isinstance(e, np.ndarray) for e in embeddings1):
                # 确保所有数组都是2D的
                normalized_embeddings = []
                for emb in embeddings1:
                    if len(emb.shape) == 1:
                        emb = emb.reshape(1, -1)
                    # 归一化
                    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
                    normalized_embeddings.append(emb)
                
                # 合并所有embeddings
                embeddings1 = np.vstack(normalized_embeddings)
        
        # 处理embeddings2列表
        if isinstance(embeddings2, (list, tuple)):
            # 检查列表中的元素类型
            if all(isinstance(e, torch.Tensor) for e in embeddings2):
                # 确保所有张量都是float类型并且归一化
                normalized_embeddings = []
                for emb in embeddings2:
                    emb = emb.float()
                    # 确保是2D张量
                    if len(emb.shape) == 1:
                        emb = emb.unsqueeze(0)
                    # 归一化
                    emb = emb / emb.norm(dim=1, keepdim=True)
                    normalized_embeddings.append(emb)
                
                # 合并所有embeddings
                embeddings2 = torch.cat(normalized_embeddings, dim=0)
            elif all(isinstance(e, np.ndarray) for e in embeddings2):
                # 确保所有数组都是2D的
                normalized_embeddings = []
                for emb in embeddings2:
                    if len(emb.shape) == 1:
                        emb = emb.reshape(1, -1)
                    # 归一化
                    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
                    normalized_embeddings.append(emb)
                
                # 合并所有embeddings
                embeddings2 = np.vstack(normalized_embeddings)
        
        # 处理单个embedding (非列表)
        # 确保是torch张量并归一化
        if isinstance(embeddings1, torch.Tensor):
            embeddings1 = embeddings1.float()
            if len(embeddings1.shape) == 1:
                embeddings1 = embeddings1.unsqueeze(0)
            embeddings1 = embeddings1 / embeddings1.norm(dim=1, keepdim=True)
        
        if isinstance(embeddings2, torch.Tensor):
            embeddings2 = embeddings2.float()
            if len(embeddings2.shape) == 1:
                embeddings2 = embeddings2.unsqueeze(0)
            embeddings2 = embeddings2 / embeddings2.norm(dim=1, keepdim=True)
        
        # 转换为numpy数组
        if isinstance(embeddings1, torch.Tensor):
            embeddings1 = embeddings1.cpu().numpy()
        if isinstance(embeddings2, torch.Tensor):
            embeddings2 = embeddings2.cpu().numpy()
        
        # 确保numpy数组是2D的
        if len(embeddings1.shape) == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if len(embeddings2.shape) == 1:
            embeddings2 = embeddings2.reshape(1, -1)
        
        # 确保numpy数组是归一化的
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        # 计算余弦相似度
        similarity_matrix = np.matmul(embeddings1, embeddings2.T)
        
        # 如果只有一对输入，返回单个相似度值
        if embeddings1.shape[0] == 1 and embeddings2.shape[0] == 1:
            return float(similarity_matrix[0, 0])
        
        return similarity_matrix
    
    def find_best_matches(self, query, candidates, is_image_query=False, top_k=5):
        """
        找到与查询最匹配的候选项
        
        Args:
            query: 查询文本或图像路径
            candidates: 候选文本列表或图像路径列表
            is_image_query: 查询是否为图像
            top_k: 返回的最佳匹配数量
            
        Returns:
            包含(索引, 相似度, 候选项)的列表，按相似度降序排序
        """
        if is_image_query:
            # 图像查询，文本候选
            if isinstance(query, str):
                query = [query]  # 转换为列表以便处理
            query_embeddings = self.get_image_embeddings(query)
            candidate_embeddings = self.get_text_embeddings(candidates)
        else:
            # 文本查询，可能是文本或图像候选
            if isinstance(query, str):
                query = [query]
            query_embeddings = self.get_text_embeddings(query)
            
            # 检查候选项类型
            if all(isinstance(c, str) and (c.endswith('.jpg') or c.endswith('.png') or c.endswith('.jpeg')) for c in candidates):
                # 图像候选
                candidate_embeddings = self.get_image_embeddings(candidates)
            else:
                # 文本候选
                candidate_embeddings = self.get_text_embeddings(candidates)
        
        # 计算相似度
        similarity = np.matmul(query_embeddings, candidate_embeddings.T)[0]  # 取第一个查询的结果
        
        # 获取top_k个最佳匹配
        top_indices = similarity.argsort()[-top_k:][::-1]
        
        # 返回结果
        results = []
        for idx in top_indices:
            results.append((int(idx), float(similarity[idx]), candidates[idx]))
        
        return results

    def save_embeddings(self, items, embeddings, output_path, is_image=False):
        """
        保存文本/图像和对应的嵌入向量
        
        Args:
            items: 文本列表或图像路径列表
            embeddings: 嵌入向量数组
            output_path: 输出文件路径
            is_image: 是否为图像嵌入
        """
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存为numpy格式
        if output_path.endswith('.npy'):
            np.save(output_path, embeddings)
            # 同时保存项目信息
            items_path = output_path.replace('.npy', '_items.json')
            with open(items_path, 'w', encoding='utf-8') as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            print(f"Embeddings saved to {output_path}")
            print(f"Items saved to {items_path}")
            
        # 保存为CSV格式
        elif output_path.endswith('.csv'):
            # 创建DataFrame
            df = pd.DataFrame({
                'item': items,
                'is_image': is_image
            })
            
            # 添加嵌入向量列
            for i in range(embeddings.shape[1]):
                df[f'embedding_{i}'] = embeddings[:, i]
                
            df.to_csv(output_path, index=False)
            print(f"Embeddings and items saved to {output_path}")
            
        else:
            raise ValueError("Output path must end with .npy or .csv")