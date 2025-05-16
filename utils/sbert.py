from sentence_transformers import SentenceTransformer,util
import torch
import numpy as np
class SBERT_Embedding:
    def __init__(self, model_name="paraphrase-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, text:str):
        return self.model.encode(text, convert_to_tensor=True)
    
    def get_and_save_text_embedding(self, text:str, output_path:str):
        try:
            embedding = self.encode(text)
            torch.save(embedding, output_path)
            print(f"文本embedding已保存到: {output_path}")
            return True
        except Exception as e:
            print(f"处理文本时出错: {str(e)}")
            return False
    def _compute_single_similarity(self, embedding1, embedding2, metric="cosine"):
        """
        计算两个单一embedding之间的相似度
        
        Args:
            embedding1: 第一个embedding，numpy数组或torch张量
            embedding2: 第二个embedding，numpy数组或torch张量
            metric: 相似度度量方式
                
        Returns:
            float: 相似度分数
        """
        # 转换为numpy数组
        if isinstance(embedding1, torch.Tensor):
            embedding1 = embedding1.cpu().numpy()
        if isinstance(embedding2, torch.Tensor):
            embedding2 = embedding2.cpu().numpy()
        
        # 确保是1D数组
        if len(embedding1.shape) > 1 and embedding1.shape[0] == 1:
            embedding1 = embedding1.flatten()
        if len(embedding2.shape) > 1 and embedding2.shape[0] == 1:
            embedding2 = embedding2.flatten()
        
        # 计算相似度
        if metric == "cosine":
            # 计算余弦相似度
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            # 避免除以零
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    def eval_embedding_similarity(self, embedding1, embedding2, metric="cosine"):
        """
        计算两个BERT embedding之间的相似度
        
        Args:
            embedding1: 第一个embedding，可以是numpy数组、torch张量或它们的列表
            embedding2: 第二个embedding，可以是numpy数组、torch张量或它们的列表
            metric: 相似度度量方式，可选值为"cosine"(余弦相似度)、
                "euclidean"(欧氏距离)、"dot"(点积)
                
        Returns:
            float或numpy数组: 如果输入是单个embedding，返回float；
                            如果输入是embedding列表，返回相似度矩阵
        """
        # 检查是否为列表
        is_list1 = isinstance(embedding1, list)
        is_list2 = isinstance(embedding2, list)
        
        # 如果两个都不是列表，使用原来的逻辑
        if not is_list1 and not is_list2:
            return self._compute_single_similarity(embedding1, embedding2, metric)
        
        # 如果只有一个是列表，将另一个转换为单元素列表
        if is_list1 and not is_list2:
            embedding2 = [embedding2]
        elif not is_list1 and is_list2:
            embedding1 = [embedding1]
        
        # 计算相似度矩阵
        similarity_matrix = np.zeros((len(embedding1), len(embedding2)))
        for i, emb1 in enumerate(embedding1):
            for j, emb2 in enumerate(embedding2):
                similarity_matrix[i, j] = self._compute_single_similarity(emb1, emb2, metric)
        
        return similarity_matrix

