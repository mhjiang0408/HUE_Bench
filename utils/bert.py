import torch
from transformers import AutoTokenizer, AutoModel
import os
import numpy as np

class BERTEmbedding:
    def __init__(self, model_name="bert-base-uncased"):
        """
        初始化BERT模型用于文本嵌入
        
        Args:
            model_name: BERT模型名称或路径，默认为"bert-base-uncased"
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # 设置为评估模式
        self.model.eval()
    
    def get_text_embedding(self, text, pooling_strategy="cls"):
        """
        获取单个文本的嵌入向量
        
        Args:
            text: 文本内容
            pooling_strategy: 池化策略，可选值为"cls"(使用[CLS]标记的表示)、
                             "mean"(平均所有token的表示)或"max"(取每个维度的最大值)
            
        Returns:
            numpy数组，形状为 (embedding_dim,)
        """
        # 对文本进行编码
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 获取模型输出
            outputs = self.model(**inputs)
            
            # 获取最后一层的隐藏状态
            last_hidden_state = outputs.last_hidden_state
            
            # 根据池化策略提取embedding
            if pooling_strategy == "cls":
                # 使用[CLS]标记的表示作为整个序列的表示
                embedding = last_hidden_state[:, 0, :]
            elif pooling_strategy == "mean":
                # 计算所有token表示的平均值
                # 创建attention mask，排除padding tokens
                attention_mask = inputs["attention_mask"]
                embedding = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
            elif pooling_strategy == "max":
                # 取每个维度的最大值
                attention_mask = inputs["attention_mask"]
                # 将padding tokens的表示设为很小的值，以便在max操作中被忽略
                masked_hidden = last_hidden_state * attention_mask.unsqueeze(-1) - (1 - attention_mask).unsqueeze(-1) * 1e10
                embedding = torch.max(masked_hidden, dim=1)[0]
            else:
                raise ValueError(f"不支持的池化策略: {pooling_strategy}")
            
            # 转换为numpy数组
            embedding = embedding.cpu().numpy()[0]
            
            return embedding
    
    def get_and_save_text_embedding(self, text, output_path, pooling_strategy="cls"):
        """
        获取文本的嵌入向量并保存到文件
        
        Args:
            text: 文本内容
            output_path: 输出文件路径
            pooling_strategy: 池化策略
            
        Returns:
            bool: 处理成功返回True，失败返回False
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 获取embedding
            embedding = self.get_text_embedding(text, pooling_strategy)
            
            # 转换为torch张量并保存
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
            torch.save(embedding_tensor, output_path)
            print(f"文本embedding已保存到: {output_path}")
            
            return True
        except Exception as e:
            print(f"处理文本时出错: {str(e)}")
            return False
    
    def get_text_embeddings(self, texts, batch_size=32, pooling_strategy="cls"):
        """
        获取多个文本的嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            pooling_strategy: 池化策略
            
        Returns:
            numpy数组，形状为 (len(texts), embedding_dim)
        """
        all_embeddings = []
        
        # 分批处理文本
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 对文本进行编码
            with torch.no_grad():
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 获取模型输出
                outputs = self.model(**inputs)
                
                # 获取最后一层的隐藏状态
                last_hidden_state = outputs.last_hidden_state
                
                # 根据池化策略提取embedding
                if pooling_strategy == "cls":
                    # 使用[CLS]标记的表示作为整个序列的表示
                    embeddings = last_hidden_state[:, 0, :]
                elif pooling_strategy == "mean":
                    # 计算所有token表示的平均值
                    attention_mask = inputs["attention_mask"]
                    embeddings = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
                elif pooling_strategy == "max":
                    # 取每个维度的最大值
                    attention_mask = inputs["attention_mask"]
                    masked_hidden = last_hidden_state * attention_mask.unsqueeze(-1) - (1 - attention_mask).unsqueeze(-1) * 1e10
                    embeddings = torch.max(masked_hidden, dim=1)[0]
                else:
                    raise ValueError(f"不支持的池化策略: {pooling_strategy}")
                
                # 转换为numpy数组
                batch_embeddings = embeddings.cpu().numpy()
                all_embeddings.append(batch_embeddings)
        
        # 合并所有批次的嵌入
        return np.vstack(all_embeddings) if all_embeddings else np.array([])
    
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
        
        elif metric == "euclidean":
            # 计算欧氏距离，并转换为相似度（距离越小，相似度越高）
            distance = np.linalg.norm(embedding1 - embedding2)
            # 使用高斯核将距离转换为相似度
            return np.exp(-distance)
        
        elif metric == "dot":
            # 计算点积
            return np.dot(embedding1, embedding2)
        
        else:
            raise ValueError(f"不支持的相似度度量方式: {metric}")
