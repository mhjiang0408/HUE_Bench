import os
import sys
sys.path.append(os.getcwd())

import torch
import argparse
import logging
import csv
import numpy as np
from tqdm import tqdm
from utils.clip import CLIPEmbedding
from utils.open_clip import OpenClip
# from utils.siglip import SigLIP2Encoder
# from utils.close_source import QwenEmbedding

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("image_embedding_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("image_embedding_extractor")

def process_image_files(base_path, output_base_path=None, model_name="google/siglip2-so400m-patch16-naflex", threshold=0.98, dry_run=True):
    """
    处理base_path下所有journal文件夹中的txt文件，提取文本embedding并保存
    
    Args:
        base_path: 基础路径，包含journal文件夹
        output_base_path: 输出基础路径，如果为None则使用base_path + "_embeddings"
        model_name: CLIP模型名称
        threshold: 相似度阈值，大于此值的文件将被标记为重复
        dry_run: 是否为试运行模式，True表示不实际删除文件
    """
    # 如果没有指定输出路径，则在原路径后添加"_embeddings"
    if output_base_path is None:
        output_base_path = os.path.join(base_path, "Cover_Embeddings")
    
    logger.info(f"开始处理文本文件，基础路径: {base_path}")
    logger.info(f"输出路径: {output_base_path}")
    logger.info(f"相似度阈值: {threshold}")
    logger.info(f"试运行模式: {dry_run}")
    
    # 初始化CLIP模型
    clip_model = CLIPEmbedding(model_name="openai/clip-vit-large-patch14-336")
    # clip_model = OpenClip(model_name="TULIP-so400m-14-384",pretrained="laion2b_s34b_b88k")
    # clip_model = SigLIP2Encoder(model_name="google/siglip2-so400m-patch16-naflex")
    # clip_model = QwenEmbedding()
    logger.info(f"CLIP模型初始化完成: {model_name}")
    
    # 创建CSV文件用于记录要删除的文件
    csv_path = os.path.join(os.path.dirname(base_path), "duplicate_files.csv")
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Journal', 'Directory', 'File', 'Similar_To', 'Similarity', 'Deleted'])
    
    # 统计数据
    total_files = 0
    processed_files = 0
    similar_pairs_count = 0
    deleted_files_count = 0
    avg_similarities = []
    
    # 设置story文件夹
    story_path = base_path
    if not os.path.exists(story_path):
        logger.error(f"错误: Cover文件夹不存在于 {base_path}")
        return {}
    
    # 遍历base_path下的所有目录
    for journal_name in os.listdir(story_path):
        journal_path = os.path.join(story_path, journal_name)
        if not os.path.isdir(journal_path):
            continue
        
        logger.info(f"处理期刊: {journal_name}")
        
        # 为每个期刊创建对应的输出目录
        journal_output_dir = os.path.join(output_base_path, journal_name)
        os.makedirs(journal_output_dir, exist_ok=True)
        
        for root, dirs, files in os.walk(journal_path):
            # 只处理图像文件
            image_files = [f for f in files if f.endswith(('.pt'))]
        
            if not image_files:
                continue
            
            # 创建对应的输出目录
            rel_path = os.path.relpath(root, journal_path)
            if rel_path != '.':
                output_dir = os.path.join(journal_output_dir, rel_path)
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = journal_output_dir
            
            logger.info(f"处理目录: {rel_path} (找到 {len(image_files)} 个图像文件)")
            
            # 获取该目录下所有图像的嵌入向量
            embeddings = {}
            for img_file in image_files:
                file_path = os.path.join(root, img_file)
                try:
                    embedding = torch.load(file_path)
                    # embedding = clip_model.get_image_embeddings([file_path])
                    if embedding is not None and len(embedding) > 0:
                        embeddings[file_path] = embedding
                    
                except Exception as e:
                    logger.error(f"提取嵌入向量时出错 {file_path}: {str(e)}")
            
            total_files += len(embeddings)
            processed_files += len(embeddings)
            
            # 如果该目录下有超过1个图像，计算相似度
            if len(embeddings) > 1:
                # 计算文件之间的相似度矩阵
                paths = list(embeddings.keys())
                vectors = np.array([embeddings[p] for p in paths])
                tensors = [embeddings[p] for p in paths]
                
                # 计算余弦相似度
                similarity_matrix = np.zeros((len(paths), len(paths)))
                for i in range(len(paths)):
                    for j in range(i+1, len(paths)):
                        # similarity = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
                        similarity = clip_model.eval_embedding_similarity(tensors[i], tensors[j])
                        similarity_matrix[i][j] = similarity
                        similarity_matrix[j][i] = similarity
                
                # 删除对角线上的值(自身与自身的相似度)
                np.fill_diagonal(similarity_matrix, 0)
                
                # 计算并记录平均相似度
                avg_similarity = similarity_matrix.sum() / (len(paths) * (len(paths) - 1))
                avg_similarities.append(avg_similarity)
                logger.info(f"目录 {journal_name} 的平均余弦相似度: {avg_similarity:.4f}")
                if avg_similarity < 0.5 :
                    logger.info(f"目录 {journal_name} 的平均余弦相似度小于0.5，跳过")
                
                # 查找高相似度对并删除文件
                to_delete = set()  # 用集合避免重复删除
                for i in range(len(paths)):
                    for j in range(i+1, len(paths)):
                        similarity = similarity_matrix[i][j]
                        if similarity > threshold:
                            similar_pairs_count += 1
                            # 选择保留较早创建的文件
                            file1_time = os.path.getmtime(paths[i])
                            file2_time = os.path.getmtime(paths[j])
                            
                            if file1_time > file2_time:
                                to_delete.add(paths[i])
                                kept_file = paths[j]
                                deleted_file = paths[i]
                            else:
                                to_delete.add(paths[j])
                                kept_file = paths[i]
                                deleted_file = paths[j]
                            
                            # 记录到CSV
                            csv_writer.writerow([
                                journal_name,
                                rel_path,
                                os.path.basename(deleted_file),
                                os.path.basename(kept_file),
                                f"{similarity:.4f}",
                                "No" if dry_run else "Yes"
                            ])
                            
                            # logger.info(f"发现高相似度对 ({similarity:.4f}): {os.path.basename(paths[i])} 和 {os.path.basename(paths[j])}")
                
                # 执行删除操作(如果不是dry_run模式)
                if not dry_run:
                    for file_path in to_delete:
                        try:
                            os.remove(file_path)
                            deleted_files_count += 1
                            logger.info(f"已删除: {file_path}")
                            # 删除对应的图片文件
                            img_path = file_path.replace('.pt', '.jpg').replace('Dataset/Comics_Embeddings', 'gocomics_downloads')
                            if os.path.exists(img_path):
                                os.remove(img_path)
                                logger.info(f"已删除: {img_path}")
                        except Exception as e:
                            logger.error(f"删除文件时出错 {file_path}: {str(e)}")
                else:
                    deleted_files_count += len(to_delete)
                    logger.info(f"[试运行模式] 将删除 {len(to_delete)} 个文件")
    
    # 关闭CSV文件
    csv_file.close()
    
    # 计算总体平均相似度
    overall_avg_similarity = sum(avg_similarities) / len(avg_similarities) if avg_similarities else 0
    
    # 输出统计信息
    logger.info(f"处理完成!")
    logger.info(f"总文件数: {total_files}")
    logger.info(f"处理成功: {processed_files}")
    logger.info(f"发现相似对: {similar_pairs_count}")
    logger.info(f"{'将要删除' if dry_run else '已删除'}的文件数: {deleted_files_count}")
    logger.info(f"总体平均余弦相似度: {overall_avg_similarity:.4f}")
    logger.info(f"相似文件记录已保存至: {csv_path}")
    
    # 返回处理结果
    return {
        "total_files": total_files,
        "processed_files": processed_files,
        "similar_pairs": similar_pairs_count,
        "deleted_files": deleted_files_count,
        "avg_similarity": overall_avg_similarity
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从图像文件提取embedding并删除高度相似的图像")
    parser.add_argument("--base_path", help="包含journal文件夹的基础路径")
    parser.add_argument("--output", help="输出基础路径，默认为base_path + '_embeddings'")
    parser.add_argument("--threshold", type=float, default=0.98, help="相似度阈值，默认为0.98")
    parser.add_argument("--no-dry-run", action="store_true", help="设置此参数表示实际删除文件，否则只记录不删除")
    # parser.add_argument("--model", default="openai/clip-vit-base-patch32", help="CLIP模型名称")
    
    args = parser.parse_args()
    
    try:
        process_image_files(
            args.base_path, 
            args.output, 
            threshold=args.threshold, 
            dry_run=not args.no_dry_run
        )
    except Exception as e:
        logger.error(f"脚本执行出错: {str(e)}")