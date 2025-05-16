import os
import sys
sys.path.append(os.getcwd())


import argparse
import logging
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

def process_image_files(base_path, output_base_path=None, model_name="google/siglip2-so400m-patch16-naflex"):
    """
    处理base_path下所有journal文件夹中的txt文件，提取文本embedding并保存
    
    Args:
        base_path: 基础路径，包含journal文件夹
        output_base_path: 输出基础路径，如果为None则使用base_path + "_embeddings"
        model_name: CLIP模型名称
    """
    # 如果没有指定输出路径，则在原路径后添加"_embeddings"
    if output_base_path is None:
        output_base_path = os.path.join(base_path, "Cover_Embeddings")
    
    logger.info(f"开始处理文本文件，基础路径: {base_path}")
    logger.info(f"输出路径: {output_base_path}")
    
    # 初始化CLIP模型
    clip_model = CLIPEmbedding(model_name="openai/clip-vit-large-patch14-336")
    # clip_model = OpenClip(model_name="TULIP-so400m-14-384",pretrained="laion2b_s34b_b88k")
    # clip_model = SigLIP2Encoder(model_name="google/siglip2-so400m-patch16-naflex")
    # clip_model = QwenEmbedding()
    logger.info(f"CLIP模型初始化完成: {model_name}")
    
    # 统计数据
    total_files = 0
    processed_files = 0
    failed_files = 0
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
            # 只处理txt文件
            txt_files = [f for f in files if f.endswith('.jpg')]
        
            if not txt_files:
                continue
            
            # 创建对应的输出目录
            rel_path = os.path.relpath(root, journal_path)
            if rel_path != '.':
                output_dir = os.path.join(journal_output_dir, rel_path)
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = journal_output_dir
            
            logger.info(f"处理目录: {rel_path} (找到 {len(txt_files)} 个txt文件)")
            
            # 处理当前目录下的所有txt文件
            for txt_file in tqdm(txt_files, desc=f"处理 {rel_path} 中的文件"):
                total_files += 1
                
                try:
                    # 构建文件路径
                    file_path = os.path.join(root, txt_file)
                    

                    
                    # 构建输出文件路径 (将.txt替换为.pt)
                    output_file = os.path.join(output_dir, txt_file.replace('.jpg', '.pt'))
                    
                    # 如果输出文件已存在，则跳过
                    if os.path.exists(output_file):
                        logger.info(f"文件已存在，跳过: {output_file}")
                        processed_files += 1
                        continue
                    
                    # 提取并保存embedding
                    success = clip_model.get_and_save_image_embedding(file_path, output_file)
                    
                    if success:
                        processed_files += 1
                        logger.debug(f"成功处理: {file_path} -> {output_file}")
                    else:
                        failed_files += 1
                        logger.error(f"处理失败: {file_path}")
                    
                except Exception as e:
                    failed_files += 1
                    logger.error(f"处理文件时出错 {txt_file}: {str(e)}")
    
    # 输出统计信息
    logger.info(f"处理完成! 总文件数: {total_files}, 成功: {processed_files}, 失败: {failed_files}")
    return processed_files, failed_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从txt文件提取文本embedding")
    parser.add_argument("--base_path", help="包含journal文件夹的基础路径")
    parser.add_argument("--output", help="输出基础路径，默认为base_path + '_embeddings'")
    # parser.add_argument("--model", default="openai/clip-vit-base-patch32", help="CLIP模型名称")
    
    args = parser.parse_args()
    
    try:
        process_image_files(args.base_path, args.output)
    except Exception as e:
        logger.error(f"脚本执行出错: {str(e)}")