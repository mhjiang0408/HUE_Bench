import cv2
import easyocr
import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import argparse
import logging
from tqdm import tqdm
# from utils.clip import CLIPEmbedding

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

def cover_text_in_image(image_path: str, tokenizer,conf_threshold: float = 0.25,reader:easyocr.Reader=easyocr.Reader(['en']), ) -> None:
    """
    Automatically detect text in images and count the number of tokens.
    
    Args:
        image_path: Path to the input image
        conf_threshold: Confidence threshold for text detection (results below this value will be ignored)
        reader: EasyOCR reader object
        tokenizer_name: Name of the tokenizer to use for token counting (default: "gpt2")
    
    Returns:
        Tuple of (list of detected texts, total token count)
        
    Note:
        Make sure you have installed easyocr, OpenCV, and transformers.
        If not installed, you can install them with:
        pip install easyocr opencv-python transformers
    """
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File does not exist: {image_path}")
        return False
    
    # Initialize tokenizer
    # try:
        
    #     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # except Exception as e:
    #     print(f"Failed to initialize tokenizer: {str(e)}")
    #     print("Falling back to character counting...")
    #     tokenizer = None
    
    # Read the image
    try:
        image = cv2.imread(image_path)
        if image is None:
            # Try using PIL as an alternative
            # print("Attempting to read image using PIL...")
            try:
                from PIL import Image
                import numpy as np
                pil_image = Image.open(image_path)
                image = np.array(pil_image.convert('RGB'))
                image = image[:, :, ::-1].copy()  # RGB to BGR
            except Exception as e:
                print(f"PIL image reading also failed: {str(e)}")
                return False
    except Exception as e:
        print(f"Exception occurred while reading image: {str(e)}")
        return False
    
    # Check image dimensions
    if image is None:
        print("Image reading failed, returned None")
        return False
    
    # Use EasyOCR to detect text
    try:
        results = reader.readtext(image)
    except Exception as e:
        print(f"Text detection failed: {str(e)}")
        return False
    
    # Store all recognized texts
    all_texts = []
    total_token_count = 0
    
    for (bbox, text, prob) in results:
        # Filter low confidence results
        if prob > conf_threshold:
            # Extract text content
            all_texts.append(text)
            
            # Count tokens 
            if tokenizer:
                # Use tokenizer to count tokens
                tokens = tokenizer.encode(text)
                token_count = len(tokens)
            else:
                # Fallback to character counting if tokenizer is not available
                token_count = len(text.strip())
            
            total_token_count += token_count
    
    # Output statistics
    # print(f"Number of texts detected: {len(all_texts)}")
    # print(f"Total token count: {total_token_count}")
    # print("All detected texts:")
    # for i, text in enumerate(all_texts):
    #     if tokenizer:
    #         tokens = tokenizer.encode(text)
    #         print(f"[{i+1}] {text} ({len(tokens)} tokens)")
    #     else:
    #         print(f"[{i+1}] {text}")
    
    # Return text statistics
    return all_texts, total_token_count

def process_image_files(base_path, output_base_path=None, model_name="openai/clip-vit-base-patch32"):
    """
    处理base_path下所有journal文件夹中的txt文件，提取文本embedding并保存
    
    Args:
        base_path: 基础路径，包含journal文件夹
        output_base_path: 输出基础路径，如果为None则使用base_path + "_embeddings"
        model_name: CLIP模型名称
    """
    # 如果没有指定输出路径，则在原路径后添加"_embeddings"
    if output_base_path is None:
        output_base_path = os.path.join(base_path, "OCRed_Cover")
    
    logger.info(f"开始处理文本文件，基础路径: {base_path}")
    logger.info(f"输出路径: {output_base_path}")
    
    # 初始化CLIP模型
    # clip_model = OpenClip(model_name="ViT-g-14",pretrained="laion2b_s34b_b88k")
    # logger.info(f"CLIP模型初始化完成: {model_name}")
    
    # 统计数据
    total_files = 0
    processed_files = 0
    failed_files = 0
    # 设置story文件夹
    story_path = os.path.join(base_path, "Cover")
    if not os.path.exists(story_path):
        logger.error(f"错误: Cover文件夹不存在于 {base_path}")
        return {}
    # 遍历base_path下的所有目录
    for journal_name in tqdm(os.listdir(story_path)):
        journal_path = os.path.join(story_path, journal_name)
        if not os.path.isdir(journal_path):
            continue
        logger.info(f"处理期刊: {journal_name}")
        # 为每个期刊创建对应的输出目录
        journal_output_dir = os.path.join(output_base_path, journal_name)
        os.makedirs(journal_output_dir, exist_ok=True)
        for root, dirs, files in os.walk(journal_path):
            # 只处理txt文件
            txt_files = [f for f in files if f.endswith('.png')]
        
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
            for txt_file in txt_files:
                total_files += 1
                
                try:
                    # 构建文件路径
                    file_path = os.path.join(root, txt_file)
                    

                    
                    # 构建输出文件路径 (将.txt替换为.pt)
                    output_file = os.path.join(output_dir, txt_file)
                    
                    # 如果输出文件已存在，则跳过
                    if os.path.exists(output_file):
                        logger.info(f"文件已存在，跳过: {output_file}")
                        processed_files += 1
                        continue
                    
                    # 提取并保存embedding
                    success = cover_text_in_image(file_path, output_file,0.25)
                    
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

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="从txt文件提取文本embedding")
#     parser.add_argument("--base_path", help="包含journal文件夹的基础路径")
#     parser.add_argument("--output", help="输出基础路径，默认为base_path + '_embeddings'")
#     # parser.add_argument("--model", default="openai/clip-vit-base-patch32", help="CLIP模型名称")
    
#     args = parser.parse_args()
    
#     try:
#         process_image_files(args.base_path, args.output)
#     except Exception as e:
#         logger.error(f"脚本执行出错: {str(e)}")


# # 使用示例，手动调用该函数：
# if __name__ == "__main__":

#     output_img = "./Dataset/test/2021_1.png"
#     cover_text_in_image(input_img)
    # print(f"处理后的图片已保存到: {output_img}")
#!/usr/bin/env python
# -*- coding: utf-8 -*-



def get_ground_truth_image(row):
    """根据ground_truth选项获取对应的图片路径"""
    gt = row['ground_truth']
    if gt in ['A', 'B', 'C', 'D']:
        return row[f'option_{gt}']
    return None

def count_text_in_images(csv_path, output_csv, base_path=""):
    """
    计算CSV文件中reference_image和ground_truth图片的文字数量，并添加到CSV中
    
    Args:
        csv_path: 输入CSV文件路径
        output_csv: 输出CSV文件路径
        base_path: 图片路径的基础目录，如果图片路径是相对路径，将与此基础路径拼接
    """
    print(f"正在读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 初始化新列
    df['ref_text_count'] = 0
    df['gt_text_count'] = 0
    df['total_text_count'] = 0
    
    # 初始化EasyOCR读取器（只初始化一次以提高效率）
    print("初始化EasyOCR读取器...")
    reader = easyocr.Reader(['en'])
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("EasyOCR读取器初始化成功")
    
    # 处理每一行
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理图片"):
        try:
            # 获取reference_image路径
            ref_img_path = row['reference_image']
            if base_path and not os.path.isabs(ref_img_path):
                ref_img_path = os.path.join(base_path, ref_img_path)
            
            # 获取ground_truth图片路径
            gt_img_path = get_ground_truth_image(row)
            if gt_img_path and base_path and not os.path.isabs(gt_img_path):
                gt_img_path = os.path.join(base_path, gt_img_path)
            
            # 计算reference_image的文字数量
            ref_text_count = 0
            if os.path.exists(ref_img_path):
                try:
                    _, ref_count = cover_text_in_image(ref_img_path, tokenizer, reader=reader)
                    if ref_count is not None:
                        ref_text_count = ref_count
                except Exception as e:
                    print(f"处理reference_image时出错: {e}")
            else:
                print(f"警告: reference_image不存在: {ref_img_path}")
            
            # 计算ground_truth图片的文字数量
            gt_text_count = 0
            if gt_img_path and os.path.exists(gt_img_path):
                try:
                    _, gt_count = cover_text_in_image(gt_img_path, tokenizer, reader=reader)
                    if gt_count is not None:
                        gt_text_count = gt_count
                except Exception as e:
                    print(f"处理ground_truth图片时出错: {e}")
            elif gt_img_path:
                print(f"警告: ground_truth图片不存在: {gt_img_path}")
            
            # 更新DataFrame
            df.at[idx, 'ref_text_count'] = ref_text_count
            df.at[idx, 'gt_text_count'] = gt_text_count
            df.at[idx, 'total_text_count'] = ref_text_count + gt_text_count
            
        except Exception as e:
            print(f"处理行 {idx} 时出错: {e}")
    
    # 保存结果
    print(f"正在保存结果到: {output_csv}")
    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df.to_csv(output_csv, index=False)
    print(f"处理完成! 结果已保存到: {output_csv}")
    
    # 显示统计信息
    print("\n统计信息:")
    print(f"平均reference_image文字数量: {df['ref_text_count'].mean():.2f}")
    print(f"平均ground_truth文字数量: {df['gt_text_count'].mean():.2f}")
    print(f"平均总文字数量: {df['total_text_count'].mean():.2f}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="计算CSV文件中reference_image和ground_truth图片的文字数量")
    parser.add_argument("--csv", required=True, help="输入CSV文件路径")
    parser.add_argument("--output", required=True, help="输出CSV文件路径")
    parser.add_argument("--base-path", default="", help="图片路径的基础目录，如果图片路径是相对路径，将与此基础路径拼接")
    args = parser.parse_args()
    
    count_text_in_images(args.csv, args.output, args.base_path)

if __name__ == "__main__":
    main()