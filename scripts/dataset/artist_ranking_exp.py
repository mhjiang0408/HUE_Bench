# -*- coding: utf-8 -*-
import os
import sys
# 确保可以导入项目根目录下的模块 (如果 utils 在上一级目录)
sys.path.append(os.getcwd())
import random
import base64
import time
import matplotlib.pyplot as plt
import json
import csv
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import re
import numpy as np

from utils.parse_jsonString import parse_json_string
from utils.llm_call import CallLLM

# --- 硬编码配置 ---
MODEL_NAME = "gpt-4o"  # 替换为你的模型名称

API_BASE = "https://api.openai.com/v1"  # 替换为你的 API Base URL
API_KEY = "sk-xxx"  # 替换为你的 API Key
DATA_PATH = "./Dataset/artist_pairs.csv"  # 修改为艺术家排名数据集路径
OUTPUT_DIR = "./experiment/results/"  # 输出目录
NUM_OPTIONS = 6  # 6个候选艺术家
OPTION_IDS = [str(i+1) for i in range(NUM_OPTIONS)] # ['1', '2', '3', '4', '5', '6']

# 图像文件扩展名
IMAGE_EXTENSION = ".jpg"
# 图像基础目录
IMAGE_BASE_DIR = "./gocomics_downloads_political" # 漫画图像目录路径

# --- 硬编码提示 ---
SYSTEM_PROMPT = """# Requirement
You are an excellent comic art style analyst. You need to analyze the given reference comic image and predict the six candidate images according to their artistic style similarity to the reference image. ONLY based on the images provided, predict the probability of stylistic similarity for each candidate and answer AS SIMPLE AS POSSIBLE. The probabilities should be up to 1.

# Response Format
```json
{
  "1": probability for candidate 1,
  "2": probability for candidate 2,
  "3": probability for candidate 3,
  "4": probability for candidate 4,
  "5": probability for candidate 5,
  "6": probability for candidate 6
}
```
"""

USER_PROMPT_TEMPLATE = """Analyze which of the following candidate comic images are more similar in ARTISTIC STYLE to the reference image shown above. The candidate images are numbered from 1 to 6.

Focus on artistic style elements like:
- Drawing technique and line work
- Color palette and shading style
- Character design and proportions
- Overall composition and panel layout

Please rank the candidates by assigning probability values indicating their stylistic similarity to the reference image. Each probability should be up to 1, with higher values indicating greater stylistic similarity.
"""

# --- 辅助函数 ---
def get_full_path(relative_path: str) -> Optional[str]:
    """构建文件的完整路径，处理 './' 前缀。"""
    if not relative_path:
        return None
    
    # 移除路径开头的 './' (如果存在)
    # cleaned_path = relative_path.lstrip('./').lstrip('/')
    # full_path = os.path.join(IMAGE_BASE_DIR, cleaned_path)

    full_path = relative_path
    # 检查文件是否存在
    if os.path.exists(full_path):
        return full_path
    else:
        # 尝试添加默认扩展名（如果路径看起来没有扩展名）
        if not os.path.splitext(full_path)[1] and IMAGE_EXTENSION:
            path_with_ext = f"{full_path}{IMAGE_EXTENSION}"
            if os.path.exists(path_with_ext):
                return path_with_ext
        else:
            path_with_ext = f"{full_path}.jpg"
            if os.path.exists(path_with_ext):
                return path_with_ext
        print(f"警告：无法找到文件路径 '{relative_path}' (尝试解析为 '{full_path}')")
        return None # 返回 None 表示未找到

def encode_image_to_base64(image_path: str) -> Optional[str]:
    """将图像文件编码为 Base64 字符串。"""
    full_path = get_full_path(image_path)
    if not full_path:
        return None
    try:
        with open(full_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            # 根据扩展名确定 MIME 类型
            ext = os.path.splitext(full_path)[1].lower()
            mime_type = f"image/{ext[1:]}" if ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp'] else "image/jpeg" # 默认为 jpeg
            return f"data:{mime_type};base64,{encoded_string}"
    except FileNotFoundError:
        print(f"错误：编码图像时未找到文件 {full_path}")
        return None
    except Exception as e:
        print(f"错误：编码图像 {full_path} 时出错: {e}")
        return None

def build_ranking_messages(reference_image_path: str, candidate_image_paths: Dict[str, str]) -> Optional[List[Dict]]:
    """
    构建发送给 MLLM 的消息列表。
    包含系统提示、用户文本提示、参考图像和所有候选图像。
    
    Args:
        reference_image_path: 参考艺术家图像路径
        candidate_image_paths: 候选艺术家图像路径字典，键为选项ID (1-6)，值为图像路径
    
    Returns:
        构建好的消息列表，如果无法编码任何图像则返回None
    """
    reference_image_base64 = encode_image_to_base64(reference_image_path)
    if not reference_image_base64:
        print(f"错误：无法编码参考图像 {reference_image_path}")
        return None

    # 编码所有候选图像
    candidate_images_base64 = {}
    for opt_id, opt_path in candidate_image_paths.items():
        encoded = encode_image_to_base64(opt_path)
        if not encoded:
            print(f"错误：无法编码候选图像 {opt_path} (选项 {opt_id})")
            return None  # 如果任何一个选项图像失败，则无法构建消息
        candidate_images_base64[opt_id] = encoded

    # 构建用户提示内容列表
    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": reference_image_base64, "detail": "low"}
        },
        {
            "type": "text",
            "text": USER_PROMPT_TEMPLATE
        }
    ]
    
    # 添加所有候选图像，按数字顺序
    for opt_id in OPTION_IDS:
        if opt_id in candidate_images_base64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": candidate_images_base64[opt_id], "detail": "low"}
            })
        else:
            print(f"警告：缺少选项 {opt_id} 的图像数据。")
            return None  # 必须有所有候选图像

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    return messages

def calculate_ranking_score(true_artist: str, predicted_ranking: Dict[str, float], candidate_artists: List[str]) -> Dict:
    """
    计算排名预测的分数。
    
    Args:
        true_artist: 真实应该排第一的艺术家
        predicted_ranking: 模型预测的排名概率字典，键为选项ID (1-6)，值为概率
        candidate_artists: 候选艺术家列表，索引对应选项ID的数字部分减1
    
    Returns:
        包含各种评估指标的字典
    """
    # 找到true_artist在候选列表中的位置
    try:
        true_idx = candidate_artists.index(true_artist)
        true_option = str(true_idx + 1)  # 转换为选项ID格式 (1-6)
    except ValueError:
        print(f"错误：真实艺术家 {true_artist} 不在候选列表中")
        return {
            "is_correct_top1": 0,
            "true_rank": -1,
            "true_probability": 0.0,
            "normalized_true_probability": 0.0,
            "reciprocal_rank": 0.0
        }
    
    # 根据概率对选项排序
    sorted_options = sorted(predicted_ranking.items(), key=lambda x: x[1], reverse=True)
    option_ranks = {opt: rank+1 for rank, (opt, _) in enumerate(sorted_options)}
    
    # 计算各种指标
    true_rank = option_ranks.get(true_option, -1)
    is_correct_top1 = 1 if true_rank == 1 else 0
    true_probability = predicted_ranking.get(true_option, 0.0)
    
    # 计算归一化概率 (当前概率 / 平均概率), 平均概率为1/6
    normalized_true_probability = true_probability * NUM_OPTIONS if NUM_OPTIONS > 0 else 0.0
    
    # 计算倒数排名 (MRR)
    reciprocal_rank = 1.0 / true_rank if true_rank > 0 else 0.0
    
    return {
        "is_correct_top1": is_correct_top1,
        "true_rank": true_rank,
        "true_probability": true_probability,
        "normalized_true_probability": normalized_true_probability,
        "reciprocal_rank": reciprocal_rank
    }

# --- 主实验函数 ---
def run_artist_ranking_experiment():
    """运行艺术家风格排名实验流程。"""
    # 初始化 LLM 客户端
    try:
        llm = CallLLM(model=MODEL_NAME, api_base=API_BASE, api_key=API_KEY)
    except NameError:
        print("错误：CallLLM 类未定义。请确保 utils/llm_call.py 可用或取消注释虚拟类。")
        return
    except Exception as e:
        print(f"错误：初始化 LLM 客户端失败: {e}")
        return

    # 加载数据
    try:
        data_df = pd.read_csv(DATA_PATH)
        
        # 给每一行添加唯一ID
        if 'id' not in data_df.columns:
            data_df['id'] = [f"ranking_{i+1}" for i in range(len(data_df))]
            
        print(f"成功从 {DATA_PATH} 加载 {len(data_df)} 条记录。")
    except FileNotFoundError:
        print(f"错误：找不到数据文件 {DATA_PATH}")
        return
    except Exception as e:
        print(f"错误：加载数据文件 {DATA_PATH} 时出错: {e}")
        return

    # 准备输出
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_csv_path = os.path.join(OUTPUT_DIR, f"artist_ranking_exp_{MODEL_NAME.replace('/','_')}_{timestamp}.csv")
    
    # 定义输出CSV的字段
    fieldnames = [
        'query_id', 'reference_artist', 'reference_image',
        'candidate1', 'candidate1_image', 'similarity1',
        'candidate2', 'candidate2_image', 'similarity2',
        'candidate3', 'candidate3_image', 'similarity3',
        'candidate4', 'candidate4_image', 'similarity4',
        'candidate5', 'candidate5_image', 'similarity5',
        'candidate6', 'candidate6_image', 'similarity6',
        'llm_response', 'pred_prob1', 'pred_prob2', 'pred_prob3', 'pred_prob4', 'pred_prob5', 'pred_prob6',
        'is_correct_top1', 'true_rank', 'true_probability', 'normalized_true_probability', 'reciprocal_rank',
        'total_tokens'
    ]

    total_correct = 0
    total_processed = 0
    total_tokens_consumed = 0
    total_mrr = 0.0

    print(f"开始处理数据，结果将保存到 {output_csv_path}")

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 迭代处理每一行数据
        for index, row in tqdm(data_df.iterrows(), total=len(data_df), desc="处理艺术家排名题目"):
            query_id = row['id'] if 'id' in row else f"ranking_{index+1}"
            reference_artist = row['reference_artist']
            reference_image = row['reference_image']
            
            # 构建候选艺术家和图像的列表
            candidate_artists = []
            candidate_images = {}
            
            for i in range(1, NUM_OPTIONS+1):
                candidate_col = f'candidate{i}'
                image_col = f'candidate{i}_image'
                similarity_col = f'similarity{i}'
                
                if candidate_col in row and image_col in row:
                    candidate_artists.append(row[candidate_col])
                    candidate_images[str(i)] = row[image_col]
            
            # 构建 MLLM 输入消息
            messages = build_ranking_messages(reference_image, candidate_images)
            if not messages:
                print(f"跳过 ID {query_id}：无法构建输入消息 (可能是图像编码失败)")
                continue

            # 调用 LLM API
            llm_response_text = None
            probabilities = None
            ranking_scores = {}
            tokens = 0
            
            try:
                print(f"调用模型 ID: {query_id}...")
                llm_response_text, tokens = llm.post_request(messages=messages)
                
                # 解析概率结果
                probabilities = parse_json_string(llm_response_text)
                
                # 计算排名分数
                ranking_scores = calculate_ranking_score(
                    reference_artist, 
                    probabilities,
                    candidate_artists
                )
                
                total_tokens_consumed += tokens if tokens else 0
            except Exception as e:
                print(f"错误：调用 LLM API 处理 ID {query_id} 时出错: {e}")
                llm_response_text = f"API Error: {e}"
                probabilities = {str(i+1): 0.0 for i in range(NUM_OPTIONS)}
                ranking_scores = {
                    "is_correct_top1": 0,
                    "true_rank": -1,
                    "true_probability": 0.0,
                    "normalized_true_probability": 0.0,
                    "reciprocal_rank": 0.0
                }

            total_processed += 1
            if ranking_scores.get("is_correct_top1", 0) == 1:
                total_correct += 1
            total_mrr += ranking_scores.get("reciprocal_rank", 0.0)

            # 准备要写入 CSV 的记录
            record = {
                'query_id': query_id,
                'reference_artist': reference_artist,
                'reference_image': reference_image,
                'llm_response': llm_response_text if llm_response_text else '',
                'total_tokens': tokens if tokens else 0,
            }
            
            # 添加候选艺术家信息
            for i in range(1, NUM_OPTIONS+1):
                candidate_col = f'candidate{i}'
                image_col = f'candidate{i}_image'
                similarity_col = f'similarity{i}'
                
                if candidate_col in row:
                    record[candidate_col] = row[candidate_col]
                if image_col in row:
                    record[image_col] = row[image_col]
                if similarity_col in row:
                    record[similarity_col] = row[similarity_col]
            
            # 添加预测概率
            for i in range(1, NUM_OPTIONS+1):
                prob_key = f'pred_prob{i}'
                record[prob_key] = probabilities.get(str(i), 0.0) if probabilities else 0.0
            
            # 添加评估指标
            for key, value in ranking_scores.items():
                record[key] = value
            
            writer.writerow(record)
            csvfile.flush()  # 确保实时写入

            # 更新进度条显示准确率和MRR
            current_accuracy = (total_correct / total_processed) * 100 if total_processed > 0 else 0
            current_mrr = (total_mrr / total_processed) if total_processed > 0 else 0
            tqdm.write(f"ID: {query_id}, Acc@1: {current_accuracy:.2f}%, MRR: {current_mrr:.4f}", end='\r')

    print(f"\n实验完成！")
    print(f"总共处理记录: {total_processed}")
    print(f"正确预测数量 (Top-1): {total_correct}")
    
    if total_processed > 0:
        final_accuracy = (total_correct / total_processed) * 100
        final_mrr = (total_mrr / total_processed)
        print(f"Top-1 准确率: {final_accuracy:.2f}%")
        print(f"平均倒数排名 (MRR): {final_mrr:.4f}")
    else:
        print("没有处理任何记录。")
        
    print(f"总消耗 Tokens (估算): {total_tokens_consumed}")
    print(f"结果已保存到: {output_csv_path}")

# --- 简单的结果分析 ---
def analyze_ranking_results(csv_path):
    """分析艺术家排名实验结果。"""
    try:
        results_df = pd.read_csv(csv_path)
        
        # 总体指标
        total_correct = results_df['is_correct_top1'].sum()
        total_records = len(results_df)
        overall_accuracy = (total_correct / total_records) * 100 if total_records > 0 else 0
        overall_mrr = results_df['reciprocal_rank'].mean()
        
        print(f"\n===== 结果分析 =====")
        print(f"总记录数: {total_records}")
        print(f"Top-1正确预测数: {total_correct}")
        print(f"Top-1准确率: {overall_accuracy:.2f}%")
        print(f"平均倒数排名 (MRR): {overall_mrr:.4f}")
        
        # 分析排名分布
        rank_distribution = results_df['true_rank'].value_counts().sort_index()
        print("\n===== 真实艺术家排名分布 =====")
        for rank, count in rank_distribution.items():
            percentage = (count / total_records) * 100
            print(f"排名 {rank}: {count} 次 ({percentage:.2f}%)")
        
        # 计算累积分布
        cumulative_ranks = [0] * (NUM_OPTIONS + 1)
        for rank in range(1, NUM_OPTIONS + 1):
            cumulative_ranks[rank] = (results_df['true_rank'] <= rank).sum()
            percentage = (cumulative_ranks[rank] / total_records) * 100
            print(f"Top-{rank} 召回率: {percentage:.2f}%")
        
        # 绘制累积排名曲线
        try:
            
            ranks = list(range(1, NUM_OPTIONS + 1))
            cumulative_percentage = [100 * cumulative_ranks[r] / total_records for r in ranks]
            
            plt.figure(figsize=(10, 6))
            plt.plot(ranks, cumulative_percentage, 'o-', linewidth=2)
            plt.xlabel('排名位置')
            plt.ylabel('累积召回率 (%)')
            plt.title('累积排名曲线')
            plt.xticks(ranks)
            plt.grid(True)
            
            # 保存图表
            plt_path = os.path.splitext(csv_path)[0] + "_ranking_curve.png"
            plt.savefig(plt_path)
            print(f"\n累积排名曲线已保存到: {plt_path}")
        except ImportError:
            print("\n警告: 未安装matplotlib，无法绘制累积排名曲线。")
        
        # 分析平均概率值
        avg_true_prob = results_df['true_probability'].mean()
        avg_normalized_prob = results_df['normalized_true_probability'].mean()
        print(f"\n真实艺术家平均概率: {avg_true_prob:.4f}")
        print(f"真实艺术家平均归一化概率: {avg_normalized_prob:.4f} (1.0表示随机)")
        
    except Exception as e:
        print(f"分析结果时出错: {e}")

# --- 执行入口 ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="运行艺术家风格排名实验")
    parser.add_argument("--analyze-only", help="只分析已有的结果文件", type=str, default=None)
    parser.add_argument("--data-path", help="艺术家排名数据的CSV文件路径", type=str, default=DATA_PATH)
    parser.add_argument("--image-dir", help="漫画图像目录的路径", type=str, default=IMAGE_BASE_DIR)
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_ranking_results(args.analyze_only)
    else:
        # 更新配置
        if args.data_path:
            DATA_PATH = args.data_path
        if args.image_dir:
            IMAGE_BASE_DIR = args.image_dir
        run_artist_ranking_experiment()