# -*- coding: utf-8 -*-
import os
import sys
# 确保可以导入项目根目录下的模块 (如果 utils 在上一级目录)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)
import random
import base64
import time
import json
import csv
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import re

from utils.parse_jsonString import parse_probabilities
from utils.llm_call import CallLLM

# --- 硬编码配置 ---
MODEL_NAME = "gpt-4o"  # 替换为你的模型名称
API_BASE = "https://api.openai.com/v1"  # 替换为你的 API Base URL
API_KEY = "sk-xxx"  # 替换为你的 API Key
DATA_PATH = "./Dataset/analysis_results/artist_style_mcq.csv"  # 艺术家风格多选题数据集路径
OUTPUT_DIR = "./experiment/results/"  # 输出目录
NUM_OPTIONS = 4  # 4个选项 A,B,C,D
OPTION_IDS = [chr(65 + i) for i in range(NUM_OPTIONS)] # ['A', 'B', 'C', 'D']
SHUFFLE_DATA = True  # 默认打乱数据

# 图像文件扩展名
IMAGE_EXTENSION = ".jpg"
# 图像基础目录
IMAGE_BASE_DIR = "." # 假设 CSV 路径相对于项目根目录

# --- 硬编码提示 ---
SYSTEM_PROMPT = """# Requirement
You are an excellent comic art style analyst. You need to identify which of the option images shares the SAME ARTIST as the reference image shown at the top. Focus ONLY on the artistic style, drawing technique, and visual characteristics - not the content or characters.

Choose the option that most likely comes from the same artist based on:
- Line work and drawing technique
- Character design and proportions
- Shading and coloring style
- Composition and panel layout

ONLY based on the images provided, predict the probability that each option was created by the same artist as the reference image. Answer AS SIMPLE AS POSSIBLE. Make sure the probabilities add up to 1.

# Response Format
```json
{
  "A": probability for option A,
  "B": probability for option B,
  "C": probability for option C,
  "D": probability for option D
}
```
"""

USER_PROMPT_TEMPLATE = """You need to identify which option image (A, B, C, or D) was created by the SAME ARTIST as the reference image at the top.

Focus only on artistic style elements such as:
- Drawing technique and line work
- Character design and proportions 
- Color palette and shading style
- Overall composition and layout

Predict the probability that each option is from the same artist as the reference image."""

# --- 辅助函数 ---

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

def build_messages(reference_image_path: str, option_image_paths: Dict[str, str]) -> Optional[List[Dict]]:
    """
    构建发送给 MLLM 的消息列表。
    包含系统提示、用户文本提示、参考图像和所有选项图像。
    """
    reference_image_base64 = encode_image_to_base64(reference_image_path)
    if not reference_image_base64:
        print(f"错误：无法编码参考图像 {reference_image_path}")
        return None

    option_images_base64 = {}
    for opt_id, opt_path in option_image_paths.items():
        encoded = encode_image_to_base64(opt_path)
        if not encoded:
            print(f"错误：无法编码选项图像 {opt_path} (选项 {opt_id})")
            return None # 如果任何一个选项图像失败，则无法构建消息
        option_images_base64[opt_id] = encoded

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
    # 添加选项图像，按 A, B, C, D 顺序
    for opt_id in OPTION_IDS:
        if opt_id in option_images_base64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": option_images_base64[opt_id], "detail": "low"}
            })
        else:
            print(f"警告：缺少选项 {opt_id} 的图像数据。")
            return None  # 必须有所有选项才能继续

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    return messages

def evaluation(ground_truth: str, predicted_answer: Optional[str]) -> int:
    """比较预测答案和真实答案。"""
    if predicted_answer is None:
        return 0 # 无法解析视为错误
    return 1 if ground_truth == predicted_answer else 0

# --- 主实验函数 ---
def run_artist_style_experiment():
    """运行艺术家风格识别实验流程。"""
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
        print(f"成功从 {DATA_PATH} 加载 {len(data_df)} 条记录。")
        
        # 随机打乱数据顺序
        if SHUFFLE_DATA:
            # 使用frac=1确保取得所有数据，random_state可以指定固定的随机种子以便复现结果
            data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"已随机打乱数据顺序。")
    except FileNotFoundError:
        print(f"错误：找不到数据文件 {DATA_PATH}")
        return
    except Exception as e:
        print(f"错误：加载数据文件 {DATA_PATH} 时出错: {e}")
        return

    # 准备输出
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_csv_path = os.path.join(OUTPUT_DIR, f"artist_style_{MODEL_NAME.replace('/','_')}_{timestamp}.csv")
    
    fieldnames = [
        'id', 'reference_artist', 'reference_image', 
        'ground_truth', 'llm_response', 'predicted_answer', 'is_correct', 
        'option_A_artist', 'option_B_artist', 'option_C_artist', 'option_D_artist',
        'ground_truth_artist', 'prediction_confidence', 'total_tokens'
    ]

    total_correct = 0
    total_processed = 0
    total_tokens_consumed = 0
    total_confidence = 0.0
    correct_confidence = 0.0

    print(f"开始处理数据，结果将保存到 {output_csv_path}")

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 迭代处理每一行数据
        for index, row in tqdm(data_df.iterrows(), total=len(data_df), desc="处理艺术家风格题目"):
            # 获取题目信息
            query_id = row['id']
            reference_artist = row['reference_artist']
            reference_image = row['reference_image']
            ground_truth = row['ground_truth']  # 正确答案(A/B/C/D)
            
            # 提取各选项的图片路径和艺术家
            option_image_paths = {}
            option_artists = {}
            
            for opt_id in OPTION_IDS:
                img_col = f'option_{opt_id}'
                artist_col = f'option_{opt_id}_artist'
                
                if img_col in row and artist_col in row:
                    option_image_paths[opt_id] = row[img_col]
                    option_artists[opt_id] = row[artist_col]
                else:
                    print(f"警告：ID {query_id} 缺少 {img_col} 或 {artist_col} 列")
                    break
            
            # 确保所有选项都有图片路径
            if len(option_image_paths) != NUM_OPTIONS:
                print(f"跳过 ID {query_id}：缺少选项图片路径")
                continue
                
            # 构建 MLLM 输入消息
            messages = build_messages(reference_image, option_image_paths)
            if not messages:
                print(f"跳过 ID {query_id}：无法构建输入消息")
                continue

            # 记录正确答案对应的艺术家
            ground_truth_artist = option_artists.get(ground_truth, "未知")

            # 调用 LLM API
            llm_response_text = None
            predicted_probabilities = None
            predicted_answer = None
            is_correct = 0
            prediction_confidence = 0.0
            tokens = 0
            
            try:
                print(f"调用模型 ID: {query_id}...")
                llm_response_text, tokens = llm.post_request(messages=messages)
                
                # 解析概率结果，获取预测结果
                predicted_probabilities = parse_probabilities(llm_response_text)
                
                if predicted_probabilities:
                    predicted_answer = max(predicted_probabilities, key=predicted_probabilities.get)
                    prediction_confidence = predicted_probabilities.get(predicted_answer, 0.0)
                    is_correct = evaluation(ground_truth, predicted_answer)
                    
                    # 累计指标
                    total_confidence += prediction_confidence
                    if is_correct == 1:
                        correct_confidence += prediction_confidence
                
                total_tokens_consumed += tokens if tokens else 0
            except Exception as e:
                print(f"错误：调用 LLM API 处理 ID {query_id} 时出错: {e}")
                llm_response_text = f"API Error: {e}"

            total_processed += 1
            if is_correct == 1:
                total_correct += 1

            # 准备要写入 CSV 的记录
            record = {
                'id': query_id,
                'reference_artist': reference_artist,
                'reference_image': reference_image,
                'ground_truth': ground_truth,
                'llm_response': llm_response_text if llm_response_text else '',
                'predicted_answer': predicted_answer if predicted_answer else 'ParseError',
                'is_correct': is_correct,
                'option_A_artist': option_artists.get('A', ''),
                'option_B_artist': option_artists.get('B', ''),
                'option_C_artist': option_artists.get('C', ''),
                'option_D_artist': option_artists.get('D', ''),
                'ground_truth_artist': ground_truth_artist,
                'prediction_confidence': prediction_confidence,
                'total_tokens': tokens if tokens else 0
            }
            writer.writerow(record)
            csvfile.flush()  # 确保实时写入

            # 更新进度条显示准确率
            current_accuracy = (total_correct / total_processed) * 100 if total_processed > 0 else 0
            avg_confidence = total_confidence / total_processed if total_processed > 0 else 0
            tqdm.write(f"ID: {query_id}, GT: {ground_truth}({ground_truth_artist}), Pred: {predicted_answer}, Conf: {prediction_confidence:.3f}, Acc: {current_accuracy:.2f}%", end='\r')

    print(f"\n实验完成！")
    print(f"总共处理记录: {total_processed}")
    print(f"正确预测数量: {total_correct}")
    
    if total_processed > 0:
        final_accuracy = (total_correct / total_processed) * 100
        avg_confidence = total_confidence / total_processed
        avg_correct_confidence = correct_confidence / total_correct if total_correct > 0 else 0
        
        print(f"最终准确率: {final_accuracy:.2f}%")
        print(f"平均预测信心: {avg_confidence:.3f}")
        print(f"正确预测的平均信心: {avg_correct_confidence:.3f}")
    else:
        print("没有处理任何记录。")
        
    print(f"总消耗 Tokens (估算): {total_tokens_consumed}")
    print(f"结果已保存到: {output_csv_path}")

# --- 简单的结果分析 ---
def analyze_results(csv_path):
    """分析实验结果，显示按艺术家分组的准确率。"""
    try:
        results_df = pd.read_csv(csv_path)
        
        # 总体准确率
        total_correct = results_df['is_correct'].sum()
        total_records = len(results_df)
        overall_accuracy = (total_correct / total_records) * 100 if total_records > 0 else 0
        avg_confidence = results_df['prediction_confidence'].mean()
        
        print(f"\n===== 结果分析 =====")
        print(f"总记录数: {total_records}")
        print(f"正确预测数: {total_correct}")
        print(f"总体准确率: {overall_accuracy:.2f}%")
        print(f"平均预测信心: {avg_confidence:.3f}")
        
        # 按参考艺术家分组分析
        artist_results = results_df.groupby('reference_artist').agg(
            correct=('is_correct', 'sum'),
            total=('is_correct', 'count'),
            avg_confidence=('prediction_confidence', 'mean')
        )
        artist_results['accuracy'] = (artist_results['correct'] / artist_results['total']) * 100
        
        # 按准确率排序
        artist_results = artist_results.sort_values('accuracy', ascending=False)
        
        print("\n===== 按参考艺术家分组的准确率 =====")
        print(artist_results.head(10))
        
        # 混淆矩阵分析 - 查看哪些选项被选得更多
        option_counts = results_df['predicted_answer'].value_counts()
        print("\n===== 预测选项分布 =====")
        print(option_counts)
        
        # 计算不同选项的准确率
        option_accuracy = {}
        for option in OPTION_IDS:
            option_df = results_df[results_df['ground_truth'] == option]
            if len(option_df) > 0:
                option_accuracy[option] = (option_df['is_correct'].sum() / len(option_df)) * 100
            else:
                option_accuracy[option] = 0
        
        print("\n===== 不同选项的准确率 =====")
        for option, acc in option_accuracy.items():
            print(f"选项 {option}: {acc:.2f}%")
            
        # 按预测信心分析准确率
        confidence_bins = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
        confidence_groups = pd.cut(results_df['prediction_confidence'], confidence_bins)
        confidence_analysis = results_df.groupby(confidence_groups).agg(
            correct=('is_correct', 'sum'),
            total=('is_correct', 'count')
        )
        confidence_analysis['accuracy'] = (confidence_analysis['correct'] / confidence_analysis['total']) * 100
        
        print("\n===== 按预测信心分组的准确率 =====")
        print(confidence_analysis)
        
    except Exception as e:
        print(f"分析结果时出错: {e}")

# --- 执行入口 ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="运行艺术家风格识别实验")
    parser.add_argument("--analyze-only", help="只分析已有的结果文件", type=str, default=None)
    parser.add_argument("--data-path", help="数据集CSV文件路径", type=str, default=DATA_PATH)
    parser.add_argument("--model", help="使用的模型名称", type=str, default=MODEL_NAME)
    parser.add_argument("--no-shuffle", help="不打乱数据顺序", action="store_true")
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_results(args.analyze_only)
    else:
        # 更新配置
        if args.data_path:
            DATA_PATH = args.data_path
        if args.model:
            MODEL_NAME = args.model
        if args.no_shuffle:
            SHUFFLE_DATA = False
            
        run_artist_style_experiment() 