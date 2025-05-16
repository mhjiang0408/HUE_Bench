# -*- coding: utf-8 -*-
import os
import sys
# 确保可以导入项目根目录下的模块 (如果 utils 在上一级目录)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
import random
import base64
import time
import json
import csv
from utils.parse_jsonString import parse_probabilities
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import re


from utils.llm_call import CallLLM



# --- 硬编码配置 ---
MODEL_NAME = "gpt-4o"  # 替换为你的模型名称
API_BASE = "https://api.openai.com/v1"  # 替换为你的 API Base URL
API_KEY = "sk-xxx"  # 替换为你的 API Key
DATA_PATH = "./Data/comic_questions.csv"  # 漫画数据集路径
OUTPUT_DIR = "./experiment/results/"  # 输出目录
NUM_OPTIONS = 4  # 4个选项 A,B,C,D
OPTION_IDS = [chr(65 + i) for i in range(NUM_OPTIONS)] # ['A', 'B', 'C', 'D']

# 图像文件扩展名
IMAGE_EXTENSION = ".jpg"
# 图像基础目录
IMAGE_BASE_DIR = "." # 假设 CSV 路径相对于项目根目录

# --- 硬编码提示 ---
SYSTEM_PROMPT = "# Requirement\n You are an excellent comic connoisseur. You need to analyze the given image and choose the option image that most closely matches its way of expressing humor. ONLY based on the image and the options provided above, predict the probability that you would choose each option and answer AS SIMPLE AS POSSIBLE. Make sure the probabilities add up to 1.\n # Response Format\n ```json\n { \"A\": probability of choosing the option A, \"B\": probability of choosing the option B, \"C\": probability of choosing the option C, \"D\": probability of choosing the option D }\n```"

USER_PROMPT_TEMPLATE = "You need to predict the probability that you would choose each option images based on the above image. The following images provided to you are, in order, A, B, C, and D."

# --- 辅助函数 ---

def get_full_path(relative_path: str) -> Optional[str]:
    """构建文件的完整路径，处理 './' 前缀。"""
    if not relative_path:
        return None
    # 如果 IMAGE_BASE_DIR 设置了，并且路径不是绝对路径，则拼接
    # if IMAGE_BASE_DIR and not os.path.isabs(relative_path):
    #     # 移除路径开头的 './' (如果存在)
    #     cleaned_path = relative_path.lstrip('./').lstrip('/')
    #     full_path = os.path.join(IMAGE_BASE_DIR, cleaned_path)
    # else:
    full_path = relative_path # 假设已经是正确的相对或绝对路径

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
            "image_url": {"url": reference_image_base64, "detail": "low"} # 假设低细节足够
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
            # 根据需要决定是否继续或返回 None
            # return None

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
def run_comic_experiment():
    """运行漫画风格识别实验流程。"""
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
        # 确保 'id' 列是字符串，以便用于文件名
        if 'id' in data_df.columns:
            data_df['id'] = data_df['id'].astype(str)
        else:
            print(f"错误：CSV 文件 {DATA_PATH} 中缺少 'id' 列。")
            return
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
    output_csv_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME.replace('/','_')}_{timestamp}.csv")
    fieldnames = [
        'query_id', 'reference_image_path',
        'option_A_path', 'option_B_path', 'option_C_path', 'option_D_path',
        'ground_truth', 'llm_response', 'predicted_answer', 'is_correct', 'total_tokens',
        'reference_author', 'ground_truth_author', 'distractor_authors'
    ]

    results_list = []
    total_correct = 0
    total_processed = 0
    total_tokens_consumed = 0

    print(f"开始处理数据，结果将保存到 {output_csv_path}")

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 迭代处理每一行数据
        for index, row in tqdm(data_df.iterrows(), total=len(data_df), desc="处理漫画数据"):
            query_id = row['id']
            ground_truth = row['correct_answer'] # 正确答案标签 (A/B/C/D)
            reference_author = row['reference_author']
            ground_truth_author = row['ground_truth_author']
            distractor_authors = row['distractor_authors']

            # 获取参考图像路径
            reference_image_path = get_full_path(row['reference_image'])
            if not reference_image_path:
                print(f"跳过 ID {query_id}：找不到参考图像 {row['reference_image']}")
                continue

            # 获取选项图像路径
            option_image_paths = {}
            valid_options = True
            for opt_id in OPTION_IDS:
                col_name = f'choice_{opt_id}'
                if col_name in row:
                    option_path = get_full_path(row[col_name])
                    if option_path:
                        option_image_paths[opt_id] = option_path
                    else:
                        print(f"跳过 ID {query_id}：找不到选项 {opt_id} 的图像 {row[col_name]}")
                        valid_options = False
                        break
                else:
                    # 尝试使用替代列名
                    alt_col_name = f'choice{opt_id}'
                    if alt_col_name in row:
                        option_path = get_full_path(row[alt_col_name])
                        if option_path:
                            option_image_paths[opt_id] = option_path
                        else:
                            print(f"跳过 ID {query_id}：找不到选项 {opt_id} 的图像 {row[alt_col_name]}")
                            valid_options = False
                            break
                    else:
                        print(f"跳过 ID {query_id}：CSV 中缺少列 {col_name} 或 {alt_col_name}")
                        valid_options = False
                        break
            if not valid_options:
                continue

            # 构建 MLLM 输入消息
            messages = build_messages(reference_image_path, option_image_paths)
            if not messages:
                print(f"跳过 ID {query_id}：无法构建输入消息 (可能是图像编码失败)")
                continue

            # 调用 LLM API
            llm_response_text = None
            answer = None
            is_correct = 0
            tokens = 0
            try:
                print(f"调用模型 ID: {query_id}...") # Debugging print
                llm_response_text, tokens = llm.post_request(messages=messages)
                
                format_answer = parse_probabilities(llm_response_text)
                answer = max(format_answer, key=format_answer.get)
                is_correct = evaluation(ground_truth, answer)
                total_tokens_consumed += tokens if tokens else 0
            except Exception as e:
                print(f"错误：调用 LLM API 处理 ID {query_id} 时出错: {e}")
                llm_response_text = f"API Error: {e}" # 记录错误信息

            total_processed += 1
            if is_correct == 1:
                total_correct += 1

            # 准备要写入 CSV 的记录
            record = {
                'query_id': query_id,
                'reference_image_path': reference_image_path,
                'option_A_path': option_image_paths.get('A', ''),
                'option_B_path': option_image_paths.get('B', ''),
                'option_C_path': option_image_paths.get('C', ''),
                'option_D_path': option_image_paths.get('D', ''),
                'ground_truth': ground_truth,
                'llm_response': llm_response_text if llm_response_text else '',
                'predicted_answer': answer if answer else 'ParseError',
                'is_correct': is_correct,
                'total_tokens': tokens if tokens else 0,
                'reference_author': reference_author,
                'ground_truth_author': ground_truth_author,
                'distractor_authors': distractor_authors
            }
            writer.writerow(record)
            csvfile.flush() # 确保实时写入

            # 更新进度条显示准确率
            current_accuracy = (total_correct / total_processed) * 100 if total_processed > 0 else 0
            tqdm.write(f"ID: {query_id}, GT: {ground_truth}, Pred: {answer}, Correct: {is_correct==1}, Acc: {current_accuracy:.2f}%", end='\r')

    print(f"\n实验完成！")
    print(f"总共处理记录: {total_processed}")
    print(f"正确预测数量: {total_correct}")
    if total_processed > 0:
        final_accuracy = (total_correct / total_processed) * 100
        print(f"最终准确率: {final_accuracy:.2f}%")
    else:
        print("没有处理任何记录。")
    print(f"总消耗 Tokens (估算): {total_tokens_consumed}")
    print(f"结果已保存到: {output_csv_path}")

# --- 简单的结果分析 ---
def analyze_results(csv_path):
    """分析实验结果，显示按作者分组的准确率。"""
    try:
        results_df = pd.read_csv(csv_path)
        
        # 总体准确率
        total_correct = results_df['is_correct'].sum()
        total_records = len(results_df)
        overall_accuracy = (total_correct / total_records) * 100 if total_records > 0 else 0
        
        print(f"\n===== 结果分析 =====")
        print(f"总记录数: {total_records}")
        print(f"正确预测数: {total_correct}")
        print(f"总体准确率: {overall_accuracy:.2f}%")
        
        # 按作者分组分析
        author_results = results_df.groupby('reference_author').agg(
            correct=('is_correct', 'sum'),
            total=('is_correct', 'count')
        )
        author_results['accuracy'] = (author_results['correct'] / author_results['total']) * 100
        
        # 按准确率排序
        author_results = author_results.sort_values('accuracy', ascending=False)
        
        print("\n===== 按作者分组的准确率 =====")
        print(author_results.head(10))
        
        # 混淆矩阵分析 - 查看哪些选项被选得更多
        option_counts = results_df['predicted_answer'].value_counts()
        print("\n===== 预测选项分布 =====")
        print(option_counts)
        
        # 计算不同选项的准确率
        option_accuracy = {}
        for option in ['A', 'B', 'C', 'D']:
            option_df = results_df[results_df['ground_truth'] == option]
            if len(option_df) > 0:
                option_accuracy[option] = (option_df['is_correct'].sum() / len(option_df)) * 100
            else:
                option_accuracy[option] = 0
        
        print("\n===== 不同选项的准确率 =====")
        for option, acc in option_accuracy.items():
            print(f"选项 {option}: {acc:.2f}%")
        
    except Exception as e:
        print(f"分析结果时出错: {e}")

# --- 执行入口 ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="运行漫画风格识别实验")
    parser.add_argument("--analyze-only", help="只分析已有的结果文件", type=str, default=None)
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_results(args.analyze_only)
    else:
        run_comic_experiment()