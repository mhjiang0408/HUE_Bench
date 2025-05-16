#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import os
import sys
from collections import Counter
sys.path.append(os.getcwd())
from utils.parse_jsonString import parse_probabilities

def count_option_choices(df):
    """
    统计CSV文件中LLM选择各选项(A,B,C,D,E)的次数
    
    Args:
        csv_path: CSV文件路径
    
    Returns:
        Counter: 各选项被选择的次数
    """

    total_samples = len(df)
    print(f"总样本数: {total_samples}")
    
    # 初始化计数器
    option_counts = Counter()
    
    # 遍历每一行进行分析
    print('here')
    for idx, row in df.iterrows():
        # 解析模型响应
        response_field = 'llm_response' if 'llm_response' in row else 'response'
        if response_field in row:
            # 优先使用5个选项的解析方式
            probabilities = parse_probabilities(str(row[response_field]), 5)
            if not probabilities:
                # 如果失败，尝试使用4个选项的解析方式
                probabilities = parse_probabilities(str(row[response_field]), 4)
            
            if probabilities:
                # 获取模型预测的选项（概率最高的）
                # 首先归一化
                total_probability = sum(probabilities.values())
                probabilities = {k: v / total_probability for k, v in probabilities.items()}
                for option, probability in probabilities.items():
                    option_counts[option] += probability
    
    # 打印结果
    # print(f"\n文件: {os.path.basename(csv_path)}")
    print("选项\t次数\t百分比")
    for opt in ['A', 'B', 'C', 'D', 'E']:
        count = option_counts.get(opt, 0)
        percentage = count / total_samples * 100 if total_samples > 0 else 0
        print(f"{opt}\t{count}\t{percentage:.2f}%")
    
    return option_counts, total_samples

def analyze_multiple_files(paths_csv, output_file=None):
    """
    分析多个CSV文件中LLM的选项分布
    
    Args:
        csv_files: CSV文件路径列表
        output_file: 输出CSV文件路径
    """
    results = []
    paths_df = pd.read_csv(paths_csv)
    
    if 'path' not in paths_df.columns:
        print("错误: CSV文件中没有'path'列")
        return
    for idx, row in paths_df.iterrows():
        csv_path = row['path']
        
        # 检查文件是否存在
        if not os.path.exists(csv_path):
            print(f"警告: 文件不存在 - {csv_path}")
            continue
        
        print(f"\n正在分析: {csv_path}")
        current_df = pd.read_csv(csv_path)
        
        try:
            option_counts, total_samples = count_option_choices(current_df)
            # 记录结果
            result = {
                'file': os.path.basename(csv_path),
                'path': csv_path,
                'total_samples': total_samples
            }
            
            # 添加各选项的计数和百分比
            for opt in ['A', 'B', 'C', 'D', 'E']:
                count = option_counts.get(opt, 0)
                percentage = count / total_samples * 100 if total_samples > 0 else 0
                result[f'{opt}_count'] = count
                result[f'{opt}_percentage'] = percentage
            
            results.append(result)
            print("-" * 50)
            
        except Exception as e:
            print(f"分析文件时出错 {csv_path}: {str(e)}")
    
    # 创建结果DataFrame
    if results:
        results_df = pd.DataFrame(results)
        
        # 如果指定了输出文件，保存结果
        if output_file:
            results_df.to_csv(output_file, index=False)
            print(f"\n结果已保存至: {output_file}")
        
        # 打印汇总表格
        print("\n选项分布汇总:")
        print(results_df[['file', 'total_samples', 'A_count', 'B_count', 'C_count', 'D_count', 'E_count']].to_string(index=False))
        
        return results_df
    
    return None

def main():
    parser = argparse.ArgumentParser(description="统计CSV文件中LLM选择各选项(A,B,C,D,E)的次数")
    parser.add_argument("--csv", required=True, help="CSV文件路径，可以指定多个文件")
    parser.add_argument("--output", type=str, default=None, help="输出CSV文件路径")
    args = parser.parse_args()
    
    analyze_multiple_files(args.csv, args.output)

if __name__ == "__main__":
    main()