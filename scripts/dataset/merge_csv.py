#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import random
import os

def merge_and_shuffle_csv_files(csv_file1, csv_file2, output_file, random_seed=42):
    """
    合并两个CSV文件，使用随机种子打乱顺序，并保存到指定路径
    
    Args:
        csv_file1: 第一个CSV文件路径
        csv_file2: 第二个CSV文件路径
        output_file: 输出CSV文件路径
        random_seed: 随机种子，默认为42
    """
    print(f"正在读取CSV文件: {csv_file1}")
    df1 = pd.read_csv(csv_file1)
    print(f"成功读取第一个CSV文件，包含 {len(df1)} 行")
    
    print(f"正在读取CSV文件: {csv_file2}")
    df2 = pd.read_csv(csv_file2)
    print(f"成功读取第二个CSV文件，包含 {len(df2)} 行")
    
    # 检查两个文件的列结构是否一致
    if list(df1.columns) != list(df2.columns):
        print("警告: 两个CSV文件的列结构不同")
        print(f"第一个CSV文件的列: {list(df1.columns)}")
        print(f"第二个CSV文件的列: {list(df2.columns)}")
        raise ValueError("两个CSV文件的列结构必须相同")
    
    # 合并数据
    print("正在合并数据...")
    merged_df = pd.concat([df1, df2], ignore_index=True)
    print(f"合并后共 {len(merged_df)} 行")
    
    # 使用随机种子打乱数据
    print(f"使用随机种子 {random_seed} 打乱数据...")
    random.seed(random_seed)
    merged_df = merged_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # 保存到输出文件
    print(f"正在保存到: {output_file}")
    merged_df.to_csv(output_file, index=False)
    print("完成！")

def main():
    parser = argparse.ArgumentParser(description='合并两个CSV文件并随机打乱')
    parser.add_argument('--csv1', type=str, required=True, help='第一个CSV文件的路径')
    parser.add_argument('--csv2', type=str, required=True, help='第二个CSV文件的路径')
    parser.add_argument('--output', type=str, required=True, help='输出CSV文件的路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，默认为42')
    
    args = parser.parse_args()
    
    try:
        merge_and_shuffle_csv_files(args.csv1, args.csv2, args.output, args.seed)
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()