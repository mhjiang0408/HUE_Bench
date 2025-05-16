#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import re
import os
import sys
sys.path.append(os.getcwd())
import csv
from utils.parse_date import extract_comic_series_and_date

def check_2025(date_str):
    date_obj = extract_comic_series_and_date(date_str)[2]
    try:
        if date_obj.year == 2025:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error: {e},str:{date_str}")
        return False
def filter_2025_records(input_csv, output_csv):
    """
    从CSV文件中筛选包含2025年的记录并保存到新的CSV文件
    采用边读取边写入的方式，找到一个匹配记录就立即写入
    
    Args:
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径
    """
    print(f"正在读取 {input_csv}...")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    
    # 记录统计
    total_rows = 0
    matched_rows = 0
    
    # 使用pandas读取CSV文件头部，获取列名
    df_header = pd.read_csv(input_csv, nrows=0)
    header = df_header.columns.tolist()
    
    # 打开输出文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=header)
        writer.writeheader()
        # 分块读取输入文件
        for chunk in pd.read_csv(input_csv, chunksize=1000):
            for idx, row in chunk.iterrows():
                total_rows += 1
                row_has_2025 = False
                
                # 检查所有列中是否有包含2025年的值
                id = row['id']
                if check_2025(id):
                    row_has_2025 = True
                
                # 如果找到匹配项，立即写入
                if row_has_2025:
                    writer.writerow(row.to_dict())
                    matched_rows += 1
                    print(f"\r已找到 {matched_rows} 条记录，处理进度: {total_rows} 行", end='')
    
    print(f"\n总共处理 {total_rows} 行，找到 {matched_rows} 条包含2025年的记录")
    print(f"结果已保存至 {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='从CSV文件中筛选包含2025年的记录')
    parser.add_argument('--input-csv', required=True, help='输入CSV文件路径')
    parser.add_argument('--output-csv', required=True, help='输出CSV文件路径')
    args = parser.parse_args()
    
    filter_2025_records(args.input_csv, args.output_csv)

if __name__ == '__main__':
    main()