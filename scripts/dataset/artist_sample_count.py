#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
from collections import Counter
import os

def count_artist_samples(input_csv, output_csv):
    """
    统计CSV文件中每个艺术家的样本数量，仅考虑reference_artist列
    
    Args:
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径
    """
    print(f"正在读取 {input_csv}...")
    # 读取CSV文件
    df = pd.read_csv(input_csv)
    
    # 初始化计数器
    artist_counts = Counter()
    
    # 只统计reference_artist列中每个艺术家的出现次数
    if 'reference_artist' in df.columns:
        artist_counts.update(df['reference_artist'].tolist())
        print(f"找到 {len(artist_counts)} 个不同的艺术家")
    else:
        print("错误: CSV文件中没有找到'reference_artist'列")
        return
    
    # 转换为DataFrame
    result_df = pd.DataFrame({
        'artist': list(artist_counts.keys()),
        'sample_count': list(artist_counts.values())
    })
    
    # 按样本数量降序排序
    result_df = result_df.sort_values(by='sample_count', ascending=False)
    
    # 保存结果
    result_df.to_csv(output_csv, index=False)
    print(f"结果已保存至 {output_csv}")
    
    # 打印一些统计信息
    total_samples = sum(artist_counts.values())
    print(f"总样本数: {total_samples}")
    print(f"艺术家数量: {len(artist_counts)}")
    print(f"平均每个艺术家的样本数: {total_samples / len(artist_counts):.2f}")
    
    # 打印前10个艺术家的样本数
    print("\n前10个艺术家的样本数:")
    for i, (artist, count) in enumerate(result_df.iloc[:10].iterrows(), 1):
        print(f"{i}. {artist['artist']}: {artist['sample_count']}")
    
    # 分析复合ID中的艺术家和漫画系列
    if any('^' in str(artist) for artist in result_df['artist']):
        print("\n分析艺术家和漫画系列...")
        # 提取艺术家名称（不含漫画系列）
        artist_only_counts = Counter()
        comic_series_counts = Counter()
        
        for artist_id, count in artist_counts.items():
            if '^' in str(artist_id):
                artist_name, comic_series = artist_id.split('^', 1)
                artist_only_counts[artist_name] += count
                comic_series_counts[comic_series] += count
        
        # 保存艺术家统计结果（不含漫画系列）
        artist_only_df = pd.DataFrame({
            'artist_name': list(artist_only_counts.keys()),
            'sample_count': list(artist_only_counts.values())
        }).sort_values(by='sample_count', ascending=False)
        
        artist_only_output = os.path.splitext(output_csv)[0] + '_artist_only.csv'
        artist_only_df.to_csv(artist_only_output, index=False)
        print(f"艺术家统计结果（不含漫画系列）已保存至 {artist_only_output}")
        
        # 保存漫画系列统计结果
        comic_series_df = pd.DataFrame({
            'comic_series': list(comic_series_counts.keys()),
            'sample_count': list(comic_series_counts.values())
        }).sort_values(by='sample_count', ascending=False)
        
        comic_series_output = os.path.splitext(output_csv)[0] + '_comic_series.csv'
        comic_series_df.to_csv(comic_series_output, index=False)
        print(f"漫画系列统计结果已保存至 {comic_series_output}")

def main():
    parser = argparse.ArgumentParser(description='统计CSV文件中每个艺术家的样本数量')
    parser.add_argument('--input-csv', required=True, help='输入CSV文件路径')
    parser.add_argument('--output-csv', default='artist_sample_count2.csv', help='输出CSV文件路径')
    args = parser.parse_args()
    
    count_artist_samples(args.input_csv, args.output_csv)

if __name__ == '__main__':
    main() 