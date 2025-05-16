#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import argparse
import re
from datetime import datetime
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())
from utils.parse_date import extract_comic_series_and_date

def extract_date_from_path(image_path):
    """从图片路径中提取日期"""
    # 提取文件名
    filename = os.path.basename(image_path)
    
    # 使用extract_comic_series_and_date函数提取日期
    _, date_str, date_obj = extract_comic_series_and_date(filename)
    
    # 确保date_str是字符串格式
    if date_str is None:
        # 尝试从文件名直接提取日期
        match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if match:
            date_str = match.group(1)
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                print(f"警告: 无法解析日期字符串 {date_str}")
                return None
    
    return date_obj

def get_artist_images_with_dates(df, artist_id):
    """获取特定艺术家的所有图片及其日期，并按日期排序"""
    # 分离艺术家ID的组件
    if '^' in artist_id:
        artist_name, comic_series = artist_id.split('^', 1)
    else:
        artist_name, comic_series = artist_id, None
    
    # 收集所有与该艺术家相关的图片
    images = []
    
    # 检查reference_image列
    for idx, row in df.iterrows():
        if row['reference_artist'] == artist_id:
            ref_img = row['reference_image']
            date = extract_date_from_path(ref_img)
            if date:
                images.append((ref_img, date))
    
    # 检查选项列
    option_columns = ['option_A', 'option_B', 'option_C', 'option_D']
    artist_columns = ['option_A_artist', 'option_B_artist', 'option_C_artist', 'option_D_artist']
    
    for idx, row in df.iterrows():
        for opt_col, artist_col in zip(option_columns, artist_columns):
            if row[artist_col] == artist_id:
                opt_img = row[opt_col]
                date = extract_date_from_path(opt_img)
                if date:
                    images.append((opt_img, date))
    
    # 去除重复项
    unique_images = list(set(images))
    
    # 按日期排序
    sorted_images = sorted(unique_images, key=lambda x: x[1])
    
    return sorted_images

def select_images_by_timeline(sorted_images):
    """根据时间线选择题目的图片"""
    if not sorted_images:
        return None, None
    
    # 如果只有一张图片，全部返回同一张
    if len(sorted_images) == 1:
        reference = sorted_images[0]
        return reference, [reference] * 5
    
    # 如果有2张图片，最早和最晚
    if len(sorted_images) == 2:
        reference = sorted_images[0]  # 用第一张作为参考
        options = [
            sorted_images[0],  # A: 最早
            sorted_images[0],  # B: 参考和最早的中间（没有中间，用参考）
            sorted_images[0],  # C: 最近的（没有，用参考）
            sorted_images[1],  # D: 参考和最晚的中间（只有最晚）
            sorted_images[1]   # E: 最晚
        ]
        return reference, options
    
    # 正常情况: 找到中间的图片作为reference
    mid_index = len(sorted_images) // 2
    reference = sorted_images[mid_index]
    
    # 找到其他选项
    earliest = sorted_images[0]  # A: 最早
    
    # B: 参考和最早的中间
    early_mid_index = mid_index // 2
    early_mid = sorted_images[early_mid_index]
    
    # C: 最近的（参考之前或之后的一张）
    if mid_index > 0:
        nearest = sorted_images[mid_index - 1]
    elif mid_index + 1 < len(sorted_images):
        nearest = sorted_images[mid_index + 1]
    else:
        nearest = reference  # 如果没有其他图片，使用参考
    
    # D: 参考和最晚的中间
    late_mid_index = (mid_index + len(sorted_images) - 1) // 2
    if late_mid_index == mid_index:
        late_mid_index = mid_index + 1
    if late_mid_index < len(sorted_images):
        late_mid = sorted_images[late_mid_index]
    else:
        late_mid = sorted_images[-1]  # 如果超出范围，使用最后一张
    
    # E: 最晚
    latest = sorted_images[-1]
    
    options = [earliest, early_mid, nearest, late_mid, latest]
    
    return reference, options

def generate_timeline_questions(input_csv, output_csv):
    """
    为每个艺术家生成基于时间线的问题
    
    Args:
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径
    """
    print(f"正在读取 {input_csv}...")
    
    # 读取CSV文件
    df = pd.read_csv(input_csv)
    
    # 获取所有唯一的艺术家ID
    all_artists = set()
    all_artists.update(df['reference_artist'].unique())
    for col in ['option_A_artist', 'option_B_artist', 'option_C_artist', 'option_D_artist']:
        all_artists.update(df[col].unique())
    
    # 过滤掉None值
    all_artists = {artist for artist in all_artists if artist is not None and isinstance(artist, str)}
    
    # 为每个艺术家创建一个题目
    results = []
    
    for artist_id in tqdm(all_artists):
        # print(f"正在处理艺术家: {artist_id}")
        
        # 获取该艺术家的所有图片及其日期
        sorted_images = get_artist_images_with_dates(df, artist_id)
        
        # 如果没有足够的图片，跳过
        if len(sorted_images) < 2:
            print(f"  警告: 艺术家 {artist_id} 没有足够的图片 ({len(sorted_images)}), 跳过")
            continue
        
        # 选择图片
        reference, options = select_images_by_timeline(sorted_images)
        
        if reference is None or options is None:
            print(f"  警告: 艺术家 {artist_id} 无法选择图片, 跳过")
            continue
        
        # 创建题目记录
        question = {
            'id': artist_id,
            'reference_artist': artist_id,
            'reference_image': reference[0],
            'reference_date': reference[1].strftime('%Y-%m-%d'),
            'option_A': options[0][0],
            'option_A_date': options[0][1].strftime('%Y-%m-%d'),
            'option_B': options[1][0],
            'option_B_date': options[1][1].strftime('%Y-%m-%d'),
            'option_C': options[2][0],
            'option_C_date': options[2][1].strftime('%Y-%m-%d'),
            'option_D': options[3][0],
            'option_D_date': options[3][1].strftime('%Y-%m-%d'),
            'option_E': options[4][0],
            'option_E_date': options[4][1].strftime('%Y-%m-%d'),
        }
        
        results.append(question)
    
    # 创建输出DataFrame
    output_df = pd.DataFrame(results)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存结果
    output_df.to_csv(output_csv, index=False)
    print(f"完成! 共生成 {len(results)} 个题目.")
    print(f"结果已保存至 {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='为每个艺术家生成基于时间线的问题')
    parser.add_argument('--input-csv', required=True, help='输入CSV文件路径')
    parser.add_argument('--output-csv', required=True, help='输出CSV文件路径')
    args = parser.parse_args()
    
    generate_timeline_questions(args.input_csv, args.output_csv)

if __name__ == '__main__':
    main()