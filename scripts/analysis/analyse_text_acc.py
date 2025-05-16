#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from utils.parse_jsonString import parse_probabilities

def analyze_text_count_group_equal_width(df, text_count_column, bins=10, min_samples=0):
    """
    分析特定文字数量列与正确率的关系，使用字数区间均匀的分组方法
    
    Args:
        df: 包含文字数量和正确性数据的DataFrame
        text_count_column: 文字数量列名
        bins: 分组数量
        min_samples: 每组最少样本数
    
    Returns:
        list: 包含每个文字数量组的统计信息
        array: 分组边界
    """
    # 确保文字数量列存在
    if text_count_column not in df.columns:
        print(f"警告: 列 '{text_count_column}' 不存在")
        return None, None
    
    # 获取文字数量范围
    min_count = df[text_count_column].min()
    max_count = df[text_count_column].max()
    
    if min_count == max_count:
        print(f"警告: 列 '{text_count_column}' 的所有值都相同 ({min_count})")
        return None, None
    
    # 创建等宽的分组边界
    bin_edges = np.linspace(min_count, max_count, bins + 1)
    
    # 将数据分配到对应的组中
    results = []
    
    for i in range(len(bin_edges) - 1):
        start_value = bin_edges[i]
        end_value = bin_edges[i + 1]
        
        # 获取当前组的数据
        if i < len(bin_edges) - 2:
            bin_df = df[(df[text_count_column] >= start_value) & (df[text_count_column] < end_value)]
        else:
            # 对于最后一个组，包含右边界
            bin_df = df[(df[text_count_column] >= start_value) & (df[text_count_column] <= end_value)]
        
        if len(bin_df) < min_samples:
            continue
        
        # 计算正确率
        correct = bin_df['is_correct'].sum()
        accuracy = correct / len(bin_df) if len(bin_df) > 0 else 0
        
        results.append({
            'bin_index': i,
            'bin_start': start_value,
            'bin_end': end_value,
            'samples': len(bin_df),
            'correct': correct,
            'accuracy': accuracy
        })
    
    if not results:
        print(f"警告: 列 '{text_count_column}' 的所有组样本数都少于 {min_samples}")
        return None, None
    
    return results, bin_edges

def analyze_text_count_group(df, text_count_column, bins=10, min_samples=0):
    """
    分析特定文字数量列与正确率的关系，使用样本数量均匀的分组方法
    
    Args:
        df: 包含文字数量和正确性数据的DataFrame
        text_count_column: 文字数量列名
        bins: 分组数量
        min_samples: 每组最少样本数
    
    Returns:
        list: 包含每个文字数量组的统计信息
        array: 分组边界
    """
    # 确保文字数量列存在
    if text_count_column not in df.columns:
        print(f"警告: 列 '{text_count_column}' 不存在")
        return None, None
    
    # 获取文字数量范围
    min_count = df[text_count_column].min()
    max_count = df[text_count_column].max()
    
    if min_count == max_count:
        print(f"警告: 列 '{text_count_column}' 的所有值都相同 ({min_count})")
        return None, None
    
    # 按文字数量排序
    sorted_df = df.sort_values(by=text_count_column)
    
    # 计算每个组应该包含的样本数
    total_samples = len(sorted_df)
    samples_per_bin = total_samples // bins
    
    # 创建分组边界
    bin_edges = []
    results = []
    
    # 添加第一个边界
    bin_edges.append(min_count)
    
    # 为每个组创建边界和统计结果
    for i in range(bins):
        # 计算当前组的起始和结束索引
        start_idx = i * samples_per_bin
        end_idx = (i + 1) * samples_per_bin if i < bins - 1 else total_samples
        
        # 获取当前组的数据
        if end_idx <= start_idx:
            continue
        
        bin_df = sorted_df.iloc[start_idx:end_idx]
        
        if len(bin_df) < min_samples:
            continue
        
        # 获取当前组的文字数量范围
        if i < bins - 1:
            end_value = sorted_df.iloc[end_idx][text_count_column]
            bin_edges.append(end_value)
        else:
            end_value = max_count
            bin_edges.append(max_count + 0.001)  # 确保最后一个值被包含
        
        start_value = bin_edges[i]
        
        # 计算正确率
        correct = bin_df['is_correct'].sum()
        accuracy = correct / len(bin_df) if len(bin_df) > 0 else 0
        
        results.append({
            'bin_index': i,
            'bin_start': start_value,
            'bin_end': end_value,
            'samples': len(bin_df),
            'correct': correct,
            'accuracy': accuracy
        })
    
    if not results:
        print(f"警告: 列 '{text_count_column}' 的所有组样本数都少于 {min_samples}")
        return None, None
    
    return results, np.array(bin_edges)

def process_results_file(results_csv, text_count_csv):
    """
    处理实验结果文件，提取正确率数据
    
    Args:
        results_csv: 包含实验结果的CSV文件路径
        text_count_csv: 包含文字数量数据的CSV文件路径
    
    Returns:
        DataFrame: 合并后的数据框
    """
    print(f"正在读取实验结果文件: {results_csv}")
    results_df = pd.read_csv(results_csv)
    
    print(f"正在读取文字数量文件: {text_count_csv}")
    text_count_df = pd.read_csv(text_count_csv)
    
    # 检查文本计数文件中必要的列是否存在
    required_columns = ['id', 'reference_artist', 'total_text_count', 'ref_text_count', 'gt_text_count']
    missing_columns = [col for col in required_columns if col not in text_count_df.columns]
    if missing_columns:
        print(f"错误: 文字数量CSV文件缺少必要的列: {', '.join(missing_columns)}")
        return None
    
    # 创建用于连接的键（id + reference_artist）
    if 'id' in results_df.columns and 'reference_artist' in results_df.columns:
        results_df['join_key'] = results_df['id'] + '|' + results_df['reference_artist']
        text_count_df['join_key'] = text_count_df['id'] + '|' + text_count_df['reference_artist']
    else:
        print("错误: 实验结果CSV文件中缺少'id'或'reference_artist'列")
        return None
    
    # 确定预测是否正确（从response中提取预测结果）
    print("解析响应数据确定预测正确性...")
    results_df['is_correct'] = False
    
    # 确定response列名
    response_col = None
    for col in ['response', 'llm_response']:
        if col in results_df.columns:
            response_col = col
            break
    
    if response_col:
        for idx, row in results_df.iterrows():
            try:
                if pd.notna(row[response_col]):
                    probabilities = parse_probabilities(row[response_col])
                    if probabilities:
                        predicted_option = max(probabilities.items(), key=lambda x: x[1])[0]
                        results_df.at[idx, 'is_correct'] = (predicted_option == row['ground_truth'])
            except Exception as e:
                print(f"解析行 {idx} 时出错: {e}")
    else:
        print("警告: 找不到响应列，无法确定预测是否正确")
        return None
    
    # 将文字数量数据合并到结果数据中
    print("合并文字数量数据...")
    merged_df = pd.merge(results_df, 
                         text_count_df[['join_key', 'total_text_count', 'ref_text_count', 'gt_text_count']], 
                         on='join_key', 
                         how='left')
    
    # 检查合并后的数据
    missing_text_count = merged_df['total_text_count'].isnull().sum()
    if missing_text_count > 0:
        print(f"警告: {missing_text_count}/{len(merged_df)} 行缺少文字数量数据")
    
    # 只保留有文字数量数据的行
    merged_df = merged_df.dropna(subset=['total_text_count'])
    print(f"合并后有效数据: {len(merged_df)} 行")
    
    # 计算参考图片与答案图片文字数比例
    merged_df['text_ratio'] = merged_df.apply(lambda x: x['ref_text_count'] / max(1, x['gt_text_count']), axis=1)
    
    return merged_df

def batch_analyze_text_acc(main_result_csv, text_count_comics_csv, text_count_political_csv, output_dir=None, bins=10, min_samples=0):
    """
    根据main_result.csv中提供的实验结果列表，批量执行文本计数与准确率分析
    
    Args:
        main_result_csv: 包含实验结果文件路径的CSV文件
        text_count_comics_csv: 漫画任务的文本计数CSV文件路径
        text_count_political_csv: 政治漫画任务的文本计数CSV文件路径
        output_dir: 输出目录，默认为main_result_csv所在目录
        bins: 文字数量分组的数量
        min_samples: 每组最少样本数
    """
    print(f"读取主实验结果文件: {main_result_csv}")
    main_df = pd.read_csv(main_result_csv)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(main_result_csv), "text_count_analysis")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 检查必要的列是否存在
    required_columns = ['model', 'path', 'task']
    missing_columns = [col for col in required_columns if col not in main_df.columns]
    if missing_columns:
        print(f"错误: 主实验结果CSV文件缺少必要的列: {', '.join(missing_columns)}")
        return
    
    # 收集所有数据
    all_comics_data = []
    all_political_data = []
    model_data = {}
    
    # 按任务类型分组处理
    task_groups = main_df.groupby('task')
    
    # 处理每个任务类型，收集数据
    for task, group in task_groups:
        print(f"\n{'='*50}")
        print(f"处理任务: {task}")
        
        # 选择对应的文本计数文件
        if task.lower() == 'comics':
            text_count_csv = text_count_comics_csv
        elif task.lower() == 'political':
            text_count_csv = text_count_political_csv
        else:
            print(f"警告: 未知任务类型 {task}，跳过")
            continue
        
        if not os.path.exists(text_count_csv):
            print(f"错误: 文本计数文件不存在: {text_count_csv}")
            continue
        
        print(f"使用文本计数文件: {text_count_csv}")
        
        # 处理该任务下的每个实验
        for idx, row in group.iterrows():
            model = row['model']
            results_csv = row['path']
            
            print(f"\n{'-'*40}")
            print(f"处理模型: {model}")
            print(f"结果文件: {results_csv}")
            
            if not os.path.exists(results_csv):
                print(f"错误: 实验结果文件不存在: {results_csv}")
                continue
            
            # 处理结果文件
            merged_df = process_results_file(results_csv, text_count_csv)
            
            if merged_df is not None:
                # 添加模型和任务标识
                merged_df['model'] = model
                merged_df['task'] = task
                
                # 收集数据
                if task.lower() == 'comics':
                    all_comics_data.append(merged_df)
                elif task.lower() == 'political':
                    all_political_data.append(merged_df)
                
                # 按模型收集数据
                if model not in model_data:
                    model_data[model] = {'comics': None, 'political': None}
                
                if task.lower() == 'comics':
                    model_data[model]['comics'] = merged_df
                elif task.lower() == 'political':
                    model_data[model]['political'] = merged_df
    
    # 合并所有数据
    comics_df = pd.concat(all_comics_data) if all_comics_data else None
    political_df = pd.concat(all_political_data) if all_political_data else None
    combined_df = pd.concat([comics_df, political_df]) if comics_df is not None and political_df is not None else None
    
    # 按分组分析数据
    results_rows = []
    
    # 记录分组边界
    comics_bins = None
    political_bins = None
    combined_bins = None
    count_name = 'gt_text_count'
    # 对每个模型进行分析
    for model in model_data:
        print(f"\n分析模型: {model}")
        
        # Comics数据
        if model_data[model]['comics'] is not None:
            comics_model_df = model_data[model]['comics']
            total_count_stats, bin_edges = analyze_text_count_group(comics_model_df, count_name, bins, min_samples)
            
            if comics_bins is None:
                comics_bins = bin_edges
            
            if total_count_stats is not None:
                for stat in total_count_stats:
                    results_rows.append({
                        'model': model,
                        'data_type': 'comics',
                        'bin_start': stat['bin_start'],
                        'bin_end': stat['bin_end'],
                        'samples': stat['samples'],
                        'correct': stat['correct'],
                        'accuracy': stat['accuracy']
                    })
        
        # Political数据
        if model_data[model]['political'] is not None:
            political_model_df = model_data[model]['political']
            total_count_stats, bin_edges = analyze_text_count_group(political_model_df, count_name, bins, min_samples)
            
            if political_bins is None:
                political_bins = bin_edges
            
            if total_count_stats is not None:
                for stat in total_count_stats:
                    results_rows.append({
                        'model': model,
                        'data_type': 'political',
                        'bin_start': stat['bin_start'],
                        'bin_end': stat['bin_end'],
                        'samples': stat['samples'],
                        'correct': stat['correct'],
                        'accuracy': stat['accuracy']
                    })
        
        # 合并数据
        if model_data[model]['comics'] is not None and model_data[model]['political'] is not None:
            combined_model_df = pd.concat([model_data[model]['comics'], model_data[model]['political']])
            total_count_stats, bin_edges = analyze_text_count_group(combined_model_df, 'gt_text_count', 3*bins, min_samples)
            # total_count_stats, bin_edges = analyze_text_count_group_equal_width(combined_model_df, count_name, bins, min_samples)
            
            if combined_bins is None:
                combined_bins = bin_edges
            
            if total_count_stats is not None:
                for stat in total_count_stats:
                    results_rows.append({
                        'model': model,
                        'data_type': 'combined',
                        'bin_start': stat['bin_start'],
                        'bin_end': stat['bin_end'],
                        'samples': stat['samples'],
                        'correct': stat['correct'],
                        'accuracy': stat['accuracy']
                    })
    
    # 创建结果数据框
    results_df = pd.DataFrame(results_rows)
    results_file = os.path.join(output_dir, "text_count_analysis_all.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\n分析结果已保存至: {results_file}")
    
    # 输出分组范围
    print("\n分组范围:")
    if comics_bins is not None:
        print("\nComics数据分组:")
        for i in range(len(comics_bins) - 1):
            print(f"  组 {i+1}: {comics_bins[i]:.1f} - {comics_bins[i+1]:.1f}")
    
    if political_bins is not None:
        print("\nPolitical数据分组:")
        for i in range(len(political_bins) - 1):
            print(f"  组 {i+1}: {political_bins[i]:.1f} - {political_bins[i+1]:.1f}")
    
    if combined_bins is not None:
        print("\n合并数据分组:")
        for i in range(len(combined_bins) - 1):
            print(f"  组 {i+1}: {combined_bins[i]:.1f} - {combined_bins[i+1]:.1f}")

def main():
    parser = argparse.ArgumentParser(description="批量分析文本计数与准确率关系")
    parser.add_argument("--main-csv", required=True, help="包含实验结果文件路径的主CSV文件")
    parser.add_argument("--comics-text-count", required=True, help="漫画任务的文本计数CSV文件路径")
    parser.add_argument("--political-text-count", required=True, help="政治漫画任务的文本计数CSV文件路径")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    parser.add_argument("--bins", type=int, default=10, help="文字数量分组的数量")
    parser.add_argument("--min-samples", type=int, default=0, help="每组最少样本数")
    args = parser.parse_args()
    
    batch_analyze_text_acc(
        args.main_csv,
        args.comics_text_count,
        args.political_text_count,
        args.output_dir,
        args.bins,
        args.min_samples
    )

if __name__ == "__main__":
    main()