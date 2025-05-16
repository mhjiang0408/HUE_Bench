import pandas as pd
import json
import os
from typing import Dict, List, Tuple
import os
import sys
import re
sys.path.append(os.getcwd())
import argparse
from utils.parse_jsonString import parse_json_string

def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    加载原始数据集
    
    Args:
        dataset_path: 数据集文件路径
    
    Returns:
        DataFrame: 包含journal和id的数据集
    """
    try:
        df = pd.read_csv(dataset_path)
        required_columns = ['id', 'reference_artist']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"数据集缺少必要的列: {required_columns}")
        return df
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        raise

def load_experiment_results(results_path: str) -> pd.DataFrame:
    """
    加载实验结果
    
    Args:
        results_path: 实验结果文件路径
    
    Returns:
        DataFrame: 包含实验结果的数据集
    """
    try:
        df = pd.read_csv(results_path)
        required_columns = ['id', 'reference_artist','options']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"实验结果缺少必要的列: {required_columns}")
        return df
    except Exception as e:
        print(f"加载实验结果时出错: {e}")
        raise

def extract_description(text):
    # 使用正则表达式匹配"description":"和"options"之间的内容
    text = text.replace('\n','')
    pattern = r'"description":"(.*?)"(?=,"options")'
    match = re.search(pattern, text)
    if "I'm sorry" in text:
        return 'delete'
    if match:
        return match.group(1)
    return "No description found"

def process_datasets(dataset_path: str, results_path: str, output_path: str):
    """
    处理数据集和实验结果，提取options描述，并删除description为'delete'的记录
    
    Args:
        dataset_path: 原始数据集路径
        results_path: 实验结果路径
        output_path: 输出文件路径
    """
    try:
        # 加载数据集
        dataset = load_dataset(dataset_path)
        print(f"成功加载数据集，共 {len(dataset)} 条记录")
        
        # 加载实验结果
        results = load_experiment_results(results_path)
        print(f"成功加载实验结果，共 {len(results)} 条记录")
        
        # 确保id列的类型一致
        dataset['id'] = dataset['id'].astype(str)
        results['reference_artist'] = results['reference_artist'].astype(str)
        
        # 合并数据集
        merged = pd.merge(
            dataset,
            results[['id', 'reference_artist','options']],
            left_on=['id', 'reference_artist'],
            right_on=['id', 'reference_artist'],
            how='left'
        )
        
        # 提取descriptions并创建临时列表存储所有数据
        data_list = []
        for _, row in merged.iterrows():
            if pd.isna(row['options']):
                description = "未找到匹配的实验结果"
            else:
                description = extract_description(row['options'])
                
            # 只添加不是'delete'的记录
            if description != 'delete':
                data_list.append({
                    'id': row['id'],
                    'reference_artist': row['reference_artist'],
                    'description': description
                })
            # data_list.append({
            #         'journal': row['journal'],
            #         'id': row['id'],
            #         'question': row['question'],
            #         'description': description
            #     })
        
        # 创建输出DataFrame
        output_df = pd.DataFrame(data_list)
        
        # 保存结果
        output_df.to_csv(output_path, index=False)
        print(f"结果已保存到: {output_path}")
        
        # 打印统计信息
        total_original = len(merged)
        total_after = len(output_df)
        print(f"原始记录数: {total_original}")
        print(f"处理后记录数: {total_after}")
        print(f"删除的记录数: {total_original - total_after}")
        
    except Exception as e:
        print(f"处理数据时出错: {e}")
        raise


if __name__ == "__main__":
    dataset_path = "./political_2025.csv"
    results_path = "./experiment/results/xx/results.csv"
    output_path = "./political_2025_description.csv"
    process_datasets(dataset_path, results_path, output_path)
