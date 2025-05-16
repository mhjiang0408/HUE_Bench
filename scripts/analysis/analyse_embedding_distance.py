import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import torch

from matplotlib import pyplot as plt

def process_results_file(results_csv, full_dataset_csv):
    """
    处理单个实验结果文件，合并ref_gt_distance
    
    Args:
        results_csv: 实验结果CSV文件路径
        full_dataset_csv: 包含ref_gt_distance的数据集CSV文件路径
    
    Returns:
        合并后的DataFrame，如果出错则返回None
    """
    print(f"  处理结果文件: {results_csv}")
    
    # 读取实验结果
    try:
        results_df = pd.read_csv(results_csv)
    except Exception as e:
        print(f"  错误: 无法读取实验结果文件: {e}")
        return None
    
    # 读取完整数据集
    try:
        full_df = pd.read_csv(full_dataset_csv)
        # 增加一列gp_option_min
        full_df['gp_option_min'] = full_df[['gt_option_distance1', 'gt_option_distance2', 'gt_option_distance3']].min(axis=1)
    except Exception as e:
        print(f"  错误: 无法读取完整数据集文件: {e}")
        return None
    
    # 检查必要的列
    if 'id' not in results_df.columns:
        print(f"  错误: 实验结果文件缺少必要的列: id")
        return None
    
    if 'id' not in full_df.columns or 'gp_option_min' not in full_df.columns:
        print(f"  错误: 完整数据集文件缺少必要的列: id 或 ref_gt_distance")
        return None
    
    # 合并数据
    # 创建连接键（支持id或id+reference_artist）
    join_keys = []
    if 'reference_artist' in results_df.columns and 'reference_artist' in full_df.columns:
        print("  使用id+reference_artist作为连接键")
        results_df['join_key'] = results_df['id'].astype(str) + '|' + results_df['reference_artist'].astype(str)
        full_df['join_key'] = full_df['id'].astype(str) + '|' + full_df['reference_artist'].astype(str)
        join_keys = ['join_key']
    else:
        print("  仅使用id作为连接键")
        results_df['join_key'] = results_df['id'].astype(str)
        full_df['join_key'] = full_df['id'].astype(str)
        join_keys = ['join_key']
    
    # 合并
    merged_df = pd.merge(results_df, full_df[join_keys + ['gp_option_min']], on=join_keys, how='left')
    
    # 检查合并后的数据
    null_count = merged_df['gp_option_min'].isnull().sum()
    if null_count > 0:
        print(f"  警告: 有{null_count}行数据未能匹配到ref_gt_distance (总行数: {len(merged_df)})")
    
    # 识别准确率列
    acc_col = None
    for c in ['accuracy', 'acc', 'is_correct']:
        if c in merged_df.columns:
            acc_col = c
            break
    
    if acc_col is None:
        print(f"  错误: 实验结果文件缺少准确率列(accuracy/acc/is_correct)")
        return None
    
    # 如果是is_correct，转为浮点数表示准确率
    if acc_col == 'is_correct':
        merged_df['accuracy'] = merged_df['is_correct'].astype(float)
    elif acc_col != 'accuracy':
        merged_df['accuracy'] = merged_df[acc_col]
    
    return merged_df

def analyze_distance_group(df, distance_column='ref_gt_distance', acc_column='accuracy', bins=10, min_samples=0):
    """
    按ref_gt_distance进行分组分析准确率，分组方式为按样本数量均匀分组
    
    Args:
        df: 包含ref_gt_distance和准确率的DataFrame
        distance_column: 距离列名
        acc_column: 准确率列名
        bins: 分组数量
        min_samples: 每组最少样本数
    
    Returns:
        分组统计结果DataFrame，如果出错则返回None
    """
    print(f"进行ref_gt_distance分组分析（按样本数量均匀分组）...")
    
    if distance_column not in df.columns or acc_column not in df.columns:
        print(f"  错误: 缺少必要列: {distance_column} 或 {acc_column}")
        return None
    
    # 去除缺失值
    valid_df = df.dropna(subset=[distance_column, acc_column])
    if len(valid_df) == 0:
        print(f"  错误: 没有有效数据进行分析")
        return None
    
    # 检查是否所有距离值都相同
    min_val = valid_df[distance_column].min()
    max_val = valid_df[distance_column].max()
    if min_val == max_val:
        print(f"  警告: {distance_column} 所有值都相同 ({min_val})")
        return pd.DataFrame({
            'bin_start': [min_val],
            'bin_end': [max_val],
            'samples': [len(valid_df)],
            'accuracy': [valid_df[acc_column].mean()]
        })
    
    # 按距离值排序
    sorted_df = valid_df.sort_values(by=distance_column)
    total_samples = len(sorted_df)
    
    # 计算每组理想样本数
    samples_per_bin = max(1, total_samples // bins)
    print(f"  样本总数: {total_samples}, 分组数: {bins}, 每组理想样本数: {samples_per_bin}")
    
    # 分组统计
    results = []
    start_idx = 0
    
    for bin_index in range(bins):
        # 计算当前组的结束索引
        remaining_bins = bins - bin_index
        remaining_samples = total_samples - start_idx
        if remaining_bins == 1:
            # 最后一组包含所有剩余样本
            end_idx = total_samples
        else:
            # 分配平均样本数
            end_idx = start_idx + remaining_samples // remaining_bins
        
        # 确保每组至少有min_samples个样本
        if end_idx - start_idx < min_samples:
            print(f"  跳过分组 {bin_index}, 样本数({end_idx - start_idx})小于最小要求({min_samples})")
            start_idx = end_idx
            continue
        
        # 获取当前组的数据
        bin_df = sorted_df.iloc[start_idx:end_idx]
        
        # 记录组的边界值和准确率
        bin_start = bin_df[distance_column].min()
        bin_end = bin_df[distance_column].max()
        bin_acc = bin_df[acc_column].mean()
        bin_samples = len(bin_df)
        
        results.append({
            'bin_index': bin_index,
            'bin_start': bin_start,
            'bin_end': bin_end,
            'samples': bin_samples,
            'accuracy': bin_acc
        })
        
        print(f"  分组 {bin_index}: [{bin_start:.4f}, {bin_end:.4f}], 样本数: {bin_samples}, 准确率: {bin_acc:.4f}")
        
        # 更新开始索引
        start_idx = end_idx
    
    if not results:
        print(f"  警告: 没有满足条件的分组")
        return None
    
    return pd.DataFrame(results)

def batch_analyze_ref_gt_distance(main_result_csv, output_dir=None, bins=10, min_samples=0):
    """
    根据main_result.csv中提供的实验结果列表，批量执行ref_gt_distance与准确率分析
    
    Args:
        main_result_csv: 包含实验结果文件路径的CSV文件
        full_dataset_comics_csv: 漫画任务的完整数据集CSV文件路径（包含ref_gt_distance）
        full_dataset_political_csv: 政治漫画任务的完整数据集CSV文件路径（包含ref_gt_distance）
        output_dir: 输出目录，默认为main_result_csv所在目录下的ref_gt_distance_analysis
        bins: 距离分组的数量
        min_samples: 每组最少样本数
    """
    print(f"读取主实验结果文件: {main_result_csv}")
    main_df = pd.read_csv(main_result_csv)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(main_result_csv), "gt_option_distance_analysis")
    
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
    all_data = []
    
    # 按任务类型分组处理
    task_groups = main_df.groupby('task')
    
    # 处理每个任务类型，收集数据
    for task, group in task_groups:
        print(f"\n{'='*50}")
        print(f"处理任务: {task}")
        
        # 选择对应的完整数据集文件
        if task.lower() == 'comics':
            full_dataset_csv = './Dataset/comics_2025_distance.csv'
        elif task.lower() == 'political':
            full_dataset_csv = './political_2025_distance.csv'
        else:
            print(f"警告: 未知任务类型 {task}，跳过")
            continue
        
        if not os.path.exists(full_dataset_csv):
            print(f"错误: 完整数据集文件不存在: {full_dataset_csv}")
            continue
        
        print(f"使用完整数据集文件: {full_dataset_csv}")
        
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
            merged_df = process_results_file(results_csv, full_dataset_csv)
            
            if merged_df is not None:
                # 添加模型和任务标识
                merged_df['model'] = model
                merged_df['task'] = task
                
                # 分组分析
                stats_df = analyze_distance_group(merged_df, bins=bins, min_samples=min_samples, distance_column='gp_option_min')
                
                if stats_df is not None:
                    # 添加模型和任务标识
                    stats_df['model'] = model
                    stats_df['task'] = task
                    
                    # 保存单个实验的结果
                    # output_file = os.path.join(output_dir, f"{model}_{task}_ref_gt_distance_group.csv")
                    # stats_df.to_csv(output_file, index=False)
                    # print(f"  结果已保存至: {output_file}")
                    
                    # 收集数据
                    if task.lower() == 'comics':
                        all_comics_data.append(stats_df)
                    elif task.lower() == 'political':
                        all_political_data.append(stats_df)
                    all_data.append(stats_df)
    
    # 合并并保存所有实验的结果
    # if all_comics_data:
    #     comics_df = pd.concat(all_comics_data, ignore_index=True)
    #     comics_file = os.path.join(output_dir, "all_comics_ref_gt_distance_group.csv")
    #     comics_df.to_csv(comics_file, index=False)
    #     print(f"\n所有漫画实验的分组分析已合并保存至: {comics_file}")
    
    # if all_political_data:
    #     political_df = pd.concat(all_political_data, ignore_index=True)
    #     political_file = os.path.join(output_dir, "all_political_ref_gt_distance_group.csv")
    #     political_df.to_csv(political_file, index=False)
    #     print(f"所有政治漫画实验的分组分析已合并保存至: {political_file}")
    
    if all_data:
        all_df = pd.concat(all_data, ignore_index=True)
        all_file = os.path.join(output_dir, "all_experiments_ref_gt_distance_group.csv")
        all_df.to_csv(all_file, index=False)
        print(f"所有实验的分组分析已合并保存至: {all_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量分析ref_gt_distance分组与准确率关系")
    parser.add_argument("--main-csv", required=True, help="包含各实验结果文件路径的主CSV文件")
    parser.add_argument("--bins", type=int, default=10, help="分组数量")
    parser.add_argument("--min-samples", type=int, default=0, help="每组最少样本数")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    args = parser.parse_args()
    
    batch_analyze_ref_gt_distance(
        args.main_csv, 
        args.output_dir,
        args.bins,
        args.min_samples
    )







def get_embedding_path(img_path):
    """
    根据图片路径返回embedding路径（.pt文件）
    """
    if img_path.startswith('./gocomics_downloads_political/'):
        emb_path = img_path.replace('./gocomics_downloads_political/', './Dataset/Political_Embeddings/')
    elif img_path.startswith('./gocomics_downloads/'):
        emb_path = img_path.replace('./gocomics_downloads/', './Dataset/Comics_Embeddings/')
    else:
        raise ValueError(f"Unknown image path prefix: {img_path}")
    emb_file = os.path.splitext(emb_path)[0] + '.pt'
    if os.path.exists(emb_file):
        return emb_file
    else:
        raise FileNotFoundError(f"Embedding not found for {img_path} (tried {emb_file})")

def load_embedding_pt(emb_path):
    """
    加载.pt embedding向量（假设为1D torch tensor或numpy array）
    """
    emb = torch.load(emb_path, map_location='cpu')
    if isinstance(emb, torch.Tensor):
        return emb.cpu().numpy().flatten()
    elif isinstance(emb, np.ndarray):
        return emb.flatten()
    elif isinstance(emb, dict) and 'embedding' in emb:
        # 支持dict格式
        return np.array(emb['embedding']).flatten()
    else:
        raise ValueError(f"Unknown embedding format in {emb_path}")

def cosine_distance(a, b):
    a = a.flatten()
    b = b.flatten()
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def process_csv(csv_path, output_csv):
    df = pd.read_csv(csv_path)
    # 新增列
    df['ref_gt_distance'] = np.nan
    distractor_cols = ['gt_option_distance1', 'gt_option_distance2', 'gt_option_distance3']
    for col in distractor_cols:
        df[col] = np.nan

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            ref_img = row['reference_image']
            gt_img = row['option_' + row['ground_truth']]
            ref_emb = load_embedding_pt(get_embedding_path(ref_img))
            gt_emb = load_embedding_pt(get_embedding_path(gt_img))
            # 1. gt与ref距离
            df.at[idx, 'ref_gt_distance'] = cosine_distance(ref_emb, gt_emb)
            # 2. gt与所有distractor距离
            distractor_opts = [opt for opt in ['A', 'B', 'C', 'D'] if opt != row['ground_truth']]
            for i, opt in enumerate(distractor_opts):
                opt_img = row[f'option_{opt}']
                opt_emb = load_embedding_pt(get_embedding_path(opt_img))
                df.at[idx, f'gt_option_distance{i+1}'] = cosine_distance(gt_emb, opt_emb)
            # for opt in ['A', 'B', 'C', 'D']:
            #     opt_img = row[f'option_{opt}']
            #     opt_emb = load_embedding_pt(get_embedding_path(opt_img))
            #     df.at[idx, f'gt_option_{opt}_distance'] = cosine_distance(gt_emb, opt_emb)
        except Exception as e:
            print(f"Error at row {idx}: {e}")

    df.to_csv(output_csv, index=False)
    print(f"Saved with distances: {output_csv}")


# compute embedding distance script
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Compute embedding distances (.pt) and save to CSV")
#     parser.add_argument('--csv', required=True, help='Input CSV file')
#     parser.add_argument('--output', required=True, help='Output CSV file')
#     args = parser.parse_args()
#     process_csv(args.csv, args.output)

