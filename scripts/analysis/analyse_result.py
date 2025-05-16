import pandas as pd
import json
import os
import sys
from transformers import AutoTokenizer
import numpy as np
sys.path.append(os.getcwd())
from utils.parse_jsonString import parse_probabilities  
import re
def extract_description(text):
    # 使用正则表达式匹配"description":"和"options"之间的内容
    text = text.replace('\n','')
    pattern = r'"reasoning_content":"(.*?)"'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return "No description found"

def calculate_rms_calibration_error(predictions, ground_truths, min_samples_per_bin=30):
    """
    使用自适应分箱计算RMS校准误差
    
    Args:
        predictions: 预测概率列表，每个元素是一个字典 {选项: 概率}
        ground_truths: 真实标签列表
        min_samples_per_bin: 每个bin的最小样本数，用于自适应分箱
        
    Returns:
        float: RMS校准误差
    """
    if not predictions or len(predictions) != len(ground_truths):
        return None
    
    # 提取每个预测的最高概率和是否正确
    confidences = []
    accuracies = []
    
    for pred, gt in zip(predictions, ground_truths):
        # 归一化预测概率
        if not sum(pred.values()) == 0:
            total = sum(pred.values())
            pred = {k: v/total for k, v in pred.items()}
        
        if gt in pred:
            # 获取最高概率及其对应的选项
            max_prob_option = max(pred.items(), key=lambda x: x[1])
            confidence = max_prob_option[1]
            is_correct = float(max_prob_option[0] == gt)
            
            confidences.append(confidence)
            accuracies.append(is_correct)
    
    if not confidences:
        return None
    
    # 将预测概率和准确性转换为numpy数组
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    # 按置信度排序
    sort_indices = np.argsort(confidences)
    confidences = confidences[sort_indices]
    accuracies = accuracies[sort_indices]
    
    # 自适应分箱
    n_samples = len(confidences)
    print(f"n_samples: {n_samples}")
    n_bins = max(n_samples // min_samples_per_bin, 1)  # 确保至少有一个bin
    
    squared_errors = []
    current_pos = 0
    
    while current_pos < n_samples:
        # 确定当前bin的结束位置
        end_pos = min(current_pos + min_samples_per_bin, n_samples)
        
        # 计算当前bin的平均置信度和准确率
        bin_confidences = confidences[current_pos:end_pos]
        bin_accuracies = accuracies[current_pos:end_pos]
        
        avg_confidence = np.mean(bin_confidences)
        avg_accuracy = np.mean(bin_accuracies)
        
        # 计算该bin的校准误差（平方）
        bin_error = (avg_confidence - avg_accuracy) ** 2
        # 根据bin中的样本数量加权
        bin_weight = (end_pos - current_pos) / n_samples
        squared_errors.append(bin_error * bin_weight)
        
        current_pos = end_pos
    
    # 计算加权RMS误差
    if squared_errors:
        rms = np.sqrt(np.sum(squared_errors))
        return rms
    else:
        return None

def calculate_ece(predictions, ground_truths, n_bins=15):
    """
    计算Expected Calibration Error (ECE)
    
    Args:
        predictions: 预测概率列表，每个元素是一个字典 {选项: 概率}
        ground_truths: 真实标签列表
        n_bins: bin的数量，默认为10
        
    Returns:
        float: ECE值
    """
    if not predictions or len(predictions) != len(ground_truths):
        return None
    
    # 提取每个预测的最高概率和是否正确
    confidences = []
    accuracies = []
    for pred, gt in zip(predictions, ground_truths):
        # 先做归一化
        if not sum(pred.values()) ==0:
            for key, value in pred.items():
                pred[key] = value / sum(pred.values())
        
        if not pred:  # 如果预测为空，跳过
            continue
            
        # 获取最高概率及其对应的选项
        max_prob_option = max(pred.items(), key=lambda x: x[1])
        confidence = max_prob_option[1]
        is_correct = float(max_prob_option[0] == gt)  # 转换为float而不是int
        
        confidences.append(confidence)
        accuracies.append(is_correct)
    
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    # 将概率值分到不同的bin中
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 找到落在这个bin中的预测
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            # 计算这个bin中的平均准确率和平均置信度
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            
            # 累加到ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def calculate_nll(predictions, ground_truths):
    """
    计算负对数似然（Negative Log-Likelihood）
    
    Args:
        predictions: 预测概率列表，每个元素是一个字典 {选项: 概率}
        ground_truths: 真实标签列表
        
    Returns:
        float: 平均NLL值
    """
    if not predictions or len(predictions) != len(ground_truths):
        return None
    
    nlls = []
    eps = 1e-15  # 防止log(0)
    
    for pred, gt in zip(predictions, ground_truths):
        # 归一化预测概率
        if not sum(pred.values()) == 0:
            total = sum(pred.values())
            pred = {k: v/total for k, v in pred.items()}
        
        # 获取真实标签的预测概率，并确保在有效范围内
        prob = pred.get(gt, eps)
        prob = max(min(prob, 1-eps), eps)  # 裁剪到[eps, 1-eps]
        
        # 计算单个样本的NLL
        nll = -np.log(prob)
        nlls.append(nll)
    
    if nlls:
        return np.mean(nlls)
    else:
        return None

def analyze_single_result(csv_path, print_result=False):
    """
    分析单个实验结果
    
    Args:
        csv_path: CSV文件路径
        print_result: 是否打印结果
    
    Returns:
        tuple: (总样本数, 正确数, top2正确数, 准确率, top2准确率, rms校准误差, 总token数, 平均每个样本token数)
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    total = len(df)
    correct = 0
    top2_correct = 0
    total_tokens = 0  # 添加总token计数
    
    # 存储所有预测和真实标签，用于计算校准误差
    all_predictions = []
    all_ground_truths = []
    
    # 遍历每一行
    for _, row in df.iterrows():
        try:
            # 累加total_tokens
            if 'total_tokens' in row:
                total_tokens += row['total_tokens']
                
            # 解析response
            probabilities = parse_probabilities(row['response'])
            if not probabilities:
                continue
                
            # 获取ground truth
            ground_truth = row['ground_truth']
            
            # 存储预测和真实标签
            all_predictions.append(probabilities)
            all_ground_truths.append(ground_truth)
            
            # 获取top1和top2预测
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            top1 = sorted_probs[0][0]
            top2 = [sorted_probs[0][0], sorted_probs[1][0]] if len(sorted_probs) > 1 else [sorted_probs[0][0]]
            
            # 计算正确率
            if top1 == ground_truth:
                correct += 1
            if ground_truth in top2:
                top2_correct += 1
                
        except Exception as e:
            print(f"处理行时出错: {e}")
            continue
    
    # 计算准确率
    accuracy = correct / total if total > 0 else 0
    top2_accuracy = top2_correct / total if total > 0 else 0
    
    # 计算RMS校准误差
    rms_error = calculate_rms_calibration_error(all_predictions, all_ground_truths)
    
    # 计算ECE
    ece = calculate_ece(all_predictions, all_ground_truths)
    
    # 计算NLL
    nll = calculate_nll(all_predictions, all_ground_truths)
    
    if print_result:
        print(f"总样本数: {total}")
        print(f"Top1正确数: {correct}")
        print(f"Top2正确数: {top2_correct}")
        print(f"Top1准确率: {accuracy:.8f}")
        print(f"Top2准确率: {top2_accuracy:.8f}")
        print(f"总Token数: {total_tokens}")
        print(f"平均每个样本Token数: {total_tokens/total:.2f}")
        if rms_error is not None:
            print(f"RMS校准误差: {rms_error:.8f}")
        if ece is not None:
            print(f"ECE: {ece:.8f}")
        if nll is not None:
            print(f"NLL: {nll:.8f}")
        print('='*50)
    
    return total, correct, top2_correct, accuracy, top2_accuracy, rms_error, ece, nll, total_tokens

def analyze_results(paths_csv,compare_csv):
    """
    分析多个实验结果，首先基于full_dataset进行筛选
    
    Args:
        paths_csv: 包含path列的CSV文件路径
    """
    # 读取full_dataset作为基准数据
    
    
    # 创建基准数据的唯一标识符集合
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # 读取包含路径的CSV文件
    paths_df = pd.read_csv(paths_csv)
    
    if 'path' not in paths_df.columns:
        print("错误: CSV文件中没有'path'列")
        return
    
    # 存储结果
    results = []
    
    # 分析每个CSV文件
    for idx, row in paths_df.iterrows():
        csv_path = row['path']
        task = row['task']
        print(task)
        if task == 'political':
            full_csv = compare_csv.replace('comics','political')
            print(full_csv)
            full_dataset = pd.read_csv(full_csv)
            valid_pairs = set(zip(full_dataset['id'].astype(str), full_dataset['reference_artist'].astype(str)))
            print(f"基准数据集中的样本数: {len(valid_pairs)}")
        else:
            full_dataset = pd.read_csv(compare_csv)
            valid_pairs = set(zip(full_dataset['id'].astype(str), full_dataset['reference_artist'].astype(str)))
            print(f"基准数据集中的样本数: {len(valid_pairs)}")
        # 检查文件是否存在
        if not os.path.exists(csv_path):
            print(f"警告: 文件不存在 - {csv_path}")
            continue
        
        print(f"\n正在分析: {csv_path}")
        
        try:
            # 读取当前实验的CSV文件
            current_df = pd.read_csv(csv_path)
            
            # 筛选出在full_dataset中存在的样本
            if 'id' in current_df.columns and 'reference_artist' in current_df.columns:
                # 创建当前数据的标识符对
                current_pairs = set(zip(current_df['id'].astype(str), current_df['reference_artist'].astype(str)))
                
                # 筛选匹配的样本
                current_df['pair'] = list(zip(current_df['id'].astype(str), current_df['reference_artist'].astype(str)))
                filtered_df = current_df[current_df['pair'].isin(valid_pairs)].copy()
                filtered_df = filtered_df.drop('pair', axis=1)
                
                print(f"原始样本数: {len(current_pairs)}")
                print(f"筛选后的样本数: {len(filtered_df)}")
                print(f"匹配率: {len(filtered_df)/len(current_pairs)*100:.2f}%")
                
                # 直接使用筛选后的DataFrame进行分析
                df_to_analyze = filtered_df
            else:
                print("警告: CSV文件中缺少'id'或'reference_artist'列，将使用全部数据")
                df_to_analyze = current_df
            
            # 分析数据
            total = len(df_to_analyze)
            correct = 0
            top2_correct = 0
            total_tokens = 0  # 添加总token计数
            total_prompt_tokens = 0
            total_completion_tokens = 0
            all_predictions = []
            all_ground_truths = []
            
            # 遍历每一行进行分析
            for _, row_data in df_to_analyze.iterrows():
                try:
                    # 累加total_tokens
                    if 'CoVR' in csv_path:
                        answer_content = row_data['options']
                        answer_content = extract_description(answer_content)
                        answer_content = tokenizer.encode(answer_content)
                        total_completion_tokens += len(answer_content)
                        parts = str(row_data['total_tokens']).split(';')
                        total_prompt_tokens += int(parts[0])+int(parts[2])
                        total_completion_tokens += int(parts[1])+int(parts[3])
                    else:
                        if 'total_tokens' in row_data:
                            if ';' in str(row_data['total_tokens']):
                                parts = str(row_data['total_tokens']).split(';')
                                prompt_tokens = int(parts[0])
                                completion_tokens = int(parts[1])
                                total_prompt_tokens += prompt_tokens
                                total_completion_tokens += completion_tokens
                            else:
                                total_prompt_tokens += int(row_data['total_tokens'])
                                total_completion_tokens += 0
                    # 解析response
                    probabilities = parse_probabilities(row_data['llm_response'])
                    if not probabilities:
                        continue
                        
                    # 获取ground truth
                    ground_truth = row_data['ground_truth']
                    
                    # 存储预测和真实标签
                    all_predictions.append(probabilities)
                    all_ground_truths.append(ground_truth)
                    
                    # 获取top1和top2预测
                    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                    top1 = sorted_probs[0][0]
                    top2 = [sorted_probs[0][0], sorted_probs[1][0]] if len(sorted_probs) > 1 else [sorted_probs[0][0]]
                    
                    # 计算正确率
                    if top1 == ground_truth:
                        correct += 1
                    if ground_truth in top2:
                        top2_correct += 1
                        
                except Exception as e:
                    print(f"处理行时出错: {e}")
                    continue
            
            # 计算指标
            accuracy = correct / total if total > 0 else 0
            top2_accuracy = top2_correct / total if total > 0 else 0
            rms_error = calculate_rms_calibration_error(all_predictions, all_ground_truths)
            
            # 计算ECE
            ece = calculate_ece(all_predictions, all_ground_truths)
            
            # 计算NLL
            nll = calculate_nll(all_predictions, all_ground_truths)
            
            # 获取实验名称（使用文件名或路径中的目录名）
            experiment_name = os.path.basename(os.path.dirname(csv_path))
            
            # 存储结果
            results.append({
                'experiment': experiment_name,
                'task': row['task'],
                # 'path': csv_path,
                'total': total,
                'correct': correct,
                'top2_correct': top2_correct,
                'accuracy': accuracy,
                'top2_accuracy': top2_accuracy,
                'rms_error': rms_error,
                'ece': ece,  # 添加ECE
                'nll': nll,  # 添加NLL
                'total_tokens': total_tokens,  # 添加总token数
                'total_prompt_tokens': total_prompt_tokens,  # 添加总token数
                'total_completion_tokens': total_completion_tokens,  # 添加总token数
                'avg_tokens': total_tokens/total if total > 0 else 0  # 添加平均token数
            })
            
            # 打印单个结果
            print(f"{'='*20} {experiment_name} 结果 {'='*20}")
            print(f"总样本数: {total}")
            print(f"Top1正确数: {correct}")
            print(f"Top2正确数: {top2_correct}")
            print(f"Top1准确率: {accuracy:.8f}")
            print(f"Top2准确率: {top2_accuracy:.8f}")
            print(f"总Token数: {total_tokens}")  # 添加总token输出
            print(f"平均每个样本Token数: {total_tokens/total:.2f}")  # 添加平均token输出
            if rms_error is not None:
                print(f"RMS校准误差: {rms_error:.8f}")
            if ece is not None:
                print(f"ECE: {ece:.8f}")
            if nll is not None:
                print(f"NLL: {nll:.8f}")
            print('='*50)
            
        except Exception as e:
            print(f"分析文件时出错 {csv_path}: {str(e)}")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    
    # 按准确率排序
    if not results_df.empty:
        results_df = results_df.sort_values('accuracy', ascending=False)
        
        # 打印汇总结果
        print("\n\n" + "="*20 + " 汇总结果 " + "="*20)
        print(results_df[['experiment', 'total', 'correct', 'accuracy','ece', 'rms_error',  'nll', 'top2_accuracy','total_tokens', 'avg_tokens']].to_string(index=False))
        print("="*50)
        
        # 保存结果
        output_path = os.path.join(os.path.dirname(paths_csv), 'analysis_summary_reasoning.csv')
        results_df.to_csv(output_path, index=False)
        print(f"\n结果已保存到: {output_path}")
    
    return results_df

def main():
    import argparse
    parser = argparse.ArgumentParser(description="分析多个实验结果")
    parser.add_argument("--csv", type=str, required=True, help="包含path列的CSV文件路径")
    parser.add_argument("--compare", type=str, default=".full_dataset.csv", help="compared CSV文件路径")
    parser.add_argument("--single", action="store_true", help="是否只分析单个实验结果")
    args = parser.parse_args()
    if args.single:
        analyze_single_result(args.csv, True)
    else:
        analyze_results(args.csv, args.compare)

def test():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    answer_content = "I'm a good boy"
    answer_content = tokenizer.encode(answer_content)
    print(answer_content)

if __name__ == "__main__":
    main()
    # test()