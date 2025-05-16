import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from matplotlib.ticker import PercentFormatter

# 定义固定的模型颜色映射
MODEL_COLORS = {
    'Qwen2.5-VL-7B': '#0066FF',  # 橙红色
    'Qwen-VL-Max': '#FF0033',  # 天蓝色
    'Step-1V-8k': '#FF6600',  # 紫色
    'Step-1o-Turbo-Vision': '#FFCC00',  # 绿色
    'Step-R1-V-mini': '#33CC33',  # 黄色
    'Gemini-2.5-flash': '#00CCFF',  # 粉色
    'GPT-4o': '#0099FF',  # 青色
    'GPT-4.1': '#9966FF',  # 红色
    # 可以根据需要添加更多模型的颜色
}

def get_model_color(model):
    """根据模型名称返回对应的颜色"""
    # 尝试精确匹配
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    
    # 尝试部分匹配
    for model_key in MODEL_COLORS:
        if model_key in model or model in model_key:
            return MODEL_COLORS[model_key]
    
    # 如果没有匹配，返回随机颜色
    import random
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def plot_distance_accuracy_histogram(csv_file, output_dir=None, plot_title=None):
    """
    根据分组的ref_gt_distance数据绘制准确率直方图
    
    Args:
        csv_file: 包含分组数据的CSV文件路径
        output_dir: 输出目录，默认为CSV文件所在目录
        plot_title: 图表标题，默认根据CSV文件名生成
    """
    print(f"读取CSV文件: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"错误: 无法读取CSV文件: {e}")
        return
    
    # 检查必要的列
    required_columns = ['bin_start', 'bin_end', 'samples', 'accuracy', 'model', 'task']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"错误: CSV文件缺少必要的列: {', '.join(missing_columns)}")
        return
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(csv_file)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 获取任务列表
    unique_tasks = ['comics', 'political']
    
    # 为每个任务(comics和political)绘制单独的图表
    for task_name in unique_tasks:
        # 过滤指定任务的数据
        task_df = df[df['task'] == task_name]
        if len(task_df) == 0:
            print(f"警告: CSV文件中没有任务 '{task_name}' 的数据")
            continue
        
        # 设置图表样式 - 修改为无网格线样式
        sns.set_style("white")  # 使用white样式替代whitegrid
        plt.figure(figsize=(15, 8))
        
        # 设置标题
        task_plot_title = plot_title
        if task_plot_title is None:
            csv_basename = os.path.basename(csv_file)
            task_plot_title = f"Embedding Distance vs. Accuracy - {task_name.capitalize()} Task"
        else:
            task_plot_title = f"{plot_title} - {task_name.capitalize()}"
        
        # plt.suptitle(task_plot_title, fontsize=16)
        
        # 获取所有模型
        unique_models = task_df['model'].unique()
        
        # 为每个模型绘制柱状图
        for model in unique_models:
            if model == 'Qwen2.5-VL-7B':
                continue
            model_df = task_df[task_df['model'] == model]
            if len(model_df) == 0:
                continue
            
            # 排序确保按bin_index或bin_start顺序显示
            if 'bin_index' in model_df.columns:
                model_df = model_df.sort_values('bin_index')
            else:
                model_df = model_df.sort_values('bin_start')
            
            # 创建X轴标签（距离范围）
            x_labels = [f"{start:.2f}-{end:.2f}" for start, end in zip(model_df['bin_start'], model_df['bin_end'])]
            x_pos = np.arange(len(x_labels)) + (list(unique_models).index(model) - len(unique_models)/2 + 0.5) * 0.8/len(unique_models)
            width = 0.8 / len(unique_models)  # 调整宽度，使得不同模型的柱子并排显示
            
            # 获取模型对应的颜色
            model_color = get_model_color(model)
            
            # 绘制柱状图，使用固定颜色
            bars = plt.bar(x_pos, model_df['accuracy'], width=width, label=model, alpha=0.8, color=model_color)
            
            # # 在柱上添加数值标签
            # for j, bar in enumerate(bars):
            #     height = bar.get_height()
            #     plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            #             f"{height:.2%}", ha='center', va='bottom', rotation=0, fontsize=8)
        
        # 设置X轴刻度
        plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45, ha='right', fontsize=9)
        
        # 设置Y轴范围和格式
        if task_name == 'comics':
            plt.ylim(0, 0.9)  # 设置Y轴上限
        else:
            plt.ylim(0, 0.4)  # 设置Y轴上限
        
        # 修改Y轴刻度，显示更简洁的百分比格式
        ax = plt.gca()
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        
        # 设置Y轴刻度为10%的间隔
        if task_name == 'comics':
            yticks = np.arange(0, 0.91, 0.1)  # 0% 到 90%，步长10%
        else:
            yticks = np.arange(0, 0.41, 0.1)  # 0% 到 40%，步长10%
        
        plt.yticks(yticks, [f"{int(y*100)}%" for y in yticks])  # 使用整数百分比标签
        
        # 添加标题和标签
        # plt.xlabel("Embedding Distance Range", fontsize=12)
        plt.ylabel("Accuracy(%)", fontsize=12)
        
        # 添加图例 - 修改为多行排列在图表上方
        plt.legend(title="Models", fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.3), 
                  ncol=4, frameon=False)  # 将ncol设为3，使图例按3列排列
        print(f"task: {task_name}")
        
        # 明确关闭所有网格线
        plt.grid(False)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(output_dir, f"{task_name}_distance_accuracy_histogram.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"{task_name}任务的图表已保存至: {output_file}")
        
        # 关闭图表
        plt.close()

def plot_models_comparison(csv_file, output_dir=None, by_task=True):
    """
    绘制模型比较图表，按任务或距离分组进行对比
    
    Args:
        csv_file: 包含分组数据的CSV文件路径
        output_dir: 输出目录，默认为CSV文件所在目录
        by_task: 是否按任务分组（True）或按距离分组（False）
    """
    print(f"读取CSV文件进行模型比较: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"错误: 无法读取CSV文件: {e}")
        return
    
    # 检查必要的列
    required_columns = ['bin_start', 'bin_end', 'samples', 'accuracy', 'model', 'task']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"错误: CSV文件缺少必要的列: {', '.join(missing_columns)}")
        return
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(csv_file)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 设置图表样式
    sns.set_style("white")  # 使用white样式替代whitegrid
    
    if by_task:
        # 按任务分组比较模型
        unique_tasks = df['task'].unique()
        
        for task in unique_tasks:
            task_df = df[df['task'] == task]
            
            plt.figure(figsize=(14, 8))
            
            # 创建折线图
            ax = plt.gca()
            # 关闭网格线
            ax.grid(False)
            
            # 获取唯一的模型列表
            unique_models = task_df['model'].unique()
            
            # 为每个模型绘制一条折线
            for model in unique_models:
                model_df = task_df[task_df['model'] == model]
                
                # 排序确保按bin_index顺序显示
                if 'bin_index' in model_df.columns:
                    model_df = model_df.sort_values('bin_index')
                else:
                    model_df = model_df.sort_values('bin_start')
                
                # 计算每个bin的中点作为X轴
                x_values = [(start + end) / 2 for start, end in zip(model_df['bin_start'], model_df['bin_end'])]
                
                # 获取模型对应的颜色
                model_color = get_model_color(model)
                
                # 绘制折线图，使用固定颜色
                plt.plot(x_values, model_df['accuracy'], marker='o', linewidth=2, markersize=8, 
                         label=model, color=model_color)
                
                # 在每个点上添加数值标签
                for x, y in zip(x_values, model_df['accuracy']):
                    ax.text(x, y + 0.01, f"{y:.2%}", ha='center', va='bottom', fontsize=8)
            
            # 设置标题和标签
            print(f"task: {task}")
            plt.title(f"Models Comparison for {task.capitalize()} Task by Embedding Distance", fontsize=16)
            plt.ylim(0, 1)
            plt.ylabel("Accuracy(%)", fontsize=12)
            
            # 格式化Y轴为百分比
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            
            # 明确关闭网格线
            plt.grid(False)
            
            # 添加图例
            plt.legend(title="Models", loc='best')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            output_file = os.path.join(output_dir, f"{task}_models_by_distance.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"{task}任务的模型比较图已保存至: {output_file}")
            
            plt.close()
    else:
        # 按距离范围分组比较模型
        # 假设每个bin_index对应一个距离范围，且在不同任务/模型之间一致
        if 'bin_index' in df.columns:
            bin_groups = df.groupby('bin_index')
        else:
            # 如果没有bin_index，尝试按bin_start和bin_end组合分组
            df['bin_range'] = df.apply(lambda row: f"{row['bin_start']:.4f}-{row['bin_end']:.4f}", axis=1)
            bin_groups = df.groupby('bin_range')
        
        # 取出所有bin_index或bin_range以供后续使用
        if 'bin_index' in df.columns:
            all_bins = sorted(df['bin_index'].unique())
            x_label = "Bin Index"
        else:
            all_bins = sorted(df['bin_range'].unique(), key=lambda x: float(x.split('-')[0]))
            x_label = "Embedding Distance Range"
        
        # 获取唯一的模型和任务组合
        model_task_combinations = df[['model', 'task']].drop_duplicates()
        
        plt.figure(figsize=(15, 10))
        
        # 创建折线图
        ax = plt.gca()
        # 关闭网格线
        ax.grid(False)
        
        # 为每个模型-任务组合绘制一条折线
        for _, row in model_task_combinations.iterrows():
            model = row['model']
            task = row['task']
            
            # 获取该模型-任务组合的所有数据
            combo_df = df[(df['model'] == model) & (df['task'] == task)]
            
            # 确保按bin排序
            if 'bin_index' in combo_df.columns:
                combo_df = combo_df.sort_values('bin_index')
            else:
                combo_df = combo_df.sort_values('bin_start')
            
            # 获取模型对应的颜色
            model_color = get_model_color(model)
            
            # 绘制折线图，使用固定颜色
            if 'bin_index' in combo_df.columns:
                plt.plot(combo_df['bin_index'], combo_df['accuracy'], marker='o', 
                         linewidth=2, markersize=6, label=f"{model} ({task})", color=model_color)
            else:
                # 使用序号作为x轴位置
                x_pos = [all_bins.index(b) for b in combo_df['bin_range']]
                plt.plot(x_pos, combo_df['accuracy'], marker='o',
                         linewidth=2, markersize=6, label=f"{model} ({task})", color=model_color)
        
        # 设置X轴刻度
        if 'bin_index' not in df.columns:
            plt.xticks(range(len(all_bins)), all_bins, rotation=45, ha='right')
        
        # 设置标题和标签
        plt.title("Models and Tasks Comparison by Embedding Distance Bins", fontsize=16)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        
        # 格式化Y轴为百分比
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        
        # 明确关闭网格线
        plt.grid(False)
        
        # 添加图例
        plt.legend(title="Model (Task)", loc='best', fontsize=10)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(output_dir, "all_models_tasks_by_bins.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"所有模型和任务的分组比较图已保存至: {output_file}")
        
        plt.close()

def plot_trend_analysis(csv_file, output_dir=None):
    """
    绘制趋势分析图，展示embedding distance与准确率的相关性
    
    Args:
        csv_file: 包含分组数据的CSV文件路径
        output_dir: 输出目录，默认为CSV文件所在目录
    """
    print(f"读取CSV文件进行趋势分析: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"错误: 无法读取CSV文件: {e}")
        return
    
    # 检查必要的列
    required_columns = ['bin_start', 'bin_end', 'samples', 'accuracy', 'model', 'task']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"错误: CSV文件缺少必要的列: {', '.join(missing_columns)}")
        return
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(csv_file)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 设置图表样式
    sns.set_style("white")  # 使用white样式替代whitegrid
    plt.figure(figsize=(12, 8))
    
    # 计算每个bin的中点
    df['distance_midpoint'] = (df['bin_start'] + df['bin_end']) / 2
    
    # 关闭网格线
    plt.grid(False)
    
    # 按任务分组绘制散点图和趋势线
    tasks = df['task'].unique()
    
    # 为每个模型和任务组合准备调色板
    model_task_palette = {}
    for task in tasks:
        task_df = df[df['task'] == task]
        models = task_df['model'].unique()
        
        # 为每个模型分配颜色
        for model in models:
            model_color = get_model_color(model)
            model_task_palette[(model, task)] = model_color
    
    # 为每个任务分别绘制散点图和趋势线
    for task in tasks:
        task_df = df[df['task'] == task]
        models = task_df['model'].unique()
        
        # 为每个模型绘制散点图
        for model in models:
            model_df = task_df[task_df['model'] == model]
            color = model_task_palette.get((model, task))
            
            # 绘制散点图
            sns.scatterplot(x='distance_midpoint', y='accuracy', data=model_df, 
                           color=color, size='samples', sizes=(50, 200), 
                           alpha=0.7, label=f"{model} ({task})")
    
        # 为整个任务添加趋势线
        sns.regplot(x='distance_midpoint', y='accuracy', data=task_df, 
                   scatter=False, line_kws={"linestyle": "--"}, 
                   label=f"{task} trend")
    
    # 设置标题和标签
    plt.title("Embedding Distance vs. Accuracy Trend Analysis", fontsize=16)
    plt.xlabel("Embedding Distance (average)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    
    # 格式化Y轴为百分比
    ax = plt.gca()
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # 明确关闭网格线
    plt.grid(False)
    
    # 添加图例
    plt.legend(title="Models and Tasks", loc='best')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_file = os.path.join(output_dir, "distance_accuracy_trend.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"趋势分析图已保存至: {output_file}")
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据分组的ref_gt_distance数据绘制准确率直方图")
    parser.add_argument("--csv-file", required=True, help="包含分组数据的CSV文件路径")
    parser.add_argument("--output-dir", default=None, help="输出目录，默认为CSV文件所在目录")
    parser.add_argument("--plot-title", default=None, help="图表标题，默认根据CSV文件名生成")
    
    args = parser.parse_args()
    
    plot_distance_accuracy_histogram(args.csv_file, args.output_dir, args.plot_title)