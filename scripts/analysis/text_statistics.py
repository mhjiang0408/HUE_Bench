import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np
def plot_token_count_distribution(csv_path, output_dir=None):
    """
    统计ref_text_count的分布并绘制密度图
    Args:
        csv_path: 输入CSV文件路径，需包含'ref_text_count'列
        output_dir: 输出目录，默认为csv文件所在目录
    """
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'ref_text_count' not in df.columns:
        print("Error: 'ref_text_count' column not found in the CSV file.")
        return
    
    counts = df['ref_text_count'].dropna().astype(int).tolist()
    if not counts:
        print("No valid ref_text_count data found.")
        return
    
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    # 绘制直方图
    n, bins, patches = plt.hist(counts, bins=30, density=True, alpha=0.6, color='#66B2FF', edgecolor='white')
    # 绘制密度曲线
    sns.kdeplot(counts, color='darkblue', linewidth=2)
    
    # 渐变色处理
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = plt.cm.get_cmap('coolwarm')(np.linspace(0.2, 0.8, len(patches)))
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', c)
    
    # 统计信息
    mean_val = round(pd.Series(counts).mean(), 2)
    median_val = round(pd.Series(counts).median(), 2)
    min_val = min(counts)
    max_val = max(counts)
    total = len(counts)
    
    plt.title('Token Count Distribution (ref_text_count)', fontsize=18, fontweight='bold')
    plt.xlabel('Token Count', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # 添加统计信息
    plt.text(0.98, 0.98,
             f"Total: {total}\nMean: {mean_val}\nMedian: {median_val}\nMin: {min_val}\nMax: {max_val}",
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             fontsize=14,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#66B2FF', pad=0.5))
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'ref_text_token_count_distribution.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Distribution plot saved to: {out_path}")
    
    # 保存详细统计信息
    stats = {
        'total': total,
        'mean': mean_val,
        'median': median_val,
        'min': min_val,
        'max': max_val
    }
    stats_path = os.path.join(output_dir, 'ref_text_token_count_stats.csv')
    pd.DataFrame([stats]).to_csv(stats_path, index=False)
    print(f"Statistics saved to: {stats_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot the distribution of ref_text_count as a density plot.")
    parser.add_argument('--csv', required=True, help='Path to the input CSV file')
    parser.add_argument('--output-dir', default=None, help='Output directory')
    args = parser.parse_args()
    plot_token_count_distribution(args.csv, args.output_dir)

if __name__ == '__main__':
    main()
