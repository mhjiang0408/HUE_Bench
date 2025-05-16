# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse
from tqdm import tqdm

def build_adjacency_matrix(result_csv, threshold=0.7):
    """
    通过实验结果CSV文件构建艺术家邻接矩阵。
    
    Args:
        result_csv: 实验结果CSV文件路径
        threshold: 相似度阈值，高于此值的艺术家被认为相似
    
    Returns:
        艺术家邻接矩阵，艺术家列表，连接信息
    """
    print(f"从 {result_csv} 加载实验结果...")
    df = pd.read_csv(result_csv)
    
    # 获取所有独特的艺术家
    artists = set()
    artists.update(df['reference_artist'].unique())
    for i in range(1, 7):  # 考虑6个候选艺术家
        col_name = f'candidate{i}'
        if col_name in df.columns:
            artists.update(df[col_name].unique())
    
    artists = sorted(list(artists))
    n_artists = len(artists)
    print(f"找到 {n_artists} 个独特的艺术家")
    
    # 创建艺术家索引映射
    artist_to_idx = {artist: idx for idx, artist in enumerate(artists)}
    
    # 初始化邻接矩阵
    adjacency_matrix = np.zeros((n_artists, n_artists))
    
    # 记录连接信息（包括实际相似度得分）
    connections = defaultdict(list)
    
    # 填充邻接矩阵
    for _, row in tqdm(df.iterrows(), total=len(df), desc="分析艺术家关系"):
        ref_artist = row['reference_artist']
        ref_idx = artist_to_idx[ref_artist]
        
        # 处理每个候选艺术家的预测相似度
        for i in range(1, 7):
            candidate_col = f'candidate{i}'
            prob_col = f'pred_prob{i}'
            
            if candidate_col in row and prob_col in row:
                candidate = row[candidate_col]
                probability = row[prob_col]
                
                # 只记录超过阈值的连接
                if probability >= threshold:
                    candidate_idx = artist_to_idx[candidate]
                    
                    # 更新邻接矩阵（取最大相似度）
                    current_value = adjacency_matrix[ref_idx, candidate_idx]
                    if probability > current_value:
                        adjacency_matrix[ref_idx, candidate_idx] = probability
                        adjacency_matrix[candidate_idx, ref_idx] = probability  # 保持对称
                    
                    # 记录连接信息
                    connection_info = {
                        "similarity": float(probability),
                        "reference_image": row['reference_image'] if 'reference_image' in row else "",
                        "candidate_image": row[f'candidate{i}_image'] if f'candidate{i}_image' in row else ""
                    }
                    
                    # 使用排序的艺术家对作为键，确保不重复记录
                    artist_pair = tuple(sorted([ref_artist, candidate]))
                    if connection_info not in connections[artist_pair]:
                        connections[artist_pair].append(connection_info)
    
    return adjacency_matrix, artists, connections

def visualize_adjacency_matrix(adjacency_matrix, artists, output_file, threshold=0.7):
    """
    可视化艺术家邻接矩阵并保存为热图
    
    Args:
        adjacency_matrix: 艺术家邻接矩阵
        artists: 艺术家列表
        output_file: 输出文件路径
        threshold: 相似度阈值
    """
    # 只显示超过阈值的连接
    visual_matrix = adjacency_matrix.copy()
    visual_matrix[visual_matrix < threshold] = 0
    
    # 创建一个更大的图表以适应标签
    # plt.figure(figsize=(max(20, len(artists)//4), max(18, len(artists)//4)))
    
    # # 创建热图
    # sns.heatmap(visual_matrix, xticklabels=artists, yticklabels=artists, 
    #             cmap="YlOrRd", vmin=threshold, vmax=1.0, 
    #             square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    # plt.title(f"艺术家相似度关系 (阈值 >= {threshold})")
    # plt.tight_layout()
    # plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # plt.close()
    
    print(f"邻接矩阵热图已保存至 {output_file}")

def generate_graph_json(adjacency_matrix, artists, connections, output_file, threshold=0.7):
    """
    生成包含超过阈值的艺术家连接的JSON图结构
    
    Args:
        adjacency_matrix: 艺术家邻接矩阵
        artists: 艺术家列表
        connections: 艺术家连接信息
        output_file: 输出文件路径
        threshold: 相似度阈值
    """
    # 准备节点和边
    nodes = []
    edges = []
    
    # 创建节点 - 直接使用艺术家名字作为ID
    artists_with_connections = set()
    for i in range(len(artists)):
        artist = artists[i]
        # 检查这个艺术家是否有超过阈值的连接
        if np.any(adjacency_matrix[i] >= threshold):
            artists_with_connections.add(artist)
            nodes.append({
                "id": artist,
                "label": artist,
                "group": i  # 保留原始索引作为组ID
            })
    
    # 创建边
    edge_count = 0
    for i in range(len(artists)):
        for j in range(i+1, len(artists)):  # 只处理上三角矩阵
            similarity = adjacency_matrix[i, j]
            if similarity >= threshold:
                artist1 = artists[i]
                artist2 = artists[j]
                
                # 确保两个艺术家都在节点中
                if artist1 in artists_with_connections and artist2 in artists_with_connections:
                    # 获取这个连接的详细信息
                    artist_pair = tuple(sorted([artist1, artist2]))
                    connection_details = connections.get(artist_pair, [])
                    
                    # 添加边 - 直接使用艺术家名字作为源和目标
                    edges.append({
                        "id": f"e{edge_count}",
                        "source": artist1,
                        "target": artist2,
                        "weight": float(similarity),
                        "details": connection_details
                    })
                    edge_count += 1
    
    # 构建完整图结构
    graph_data = {
        "nodes": nodes,
        "edges": edges,
        "threshold": threshold,
        "metadata": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "total_artists": len(artists),
            "artists_with_connections": len(artists_with_connections)
        }
    }
    
    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    print(f"艺术家关系图JSON已保存至 {output_file}")
    print(f"节点数：{len(nodes)}，边数：{len(edges)}")
    return graph_data

def analyze_artist_communities(adjacency_matrix, artists, threshold=0.7):
    """
    分析艺术家社区（连通分量）
    
    Args:
        adjacency_matrix: 艺术家邻接矩阵
        artists: 艺术家列表
        threshold: 相似度阈值
    
    Returns:
        社区列表，每个社区是艺术家名称的列表
    """
    # 创建二值邻接矩阵 (1表示连接，0表示无连接)
    binary_matrix = (adjacency_matrix >= threshold).astype(int)
    n = len(artists)
    
    # 使用DFS发现连通分量
    visited = [False] * n
    communities = []
    
    def dfs(node, community):
        visited[node] = True
        community.append(artists[node])
        
        for neighbor in range(n):
            if binary_matrix[node, neighbor] == 1 and not visited[neighbor]:
                dfs(neighbor, community)
    
    # 对每个未访问的节点执行DFS
    for i in range(n):
        if not visited[i]:
            # 检查是否有任何连接
            if np.sum(binary_matrix[i]) > 0:
                new_community = []
                dfs(i, new_community)
                if len(new_community) > 1:  # 只保留有多个艺术家的社区
                    communities.append(new_community)
    
    # 按社区大小排序
    communities.sort(key=len, reverse=True)
    
    return communities

def generate_simple_json(adjacency_matrix, artists, output_file, threshold=0.7):
    """
    生成简化版的JSON格式，直接使用艺术家名字作为键，值为包含其他相似艺术家名字及其相似度的字典
    格式: {"艺术家1":{"艺术家2":相似度,"艺术家3":相似度,...},...}
    如果某艺术家没有相似艺术家，则显示为空字典{}
    
    Args:
        adjacency_matrix: 艺术家邻接矩阵
        artists: 艺术家列表
        output_file: 输出文件路径
        threshold: 相似度阈值
    """
    # 创建简化的JSON结构
    simplified_data = {}
    
    # 对每个艺术家
    for i, artist in enumerate(artists):
        # 查找相似度高于阈值的其他艺术家
        similar_artists = {}
        for j, other_artist in enumerate(artists):
            if i != j and adjacency_matrix[i, j] >= threshold:
                similar_artists[other_artist] = float(adjacency_matrix[i, j])
        
        # 无论是否有相似的艺术家，都添加到结果中
        simplified_data[artist] = similar_artists
    
    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(simplified_data, f, ensure_ascii=False, indent=2)
    
    print(f"简化版艺术家关系JSON已保存至 {output_file}")
    print(f"总共包含 {len(simplified_data)} 个艺术家")
    
    # 统计有相似艺术家的数量
    artists_with_similar = sum(1 for similar in simplified_data.values() if similar)
    print(f"其中 {artists_with_similar} 个艺术家有相似艺术家")
    
    return simplified_data

def main():
    parser = argparse.ArgumentParser(description="分析艺术家相似度关系并生成图结构")
    parser.add_argument("--result-csv", required=True, help="实验结果CSV文件路径")
    parser.add_argument("--threshold", type=float, default=0.7, help="相似度阈值，默认为0.7")
    parser.add_argument("--output-dir", default="./Dataset/analysis_results", help="输出目录")
    parser.add_argument("--simple", action="store_true", help="只生成简化的JSON格式")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置输出文件路径
    base_name = os.path.splitext(os.path.basename(args.result_csv))[0]
    simple_json_output = os.path.join(args.output_dir, f"{base_name}_artist_similarity.json")
    
    # 构建邻接矩阵
    adjacency_matrix, artists, connections = build_adjacency_matrix(args.result_csv, args.threshold)
    
    # 生成简化的JSON格式
    simplified_data = generate_simple_json(adjacency_matrix, artists, simple_json_output, args.threshold)
    
    # 如果不是简化模式，则生成其他输出
    if not args.simple:
        matrix_output = os.path.join(args.output_dir, f"{base_name}_adjacency_matrix.png")
        graph_output = os.path.join(args.output_dir, f"{base_name}_artist_graph.json")
        community_output = os.path.join(args.output_dir, f"{base_name}_artist_communities.json")
        
        # 可视化邻接矩阵
        visualize_adjacency_matrix(adjacency_matrix, artists, matrix_output, args.threshold)
        
        # 生成图结构
        graph_data = generate_graph_json(adjacency_matrix, artists, connections, graph_output, args.threshold)
        
        # 分析艺术家社区
        communities = analyze_artist_communities(adjacency_matrix, artists, args.threshold)
        
        # 保存社区信息
        community_data = {
            "threshold": args.threshold,
            "total_communities": len(communities),
            "communities": [
                {
                    "id": i,
                    "size": len(community),
                    "artists": community
                }
                for i, community in enumerate(communities)
            ]
        }
        
        with open(community_output, 'w', encoding='utf-8') as f:
            json.dump(community_data, f, ensure_ascii=False, indent=2)
        
        print(f"找到 {len(communities)} 个艺术家社区")
        print(f"社区信息已保存至 {community_output}")
        
        # 打印最大的几个社区
        if communities:
            print("\n最大的艺术家社区:")
            for i, community in enumerate(communities[:3]):  # 只显示前3个最大的社区
                print(f"社区 {i+1} (艺术家数: {len(community)}): {', '.join(community[:5])}" + 
                      (f"... 等{len(community)-5}个" if len(community) > 5 else ""))

if __name__ == "__main__":
    main() 