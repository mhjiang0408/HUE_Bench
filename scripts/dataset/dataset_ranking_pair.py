import csv
import random
import numpy as np
import torch
import argparse
from tqdm import tqdm
import os
import sys
sys.path.append(os.getcwd())
from utils.clip import CLIPEmbedding
from utils.open_clip import OpenClip
from utils.parse_date import extract_comic_series_and_date
# from utils.siglip import SigLIP2Encoder
# from utils.close_source import QwenEmbedding
from collections import defaultdict
import json
import re
import datetime



def sample_latest_comics_by_series(base_path, output_csv=None, extensions=None):
    """
    从基础路径下的每个艺术家文件夹中，按漫画集分类，并从每个漫画集中选择最新的一个文件
    
    Args:
        base_path: 包含艺术家文件夹的基础路径
        output_csv: 输出CSV文件的路径，如果为None则使用"comics_latest_samples.csv"
        extensions: 要筛选的文件扩展名列表，如果为None则使用['.jpg', '.jpeg', '.png']
    
    Returns:
        bool: 操作是否成功
    """
    if output_csv is None:
        output_csv = "./comics_latest_samples.csv"
    
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png']
    
    # 确保目录存在
    if not os.path.exists(base_path):
        print(f"错误: 路径不存在 {base_path}")
        return False
    
    print(f"开始从 {base_path} 中按漫画集选择最新文件")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    
    # 创建CSV文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['artist', 'comic_series', 'date', 'file_name', 'file_path'])
        
        # 获取所有艺术家文件夹
        artists = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        print(f"找到 {len(artists)} 个艺术家文件夹")
        
        total_series = 0
        skipped_artists = 0
        date_missing_count = 0
        
        # 遍历每个艺术家文件夹
        for artist in tqdm(artists, desc="处理艺术家"):
            artist_path = os.path.join(base_path, artist)
            
            # 获取该艺术家目录下的所有符合扩展名的文件
            all_files = []
            for root, _, files in os.walk(artist_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        file_path = os.path.join(root, file)
                        all_files.append((file, file_path))
            
            if not all_files:
                print(f"警告: 艺术家 {artist} 没有符合条件的文件，已跳过")
                skipped_artists += 1
                continue
            
            # 按漫画集分组
            series_files = defaultdict(list)
            
            for file_name, file_path in all_files:
                series_name, date_str, date_obj = extract_comic_series_and_date(file_name)
                series_files[series_name].append((file_name, file_path, date_str, date_obj))
            
            # 从每个漫画集中选择最新的一个文件
            for series_name, files in series_files.items():
                # 优先使用日期对象排序
                files_with_dates = [(f, p, d_str, d_obj) for f, p, d_str, d_obj in files if d_obj is not None]
                
                if files_with_dates:
                    # 按日期排序并选择最新的文件
                    latest_file = sorted(files_with_dates, key=lambda x: x[3], reverse=True)[0]
                    file_name, file_path, date_str, _ = latest_file
                else:
                    # 如果没有有效日期的文件，使用文件名排序作为后备方案
                    date_missing_count += 1
                    files_sorted = sorted(files, key=lambda x: x[0], reverse=True)
                    file_name, file_path, date_str, _ = files_sorted[0]
                    print(f"警告: 无法提取漫画集 '{series_name}' 的日期，使用文件名排序替代")
                
                # 写入CSV
                # rel_file_path = os.path.relpath(file_path, base_path)
                csv_writer.writerow([artist, series_name, date_str, file_name, file_path])
                total_series += 1
    
    print(f"完成! 共处理了 {len(artists) - skipped_artists} 个艺术家, {total_series} 个漫画集")
    if date_missing_count > 0:
        print(f"警告: 有 {date_missing_count} 个漫画集无法提取日期，使用了文件名排序作为替代")
    print(f"结果已保存到 {output_csv}")
    return True

def analyze_comic_series(base_path, extensions=None):
    """
    分析基础路径下艺术家的漫画集分布，不进行抽样，仅统计信息
    
    Args:
        base_path: 包含艺术家文件夹的基础路径
        extensions: 要筛选的文件扩展名列表，如果为None则使用['.jpg', '.jpeg', '.png']
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png']
    
    # 确保目录存在
    if not os.path.exists(base_path):
        print(f"错误: 路径不存在 {base_path}")
        return
    
    print(f"分析 {base_path} 中的漫画集分布")
    
    # 获取所有艺术家文件夹
    artists = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    print(f"找到 {len(artists)} 个艺术家文件夹")
    
    total_series = 0
    artist_stats = {}
    
    # 遍历每个艺术家文件夹
    for artist in tqdm(artists, desc="分析艺术家"):
        artist_path = os.path.join(base_path, artist)
        
        # 获取该艺术家目录下的所有符合扩展名的文件
        all_files = []
        for root, _, files in os.walk(artist_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    all_files.append(file)
        
        if not all_files:
            continue
        
        # 按漫画集分组
        series_files = defaultdict(list)
        
        for file_name in all_files:
            series_name, _ = extract_comic_series_and_date(file_name)
            series_files[series_name].append(file_name)
        
        # 记录该艺术家的漫画集统计
        artist_stats[artist] = {
            'total_files': len(all_files),
            'series_count': len(series_files),
            'series': {name: len(files) for name, files in series_files.items()}
        }
        
        total_series += len(series_files)
    
    # 输出统计信息
    print(f"\n分析完成! 共 {len(artists)} 个艺术家, {total_series} 个漫画集")
    
    # 输出漫画集数量最多的前10个艺术家
    sorted_artists = sorted(artist_stats.items(), key=lambda x: x[1]['series_count'], reverse=True)
    print("\n漫画集数量最多的艺术家:")
    for i, (artist, stats) in enumerate(sorted_artists[:10], 1):
        print(f"{i}. {artist}: {stats['series_count']} 个漫画集, {stats['total_files']} 个文件")
    
    # 输出文件数量最多的前10个艺术家
    sorted_by_files = sorted(artist_stats.items(), key=lambda x: x[1]['total_files'], reverse=True)
    print("\n文件数量最多的艺术家:")
    for i, (artist, stats) in enumerate(sorted_by_files[:10], 1):
        print(f"{i}. {artist}: {stats['total_files']} 个文件, {stats['series_count']} 个漫画集")

def sample_artist_files(base_path, output_csv=None, extensions=None):
    """
    从基础路径下的每个子文件夹（艺术家）中随机抽取一个文件，并记录到CSV文件中
    
    Args:
        base_path: 包含艺术家文件夹的基础路径
        output_csv: 输出CSV文件的路径，如果为None则使用"artist_samples.csv"
        extensions: 要筛选的文件扩展名列表，如果为None则使用['.jpg', '.jpeg', '.png', '.pt']
    """
    if output_csv is None:
        output_csv = "./dataset/artist_samples.csv"
    
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.pt']
    
    # 确保目录存在
    if not os.path.exists(base_path):
        print(f"错误: 路径不存在 {base_path}")
        return False
    
    print(f"开始从 {base_path} 中随机抽取文件")
    
    # 创建CSV文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['artist', 'file_name', 'file_path'])
        
        # 获取所有艺术家文件夹
        artists = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        print(f"找到 {len(artists)} 个艺术家文件夹")
        
        sampled_count = 0
        skipped_count = 0
        
        # 遍历每个艺术家文件夹
        for artist in tqdm(artists, desc="处理艺术家"):
            artist_path = os.path.join(base_path, artist)
            
            # 获取该艺术家目录下的所有符合扩展名的文件
            all_files = []
            for root, _, files in os.walk(artist_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        all_files.append(os.path.join(root, file))
            
            if not all_files:
                print(f"警告: 艺术家 {artist} 没有符合条件的文件，已跳过")
                skipped_count += 1
                continue
            
            # 随机选择一个文件
            selected_file = random.choice(all_files)
            file_name = os.path.basename(selected_file)
            file_path = os.path.relpath(selected_file, base_path)
            
            # 写入CSV
            csv_writer.writerow([artist, file_name, file_path])
            sampled_count += 1
    
    print(f"抽样完成！已将 {sampled_count} 个艺术家的样本文件记录到 {output_csv}")
    if skipped_count > 0:
        print(f"注意: {skipped_count} 个艺术家文件夹中没有找到符合条件的文件")
    
    return True

def convert_image_path_to_embedding_path(image_path, image_base_path="./gocomics_downloads1", embedding_base_path="./Dataset/Comics_Embeddings"):
    """
    将图片路径转换为对应的embedding路径
    
    Args:
        image_path: 图片的路径
        image_base_path: 图片的基础路径
        embedding_base_path: embedding的基础路径
    
    Returns:
        对应的embedding路径
    """
    # 获取相对路径
    if image_path.startswith(image_base_path):
        rel_path = os.path.relpath(image_path, image_base_path)
    else:
        rel_path = image_path
    
    # 更改扩展名并构建embedding路径
    embedding_file = os.path.splitext(rel_path)[0] + '.pt'
    embedding_path = os.path.join(embedding_base_path, embedding_file)
    
    return embedding_path

def convert_embedding_path_to_image_path(embedding_path, image_base_path="./gocomics_downloads1", embedding_base_path="./Dataset/Comics_Embeddings"):
    """
    将embedding路径转换为对应的图片路径
    
    Args:
        embedding_path: embedding的路径
        image_base_path: 图片的基础路径
        embedding_base_path: embedding的基础路径
    
    Returns:
        对应的图片路径
    """
    # 获取相对路径
    if embedding_path.startswith(embedding_base_path):
        rel_path = os.path.relpath(embedding_path, embedding_base_path)
    else:
        rel_path = embedding_path
    
    # 更改扩展名并构建图片路径
    image_file = os.path.splitext(rel_path)[0] + '.jpg'
    image_path = os.path.join(image_base_path, image_file)
    
    return image_path

def find_all_images(base_path, extensions=['.jpg', '.jpeg', '.png']):
    """
    查找给定路径下所有符合扩展名的图片文件
    
    Args:
        base_path: 图片文件的基础路径
        extensions: 要查找的文件扩展名列表
    
    Returns:
        图片路径列表
    """
    print(f"正在查找{base_path}下的所有图片...")
    image_paths = []
    file_count = 0
    
    for root, _, files in os.walk(base_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                image_paths.append(file_path)
                file_count += 1
                if file_count % 1000 == 0:
                    print(f"已找到 {file_count} 个图片文件")
    
    print(f"共找到 {len(image_paths)} 个图片文件")
    return image_paths

def load_embeddings(samples, image_base_path, embedding_base_path):
    """
    加载给定图片路径对应的所有embedding
    
    Args:
        image_paths: 图片路径列表
        image_base_path: 图片的基础路径
        embedding_base_path: embedding的基础路径
    
    Returns:
        字典，键为原始图片路径，值为embedding张量
    """
    print(f"正在加载对应的embedding文件...")
    image_paths = [sample[2] for sample in samples]
    embeddings = {}
    loaded_count = 0
    failed_count = 0
    
    for image_path in tqdm(image_paths, desc="加载embeddings"):
        # 转换为embedding路径
        embedding_path = convert_image_path_to_embedding_path(image_path, image_base_path, embedding_base_path)
        
        # 尝试加载embedding
        try:
            if os.path.exists(embedding_path):
                embedding = torch.load(embedding_path)
                embeddings[image_path] = embedding
                loaded_count += 1
                if loaded_count % 1000 == 0:
                    print(f"已加载 {loaded_count} 个embedding文件")
            else:
                # print(f"警告: 未找到embedding文件 {embedding_path}")
                failed_count += 1
        except Exception as e:
            # print(f"加载 {embedding_path} 时出错: {str(e)}")
            failed_count += 1
    
    print(f"共加载了 {loaded_count} 个embedding文件，{failed_count} 个文件加载失败")
    return embeddings

def load_embeddings_all(image_paths, image_base_path, embedding_base_path):
    """
    加载给定图片路径对应的所有embedding
    
    Args:
        image_paths: 图片路径列表
        image_base_path: 图片的基础路径
        embedding_base_path: embedding的基础路径
    
    Returns:
        字典，键为原始图片路径，值为embedding张量
    """
    print(f"正在加载对应的embedding文件...")

    embeddings = {}
    loaded_count = 0
    failed_count = 0
    
    for image_path in tqdm(image_paths, desc="加载embeddings"):
        # 转换为embedding路径
        embedding_path = convert_image_path_to_embedding_path(image_path, image_base_path, embedding_base_path)
        
        # 尝试加载embedding
        try:
            if os.path.exists(embedding_path):
                embedding = torch.load(embedding_path)
                embeddings[image_path] = embedding
                loaded_count += 1
                if loaded_count % 1000 == 0:
                    print(f"已加载 {loaded_count} 个embedding文件")
            else:
                # print(f"警告: 未找到embedding文件 {embedding_path}")
                failed_count += 1
        except Exception as e:
            # print(f"加载 {embedding_path} 时出错: {str(e)}")
            failed_count += 1
    
    print(f"共加载了 {loaded_count} 个embedding文件，{failed_count} 个文件加载失败")
    return embeddings

def find_similar_images(sample_image_path, all_embeddings, clip_model, top_k=6):
    """
    为样本图片找到最相似的K张图片
    
    Args:
        sample_image_path: 样本图片的路径
        all_embeddings: 所有图片的embedding字典 (键为图片路径，值为embedding)
        clip_model: CLIP模型对象，用于计算相似度
        top_k: 返回相似度最高的前K张图片
    
    Returns:
        列表，包含(图片路径, 相似度)元组，按相似度降序排列
    """
    if sample_image_path not in all_embeddings:
        print(f"警告: 未找到 {sample_image_path} 的embedding")
        return []
    
    sample_embedding = all_embeddings[sample_image_path]
    similarities = []
    
    for path, embedding in all_embeddings.items():
        if path != sample_image_path:  # 排除自身
            similarity = clip_model.eval_embedding_similarity(sample_embedding, embedding)
            similarities.append((path, similarity))
    
    # 按相似度降序排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 返回前K个结果
    return similarities[:top_k]

def get_another_image_from_same_artist(image_path, all_embeddings, image_base_path):
    """
    从同一艺术家的文件夹中随机选择另一张图片
    
    Args:
        image_path: 图片路径
        all_embeddings: 所有图片的embedding字典 (键为图片路径)
        image_base_path: 图片基础路径
    
    Returns:
        选中的图片路径，如果没有其他图片则返回None
    """
    # 获取艺术家文件夹路径
    rel_path = os.path.relpath(image_path, image_base_path)
    artist_name = rel_path.split(os.sep)[0]
    artist_path = os.path.join(image_base_path, artist_name)
    
    # 获取该艺术家的所有图片
    artist_images = []
    for path in all_embeddings.keys():
        if path.startswith(artist_path) and path != image_path:
            artist_images.append(path)
    
    if not artist_images:
        return None
    
    # 随机选择一张图片
    return random.choice(artist_images)

def generate_similarity_pairs(samples_csv, image_base_path, embedding_base_path, output_csv, top_k=6, model_name="openai/clip-vit-large-patch14-336"):
    """
    根据样本CSV文件生成相似度配对数据集
    
    Args:
        samples_csv: 样本图片CSV文件路径
        image_base_path: 图片文件所在的基础路径
        embedding_base_path: embedding文件所在的基础路径
        output_csv: 输出CSV文件路径
        top_k: 为每个样本生成的配对数量
        model_name: CLIP模型名称
    """
    # 初始化CLIP模型
    print(f"初始化CLIP模型: {model_name}")
    try:
        clip_model = CLIPEmbedding(model_name=model_name)
        # clip_model = OpenClip(model_name="TULIP-so400m-14-384",pretrained="laion2b_s34b_b88k")
        # clip_model = SigLIP2Encoder(model_name="google/siglip2-so400m-patch16-naflex")
        # clip_model = QwenEmbedding()
    except Exception as e:
        print(f"初始化CLIP模型时出错: {str(e)}")
        return False
    
    # 查找所有图片文件
    all_image_paths = find_all_images(image_base_path)
    if not all_image_paths:
        print("错误: 未能找到任何图片文件")
        return False
    
    # 加载所有embedding
    all_embeddings = load_embeddings_all(all_image_paths, image_base_path, embedding_base_path)
    if not all_embeddings:
        print("错误: 未能加载任何embedding文件")
        return False
    
    # 读取样本CSV文件
    samples = []
    try:
        with open(samples_csv, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # 跳过标题行
            for row in reader:
                if len(row) >= 3:
                    artist, filename, filepath = row[0], row[1], row[2]
                    # 构建完整的图片路径
                    image_path = os.path.join(image_base_path, filepath)
                    samples.append((artist, filename, image_path))
    except Exception as e:
        print(f"读取样本CSV文件时出错: {str(e)}")
        return False
    
    print(f"从CSV中读取了 {len(samples)} 个样本")

    samples_embeddings = load_embeddings(samples, image_base_path, embedding_base_path)
    if not samples_embeddings:
        print("错误: 未能加载任何embedding文件")
        return False
    
    # 创建输出CSV文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['reference_image', 'option_a(same_artist)', 'option_b(similar_image)', 'similarity'])
        
        total_pairs = 0
        skipped_samples = 0
        
        # 为每个样本生成配对
        for artist, filename, sample_image_path in tqdm(samples, desc="生成配对"):
            # 检查样本是否有embedding
            if sample_image_path not in samples_embeddings:
                print(f"警告: 样本 {sample_image_path} 没有对应的embedding，已跳过")
                skipped_samples += 1
                continue
                
            # 寻找相似图片
            similar_images = find_similar_images(sample_image_path, samples_embeddings, clip_model, top_k)
            if not similar_images:
                print(f"警告: 未能为 {sample_image_path} 找到相似图片，已跳过")
                skipped_samples += 1
                continue
            
            # 从同一艺术家获取另一张图片
            same_artist_image = get_another_image_from_same_artist(sample_image_path, all_embeddings, image_base_path)
            if not same_artist_image:
                print(f"警告: 未能为 {artist} 找到另一张图片，已跳过")
                skipped_samples += 1
                continue
            
            # 生成配对记录
            for similar_image_path, similarity in similar_images:
                # 转换为相对路径以便输出到CSV
                ref_image_rel_path = os.path.relpath(sample_image_path, image_base_path)
                option_a_rel_path = os.path.relpath(same_artist_image, image_base_path)
                option_b_rel_path = os.path.relpath(similar_image_path, image_base_path)
                
                # 写入CSV
                csv_writer.writerow([ref_image_rel_path, option_a_rel_path, option_b_rel_path, f"{similarity:.4f}"])
                total_pairs += 1
    
    print(f"配对生成完成！共生成 {total_pairs} 个配对记录，已保存到 {output_csv}")
    if skipped_samples > 0:
        print(f"注意: {skipped_samples} 个样本被跳过")
    
    return True

def generate_artist_ranking_samples(samples_csv, output_csv, image_base_path="./gocomics_downloads1", embedding_base_path="./Dataset/Comics_Embeddings", num_candidates=6, model_name="openai/clip-vit-large-patch14-336"):
    """
    为每个艺术家生成一道题目，包含六个候选艺术家（基于embedding相似度）
    
    Args:
        samples_csv: 艺术家代表样本CSV文件路径，格式为：artist,file_name,file_path
        output_csv: 输出CSV文件路径
        image_base_path: 图片文件所在的基础路径
        embedding_base_path: embedding文件所在的基础路径
        num_candidates: 每个题目包含的候选艺术家数量
        model_name: CLIP模型名称
    
    Returns:
        生成的题目数量
    """
    # 初始化CLIP模型
    print(f"初始化CLIP模型: {model_name}")
    try:
        clip_model = CLIPEmbedding(model_name=model_name)
    except Exception as e:
        print(f"初始化CLIP模型时出错: {str(e)}")
        return 0
    
    # 创建输出目录
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 从CSV文件读取艺术家代表样本
    artist_samples = {}
    try:
        with open(samples_csv, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            # 跳过标题行
            header = next(reader)
            for row in reader:
                if len(row) >= 3:
                    artist,comic_series,date,file_name,file_path = row
                    key = f"{artist}^{comic_series}"
                    # 构建完整图片路径
                    image_path = file_path
                    artist_samples[key] = image_path
    except Exception as e:
        print(f"读取代表样本CSV文件时出错: {str(e)}")
        return 0
    
    if not artist_samples:
        print(f"从CSV文件 {samples_csv} 中未能读取任何艺术家样本")
        return 0
    
    print(f"从CSV文件中读取了 {len(artist_samples)} 个艺术家代表样本")
    
    # 为每个艺术家加载embedding
    artist_embeddings = {}
    
    for artist, image_path in tqdm(artist_samples.items(), desc="加载艺术家样本embedding"):
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"警告: 艺术家 {artist} 的样本图片不存在: {image_path}")
            continue
            
        # 加载该样本的embedding
        embedding_path = convert_image_path_to_embedding_path(image_path, image_base_path, embedding_base_path)
        try:
            if os.path.exists(embedding_path):
                embedding = torch.load(embedding_path)
                artist_embeddings[artist] = embedding
            else:
                print(f"艺术家 {artist} 的样本 {image_path} 没有对应的embedding: {embedding_path}")
        except Exception as e:
            print(f"加载 {embedding_path} 时出错: {str(e)}")
    
    if len(artist_embeddings) < num_candidates + 1:
        print(f"有效艺术家数量 ({len(artist_embeddings)}) 少于所需的候选数量 ({num_candidates+1})，无法生成题目")
        return 0
    
    print(f"成功加载了 {len(artist_embeddings)} 个艺术家的embedding")
    
    # 计算所有艺术家之间的相似度
    print("计算艺术家间的相似度...")
    artist_similarities = {}
    
    for artist1, emb1 in tqdm(artist_embeddings.items(), desc="计算艺术家相似度"):
        similarities = []
        for artist2, emb2 in artist_embeddings.items():
            if artist1 != artist2:
                similarity = clip_model.eval_embedding_similarity(emb1, emb2)
                similarities.append((artist2, similarity))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        artist_similarities[artist1] = similarities
    
    # 为每个艺术家生成一道题目
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'reference_artist', 'reference_image', 
            'candidate1', 'candidate1_image', 'similarity1',
            'candidate2', 'candidate2_image', 'similarity2',
            'candidate3', 'candidate3_image', 'similarity3',
            'candidate4', 'candidate4_image', 'similarity4',
            'candidate5', 'candidate5_image', 'similarity5',
            'candidate6', 'candidate6_image', 'similarity6'
        ])
        
        question_count = 0
        
        for artist, similarities in tqdm(artist_similarities.items(), desc="生成艺术家测试题目"):
            # 从artist key中提取真正的艺术家名称，格式为"artist_comic_series"
            reference_artist_name = artist.split('^')[0]
            
            # 筛选不是同一个艺术家的候选项
            filtered_similar = []
            i = 0
            while len(filtered_similar) < num_candidates and i < len(similarities):
                similar_artist, similarity = similarities[i]
                similar_artist_name = similar_artist.split('^')[0]
                
                # 检查是否为同一个艺术家的不同漫画集
                if similar_artist_name != reference_artist_name:
                    filtered_similar.append((similar_artist, similarity))
                else:
                    print(f"跳过同一艺术家({reference_artist_name})的不同漫画集: {similar_artist}")
                
                i += 1
            
            # 如果筛选后的相似艺术家不足，跳过
            if len(filtered_similar) < num_candidates:
                print(f"艺术家 {artist} 的有效相似艺术家数量不足 {num_candidates}，跳过")
                continue
            
            # 准备题目数据
            row_data = [
                artist, 
                artist_samples[artist]
            ]
            
            # 添加候选艺术家数据
            for i, (similar_artist, similarity) in enumerate(filtered_similar, 1):
                row_data.extend([
                    similar_artist,
                    artist_samples[similar_artist],
                    f"{similarity:.4f}"
                ])
            
            # 写入CSV
            csv_writer.writerow(row_data)
            question_count += 1
    
    print(f"题目生成完成！共生成 {question_count} 道题目，已保存到 {output_csv}")
    return question_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成相似度配对数据集")
    parser.add_argument("--samples_csv", help="样本图片CSV文件路径")
    parser.add_argument("--image_base_path", default="./gocomics_downloads1", help="图片文件所在的基础路径")
    parser.add_argument("--embedding_base_path", default="./Dataset/Comics_Embeddings", help="embedding文件所在的基础路径")
    parser.add_argument("--output_csv", default="./Dataset/similarity_pairs.csv", help="输出CSV文件路径")
    parser.add_argument("--top_k", type=int, default=6, help="为每个样本生成的配对数量")
    parser.add_argument("--artist_ranking", action="store_true", help="生成艺术家相似度排名测试题目")
    parser.add_argument("--sample", action="store_true", help="sample artist")
    parser.add_argument("--artist_samples", default="./Dataset/artist_samples.csv", help="艺术家代表样本CSV文件路径")
    parser.add_argument("--ranking_output", default="./Dataset/artist_ranking_samples.csv", help="艺术家排名题目输出CSV文件路径")
    
    args = parser.parse_args()
    
    try:
        if args.artist_ranking:
            generate_artist_ranking_samples(
                args.samples_csv,
                args.output_csv,
                args.image_base_path,
                args.embedding_base_path,
                args.top_k
            )
        elif args.sample:
            # sample_artist_files(args.samples_csv, args.output_csv)
            sample_latest_comics_by_series(args.samples_csv, args.output_csv)
        else:
            generate_similarity_pairs(
                args.samples_csv,
                args.image_base_path,
                args.embedding_base_path,
                args.output_csv,
                args.top_k
            )
    except Exception as e:
        print(f"脚本执行出错: {str(e)}")