# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.getcwd())
import json
import random
import csv
import argparse
import re
import datetime
import time
import torch
import gc
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from utils.clip import CLIPEmbedding
import numpy as np

def extract_date_from_filename(filename):
    """
    从文件名中提取日期
    
    Args:
        filename: 文件名，如 "W.T. Duck_2022-09-09.jpg"
        
    Returns:
        datetime.date对象，如果无法提取则返回None
    """
    # 尝试匹配常见的日期格式 (YYYY-MM-DD)
    date_pattern = r'_(\d{4}-\d{2}-\d{2})\.'
    match = re.search(date_pattern, filename)
    
    if match:
        # 提取日期字符串
        date_str = match.group(1)
        try:
            # 转换为日期对象
            date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            return date_obj
        except ValueError:
            pass
    
    # 如果没有找到标准日期格式，尝试其他格式
    # 例如 "Series Name_YYYYMMDD.jpg"
    date_pattern2 = r'_(\d{8})\.'
    match = re.search(date_pattern2, filename)
    
    if match:
        date_str = match.group(1)
        try:
            # 转换为标准格式
            date_obj = datetime.datetime.strptime(date_str, '%Y%m%d').date()
            return date_obj
        except ValueError:
            pass
    
    return None

def get_artist_images_with_dates(image_root, artist_name, comic_series=None):
    """
    获取某艺术家所有图片路径，并按日期排序
    
    Args:
        image_root: 图片根目录
        artist_name: 艺术家名称
        comic_series: 漫画系列名称（可选）
    
    Returns:
        按日期排序的图片路径列表和日期字典
    """
    artist_dir = os.path.join(image_root, artist_name)
    if not os.path.isdir(artist_dir):
        return [], {}
    
    # 支持jpg/jpeg/png
    images = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        images.extend(glob(os.path.join(artist_dir, ext)))
    
    # 提取每个图片的日期并排序
    dated_images = []
    image_dates = {}
    
    for img_path in images:
        filename = os.path.basename(img_path)
        
        # 如果指定了漫画系列，过滤不匹配的图片
        if comic_series and not filename.startswith(f"{comic_series}_"):
            continue
            
        date_obj = extract_date_from_filename(filename)
        if date_obj:
            dated_images.append((img_path, date_obj))
            image_dates[img_path] = date_obj
    
    # 按日期排序（从新到旧）
    dated_images.sort(key=lambda x: x[1], reverse=True)
    
    # 返回排序后的图片路径列表和日期字典
    return [img for img, _ in dated_images], image_dates

def load_artist_embeddings(image_root, embedding_base_path, artist_name, comic_series=None):
    """
    只加载指定艺术家的所有图片embeddings
    
    Args:
        image_root: 图片根目录
        embedding_base_path: embedding文件根目录
        artist_name: 艺术家名称
        
    Returns:
        embeddings字典，键为图片路径，值为embedding张量
    """
    print(f"加载艺术家 {artist_name} 的embeddings...")
    embeddings = {}
    
    # 构建艺术家embedding目录路径
    artist_embedding_dir = os.path.join(embedding_base_path, artist_name)
    
    # 检查目录是否存在
    if not os.path.exists(artist_embedding_dir):
        print(f"找不到艺术家 {artist_name} 的embedding目录")
        return embeddings
    
    # 遍历embedding目录
    for root, dirs, files in os.walk(artist_embedding_dir):
        for file in files:
            if comic_series and not comic_series in file:
                continue
            if file.endswith('.pt'):
                embedding_path = os.path.join(root, file)
                try:
                    # 加载embedding
                    embedding = torch.load(embedding_path)
                    
                    # 转换为图片路径
                    rel_path = os.path.relpath(embedding_path, embedding_base_path)
                    img_rel_path = os.path.splitext(rel_path)[0] + '.jpg'
                    img_path = os.path.join(image_root, img_rel_path)
                    
                    # 检查图片是否存在
                    if os.path.exists(img_path):
                        embeddings[img_path] = embedding
                    else:
                        # 尝试其他扩展名
                        alt_img_path = os.path.splitext(img_path)[0] + '.png'
                        if os.path.exists(alt_img_path):
                            embeddings[alt_img_path] = embedding
                except Exception as e:
                    print(f"加载embedding出错 {embedding_path}: {e}")
    
    print(f"成功加载 {len(embeddings)} 个 {artist_name} 的图片embeddings")
    return embeddings

def find_most_similar_image(ref_img, artist_images, embeddings, clip_model:CLIPEmbedding=None):
    """
    找到与参考图片embedding相似度最高的图片
    
    Args:
        ref_img: 参考图片路径
        artist_images: 艺术家的所有图片路径
        embeddings: 所有图片的embedding字典
        clip_model: CLIPEmbedding实例，用于计算相似度
        
    Returns:
        相似度最高的图片路径，如果没有可用embeddings则返回随机图片
    """
    if ref_img not in embeddings or not artist_images:
        # 如果没有embeddings，返回随机图片
        return random.choice(artist_images) if artist_images else None
    
    ref_embedding = embeddings[ref_img]
    
    # 筛选出有embedding的图片，排除参考图片本身
    valid_images = [img for img in artist_images if img in embeddings and img != ref_img]
    
    if not valid_images:
        # 如果没有有效的图片，随机选择一个
        return random.choice([img for img in artist_images if img != ref_img]) if artist_images else None
    
    # 收集所有有效图片的embedding
    candidate_embeddings = [embeddings[img] for img in valid_images]
    
    # 批量计算相似度
    if clip_model:
        # 使用CLIPEmbedding实例批量计算相似度
        similarities = clip_model.eval_embedding_similarity(ref_embedding, candidate_embeddings)
        
        # 如果返回的是矩阵，获取第一行(假设ref_embedding只有一个embedding)
        if hasattr(similarities, 'shape') and len(similarities.shape) > 1:
            similarities = similarities[0]
        
        # 找到最大相似度的索引
        best_idx = int(torch.argmax(torch.tensor(similarities)).item()) if isinstance(similarities, list) else int(np.argmax(similarities))
    else:
        # 如果没有clip_model实例，无法批量计算，回退到一个一个比较
        best_similarity = -1
        best_idx = 0
        
        for i, img in enumerate(valid_images):
            # 简单计算余弦相似度
            ref_norm = ref_embedding / torch.norm(ref_embedding, dim=-1, keepdim=True)
            img_norm = embeddings[img] / torch.norm(embeddings[img], dim=-1, keepdim=True)
            similarity = torch.sum(ref_norm * img_norm).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = i
    
    # 返回最相似的图片
    return valid_images[best_idx]

def get_previous_image(ref_img, artist_images, image_dates):
    """
    获取参考图片时间线上的前一幅图片
    
    Args:
        ref_img: 参考图片路径
        artist_images: 按日期排序的艺术家图片列表
        image_dates: 图片日期字典
        
    Returns:
        时间上前一幅图片，如果没有则返回None
    """
    if ref_img not in image_dates or not artist_images:
        return None
    
    ref_date = image_dates[ref_img]
    previous_images = []
    
    # 查找日期早于参考图片的所有图片
    for img in artist_images:
        if img != ref_img and img in image_dates:
            if image_dates[img] < ref_date:
                previous_images.append((img, image_dates[img]))
    
    # 如果有前序图片，返回日期最接近的一个
    if previous_images:
        previous_images.sort(key=lambda x: x[1], reverse=True)
        return previous_images[0][0]
    
    return None

def parse_composite_artist_id(artist_id):
    """
    解析复合艺术家ID
    
    Args:
        artist_id: 复合ID，格式为"艺术家名^漫画集名"
        
    Returns:
        (艺术家名, 漫画集名)元组
    """
    if '^' in artist_id:
        parts = artist_id.split('^', 1)
        return parts[0], parts[1]
    return artist_id, None

def generate_mcq_questions(sim_json_path, image_root, output_csv, embedding_base_path=None):
    """
    生成多选题CSV文件
    
    Args:
        sim_json_path: 艺术家相似度JSON文件路径
        image_root: 图片根目录
        output_csv: 输出CSV文件路径
        embedding_base_path: embedding根目录（可选）
    """
    with open(sim_json_path, 'r', encoding='utf-8') as f:
        sim_dict = json.load(f)
    
    all_artist_ids = list(sim_dict.keys())
    questions = []
    
    # 不再一次性加载所有embeddings
    use_embeddings = embedding_base_path is not None
    
    # 创建CLIPEmbedding实例（如果需要）
    clip_model = None
    if use_embeddings:
        clip_model = CLIPEmbedding()
    
    # 按艺术家名称分组图片和日期
    artist_images_cache = {}
    artist_dates_cache = {}
    
    for artist_id in tqdm(all_artist_ids, desc='艺术家遍历'):
        # 解析复合艺术家ID
        artist_name, comic_series = parse_composite_artist_id(artist_id)
        
        # 从相似字典中获取相似艺术家
        similar_artist_ids = list(sim_dict[artist_id].keys())
        if len(similar_artist_ids) < 3:
            continue  # 跳过相似艺术家不足3的
        
        # 生成缓存键，包含艺术家和漫画系列
        cache_key = f"{artist_name}_{comic_series}" if comic_series else artist_name
        
        # 获取本艺术家所有图片和日期（按日期排序），并基于漫画系列过滤
        if cache_key not in artist_images_cache:
            artist_images, image_dates = get_artist_images_with_dates(image_root, artist_name, comic_series)
            artist_images_cache[cache_key] = artist_images
            artist_dates_cache[cache_key] = image_dates
        else:
            artist_images = artist_images_cache[cache_key]
            image_dates = artist_dates_cache[cache_key]
        
        if len(artist_images) < 2:
            continue  # 至少要有两张图片
        
        # 需要处理的艺术家和漫画系列组合
        artists_to_process = []
        
        # 添加当前艺术家和漫画系列
        artists_to_process.append((artist_name, comic_series))
        
        # 添加相似艺术家到处理列表
        for similar_id in similar_artist_ids:
            similar_artist, similar_comic = parse_composite_artist_id(similar_id)
            similar_key = (similar_artist, similar_comic)
            if similar_key not in artists_to_process:
                artists_to_process.append(similar_key)
        
        # 为当前批次加载embeddings
        current_batch_embeddings = {}
        if use_embeddings:
            for artist_to_load, comic_to_load in artists_to_process:
                # 只加载需要的艺术家目录
                artist_embeddings = load_artist_embeddings(image_root, embedding_base_path, artist_to_load, comic_to_load)
                current_batch_embeddings.update(artist_embeddings)
                # 对于有comic_series的情况，筛选只属于该系列的图片
                # if comic_to_load:
                #     filtered_embeddings = {}
                #     for img_path, embedding in artist_embeddings.items():
                #         filename = os.path.basename(img_path)
                #         if filename.startswith(f"{comic_to_load}_"):
                #             filtered_embeddings[img_path] = embedding
                #     current_batch_embeddings.update(filtered_embeddings)
                # else:
                #     current_batch_embeddings.update(artist_embeddings)
        
        for ref_img in tqdm(artist_images, desc=f'处理{artist_id}图片', leave=False):
            # 提取文件名（不含后缀）作为ID
            time1 = time.time()
            ref_basename = os.path.basename(ref_img)
            ref_id = os.path.splitext(ref_basename)[0]
            
            # ground truth: 时间线上前一幅图片
            gt_img = get_previous_image(ref_img, artist_images, image_dates)
            if not gt_img:
                # 如果没有时间线上前一幅，则跳过
                continue
            
            # 干扰项：从相似艺术家中选3个，每个选择与参考图片embedding最相似的
            distractor_imgs = []
            
            # 打乱相似艺术家列表以随机选择
            random.shuffle(similar_artist_ids)
            time2 = time.time()
            # print(f'打乱相似艺术家列表用时{time2 - time1}秒')
            for similar_id in similar_artist_ids:
                if len(distractor_imgs) >= 3:
                    break
                
                # 解析相似艺术家ID，包括艺术家名和漫画系列
                similar_artist, similar_comic = parse_composite_artist_id(similar_id)
                
                # 生成缓存键
                similar_cache_key = f"{similar_artist}_{similar_comic}" if similar_comic else similar_artist
                
                # 获取相似艺术家的图片，并基于漫画系列过滤
                if similar_cache_key not in artist_images_cache:
                    similar_images, similar_dates = get_artist_images_with_dates(image_root, similar_artist, similar_comic)
                    artist_images_cache[similar_cache_key] = similar_images
                    artist_dates_cache[similar_cache_key] = similar_dates
                else:
                    similar_images = artist_images_cache[similar_cache_key]
                
                if not similar_images:
                    continue
                
                # 选择与参考图片embedding最相似的图片
                if use_embeddings and ref_img in current_batch_embeddings:
                    best_img = find_most_similar_image(ref_img, similar_images, current_batch_embeddings, clip_model)
                else:
                    # 如果没有embedding信息，随机选择
                    best_img = random.choice(similar_images)
                
                if best_img:
                    distractor_imgs.append((similar_id, best_img))
            time3 = time.time()
            # print(f'选择干扰项用时{time3 - time2}秒')
            # print(f'处理{artist_id}图片用时{time.time() - cur_time}秒')
            # 如果干扰项不足3个，跳过
            if len(distractor_imgs) < 3:
                continue
                
            # 构建选项列表：[(artist_id, image), ...]
            options = []
            
            # 先添加干扰项（前3个）
            for i in range(3):
                options.append((distractor_imgs[i][0], distractor_imgs[i][1]))
            
            # 然后以随机位置插入正确答案
            correct_idx = random.randint(0, 3)
            options.insert(correct_idx, (artist_id, gt_img))
            
            # 将索引转为选项字母 (0->A, 1->B, 2->C, 3->D)
            correct_option = chr(ord('A') + correct_idx)
            
            # 构建记录行
            row = {
                'id': ref_id,
                'reference_artist': artist_id,
                'reference_image': ref_img,
                'ground_truth': correct_option,
            }
            
            # 添加选项
            for idx, (opt_artist, opt_img) in enumerate(options):
                col = chr(ord('A') + idx)
                row[f'option_{col}'] = opt_img
                row[f'option_{col}_artist'] = opt_artist
            time4 = time.time()
            # print(f'添加选项用时{time4 - time3}秒')
            questions.append(row)
        
        # 清理当前批次的embeddings，释放内存
        if use_embeddings:
            current_batch_embeddings.clear()
            gc.collect()  # 强制垃圾回收，释放内存
    
    # 清理CLIPEmbedding实例
    if clip_model:
        del clip_model
        gc.collect()
    
    # 写入csv，确保id是第一列
    fieldnames = ['id', 'reference_artist', 'reference_image', 'ground_truth',
                  'option_A', 'option_B', 'option_C', 'option_D',
                  'option_A_artist', 'option_B_artist', 'option_C_artist', 'option_D_artist']
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for q in questions:
            writer.writerow(q)
    print(f'共生成 {len(questions)} 道题，已保存到 {output_csv}')

def main():
    parser = argparse.ArgumentParser(description='根据相似度json和图片目录生成风格多选题csv')
    parser.add_argument('--sim-json', required=True, help='艺术家相似度json文件路径')
    parser.add_argument('--image-root', required=True, help='图片根目录（如gocomics_downloads）')
    parser.add_argument('--embedding-base', help='embedding根目录', default=None)
    parser.add_argument('--output-csv', required=True, help='输出csv文件路径')
    args = parser.parse_args()
    
    generate_mcq_questions(args.sim_json, args.image_root, args.output_csv, args.embedding_base)

if __name__ == '__main__':
    main()
