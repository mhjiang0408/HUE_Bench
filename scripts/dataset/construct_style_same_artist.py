# -*- coding: utf-8 -*-
import os
import json
import random
import csv
import argparse
from glob import glob
from tqdm import tqdm

def get_artist_images(image_root, artist):
    """
    获取某艺术家所有图片路径
    """
    artist_dir = os.path.join(image_root, artist)
    if not os.path.isdir(artist_dir):
        return []
    # 支持jpg/jpeg/png
    images = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        images.extend(glob(os.path.join(artist_dir, ext)))
    return images

def generate_mcq_questions_same_artist(image_root, output_csv, similarity_json="./Dataset/comic_artist_similarity.json", min_images=6, min_similar_artists=3):
    """
    生成多选题，其中干扰项来自参考图像所属的同一艺术家
    
    Args:
        image_root: 图片根目录
        output_csv: 输出CSV文件路径
        similarity_json: 艺术家相似度JSON文件路径
        min_images: 艺术家最少需要的图片数量(至少5张：1张参考+4个选项)
        min_similar_artists: 艺术家最少需要的相似艺术家数量
    """
    # 加载艺术家相似度数据
    try:
        with open(similarity_json, 'r', encoding='utf-8') as f:
            similarity_data = json.load(f)
        print(f"已加载艺术家相似度数据，共 {len(similarity_data)} 位艺术家")
    except Exception as e:
        print(f"加载艺术家相似度文件失败: {e}")
        print(f"将不检查艺术家相似度条件，继续执行...")
        similarity_data = {}
    
    # 获取所有艺术家
    artist_dirs = [d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))]
    
    questions = []
    skipped_artists = {"not_in_json": [], "few_similar_artists": [], "few_images": []}
    
    for artist in tqdm(artist_dirs, desc='艺术家遍历'):
        # 检查是否在相似度JSON中且有足够的相似艺术家
        if similarity_data:
            if artist not in similarity_data:
                skipped_artists["not_in_json"].append(artist)
                continue
                
            similar_artists = similarity_data[artist]
            if len(similar_artists) < min_similar_artists:
                skipped_artists["few_similar_artists"].append(artist)
                continue
        
        # 获取本艺术家所有图片
        artist_images = get_artist_images(image_root, artist)
        
        # 确保每个艺术家至少有min_images张图片(5张：1张参考+4个选项)
        if len(artist_images) < min_images:
            skipped_artists["few_images"].append(artist)
            continue
        
        for ref_img in tqdm(artist_images, desc=f'处理{artist}图片', leave=False):
            # 提取文件名（不含后缀）作为ID
            ref_basename = os.path.basename(ref_img)
            ref_id = os.path.splitext(ref_basename)[0]
            
            # 除参考图片外的其他图片
            other_imgs = [img for img in artist_images if img != ref_img]
            
            # 随机选择4张用作选项（1个正确答案+3个干扰项）
            if len(other_imgs) < 4:
                continue
            
            option_imgs = random.sample(other_imgs, 4)
            
            # 随机选择一个作为正确答案
            correct_idx = random.randint(0, 3)
            
            # 将索引转为选项字母 (0->A, 1->B, 2->C, 3->D)
            correct_option = chr(ord('A') + correct_idx)
            
            # 构建记录行
            row = {
                'id': ref_id,
                'reference_artist': artist,
                'reference_image': ref_img,
                'ground_truth': correct_option,  # 记录哪个选项是正确答案
            }
            
            # 添加选项（所有选项都是同一艺术家）
            for idx in range(4):
                col = chr(ord('A') + idx)
                row[f'option_{col}'] = option_imgs[idx]
                row[f'option_{col}_artist'] = artist
            
            questions.append(row)
    
    # 写入csv，确保id是第一列
    fieldnames = ['id', 'reference_artist', 'reference_image', 'ground_truth',
                  'option_A', 'option_B', 'option_C', 'option_D',
                  'option_A_artist', 'option_B_artist', 'option_C_artist', 'option_D_artist']
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for q in questions:
            writer.writerow(q)
    
    print(f'共生成 {len(questions)} 道题，已保存到 {output_csv}')
    
    # 打印跳过的艺术家统计信息
    print(f"跳过的艺术家统计:")
    print(f"  不在相似度JSON中: {len(skipped_artists['not_in_json'])} 位")
    print(f"  相似艺术家少于{min_similar_artists}个: {len(skipped_artists['few_similar_artists'])} 位")
    print(f"  图片少于{min_images}张: {len(skipped_artists['few_images'])} 位")

def main():
    parser = argparse.ArgumentParser(description='根据图片目录生成风格多选题CSV，其中干扰项来自同一艺术家')
    parser.add_argument('--image-root', required=True, help='图片根目录（如gocomics_downloads）')
    parser.add_argument('--output-csv', required=True, help='输出csv文件路径')
    parser.add_argument('--similarity-json', default='./Dataset/comic_artist_similarity.json', help='艺术家相似度JSON文件路径')
    parser.add_argument('--min-images', type=int, default=5, help='艺术家最少需要的图片数量')
    parser.add_argument('--min-similar-artists', type=int, default=3, help='艺术家最少需要的相似艺术家数量')
    args = parser.parse_args()
    
    generate_mcq_questions_same_artist(args.image_root, args.output_csv, 
                                      args.similarity_json, 
                                      args.min_images, 
                                      args.min_similar_artists)

if __name__ == '__main__':
    main() 