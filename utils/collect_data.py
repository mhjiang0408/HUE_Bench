import os
import pandas as pd
from pathlib import Path

def create_dataset(base_dir: str) -> pd.DataFrame:
    """
    创建整合数据集
    
    Args:
        base_dir: 基础目录路径
    
    Returns:
        pd.DataFrame: 包含所有数据样本信息的DataFrame
    """
    data = []
    
    # 分别获取Cover、Story、Article目录
    cover_base = os.path.join(base_dir, 'Cover')
    story_base = os.path.join(base_dir, 'Story')
    article_base = os.path.join(base_dir, 'Article')
    
    # 获取Cover目录下的所有期刊目录
    if os.path.exists(cover_base):
        journals = [d for d in os.listdir(cover_base) if os.path.isdir(os.path.join(cover_base, d))]
        
        for journal in journals:
            print(f"Processing journal: {journal}")
            # 构建各个目录的期刊路径
            cover_journal_dir = os.path.join(cover_base, journal)
            story_journal_dir = os.path.join(story_base, journal)
            article_journal_dir = os.path.join(article_base, journal)
            
            # 获取所有cover文件
            cover_files = Path(cover_journal_dir).rglob('*.*')
            
            for cover in cover_files:
                # 获取相对路径
                cover_path = os.path.relpath(cover, base_dir)
                
                # 构建可能的story和article路径，并将扩展名改为txt
                relative_path = os.path.relpath(cover, cover_journal_dir)
                relative_path_no_ext = os.path.splitext(relative_path)[0]  # 移除原始扩展名
                potential_story = os.path.join(story_journal_dir, relative_path_no_ext + '.txt')
                potential_article = os.path.join(article_journal_dir, relative_path_no_ext + '.txt')
                
                # 检查story和article是否存在
                story_path = os.path.relpath(potential_story, base_dir) if os.path.exists(potential_story) else None
                article_path = os.path.relpath(potential_article, base_dir) if os.path.exists(potential_article) else None
                
                data.append({
                    'journal': journal,
                    'cover_path': cover_path,
                    'story_path': story_path,
                    'article_path': article_path
                })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # 设置基础目录
    base_dir = "./Cell"
    
    # 创建数据集
    dataset = create_dataset(base_dir)
    
    # 保存为CSV文件
    output_file = "dataset_cell.csv"
    dataset.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    print(f"Total samples: {len(dataset)}")