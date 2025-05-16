import os
import sys
import json
sys.path.append(os.getcwd())
from utils.clip import CLIPEmbedding
import torch
from typing import Dict
def calculate_clip_answer(clip:CLIPEmbedding,reference_image_path: str, option_image_paths: Dict[str, str]):
    if "political" in reference_image_path:
        type = "political"
    else:
        type = "comics"
    def transfer_embedding_path(image_path:str,type:str):
        if type == "political":
            return image_path.replace("gocomics_downloads_political","Dataset/Political_Embeddings").replace(".jpg",".pt")
        else:
            return image_path.replace("gocomics_downloads","Dataset/Comics_Embeddings").replace(".jpg",".pt")
    reference_image_embedding_path = transfer_embedding_path(reference_image_path,type)
    
    # 确保按照'A','B','C','D'的顺序处理选项
    ordered_keys = sorted(option_image_paths.keys())  # 按字母顺序排序键
    ordered_paths = [option_image_paths[key] for key in ordered_keys]
    option_image_embedding_paths = [transfer_embedding_path(path, type) for path in ordered_paths]
    reference_embedding = torch.load(reference_image_embedding_path)
    option_embeddings = [torch.load(path) for path in option_image_embedding_paths]

    sim = clip.eval_embedding_similarity(reference_embedding, option_embeddings)


    response = {
        "A": float(sim[0][0]),
        "B": float(sim[0][1]),
        "C": float(sim[0][2]),
        "D": float(sim[0][3])
    }
    # 返回计算结果
    return str(response),0,0,0



