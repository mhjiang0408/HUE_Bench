import os
import sys
import json
sys.path.append(os.getcwd())
import base64
from utils.config_loader import ConfigLoader
from utils.parse_jsonString import parse_probabilities
from utils.llm_call import CallLLM
from utils.clip import CLIPEmbedding
from utils.test_api import test_model_api
from scripts.method.method import CoVR
from scripts.experiment.clip_experiment import calculate_clip_answer
import openai
from typing import List, Dict, Tuple, Optional
import json
import csv, os, re
import random
import argparse
import pandas as pd
from tqdm import tqdm
import wandb
from datetime import datetime
import os
import logging
import time
from functools import wraps
import multiprocessing  # 添加多进程支持
import math  # 添加数学函数支持，用于计算分组
import threading
import queue

# logger = logging.getLogger('journal_processor')

# def setup_logger(log_file: str = "./debug/journal_processing.log") -> None:
#     """
#     设置全局日志记录器
#     """
#     # 确保日志文件所在目录存在
#     os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
#     # 设置日志级别
#     logger.setLevel(logging.INFO)
    
#     # 如果logger已经有处理器，先清除
#     if logger.handlers:
#         logger.handlers.clear()
    
#     # 创建文件处理器
#     file_handler = logging.FileHandler(log_file, encoding='utf-8')
#     file_handler.setLevel(logging.INFO)
    
#     # 设置日志格式
#     formatter = logging.Formatter('%(asctime)s - %(message)s')
#     file_handler.setFormatter(formatter)
    
#     # 添加处理器
#     logger.addHandler(file_handler)



class MultiChoiceEvaluation:
    def __init__(self, model:str = "Qwen/Qwen2.5-7B-Instruct", 
                 api_base:str = "https://api.siliconflow.cn/v1", 
                 api_key:str = "sk-ogfnmwnolxpgzpisetqjbgikyqawdazfjuhcavykqyphvgvc", 
                 prompt_template:dict[str, str] = None,
                 num_options:int = 4):  # 添加选项数量参数
        self.model = model
        if 'ocr' in self.model:
            self.ocr = True
        else:
            self.ocr = False
        if 'few_shot' in self.model:
            self.few_shot = True
            if 'few_shot_1' in self.model:
                self.shot_num = 1
                self.model = self.model.replace("_few_shot_1", "")
            elif 'few_shot_2' in self.model:
                self.shot_num = 2
                self.model = self.model.replace("_few_shot_2", "")
            elif 'few_shot_5' in self.model:
                self.shot_num = 5
                self.model = self.model.replace("_few_shot_5", "")
        else:
            self.few_shot = False
            self.shot_num = 0
        if 'descripted' in self.model:
            self.descripted = True
            self.model = self.model.replace("_descripted", "")
        else:
            self.descripted = False
        self.api_base = api_base
        self.api_key = api_key
        self.majority = False
        self.prompt_template = prompt_template or loader.load_config('Config/prompt_template/template.json')
        # if self.model =='CoVR':
        #     if self.ocr:
        #         self.llm = CoVRr3(ocr=True,mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        #     else:
        #         self.llm = CoVRr3(mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        # elif self.model == 'qwen2.5-vl-7b-instruct_majority':
        #     self.llm = CoVRr1(mllm_model='qwen2.5-vl-7b-instruct')
        #     self.majority = True
        # elif self.model == 'qwen-vl-max_majority':
        #     self.llm = CoVRr1(mllm_model='qwen-vl-max')
        #     self.majority = True
        # elif self.model == 'CoVR-qwen-max':
        #     if self.ocr:
        #         self.llm = CoVRr3(ocr=True,mllm_model='qwen-vl-max',mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        #     else:
        #         self.llm = CoVRr3(mllm_model='qwen-vl-max',mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        # elif self.model == 'CoVR-step-1v-8k':
        #     if self.ocr:
        #         self.llm = CoVRr3(ocr=True,mllm_model='step-1v-8k',mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        #     else:
        #         self.llm = CoVRr3(mllm_model='step-1v-8k',mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        # elif self.model == 'CoVR-step-1o-turbo-vision':
        #     if self.ocr:
        #         self.llm = CoVRr3(ocr=True,mllm_model='step-1o-turbo-vision',mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        #     else:
        #         self.llm = CoVRr3(mllm_model='step-1o-turbo-vision',mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        # elif self.model == 'CoVR-gemini':
        #     if self.ocr:
        #         self.llm = CoVRr3(ocr=True,mllm_model='gemini-1.5-pro-latest', mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        #     else:
        #         self.llm = CoVRr3(mllm_model='gemini-1.5-pro-latest', mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        # elif self.model == 'CoVR-ernie-4.5-8k':
        #     if self.ocr:
        #         self.llm = CoVRr3(ocr=True,mllm_model='ernie-4.5-8k-preview', low_detail=True,mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        #     else:
        #         self.llm = CoVRr3(mllm_model='ernie-4.5-8k-preview', low_detail=True,mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        # elif self.model == 'CoVR-gpt-4o':
        #     if self.ocr:
        #         self.llm = CoVRr3(ocr=True,mllm_model='gpt-4o', mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        #     else:
        #         self.llm = CoVRr3(mllm_model='gpt-4o', mllm_api_base=self.api_base, mllm_api_key=self.api_key)
        # elif self.model == 'CoVR-r1':
        #     if self.ocr:
        #         self.llm = CoVRr3(ocr=True,reasoning_model='deepseek-r1-250120', reasoning_api_base=self.api_base, reasoning_api_key=self.api_key)
        #     else:
        #         self.llm = CoVRr3(reasoning_model='deepseek-r1-250120', reasoning_api_base=self.api_base, reasoning_api_key=self.api_key)
        # elif self.model == 'CoVR-o3-mini':
        #     if self.ocr:
        #         self.llm = CoVRr3(ocr=True,reasoning_model='o3-mini',reasoning_api_base=self.api_base, reasoning_api_key=self.api_key)
        #     else:
        #         self.llm = CoVRr3(reasoning_model='o3-mini',reasoning_api_base=self.api_base, reasoning_api_key=self.api_key)
        # else:
        if self.model == 'DAD':
            self.llm = CoVR(mllm_model='qwen-vl-max-1119', reasoning_model='deepseek-r1-250120', mllm_api_base=self.api_base, mllm_api_key=self.api_key, reasoning_api_base=self.api_base, reasoning_api_key=self.api_key)
        elif self.model == 'clip':
            self.llm = CLIPEmbedding()
        elif self.model == 'DAD-gpt-4o':
            self.llm = CoVR(mllm_model='gpt-4o', reasoning_model='deepseek-r1-250120', mllm_api_base=self.api_base, mllm_api_key=self.api_key, reasoning_api_base=self.api_base, reasoning_api_key=self.api_key)
        elif self.model == 'DAD-gpt-4.1':
            self.llm = CoVR(mllm_model='gpt-4.1-2025-04-14', reasoning_model='deepseek-r1-250120', mllm_api_base=self.api_base, mllm_api_key=self.api_key, reasoning_api_base=self.api_base, reasoning_api_key=self.api_key)
        elif self.model == 'DAD-gemini':
            self.llm = CoVR(mllm_model='vertex-gemini-2.5-flash-preview-04-17', reasoning_model='deepseek-r1-250120', mllm_api_base=self.api_base, mllm_api_key=self.api_key, reasoning_api_base=self.api_base, reasoning_api_key=self.api_key)
        elif self.model == 'DAD-o3-mini':
            self.llm = CoVR(mllm_model='gpt-4.1-2025-04-14', reasoning_model='o3-mini', mllm_api_base=self.api_base, mllm_api_key=self.api_key, reasoning_api_base=self.api_base, reasoning_api_key=self.api_key)
        elif self.model == 'DAD-qwq':
            self.llm = CoVR(mllm_model='gpt-4.1-2025-04-14', reasoning_model='qwq-32b', mllm_api_base=self.api_base, mllm_api_key=self.api_key, reasoning_api_base=self.api_base, reasoning_api_key=self.api_key)
        else:
            print(f"model: {self.model}")
            self.llm = CallLLM(model=self.model, api_base=self.api_base, api_key=self.api_key)
        self.num_options = num_options
        # 生成选项ID列表 (A-H 或其他数量)
        self.option_ids = [chr(65 + i) for i in range(num_options)]  # 65是'A'的ASCII码

    def split_dataset(self,data: pd.DataFrame, num_splits: int = 5):
        """
        将数据集均匀拆分成指定份数
        
        Args:
            data: 原始数据集
            num_splits: 拆分份数
        
        Returns:
            list: 包含拆分后的数据集列表
        """
        # 计算每份的大小
        split_size = len(data) // num_splits
        remainder = len(data) % num_splits
        
        splits = []
        start = 0
        for i in range(num_splits):
            # 如果有余数，前几份多分配一条数据
            end = start + split_size + (1 if i < remainder else 0)
            splits.append(data.iloc[start:end].copy())
            start = end
        
        return splits
    # def multi_choice_message(self, question:str, options:str, image_path:str):
    #     """
    #     多选题的prompt
    #     """
    #     if self.few_shot:
    #         system_prompt = self.prompt_template['system_prompt']
    #         user_prompt = self.prompt_template['user_prompt']
    #         few_shot_prompt = self.prompt_template['few_shot_prompt']
    #         few_shot_image = self.prompt_template['few_shot_image']
    #     else:
    #         system_prompt = self.prompt_template['system_prompt']
    #         user_prompt = self.prompt_template['user_prompt']
    #     user_content = user_prompt.format(
    #         question=question, 
    #         options=options
    #     )
    #     if self.ocr:
    #         image_path = image_path.replace('Cover','OCRed_Cover')
    #     try:
    #         with open(image_path, 'rb') as image_file:
    #             # 读取图片并转换为base64
    #             base64_data = base64.b64encode(image_file.read())
    #             # 转换为字符串并去除b''
    #             image_base64 = base64_data.decode('utf-8')
    #             if self.few_shot:
    #                 few_shot_base64 = base64.b64encode(open(few_shot_image, 'rb').read())
    #                 few_shot_base64 = few_shot_base64.decode('utf-8')

    #     except Exception as e:
    #         print(f"Error: {e}")
    #         return None
    #     if self.few_shot:
    #         return [
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": [
    #                 {
    #                     "type": "text",
    #                     "text": user_content
    #                 },
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/png;base64,{few_shot_base64}",
    #                         "detail": "low"
    #                     }
    #                 },
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/png;base64,{image_base64}",
    #                         "detail": "low"
    #                     }
    #                 }
    #             ]}
    #         ]
    #     else:
    #         return [
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": [
    #                 {
    #                     "type": "text",
    #                     "text": user_content
    #                 },
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/png;base64,{image_base64}",
    #                         "detail": "low"
    #                     }
    #                 }
    #             ]}
    #         ]
    def encode_image_to_base64(self,image_path: str) -> Optional[str]:
        """将图像文件编码为 Base64 字符串。"""
        if not image_path or not os.path.exists(image_path):
            print(f"错误：图像文件不存在 {image_path}")
            return None
        
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                # 根据扩展名确定 MIME 类型
                ext = os.path.splitext(image_path)[1].lower()
                mime_type = "image/jpeg" # 默认为 jpeg
                return f"data:{mime_type};base64,{encoded_string}"
        except FileNotFoundError:
            print(f"错误：编码图像时未找到文件 {image_path}")
            return None
        except Exception as e:
            print(f"错误：编码图像 {image_path} 时出错: {e}")
            return None
    def multi_choice_message(self, reference_image_path: str, option_image_paths: Dict[str, str]):
        """
        多选题的prompt
        """
        # if self.few_shot:
        #     system_prompt = self.prompt_template['system_prompt']
        #     user_prompt = self.prompt_template['user_prompt']
        #     few_shot_story = self.prompt_template['few_shot_story']
        #     few_shot_optionA = self.prompt_template['few_shot_optionA']
        #     few_shot_optionB = self.prompt_template['few_shot_optionB']
        #     few_shot_optionC = self.prompt_template['few_shot_optionC']
        #     few_shot_optionD = self.prompt_template['few_shot_optionD']
        # else:
        system_prompt = self.prompt_template['system_prompt']
        user_prompt = self.prompt_template['user_prompt']
        reference_image_base64 = self.encode_image_to_base64(reference_image_path)
        if not reference_image_base64:
            print(f"错误：无法编码参考图像 {reference_image_path}")
            return None
        option_images_base64 = {}
        for opt_id, opt_path in option_image_paths.items():
            encoded = self.encode_image_to_base64(opt_path)
            if not encoded:
                print(f"错误：无法编码选项图像 {opt_path} (选项 {opt_id})")
                return None # 如果任何一个选项图像失败，则无法构建消息
            option_images_base64[opt_id] = encoded

        # 构建用户提示内容列表
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": reference_image_base64, "detail": "low"}
            },
            {
                "type": "text",
                "text": user_prompt
            }
        ]
        # 添加选项图像，按 A, B, C, D 顺序
        for opt_id in self.option_ids:
            if opt_id in option_images_base64:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": option_images_base64[opt_id], "detail": "low"}
                })
            else:
                print(f"警告：缺少选项 {opt_id} 的图像数据。")
                return None  # 必须有所有选项才能继续
        if self.few_shot:
            few_shot_content = []
            for i in range(self.shot_num):
                start_prompt = f"# Following is the {i+1}th few-shot example."
                few_shot_content.append({"type": "text", "text": start_prompt})
                prompt_key = f"few_shot_prompt{i+1}"
                image_key = f"few_shot_image{i+1}"
                try:
                    few_shot_prompt = self.prompt_template[prompt_key]
                    few_shot_image = self.prompt_template[image_key]
                    json_few_shot_image = json.loads(few_shot_image.replace("'", '"'))
                    few_shot_content.append({"type": "text", "text": few_shot_prompt})
                    shot_reference_image_base64 = self.encode_image_to_base64(json_few_shot_image['reference_image'])
                    few_shot_content.append({"type": "image_url", "image_url": {"url": shot_reference_image_base64, "detail": "low"}})
                    few_shot_content.append({"type": "text", "text": user_prompt})
                    for opt_id in self.option_ids:
                        option_image_base64 = self.encode_image_to_base64(json_few_shot_image[opt_id])
                        few_shot_content.append({"type": "image_url", "image_url": {"url": option_image_base64, "detail": "low"}})
                    few_shot_content.append({"type": "text", "text": "The above is all the few shot examples. The following is the question and options you need to answer."})
                except Exception as e:
                    print(f"错误：构建few-shot内容时出错: {e}")
                    return None
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": few_shot_content},
                {"role": "user", "content": user_content}
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        return messages

    
    @staticmethod
    def prepare_dataset(data_path: str, scaling_factor: float = 1.0, seed: int = 42):
        """
        准备数据集，使用随机种子打乱数据集顺序
        
        Args:
            data_path: 数据集路径
            scaling_factor: 采样比例
            seed: 随机种子，确保可重复性
            
        Returns:
            打乱后的数据集
        """
        data = pd.read_csv(data_path)
        # 无论scaling_factor是否为1，都进行随机打乱
        if scaling_factor < 1.0:
            # 如果需要采样，则采样后返回
            return data.sample(frac=scaling_factor, random_state=seed)
        else:
            # 如果不需要采样，只进行随机打乱
            return data.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # 添加一个写入进程函数
    def writer_process(self, csv_path, fieldnames, result_queue, stop_event):
        """
        专门的写入进程，从队列中读取结果并写入CSV文件
        
        Args:
            csv_path: CSV文件路径
            fieldnames: CSV文件的字段名
            result_queue: 结果队列
            stop_event: 停止事件，用于通知写入进程结束
        """
        # 打开CSV文件进行追加写入
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 持续从队列中获取结果并写入
            while not stop_event.is_set() or not result_queue.empty():
                try:
                    # 非阻塞方式获取队列中的结果，超时1秒
                    result = result_queue.get(timeout=1)
                    
                    # 写入结果
                    writer.writerow(result['record'])
                    csvfile.flush()  # 确保立即写入磁盘
                    
                    # 更新wandb日志
                    if 'wandb_data' in result:
                        wandb.log(result['wandb_data'])
                        
                except queue.Empty:
                    # 队列为空，继续等待
                    continue
                except Exception as e:
                    print(f"写入进程出错: {e}")
    
    # 处理数据块的工作进程函数
    def process_data_chunk(self, data_chunk, output_path, result_queue, worker_id, existing_results=None):
        """
        处理数据集的一个子集
        
        Args:
            data_chunk: 数据子集
            output_path: 输出目录
            result_queue: 结果队列，用于将结果传递给写入进程
            worker_id: 工作进程ID
            existing_results: 已存在的结果，用于跳过已处理的记录
        """
        total_count = 0
        correct_count = 0
        all_tokens = 0
        
        # 创建进度条
        pbar = tqdm(total=len(data_chunk), desc=f"Worker {worker_id}, model: {self.model}")
        description_df = pd.read_csv('./political_2025_description.csv')
        for _, row in data_chunk.iterrows():
            id = row['id']
            reference_artist = row['reference_artist']
            reference_image = row['reference_image']
            ground_truth = row['ground_truth']
            # 检查是否已处理过
            if existing_results is not None and any(
                r['id'] == id 
                for r in existing_results
            ):
                pbar.update(1)
                continue
            
            # 根据类型处理不同的输入
            options_image_paths = {}
            options_artists = {}
            for opt_id in self.option_ids:
                img_col = f'option_{opt_id}'
                artist_col = f'option_{opt_id}_artist'
                if img_col in row:
                    options_image_paths[opt_id] = row[img_col]
                    options_artists[opt_id] = row[artist_col]
                else:
                    print(f"警告：缺少选项 {opt_id} 的图像数据。")
            if 'DAD' in self.model:
                try:
                    if self.descripted:
                        descriptions = description_df[(description_df['id'] == id) & (description_df['reference_artist'] == reference_artist)]['description'].values[0]
                        if descriptions is None:
                            print(f"Error: lack of description for {id}")
                            continue
                        response, prompt_tokens, completion_tokens,total_tokens, options_image_paths = self.llm.post_existing_request(options_image_paths, descriptions)
                    else:
                        response, prompt_tokens, completion_tokens,total_tokens, options_image_paths = self.llm.post_request(options_image_paths, reference_image)
                except Exception as e:
                    print(f"Error: LLM请求失败且重试耗尽: {e}")
                    continue
            elif self.model == 'clip':
                try:
                    response,prompt_tokens,completion_tokens,total_tokens = calculate_clip_answer(self.llm,reference_image, options_image_paths)
                except Exception as e:
                    print(f"Error: CLIP请求失败且重试耗尽: {e}")
                    continue
            else:
                messages = self.multi_choice_message(reference_image, options_image_paths)
                if messages is None:
                    continue
                try:
                    if 'qvq' in self.model:
                        response, prompt_tokens, completion_tokens, reasoning_content = self.llm.post_reasoning_request(messages=messages)
                        # print(f"reasoning_content: {reasoning_content}")
                        total_tokens = prompt_tokens + completion_tokens
                    else:
                        # print(f"messages: {messages}")
                        with open('messages.json', 'w') as f:
                            json.dump(messages, f)
                        response, prompt_tokens, completion_tokens = self.llm.post_request(messages=messages)
                        total_tokens = prompt_tokens + completion_tokens
                except Exception as e:
                    print(f"Error: LLM请求失败且重试耗尽: {e}")
                    # answer = "None"
                    # judge = 0
                    # total_count += 1
                    # # 创建记录
                    # record = {
                    #     'id': id,
                    #     'reference_artist': reference_artist,
                    #     'reference_image': reference_image,
                    #     'ground_truth': ground_truth,
                    #     'llm_response': 'Error',
                    #     'predicted_answer': answer,
                    #     'options': options_image_paths,
                    #     'is_correct': judge,
                    #     'option_A_artist': options_artists['A'],
                    #     'option_B_artist': options_artists['B'],
                    #     'option_C_artist': options_artists['C'],
                    #     'option_D_artist': options_artists['D'],
                    #     'total_tokens': 0
                    #     # 'total_prompt_tokens': total_prompt_tokens
                    # }
                    # pbar.update(1)
                    continue
            
            
            # 解析回答
            format_answer = parse_probabilities(response)
            if not format_answer:
                answer = "None"
                judge = 0
                total_count += 1
                # 创建记录
                record = {
                    'id': id,
                    'reference_artist': reference_artist,
                    'reference_image': reference_image,
                    'ground_truth': ground_truth,
                    'llm_response': response,
                    'predicted_answer': answer,
                    'options': options_image_paths,
                    'is_correct': judge,
                    'option_A_artist': options_artists['A'],
                    'option_B_artist': options_artists['B'],
                    'option_C_artist': options_artists['C'],
                    'option_D_artist': options_artists['D'],
                    'total_tokens': f'{prompt_tokens};{completion_tokens}'
                    # 'total_prompt_tokens': total_prompt_tokens
                }
                
                # 将结果放入队列
                result_queue.put({
                    'record': record,
                    'wandb_data': {
                        "accuracy": correct_count/total_count, 
                        "all_tokens": all_tokens + total_tokens,
                        "num_options": self.num_options
                    }
                })
                
                all_tokens += total_tokens
                pbar.update(1)
                continue
            
            answer = max(format_answer, key=format_answer.get)
            total_count += 1
            judge = self.evaluation(ground_truth, answer)
            correct_count += judge
            
            # 创建记录
            record = {
                'id': id,
                'reference_artist': reference_artist,
                'reference_image': reference_image,
                'ground_truth': ground_truth,
                'llm_response': response,
                'predicted_answer': answer,
                'options': options_image_paths,
                'is_correct': judge,
                'option_A_artist': options_artists['A'],
                'option_B_artist': options_artists['B'],
                'option_C_artist': options_artists['C'],
                'option_D_artist': options_artists['D'],
                'total_tokens': f'{prompt_tokens};{completion_tokens}'
                # 'total_prompt_tokens': total_prompt_tokens
            }
            
            # 将结果放入队列
            result_queue.put({
                'record': record,
                'wandb_data': {
                    "accuracy": correct_count/total_count, 
                    "all_tokens": all_tokens + total_tokens,
                    "num_options": self.num_options
                }
            })
            
            all_tokens += total_tokens
            
            pbar.update(1)
        
        pbar.close()
        
        # 返回统计信息
        return {
            'worker_id': worker_id,
            'total_count': total_count,
            'correct_count': correct_count,
            'all_tokens': all_tokens
        }


    def evaluation(self, ground_truth:str, answer:str):
        """
        评价答案
        """
        if ground_truth == answer:
            return 1
        else:
            return 0

    def experiment_with_threads(self, data: pd.DataFrame, output_path: str, resume: bool = False, num_workers: int = 4):
        """
        使用线程而不是进程来运行实验
        """
        # 设置CSV文件路径
        if not os.path.exists(output_path) and not resume:
            os.makedirs(output_path, exist_ok=True)
            
            # 设置CSV文件路径和字段
            csv_path = os.path.join(output_path, 'results.csv')
            
        # 设置字段
        fieldnames = ['id', 'reference_artist', 'reference_image', 'ground_truth', 'options', 'llm_response', 'predicted_answer', 'is_correct', 'option_A_artist', 'option_B_artist', 'option_C_artist', 'option_D_artist','total_tokens']
        
        # 检查是否继续上次实验
        existing_results = []
        if resume and os.path.exists(output_path):
            try:
                # 使用pandas只读取需要的列
                df = pd.read_csv(output_path, usecols=['id'])
                existing_results = df.to_dict('records')
                print(f"继续上次实验，已有 {len(existing_results)} 条记录")
            except Exception as e:
                print(f"读取CSV文件时出错: {e}")
                print("尝试使用替代方法读取...")
                # 如果pandas读取失败，使用手动读取方式
                with open(csv_path, 'r', encoding='utf-8') as f:
                    # 读取header行找到需要的列的索引
                    header = next(csv.reader(f))
                    id_idx = header.index('id')
                    # 只读取需要的列
                    existing_results = []
                    for row in csv.reader(f):
                        existing_results.append({
                            'id': row[id_idx]
                        })
                print(f"使用替代方法成功读取，已有 {len(existing_results)} 条记录")
            csv_path = output_path
        else:
            # 如果不是继续上次实验，创建新文件
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        
        # 创建线程安全的队列和文件锁
        result_queue = queue.Queue()
        file_lock = threading.Lock()
        
        # 创建并启动写入线程
        stop_event = threading.Event()
        writer_thread = threading.Thread(
            target=self.writer_thread,
            args=(csv_path, fieldnames, result_queue, stop_event, file_lock)
        )
        writer_thread.start()
        
        # 分割数据
        data_chunks = self.split_dataset(data, num_workers)
        
        # 创建并启动工作线程
        threads = []
        for i, chunk in enumerate(data_chunks):
            t = threading.Thread(
                target=self.process_data_chunk,
                args=(chunk, output_path, result_queue, i, existing_results)
            )
            threads.append(t)
            t.start()
        
        # 等待所有工作线程完成
        for t in threads:
            t.join()
        
        # 通知写入线程结束并等待完成
        stop_event.set()
        writer_thread.join()
        
        print(f"实验完成，结果已保存到 {csv_path}")

    def writer_thread(self, csv_path, fieldnames, result_queue, stop_event, file_lock):
        """
        专门的写入线程，从队列中读取结果并写入CSV文件
        
        Args:
            csv_path: CSV文件路径
            fieldnames: CSV文件的字段名
            result_queue: 结果队列
            stop_event: 停止事件，用于通知写入线程结束
            file_lock: 线程锁，用于同步文件写入
        """
        # 在写入线程开始时使用固定值
        csv.field_size_limit(2**27 - 1)  # 约等于 134,217,727
        
        while not stop_event.is_set() or not result_queue.empty():
            try:
                # 非阻塞方式获取队列中的结果，超时1秒
                result = result_queue.get(timeout=1)
                
                # 使用文件锁保护写入操作
                with file_lock:
                    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow(result['record'])
                        csvfile.flush()  # 确保立即写入磁盘
                
                # 更新wandb日志
                if 'wandb_data' in result:
                    wandb.log(result['wandb_data'])
                    
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                print(f"写入线程出错: {e}")
                print(f"错误详情: {str(e)}")  # 添加更详细的错误信息

# 添加一个新函数来处理单个模型的实验
def run_single_model_experiment(model_config, data, num_options, config, wandb_key):
    """
    运行单个模型的实验
    """
    try:
        # 测试API连接
        test_model_api(model_config['name'], model_config['api_base'], model_config['api_key'])
        
        # 加载提示模板
        loader = ConfigLoader()
        prompt_template = loader.load_config(model_config['prompt_template'])
        
        # 初始化wandb
        wandb.login(key=wandb_key)
        run_name = f"MCQ_{model_config['name'].replace('/', '_')}_{config['data']['data_path'].replace('/', '_')}"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        wandb.init(
            project="humor_style_mcq",
            name=run_name,
            config={
                "model": model_config['name'],
                "data_path": config['data']['data_path'],
                "sampling_factor": config['data']['scaling_factor'],
                "timestamp": timestamp
            }
        )
        
        # 创建时间戳
        
        
        # 初始化实验
        experiment = MultiChoiceEvaluation(
            model=model_config['name'], 
            api_base=model_config['api_base'], 
            api_key=model_config['api_key'], 
            prompt_template=prompt_template,
            num_options=num_options
        )
        
        # 运行实验
        if model_config['resume']:
            output_path = model_config['resume_path']
            resume = True
        else:
            output_path = os.path.join(
                config['data']['output_folder'],
                f"{model_config['name'].replace('/','_')}_{num_options}options_{timestamp}"
            )
            resume = False
        
        # 获取并行工作进程数量
        num_workers = max(model_config.get('num_workers'),1)
        
        # 使用线程而不是进程来处理数据
        experiment.experiment_with_threads(
            data=data, 
            output_path=output_path, 
            resume=resume,
            num_workers=num_workers
        )
        
        # 完成wandb记录
        wandb.finish()
        
        print(f"模型 {model_config['name']} 实验完成，结果保存到 {output_path}")
        return True
        
    except Exception as e:
        print(f"模型 {model_config['name']} 实验失败: {str(e)}")
        try:
            wandb.finish()
        except:
            pass
        return False

def run_models_in_batch(models_batch, data, num_options, config, wandb_key):
    """
    批量运行一组模型
    """
    # 确保在Windows上正确启动多进程
    multiprocessing.freeze_support()
    
    # 创建进程池，最多使用10个进程
    pool = multiprocessing.Pool(processes=min(len(models_batch), 10))
    
    # 准备参数
    args_list = [(model, data, num_options, config, wandb_key) for model in models_batch]
    
    # 并行运行实验
    results = pool.starmap(run_single_model_experiment, args_list)
    
    # 关闭进程池
    pool.close()
    pool.join()
    
    # 返回成功数量
    return sum(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./Config/understanding_config.json", help="config file path")
    parser.add_argument("--batch_size", type=int, default=10, help="每批运行的模型数量")
    args = parser.parse_args()
    
    loader = ConfigLoader()
    config = loader.load_config(args.config)
    models = config['models']
    
    # 从配置中获取选项数量
    num_options = config['data']['num_options']
    
    # 准备数据集
    data = MultiChoiceEvaluation.prepare_dataset(
        data_path=config['data']['data_path'],
        scaling_factor=config['data']['scaling_factor'],
        seed=config['data']['random_seed']
    )
    
    # wandb API密钥
    wandb_key = "75c71a00697e97575abad4cafddb5cfc37de3305"
    
    # 分批运行模型
    batch_size = args.batch_size
    total_models = len(models)
    num_batches = math.ceil(total_models / batch_size)
    
    print(f"将 {total_models} 个模型分成 {num_batches} 批运行，每批 {batch_size} 个模型")
    
    total_success = 0
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_models)
        current_batch = models[start_idx:end_idx]
        
        print(f"\n开始运行第 {i+1}/{num_batches} 批模型 ({len(current_batch)} 个模型)")
        batch_success = run_models_in_batch(current_batch, data, num_options, config, wandb_key)
        total_success += batch_success
        
        print(f"第 {i+1}/{num_batches} 批完成! 成功: {batch_success}/{len(current_batch)}")
    
    print(f"\n所有批次运行完成! 总成功: {total_success}/{total_models}")



