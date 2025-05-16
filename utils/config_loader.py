import yaml
import json
import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """模型相关配置"""
    model_name: str
    batch_size: int
    learning_rate: float
    # 添加其他模型参数...

@dataclass
class DataConfig:
    """数据相关配置"""
    data_path: str
    train_ratio: float
    val_ratio: float
    # 添加其他数据参数...

@dataclass
class Config:
    """总配置类"""
    model: ModelConfig
    data: DataConfig
    # 可以添加其他配置类...

class ConfigLoader:
    @staticmethod
    def load_config(config_path):
        """加载YAML或JSON配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件 {config_path} 不存在")
        
        file_ext = os.path.splitext(config_path)[1].lower()
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if file_ext == '.json':
                    return json.load(f)
                elif file_ext in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {file_ext}")
        except Exception as e:
            raise Exception(f"加载配置文件 {config_path} 时出错: {e}")
    


