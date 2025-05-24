"""
训练配置模块
"""
import os
import json
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any, Optional

def set_seed(seed: int) -> None:
    """
    设置随机种子，确保实验可重复
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置字典
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    保存配置文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
        
def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    获取设备（CPU/GPU）
    
    Args:
        gpu_id: GPU ID，默认为None（使用第一个可用的GPU）
        
    Returns:
        device: PyTorch设备
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    return device

def get_class_weights(dataset_path: str, split: str = 'train') -> torch.Tensor:
    """
    计算类别权重，用于处理类别不平衡，特别加强极少数类别
    
    Args:
        dataset_path: 数据集路径
        split: 数据划分，默认为'train'
        
    Returns:
        class_weights: 类别权重张量
    """
    import pandas as pd
    import numpy as np
    
    # 读取CSV文件
    csv_path = os.path.join(dataset_path, f"{split}_sent_emo.csv")
    df = pd.read_csv(csv_path)
    
    # 统计每个类别的样本数
    emotion_counts = df['Emotion'].value_counts().to_dict()
    
    # 情感映射
    emotion_mapping = {
        'neutral': 0,
        'joy': 1,
        'sadness': 2,
        'anger': 3,
        'surprise': 4,
        'fear': 5,
        'disgust': 6
    }
    
    # 计算增强的类别权重
    total_samples = len(df)
    n_classes = len(emotion_mapping)
    class_weights = torch.zeros(n_classes)
    
    # 获取所有类别的样本数
    counts = []
    for emotion, idx in emotion_mapping.items():
        count = emotion_counts.get(emotion, 1)  # 避免除零
        counts.append(count)
    
    max_count = max(counts)
    
    for emotion, idx in emotion_mapping.items():
        count = emotion_counts.get(emotion, 1)
        
        # 基础权重：反比例
        base_weight = max_count / count
        
        # 对极少数类别进行特别加强
        if count < 100:  # Fear和Disgust等极少数类别
            # 使用对数缩放 + 额外惩罚因子
            boost_factor = np.log(100 / count) * 3  # 大幅提升权重
            enhanced_weight = base_weight * (1 + boost_factor)
        elif count < 300:  # 较少类别如Sadness
            boost_factor = np.log(300 / count) * 1.5
            enhanced_weight = base_weight * (1 + boost_factor)
        else:
            enhanced_weight = base_weight
        
        # 设置最小权重阈值，确保主要类别不会权重过小
        enhanced_weight = max(enhanced_weight, 0.5)
        
        class_weights[idx] = enhanced_weight
    
    print(f"增强类别权重: {class_weights}")
    print(f"类别样本分布: {emotion_counts}")
    
    return class_weights

def create_experiment_dir(config: Dict[str, Any]) -> str:
    """
    创建实验目录
    
    Args:
        config: 配置字典
        
    Returns:
        exp_dir: 实验目录路径
    """
    # 基础目录
    checkpoint_dir = config['paths']['checkpoint_dir']
    
    # 创建实验目录
    model_name = config['model_name']
    exp_name = f"{model_name}_{config['random_seed']}"
    exp_dir = os.path.join(checkpoint_dir, exp_name)
    
    # 创建目录
    os.makedirs(exp_dir, exist_ok=True)
    
    # 保存配置
    save_config(config, os.path.join(exp_dir, 'config.json'))
    
    return exp_dir 