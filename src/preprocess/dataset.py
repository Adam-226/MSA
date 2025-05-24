"""
MELD数据集加载和处理模块
"""
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class MELDDataset(Dataset):
    """MELD数据集类"""
    
    def __init__(self, dataset_path, split='train', text_feature_path=None, audio_feature_path=None, 
                 video_feature_path=None, transform=None, emotion_mapping=None):
        """
        初始化MELDDataset
        
        Args:
            dataset_path: MELD数据集路径
            split: 数据划分（'train', 'dev', 'test'）
            text_feature_path: 文本特征文件路径
            audio_feature_path: 音频特征文件路径
            video_feature_path: 视频特征文件路径
            transform: 特征变换函数
            emotion_mapping: 情感标签映射字典
        """
        self.dataset_path = dataset_path
        self.split = split
        self.transform = transform
        
        # 情感标签映射
        if emotion_mapping is None:
            self.emotion_mapping = {
                'neutral': 0,
                'joy': 1,
                'sadness': 2,
                'anger': 3,
                'surprise': 4,
                'fear': 5,
                'disgust': 6
            }
        else:
            self.emotion_mapping = emotion_mapping
            
        # 加载CSV文件
        self.csv_path = os.path.join(dataset_path, f"{split}_sent_emo.csv")
        self.df = pd.read_csv(self.csv_path)
        
        # 创建DialogID_UtteranceID到索引的映射
        self.dia_utt_to_idx = {}
        for i, row in self.df.iterrows():
            dia_id = row['Dialogue_ID']
            utt_id = row['Utterance_ID']
            dia_utt_id = f"dia{dia_id}_utt{utt_id}"
            self.dia_utt_to_idx[dia_utt_id] = i
            
        # 加载特征
        self.text_features = self.load_features(text_feature_path)
        self.audio_features = self.load_features(audio_feature_path)
        self.video_features = self.load_features(video_feature_path)
        
        # 检查有效样本
        self.valid_indices = self.get_valid_indices()
        
        print(f"加载MELD {split}集完成，共{len(self.valid_indices)}个有效样本，总样本数：{len(self.df)}")
        
    def load_features(self, feature_path):
        """加载特征"""
        if feature_path and os.path.exists(feature_path):
            print(f"加载特征: {feature_path}")
            return torch.load(feature_path, weights_only=False)
        return None
        
    def get_valid_indices(self):
        """获取有效样本索引"""
        valid_indices = []
        for i in range(len(self.df)):
            # 如果所有特征都有效，则添加到有效索引列表
            if ((self.text_features is None or i < len(self.text_features['features'])) and
                (self.audio_features is None or i < len(self.audio_features['features'])) and
                (self.video_features is None or i < len(self.video_features['features']))):
                valid_indices.append(i)
        return valid_indices
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.valid_indices)
        
    def __getitem__(self, idx):
        """获取样本"""
        # 获取有效索引
        valid_idx = self.valid_indices[idx]
        
        # 获取原始数据
        row = self.df.iloc[valid_idx]
        
        # 提取情感标签
        emotion = row['Emotion']
        emotion_id = self.emotion_mapping.get(emotion.lower(), 0)
        
        # 提取特征
        text_feature = self.text_features['features'][valid_idx] if self.text_features else np.zeros((1, 768))
        audio_feature = self.audio_features['features'][valid_idx] if self.audio_features else np.zeros((1, 768))
        video_feature = self.video_features['features'][valid_idx] if self.video_features else np.zeros((1, 768))
        
        # 转换为PyTorch张量
        text_feature = torch.tensor(text_feature, dtype=torch.float32).squeeze(0)
        audio_feature = torch.tensor(audio_feature, dtype=torch.float32).squeeze(0)
        video_feature = torch.tensor(video_feature, dtype=torch.float32).squeeze(0)
        
        # 应用变换
        if self.transform:
            text_feature = self.transform(text_feature)
            audio_feature = self.transform(audio_feature)
            video_feature = self.transform(video_feature)
            
        # 创建样本字典
        sample = {
            'text': text_feature,
            'audio': audio_feature,
            'video': video_feature,
            'emotion': torch.tensor(emotion_id, dtype=torch.long),
            'dia_id': row['Dialogue_ID'],
            'utt_id': row['Utterance_ID'],
            'utterance': row['Utterance'],
            'speaker': row['Speaker']
        }
        
        return sample


def get_meld_dataloaders(config):
    """
    获取MELD数据集的DataLoader
    
    Args:
        config: 配置字典
        
    Returns:
        dataloaders: 包含训练集、验证集和测试集DataLoader的字典
    """
    dataset_path = config['data']['dataset_path']
    cache_path = config['data']['cache_path']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    
    # 特征路径
    text_features = {
        'train': os.path.join(cache_path, 'text_features', 'train_sent_emo_text_features.pt'),
        'dev': os.path.join(cache_path, 'text_features', 'dev_sent_emo_text_features.pt'),
        'test': os.path.join(cache_path, 'text_features', 'test_sent_emo_text_features.pt')
    }
    
    audio_features = {
        'train': os.path.join(cache_path, 'audio_features', 'train_sent_emo_audio_features.pt'),
        'dev': os.path.join(cache_path, 'audio_features', 'dev_sent_emo_audio_features.pt'),
        'test': os.path.join(cache_path, 'audio_features', 'test_sent_emo_audio_features.pt')
    }
    
    video_features = {
        'train': os.path.join(cache_path, 'video_features', 'train_sent_emo_video_features.pt'),
        'dev': os.path.join(cache_path, 'video_features', 'dev_sent_emo_video_features.pt'),
        'test': os.path.join(cache_path, 'video_features', 'test_sent_emo_video_features.pt')
    }
    
    # 创建数据集
    train_dataset = MELDDataset(
        dataset_path=dataset_path,
        split='train',
        text_feature_path=text_features['train'],
        audio_feature_path=audio_features['train'],
        video_feature_path=video_features['train']
    )
    
    dev_dataset = MELDDataset(
        dataset_path=dataset_path,
        split='dev',
        text_feature_path=text_features['dev'],
        audio_feature_path=audio_features['dev'],
        video_feature_path=video_features['dev']
    )
    
    test_dataset = MELDDataset(
        dataset_path=dataset_path,
        split='test',
        text_feature_path=text_features['test'],
        audio_feature_path=audio_features['test'],
        video_feature_path=video_features['test']
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'dev': dev_loader,
        'test': test_loader
    }


if __name__ == "__main__":
    # 示例用法
    dataset_path = "data/MELD.Raw"
    cache_path = "data/processed"
    
    # 特征路径
    text_feature_path = os.path.join(cache_path, 'text_features', 'train_sent_emo_text_features.pt')
    audio_feature_path = os.path.join(cache_path, 'audio_features', 'train_sent_emo_audio_features.pt')
    video_feature_path = os.path.join(cache_path, 'video_features', 'train_sent_emo_video_features.pt')
    
    # 检查特征文件是否存在
    if os.path.exists(text_feature_path) and os.path.exists(audio_feature_path) and os.path.exists(video_feature_path):
        # 创建数据集
        dataset = MELDDataset(
            dataset_path=dataset_path,
            split='train',
            text_feature_path=text_feature_path,
            audio_feature_path=audio_feature_path,
            video_feature_path=video_feature_path
        )
        
        # 创建DataLoader
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # 获取一个批次
        for batch in dataloader:
            print(f"文本特征形状: {batch['text'].shape}")
            print(f"音频特征形状: {batch['audio'].shape}")
            print(f"视频特征形状: {batch['video'].shape}")
            print(f"情感标签形状: {batch['emotion'].shape}")
            print(f"情感标签: {batch['emotion']}")
            break
    else:
        print("特征文件不存在，请先运行特征提取脚本") 