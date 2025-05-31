"""
端到端多模态数据集
支持直接处理原始文本，进行端到端训练
"""
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from src.preprocess.dataset import DataAugmentation

class EndToEndMELDDataset(Dataset):
    """端到端MELD数据集类，支持原始文本处理"""
    
    def __init__(self, dataset_path, split='train', audio_feature_path=None, 
                 video_feature_path=None, text_model_name="roberta-base",
                 max_seq_len=128, transform=None, emotion_mapping=None, config=None):
        """
        初始化EndToEndMELDDataset
        
        Args:
            dataset_path: MELD数据集路径
            split: 数据划分（'train', 'dev', 'test'）
            audio_feature_path: 音频特征文件路径
            video_feature_path: 视频特征文件路径
            text_model_name: 文本模型名称，用于tokenizer
            max_seq_len: 最大序列长度
            transform: 特征变换函数
            emotion_mapping: 情感标签映射字典
            config: 配置字典，包含数据增强参数
        """
        self.dataset_path = dataset_path
        self.split = split
        self.transform = transform
        self.max_seq_len = max_seq_len
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 初始化数据增强器
        if config is not None:
            self.augmenter = DataAugmentation(config)
            # 只在训练集上应用数据增强
            self.apply_augmentation = (split == 'train')
        else:
            self.augmenter = None
            self.apply_augmentation = False
        
        # 情感标签映射
        if emotion_mapping is None:
            self.emotion_mapping = {
                'neutral': 0, 'joy': 1, 'sadness': 2, 'anger': 3, 
                'surprise': 4, 'fear': 5, 'disgust': 6
            }
        else:
            self.emotion_mapping = emotion_mapping
            
        # 加载CSV文件
        self.csv_path = os.path.join(dataset_path, f"{split}_sent_emo.csv")
        self.df = pd.read_csv(self.csv_path)
        
        # 说话人映射
        unique_speakers = self.df['Speaker'].unique()
        self.speaker_mapping = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
        self.num_speakers = len(unique_speakers)
        print(f"发现 {self.num_speakers} 个不同的说话人: {list(unique_speakers)[:10]}{'...' if len(unique_speakers) > 10 else ''}")
        
        # 加载非文本特征
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
            # 检查文本是否有效
            utterance = self.df.iloc[i]['Utterance']
            if pd.isna(utterance) or len(str(utterance).strip()) == 0:
                continue
                
            # 检查其他特征是否有效
            if ((self.audio_features is None or i < len(self.audio_features['features'])) and
                (self.video_features is None or i < len(self.video_features['features']))):
                valid_indices.append(i)
        return valid_indices
        
    def tokenize_text(self, text):
        """对文本进行tokenization"""
        # 清理文本
        text = str(text).strip()
        if len(text) == 0:
            text = "[UNK]"  # 空文本处理
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }
        
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
        
        # 处理文本 - 端到端方式
        utterance = row['Utterance']
        tokenized = self.tokenize_text(utterance)
        
        # 提取其他模态特征
        audio_feature = self.audio_features['features'][valid_idx] if self.audio_features else np.zeros((1, 768))
        video_feature = self.video_features['features'][valid_idx] if self.video_features else np.zeros((1, 768))
        
        # 转换为PyTorch张量
        audio_feature = torch.tensor(audio_feature, dtype=torch.float32).squeeze(0)
        video_feature = torch.tensor(video_feature, dtype=torch.float32).squeeze(0)
        
        # 应用变换
        if self.transform:
            audio_feature = self.transform(audio_feature)
            video_feature = self.transform(video_feature)
            
        # 应用数据增强（只对非文本特征）
        if self.apply_augmentation and self.augmenter:
            audio_feature = self.augmenter.apply_augmentation(audio_feature)
            video_feature = self.augmenter.apply_augmentation(video_feature)
            
        # 创建样本字典
        sample = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'audio': audio_feature,
            'video': video_feature,
            'emotion': torch.tensor(emotion_id, dtype=torch.long),
            'speaker_id': torch.tensor(self.speaker_mapping[row['Speaker']], dtype=torch.long),
            'dia_id': row['Dialogue_ID'],
            'utt_id': row['Utterance_ID'],
            'utterance': utterance,
            'speaker': row['Speaker']
        }
        
        return sample


def get_end_to_end_dataloaders(config):
    """
    获取端到端MELD数据集的DataLoader
    
    Args:
        config: 配置字典
        
    Returns:
        dataloaders: 包含训练集、验证集和测试集DataLoader的字典
    """
    dataset_path = config['data']['dataset_path']
    cache_path = config['data']['cache_path']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    max_seq_len = config['data']['max_seq_len']
    text_model = config['model']['text']['pretrained_model']
    
    # 获取启用的模态列表
    enabled_modalities = config['data'].get('enabled_modalities', ['text', 'audio', 'video'])
    
    print(f"端到端训练启用的模态: {enabled_modalities}")
    print(f"文本模型: {text_model}")
    print(f"最大序列长度: {max_seq_len}")
    
    # 音频和视频特征路径（如果需要）
    audio_features = {
        'train': os.path.join(cache_path, 'audio_features', 'train_sent_emo_audio_features.pt'),
        'dev': os.path.join(cache_path, 'audio_features', 'dev_sent_emo_audio_features.pt'),
        'test': os.path.join(cache_path, 'audio_features', 'test_sent_emo_audio_features.pt')
    } if 'audio' in enabled_modalities else {'train': None, 'dev': None, 'test': None}
    
    video_features = {
        'train': os.path.join(cache_path, 'video_features', 'train_sent_emo_video_features.pt'),
        'dev': os.path.join(cache_path, 'video_features', 'dev_sent_emo_video_features.pt'),
        'test': os.path.join(cache_path, 'video_features', 'test_sent_emo_video_features.pt')
    } if 'video' in enabled_modalities else {'train': None, 'dev': None, 'test': None}
    
    # 创建数据集
    train_dataset = EndToEndMELDDataset(
        dataset_path=dataset_path,
        split='train',
        audio_feature_path=audio_features['train'],
        video_feature_path=video_features['train'],
        text_model_name=text_model,
        max_seq_len=max_seq_len,
        config=config
    )
    
    dev_dataset = EndToEndMELDDataset(
        dataset_path=dataset_path,
        split='dev',
        audio_feature_path=audio_features['dev'],
        video_feature_path=video_features['dev'],
        text_model_name=text_model,
        max_seq_len=max_seq_len,
        config=config
    )
    
    test_dataset = EndToEndMELDDataset(
        dataset_path=dataset_path,
        split='test',
        audio_feature_path=audio_features['test'],
        video_feature_path=video_features['test'],
        text_model_name=text_model,
        max_seq_len=max_seq_len,
        config=config
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 端到端训练时保持批次大小一致
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
    # 测试数据集
    import json
    
    # 加载配置
    config_path = "configs/config_end_to_end_emotion.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 创建数据加载器
        dataloaders = get_end_to_end_dataloaders(config)
        
        # 测试一个批次
        train_loader = dataloaders['train']
        for batch in train_loader:
            print(f"输入IDs形状: {batch['input_ids'].shape}")
            print(f"注意力掩码形状: {batch['attention_mask'].shape}")
            print(f"音频特征形状: {batch['audio'].shape}")
            print(f"情感标签: {batch['emotion']}")
            print(f"样本文本: {batch['utterance'][0]}")
            break
    else:
        print(f"配置文件不存在: {config_path}") 