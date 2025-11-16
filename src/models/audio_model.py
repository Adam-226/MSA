"""
音频模型模块
使用预训练模型处理音频数据
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
import os

class AudioEncoder(nn.Module):
    """音频编码器类"""
    
    def __init__(self, pretrained_model="facebook/wav2vec2-base-960h", feature_dim=768, dropout=0.1, freeze_encoder=True, use_local=True, local_model_dir="models"):
        """
        初始化AudioEncoder
        
        Args:
            pretrained_model: 预训练模型名称，默认为"facebook/wav2vec2-base-960h"
            feature_dim: 输出特征维度，默认为768
            dropout: dropout概率，默认为0.1
            freeze_encoder: 是否冻结编码器参数，默认为True
            use_local: 是否使用本地预训练模型，默认为True
            local_model_dir: 本地预训练模型目录，默认为"models"
        """
        super(AudioEncoder, self).__init__()
        
        # 构建本地模型路径
        if use_local:
            model_folder = os.path.basename(pretrained_model.rstrip("/"))
            local_path = os.path.join(local_model_dir, model_folder)
            if os.path.exists(local_path):
                pretrained_model = local_path
                print(f"使用本地音频预训练模型: {local_path}")
            else:
                print(f"警告：本地音频模型路径 {local_path} 不存在，尝试在线下载。")
        
        # 加载预训练模型配置
        self.config = Wav2Vec2Config.from_pretrained(pretrained_model, local_files_only=use_local)
        
        # 加载预训练模型
        try:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model, local_files_only=use_local)
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            print("使用随机初始化的模型")
            self.wav2vec2 = Wav2Vec2Model(self.config)
        
        # 冻结预训练模型参数
        if freeze_encoder:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
                
        # 特征转换层
        self.fc = nn.Sequential(
            nn.Linear(self.config.hidden_size, feature_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
    def forward(self, features):
        """
        前向传播
        
        Args:
            features: 音频特征，形状为[batch_size, feature_dim]
            
        Returns:
            output: 编码后的音频特征，形状为[batch_size, feature_dim]
        """
        # 如果输入是原始特征（不是通过wav2vec2提取的），直接使用特征转换层
        output = self.fc(features)
        
        return output
        
    def extract_features(self, waveform, attention_mask=None):
        """
        从原始音频中提取特征
        
        Args:
            waveform: 音频波形，形状为[batch_size, seq_length]
            attention_mask: 注意力掩码，形状为[batch_size, seq_length]
            
        Returns:
            features: 提取的音频特征，形状为[batch_size, feature_dim]
        """
        # 提取wav2vec2特征
        outputs = self.wav2vec2(input_values=waveform, attention_mask=attention_mask)
        
        # 获取最后一层隐藏状态
        hidden_states = outputs.last_hidden_state
        
        # 对时间维度进行平均池化，得到单一特征向量
        features = torch.mean(hidden_states, dim=1)
        
        # 特征转换
        features = self.fc(features)
        
        return features


class AudioClassifier(nn.Module):
    """音频分类器类"""
    
    def __init__(self, encoder_model="facebook/wav2vec2-base-960h", feature_dim=768, num_classes=7, dropout=0.1, freeze_encoder=True, use_local=True, local_model_dir="models"):
        """
        初始化AudioClassifier
        
        Args:
            encoder_model: 编码器模型名称，默认为wav2vec2-base-960h
            feature_dim: 特征维度，默认为768
            num_classes: 类别数量，默认为7（MELD数据集的情感类别数）
            dropout: Dropout比率，默认为0.1
            freeze_encoder: 是否冻结预训练模型参数，默认为True
            use_local: 是否使用本地预训练模型，默认为True
            local_model_dir: 本地预训练模型目录，默认为"models"
        """
        super(AudioClassifier, self).__init__()
        
        # 音频编码器
        self.encoder = AudioEncoder(
            pretrained_model=encoder_model,
            feature_dim=feature_dim,
            dropout=dropout,
            freeze_encoder=freeze_encoder,
            use_local=use_local,
            local_model_dir=local_model_dir
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
    def forward(self, features):
        """
        前向传播
        
        Args:
            features: 音频特征，形状为[batch_size, feature_dim]
            
        Returns:
            logits: 类别logits，形状为[batch_size, num_classes]
        """
        # 编码音频特征
        encoded = self.encoder(features)
        
        # 分类
        logits = self.classifier(encoded)
        
        return logits
        
    def extract_and_classify(self, waveform, attention_mask=None):
        """
        从原始音频中提取特征并分类
        
        Args:
            waveform: 音频波形，形状为[batch_size, seq_length]
            attention_mask: 注意力掩码，形状为[batch_size, seq_length]
            
        Returns:
            logits: 类别logits，形状为[batch_size, num_classes]
        """
        # 提取特征
        features = self.encoder.extract_features(waveform, attention_mask)
        
        # 分类
        logits = self.classifier(features)
        
        return logits


if __name__ == "__main__":
    # 示例用法
    
    # 创建编码器
    encoder = AudioEncoder(pretrained_model="facebook/wav2vec2-base-960h")
    
    # 随机生成特征
    batch_size = 16
    feature_dim = 768
    features = torch.randn(batch_size, feature_dim)
    
    # 编码特征
    encoded = encoder(features)
    print(f"编码后的特征形状: {encoded.shape}")
    
    # 创建分类器
    classifier = AudioClassifier(encoder_model="facebook/wav2vec2-base-960h", num_classes=7)
    
    # 分类
    logits = classifier(features)
    print(f"分类logits形状: {logits.shape}") 