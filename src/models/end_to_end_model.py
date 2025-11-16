"""
端到端多模态情感分析模型
支持直接处理原始文本，进行完全端到端训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import math

from transformers import AutoModel, AutoConfig
from src.models.audio_model import AudioEncoder
from src.models.video_model import VideoEncoder
from src.models.model import ImprovedFocalLoss, ContrastiveLoss, MultiHeadAttention


class EndToEndTextEncoder(nn.Module):
    """端到端文本编码器，直接处理tokenized输入"""
    
    def __init__(self, pretrained_model="j-hartmann/emotion-english-distilroberta-base", 
                 feature_dim=768, dropout=0.1, freeze_bert=False):
        """
        初始化端到端文本编码器
        
        Args:
            pretrained_model: 预训练模型名称
            feature_dim: 输出特征维度
            dropout: Dropout比率
            freeze_bert: 是否冻结BERT参数
        """
        super(EndToEndTextEncoder, self).__init__()
        
        print(f"🔤 加载端到端文本模型: {pretrained_model}")
        
        # 加载预训练模型
        self.config = AutoConfig.from_pretrained(pretrained_model)
        self.bert = AutoModel.from_pretrained(pretrained_model)
        
        # 冻结参数（如果需要）
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("❄️ BERT参数已冻结")
        else:
            print("🔥 BERT参数可训练")
                
        # 特征投影层
        self.projection = nn.Sequential(
            nn.Linear(self.config.hidden_size, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, input_ids, attention_mask=None):
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs，形状为[batch_size, seq_length]
            attention_mask: 注意力掩码，形状为[batch_size, seq_length]
            
        Returns:
            encoded_features: 编码后的文本特征，形状为[batch_size, feature_dim]
        """
        # 通过BERT提取特征
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用[CLS]标记的表示
        cls_features = outputs.last_hidden_state[:, 0, :]
        
        # 特征投影
        encoded_features = self.projection(cls_features)
        
        return encoded_features


class AdvancedFusionModule(nn.Module):
    """高级融合模块，使用注意力机制"""
    
    def __init__(self, input_dims, hidden_dim, num_heads=8, num_layers=3, dropout=0.1):
        """
        初始化高级融合模块
        
        Args:
            input_dims: 各模态输入维度字典
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            num_layers: 层数
            dropout: Dropout比率
        """
        super(AdvancedFusionModule, self).__init__()
        
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 模态特定投影层
        self.modality_projections = nn.ModuleDict()
        for modality, dim in input_dims.items():
            self.modality_projections[modality] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # 多层交叉注意力
        self.cross_attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.cross_attention_layers.append(
                MultiHeadAttention(hidden_dim, num_heads, dropout)
            )
        
        # 模态间交互层
        self.interaction_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.interaction_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ))
        
        # 融合权重学习
        num_modalities = len(input_dims)
        self.fusion_weights = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=1)
        )
        
    def forward(self, modality_features):
        """
        前向传播
        
        Args:
            modality_features: 各模态特征字典
            
        Returns:
            fused_features: 融合后的特征
        """
        # 投影到统一维度
        projected_features = {}
        for modality, features in modality_features.items():
            if modality in self.modality_projections:
                projected_features[modality] = self.modality_projections[modality](features)
        
        # 转换为列表便于处理
        modality_names = list(projected_features.keys())
        features_list = [projected_features[mod] for mod in modality_names]
        
        # 多层交叉注意力和交互
        for i in range(self.num_layers):
            # 交叉注意力
            attended_features = []
            for j, features in enumerate(features_list):
                # 为注意力添加序列维度
                features_seq = features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
                
                # 与其他模态进行注意力
                for k, other_features in enumerate(features_list):
                    if j != k:
                        other_seq = other_features.unsqueeze(1)
                        attended, _ = self.cross_attention_layers[i](
                            features_seq, other_seq, other_seq
                        )
                        features_seq = features_seq + attended
                
                attended_features.append(features_seq.squeeze(1))
            
            # 交互层
            for j in range(len(features_list)):
                features_list[j] = features_list[j] + self.interaction_layers[i](attended_features[j])
        
        # 计算融合权重
        concatenated = torch.cat(features_list, dim=1)
        weights = self.fusion_weights(concatenated)  # [batch_size, num_modalities]
        
        # 加权融合
        weighted_features = []
        for i, features in enumerate(features_list):
            weight = weights[:, i:i+1]  # [batch_size, 1]
            weighted_features.append(weight * features)
        
        fused_features = sum(weighted_features)
        
        return fused_features


class EndToEndMultiModalModel(nn.Module):
    """端到端多模态情感分析模型"""
    
    def __init__(self, config):
        super(EndToEndMultiModalModel, self).__init__()
        
        self.config = config
        self.enabled_modalities = config['data'].get('enabled_modalities', ['text', 'audio'])
        
        print(f"🚀 初始化端到端多模态模型")
        print(f"📱 启用的模态: {self.enabled_modalities}")
        
        # 文本编码器 - 端到端
        if 'text' in self.enabled_modalities:
            self.text_encoder = EndToEndTextEncoder(
                pretrained_model=config['model']['text']['pretrained_model'],
                feature_dim=config['model']['text']['feature_dim'],
                dropout=config['model']['text']['dropout'],
                freeze_bert=config['model']['text']['freeze_bert']
            )
        
        # 音频编码器
        if 'audio' in self.enabled_modalities:
            self.audio_encoder = AudioEncoder(
                pretrained_model=config['model']['audio']['pretrained_model'],
                feature_dim=config['model']['audio']['feature_dim'],
                dropout=config['model']['audio']['dropout'],
                freeze_encoder=config['model']['audio']['freeze_encoder']
            )
        
        # 视频编码器
        if 'video' in self.enabled_modalities:
            self.video_encoder = VideoEncoder(
                pretrained_model=config['model']['video']['pretrained_model'],
                feature_dim=config['model']['video']['feature_dim'],
                dropout=config['model']['video']['dropout'],
                freeze_encoder=config['model']['video']['freeze_encoder']
            )
        
        # 高级融合模块
        input_dims = {}
        for modality in self.enabled_modalities:
            input_dims[modality] = config['model'][modality]['feature_dim']
        
        fusion_config = config['model']['fusion']
        self.fusion_module = AdvancedFusionModule(
            input_dims=input_dims,
            hidden_dim=fusion_config['hidden_dim'],
            num_heads=8,
            num_layers=fusion_config.get('num_layers', 3),
            dropout=fusion_config['dropout']
        )
        
        # 分类器
        hidden_dim = fusion_config['hidden_dim']
        num_classes = config['model']['output']['num_classes']
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_config['dropout']),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(fusion_config['dropout']),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 损失函数
        training_config = config.get('training', {})
        
        # Focal Loss
        self.focal_loss = ImprovedFocalLoss(
            gamma=training_config.get('focal_loss_gamma', 2.5),
            rare_class_boost=training_config.get('rare_class_boost', True),
            label_smoothing=training_config.get('label_smoothing', 0.08)
        )
        
        # 对比损失
        self.use_contrastive_loss = config.get('model_architecture', {}).get('use_contrastive_loss', True)
        if self.use_contrastive_loss:
            self.contrastive_loss = ContrastiveLoss(
                temperature=training_config.get('contrastive_temperature', 0.12)
            )
            self.contrastive_weight = config.get('model_architecture', {}).get('contrastive_weight', 0.08)
        
    def forward(self, input_ids=None, attention_mask=None, audio_features=None, 
                video_features=None, return_features=False):
        """
        前向传播
        
        Args:
            input_ids: 文本输入IDs
            attention_mask: 文本注意力掩码
            audio_features: 音频特征
            video_features: 视频特征
            return_features: 是否返回中间特征
            
        Returns:
            Dict: 包含logits和可能的中间特征
        """
        modality_features = {}
        
        # 文本编码
        if 'text' in self.enabled_modalities and input_ids is not None:
            text_encoded = self.text_encoder(input_ids, attention_mask)
            modality_features['text'] = text_encoded
        
        # 音频编码
        if 'audio' in self.enabled_modalities and audio_features is not None:
            audio_encoded = self.audio_encoder(audio_features)
            modality_features['audio'] = audio_encoded
        
        # 视频编码
        if 'video' in self.enabled_modalities and video_features is not None:
            video_encoded = self.video_encoder(video_features)
            modality_features['video'] = video_encoded
        
        # 高级融合
        fused_features = self.fusion_module(modality_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        result = {'logits': logits}
        
        if return_features:
            result.update({
                'modality_features': modality_features,
                'fused_features': fused_features
            })
        
        return result
    
    def calculate_losses(self, outputs, labels):
        """
        计算损失
        
        Args:
            outputs: 模型输出
            labels: 真实标签
            
        Returns:
            Dict: 包含各种损失的字典
        """
        logits = outputs['logits']
        
        # 主损失（Focal Loss）
        main_loss = self.focal_loss(logits, labels)
        
        losses = {
            'main_loss': main_loss,
            'total_loss': main_loss
        }
        
        # 对比损失
        if self.use_contrastive_loss and 'modality_features' in outputs:
            modality_features = outputs['modality_features']
            
            # 计算文本-音频对比损失
            if 'text' in modality_features and 'audio' in modality_features:
                contrastive_loss = self.contrastive_loss(
                    modality_features['text'],
                    modality_features['audio'],
                    labels
                )
                losses['contrastive_loss'] = contrastive_loss
                losses['total_loss'] = losses['total_loss'] + self.contrastive_weight * contrastive_loss
        
        return losses


def create_end_to_end_model(config):
    """
    创建端到端模型
    
    Args:
        config: 配置字典
        
    Returns:
        EndToEndMultiModalModel: 端到端模型实例
    """
    return EndToEndMultiModalModel(config)


if __name__ == "__main__":
    # 测试模型
    import json
    
    # 测试配置
    test_config = {
        "data": {
            "enabled_modalities": ["text", "audio"]
        },
        "model": {
            "text": {
                "pretrained_model": "j-hartmann/emotion-english-distilroberta-base",
                "feature_dim": 768,
                "dropout": 0.1,
                "freeze_bert": False
            },
            "audio": {
                "pretrained_model": "facebook/wav2vec2-base-960h",
                "feature_dim": 768,
                "dropout": 0.1,
                "freeze_encoder": False
            },
            "fusion": {
                "hidden_dim": 512,
                "dropout": 0.1,
                "num_layers": 2
            },
            "output": {
                "num_classes": 7
            }
        },
        "model_architecture": {
            "use_contrastive_loss": True,
            "contrastive_weight": 0.1
        },
        "training": {
            "focal_loss_gamma": 2.5,
            "rare_class_boost": True,
            "label_smoothing": 0.08,
            "contrastive_temperature": 0.12
        }
    }
    
    # 创建模型
    model = create_end_to_end_model(test_config)
    
    # 测试输入
    batch_size = 4
    seq_len = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    audio_features = torch.randn(batch_size, 768)
    labels = torch.randint(0, 7, (batch_size,))
    
    # 前向传播
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        audio_features=audio_features,
        return_features=True
    )
    
    # 计算损失
    losses = model.calculate_losses(outputs, labels)
    
    print(f"✅ 模型测试成功!")
    print(f"📊 输出logits形状: {outputs['logits'].shape}")
    print(f"📉 总损失: {losses['total_loss'].item():.4f}")
    print(f"🎯 主损失: {losses['main_loss'].item():.4f}")
    if 'contrastive_loss' in losses:
        print(f"🔗 对比损失: {losses['contrastive_loss'].item():.4f}") 