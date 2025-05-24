"""
多模态融合情感分析模型
基于高效的融合技术实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from src.models.text_model import TextEncoder
from src.models.audio_model import AudioEncoder
from src.models.video_model import VideoEncoder


class FocalLoss(nn.Module):
    """改进的Focal Loss"""
    
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ModalitySpecificEncoder(nn.Module):
    """模态特定编码器"""
    
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(ModalitySpecificEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.encoder(x)


class FusionModel(nn.Module):
    """多模态融合情感分析模型"""
    
    def __init__(self, config):
        super(FusionModel, self).__init__()
        
        self.config = config
        
        # 基础编码器
        self.text_encoder = TextEncoder(
            pretrained_model=config['model']['text']['pretrained_model'],
            feature_dim=config['model']['text']['feature_dim'],
            dropout=config['model']['text']['dropout'],
            freeze_bert=config['model']['text']['freeze_bert'],
            use_local=True,
            local_model_dir="models"
        )
        
        self.audio_encoder = AudioEncoder(
            pretrained_model=config['model']['audio']['pretrained_model'],
            feature_dim=config['model']['audio']['feature_dim'],
            dropout=config['model']['audio']['dropout'],
            freeze_encoder=config['model']['audio']['freeze_encoder'],
            use_local=True,
            local_model_dir="models"
        )
        
        self.video_encoder = VideoEncoder(
            pretrained_model=config['model']['video']['pretrained_model'],
            feature_dim=config['model']['video']['feature_dim'],
            dropout=config['model']['video']['dropout'],
            freeze_encoder=config['model']['video']['freeze_encoder'],
            use_local=True,
            local_model_dir="models"
        )
        
        # 模态特定编码器
        hidden_dim = config['model']['fusion']['hidden_dim']
        dropout = config['model']['fusion']['dropout']
        
        self.modality_encoders = nn.ModuleDict({
            'text': ModalitySpecificEncoder(config['model']['text']['feature_dim'], hidden_dim, dropout),
            'audio': ModalitySpecificEncoder(config['model']['audio']['feature_dim'], hidden_dim, dropout),
            'video': ModalitySpecificEncoder(config['model']['video']['feature_dim'], hidden_dim, dropout)
        })
        
        # 单模态分类器（用于集成）
        self.single_modal_classifiers = nn.ModuleDict({
            'text': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, config['model']['output']['num_classes'])
            ),
            'audio': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, config['model']['output']['num_classes'])
            ),
            'video': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, config['model']['output']['num_classes'])
            )
        })
        
        # 融合分类器
        self.fusion_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, config['model']['output']['num_classes'])
        )
        
        # 动态权重学习
        self.fusion_weights = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # 3个单模态 + 1个融合
            nn.Softmax(dim=1)
        )
        
        # Focal Loss
        self.focal_loss = FocalLoss(gamma=2)
        
    def forward(self, text_features, audio_features, video_features):
        """前向传播"""
        
        # 基础特征编码
        text_encoded = self.text_encoder(text_features)
        audio_encoded = self.audio_encoder(audio_features)
        video_encoded = self.video_encoder(video_features)
        
        # 模态特定编码
        text_modal = self.modality_encoders['text'](text_encoded)
        audio_modal = self.modality_encoders['audio'](audio_encoded)
        video_modal = self.modality_encoders['video'](video_encoded)
        
        # 单模态预测
        text_logits = self.single_modal_classifiers['text'](text_modal)
        audio_logits = self.single_modal_classifiers['audio'](audio_modal)
        video_logits = self.single_modal_classifiers['video'](video_modal)
        
        # 融合特征
        fused_features = torch.cat([text_modal, audio_modal, video_modal], dim=1)
        fusion_logits = self.fusion_classifier(fused_features)
        
        # 动态权重
        weights = self.fusion_weights(fused_features)
        
        # 加权融合预测
        final_logits = (weights[:, 0:1] * text_logits + 
                       weights[:, 1:2] * audio_logits + 
                       weights[:, 2:3] * video_logits + 
                       weights[:, 3:4] * fusion_logits)
        
        return {
            'logits': final_logits,
            'text_logits': text_logits,
            'audio_logits': audio_logits,
            'video_logits': video_logits,
            'fusion_logits': fusion_logits,
            'weights': weights,
            'text_features': text_modal,
            'audio_features': audio_modal,
            'video_features': video_modal
        }
    
    def calculate_losses(self, output, labels):
        """计算损失"""
        
        # 主要损失：最终预测
        main_loss = self.focal_loss(output['logits'], labels)
        
        # 辅助损失：单模态预测
        text_loss = F.cross_entropy(output['text_logits'], labels)
        audio_loss = F.cross_entropy(output['audio_logits'], labels)
        video_loss = F.cross_entropy(output['video_logits'], labels)
        fusion_loss = F.cross_entropy(output['fusion_logits'], labels)
        
        # 总损失
        total_loss = (main_loss + 
                     0.3 * text_loss + 
                     0.3 * audio_loss + 
                     0.3 * video_loss + 
                     0.3 * fusion_loss)
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'text_loss': text_loss,
            'audio_loss': audio_loss,
            'video_loss': video_loss,
            'fusion_loss': fusion_loss
        } 