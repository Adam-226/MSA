"""
多模态融合情感分析模型
基于高效的融合技术实现，包含数据增强、深度架构和改进损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import math

from src.models.text_model import TextEncoder
from src.models.audio_model import AudioEncoder
from src.models.video_model import VideoEncoder


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换和reshape
        Q = self.w_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 输出投影和残差连接
        output = self.w_o(context)
        output = self.layer_norm(output + query)
        
        return output, attn_weights


class DeepModalityEncoder(nn.Module):
    """深度模态编码器"""
    
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.1, use_attention=True):
        super(DeepModalityEncoder, self).__init__()
        
        self.use_attention = use_attention
        self.num_layers = num_layers
        
        # 多层编码器
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(nn.Sequential(
                nn.Linear(layer_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),  # 使用GELU激活函数
                nn.Dropout(dropout)
            ))
        
        # 注意力层
        if use_attention:
            self.attention = MultiHeadAttention(hidden_dim, nhead=8, dropout=dropout)
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # 逐层编码
        for layer in self.layers:
            x = layer(x)
        
        # 应用注意力（如果启用）
        if self.use_attention:
            # 为了使用注意力，需要添加序列维度
            x_with_seq = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            attended, _ = self.attention(x_with_seq, x_with_seq, x_with_seq)
            x = attended.squeeze(1)  # [batch_size, hidden_dim]
        
        # 输出投影
        x = self.output_projection(x)
        return x


class ContrastiveLoss(nn.Module):
    """对比损失函数"""
    
    def __init__(self, temperature=0.1, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, features1, features2, labels):
        """
        计算对比损失
        features1, features2: 两个模态的特征 [batch_size, feature_dim]
        labels: 标签 [batch_size]
        """
        batch_size = features1.size(0)
        
        # 归一化特征
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        
        # 计算相似度
        similarity = torch.mm(features1, features2.t()) / self.temperature
        
        # 创建正样本掩码（相同标签）
        labels_expanded = labels.unsqueeze(1).expand(batch_size, batch_size)
        positive_mask = (labels_expanded == labels_expanded.t()).float()
        
        # 移除对角线（自己和自己）
        positive_mask = positive_mask - torch.eye(batch_size).to(positive_mask.device)
        
        # 计算对比损失
        exp_sim = torch.exp(similarity)
        positive_sum = (exp_sim * positive_mask).sum(dim=1)
        total_sum = exp_sim.sum(dim=1)
        
        # 避免除零
        positive_sum = torch.clamp(positive_sum, min=1e-8)
        loss = -torch.log(positive_sum / total_sum)
        
        return loss.mean()


class ImprovedFocalLoss(nn.Module):
    """改进的Focal Loss，支持标签平滑和类别权重"""
    
    def __init__(self, alpha=None, gamma=2, label_smoothing=0.1, reduction='mean', rare_class_boost=True):
        super(ImprovedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.rare_class_boost = rare_class_boost
        
        # 定义极少数类别的索引 (fear=5, disgust=6)
        self.rare_classes = [5, 6]
        
    def forward(self, inputs, targets):
        # 标签平滑
        if self.label_smoothing > 0:
            num_classes = inputs.size(1)
            smoothed_targets = torch.zeros_like(inputs)
            smoothed_targets.fill_(self.label_smoothing / (num_classes - 1))
            smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            
            # 使用KL散度计算损失
            log_probs = F.log_softmax(inputs, dim=1)
            loss = -torch.sum(smoothed_targets * log_probs, dim=1)
        else:
            # 标准交叉熵损失
            loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算概率
        pt = torch.exp(-loss)
        
        # 基础focal loss
        focal_loss = (1 - pt) ** self.gamma * loss
        
        # 对极少数类别进行特殊增强
        if self.rare_class_boost:
            # 创建增强掩码
            rare_mask = torch.zeros_like(targets, dtype=torch.float)
            for rare_idx in self.rare_classes:
                rare_mask += (targets == rare_idx).float()
            
            # 对极少数类别使用更大的gamma值和额外权重
            enhanced_gamma = self.gamma * 1.8  # 进一步增强
            enhanced_focal = (1 - pt) ** enhanced_gamma * loss
            
            # 额外的类别特定权重
            rare_boost = 4.0  # 极少数类别额外4倍权重
            
            # 混合损失
            focal_loss = (1 - rare_mask) * focal_loss + rare_mask * enhanced_focal * rare_boost
        
        # 应用alpha权重
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


class DialogueContextModule(nn.Module):
    """简化的对话上下文模块 - 减少复杂度"""
    
    def __init__(self, feature_dim, speaker_dim=32, context_dim=64, dropout=0.1):  # 减少隐藏维度
        super(DialogueContextModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.speaker_dim = speaker_dim
        self.context_dim = context_dim
        
        # 简化的说话人嵌入
        self.speaker_embedding = nn.Embedding(300, speaker_dim)  # 降低维度
        
        # 简化的说话人状态GRU - 单层
        self.speaker_gru = nn.GRU(
            input_size=feature_dim + speaker_dim,
            hidden_size=context_dim,
            num_layers=1,  # 减少层数
            batch_first=True,
            dropout=0 if 1 == 1 else dropout  # 单层不需要dropout
        )
        
        # 简化的全局上下文GRU - 单层
        self.global_gru = nn.GRU(
            input_size=context_dim,
            hidden_size=context_dim,
            num_layers=1,  # 减少层数
            batch_first=True,
            dropout=0 if 1 == 1 else dropout
        )
        
        # 简化的情绪状态更新网络
        self.emotion_update = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),  # 减少隐藏层大小
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim // 2, feature_dim)
        )
        
        # 移除复杂的多头注意力，使用简单的线性层
        self.context_fusion = nn.Sequential(
            nn.Linear(feature_dim + context_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 初始化隐藏状态
        self.speaker_hidden = None
        self.global_hidden = None
        
    def reset_states(self):
        """重置所有隐藏状态"""
        self.speaker_hidden = None
        self.global_hidden = None
        
    def forward(self, features, speaker_ids=None, use_context_attention=True):
        """
        前向传播
        
        Args:
            features: 输入特征 [batch_size, feature_dim]
            speaker_ids: 说话人ID [batch_size]
            use_context_attention: 是否使用上下文注意力
            
        Returns:
            context_features: 上下文增强特征 [batch_size, feature_dim]
        """
        batch_size, feature_dim = features.size()
        device = features.device
        
        # 处理说话人信息
        if speaker_ids is not None:
            speaker_emb = self.speaker_embedding(speaker_ids)  # [batch_size, speaker_dim]
            speaker_input = torch.cat([features, speaker_emb], dim=1)  # [batch_size, feature_dim + speaker_dim]
        else:
            # 使用零嵌入
            zero_speaker_emb = torch.zeros(batch_size, self.speaker_dim, device=device)
            speaker_input = torch.cat([features, zero_speaker_emb], dim=1)
        
        # 说话人状态建模
        speaker_input = speaker_input.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # 初始化或调整说话人隐藏状态
        if self.speaker_hidden is None or self.speaker_hidden.size(1) != batch_size:
            self.speaker_hidden = torch.zeros(1, batch_size, self.context_dim, device=device)
        
        speaker_output, self.speaker_hidden = self.speaker_gru(speaker_input, self.speaker_hidden.detach())
        speaker_context = speaker_output.squeeze(1)  # [batch_size, context_dim]
        
        # 全局上下文建模
        speaker_context_input = speaker_context.unsqueeze(1)  # [batch_size, 1, context_dim]
        
        # 初始化或调整全局隐藏状态
        if self.global_hidden is None or self.global_hidden.size(1) != batch_size:
            self.global_hidden = torch.zeros(1, batch_size, self.context_dim, device=device)
        
        global_output, self.global_hidden = self.global_gru(speaker_context_input, self.global_hidden.detach())
        global_context = global_output.squeeze(1)  # [batch_size, context_dim]
        
        # 简化的上下文融合
        context_input = torch.cat([features, global_context], dim=1)  # [batch_size, feature_dim + context_dim]
        context_features = self.context_fusion(context_input)  # [batch_size, feature_dim]
        
        return context_features


class SpeakerAwareDialogueModel(nn.Module):
    """简化的说话人感知对话模型"""
    
    def __init__(self, input_dim, output_dim, num_speakers=10, dropout=0.1):
        super(SpeakerAwareDialogueModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_speakers = num_speakers
        
        # 简化的对话上下文模块
        self.dialogue_context = DialogueContextModule(
            feature_dim=input_dim,
            speaker_dim=32,  # 减少维度
            context_dim=64,  # 减少维度
            dropout=dropout
        )
        
        # 简化的说话人特定变换
        self.speaker_transform = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 移除复杂的情绪转移建模，使用简单的残差连接
        self.emotion_transition = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        # 简化状态管理
        self.prev_emotion_state = None
        
    def reset_dialogue(self):
        """重置对话状态"""
        self.dialogue_context.reset_states()
        self.prev_emotion_state = None
        
    def forward(self, features, speaker_ids=None):
        """
        前向传播
        
        Args:
            features: 输入特征 [batch_size, input_dim]
            speaker_ids: 说话人ID [batch_size]
            
        Returns:
            context_features: 对话上下文特征 [batch_size, output_dim]
        """
        batch_size = features.size(0)
        
        # 对话上下文建模
        context_features = self.dialogue_context(features, speaker_ids)
        
        # 说话人特定变换
        speaker_features = self.speaker_transform(context_features)
        
        # 简化的情绪转移 - 只在维度匹配时使用
        if self.prev_emotion_state is not None and self.prev_emotion_state.size(0) == batch_size:
            if self.prev_emotion_state.device == speaker_features.device:
                # 使用残差连接而不是复杂的转移网络
                transition_features = self.emotion_transition(self.prev_emotion_state)
                speaker_features = speaker_features + 0.1 * transition_features  # 小权重的历史信息
        
        # 更新前一状态
        self.prev_emotion_state = speaker_features.detach()
        
        return speaker_features


class ModalitySpecificEncoder(nn.Module):
    """改进的模态特定编码器"""
    
    def __init__(self, input_dim, hidden_dim, dropout=0.1, use_deep_encoder=True):
        super(ModalitySpecificEncoder, self).__init__()
        
        if use_deep_encoder:
            self.encoder = DeepModalityEncoder(
                input_dim=input_dim, 
                hidden_dim=hidden_dim, 
                num_layers=3, 
                dropout=dropout,
                use_attention=True
            )
        else:
            # 原始的简单编码器
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
        
        # 获取启用的模态
        self.enabled_modalities = config['data'].get('enabled_modalities', ['text', 'audio', 'video'])
        self.num_modalities = len(self.enabled_modalities)
        
        # 获取模型配置
        model_config = config.get('model_architecture', {})
        self.use_deep_encoder = model_config.get('use_deep_encoder', True)
        self.use_contrastive_loss = model_config.get('use_contrastive_loss', True)
        self.contrastive_weight = model_config.get('contrastive_weight', 0.1)
        
        # DialogueRNN相关配置
        self.use_dialogue_context = model_config.get('use_dialogue_context', True)
        self.use_speaker_info = model_config.get('use_speaker_info', True)
        
        print(f"模型启用的模态: {self.enabled_modalities}")
        print(f"深度编码器: {self.use_deep_encoder}, 对比损失: {self.use_contrastive_loss}")
        print(f"对话上下文建模: {self.use_dialogue_context}, 说话人信息: {self.use_speaker_info}")
        
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
        
        # 模态特定编码器（深度版本）
        hidden_dim = config['model']['fusion']['hidden_dim']
        dropout = config['model']['fusion']['dropout']
        
        self.modality_encoders = nn.ModuleDict({
            'text': ModalitySpecificEncoder(
                config['model']['text']['feature_dim'], 
                hidden_dim, 
                dropout, 
                use_deep_encoder=self.use_deep_encoder
            ),
            'audio': ModalitySpecificEncoder(
                config['model']['audio']['feature_dim'], 
                hidden_dim, 
                dropout, 
                use_deep_encoder=self.use_deep_encoder
            ),
            'video': ModalitySpecificEncoder(
                config['model']['video']['feature_dim'], 
                hidden_dim, 
                dropout, 
                use_deep_encoder=self.use_deep_encoder
            )
        })
        
        # DialogueRNN对话上下文建模
        if self.use_dialogue_context:
            dialogue_input_dim = hidden_dim * self.num_modalities
            dialogue_output_dim = hidden_dim
            
            self.dialogue_model = SpeakerAwareDialogueModel(
                input_dim=dialogue_input_dim,
                output_dim=dialogue_output_dim,
                num_speakers=config.get('data', {}).get('num_speakers', 10),
                dropout=dropout
            )
            
            # 对话感知的融合分类器
            self.dialogue_fusion_classifier = nn.Sequential(
                nn.Linear(dialogue_output_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, config['model']['output']['num_classes'])
            )
        
        # 单模态分类器
        self.single_modal_classifiers = nn.ModuleDict({
            'text': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, config['model']['output']['num_classes'])
            ),
            'audio': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, config['model']['output']['num_classes'])
            ),
            'video': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, config['model']['output']['num_classes'])
            )
        })
        
        # 改进的融合分类器
        fusion_input_dim = hidden_dim * self.num_modalities
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, config['model']['output']['num_classes'])
        )
        
        # 动态权重学习
        if self.use_dialogue_context:
            # 对话模式：需要考虑额外的dialogue_features
            weight_input_dim = fusion_input_dim + hidden_dim  # fused_features + dialogue_features
            weight_output_dim = self.num_modalities + 2  # 单模态 + 传统融合 + 对话感知
        else:
            # 传统模式
            weight_input_dim = fusion_input_dim
            weight_output_dim = self.num_modalities + 1  # 单模态 + 传统融合
            
        self.fusion_weights = nn.Sequential(
            nn.Linear(weight_input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, weight_output_dim),
            nn.Softmax(dim=1)
        )
        
        # 损失函数
        focal_gamma = config.get('training', {}).get('focal_loss_gamma', 2.0)
        rare_boost = config.get('training', {}).get('rare_class_boost', True)
        label_smoothing = config.get('training', {}).get('label_smoothing', 0.1)
        
        self.focal_loss = ImprovedFocalLoss(
            gamma=focal_gamma, 
            rare_class_boost=rare_boost,
            label_smoothing=label_smoothing
        )
        
        # 对比损失
        if self.use_contrastive_loss:
            contrastive_temp = config.get('training', {}).get('contrastive_temperature', 0.1)
            self.contrastive_loss = ContrastiveLoss(temperature=contrastive_temp)
        
    def forward(self, text_features, audio_features, video_features, speaker_ids=None, reset_dialogue=False):
        """前向传播"""
        
        # 重置对话状态（新对话开始时）
        if reset_dialogue and self.use_dialogue_context:
            self.dialogue_model.reset_dialogue()
        
        # 基础特征编码
        text_encoded = self.text_encoder(text_features)
        audio_encoded = self.audio_encoder(audio_features)
        video_encoded = self.video_encoder(video_features)
        
        # 模态特定编码
        text_modal = self.modality_encoders['text'](text_encoded)
        audio_modal = self.modality_encoders['audio'](audio_encoded)
        video_modal = self.modality_encoders['video'](video_encoded)
        
        # 根据启用的模态收集特征和预测
        modal_features = []
        modal_logits = []
        logits_dict = {}
        
        if 'text' in self.enabled_modalities:
            modal_features.append(text_modal)
            text_logits = self.single_modal_classifiers['text'](text_modal)
            modal_logits.append(text_logits)
            logits_dict['text_logits'] = text_logits
        else:
            logits_dict['text_logits'] = torch.zeros(text_features.size(0), 7).to(text_features.device)
            
        if 'audio' in self.enabled_modalities:
            modal_features.append(audio_modal)
            audio_logits = self.single_modal_classifiers['audio'](audio_modal)
            modal_logits.append(audio_logits)
            logits_dict['audio_logits'] = audio_logits
        else:
            logits_dict['audio_logits'] = torch.zeros(text_features.size(0), 7).to(text_features.device)
            
        if 'video' in self.enabled_modalities:
            modal_features.append(video_modal)
            video_logits = self.single_modal_classifiers['video'](video_modal)
            modal_logits.append(video_logits)
            logits_dict['video_logits'] = video_logits
        else:
            logits_dict['video_logits'] = torch.zeros(text_features.size(0), 7).to(text_features.device)
        
        # 融合多模态特征
        if len(modal_features) > 1:
            fused_features = torch.cat(modal_features, dim=1)
        else:
            fused_features = modal_features[0]
        
        # DialogueRNN对话上下文建模
        if self.use_dialogue_context:
            # 使用对话模型获取上下文感知的特征
            dialogue_features = self.dialogue_model(fused_features, speaker_ids)
            
            # 基于对话上下文的最终预测
            dialogue_logits = self.dialogue_fusion_classifier(dialogue_features)
            logits_dict['dialogue_logits'] = dialogue_logits
            
            # 传统融合预测（作为对比）
            fusion_logits = self.fusion_classifier(fused_features)
            logits_dict['fusion_logits'] = fusion_logits
            
            # 动态权重学习（包含对话上下文）
            extended_features = torch.cat([fused_features, dialogue_features], dim=1)
            weights = self.fusion_weights(extended_features)
            
            # 权重融合：单模态 + 传统融合 + 对话感知
            weighted_logits = torch.zeros_like(dialogue_logits)
            for i, modal_logit in enumerate(modal_logits):
                weighted_logits += weights[:, i:i+1] * modal_logit
            
            # 传统融合权重 - 增强权重
            fusion_weight = weights[:, -2:-1] * 1.5  # 增强传统融合
            weighted_logits += fusion_weight * fusion_logits
            
            # 对话感知权重 - 减少权重
            dialogue_weight = weights[:, -1:] * 0.5  # 减少DialogueRNN影响
            weighted_logits += dialogue_weight * dialogue_logits
            
            logits_dict['weighted_logits'] = weighted_logits
            
            # 返回权重融合的结果而不是纯对话感知结果
            return weighted_logits, logits_dict
            
        else:
            # 原始的非对话版本
            fusion_logits = self.fusion_classifier(fused_features)
            logits_dict['fusion_logits'] = fusion_logits
            
            # 动态权重学习
            weights = self.fusion_weights(fused_features)
            
            # 权重融合
            weighted_logits = torch.zeros_like(fusion_logits)
            for i, modal_logit in enumerate(modal_logits):
                weighted_logits += weights[:, i:i+1] * modal_logit
            weighted_logits += weights[:, -1:] * fusion_logits
            
            logits_dict['weighted_logits'] = weighted_logits
            
            return weighted_logits, logits_dict
    
    def calculate_losses(self, main_logits, logits_dict, labels):
        """计算损失"""
        
        # 主要损失：对话感知的最终预测（或传统融合预测）
        main_loss = self.focal_loss(main_logits, labels)
        
        # 辅助损失：根据启用的模态计算
        aux_losses = []
        aux_loss_dict = {}
        
        if 'text' in self.enabled_modalities and 'text_logits' in logits_dict:
            text_loss = F.cross_entropy(logits_dict['text_logits'], labels)
            aux_losses.append(text_loss)
            aux_loss_dict['text_loss'] = text_loss
        else:
            aux_loss_dict['text_loss'] = torch.tensor(0.0).to(labels.device)
            
        if 'audio' in self.enabled_modalities and 'audio_logits' in logits_dict:
            audio_loss = F.cross_entropy(logits_dict['audio_logits'], labels)
            aux_losses.append(audio_loss)
            aux_loss_dict['audio_loss'] = audio_loss
        else:
            aux_loss_dict['audio_loss'] = torch.tensor(0.0).to(labels.device)
            
        if 'video' in self.enabled_modalities and 'video_logits' in logits_dict:
            video_loss = F.cross_entropy(logits_dict['video_logits'], labels)
            aux_losses.append(video_loss)
            aux_loss_dict['video_loss'] = video_loss
        else:
            aux_loss_dict['video_loss'] = torch.tensor(0.0).to(labels.device)
        
        # 融合损失（传统融合）
        if 'fusion_logits' in logits_dict:
            fusion_loss = F.cross_entropy(logits_dict['fusion_logits'], labels)
            aux_losses.append(fusion_loss)
            aux_loss_dict['fusion_loss'] = fusion_loss
        else:
            aux_loss_dict['fusion_loss'] = torch.tensor(0.0).to(labels.device)
        
        # 对话上下文损失（如果使用DialogueRNN）
        if self.use_dialogue_context and 'dialogue_logits' in logits_dict:
            dialogue_loss = F.cross_entropy(logits_dict['dialogue_logits'], labels)
            aux_losses.append(dialogue_loss)
            aux_loss_dict['dialogue_loss'] = dialogue_loss
        else:
            aux_loss_dict['dialogue_loss'] = torch.tensor(0.0).to(labels.device)
        
        # 权重融合损失
        if 'weighted_logits' in logits_dict:
            weighted_loss = F.cross_entropy(logits_dict['weighted_logits'], labels)
            aux_losses.append(weighted_loss)
            aux_loss_dict['weighted_loss'] = weighted_loss
        else:
            aux_loss_dict['weighted_loss'] = torch.tensor(0.0).to(labels.device)
        
        # 对比损失（如果启用）
        contrastive_loss = torch.tensor(0.0).to(labels.device)
        if self.use_contrastive_loss and self.num_modalities >= 2:
            # 简化的对比损失计算（基于logits）
            contrastive_losses = []
            modal_logits = []
            
            if 'text_logits' in logits_dict:
                modal_logits.append(logits_dict['text_logits'])
            if 'audio_logits' in logits_dict:
                modal_logits.append(logits_dict['audio_logits'])
            if 'video_logits' in logits_dict:
                modal_logits.append(logits_dict['video_logits'])
            
            # 计算模态间的一致性损失
            for i in range(len(modal_logits)):
                for j in range(i + 1, len(modal_logits)):
                    # 使用KL散度作为一致性损失
                    prob_i = F.softmax(modal_logits[i], dim=1)
                    prob_j = F.softmax(modal_logits[j], dim=1)
                    kl_loss = F.kl_div(prob_i.log(), prob_j, reduction='batchmean')
                    contrastive_losses.append(kl_loss)
            
            if contrastive_losses:
                contrastive_loss = torch.stack(contrastive_losses).mean()
        
        aux_loss_dict['contrastive_loss'] = contrastive_loss
        
        # 总损失：主损失 + 辅助损失 + 对比损失
        aux_loss_weight = 0.2 / len(aux_losses) if aux_losses else 0.0
        total_loss = (main_loss + 
                     aux_loss_weight * sum(aux_losses) + 
                     self.contrastive_weight * contrastive_loss)
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            **aux_loss_dict
        } 