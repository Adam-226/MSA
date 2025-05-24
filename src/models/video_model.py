"""
视频模型模块
使用预训练模型处理视频数据
"""
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class VideoEncoder(nn.Module):
    """视频编码器类"""
    
    def __init__(self, pretrained_model="facebook/dino-vitb16", feature_dim=768, dropout=0.1, freeze_encoder=True, use_local=True, local_model_dir="models"):
        """
        初始化VideoEncoder
        
        Args:
            pretrained_model: 预训练模型名称，默认为"facebook/dino-vitb16"
            feature_dim: 输出特征维度，默认为768
            dropout: dropout概率，默认为0.1
            freeze_encoder: 是否冻结编码器参数，默认为True
            use_local: 是否使用本地预训练模型，默认为True
            local_model_dir: 本地预训练模型目录，默认为"models"
        """
        super(VideoEncoder, self).__init__()
        
        # 构建本地模型路径
        if use_local:
            import os
            model_folder = os.path.basename(pretrained_model.rstrip("/"))
            local_path = os.path.join(local_model_dir, model_folder)
            if os.path.exists(local_path):
                pretrained_model = local_path
                print(f"使用本地视频预训练模型: {local_path}")
            else:
                print(f"警告：本地视频模型路径 {local_path} 不存在，尝试在线下载。")
        
        # 加载预训练模型配置
        self.config = ViTConfig.from_pretrained(pretrained_model, local_files_only=use_local)
        
        # 加载预训练模型
        try:
            self.vit = ViTModel.from_pretrained(pretrained_model, local_files_only=use_local)
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            print("使用随机初始化的模型")
            self.vit = ViTModel(self.config)
        
        # 冻结预训练模型参数
        if freeze_encoder:
            for param in self.vit.parameters():
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
            features: 视频特征，形状为[batch_size, feature_dim]
            
        Returns:
            output: 编码后的视频特征，形状为[batch_size, feature_dim]
        """
        # 如果输入是原始特征（不是通过vit提取的），直接使用特征转换层
        output = self.fc(features)
        
        return output
        
    def extract_features(self, pixel_values, attention_mask=None):
        """
        从原始视频帧中提取特征
        
        Args:
            pixel_values: 视频帧像素值，形状为[batch_size, num_frames, 3, height, width]
            attention_mask: 注意力掩码，形状为[batch_size, num_frames]
            
        Returns:
            features: 提取的视频特征，形状为[batch_size, feature_dim]
        """
        batch_size, num_frames = pixel_values.shape[0], pixel_values.shape[1]
        
        # 重塑为[batch_size * num_frames, 3, height, width]
        pixel_values = pixel_values.view(-1, pixel_values.shape[2], pixel_values.shape[3], pixel_values.shape[4])
        
        # 提取ViT特征
        outputs = self.vit(pixel_values=pixel_values)
        
        # 获取[CLS]标记的表示
        cls_features = outputs.last_hidden_state[:, 0, :]
        
        # 重塑为[batch_size, num_frames, hidden_size]
        cls_features = cls_features.view(batch_size, num_frames, -1)
        
        # 对帧维度进行平均池化，得到视频特征
        features = torch.mean(cls_features, dim=1)
        
        # 特征转换
        features = self.fc(features)
        
        return features


class VideoClassifier(nn.Module):
    """视频分类器类"""
    
    def __init__(self, encoder_model="facebook/dino-vitb16", feature_dim=768, num_classes=7, dropout=0.1, freeze_encoder=True, use_local=True, local_model_dir="models"):
        """
        初始化VideoClassifier
        
        Args:
            encoder_model: 编码器模型名称，默认为dino-vitb16
            feature_dim: 特征维度，默认为768
            num_classes: 类别数量，默认为7（MELD数据集的情感类别数）
            dropout: Dropout比率，默认为0.1
            freeze_encoder: 是否冻结预训练模型参数，默认为True
            use_local: 是否使用本地预训练模型，默认为True
            local_model_dir: 本地预训练模型目录，默认为"models"
        """
        super(VideoClassifier, self).__init__()
        
        # 视频编码器
        self.encoder = VideoEncoder(
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
            features: 视频特征，形状为[batch_size, feature_dim]
            
        Returns:
            logits: 类别logits，形状为[batch_size, num_classes]
        """
        # 编码视频特征
        encoded = self.encoder(features)
        
        # 分类
        logits = self.classifier(encoded)
        
        return logits
        
    def extract_and_classify(self, pixel_values, attention_mask=None):
        """
        从原始视频帧中提取特征并分类
        
        Args:
            pixel_values: 视频帧像素值，形状为[batch_size, num_frames, 3, height, width]
            attention_mask: 注意力掩码，形状为[batch_size, num_frames]
            
        Returns:
            logits: 类别logits，形状为[batch_size, num_classes]
        """
        # 提取特征
        features = self.encoder.extract_features(pixel_values, attention_mask)
        
        # 分类
        logits = self.classifier(features)
        
        return logits


if __name__ == "__main__":
    # 示例用法
    
    # 创建编码器
    encoder = VideoEncoder(pretrained_model="facebook/dino-vitb16")
    
    # 随机生成特征
    batch_size = 16
    feature_dim = 768
    features = torch.randn(batch_size, feature_dim)
    
    # 编码特征
    encoded = encoder(features)
    print(f"编码后的特征形状: {encoded.shape}")
    
    # 创建分类器
    classifier = VideoClassifier(encoder_model="facebook/dino-vitb16", num_classes=7)
    
    # 分类
    logits = classifier(features)
    print(f"分类logits形状: {logits.shape}") 