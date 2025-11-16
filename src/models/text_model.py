"""
文本模型模块
使用预训练语言模型处理文本数据
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TextEncoder(nn.Module):
    """文本编码器类"""
    
    def __init__(self, pretrained_model="roberta-base", feature_dim=768, dropout=0.1, freeze_bert=False, use_local=True, local_model_dir="models"):
        """
        初始化TextEncoder
        
        Args:
            pretrained_model: 预训练模型名称，默认为roberta-base
            feature_dim: 特征维度，默认为768
            dropout: Dropout比率，默认为0.1
            freeze_bert: 是否冻结BERT参数，默认为False
            use_local: 是否使用本地预训练模型，默认为True
            local_model_dir: 本地预训练模型目录，默认为"models"
        """
        super(TextEncoder, self).__init__()
        
        # 构建本地模型路径
        if use_local:
            import os
            model_folder = os.path.basename(pretrained_model.rstrip("/"))
            local_path = os.path.join(local_model_dir, model_folder)
            if os.path.exists(local_path):
                pretrained_model = local_path
                print(f"使用本地预训练模型: {local_path}")
            else:
                print(f"警告：本地模型路径 {local_path} 不存在，尝试在线下载。")
        
        # 加载预训练模型配置
        self.config = AutoConfig.from_pretrained(pretrained_model, local_files_only=use_local)
        
        # 加载预训练模型
        self.bert = AutoModel.from_pretrained(pretrained_model, local_files_only=use_local)
        
        # 冻结BERT参数
        if freeze_bert:
            for param in self.bert.parameters():
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
            features: 文本特征，形状为[batch_size, feature_dim]
            
        Returns:
            output: 编码后的文本特征，形状为[batch_size, feature_dim]
        """
        # 如果输入是原始特征（不是通过BERT提取的），直接使用特征转换层
        output = self.fc(features)
        
        return output
        
    def extract_features(self, input_ids, attention_mask=None):
        """
        从原始文本中提取特征
        
        Args:
            input_ids: 输入token IDs，形状为[batch_size, seq_length]
            attention_mask: 注意力掩码，形状为[batch_size, seq_length]
            
        Returns:
            features: 提取的文本特征，形状为[batch_size, feature_dim]
        """
        # 提取BERT特征
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用[CLS]标记的表示作为句子表示
        features = outputs.last_hidden_state[:, 0, :]
        
        # 特征转换
        features = self.fc(features)
        
        return features


class TextClassifier(nn.Module):
    """文本分类器类"""
    
    def __init__(self, encoder_model="roberta-base", feature_dim=768, num_classes=7, dropout=0.1, freeze_bert=False, use_local=True, local_model_dir="models"):
        """
        初始化TextClassifier
        
        Args:
            encoder_model: 编码器模型名称，默认为roberta-base
            feature_dim: 特征维度，默认为768
            num_classes: 类别数量，默认为7（MELD数据集的情感类别数）
            dropout: Dropout比率，默认为0.1
            freeze_bert: 是否冻结BERT参数，默认为False
            use_local: 是否使用本地预训练模型，默认为True
            local_model_dir: 本地预训练模型目录，默认为"models"
        """
        super(TextClassifier, self).__init__()
        
        # 文本编码器
        self.encoder = TextEncoder(
            pretrained_model=encoder_model,
            feature_dim=feature_dim,
            dropout=dropout,
            freeze_bert=freeze_bert,
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
            features: 文本特征，形状为[batch_size, feature_dim]
            
        Returns:
            logits: 类别logits，形状为[batch_size, num_classes]
        """
        # 编码文本特征
        encoded = self.encoder(features)
        
        # 分类
        logits = self.classifier(encoded)
        
        return logits
        
    def extract_and_classify(self, input_ids, attention_mask=None):
        """
        从原始文本中提取特征并分类
        
        Args:
            input_ids: 输入token IDs，形状为[batch_size, seq_length]
            attention_mask: 注意力掩码，形状为[batch_size, seq_length]
            
        Returns:
            logits: 类别logits，形状为[batch_size, num_classes]
        """
        # 提取特征
        features = self.encoder.extract_features(input_ids, attention_mask)
        
        # 分类
        logits = self.classifier(features)
        
        return logits


def create_text_encoder(
    pretrained_model: str = "roberta-base",
    feature_dim: int = 768,
    dropout: float = 0.2,
    freeze_bert: bool = False, 
    use_local: bool = True, 
    local_model_dir: str = "models"
):
    """
    创建文本编码器
    
    Args:
        pretrained_model: 预训练模型名称，默认为"roberta-base"
        feature_dim: 特征维度，默认为768
        dropout: dropout概率，默认为0.2
        freeze_bert: 是否冻结BERT层，默认为False
        use_local: 是否使用本地模型，默认为True
        local_model_dir: 本地预训练模型目录，默认为"models"
        
    Returns:
        TextEncoder: 文本编码器实例
    """


if __name__ == "__main__":
    # 示例用法
    
    # 创建编码器
    encoder = TextEncoder(pretrained_model="roberta-base")
    
    # 随机生成特征
    batch_size = 16
    feature_dim = 768
    features = torch.randn(batch_size, feature_dim)
    
    # 编码特征
    encoded = encoder(features)
    print(f"编码后的特征形状: {encoded.shape}")
    
    # 创建分类器
    classifier = TextClassifier(encoder_model="roberta-base", num_classes=7)
    
    # 分类
    logits = classifier(features)
    print(f"分类logits形状: {logits.shape}") 