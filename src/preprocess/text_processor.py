"""
文本特征提取模块
用于从MELD数据集中提取文本特征
"""
import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

class TextProcessor:
    """文本处理类，用于从文本中提取特征"""
    
    def __init__(self, model_name="roberta-base", max_length=128, device='cuda' if torch.cuda.is_available() else 'cpu', use_local=True, local_model_dir="models"):
        """
        初始化TextProcessor
        
        Args:
            model_name: 预训练模型名称，默认为roberta-base
            max_length: 最大序列长度，默认为128
            device: 运行设备，默认为GPU（如果可用）
            use_local: 是否使用本地预训练模型，默认为True
            local_model_dir: 本地预训练模型目录，默认为"models"
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        
        # 构建本地模型路径
        if use_local:
            local_path = os.path.join(local_model_dir, os.path.basename(model_name))
            if os.path.exists(local_path):
                model_name = local_path
                print(f"使用本地预训练模型: {local_path}")
            else:
                print(f"警告：本地模型路径 {local_path} 不存在，尝试在线下载。")
                print(f"可以从 https://huggingface.co/{model_name} 下载模型文件并放置在 {local_path} 目录中。")
        
        print(f"加载文本预训练模型: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            print("使用随机初始化的模型代替。这只是临时的解决方案，最终结果可能不准确。")
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name, local_files_only=use_local)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=use_local)
            self.model = AutoModel(config).to(device)
        
    def extract_features(self, text):
        """
        从文本中提取特征
        
        Args:
            text: 输入文本
            
        Returns:
            features: 文本特征向量
        """
        # 对文本进行编码
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 使用[CLS]标记的表示作为句子表示
        features = outputs.last_hidden_state[:, 0, :]
        
        return features.cpu().numpy()
    
    def process_csv(self, csv_path, output_dir):
        """
        处理CSV文件中的文本并保存特征
        
        Args:
            csv_path: CSV文件路径
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 提取文本特征
        features = []
        texts = []
        
        for text in tqdm(df['Utterance'].tolist(), desc="提取文本特征"):
            texts.append(text)
            feature = self.extract_features(text)
            features.append(feature)
            
        # 保存特征
        output_path = os.path.join(output_dir, f"{os.path.basename(csv_path).split('.')[0]}_text_features.pt")
        torch.save({
            'features': features,
            'texts': texts
        }, output_path)
        
        print(f"文本特征已保存至: {output_path}")
        
        return output_path
        
    def process_dataset(self, dataset_path, output_dir):
        """
        处理MELD数据集
        
        Args:
            dataset_path: 数据集路径
            output_dir: 输出目录
            
        Returns:
            outputs: 包含训练集、验证集和测试集特征路径的字典
        """
        outputs = {}
        
        # 处理训练集
        train_csv = os.path.join(dataset_path, "train_sent_emo.csv")
        if os.path.exists(train_csv):
            outputs['train'] = self.process_csv(train_csv, output_dir)
            
        # 处理验证集
        dev_csv = os.path.join(dataset_path, "dev_sent_emo.csv")
        if os.path.exists(dev_csv):
            outputs['dev'] = self.process_csv(dev_csv, output_dir)
            
        # 处理测试集
        test_csv = os.path.join(dataset_path, "test_sent_emo.csv")
        if os.path.exists(test_csv):
            outputs['test'] = self.process_csv(test_csv, output_dir)
            
        return outputs


if __name__ == "__main__":
    # 示例用法
    processor = TextProcessor(model_name="roberta-base")
    
    # 测试单个文本
    text = "I'm feeling really happy today!"
    features = processor.extract_features(text)
    print(f"文本: {text}")
    print(f"特征形状: {features.shape}")
    
    # 处理整个数据集
    dataset_path = "data/MELD.Raw"
    output_dir = "data/processed/text_features"
    
    if os.path.exists(dataset_path):
        outputs = processor.process_dataset(dataset_path, output_dir)
        print(f"处理完成，输出: {outputs}")
    else:
        print(f"数据集路径不存在: {dataset_path}") 