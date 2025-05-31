#!/usr/bin/env python3
"""
专门用于情感模型的预处理脚本
使用CardiffNLP的twitter-roberta-base-emotion模型提取文本特征
"""
import os
import sys
import warnings
import tarfile
from pathlib import Path

# 抑制警告信息
warnings.filterwarnings("ignore")

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from preprocess.text_processor import TextProcessor
from preprocess.audio_processor import AudioProcessor

def check_and_extract_data():
    """检查并解压MELD数据"""
    dataset_path = "data/MELD.Raw"
    tar_path = "data/MELD.Raw.tar.gz"
    
    if os.path.exists(dataset_path):
        print(f"✅ 发现数据目录: {dataset_path}")
        return dataset_path
    
    if os.path.exists(tar_path):
        print(f"📦 发现压缩文件: {tar_path}")
        print("正在解压数据...")
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path="data/")
        
        print("✅ 数据解压完成")
        return dataset_path
    
    print("❌ 未找到MELD数据，请确保 data/MELD.Raw.tar.gz 或 data/MELD.Raw 存在")
    return None

def main():
    print("🚀 开始使用情感专门模型提取特征...")
    
    # 检查和准备数据
    dataset_path = check_and_extract_data()
    if not dataset_path:
        print("❌ 数据准备失败，退出")
        return
    
    # 配置参数
    output_dir = "data/processed_emotion"
    emotion_model = "cardiffnlp/twitter-roberta-base-emotion"
    audio_model = "facebook/wav2vec2-base-960h"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n配置信息:")
    print(f"📁 数据集路径: {dataset_path}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🤖 情感文本模型: {emotion_model}")
    print(f"🤖 音频模型: {audio_model}")
    
    # 检查必要的CSV文件
    required_files = ["train_sent_emo.csv", "dev_sent_emo.csv", "test_sent_emo.csv"]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(dataset_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        print("请确保MELD.Raw目录包含完整的CSV文件")
        return
    
    # 提取文本特征（使用情感专门模型）
    print("\n=== 🔤 提取文本特征（情感专门模型）===")
    try:
        text_processor = TextProcessor(
            model_name=emotion_model,
            use_local=False,  # 直接从HuggingFace下载
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        text_output_dir = os.path.join(output_dir, "text_features")
        text_results = text_processor.process_dataset(dataset_path, text_output_dir)
        print(f"✅ 文本特征提取完成: {text_results}")
        
    except Exception as e:
        print(f"❌ 文本特征提取失败: {str(e)}")
        print("可能需要安装transformers: pip install transformers")
        return
    
    # 提取音频特征（复用原有特征或重新提取）
    print("\n=== 🔊 处理音频特征 ===")
    existing_audio_dir = "data/processed/audio_features"
    if os.path.exists(existing_audio_dir):
        print("发现已有音频特征，复制到新目录...")
        import shutil
        new_audio_dir = os.path.join(output_dir, "audio_features")
        if os.path.exists(new_audio_dir):
            shutil.rmtree(new_audio_dir)
        shutil.copytree(existing_audio_dir, new_audio_dir)
        print(f"✅ 音频特征复制完成: {new_audio_dir}")
    else:
        print("⚠️  未找到已有音频特征")
        print("如需重新提取音频特征，请运行:")
        print("python src/preprocess/preprocess_all.py --modality audio")
        print("现在跳过音频特征...")
        
        # 创建空的音频特征目录
        audio_output_dir = os.path.join(output_dir, "audio_features")
        os.makedirs(audio_output_dir, exist_ok=True)
    
    print("\n🎉 情感模型特征提取完成!")
    print(f"📁 特征保存位置: {output_dir}")
    print(f"📂 文本特征: {os.path.join(output_dir, 'text_features')}")
    print(f"📂 音频特征: {os.path.join(output_dir, 'audio_features')}")
    
    print("\n🚀 下一步: 运行训练命令")
    print(f"python main.py train --config configs/config_bimodal_emotion.json")

if __name__ == "__main__":
    # 导入torch用于检查CUDA
    try:
        import torch
    except ImportError:
        print("请安装PyTorch: pip install torch")
        sys.exit(1)
    
    main() 