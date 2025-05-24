#!/usr/bin/env python3
"""
检查预训练模型是否已正确下载到本地
"""
import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(str(Path(__file__).parent.parent))

def check_model(model_dir, model_name):
    """检查模型文件是否完整"""
    # 模型基本路径
    base_path = os.path.join(model_dir, model_name)
    
    # 必要的模型文件
    required_files = ['config.json', 'pytorch_model.bin']
    
    # 检查目录是否存在
    if not os.path.isdir(base_path):
        print(f"❌ 模型目录不存在：{base_path}")
        return False
    
    # 检查必要文件
    for file in required_files:
        file_path = os.path.join(base_path, file)
        if not os.path.exists(file_path):
            print(f"❌ 缺少必要文件：{file_path}")
            return False
    
    # 所有检查通过
    print(f"✅ 模型检查通过：{base_path}")
    return True

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='检查预训练模型是否已正确下载')
    parser.add_argument('--model_dir', type=str, default='models', help='预训练模型目录')
    args = parser.parse_args()
    
    # 要检查的模型
    models = {
        'roberta-base': '文本模型 (RoBERTa-base)',
        'wav2vec2-base-960h': '音频模型 (Wav2Vec2)',
        'dino-vitb16': '视频模型 (DINO-ViT)'
    }
    
    print(f"\n===== 检查预训练模型 =====")
    
    # 检查模型目录
    if not os.path.isdir(args.model_dir):
        print(f"❌ 预训练模型目录不存在：{args.model_dir}")
        print(f"请创建目录并下载模型：mkdir -p {args.model_dir}")
        return
    
    # 检查各个模型
    missing_models = []
    for model_name, description in models.items():
        print(f"\n检查{description}...")
        if not check_model(args.model_dir, model_name):
            missing_models.append(model_name)
    
    # 总结
    print("\n===== 检查结果 =====")
    if missing_models:
        print(f"❌ 以下模型需要下载：")
        for model in missing_models:
            if model == 'roberta-base':
                print(f"  1. {models[model]}:")
                print(f"     git clone https://huggingface.co/{model} {args.model_dir}/{model}")
                print(f"     或访问 https://huggingface.co/{model} 手动下载")
            else:
                print(f"  2. {models[model]}:")
                print(f"     git clone https://huggingface.co/facebook/{model} {args.model_dir}/{model}")
                print(f"     或访问 https://huggingface.co/facebook/{model} 手动下载")
    else:
        print("✅ 所有模型检查通过，可以开始数据预处理！")
        print("运行命令：python src/preprocess/preprocess_all.py --use_local")

if __name__ == "__main__":
    main() 