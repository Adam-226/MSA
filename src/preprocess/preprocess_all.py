"""
统一的数据预处理脚本
用于提取文本、音频和视频特征
"""
import os
import sys
import argparse
import warnings
from tqdm import tqdm

# 抑制所有警告信息，保持输出清洁
warnings.filterwarnings("ignore")

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocess.text_processor import TextProcessor
from preprocess.audio_processor import AudioProcessor
from preprocess.video_processor import VideoProcessor

def process_text(args, output_dir):
    """
    处理文本模态数据
    
    Args:
        args: 命令行参数
        output_dir: 输出目录
        
    Returns:
        dict: 处理结果
    """
    print("处理文本模态...")
    text_processor = TextProcessor(
        model_name=getattr(args, 'text_model', 'roberta-base'),
        use_local=getattr(args, 'use_local', False), 
        local_model_dir=getattr(args, 'local_model_dir', 'models')
    )
    text_output_dir = os.path.join(output_dir, "text_features")
    result = text_processor.process_dataset(args.dataset_path, text_output_dir)
    print(f"文本特征提取完成: {result}")
    return result

def process_audio(args, output_dir):
    """
    处理音频模态数据
    
    Args:
        args: 命令行参数
        output_dir: 输出目录
        
    Returns:
        dict: 处理结果
    """
    print("处理音频模态...")
    audio_processor = AudioProcessor(
        model_name=getattr(args, 'audio_model', 'facebook/wav2vec2-base-960h'),
        use_local=getattr(args, 'use_local', False), 
        local_model_dir=getattr(args, 'local_model_dir', 'models')
    )
    audio_output_dir = os.path.join(output_dir, "audio_features")
    result = audio_processor.process_dataset(args.dataset_path, audio_output_dir)
    print(f"音频特征提取完成: {result}")
    return result

def process_video(args, output_dir):
    """
    处理视频模态数据
    
    Args:
        args: 命令行参数
        output_dir: 输出目录
        
    Returns:
        dict: 处理结果
    """
    print("处理视频模态...")
    video_processor = VideoProcessor(
        model_name=getattr(args, 'video_model', 'facebook/dino-vitb16'),
        use_local=getattr(args, 'use_local', False), 
        local_model_dir=getattr(args, 'local_model_dir', 'models')
    )
    video_output_dir = os.path.join(output_dir, "video_features")
    result = video_processor.process_dataset(args.dataset_path, video_output_dir)
    print(f"视频特征提取完成: {result}")
    return result

def main():
    parser = argparse.ArgumentParser(description='预处理MELD数据集')
    parser.add_argument('--dataset_path', default='data/MELD.Raw', help='数据集路径')
    parser.add_argument('--output_dir', default='data/processed', help='输出目录')
    parser.add_argument('--use_local', action='store_true', help='使用本地预训练模型')
    parser.add_argument('--local_model_dir', default='models', help='本地预训练模型目录')
    parser.add_argument('--modality', choices=['text', 'audio', 'video', 'all'], default='all', 
                       help='要处理的模态: text, audio, video, 或 all (默认处理所有模态)')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    
    # 根据选择的模态进行处理
    if args.modality in ['text', 'all']:
        print("\n=== 处理文本模态 ===")
        results['text'] = process_text(args, args.output_dir)
    
    if args.modality in ['audio', 'all']:
        print("\n=== 处理音频模态 ===")
        results['audio'] = process_audio(args, args.output_dir)
    
    if args.modality in ['video', 'all']:
        print("\n=== 处理视频模态 ===")
        results['video'] = process_video(args, args.output_dir)
    
    print("\n=== 预处理完成 ===")
    print(f"特征已保存至: {args.output_dir}")
    print(f"处理结果: {results}")

if __name__ == "__main__":
    main() 