"""
音频特征提取模块
用于从MELD数据集中提取音频特征
"""
import os
import glob
import torch
import librosa
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# 抑制重复的警告信息
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", message="librosa.core.audio.__audioread_load")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

class AudioProcessor:
    """音频处理类，用于从音频中提取特征"""
    
    def __init__(
        self, 
        model_name="facebook/wav2vec2-base-960h",
        target_sample_rate=16000,
        max_length=16000*10,  # 10秒
        batch_size=8,
        device='cpu',
        use_local=True,
        local_model_dir="models"
    ):
        """
        初始化AudioProcessor
        
        Args:
            model_name: 预训练模型名称
            target_sample_rate: 目标采样率
            max_length: 最大音频长度（样本数）
            batch_size: 批次大小
            device: 运行设备
            use_local: 是否使用本地预训练模型，默认为True
            local_model_dir: 本地预训练模型目录，默认为"models"
        """
        self.model_name = model_name
        self.target_sample_rate = target_sample_rate
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        
        # 构建本地模型路径
        if use_local:
            model_folder = os.path.basename(model_name.rstrip("/"))
            local_path = os.path.join(local_model_dir, model_folder)
            if os.path.exists(local_path):
                model_name = local_path
                print(f"使用本地预训练模型: {local_path}")
            else:
                print(f"警告：本地模型路径 {local_path} 不存在，尝试在线下载。")
                print(f"可以从 https://huggingface.co/{model_name} 下载模型文件并放置在 {local_path} 目录中。")
        
        print(f"加载音频预训练模型: {model_name}")
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            print("使用随机初始化的模型代替。这只是临时的解决方案，最终结果可能不准确。")
            from transformers import Wav2Vec2Config
            config = Wav2Vec2Config.from_pretrained(model_name, local_files_only=use_local)
            self.processor = Wav2Vec2Processor.from_pretrained(model_name, local_files_only=use_local)
            self.model = Wav2Vec2Model(config).to(device)
        
    def load_audio(self, audio_path, max_duration=10):
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            max_duration: 最大音频时长（秒），默认为10秒
            
        Returns:
            waveform: 音频波形
        """
        try:
            # 加载音频
            waveform, sr = librosa.load(audio_path, sr=self.target_sample_rate, mono=True)
            
            # 如果音频太长，截断
            max_samples = int(max_duration * self.target_sample_rate)
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]
                
            # 如果音频太短，填充
            if len(waveform) < self.target_sample_rate:
                waveform = np.pad(waveform, (0, self.target_sample_rate - len(waveform)), 'constant')
                
            return waveform
        except Exception as e:
            print(f"加载音频文件失败: {audio_path}, 错误: {e}")
            # 返回空音频
            return np.zeros(self.target_sample_rate)
        
    def load_audio_from_video(self, video_path, max_duration=10):
        """
        从视频文件中加载音频
        
        Args:
            video_path: 视频文件路径
            max_duration: 最大音频时长（秒），默认为10秒
            
        Returns:
            waveform: 音频波形
        """
        try:
            # 使用librosa从视频中提取音频
            waveform, sr = librosa.load(video_path, sr=self.target_sample_rate, mono=True)
            
            # 如果音频太长，截断
            max_samples = int(max_duration * self.target_sample_rate)
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]
                
            # 如果音频太短，填充
            if len(waveform) < self.target_sample_rate:
                waveform = np.pad(waveform, (0, self.target_sample_rate - len(waveform)), 'constant')
                
            return waveform
        except Exception as e:
            # 返回空音频，但不打印错误（只在debug模式下打印）
            return np.zeros(self.target_sample_rate)

    def extract_features(self, audio_path):
        """
        从音频中提取特征
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            features: 音频特征向量
        """
        # 加载音频
        waveform = self.load_audio(audio_path)
        
        # 对音频进行编码
        inputs = self.processor(waveform, sampling_rate=self.target_sample_rate, return_tensors="pt").to(self.device)
        
        # 提取特征
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 获取最后一层隐藏状态
        features = outputs.last_hidden_state
        
        # 对时间维度进行平均池化，得到单一特征向量
        features = torch.mean(features, dim=1)
        
        return features.cpu().numpy()
    
    def extract_features_from_video(self, video_path):
        """
        从视频文件中提取音频特征
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            features: 音频特征向量
        """
        try:
            # 从视频加载音频
            waveform = self.load_audio_from_video(video_path)
            
            # 对音频进行编码
            inputs = self.processor(waveform, sampling_rate=self.target_sample_rate, return_tensors="pt").to(self.device)
            
            # 提取特征
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # 获取最后一层隐藏状态
            features = outputs.last_hidden_state
            
            # 对时间维度进行平均池化，得到单一特征向量
            features = torch.mean(features, dim=1)
            
            return features.cpu().numpy()
        except Exception as e:
            # 返回零向量
            return np.zeros((1, 768))
    
    def find_media_file(self, split_dir, dia_id, utt_id, extension):
        """
        智能查找媒体文件，支持多种可能的文件组织结构
        
        Args:
            split_dir: 数据集分割目录
            dia_id: 对话ID
            utt_id: 话语ID
            extension: 文件扩展名（如.mp4, .wav）
            
        Returns:
            文件路径，如果找不到返回None
        """
        filename = f"dia{dia_id}_utt{utt_id}{extension}"
        
        # 可能的路径列表
        possible_paths = [
            # 直接在split_dir中
            os.path.join(split_dir, filename),
            # 在dia{dia_id}子目录中
            os.path.join(split_dir, f"dia{dia_id}", filename),
            # 尝试不同的子目录结构
            os.path.join(split_dir, f"{dia_id}", filename),
        ]
        
        # 检查每个可能的路径
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        return None
    
    def process_meld_split(self, split_dir, csv_path, output_dir):
        """
        处理MELD数据集的一个划分（训练集/验证集/测试集）
        
        Args:
            split_dir: 数据集划分目录
            csv_path: 标签CSV文件路径
            output_dir: 输出目录
            
        Returns:
            output_path: 输出特征文件路径
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 提取音频特征
        features = []
        audio_paths = []
        
        # 按CSV文件的每一行处理
        processed_count = 0
        error_count = 0
        
        with tqdm(total=len(df), desc="提取音频特征", leave=True, ncols=100) as pbar:
            for idx, row in df.iterrows():
                dia_id = row['Dialogue_ID']
                utt_id = row['Utterance_ID']
                
                try:
                    # 首先尝试查找独立的音频文件
                    audio_file_path = self.find_media_file(split_dir, dia_id, utt_id, '.wav')
                    
                    # 如果没有找到音频文件，尝试从视频文件提取音频
                    if not audio_file_path:
                        video_file_path = self.find_media_file(split_dir, dia_id, utt_id, '.mp4')
                        if video_file_path:
                            audio_file_path = video_file_path
                    
                    # 提取特征
                    if audio_file_path:
                        # 根据文件类型选择处理方法
                        if audio_file_path.endswith('.wav'):
                            feature = self.extract_features(audio_file_path)
                        else:  # 视频文件
                            feature = self.extract_features_from_video(audio_file_path)
                        features.append(feature)
                        audio_paths.append(audio_file_path)
                        processed_count += 1
                    else:
                        # 如果音频/视频文件不存在，使用零向量
                        features.append(np.zeros((1, 768)))
                        audio_paths.append("")
                        error_count += 1
                        
                except Exception as e:
                    # 使用零向量作为特征
                    features.append(np.zeros((1, 768)))
                    audio_paths.append("")
                    error_count += 1
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    '处理': processed_count, 
                    '错误': error_count
                })
        
        # 保存特征
        output_path = os.path.join(output_dir, f"{os.path.basename(csv_path).split('.')[0]}_audio_features.pt")
        torch.save({
            'features': features,
            'audio_paths': audio_paths
        }, output_path)
        
        print(f"音频特征已保存至: {output_path}")
        
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
        train_dir = os.path.join(dataset_path, "train_splits")
        train_csv = os.path.join(dataset_path, "train_sent_emo.csv")
        if os.path.exists(train_dir) and os.path.exists(train_csv):
            outputs['train'] = self.process_meld_split(train_dir, train_csv, output_dir)
            
        # 处理验证集
        dev_dir = os.path.join(dataset_path, "dev_splits_complete")
        dev_csv = os.path.join(dataset_path, "dev_sent_emo.csv")
        if os.path.exists(dev_dir) and os.path.exists(dev_csv):
            outputs['dev'] = self.process_meld_split(dev_dir, dev_csv, output_dir)
            
        # 处理测试集
        test_dir = os.path.join(dataset_path, "output_repeated_splits_test")
        test_csv = os.path.join(dataset_path, "test_sent_emo.csv")
        if os.path.exists(test_dir) and os.path.exists(test_csv):
            outputs['test'] = self.process_meld_split(test_dir, test_csv, output_dir)
            
        return outputs


if __name__ == "__main__":
    # 示例用法
    processor = AudioProcessor()
    
    # 处理整个数据集
    dataset_path = "data/MELD.Raw"
    output_dir = "data/processed/audio_features"
    
    if os.path.exists(dataset_path):
        outputs = processor.process_dataset(dataset_path, output_dir)
        print(f"处理完成，输出: {outputs}")
    else:
        print(f"数据集路径不存在: {dataset_path}") 