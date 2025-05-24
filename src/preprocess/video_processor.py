"""
视频特征提取模块
用于从MELD数据集中提取视频特征
"""
import os
import glob
import torch
import numpy as np
import pandas as pd
import cv2
import warnings
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel

# 抑制重复的警告信息
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message="The class ViTFeatureExtractor is deprecated")

class VideoProcessor:
    """视频处理类，用于从视频中提取特征"""
    
    def __init__(
        self, 
        model_name="facebook/dino-vitb16",
        image_size=224,
        max_frames=3,
        fps=1,
        batch_size=8,
        device='cpu',
        use_local=True,
        local_model_dir="models"
    ):
        """
        初始化VideoProcessor
        
        Args:
            model_name: 预训练模型名称
            image_size: 图像尺寸
            max_frames: 最大帧数
            fps: 每秒帧数
            batch_size: 批次大小
            device: 运行设备
            use_local: 是否使用本地预训练模型，默认为True
            local_model_dir: 本地预训练模型目录，默认为"models"
        """
        self.model_name = model_name
        self.image_size = image_size
        self.max_frames = max_frames
        self.fps = fps
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
        
        print(f"加载视频预训练模型: {model_name}")
        try:
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
            # 尝试先加载safetensors格式，失败则自动使用其他可用格式
            try:
                self.model = ViTModel.from_pretrained(model_name, use_safetensors=True).to(device)
            except:
                # 如果safetensors不可用，则使用默认加载方式（会自动选择可用格式）
                self.model = ViTModel.from_pretrained(model_name).to(device)
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            print("使用随机初始化的模型代替。这只是临时的解决方案，最终结果可能不准确。")
            from transformers import ViTConfig
            config = ViTConfig.from_pretrained(model_name, local_files_only=use_local)
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name, local_files_only=use_local)
            self.model = ViTModel(config).to(device)
        
        # 定义图像变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def extract_frames(self, video_path):
        """
        从视频中提取帧
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            frames: 提取的帧列表
        """
        frames = []
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            
            # 检查视频是否成功打开
            if not cap.isOpened():
                cap.release()
                return []
            
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # 如果视频为空，返回空列表
            if total_frames == 0 or duration == 0:
                cap.release()
                return []
                
            # 计算均匀采样的帧索引
            frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
            
            # 提取帧
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # 转换为RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 转换为PIL图像
                    pil_image = Image.fromarray(frame)
                    # 应用图像变换
                    transformed_frame = self.transform(pil_image)
                    frames.append(transformed_frame)
                    
            cap.release()
            
            # 如果没有提取到任何帧，返回空列表
            if len(frames) == 0:
                return []
                
            # 如果提取的帧数少于要求的帧数，复制最后一帧填充
            while len(frames) < self.max_frames:
                frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))
                
            return frames
        except Exception as e:
            return []
        
    def extract_features(self, video_path):
        """
        从视频中提取特征
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            features: 视频特征向量
        """
        # 提取帧
        frames = self.extract_frames(video_path)
        
        # 如果没有提取到帧，返回零向量
        if not frames:
            return np.zeros((1, 768))
            
        # 将帧转换为numpy数组并批量处理
        frames_np = []
        for frame in frames:
            frame_np = frame.cpu().numpy()
            
            # 如果需要，将CHW转换为HWC
            if frame_np.shape[0] == 3:  # CHW格式
                frame_np = frame_np.transpose(1, 2, 0)
            
            # 确保值在[0,1]范围内
            frame_np = np.clip(frame_np, 0, 1)
            
            # 转换为PIL图像所需的[0,255]范围
            frame_np = (frame_np * 255).astype(np.uint8)
            frames_np.append(frame_np)
        
        # 批量处理所有帧
        with torch.no_grad():
            # 使用feature_extractor批量处理
            inputs = self.feature_extractor(images=frames_np, return_tensors="pt").to(self.device)
            
            # 提取特征
            outputs = self.model(**inputs)
                
            # 使用[CLS]标记的表示作为帧表示
            frame_features = outputs.last_hidden_state[:, 0, :]
            
            # 对所有帧特征进行平均池化，得到视频特征
            video_features = torch.mean(frame_features, dim=0, keepdim=True)
        
        return video_features.cpu().numpy()
    
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
        
        # 提取视频特征
        features = []
        video_paths = []
        
        # 按CSV文件的每一行处理
        processed_count = 0
        error_count = 0
        
        with tqdm(total=len(df), desc="提取视频特征", leave=True, ncols=100) as pbar:
            for idx, row in df.iterrows():
                dia_id = row['Dialogue_ID']
                utt_id = row['Utterance_ID']
                
                try:
                    # 使用智能文件查找
                    video_file_path = self.find_media_file(split_dir, dia_id, utt_id, '.mp4')
                    
                    # 提取特征
                    if video_file_path:
                        feature = self.extract_features(video_file_path)
                        features.append(feature)
                        video_paths.append(video_file_path)
                        processed_count += 1
                    else:
                        # 如果视频文件不存在，使用零向量
                        features.append(np.zeros((1, 768)))
                        video_paths.append("")
                        error_count += 1
                        
                except Exception as e:
                    # 使用零向量作为特征
                    features.append(np.zeros((1, 768)))
                    video_paths.append("")
                    error_count += 1
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    '处理': processed_count, 
                    '错误': error_count
                })
        
        # 保存特征
        output_path = os.path.join(output_dir, f"{os.path.basename(csv_path).split('.')[0]}_video_features.pt")
        torch.save({
            'features': features,
            'video_paths': video_paths
        }, output_path)
        
        print(f"视频特征已保存至: {output_path}")
        
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
    processor = VideoProcessor()
    
    # 处理整个数据集
    dataset_path = "data/MELD.Raw"
    output_dir = "data/processed/video_features"
    
    if os.path.exists(dataset_path):
        outputs = processor.process_dataset(dataset_path, output_dir)
        print(f"处理完成，输出: {outputs}")
    else:
        print(f"数据集路径不存在: {dataset_path}") 