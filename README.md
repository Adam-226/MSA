# MELD数据集多模态情感分析

本项目使用**MELD** (Multimodal EmotionLines Dataset) 数据集进行多模态情感分析，实现7类情绪识别（joy, anger, sadness, surprise, fear, disgust, neutral）。

基于高效的多模态融合技术，在测试集上达到**59.46%的准确率**和**56.18%的F1加权分数**。

## 项目结构

```
├── README.md               # 项目说明文档
├── requirements.txt        # 项目依赖
├── main.py                 # 主程序入口
├── configs/                # 配置文件
│   └── config.json         # 模型配置文件
├── scripts/                # 辅助脚本
│   └── check_models.py     # 预训练模型检查脚本
├── logs/                   # 日志文件目录
├── data/                   # 数据存放目录
├── models/                 # 预训练模型目录
├── checkpoints/            # 模型检查点保存目录
└── src/                    # 源代码
    ├── preprocess/         # 数据预处理
    │   ├── text_processor.py      # 文本特征提取
    │   ├── audio_processor.py     # 音频特征提取
    │   ├── video_processor.py     # 视频特征提取
    │   ├── preprocess_all.py      # 统一预处理脚本
    │   └── dataset.py             # 数据集加载类
    ├── models/             # 模型定义
    │   ├── text_model.py          # 文本编码器
    │   ├── audio_model.py         # 音频编码器
    │   ├── video_model.py         # 视频编码器
    │   └── model.py               # 多模态融合模型
    ├── utils/              # 工具函数
    │   ├── evaluator.py           # 完整的评估和可视化工具
    │   └── evaluate_model.py      # 独立评估脚本
    └── training/           # 训练相关
        ├── trainer.py             # 训练脚本
        └── config.py              # 配置管理
```

## 模型特点

- **高效融合架构**: 采用动态权重学习的多模态融合策略
- **多级预测**: 结合单模态预测和融合预测，提升模型鲁棒性
- **Focal Loss**: 解决类别不平衡问题，提升少数类别识别效果
- **早停机制**: 防止过拟合，自动保存最佳模型
- **预训练模型**: 使用RoBERTa (文本)、Wav2Vec2 (音频)、DINO-ViT (视频)
- **自动化流程**: 完整的数据预处理、训练、评估、可视化流程

## 数据集获取

MELD数据集可以通过以下方式获取：

1. 下载链接：[MELD.Raw.tar.gz](http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz)
2. 下载命令：
```bash
wget http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz -P data/
```
3. 解压数据集：
```bash
cd data
tar -xzvf MELD.Raw.tar.gz
cd MELD.Raw
tar -xzvf train.tar.gz
tar -xzvf dev.tar.gz
tar -xzvf test.tar.gz
```

解压后的MELD.Raw目录结构如下：
```
MELD.Raw/
├── README.txt
├── dev_sent_emo.csv           # 验证集情感标签
├── dev_splits_complete        # 验证集数据
├── dev.tar.gz                 # 验证集压缩包
├── output_repeated_splits_test # 测试集重复分割数据
├── test_sent_emo.csv          # 测试集情感标签
├── test.tar.gz                # 测试集压缩包
├── train_sent_emo.csv         # 训练集情感标签
├── train_splits               # 训练集数据
└── train.tar.gz               # 训练集压缩包
```

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 预训练模型准备

本项目使用的预训练模型如下：

1. **文本模型**: RoBERTa-base
2. **音频模型**: Wav2Vec2-base-960h  
3. **视频模型**: DINO-ViT-B/16

手动下载预训练模型：

1. 下载预训练模型到/models目录：

```bash
# Make sure git-lfs is installed (https://git-lfs.com)
git lfs install

# 文本模型 (RoBERTa-base)
git clone https://huggingface.co/roberta-base models/roberta-base

# 音频模型 (Wav2Vec2)
git clone https://huggingface.co/facebook/wav2vec2-base-960h models/wav2vec2-base-960h

# 视频模型 (DINO-ViT)
git clone https://huggingface.co/facebook/dino-vitb16 models/dino-vitb16
```

2. 检查模型是否下载完整：
```bash
python scripts/check_models.py
```

该脚本会检查所有必要的模型文件是否存在，如果有缺失会提示下载方法。

## 多模态情感分析

### 1. 数据预处理

#### 预处理所有模态

首先需要提取多模态特征：

```bash
python src/preprocess/preprocess_all.py --use_local --local_model_dir models
```

#### 预处理单一模态

如果只需要预处理某一模态，可以使用 `--modality` 参数：

**只预处理文本模态：**
```bash
python src/preprocess/preprocess_all.py --use_local --local_model_dir models --modality text
```

**只预处理音频模态：**
```bash
python src/preprocess/preprocess_all.py --use_local --local_model_dir models --modality audio
```

**只预处理视频模态：**
```bash
python src/preprocess/preprocess_all.py --use_local --local_model_dir models --modality video
```

预处理完成后，特征文件将保存在 `data/processed/` 目录下：
```
data/processed/
├── text_features/
├── audio_features/
└── video_features/
```

### 2. 训练模型

```bash
python main.py train --config configs/config.json
```

训练过程会自动：
- 使用早停机制防止过拟合
- 保存最佳模型到 `checkpoints/FusionModel_42/best_model.pth`
- 记录训练历史到 `checkpoints/FusionModel_42/history.json`
- 在测试集上评估并保存结果到 `checkpoints/FusionModel_42/test_results.json`

### 3. 评估模型

#### 方法一：使用主程序测试

```bash
python main.py test --checkpoint checkpoints/FusionModel_42/best_model.pth
```

#### 方法二：使用独立评估脚本

```bash
# 评估测试集
python src/utils/evaluate_model.py --checkpoint checkpoints/FusionModel_42/best_model.pth

# 评估验证集
python src/utils/evaluate_model.py --checkpoint checkpoints/FusionModel_42/best_model.pth --split dev

# 指定输出目录
python src/utils/evaluate_model.py --checkpoint checkpoints/FusionModel_42/best_model.pth --output_dir results/
```

#### 评估输出内容

评估完成后会生成以下文件：

```
evaluation_results/
├── test_evaluation_results.json    # 详细的JSON评估结果
├── test_evaluation_summary.txt     # 文本格式的评估摘要
└── plots/                          # 可视化图表
    ├── confusion_matrix_raw.png       # 原始混淆矩阵
    ├── confusion_matrix_normalized.png # 归一化混淆矩阵
    ├── metrics_per_class.png          # 各类别指标对比
    ├── label_distribution.png         # 标签分布对比
    ├── detailed_performance_heatmap.png # 性能热图
    └── training_history.png           # 训练历史曲线
```

## 性能表现

在MELD测试集上的性能：

| 指标 | 分数 |
|------|------|
| 准确率 (Accuracy) | 59.46% |
| F1-Macro | 32.94% |
| F1-Weighted | 56.18% |

### 7类情感分布：
- **neutral**: 中性
- **joy**: 快乐
- **sadness**: 悲伤  
- **anger**: 愤怒
- **surprise**: 惊讶
- **fear**: 恐惧
- **disgust**: 厌恶
