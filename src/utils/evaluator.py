"""
模型评估器
提供完整的模型评估和可视化功能
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# 设置matplotlib显示配置
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


# ============================= 评估指标计算函数 =============================

def compute_metrics(
    preds: Union[np.ndarray, List[int], torch.Tensor],
    labels: Union[np.ndarray, List[int], torch.Tensor]
) -> Dict[str, float]:
    """
    计算各种评估指标
    
    Args:
        preds: 预测标签
        labels: 真实标签
        
    Returns:
        metrics: 评估指标字典
    """
    # 转换为numpy数组
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # 计算指标
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    precision_macro = precision_score(labels, preds, average='macro')
    precision_weighted = precision_score(labels, preds, average='weighted')
    recall_macro = recall_score(labels, preds, average='macro')
    recall_weighted = recall_score(labels, preds, average='weighted')
    
    # 返回指标字典
    metrics = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted
    }
    
    return metrics


def compute_metrics_per_class(
    preds: Union[np.ndarray, List[int], torch.Tensor],
    labels: Union[np.ndarray, List[int], torch.Tensor],
    label_names: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    计算每个类别的评估指标
    
    Args:
        preds: 预测标签
        labels: 真实标签
        label_names: 标签名称列表
        
    Returns:
        metrics: 每个类别的评估指标字典
    """
    # 转换为numpy数组
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # 计算每个类别的指标
    f1 = f1_score(labels, preds, average=None)
    precision = precision_score(labels, preds, average=None)
    recall = recall_score(labels, preds, average=None)
    
    # 获取类别列表
    classes = np.unique(np.concatenate([labels, preds]))
    
    # 如果没有提供标签名称，使用类别索引作为标签名称
    if label_names is None:
        label_names = [f"class_{c}" for c in classes]
    
    # 构建每个类别的指标字典
    metrics_per_class = {}
    for i, c in enumerate(classes):
        class_name = label_names[c] if c < len(label_names) else f"class_{c}"
        metrics_per_class[class_name] = {
            'f1': f1[i],
            'precision': precision[i],
            'recall': recall[i]
        }
    
    return metrics_per_class


def compute_confusion_matrix_func(
    preds: Union[np.ndarray, List[int], torch.Tensor],
    labels: Union[np.ndarray, List[int], torch.Tensor],
    normalize: bool = False
) -> np.ndarray:
    """
    计算混淆矩阵
    
    Args:
        preds: 预测标签
        labels: 真实标签
        normalize: 是否归一化
        
    Returns:
        cm: 混淆矩阵
    """
    # 转换为numpy数组
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds)
    
    # 归一化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return cm


# ============================= 可视化函数 =============================

def plot_training_history(
    history: Dict[str, List[float]], 
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史字典
        save_path: 保存路径，默认为None（不保存）
        
    Returns:
        fig: matplotlib图形对象
        axes: matplotlib坐标轴对象元组
    """
    # 创建图形和坐标轴
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 绘制损失曲线
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制F1分数曲线
    ax2.plot(history['train_f1_weighted'], label='Train F1 Weighted')
    ax2.plot(history['val_f1_weighted'], label='Val F1 Weighted')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Weighted Score')
    ax2.set_title('Training and Validation F1 Weighted Score')
    ax2.legend()
    ax2.grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2)


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    cmap: str = 'Blues',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        classes: 类别名称列表
        normalize: 是否归一化
        title: 图表标题
        cmap: 颜色映射
        save_path: 保存路径，默认为None（不保存）
        
    Returns:
        fig: matplotlib图形对象
    """
    # 如果需要归一化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制热图
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=classes, yticklabels=classes)
    
    # 设置标题和标签
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_metrics_per_class(
    metrics_per_class: Dict[str, Dict[str, float]],
    title: str = 'Metrics Per Class',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制每个类别的指标
    
    Args:
        metrics_per_class: 每个类别的指标字典
        title: 图表标题
        save_path: 保存路径，默认为None（不保存）
        
    Returns:
        fig: matplotlib图形对象
    """
    # 提取类别和指标
    classes = list(metrics_per_class.keys())
    f1_scores = [metrics_per_class[c]['f1'] for c in classes]
    precision_scores = [metrics_per_class[c]['precision'] for c in classes]
    recall_scores = [metrics_per_class[c]['recall'] for c in classes]
    
    # 创建数据框
    df = pd.DataFrame({
        'Class': classes,
        'F1 Score': f1_scores,
        'Precision': precision_scores,
        'Recall': recall_scores
    })
    
    # 转换为长格式
    df_long = pd.melt(df, id_vars=['Class'], var_name='Metric', value_name='Score')
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 绘制条形图
    sns.barplot(x='Class', y='Score', hue='Metric', data=df_long)
    
    # 设置标题和标签
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Score')
    
    # 调整布局
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 保存图形
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


# ============================= 模型评估器 =============================

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model: nn.Module, device: torch.device, emotion_labels: List[str] = None):
        """
        初始化评估器
        
        Args:
            model: 要评估的模型
            device: 设备
            emotion_labels: 情感标签列表
        """
        self.model = model
        self.device = device
        self.emotion_labels = emotion_labels or ['neutral', 'joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust']
        
    def evaluate_on_loader(self, data_loader: DataLoader) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        在数据加载器上评估模型
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            metrics: 评估指标
            predictions: 预测结果
            true_labels: 真实标签
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_logits = []
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # 获取数据
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device) 
                video = batch['video'].to(self.device)
                labels = batch['emotion'].to(self.device)
                
                batch_size = text.size(0)
                
                # 前向传播
                main_logits, logits_dict = self.model(text, audio, video)
                
                # 计算损失
                losses = self.model.calculate_losses(main_logits, logits_dict, labels)
                loss = losses['total_loss']
                
                # 累积损失
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # 获取预测结果
                logits = main_logits.cpu().numpy()
                preds = np.argmax(logits, axis=1)
                labels_np = labels.cpu().numpy()
                
                all_logits.extend(logits)
                all_preds.extend(preds)
                all_labels.extend(labels_np)
        
        # 转换为numpy数组
        predictions = np.array(all_preds)
        true_labels = np.array(all_labels)
        logits = np.array(all_logits)
        
        # 计算指标
        metrics = {
            'loss': total_loss / total_samples,
            'predictions': predictions.tolist(),
            'true_labels': true_labels.tolist(),
            'logits': logits.tolist()
        }
        
        # 添加详细指标
        detailed_metrics = compute_metrics(predictions, true_labels)
        metrics.update(detailed_metrics)
        
        # 混淆矩阵
        cm = compute_confusion_matrix_func(predictions, true_labels)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics, predictions, true_labels
    
    def generate_detailed_report(self, predictions: np.ndarray, true_labels: np.ndarray) -> Dict:
        """
        生成详细的评估报告
        
        Args:
            predictions: 预测结果
            true_labels: 真实标签
            
        Returns:
            detailed_report: 详细报告
        """
        # 基础指标
        basic_metrics = compute_metrics(predictions, true_labels)
        
        # 每类别指标
        per_class_metrics = compute_metrics_per_class(predictions, true_labels, self.emotion_labels)
        
        # 混淆矩阵
        cm = compute_confusion_matrix_func(predictions, true_labels)
        cm_normalized = compute_confusion_matrix_func(predictions, true_labels, normalize=True)
        
        # sklearn classification report
        sklearn_report = classification_report(
            true_labels, predictions, 
            target_names=self.emotion_labels,
            output_dict=True
        )
        
        # 类别分布统计
        unique_labels, label_counts = np.unique(true_labels, return_counts=True)
        label_distribution = {
            self.emotion_labels[label]: int(count) 
            for label, count in zip(unique_labels, label_counts)
        }
        
        # 预测分布统计
        unique_preds, pred_counts = np.unique(predictions, return_counts=True)
        pred_distribution = {
            self.emotion_labels[pred]: int(count) 
            for pred, count in zip(unique_preds, pred_counts)
        }
        
        detailed_report = {
            'basic_metrics': basic_metrics,
            'per_class_metrics': per_class_metrics,
            'sklearn_report': sklearn_report,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_normalized': cm_normalized.tolist(),
            'label_distribution': label_distribution,
            'pred_distribution': pred_distribution,
            'total_samples': len(true_labels)
        }
        
        return detailed_report
    
    def create_visualizations(self, predictions: np.ndarray, true_labels: np.ndarray, 
                            save_dir: str, history: Dict = None) -> Dict[str, str]:
        """
        创建可视化图表
        
        Args:
            predictions: 预测结果
            true_labels: 真实标签
            save_dir: 保存目录
            history: 训练历史（可选）
            
        Returns:
            saved_plots: 保存的图表路径
        """
        os.makedirs(save_dir, exist_ok=True)
        saved_plots = {}
        
        # 1. 混淆矩阵（原始）
        cm = compute_confusion_matrix_func(predictions, true_labels)
        cm_raw_path = os.path.join(save_dir, 'confusion_matrix_raw.png')
        plot_confusion_matrix(
            cm=cm,
            classes=self.emotion_labels,
            normalize=False,
            title='Confusion Matrix (Raw Counts)',
            save_path=cm_raw_path
        )
        saved_plots['confusion_matrix_raw'] = cm_raw_path
        
        # 2. 混淆矩阵（归一化）
        cm_norm_path = os.path.join(save_dir, 'confusion_matrix_normalized.png')
        plot_confusion_matrix(
            cm=cm,
            classes=self.emotion_labels,
            normalize=True,
            title='Confusion Matrix (Normalized)',
            save_path=cm_norm_path
        )
        saved_plots['confusion_matrix_normalized'] = cm_norm_path
        
        # 3. 每类别指标
        per_class_metrics = compute_metrics_per_class(predictions, true_labels, self.emotion_labels)
        metrics_path = os.path.join(save_dir, 'metrics_per_class.png')
        plot_metrics_per_class(
            metrics_per_class=per_class_metrics,
            title='Performance Metrics by Emotion Class',
            save_path=metrics_path
        )
        saved_plots['metrics_per_class'] = metrics_path
        
        # 4. 类别分布对比
        self._plot_label_distribution(predictions, true_labels, save_dir)
        saved_plots['label_distribution'] = os.path.join(save_dir, 'label_distribution.png')
        
        # 5. 训练历史（如果提供）
        if history is not None:
            history_path = os.path.join(save_dir, 'training_history.png')
            plot_training_history(history=history, save_path=history_path)
            saved_plots['training_history'] = history_path
        
        # 6. 详细性能热图
        self._plot_detailed_heatmap(predictions, true_labels, save_dir)
        saved_plots['detailed_heatmap'] = os.path.join(save_dir, 'detailed_performance_heatmap.png')
        
        return saved_plots
    
    def _plot_label_distribution(self, predictions: np.ndarray, true_labels: np.ndarray, save_dir: str):
        """绘制标签分布对比图"""
        
        # 统计真实标签和预测标签分布
        true_counts = np.bincount(true_labels, minlength=len(self.emotion_labels))
        pred_counts = np.bincount(predictions, minlength=len(self.emotion_labels))
        
        # 创建DataFrame
        df = pd.DataFrame({
            'Emotion': self.emotion_labels,
            'True Labels': true_counts,
            'Predictions': pred_counts
        })
        
        # 转换为长格式
        df_long = pd.melt(df, id_vars=['Emotion'], var_name='Type', value_name='Count')
        
        # 绘图
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_long, x='Emotion', y='Count', hue='Type')
        plt.title('True Labels vs Predictions Distribution')
        plt.xlabel('Emotion Category')
        plt.ylabel('Sample Count')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'label_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_detailed_heatmap(self, predictions: np.ndarray, true_labels: np.ndarray, save_dir: str):
        """绘制详细性能热图"""
        
        # 计算每个类别的详细指标
        per_class_metrics = compute_metrics_per_class(predictions, true_labels, self.emotion_labels)
        
        # 准备数据
        metrics_data = []
        for emotion in self.emotion_labels:
            if emotion in per_class_metrics:
                metrics_data.append([
                    per_class_metrics[emotion]['precision'],
                    per_class_metrics[emotion]['recall'],
                    per_class_metrics[emotion]['f1']
                ])
            else:
                metrics_data.append([0.0, 0.0, 0.0])
        
        metrics_array = np.array(metrics_data)
        
        # 绘制热图
        plt.figure(figsize=(8, 10))
        sns.heatmap(
            metrics_array,
            xticklabels=['Precision', 'Recall', 'F1-Score'],
            yticklabels=self.emotion_labels,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            cbar_kws={'label': 'Score'}
        )
        plt.title('Detailed Performance Heatmap by Emotion')
        plt.xlabel('Metrics')
        plt.ylabel('Emotion Category')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'detailed_performance_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results: Dict, save_path: str):
        """
        保存评估结果到JSON文件
        
        Args:
            results: 评估结果
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
    
    def print_summary(self, results: Dict):
        """
        打印评估结果摘要
        
        Args:
            results: 评估结果
        """
        basic_metrics = results['basic_metrics']
        per_class = results['per_class_metrics']
        
        print("=" * 70)
        print("模型评估结果摘要")
        print("=" * 70)
        print(f"总样本数: {results['total_samples']}")
        print(f"总体准确率: {basic_metrics['accuracy']:.4f}")
        print(f"F1-Macro: {basic_metrics['f1_macro']:.4f}")
        print(f"F1-Weighted: {basic_metrics['f1_weighted']:.4f}")
        print(f"Precision-Macro: {basic_metrics['precision_macro']:.4f}")
        print(f"Recall-Macro: {basic_metrics['recall_macro']:.4f}")
        
        print("\n各类别详细指标:")
        print("-" * 70)
        print(f"{'类别':<12} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Support':<8}")
        print("-" * 70)
        
        for emotion in self.emotion_labels:
            if emotion in per_class:
                metrics = per_class[emotion]
                support = results['label_distribution'].get(emotion, 0)
                print(f"{emotion:<12} {metrics['f1']:<8.4f} {metrics['precision']:<10.4f} "
                      f"{metrics['recall']:<8.4f} {support:<8}")
        
        print("-" * 70) 