#!/usr/bin/env python3
"""
端到端多模态情感分析模型评估脚本
专门用于评估通过train_end_to_end.py训练的模型
"""
import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

# 抑制警告
warnings.filterwarnings("ignore")

# 添加项目根目录到系统路径
sys.path.append(str(Path(__file__).parent.parent))

from src.models.end_to_end_model import EndToEndMultiModalModel
from src.preprocess.end_to_end_dataset import get_end_to_end_dataloaders
from src.training.config import load_config, get_device
from src.utils.evaluator import ModelEvaluator, compute_metrics, compute_metrics_per_class, plot_confusion_matrix

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 情感标签
EMOTION_LABELS = ['neutral', 'joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust']


class EndToEndModelEvaluator:
    """端到端模型评估器，与标准ModelEvaluator功能一致"""
    
    def __init__(self, model: nn.Module, device: torch.device, emotion_labels: List[str] = None):
        """
        初始化端到端模型评估器
        
        Args:
            model: 端到端模型
            device: 设备
            emotion_labels: 情感标签列表
        """
        self.model = model
        self.device = device
        self.emotion_labels = emotion_labels or EMOTION_LABELS
        
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
                # 获取数据并移动到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                audio_features = batch['audio'].to(self.device)
                labels = batch['emotion'].to(self.device)
                
                batch_size = input_ids.size(0)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audio_features=audio_features,
                    return_features=True
                )
                
                # 计算损失
                losses = self.model.calculate_losses(outputs, labels)
                loss = losses['total_loss']
                
                # 累积损失和样本数
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # 获取预测结果
                logits = outputs['logits']
                preds = torch.argmax(logits, dim=1)
                
                all_logits.extend(logits.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
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
        cm = confusion_matrix(true_labels, predictions)
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
        cm = confusion_matrix(predictions, true_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # sklearn classification report
        sklearn_report = classification_report(
            true_labels, predictions, 
            target_names=self.emotion_labels,
            output_dict=True,
            zero_division=0
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
        from src.utils.evaluator import (
            plot_confusion_matrix, plot_metrics_per_class, 
            plot_training_history, compute_confusion_matrix_func
        )
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        
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
        plt.close()
        
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
        plt.close()
        
        # 3. 每类别指标
        per_class_metrics = compute_metrics_per_class(predictions, true_labels, self.emotion_labels)
        metrics_path = os.path.join(save_dir, 'metrics_per_class.png')
        plot_metrics_per_class(
            metrics_per_class=per_class_metrics,
            title='Performance Metrics by Emotion Class',
            save_path=metrics_path
        )
        saved_plots['metrics_per_class'] = metrics_path
        plt.close()
        
        # 4. 类别分布对比
        self._plot_label_distribution(predictions, true_labels, save_dir)
        saved_plots['label_distribution'] = os.path.join(save_dir, 'label_distribution.png')
        
        # 5. 训练历史（如果提供）
        if history is not None:
            history_path = os.path.join(save_dir, 'training_history.png')
            plot_training_history(history=history, save_path=history_path)
            saved_plots['training_history'] = history_path
            plt.close()
        
        # 6. 详细性能热图
        self._plot_detailed_heatmap(predictions, true_labels, save_dir)
        saved_plots['detailed_heatmap'] = os.path.join(save_dir, 'detailed_performance_heatmap.png')
        
        return saved_plots
    
    def _plot_label_distribution(self, predictions: np.ndarray, true_labels: np.ndarray, save_dir: str):
        """绘制标签分布对比图"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        
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
        import matplotlib.pyplot as plt
        import seaborn as sns
        
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
        
        print("=" * 80)
        print("端到端模型评估结果摘要")
        print("=" * 80)
        print(f"📊 总样本数: {results['total_samples']}")
        print(f"📈 准确率 (Accuracy): {basic_metrics['accuracy']:.4f}")
        print(f"📈 F1-Macro: {basic_metrics['f1_macro']:.4f}")
        print(f"📈 F1-Weighted: {basic_metrics['f1_weighted']:.4f}")
        print(f"📈 Precision-Macro: {basic_metrics['precision_macro']:.4f}")
        print(f"📈 Recall-Macro: {basic_metrics['recall_macro']:.4f}")
        
        print(f"\n各类别详细指标:")
        print("-" * 80)
        print(f"{'类别':<12} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Support':<8}")
        print("-" * 80)
        
        for emotion in self.emotion_labels:
            if emotion in per_class:
                metrics = per_class[emotion]
                support = results['label_distribution'].get(emotion, 0)
                print(f"{emotion:<12} {metrics['f1']:<8.4f} {metrics['precision']:<10.4f} "
                      f"{metrics['recall']:<8.4f} {support:<8}")
        
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description='评估端到端多模态情感分析模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'dev'], help='评估数据集分割')
    parser.add_argument('--output_dir', type=str, default='results', help='结果输出目录')
    parser.add_argument('--save_predictions', action='store_true', help='是否保存预测结果')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"检查点文件不存在: {args.checkpoint}")
    
    # 加载配置
    config = load_config(args.config)
    device = get_device()
    
    logger.info(f"🔧 配置文件: {args.config}")
    logger.info(f"💾 检查点: {args.checkpoint}")
    logger.info(f"🖥️  设备: {device}")
    logger.info(f"📊 评估数据集: {args.split}")
    
    # 获取数据加载器
    dataloaders = get_end_to_end_dataloaders(config)
    
    if args.split == 'test':
        data_loader = dataloaders['test']
    else:
        data_loader = dataloaders['dev']
    
    # 创建模型
    logger.info("🔨 创建端到端模型...")
    from src.models.end_to_end_model import create_end_to_end_model
    model = create_end_to_end_model(config)
    model = model.to(device)
    
    # 加载检查点
    logger.info("📂 加载模型检查点...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"✅ 模型加载完成，训练epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # 获取实验目录
    checkpoint_dir = os.path.dirname(args.checkpoint)
    
    # 创建评估器
    evaluator = EndToEndModelEvaluator(model, device, EMOTION_LABELS)
    
    # 评估模型
    logger.info(f"📊 在{args.split}集上评估模型...")
    metrics, predictions, true_labels = evaluator.evaluate_on_loader(data_loader)
    
    # 生成详细报告
    logger.info("📋 生成详细评估报告...")
    detailed_report = evaluator.generate_detailed_report(predictions, true_labels)
    
    # 加载训练历史（如果存在）
    history = None
    history_path = os.path.join(checkpoint_dir, 'history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
    
    # 创建可视化
    logger.info("🎨 生成可视化图表...")
    saved_plots = evaluator.create_visualizations(
        predictions, true_labels,
        save_dir=os.path.join(checkpoint_dir, 'evaluation_plots'),
        history=history
    )
    
    # 保存详细结果
    results_path = os.path.join(checkpoint_dir, 'comprehensive_evaluation.json')
    final_results = {
        'evaluation_metrics': metrics,
        'detailed_report': detailed_report,
        'saved_plots': saved_plots,
        'model_info': {
            'checkpoint_path': args.checkpoint,
            'config_path': args.config,
            'model_config': config
        }
    }
    
    evaluator.save_results(final_results, results_path)
    
    # 打印摘要
    evaluator.print_summary(detailed_report)
    
    logger.info(f"\n📄 详细评估结果已保存至: {results_path}")
    logger.info(f"🎯 可视化图表已保存至: {os.path.join(checkpoint_dir, 'evaluation_plots')}")
    
    # 可选：保存预测结果
    if args.save_predictions:
        predictions_file = os.path.join(args.output_dir, f"EndToEndEmotionModel_{args.split}_predictions.json")
        predictions_data = {
            'predictions': predictions.tolist(),
            'true_labels': true_labels.tolist(),
            'emotion_labels': EMOTION_LABELS
        }
        os.makedirs(args.output_dir, exist_ok=True)
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions_data, f, indent=4)
        logger.info(f"📦 预测结果已保存到: {predictions_file}")
    
    logger.info("🎉 评估完成！")
    
    return final_results


if __name__ == "__main__":
    main() 