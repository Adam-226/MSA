#!/usr/bin/env python3
"""
独立的模型评估脚本
提供完整的模型评估和可视化功能
"""
import os
import sys
import argparse
import logging
import json
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from src.models.model import FusionModel
from src.preprocess.dataset import get_meld_dataloaders
from src.training.config import set_seed, load_config, get_device
from src.utils.evaluator import ModelEvaluator

# 配置日志
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/evaluation.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MELD数据集多模态情感分析模型评估')
    parser.add_argument('--config', type=str, default='configs/config.json',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录，默认为checkpoint同目录下的evaluation_results')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'dev', 'test'],
                       help='评估的数据集分割')
    
    args = parser.parse_args()
    
    logger.info("开始模型评估...")
    logger.info(f"模型检查点: {args.checkpoint}")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"评估数据集: {args.split}")
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置随机种子
    set_seed(config['random_seed'])
    
    # 获取设备
    device = get_device()
    logger.info(f"使用设备: {device}")
    
    # 获取数据加载器
    dataloaders = get_meld_dataloaders(config)
    data_loader = dataloaders[args.split]
    
    # 创建模型
    model = FusionModel(config)
    
    # 加载模型权重
    logger.info(f"加载模型权重...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # 确定输出目录
    if args.output_dir is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        output_dir = os.path.join(checkpoint_dir, 'evaluation_results')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"评估结果将保存至: {output_dir}")
    
    # 创建评估器
    emotion_labels = ['neutral', 'joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust']
    evaluator = ModelEvaluator(model, device, emotion_labels)
    
    # 评估模型
    logger.info(f"在{args.split}集上评估模型...")
    metrics, predictions, true_labels = evaluator.evaluate_on_loader(data_loader)
    
    # 生成详细报告
    logger.info("生成详细评估报告...")
    detailed_report = evaluator.generate_detailed_report(predictions, true_labels)
    
    # 加载训练历史（如果存在）
    history = None
    checkpoint_dir = os.path.dirname(args.checkpoint)
    history_path = os.path.join(checkpoint_dir, 'history.json')
    if os.path.exists(history_path):
        logger.info("加载训练历史...")
        with open(history_path, 'r') as f:
            history = json.load(f)
    
    # 创建可视化
    logger.info("生成可视化图表...")
    plots_dir = os.path.join(output_dir, 'plots')
    saved_plots = evaluator.create_visualizations(
        predictions, true_labels, 
        save_dir=plots_dir,
        history=history
    )
    
    # 保存详细结果
    results_path = os.path.join(output_dir, f'{args.split}_evaluation_results.json')
    final_results = {
        'evaluation_info': {
            'dataset_split': args.split,
            'checkpoint_path': args.checkpoint,
            'config_path': args.config,
            'output_dir': output_dir
        },
        'evaluation_metrics': metrics,
        'detailed_report': detailed_report,
        'saved_plots': saved_plots,
        'model_config': config
    }
    
    evaluator.save_results(final_results, results_path)
    
    # 打印摘要
    evaluator.print_summary(detailed_report)
    
    # 保存简化的文本报告
    text_report_path = os.path.join(output_dir, f'{args.split}_evaluation_summary.txt')
    with open(text_report_path, 'w', encoding='utf-8') as f:
        f.write("MELD数据集多模态情感分析模型评估报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"模型检查点: {args.checkpoint}\n")
        f.write(f"评估数据集: {args.split}\n")
        f.write(f"总样本数: {detailed_report['total_samples']}\n\n")
        
        basic_metrics = detailed_report['basic_metrics']
        f.write("总体性能指标:\n")
        f.write("-" * 30 + "\n")
        f.write(f"准确率 (Accuracy): {basic_metrics['accuracy']:.4f}\n")
        f.write(f"F1-Macro: {basic_metrics['f1_macro']:.4f}\n")
        f.write(f"F1-Weighted: {basic_metrics['f1_weighted']:.4f}\n")
        f.write(f"Precision-Macro: {basic_metrics['precision_macro']:.4f}\n")
        f.write(f"Recall-Macro: {basic_metrics['recall_macro']:.4f}\n\n")
        
        per_class = detailed_report['per_class_metrics']
        f.write("各类别详细指标:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'类别':<12} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Support':<8}\n")
        f.write("-" * 60 + "\n")
        
        for emotion in emotion_labels:
            if emotion in per_class:
                metrics_data = per_class[emotion]
                support = detailed_report['label_distribution'].get(emotion, 0)
                f.write(f"{emotion:<12} {metrics_data['f1']:<8.4f} {metrics_data['precision']:<10.4f} "
                       f"{metrics_data['recall']:<8.4f} {support:<8}\n")
        
        f.write("-" * 60 + "\n")
    
    logger.info(f"\n详细评估结果已保存至: {results_path}")
    logger.info(f"文本报告已保存至: {text_report_path}")
    logger.info(f"可视化图表已保存至: {plots_dir}")
    logger.info("评估完成！")


if __name__ == "__main__":
    main() 