#!/usr/bin/env python3
"""
集成评估脚本 - 加载多个模型进行投票集成
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# 添加项目根目录到系统路径
sys.path.append(str(Path(__file__).parent.parent))

from src.models.model import FusionModel
from src.preprocess.dataset import get_meld_dataloaders
from src.training.config import get_device

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 情感标签映射
EMOTION_LABELS = ['neutral', 'joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust']

def load_ensemble_models(ensemble_info_path: str, device: torch.device) -> List[Dict[str, Any]]:
    """
    加载集成模型
    
    Args:
        ensemble_info_path: 集成信息文件路径
        device: 计算设备
        
    Returns:
        models: 加载的模型列表
    """
    
    # 加载集成信息
    with open(ensemble_info_path, 'r', encoding='utf-8') as f:
        ensemble_info = json.load(f)
    
    models = []
    
    for model_info in ensemble_info['models']:
        model_dir = model_info['model_dir']
        best_model_path = os.path.join(model_dir, 'best_model.pth')
        
        if not os.path.exists(best_model_path):
            logger.warning(f"模型文件不存在: {best_model_path}")
            continue
        
        try:
            # 加载模型检查点
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            config = checkpoint['config']
            
            # 创建模型
            model = FusionModel(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            models.append({
                'model': model,
                'seed': model_info['seed'],
                'config': config,
                'val_metrics': checkpoint.get('val_metrics', {}),
                'best_val_f1': model_info.get('best_val_f1_weighted', 0.0)
            })
            
            logger.info(f"成功加载模型: 种子{model_info['seed']}, 验证F1={model_info.get('best_val_f1_weighted', 0.0):.4f}")
            
        except Exception as e:
            logger.error(f"加载模型失败 (种子{model_info['seed']}): {str(e)}")
            continue
    
    logger.info(f"成功加载 {len(models)} 个模型用于集成")
    return models

def ensemble_predict(models: List[Dict[str, Any]], data_loader, device: torch.device, 
                    method: str = 'voting') -> tuple:
    """
    集成预测
    
    Args:
        models: 模型列表
        data_loader: 数据加载器
        device: 计算设备
        method: 集成方法 ('voting', 'soft_voting', 'weighted_voting')
        
    Returns:
        all_labels: 真实标签
        ensemble_preds: 集成预测结果
        individual_preds: 各个模型的预测结果
    """
    
    all_labels = []
    all_individual_preds = []  # [num_models, num_samples]
    all_individual_probs = []  # [num_models, num_samples, num_classes]
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="集成预测"):
            # 获取数据
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            labels = batch['emotion'].to(device)
            
            batch_labels = labels.cpu().numpy()
            all_labels.extend(batch_labels)
            
            batch_preds = []
            batch_probs = []
            
            # 每个模型进行预测
            for model_info in models:
                model = model_info['model']
                output = model(text, audio, video)
                
                # 获取预测结果
                logits = output['logits']
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                batch_preds.append(preds.cpu().numpy())
                batch_probs.append(probs.cpu().numpy())
            
            all_individual_preds.append(np.array(batch_preds))  # [num_models, batch_size]
            all_individual_probs.append(np.array(batch_probs))  # [num_models, batch_size, num_classes]
    
    # 合并所有批次
    all_labels = np.array(all_labels)
    all_individual_preds = np.concatenate(all_individual_preds, axis=1)  # [num_models, total_samples]
    all_individual_probs = np.concatenate(all_individual_probs, axis=1)  # [num_models, total_samples, num_classes]
    
    # 根据集成方法生成最终预测
    if method == 'voting':
        # 硬投票
        ensemble_preds = []
        for i in range(all_individual_preds.shape[1]):
            votes = all_individual_preds[:, i]
            ensemble_pred = Counter(votes).most_common(1)[0][0]
            ensemble_preds.append(ensemble_pred)
        ensemble_preds = np.array(ensemble_preds)
        
    elif method == 'soft_voting':
        # 软投票 - 平均概率
        avg_probs = np.mean(all_individual_probs, axis=0)  # [total_samples, num_classes]
        ensemble_preds = np.argmax(avg_probs, axis=1)
        
    elif method == 'weighted_voting':
        # 加权投票 - 根据验证性能加权
        weights = np.array([model_info['best_val_f1'] for model_info in models])
        weights = weights / np.sum(weights)  # 归一化权重
        
        weighted_probs = np.zeros_like(all_individual_probs[0])  # [total_samples, num_classes]
        for i, weight in enumerate(weights):
            weighted_probs += weight * all_individual_probs[i]
        
        ensemble_preds = np.argmax(weighted_probs, axis=1)
    
    else:
        raise ValueError(f"不支持的集成方法: {method}")
    
    return all_labels, ensemble_preds, all_individual_preds.T  # 转置以便后续分析

def evaluate_ensemble(ensemble_info_path: str, split: str = 'test', method: str = 'voting',
                     output_dir: str = None):
    """
    评估集成模型
    
    Args:
        ensemble_info_path: 集成信息文件路径
        split: 评估数据集 ('train', 'dev', 'test')
        method: 集成方法
        output_dir: 输出目录
    """
    
    # 获取设备
    device = get_device()
    logger.info(f"使用设备: {device}")
    
    # 加载集成模型
    models = load_ensemble_models(ensemble_info_path, device)
    if not models:
        logger.error("没有成功加载任何模型!")
        return
    
    # 加载数据
    config = models[0]['config']  # 使用第一个模型的配置
    dataloaders = get_meld_dataloaders(config)
    data_loader = dataloaders[split]
    
    logger.info(f"开始评估集成模型 (数据集: {split}, 方法: {method})")
    
    # 集成预测
    all_labels, ensemble_preds, individual_preds = ensemble_predict(
        models, data_loader, device, method
    )
    
    # 计算集成性能
    ensemble_acc = accuracy_score(all_labels, ensemble_preds)
    ensemble_f1_macro = f1_score(all_labels, ensemble_preds, average='macro')
    ensemble_f1_weighted = f1_score(all_labels, ensemble_preds, average='weighted')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"集成模型性能 ({method}):")
    logger.info(f"  准确率: {ensemble_acc:.4f}")
    logger.info(f"  F1-Macro: {ensemble_f1_macro:.4f}")
    logger.info(f"  F1-Weighted: {ensemble_f1_weighted:.4f}")
    
    # 计算各个模型的性能
    logger.info(f"\n单个模型性能:")
    individual_results = []
    for i, model_info in enumerate(models):
        preds = individual_preds[:, i]
        acc = accuracy_score(all_labels, preds)
        f1_macro = f1_score(all_labels, preds, average='macro')
        f1_weighted = f1_score(all_labels, preds, average='weighted')
        
        individual_results.append({
            'seed': model_info['seed'],
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        })
        
        logger.info(f"  模型{i+1} (种子{model_info['seed']}): "
                   f"Acc={acc:.4f}, F1-Macro={f1_macro:.4f}, F1-Weighted={f1_weighted:.4f}")
    
    # 计算平均性能和提升
    avg_acc = np.mean([r['accuracy'] for r in individual_results])
    avg_f1_macro = np.mean([r['f1_macro'] for r in individual_results])
    avg_f1_weighted = np.mean([r['f1_weighted'] for r in individual_results])
    
    acc_improvement = ensemble_acc - avg_acc
    f1_macro_improvement = ensemble_f1_macro - avg_f1_macro
    f1_weighted_improvement = ensemble_f1_weighted - avg_f1_weighted
    
    logger.info(f"\n性能对比:")
    logger.info(f"  单模型平均: Acc={avg_acc:.4f}, F1-Macro={avg_f1_macro:.4f}, F1-Weighted={avg_f1_weighted:.4f}")
    logger.info(f"  集成模型:   Acc={ensemble_acc:.4f}, F1-Macro={ensemble_f1_macro:.4f}, F1-Weighted={ensemble_f1_weighted:.4f}")
    logger.info(f"  提升幅度:   Acc={acc_improvement:+.4f}, F1-Macro={f1_macro_improvement:+.4f}, F1-Weighted={f1_weighted_improvement:+.4f}")
    
    # 详细分类报告
    logger.info(f"\n分类报告:")
    report = classification_report(all_labels, ensemble_preds, 
                                 target_names=EMOTION_LABELS, 
                                 digits=4)
    logger.info(f"\n{report}")
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'ensemble_method': method,
            'split': split,
            'ensemble_metrics': {
                'accuracy': float(ensemble_acc),
                'f1_macro': float(ensemble_f1_macro),
                'f1_weighted': float(ensemble_f1_weighted)
            },
            'individual_results': individual_results,
            'average_metrics': {
                'accuracy': float(avg_acc),
                'f1_macro': float(avg_f1_macro),
                'f1_weighted': float(avg_f1_weighted)
            },
            'improvements': {
                'accuracy': float(acc_improvement),
                'f1_macro': float(f1_macro_improvement),
                'f1_weighted': float(f1_weighted_improvement)
            },
            'confusion_matrix': confusion_matrix(all_labels, ensemble_preds).tolist()
        }
        
        results_path = os.path.join(output_dir, f'ensemble_results_{split}_{method}.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        logger.info(f"结果已保存到: {results_path}")

def main():
    parser = argparse.ArgumentParser(description='集成模型评估')
    parser.add_argument('--ensemble_info', type=str, required=True,
                       help='集成信息文件路径')
    parser.add_argument('--split', type=str, default='test', 
                       choices=['train', 'dev', 'test'],
                       help='评估数据集')
    parser.add_argument('--method', type=str, default='voting',
                       choices=['voting', 'soft_voting', 'weighted_voting'],
                       help='集成方法')
    parser.add_argument('--output_dir', type=str, default='evaluation_results/ensemble',
                       help='输出目录')
    
    args = parser.parse_args()
    
    logger.info("开始集成评估...")
    logger.info(f"集成信息文件: {args.ensemble_info}")
    logger.info(f"评估数据集: {args.split}")
    logger.info(f"集成方法: {args.method}")
    
    # 评估集成模型
    evaluate_ensemble(
        ensemble_info_path=args.ensemble_info,
        split=args.split,
        method=args.method,
        output_dir=args.output_dir
    )
    
    logger.info("集成评估完成!")

if __name__ == "__main__":
    main() 