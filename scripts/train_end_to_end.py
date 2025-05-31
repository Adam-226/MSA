#!/usr/bin/env python3
"""
端到端多模态情感分析训练脚本
支持直接处理原始文本，进行完全端到端训练
"""
import os
import sys
import time
import json
import logging
import warnings
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 抑制警告
warnings.filterwarnings("ignore")

# 添加项目根目录到系统路径
sys.path.append(str(Path(__file__).parent.parent))

from src.models.end_to_end_model import EndToEndMultiModalModel
from src.preprocess.end_to_end_dataset import get_end_to_end_dataloaders
from src.training.config import set_seed, load_config, get_device, get_class_weights, create_experiment_dir

# 配置日志
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/end_to_end_training.log')
    ]
)
logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    clip_grad_val: float = 1.0,
    accumulation_steps: int = 1
) -> Dict[str, float]:
    """训练一个epoch"""
    
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()  # 初始化梯度
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        # 获取数据并移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        audio_features = batch['audio'].to(device)
        labels = batch['emotion'].to(device)
        
        batch_size = input_ids.size(0)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            return_features=True  # 用于对比损失
        )
        
        # 计算损失
        losses = model.calculate_losses(outputs, labels)
        loss = losses['total_loss'] / accumulation_steps  # 梯度累积
        
        # 反向传播
        loss.backward()
        
        # 梯度累积
        if (batch_idx + 1) % accumulation_steps == 0:
            # 梯度裁剪
            if clip_grad_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_val)
            
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
            
            # 更新学习率
            if scheduler is not None and isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step()
        
        # 累积损失和样本数
        total_loss += losses['total_loss'].item() * batch_size
        total_samples += batch_size
        
        # 预测结果
        preds = torch.argmax(outputs['logits'], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f"{losses['total_loss'].item():.4f}",
            'Avg_Loss': f"{total_loss/total_samples:.4f}"
        })
    
    # 处理最后的梯度累积
    if (len(train_loader) % accumulation_steps) != 0:
        if clip_grad_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_val)
        optimizer.step()
        optimizer.zero_grad()
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / total_samples
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """评估模型性能"""
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # 获取数据并移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_features = batch['audio'].to(device)
            labels = batch['emotion'].to(device)
            
            batch_size = input_ids.size(0)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_features=audio_features,
                return_features=True
            )
            
            # 计算损失
            losses = model.calculate_losses(outputs, labels)
            loss = losses['total_loss']
            
            # 累积损失和样本数
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 保存logits和标签
            all_logits.append(outputs['logits'].cpu())
            all_labels.extend(labels.cpu().numpy())
    
    # 合并所有logits
    all_logits = torch.cat(all_logits, dim=0)
    all_preds = torch.argmax(all_logits, dim=1).numpy()
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / total_samples
    
    return {
        'loss': avg_loss,
        'acc': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


def create_optimizers(model, config):
    """创建优化器，支持不同模态的不同学习率"""
    
    training_config = config['training']
    
    # 获取不同的学习率
    base_lr = training_config['lr']
    text_lr = training_config.get('text_lr', base_lr * 0.3)
    audio_lr = training_config.get('audio_lr', base_lr * 0.5)
    
    # 分组参数
    text_params = []
    audio_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'text_encoder' in name:
            text_params.append(param)
        elif 'audio_encoder' in name:
            audio_params.append(param)
        else:
            other_params.append(param)
    
    # 创建参数组
    param_groups = [
        {'params': text_params, 'lr': text_lr, 'name': 'text'},
        {'params': audio_params, 'lr': audio_lr, 'name': 'audio'},
        {'params': other_params, 'lr': base_lr, 'name': 'other'}
    ]
    
    # 过滤空组
    param_groups = [group for group in param_groups if len(group['params']) > 0]
    
    logger.info(f"📊 参数组信息:")
    for group in param_groups:
        logger.info(f"  - {group['name']}: {len(group['params'])} 参数, lr={group['lr']:.2e}")
    
    # 创建优化器
    optimizer = optim.AdamW(
        param_groups,
        weight_decay=training_config['weight_decay']
    )
    
    return optimizer


def train_end_to_end_model(config_path: str) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """端到端训练模型"""
    
    # 加载配置
    config = load_config(config_path)
    
    # 设置随机种子
    set_seed(config['random_seed'])
    
    # 获取设备
    device = get_device()
    logger.info(f"🔥 使用设备: {device}")
    
    # 创建实验目录
    exp_dir = create_experiment_dir(config)
    logger.info(f"📁 实验目录: {exp_dir}")
    
    # 获取端到端数据加载器
    try:
        dataloaders = get_end_to_end_dataloaders(config)
        train_loader = dataloaders['train']
        dev_loader = dataloaders['dev']
        test_loader = dataloaders['test']
        logger.info(f"✅ 数据加载器创建成功")
    except Exception as e:
        logger.error(f"❌ 数据加载器创建失败: {e}")
        raise
    
    # 创建端到端模型
    try:
        model = EndToEndMultiModalModel(config)
        model.to(device)
        logger.info(f"✅ 端到端模型创建成功")
        
        # 打印模型参数信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"📊 模型参数统计:")
        logger.info(f"  - 总参数: {total_params:,}")
        logger.info(f"  - 可训练参数: {trainable_params:,}")
        logger.info(f"  - 可训练比例: {trainable_params/total_params*100:.1f}%")
        
    except Exception as e:
        logger.error(f"❌ 模型创建失败: {e}")
        raise
    
    # 创建优化器
    optimizer = create_optimizers(model, config)
    
    # 创建学习率调度器
    training_config = config['training']
    num_training_steps = len(train_loader) * training_config['epochs'] // training_config.get('accumulation_steps', 1)
    warmup_steps = int(num_training_steps * training_config['warmup_ratio'])
    
    if training_config.get('scheduler') == 'cosine_with_restarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=warmup_steps,
            T_mult=2
        )
    elif training_config.get('scheduler') == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )
    else:
        scheduler = None
    
    # 初始化训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1_macro': [],
        'val_f1_weighted': []
    }
    
    # 初始化最佳验证F1分数
    best_val_f1 = 0.0
    patience_counter = 0
    early_stopping_patience = training_config.get('early_stopping_patience', 10)
    
    # 训练循环
    logger.info(f"🚀 开始端到端训练...")
    for epoch in range(training_config['epochs']):
        logger.info(f"Epoch {epoch+1}/{training_config['epochs']}")
        
        # 训练一个epoch
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            clip_grad_val=training_config['gradient_clip_val'],
            accumulation_steps=training_config.get('accumulation_steps', 1)
        )
        
        # 在验证集上评估
        val_metrics = evaluate(
            model=model,
            data_loader=dev_loader,
            device=device
        )
        
        # 更新学习率（如果使用ReduceLROnPlateau）
        if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_metrics['f1_weighted'])
        
        # 记录指标
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        history['val_f1_weighted'].append(val_metrics['f1_weighted'])
        
        # 打印指标
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}")
        
        # 保存最佳模型
        if val_metrics['f1_weighted'] > best_val_f1:
            best_val_f1 = val_metrics['f1_weighted']
            patience_counter = 0
            
            # 保存模型
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_metrics': val_metrics,
                'history': history
            }
            
            best_model_path = os.path.join(exp_dir, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            
            logger.info(f"💾 保存最佳模型，F1 Weighted: {val_metrics['f1_weighted']:.4f}")
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= early_stopping_patience:
            logger.info(f"⏰ 早停触发，最佳F1: {best_val_f1:.4f}")
            break
    
    # 在测试集上评估最佳模型
    logger.info("🎯 在测试集上评估最佳模型...")
    
    # 加载最佳模型
    best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # 测试集评估
    test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        device=device
    )
    
    logger.info(f"🏆 测试集指标: Acc: {test_metrics['acc']:.4f}, F1 Macro: {test_metrics['f1_macro']:.4f}, F1 Weighted: {test_metrics['f1_weighted']:.4f}")
    
    # 保存训练历史
    history_path = os.path.join(exp_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # 保存测试结果
    test_results = {
        'test_metrics': test_metrics,
        'best_val_f1': best_val_f1,
        'total_epochs': epoch + 1
    }
    
    results_path = os.path.join(exp_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"✅ 端到端训练完成！")
    logger.info(f"📊 结果保存在: {exp_dir}")
    
    return model, history


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='端到端多模态情感分析训练')
    parser.add_argument('--config', type=str, default='configs/config_end_to_end_emotion.json',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    logger.info(f"🎯 开始端到端训练")
    logger.info(f"📝 配置文件: {args.config}")
    
    try:
        model, history = train_end_to_end_model(args.config)
        logger.info(f"🎉 训练成功完成！")
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 