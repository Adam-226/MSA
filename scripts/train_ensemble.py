#!/usr/bin/env python3
"""
集成训练脚本 - 训练多个不同随机种子的模型
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer import train_model
from src.training.config import load_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_ensemble_models(config_path: str, seeds: list, base_output_dir: str = "checkpoints/ensemble"):
    """
    训练多个不同随机种子的模型进行集成
    
    Args:
        config_path: 配置文件路径
        seeds: 随机种子列表
        base_output_dir: 输出目录基路径
    """
    
    # 创建集成输出目录
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 加载原始配置
    base_config = load_config(config_path)
    
    # 存储每个模型的训练结果
    ensemble_results = {
        'models': [],
        'seeds': seeds,
        'base_config': base_config
    }
    
    for i, seed in enumerate(seeds):
        logger.info(f"\n{'='*60}")
        logger.info(f"开始训练集成模型 {i+1}/{len(seeds)} (随机种子: {seed})")
        logger.info(f"{'='*60}")
        
        # 复制配置并修改随机种子
        config = base_config.copy()
        config['random_seed'] = seed
        config['model_name'] = f"FusionModel_{seed}"
        
        # 创建临时配置文件
        temp_config_path = f"configs/temp_config_seed_{seed}.json"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        try:
            # 训练模型
            model, history = train_model(temp_config_path)
            
            # 获取模型保存路径
            model_dir = f"checkpoints/FusionModel_{seed}"
            
            # 记录模型信息
            model_info = {
                'seed': seed,
                'model_dir': model_dir,
                'config_path': temp_config_path,
                'final_train_acc': history['train_acc'][-1],
                'final_val_acc': history['val_acc'][-1],
                'best_val_f1_weighted': max(history['val_f1_weighted']),
                'best_epoch': history['val_f1_weighted'].index(max(history['val_f1_weighted'])) + 1
            }
            
            ensemble_results['models'].append(model_info)
            
            logger.info(f"模型 {i+1} 训练完成:")
            logger.info(f"  - 随机种子: {seed}")
            logger.info(f"  - 最佳验证F1: {model_info['best_val_f1_weighted']:.4f}")
            logger.info(f"  - 最佳epoch: {model_info['best_epoch']}")
            logger.info(f"  - 模型保存路径: {model_dir}")
            
        except Exception as e:
            logger.error(f"训练模型 {i+1} (种子{seed}) 时出错: {str(e)}")
            continue
        
        finally:
            # 清理临时配置文件
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    
    # 保存集成结果信息
    ensemble_info_path = os.path.join(base_output_dir, 'ensemble_info.json')
    with open(ensemble_info_path, 'w', encoding='utf-8') as f:
        json.dump(ensemble_results, f, indent=4, ensure_ascii=False)
    
    logger.info(f"\n{'='*60}")
    logger.info("集成训练完成!")
    logger.info(f"成功训练了 {len(ensemble_results['models'])} 个模型")
    logger.info(f"集成信息保存在: {ensemble_info_path}")
    
    # 输出模型性能总结
    if ensemble_results['models']:
        logger.info("\n模型性能总结:")
        for i, model_info in enumerate(ensemble_results['models']):
            logger.info(f"  模型{i+1} (种子{model_info['seed']}): "
                       f"验证F1={model_info['best_val_f1_weighted']:.4f}")
        
        # 计算平均性能
        avg_val_f1 = sum(m['best_val_f1_weighted'] for m in ensemble_results['models']) / len(ensemble_results['models'])
        logger.info(f"\n平均验证F1: {avg_val_f1:.4f}")
    
    return ensemble_results


def main():
    parser = argparse.ArgumentParser(description='训练集成模型')
    parser.add_argument('--config', type=str, default='configs/config.json',
                       help='配置文件路径')
    parser.add_argument('--seeds', type=int, nargs='+', 
                       default=[42, 123, 456, 789, 999],
                       help='随机种子列表')
    parser.add_argument('--output_dir', type=str, default='checkpoints/ensemble',
                       help='集成模型输出目录')
    
    args = parser.parse_args()
    
    logger.info("开始集成训练...")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"随机种子: {args.seeds}")
    logger.info(f"输出目录: {args.output_dir}")
    
    # 训练集成模型
    results = train_ensemble_models(
        config_path=args.config,
        seeds=args.seeds,
        base_output_dir=args.output_dir
    )
    
    logger.info("集成训练全部完成!")


if __name__ == "__main__":
    main() 