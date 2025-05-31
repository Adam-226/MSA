"""
多模态情感分析模型训练脚本
"""
import os
import sys
import argparse
import logging
import json
from pathlib import Path

# 添加src目录到系统路径
sys.path.append(str(Path(__file__).parent))

import torch
from src.models.model import FusionModel
from src.preprocess.dataset import get_meld_dataloaders
from src.training.config import set_seed, load_config, get_device, get_class_weights, create_experiment_dir
from src.training.trainer import train_model
from src.utils.evaluator import ModelEvaluator

# 配置日志
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/training.log')
    ]
)
logger = logging.getLogger(__name__)


def train(args):
    """训练模型"""
    logger.info("训练多模态情感分析模型...")
    try:
        model, history = train_model(args.config)
        logger.info("训练完成！")
        return model, history
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise


def test(args):
    """测试模型"""
    logger.info("测试模型...")
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置随机种子
    set_seed(config['random_seed'])
    
    # 获取设备
    device = get_device()
    logger.info(f"使用设备: {device}")
    
    # 获取数据加载器
    dataloaders = get_meld_dataloaders(config)
    test_loader = dataloaders['test']
    
    # 创建模型
    model = FusionModel(config)
    
    # 加载模型权重
    logger.info(f"加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # 获取实验目录
    checkpoint_dir = os.path.dirname(args.checkpoint)
    
    # 创建评估器
    emotion_labels = ['neutral', 'joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust']
    evaluator = ModelEvaluator(model, device, emotion_labels)
    
    # 评估模型
    logger.info("在测试集上评估模型...")
    metrics, predictions, true_labels = evaluator.evaluate_on_loader(test_loader)
    
    # 生成详细报告
    logger.info("生成详细评估报告...")
    detailed_report = evaluator.generate_detailed_report(predictions, true_labels)
    
    # 加载训练历史（如果存在）
    history = None
    history_path = os.path.join(checkpoint_dir, 'history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
    
    # 创建可视化
    logger.info("生成可视化图表...")
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
    
    logger.info(f"\n详细评估结果已保存至: {results_path}")
    logger.info(f"可视化图表已保存至: {os.path.join(checkpoint_dir, 'evaluation_plots')}")
    
    return final_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多模态情感分析模型')
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', type=str, default='configs/config.json', 
                            help='配置文件路径')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='测试模型')
    test_parser.add_argument('--config', type=str, default='configs/config.json',
                           help='配置文件路径')
    test_parser.add_argument('--checkpoint', type=str, required=True,
                           help='模型检查点路径')
    
    # 解析参数
    args = parser.parse_args()
    
    # 根据命令执行相应操作
    if args.command == 'train':
        train(args)
    elif args.command == 'test':
        test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 