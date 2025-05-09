#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
壁纸生成AI模型训练脚本

此脚本用于训练壁纸生成AI模型。支持从头开始训练或进行增量训练。
可以使用自定义数据集，并支持进度报告，方便在Electron应用中显示训练进度。

基本用法：
    python train.py '{"dataset_path":"./dataset","epochs":10,"model_path":"./models/model.pth"}'

增量训练：
    python train.py '{"dataset_path":"./dataset","epochs":5,"base_model_path":"./models/pretrained.pth","model_path":"./models/fine_tuned.pth"}'
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

# 尝试导入torch和相关库
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("警告: 未检测到PyTorch，训练功能将不可用")

# 默认训练配置
DEFAULT_CONFIG = {
    "dataset_path": "./dataset",  # 数据集路径
    "model_path": "./models/wallpaper_model.pth",  # 保存模型路径
    "base_model_path": None,  # 基础模型路径（用于增量训练）
    "epochs": 10,  # 训练轮数
    "batch_size": 4,  # 批处理大小
    "learning_rate": 0.0002,  # 学习率
    "image_size": 256,  # 训练图像大小
    "save_interval": 1,  # 每多少个epoch保存一次
    "report_interval": 10,  # 每多少个批次报告一次进度
    "seed": None,  # 随机种子
    "model_type": "basic"  # 模型类型：basic、style_transfer、diffusion
}

# 训练数据集类
class WallpaperDataset(Dataset):
    """壁纸训练数据集类"""
    
    def __init__(self, dataset_path, transform=None):
        """初始化数据集"""
        self.dataset_path = Path(dataset_path)
        self.image_files = []
        
        # 支持的图像格式
        valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        
        # 收集所有图像文件
        if self.dataset_path.exists():
            for ext in valid_extensions:
                self.image_files.extend(list(self.dataset_path.glob(f"**/*{ext}")))
        else:
            print(f"警告: 数据集路径 {dataset_path} 不存在")
        
        # 设置转换器
        self.transform = transform
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """获取数据集项"""
        img_path = self.image_files[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
                
            # 对于简单的自编码器训练，输入和目标相同
            return image, image
            
        except Exception as e:
            print(f"警告: 无法加载图像 {img_path}: {e}")
            # 返回随机数据作为替代
            if self.transform:
                return torch.randn(3, 256, 256), torch.randn(3, 256, 256)
            else:
                return None, None

# 简单的自编码器模型
class SimpleAutoencoder(nn.Module):
    """简单的自编码器模型，用于生成壁纸"""
    
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU()
        )
        
        # 潜在空间
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 256 * 16 * 16)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 256x256
            nn.Sigmoid()  # 输出范围为[0,1]
        )
    
    def encode(self, x):
        """编码过程"""
        x = self.encoder(x)
        x = x.view(-1, 256 * 16 * 16)
        x = self.fc1(x)
        return x
    
    def decode(self, x):
        """解码过程"""
        x = self.fc2(x)
        x = x.view(-1, 256, 16, 16)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        """前向传播"""
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

# 风格迁移模型 (简化版)
class StyleTransferModel(nn.Module):
    """简化的风格迁移模型"""
    
    def __init__(self):
        super(StyleTransferModel, self).__init__()
        # 使用更简化的网络架构
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x128
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # 残差块
        self.res_blocks = nn.Sequential(
            self._make_res_block(128),
            self._make_res_block(128),
            self._make_res_block(128),
            self._make_res_block(128)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 256x256
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def _make_res_block(self, channels):
        """创建残差块"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        """前向传播"""
        x = self.encoder(x)
        x = self.res_blocks(x) + x  # 残差连接
        x = self.decoder(x)
        return x

# 模型工厂函数
def create_model(model_type):
    """根据类型创建模型"""
    if model_type == "style_transfer":
        return StyleTransferModel()
    else:  # "basic" 或 其他
        return SimpleAutoencoder()

# 训练类
class WallpaperModelTrainer:
    """壁纸模型训练器"""
    
    def __init__(self, config):
        """初始化训练器"""
        self.config = {**DEFAULT_CONFIG, **config}
        
        # 设置随机种子
        if self.config["seed"] is not None:
            random.seed(self.config["seed"])
            np.random.seed(self.config["seed"])
            torch.manual_seed(self.config["seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config["seed"])
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 创建数据转换
        self.transform = transforms.Compose([
            transforms.Resize((self.config["image_size"], self.config["image_size"])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        
        # 创建数据集和加载器
        self.dataset = WallpaperDataset(self.config["dataset_path"], self.transform)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=0  # 可以根据系统调整
        )
        
        print(f"数据集大小: {len(self.dataset)} 图像")
        
        # 创建模型
        self.model = create_model(self.config["model_type"]).to(self.device)
        
        # 如果提供了基础模型，加载它
        if self.config["base_model_path"] and os.path.exists(self.config["base_model_path"]):
            try:
                self.model.load_state_dict(torch.load(self.config["base_model_path"]))
                print(f"成功加载基础模型: {self.config['base_model_path']}")
            except Exception as e:
                print(f"加载模型失败: {e}")
        
        # 定义损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config["learning_rate"]
        )
        
        # 创建模型保存目录
        os.makedirs(os.path.dirname(self.config["model_path"]), exist_ok=True)
        
        # 进度回调函数
        self.progress_callback = lambda progress: None
    
    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def train(self):
        """开始训练"""
        total_epochs = self.config["epochs"]
        total_batches = len(self.dataloader)
        
        print(f"开始训练 {total_epochs} 轮，每轮 {total_batches} 批次")
        
        # 训练主循环
        for epoch in range(total_epochs):
            epoch_start_time = time.time()
            running_loss = 0.0
            
            # 遍历批次
            for batch_idx, (inputs, targets) in enumerate(self.dataloader):
                # 将数据移到设备
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # 反向传播和优化
                loss.backward()
                self.optimizer.step()
                
                # 更新统计
                running_loss += loss.item()
                
                # 报告进度
                if batch_idx % self.config["report_interval"] == 0:
                    avg_loss = running_loss / (batch_idx + 1)
                    progress = {
                        "epoch": epoch + 1,
                        "total_epochs": total_epochs,
                        "batch": batch_idx + 1,
                        "total_batches": total_batches,
                        "loss": avg_loss,
                        "progress": (epoch * total_batches + batch_idx + 1) / (total_epochs * total_batches)
                    }
                    
                    self.progress_callback(progress)
                    print(f"轮次 [{epoch+1}/{total_epochs}], 批次 [{batch_idx+1}/{total_batches}], 损失: {avg_loss:.4f}")
            
            # 计算整个epoch的平均损失
            epoch_loss = running_loss / total_batches
            epoch_time = time.time() - epoch_start_time
            
            print(f"轮次 {epoch+1} 完成，损失: {epoch_loss:.4f}, 用时: {epoch_time:.2f}s")
            
            # 保存模型检查点
            if (epoch + 1) % self.config["save_interval"] == 0:
                checkpoint_path = self.config["model_path"].replace(".pth", f"_epoch{epoch+1}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"保存检查点: {checkpoint_path}")
        
        # 保存最终模型
        torch.save(self.model.state_dict(), self.config["model_path"])
        print(f"训练完成，保存模型到: {self.config['model_path']}")
        
        return {
            "status": "完成",
            "epochs": total_epochs,
            "final_loss": epoch_loss,
            "model_path": self.config["model_path"]
        }

# 用于模拟训练的类（当没有PyTorch时使用）
class MockTrainer:
    """模拟训练过程，用于在没有PyTorch时提供接口一致性"""
    
    def __init__(self, config):
        self.config = {**DEFAULT_CONFIG, **config}
        print("警告: 使用模拟训练器（PyTorch不可用）")
        
        # 进度回调函数
        self.progress_callback = lambda progress: None
    
    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def train(self):
        """模拟训练过程"""
        total_epochs = self.config["epochs"]
        
        # 确保模型目录存在
        os.makedirs(os.path.dirname(self.config["model_path"]), exist_ok=True)
        
        # 模拟训练主循环
        for epoch in range(total_epochs):
            # 随机训练损失（模拟训练过程）
            loss = 100.0 / (epoch + 1)
            
            # 模拟多个训练批次
            batches = 10
            for batch in range(batches):
                time.sleep(0.1)  # 模拟计算时间
                
                # 报告进度
                progress = {
                    "epoch": epoch + 1,
                    "total_epochs": total_epochs,
                    "batch": batch + 1,
                    "total_batches": batches,
                    "loss": loss * (1 - batch / batches * 0.1),  # 每个批次损失略微下降
                    "progress": (epoch * batches + batch + 1) / (total_epochs * batches)
                }
                
                self.progress_callback(progress)
            
            print(f"模拟轮次 {epoch+1}/{total_epochs} 完成，损失: {loss:.4f}")
        
        # 创建一个假的模型文件
        model_file = self.config["model_path"]
        with open(model_file, "w") as f:
            f.write(f"模拟模型文件 - 创建于 {datetime.now().isoformat()}")
        
        print(f"模拟训练完成，模型保存到: {model_file}")
        
        return {
            "status": "完成（模拟）",
            "epochs": total_epochs,
            "final_loss": loss,
            "model_path": model_file
        }

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="壁纸模型训练器")
    parser.add_argument('json_config', nargs='?', default=None, 
                        help='JSON格式的配置参数')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 从命令行JSON或使用默认配置
    if args.json_config:
        try:
            config = json.loads(args.json_config)
        except json.JSONDecodeError:
            print(f"错误: 无法解析JSON配置: {args.json_config}")
            return 1
    else:
        # 从stdin读取（用于Electron通信）
        if not sys.stdin.isatty():
            try:
                config = json.load(sys.stdin)
            except json.JSONDecodeError:
                print("错误: 无法从标准输入解析JSON配置")
                return 1
        else:
            config = DEFAULT_CONFIG
    
    # 创建训练器
    if HAS_TORCH:
        trainer = WallpaperModelTrainer(config)
    else:
        trainer = MockTrainer(config)
    
    # 设置进度回调，将进度写入stdout以便Electron读取
    def progress_callback(progress):
        print(json.dumps({"progress": progress}), flush=True)
    
    trainer.set_progress_callback(progress_callback)
    
    # 执行训练
    try:
        result = trainer.train()
        print(json.dumps({
            "success": True,
            "result": result
        }))
        return 0
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e)
        }))
        return 1

if __name__ == "__main__":
    sys.exit(main())
