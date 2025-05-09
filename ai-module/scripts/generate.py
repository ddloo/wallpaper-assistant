#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
壁纸生成模块 - 基于扩散模型生成高质量壁纸

此脚本用于根据文本提示生成壁纸图像。它可以使用预训练的模型（如Stable Diffusion）
或自定义训练的模型。脚本设计为可以从Electron应用调用，支持JSON参数传递。

基本用法：
    python generate.py '{"prompt":"美丽的山脉日落","width":1920,"height":1080,"output_path":"wallpaper.png"}'

高级用法：
    python generate.py '{"prompt":"美丽的山脉日落","width":1920,"height":1080,"model_path":"./models/custom_model.pth","guidance_scale":7.5,"output_path":"wallpaper.png"}'
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# 尝试导入torch和相关库，如果导入失败则使用基本图像生成
try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("警告: 未检测到PyTorch，将使用基本图像生成方法")

# 默认配置
DEFAULT_CONFIG = {
    "prompt": "美丽的自然风景",
    "width": 1920,
    "height": 1080,
    "model_path": None,  # 如果为None则使用默认模型
    "guidance_scale": 7.5,
    "num_inference_steps": 50,
    "seed": None,  # 随机种子，None表示随机生成
    "output_path": "wallpaper.png",
    "style": "自然",  # 可选风格：自然、抽象、极简、科幻、梦幻
    "local_files_only": False,  # 是否只使用本地缓存的文件
    "local_cache_dir": None,  # 本地模型缓存目录，None使用默认缓存路径
    "max_retries": 3  # 模型下载重试次数
}

# AI模型基类
class WallpaperGenerator:
    """壁纸生成器基类"""
    def __init__(self, config):
        self.config = {**DEFAULT_CONFIG, **config}
        
        # 设置随机种子以确保结果可重现（如果指定了种子）
        if self.config["seed"] is not None:
            random.seed(self.config["seed"])
            np.random.seed(self.config["seed"])
            if HAS_TORCH:
                torch.manual_seed(self.config["seed"])
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.config["seed"])
    
    def generate(self):
        """生成壁纸的主方法，需要在子类中实现"""
        raise NotImplementedError("子类必须实现generate方法")

    def save_image(self, image, output_path=None):
        """保存生成的图像到指定路径"""
        if output_path is None:
            output_path = self.config["output_path"]
            
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # 保存图像
        image.save(output_path)
        return output_path

# 基本壁纸生成器（无需AI模型）
class BasicWallpaperGenerator(WallpaperGenerator):
    """基本壁纸生成器，使用PIL创建简单图像"""
    
    def __init__(self, config):
        super().__init__(config)
        self.styles = {
            "自然": self._generate_nature,
            "抽象": self._generate_abstract,
            "极简": self._generate_minimalist,
            "科幻": self._generate_scifi,
            "梦幻": self._generate_dreamy
        }
    
    def generate(self):
        """根据配置生成壁纸"""
        style = self.config["style"]
        if style in self.styles:
            image = self.styles[style]()
        else:
            # 默认使用自然风格
            image = self._generate_nature()
            
        return self.save_image(image)
    
    def _generate_gradient(self, colors, direction="horizontal"):
        """生成渐变背景"""
        width, height = self.config["width"], self.config["height"]
        image = Image.new('RGB', (width, height), color=colors[0])
        draw = ImageDraw.Draw(image)
        
        steps = 255
        if direction == "horizontal":
            for i in range(steps):
                # 计算当前位置和颜色
                x = int((width / steps) * i)
                if len(colors) == 2:
                    r = int(colors[0][0] + (colors[1][0] - colors[0][0]) * i / steps)
                    g = int(colors[0][1] + (colors[1][1] - colors[0][1]) * i / steps)
                    b = int(colors[0][2] + (colors[1][2] - colors[0][2]) * i / steps)
                    color = (r, g, b)
                    draw.line([(x, 0), (x, height)], fill=color, width=2)
                else:  # 多点渐变
                    idx = int((len(colors) - 1) * i / steps)
                    blend = (len(colors) - 1) * i / steps - idx
                    r = int(colors[idx][0] * (1 - blend) + colors[idx + 1][0] * blend)
                    g = int(colors[idx][1] * (1 - blend) + colors[idx + 1][1] * blend)
                    b = int(colors[idx][2] * (1 - blend) + colors[idx + 1][2] * blend)
                    color = (r, g, b)
                    draw.line([(x, 0), (x, height)], fill=color, width=2)
        else:  # vertical
            for i in range(steps):
                y = int((height / steps) * i)
                if len(colors) == 2:
                    r = int(colors[0][0] + (colors[1][0] - colors[0][0]) * i / steps)
                    g = int(colors[0][1] + (colors[1][1] - colors[0][1]) * i / steps)
                    b = int(colors[0][2] + (colors[1][2] - colors[0][2]) * i / steps)
                    color = (r, g, b)
                    draw.line([(0, y), (width, y)], fill=color, width=2)
                else:  # 多点渐变
                    idx = int((len(colors) - 1) * i / steps)
                    blend = (len(colors) - 1) * i / steps - idx
                    r = int(colors[idx][0] * (1 - blend) + colors[idx + 1][0] * blend)
                    g = int(colors[idx][1] * (1 - blend) + colors[idx + 1][1] * blend)
                    b = int(colors[idx][2] * (1 - blend) + colors[idx + 1][2] * blend)
                    color = (r, g, b)
                    draw.line([(0, y), (width, y)], fill=color, width=2)
                    
        return image
    
    def _generate_nature(self):
        """生成自然风格壁纸"""
        # 使用蓝色和绿色的渐变模拟天空和山脉
        colors = [(66, 165, 245), (100, 181, 246), (144, 202, 249)]  # 蓝色渐变
        image = self._generate_gradient(colors, "vertical")
        
        # 添加模拟山脉
        width, height = self.config["width"], self.config["height"]
        draw = ImageDraw.Draw(image)
        
        # 山脉底部高度
        mountain_base = int(height * 0.6)
        
        # 生成山脉轮廓点
        points = [(0, height)]
        n_peaks = random.randint(5, 10)
        for i in range(n_peaks + 2):
            x = int(width * i / (n_peaks + 1))
            y_offset = random.randint(-int(height * 0.2), int(height * 0.1))
            y = mountain_base + y_offset
            points.append((x, y))
        points.append((width, height))
        
        # 绘制山脉
        mountain_color = (46, 125, 50)
        draw.polygon(points, fill=mountain_color)
        
        # 添加太阳或月亮
        if random.random() > 0.5:  # 太阳
            sun_x = random.randint(int(width * 0.1), int(width * 0.9))
            sun_y = random.randint(int(height * 0.1), int(height * 0.3))
            sun_radius = random.randint(int(height * 0.05), int(height * 0.1))
            sun_color = (255, 236, 179)
            draw.ellipse(
                [(sun_x - sun_radius, sun_y - sun_radius), 
                 (sun_x + sun_radius, sun_y + sun_radius)], 
                fill=sun_color
            )
        else:  # 月亮
            moon_x = random.randint(int(width * 0.1), int(width * 0.9))
            moon_y = random.randint(int(height * 0.1), int(height * 0.3))
            moon_radius = random.randint(int(height * 0.03), int(height * 0.07))
            moon_color = (255, 255, 255)
            draw.ellipse(
                [(moon_x - moon_radius, moon_y - moon_radius), 
                 (moon_x + moon_radius, moon_y + moon_radius)], 
                fill=moon_color
            )
        
        return image
    
    def _generate_abstract(self):
        """生成抽象风格壁纸"""
        width, height = self.config["width"], self.config["height"]
        image = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # 创建随机形状
        num_shapes = random.randint(20, 50)
        for _ in range(num_shapes):
            shape_type = random.choice(['circle', 'rectangle', 'line'])
            color = (random.randint(0, 255), 
                     random.randint(0, 255), 
                     random.randint(0, 255))
            
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            
            if shape_type == 'circle':
                radius = random.randint(10, 100)
                draw.ellipse(
                    [(x1 - radius, y1 - radius), 
                     (x1 + radius, y1 + radius)], 
                    fill=color
                )
            elif shape_type == 'rectangle':
                x2 = random.randint(x1, min(x1 + 200, width))
                y2 = random.randint(y1, min(y1 + 200, height))
                draw.rectangle([(x1, y1), (x2, y2)], fill=color)
            else:  # line
                x2 = random.randint(0, width)
                y2 = random.randint(0, height)
                line_width = random.randint(1, 10)
                draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
        
        # 应用模糊效果
        image = image.filter(ImageFilter.GaussianBlur(radius=5))
        
        return image
    
    def _generate_minimalist(self):
        """生成极简风格壁纸"""
        width, height = self.config["width"], self.config["height"]
        
        # 选择背景色
        bg_color = random.choice([
            (240, 240, 240),  # 浅灰色
            (250, 250, 250),  # 接近白色
            (245, 245, 245),  # 灰白色
            (235, 235, 235),  # 浅灰色
        ])
        
        # 创建背景
        image = Image.new('RGB', (width, height), color=bg_color)
        draw = ImageDraw.Draw(image)
        
        # 选择前景色
        fg_color = random.choice([
            (52, 58, 64),    # 深灰色
            (33, 37, 41),    # 近黑色
            (0, 123, 255),   # 蓝色
            (220, 53, 69),   # 红色
            (40, 167, 69),   # 绿色
        ])
        
        # 绘制简单几何图形
        shape_type = random.choice(['line', 'circle', 'square'])
        
        if shape_type == 'line':
            # 绘制水平或垂直线
            if random.random() > 0.5:  # 水平线
                y_pos = height // 2
                line_width = random.randint(1, 5)
                draw.line([(0, y_pos), (width, y_pos)], fill=fg_color, width=line_width)
            else:  # 垂直线
                x_pos = width // 2
                line_width = random.randint(1, 5)
                draw.line([(x_pos, 0), (x_pos, height)], fill=fg_color, width=line_width)
                
        elif shape_type == 'circle':
            # 绘制一个圆形
            center_x = width // 2
            center_y = height // 2
            radius = min(width, height) // 6
            draw.ellipse(
                [(center_x - radius, center_y - radius), 
                 (center_x + radius, center_y + radius)], 
                outline=fg_color, width=2
            )
            
        else:  # square
            # 绘制一个正方形
            center_x = width // 2
            center_y = height // 2
            side = min(width, height) // 5
            half_side = side // 2
            draw.rectangle(
                [(center_x - half_side, center_y - half_side),
                 (center_x + half_side, center_y + half_side)],
                outline=fg_color, width=2
            )
            
        return image
    
    def _generate_scifi(self):
        """生成科幻风格壁纸"""
        width, height = self.config["width"], self.config["height"]
        
        # 选择深色背景
        bg_color = random.choice([
            (0, 0, 0),        # 黑色
            (10, 20, 30),     # 深蓝黑色
            (20, 20, 40),     # 深紫黑色
            (10, 30, 30),     # 深青黑色
        ])
        
        # 创建背景
        image = Image.new('RGB', (width, height), color=bg_color)
        draw = ImageDraw.Draw(image)
        
        # 添加网格线
        grid_color = random.choice([
            (0, 123, 255),   # 蓝色
            (80, 200, 255),  # 亮蓝色
            (0, 200, 200),   # 青色
            (200, 0, 200),   # 紫色
        ])
        
        # 生成网格
        grid_spacing = random.randint(40, 100)
        line_width = random.randint(1, 2)
        
        for x in range(0, width, grid_spacing):
            draw.line([(x, 0), (x, height)], fill=grid_color, width=line_width)
            
        for y in range(0, height, grid_spacing):
            draw.line([(0, y), (width, y)], fill=grid_color, width=line_width)
        
        # 添加科技元素
        num_elements = random.randint(10, 30)
        for _ in range(num_elements):
            element_type = random.choice(['dot', 'small_circle', 'rectangle'])
            x = random.randint(0, width)
            y = random.randint(0, height)
            
            if element_type == 'dot':
                draw.point((x, y), fill=grid_color)
            elif element_type == 'small_circle':
                radius = random.randint(2, 8)
                draw.ellipse(
                    [(x - radius, y - radius), (x + radius, y + radius)],
                    outline=grid_color, width=1
                )
            else:  # rectangle
                rect_width = random.randint(5, 30)
                rect_height = random.randint(5, 30)
                draw.rectangle(
                    [(x, y), (x + rect_width, y + rect_height)],
                    outline=grid_color, width=1
                )
        
        # 添加一个大圆形作为焦点
        focus_x = random.randint(width // 4, 3 * width // 4)
        focus_y = random.randint(height // 4, 3 * height // 4)
        focus_radius = random.randint(min(width, height) // 10, min(width, height) // 5)
        draw.ellipse(
            [(focus_x - focus_radius, focus_y - focus_radius), 
             (focus_x + focus_radius, focus_y + focus_radius)], 
            outline=grid_color, width=2
        )
        
        # 添加随机的小文本（模拟代码或数据）
        for _ in range(5):
            text_x = random.randint(0, width - 100)
            text_y = random.randint(0, height - 20)
            # 模拟代码或数据的文本
            text = "0101" * random.randint(1, 5)
            draw.text((text_x, text_y), text, fill=grid_color)
        
        return image
    
    def _generate_dreamy(self):
        """生成梦幻风格壁纸"""
        width, height = self.config["width"], self.config["height"]
        
        # 选择梦幻色彩
        colors = random.choice([
            [(147, 112, 219), (199, 125, 255)],  # 紫色渐变
            [(255, 182, 193), (255, 218, 233)],  # 粉色渐变
            [(135, 206, 235), (176, 226, 255)],  # 天蓝色渐变
            [(255, 165, 0), (255, 215, 0)],      # 金色渐变
            [(64, 224, 208), (0, 255, 127)],     # 绿松石渐变
        ])
        
        # 创建渐变背景
        image = self._generate_gradient(colors, random.choice(["horizontal", "vertical"]))
        
        # 添加大量的模糊光点
        draw = ImageDraw.Draw(image)
        num_lights = random.randint(30, 70)
        
        for _ in range(num_lights):
            light_color = random.choice([
                (255, 255, 255),  # 白色
                (255, 255, 220),  # 淡黄色
                (255, 220, 255),  # 淡紫色
                (220, 255, 255),  # 淡青色
            ])
            
            x = random.randint(0, width)
            y = random.randint(0, height)
            radius = random.randint(1, 10)
            
            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)],
                fill=light_color
            )
        
        # 添加模糊效果以创建梦幻感
        image = image.filter(ImageFilter.GaussianBlur(radius=random.randint(2, 6)))
        
        # 增加亮度
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)
        
        return image

# 如果有PyTorch，使用更高级的方法
if HAS_TORCH:
    class AIWallpaperGenerator(WallpaperGenerator):
        """基于神经网络的壁纸生成器"""
        
        def __init__(self, config):
            super().__init__(config)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"使用设备: {self.device}")
            
            # 尝试加载模型，如果模型加载失败则使用基本生成器
            try:
                self._load_model()
                self.model_loaded = True
            except Exception as e:
                print(f"模型加载失败: {e}")
                print("将使用基本生成方法")
                self.model_loaded = False
                self.fallback_generator = BasicWallpaperGenerator(config)
        
        def _load_model(self):
            """加载预训练模型或微调模型"""
            # 决定使用哪种模型加载策略
            model_strategy = self.config.get("model_strategy", "custom")
            
            if model_strategy == "diffusion":
                # 使用Stable Diffusion或其微调版本
                self._load_diffusion_model()
            else:
                # 使用自定义训练的模型
                self._load_custom_model()
        
        def _load_custom_model(self):
            """加载使用train.py训练的自定义模型"""
            # 确定模型路径
            model_path = self.config.get("model_path")
            if not model_path or not os.path.exists(model_path):
                # 如果未指定模型或模型不存在，尝试使用默认路径
                model_path = os.path.join(os.path.dirname(__file__), "../models/wallpaper_model.pth")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 根据模型类型创建模型实例
            # 导入train模块中定义的模型类
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            try:
                from scripts.train import create_model
                
                model_type = self.config.get("model_type", "basic")
                self.model = create_model(model_type)
                
                # 加载模型权重
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()  # 设置为评估模式
                
                print(f"成功加载自定义模型: {model_path}")
                self.model_type = "custom"
            except Exception as e:
                raise RuntimeError(f"加载自定义模型失败: {e}")
        
        def _load_diffusion_model(self):
            """加载Stable Diffusion或其微调版本"""
            try:
                from diffusers import StableDiffusionPipeline
                import torch
                import time
                import os
                from huggingface_hub import snapshot_download, HfFolder, HfApi
                from huggingface_hub.utils import HfHubHTTPError
                from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
                
                # 设置环境变量来优化HF下载器
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # 启用更快的下载器
                
                # 检查是否有微调模型路径
                fine_tuned_model_path = self.config.get("model_path")
                base_model_id = self.config.get("base_model_id", "runwayml/stable-diffusion-v1-5")
                local_files_only = self.config.get("local_files_only", False)
                max_retries = self.config.get("max_retries", 3)  # 最大重试次数
                local_cache_dir = self.config.get("local_cache_dir", None)
                resume_download = self.config.get("resume_download", True)  # 默认尝试断点续传
                
                # 确定torch数据类型
                torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
                
                # 是否禁用安全检查器
                disable_safety_checker = self.config.get("disable_safety_checker", False)
        
                # 预先准备参数
                pretrained_kwargs = {
                    "torch_dtype": torch_dtype,
                    "local_files_only": local_files_only,
                    "resume_download": resume_download,
                    "low_cpu_mem_usage": True,  # 降低内存使用
                }
        
                # 如果禁用安全检查器
                if disable_safety_checker:
                    pretrained_kwargs["safety_checker"] = None
                    print("警告: 已禁用NSFW内容检测")
                
                # 添加缓存目录参数（如果指定了）
                if local_cache_dir:
                    if not os.path.exists(local_cache_dir):
                        os.makedirs(local_cache_dir, exist_ok=True)
                    pretrained_kwargs["cache_dir"] = local_cache_dir
                    print(f"使用自定义缓存目录: {local_cache_dir}")
                    
                    # 预先检查缓存目录中是否已有完整模型
                    model_cache_path = None
                    if not fine_tuned_model_path and not local_files_only:
                        potential_model_path = os.path.join(local_cache_dir, base_model_id.split('/')[-1])
                        if os.path.exists(potential_model_path) and \
                           os.path.exists(os.path.join(potential_model_path, 'model_index.json')):
                            print(f"发现缓存中存在完整模型: {potential_model_path}")
                            model_cache_path = potential_model_path
                            local_files_only = True
                            pretrained_kwargs["local_files_only"] = True
                
                # 加载模型
                if fine_tuned_model_path:
                    # 检查是否为本地路径
                    is_local_path = os.path.exists(fine_tuned_model_path)
                    
                    # 如果是Hugging Face模型ID (形如 'organization/model') 或本地存在
                    if '/' in fine_tuned_model_path or is_local_path:
                        print(f"加载模型: {fine_tuned_model_path}")
                        
                        # 处理本地目录
                        if is_local_path and os.path.isdir(fine_tuned_model_path):
                            self.pipe = self._load_with_retry(
                                lambda: StableDiffusionPipeline.from_pretrained(
                                    fine_tuned_model_path,
                                    **pretrained_kwargs
                                ),
                                max_retries=max_retries
                            )
                        # 处理Hugging Face模型ID或本地单文件
                        elif '/' in fine_tuned_model_path:  # 处理Hugging Face模型ID
                            self.pipe = self._load_with_retry(
                                lambda: StableDiffusionPipeline.from_pretrained(
                                    fine_tuned_model_path,
                                    **pretrained_kwargs
                                ),
                                max_retries=max_retries
                            )
                        # 处理本地单文件模型
                        elif is_local_path:
                            # 先加载基础模型
                            self.pipe = self._load_with_retry(
                                lambda: StableDiffusionPipeline.from_pretrained(
                                    base_model_id,
                                    **pretrained_kwargs
                                ),
                                max_retries=max_retries
                            )
                            # 加载微调的UNet权重
                            unet_state_dict = torch.load(fine_tuned_model_path, map_location=self.device)
                            # 仅当状态字典包含UNet参数时才尝试加载
                            if any(k.startswith("unet.") for k in unet_state_dict.keys()):
                                # 处理带前缀的情况
                                if list(unet_state_dict.keys())[0].startswith("unet."):
                                    self.pipe.unet.load_state_dict(unet_state_dict)
                                else:
                                    self.pipe.unet.load_state_dict(unet_state_dict, strict=False)
                            else:
                                # 如果是完整模型状态字典，直接加载
                                self.pipe.load_state_dict(unet_state_dict, strict=False)
                    else:
                        print(f"未找到模型 {fine_tuned_model_path}，使用基础模型: {base_model_id}")
                        # 使用基础模型
                        self.pipe = self._load_with_retry(
                            lambda: StableDiffusionPipeline.from_pretrained(
                                base_model_id,
                                safety_checker=None,  # 禁用NSFW检查
                                **pretrained_kwargs
                            ),
                            max_retries=max_retries
                        )
                else:
                    print(f"使用基础模型: {base_model_id}")
                    
                    # 如果有已缓存的模型路径，使用它
                    if model_cache_path:
                        self.pipe = self._load_with_retry(
                            lambda: StableDiffusionPipeline.from_pretrained(
                                model_cache_path,
                                **pretrained_kwargs
                            ),
                            max_retries=max_retries
                        )
                    else:
                        # 否则从 Hugging Face 加载
                        self.pipe = self._load_with_retry(
                            lambda: StableDiffusionPipeline.from_pretrained(
                                base_model_id,
                                safety_checker=None,  # 禁用NSFW检查
                                **pretrained_kwargs
                            ),
                            max_retries=max_retries
                        )
                        
                        # 如果指定了缓存目录，保存完整模型副本
                        if local_cache_dir and not local_files_only:
                            target_dir = os.path.join(local_cache_dir, base_model_id.split('/')[-1])
                            if not os.path.exists(target_dir):
                                os.makedirs(target_dir, exist_ok=True)
                                print(f"保存模型副本到: {target_dir}")
                                self.pipe.save_pretrained(target_dir)
                
                # 移动到设备
                self.pipe = self.pipe.to(self.device)
                
                # 优化推理
                if self.device == "cuda":
                    self.pipe.enable_attention_slicing()
                    # 可选：启用CUDA图优化
                    if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                        self.pipe.enable_xformers_memory_efficient_attention()
                
                print("Diffusion模型加载成功!")
                self.model_type = "diffusion"
                
            except ImportError as e:
                raise ImportError(f"缺少必要的依赖: {e}. 请安装: pip install diffusers transformers accelerate huggingface_hub")
            except Exception as e:
                print(f"加载 Diffusion 模型时出错: {e}")
                raise

        def _load_with_retry(self, load_fn, max_retries=3):
            """
            使用重试机制加载模型
            """
            retries = 0
            last_error = None
            
            while retries < max_retries:
                try:
                    print(f"尝试加载模型... (尝试 {retries+1}/{max_retries})")
                    return load_fn()
                except Exception as e:
                    last_error = e
                    retries += 1
                    print(f"加载失败: {e}")
                    if "IncompleteRead" in str(e) or "Connection broken" in str(e):
                        print(f"网络连接问题，等待后重试...")
                        time.sleep(5)  # 等待5秒后重试
                    else:
                        # 对于非网络问题，立即抛出异常
                        raise
            
            print(f"达到最大重试次数({max_retries})，无法加载模型")
            raise RuntimeError(f"模型加载失败，最后错误: {last_error}")
        
        def generate(self):
            """使用AI模型生成壁纸"""
            if not self.model_loaded:
                return self.fallback_generator.generate()
            
            # 使用自定义训练模型的情况
            if hasattr(self, 'model_type') and self.model_type == "custom":
                # 创建随机噪声或示例输入
                input_size = self.config.get("image_size", 256)
                input_tensor = torch.randn(1, 3, input_size, input_size).to(self.device)
                
                # 生成图像
                with torch.no_grad():
                    output_tensor = self.model(input_tensor)
                
                # 转换为PIL图像
                output_np = output_tensor[0].cpu().numpy().transpose(1, 2, 0)
                output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
                image = Image.fromarray(output_np)
                
            # 使用Diffusion模型的情况
            elif hasattr(self, 'model_type') and self.model_type == "diffusion":
                try:
                    # 获取生成参数
                    prompt = self.config["prompt"]
                    negative_prompt = self.config.get("negative_prompt", "模糊,低质量,低分辨率")
                    guidance_scale = float(self.config.get("guidance_scale", 7.5))
                    num_inference_steps = int(self.config.get("num_inference_steps", 50))
                    seed = self.config.get("seed")
                    
                    # 设置随机种子
                    if seed is not None:
                        generator = torch.Generator(device=self.device).manual_seed(int(seed))
                    else:
                        generator = None
                    
                    # 生成图像
                    print(f"使用提示词 '{prompt}' 生成图像...")
                    print(f"参数: guidance_scale={guidance_scale}, steps={num_inference_steps}")
                    
                    result = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        # 高级参数
                        width=self.config.get("diffusion_width", 512),
                        height=self.config.get("diffusion_height", 512),
                    )
                    
                    # 获取生成的图像
                    image = result.images[0]
                    print("图像生成成功!")
                    
                except Exception as e:
                    print(f"使用Diffusion模型生成图像失败: {e}")
                    return self.fallback_generator.generate()
            else:
                print("未知的模型类型，使用基本生成器作为后备方案")
                return self.fallback_generator.generate()
            
            # TODO: 需要自适应图片尺寸，还没做!!!!
            # # 调整为目标尺寸
            # if image.size != (self.config["width"], self.config["height"]):
            #     image = image.resize(
            #         (self.config["width"], self.config["height"]), 
            #         Image.LANCZOS
            #     )
            
            return self.save_image(image)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="壁纸生成器")
    parser.add_argument('json_config', nargs='?', default=None, 
                        help='JSON格式的配置参数或JSON配置文件路径')
    parser.add_argument('--file', '-f', action='store_true',
                        help='指定参数是文件路径而非JSON字符串')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 从命令行参数或文件中加载配置
    if args.json_config:
        # 如果指定了--file选项或者参数是json文件路径
        if args.file or (args.json_config.endswith('.json') and os.path.exists(args.json_config)):
            try:
                with open(args.json_config, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"从文件读取配置: {args.json_config}")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"错误: 无法从文件加载配置: {e}")
                return 1
        else:  # 尝试作为JSON字符串解析
            try:
                config = json.loads(args.json_config)
            except json.JSONDecodeError:
                print(f"错误: 无法解析JSON配置字符串")
                # 尝试作为文件路径处理
                if os.path.exists(args.json_config):
                    try:
                        with open(args.json_config, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        print(f"从文件读取配置: {args.json_config}")
                    except json.JSONDecodeError:
                        print(f"错误: 无法从文件加载 JSON: {args.json_config}")
                        return 1
                else:
                    return 1
    else:
        # 从stdin读取（用于Electron通信）
        if not sys.stdin.isatty():
            try:
                config = json.load(sys.stdin)
                print("从标准输入读取配置")
            except json.JSONDecodeError:
                print("错误: 无法从标准输入解析JSON配置")
                return 1
        else:
            config = DEFAULT_CONFIG
            print("使用默认配置")
    
    # 创建生成器
    if HAS_TORCH and config.get("use_ai", True):
        generator = AIWallpaperGenerator(config)
    else:
        generator = BasicWallpaperGenerator(config)
    
    # 生成壁纸
    try:
        start_time = time.time()
        output_path = generator.generate()
        end_time = time.time()
        
        print(json.dumps({
            "success": True,
            "path": output_path,
            "generation_time": end_time - start_time,
            "config": config
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
