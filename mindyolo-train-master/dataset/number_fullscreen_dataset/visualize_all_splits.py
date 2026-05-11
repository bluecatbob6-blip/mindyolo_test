#!/usr/bin/env python3
"""
批量可视化数据集划分的标注
支持可视化 train/val/test 三个数据集的标注
"""
import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict


class SplitDatasetVisualizer:
    def __init__(self, base_dir: str = ".", class_names: Optional[List[str]] = None):
        """
        初始化可视化器
        
        Args:
            base_dir: 数据集基础目录（包含train/val/test）
            class_names: 类别名称列表，如果为None则使用数字0-9
        """
        self.base_dir = base_dir
        self.class_names = class_names or [str(i) for i in range(10)]
        self.splits = ['train', 'val', 'test']
        
        # 支持的图片格式
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', 
                                '.JPG', '.JPEG', '.PNG', '.BMP')
        
        # 颜色列表
        self.colors = self._generate_colors(len(self.class_names))
        
        # 当前状态
        self.current_split_idx = 0
        self.current_image_idx = 0
        self.all_images = {}  # {split_name: [image_files]}
        
        # 加载所有图片
        self._load_all_images()
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """生成不同颜色用于不同类别"""
        colors = []
        np.random.seed(42)
        
        for i in range(num_classes):
            hue = int(180 * i / num_classes)
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color_bgr)))
        
        return colors
    
    def _load_all_images(self):
        """加载所有数据集的图片列表"""
        for split_name in self.splits:
            img_dir = os.path.join(self.base_dir, split_name, "images")
            if os.path.exists(img_dir):
                image_files = [
                    f for f in os.listdir(img_dir)
                    if f.lower().endswith(self.image_extensions)
                ]
                self.all_images[split_name] = sorted(image_files)
            else:
                self.all_images[split_name] = []
    
    def parse_xml_annotation(self, xml_path: str, img_width: int, img_height: int) -> List[dict]:
        """解析XML标注文件"""
        annotations = []
        
        if not os.path.exists(xml_path):
            return annotations
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                name = obj.find('name').text
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                
                # 确保坐标在图片范围内
                xmin = max(0, min(img_width - 1, xmin))
                ymin = max(0, min(img_height - 1, ymin))
                xmax = max(0, min(img_width - 1, xmax))
                ymax = max(0, min(img_height - 1, ymax))
                
                annotations.append({
                    'class': int(name),
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax
                })
        except Exception as e:
            print(f"⚠️  解析XML失败 {xml_path}: {e}")
        
        return annotations
    
    def draw_annotations(self, img: np.ndarray, annotations: List[dict]) -> np.ndarray:
        """在图片上绘制标注框"""
        img_copy = img.copy()
        
        for ann in annotations:
            class_id = ann['class']
            xmin, ymin, xmax, ymax = ann['xmin'], ann['ymin'], ann['xmax'], ann['ymax']
            
            color = self.colors[class_id % len(self.colors)]
            
            # 绘制边界框
            cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), color, 2)
            
            # 准备标签文本
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
            label = f"{class_name}"
            
            # 计算文本大小
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 绘制标签背景
            label_y = max(ymin, text_height + 5)
            cv2.rectangle(img_copy, 
                         (xmin, label_y - text_height - 5),
                         (xmin + text_width, label_y + baseline),
                         color, -1)
            
            # 绘制标签文本
            cv2.putText(img_copy, label, (xmin, label_y), 
                       font, font_scale, (255, 255, 255), thickness)
        
        return img_copy
    
    def get_current_image_info(self) -> Optional[Tuple[str, str, str]]:
        """获取当前图片信息"""
        split_name = self.splits[self.current_split_idx]
        images = self.all_images[split_name]
        
        if len(images) == 0:
            return None
        
        if self.current_image_idx >= len(images):
            self.current_image_idx = len(images) - 1
        
        img_file = images[self.current_image_idx]
        return split_name, img_file, os.path.join(self.base_dir, split_name, "images", img_file)
    
    def visualize_current_image(self) -> Optional[np.ndarray]:
        """可视化当前图片"""
        info = self.get_current_image_info()
        if info is None:
            return None
        
        split_name, img_filename, img_path = info
        
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  无法读取图片: {img_path}")
            return None
        
        h, w = img.shape[:2]
        
        # 查找对应的标注文件
        base_name = os.path.splitext(img_filename)[0]
        ann_path = os.path.join(self.base_dir, split_name, "annotations", base_name + '.xml')
        
        # 解析标注
        annotations = self.parse_xml_annotation(ann_path, w, h)
        
        # 绘制标注
        img_with_annotations = self.draw_annotations(img, annotations)
        
        # 添加信息文本
        total_images = sum(len(imgs) for imgs in self.all_images.values())
        current_split_images = len(self.all_images[split_name])
        
        info_text = [
            f"Split: {split_name.upper()}",
            f"Image: {self.current_image_idx + 1}/{current_split_images}",
            f"Total: {self.current_split_idx + 1}/{len([s for s in self.splits if len(self.all_images[s]) > 0])} splits",
            f"File: {img_filename}",
            f"Size: {w}x{h}",
            f"Annotations: {len(annotations)}"
        ]
        
        # 在图片左上角绘制信息
        y_offset = 25
        for i, text in enumerate(info_text):
            cv2.putText(img_with_annotations, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加操作提示
        help_text = [
            "← → / A D : Previous/Next",
            "T : Switch Split (Train/Val/Test)",
            "F/L : First/Last",
            "S : Save",
            "I : Info",
            "Q/ESC : Quit"
        ]
        for i, text in enumerate(help_text):
            cv2.putText(img_with_annotations, text, 
                       (10, h - 150 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img_with_annotations
    
    def switch_split(self, direction: int = 1):
        """切换数据集划分"""
        available_splits = [i for i, s in enumerate(self.splits) 
                          if len(self.all_images[s]) > 0]
        
        if len(available_splits) == 0:
            return
        
        current_idx_in_available = available_splits.index(self.current_split_idx) if self.current_split_idx in available_splits else 0
        
        if direction > 0:
            # 下一个
            next_idx = (current_idx_in_available + 1) % len(available_splits)
        else:
            # 上一个
            next_idx = (current_idx_in_available - 1) % len(available_splits)
        
        self.current_split_idx = available_splits[next_idx]
        self.current_image_idx = 0
        split_name = self.splits[self.current_split_idx]
        print(f"📂 切换到: {split_name} ({len(self.all_images[split_name])} 张图片)")
    
    def show(self, window_name: str = "Dataset Split Visualizer", 
             start_split: str = "train", start_index: int = 0, scale: float = 0.5):
        """显示可视化窗口"""
        # 设置起始划分
        if start_split in self.splits:
            self.current_split_idx = self.splits.index(start_split)
        else:
            # 找到第一个有图片的划分
            for i, split_name in enumerate(self.splits):
                if len(self.all_images[split_name]) > 0:
                    self.current_split_idx = i
                    break
        
        self.current_image_idx = start_index
        
        # 统计信息
        total_images = sum(len(imgs) for imgs in self.all_images.values())
        available_splits = [s for s in self.splits if len(self.all_images[s]) > 0]
        
        if total_images == 0:
            print("❌ 没有找到图片文件")
            return
        
        print(f"📸 找到 {total_images} 张图片")
        for split_name in self.splits:
            count = len(self.all_images[split_name])
            if count > 0:
                print(f"   {split_name}: {count} 张")
        print(f"🎨 类别数量: {len(self.class_names)}")
        print(f"⌨️  操作说明:")
        print(f"   ← → / A D : 上一张/下一张")
        print(f"   T         : 切换数据集划分 (Train/Val/Test)")
        print(f"   F / L     : 第一张/最后一张")
        print(f"   S         : 保存当前图片（带标注）")
        print(f"   I         : 显示详细信息")
        print(f"   Q / ESC   : 退出")
        print("-" * 60)
        
        while True:
            # 可视化当前图片
            img = self.visualize_current_image()
            if img is None:
                break
            
            # 缩放图片
            if scale != 1.0:
                h, w = img.shape[:2]
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 显示图片
            cv2.imshow(window_name, img)
            
            # 等待键盘输入
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == 27:  # Q 或 ESC
                break
            elif key == 81 or key == 2 or key == ord('a') or key == ord('A'):  # 左箭头或A键
                split_name = self.splits[self.current_split_idx]
                self.current_image_idx = max(0, self.current_image_idx - 1)
                print(f"📄 上一张: {split_name}/{self.all_images[split_name][self.current_image_idx]}")
            elif key == 83 or key == 3 or key == ord('d') or key == ord('D'):  # 右箭头或D键
                split_name = self.splits[self.current_split_idx]
                images = self.all_images[split_name]
                self.current_image_idx = min(len(images) - 1, self.current_image_idx + 1)
                print(f"📄 下一张: {split_name}/{images[self.current_image_idx]}")
            elif key == ord('t') or key == ord('T'):  # T键：切换划分
                self.switch_split(1)
            elif key == ord('f') or key == ord('F'):  # F键：第一张
                self.current_image_idx = 0
            elif key == ord('l') or key == ord('L'):  # L键：最后一张
                split_name = self.splits[self.current_split_idx]
                self.current_image_idx = len(self.all_images[split_name]) - 1
            elif key == ord('s') or key == ord('S'):  # S键：保存
                self._save_current_image(img)
            elif key == ord('i') or key == ord('I'):  # I键：详细信息
                self._print_image_info()
        
        cv2.destroyAllWindows()
        print("✅ 可视化结束")
    
    def _save_current_image(self, img: np.ndarray):
        """保存当前图片（带标注）"""
        info = self.get_current_image_info()
        if info is None:
            return
        
        split_name, img_filename, _ = info
        base_name = os.path.splitext(img_filename)[0]
        output_filename = f"{split_name}_{base_name}_annotated.jpg"
        output_path = os.path.join(self.base_dir, "visualized", output_filename)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"💾 已保存: {output_path}")
    
    def _print_image_info(self):
        """打印当前图片的详细信息"""
        info = self.get_current_image_info()
        if info is None:
            return
        
        split_name, img_filename, img_path = info
        img = cv2.imread(img_path)
        if img is None:
            return
        
        h, w = img.shape[:2]
        base_name = os.path.splitext(img_filename)[0]
        ann_path = os.path.join(self.base_dir, split_name, "annotations", base_name + '.xml')
        annotations = self.parse_xml_annotation(ann_path, w, h)
        
        print(f"\n📊 图片信息:")
        print(f"  数据集: {split_name}")
        print(f"  文件名: {img_filename}")
        print(f"  尺寸: {w}x{h}")
        print(f"  标注数量: {len(annotations)}")
        
        # 统计各类别数量
        class_counts = {}
        for ann in annotations:
            class_id = ann['class']
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        if class_counts:
            print(f"  类别分布:")
            for class_id, count in sorted(class_counts.items()):
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
                print(f"    {class_name}: {count}")
        print()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量可视化数据集划分的标注')
    parser.add_argument('--base_dir', type=str, default='.',
                       help='数据集基础目录（默认: 当前目录）')
    parser.add_argument('--start_split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='起始数据集划分（默认: train）')
    parser.add_argument('--start', type=int, default=0,
                       help='起始图片索引（默认: 0）')
    parser.add_argument('--scale', type=float, default=0.5,
                       help='显示缩放比例（默认: 0.5）')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = SplitDatasetVisualizer(base_dir=args.base_dir)
    
    # 显示
    visualizer.show(start_split=args.start_split, 
                   start_index=args.start, 
                   scale=args.scale)


if __name__ == '__main__':
    main()

