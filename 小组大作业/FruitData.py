from torch.utils.data import Dataset
import os
import torch
from PIL import Image

# 定义自己的数据集
# 获取水果分类:将目标标签改为整数索引
fruit_classes = os.listdir('./dataset/Training')
fruit_classes_dict = {fruit_label: idx for idx,
                      fruit_label in enumerate(fruit_classes)}


class FruitData(Dataset):

    def __init__(self, root_dir, fruit_class_dir, transforms):
        """
        构造函数
        :param root_dir:数据集根目录(相对路径)
        :param fruit_class_dir:水果类别目录(相对路径)
        """
        self.root_dir = root_dir
        self.fruit_class_dir = fruit_class_dir
        self.path = os.path.join(
            self.root_dir, self.fruit_class_dir)
        self.img_path = os.listdir(self.path)
        self.transforms = transforms
        self.fruit_classes_dict = fruit_classes_dict

    def __getitem__(self, index):
        """
        获取特定水果目录下的图片
        """
        img_name = self.img_path[index]
        img_item_path = os.path.join(
            self.root_dir, self.fruit_class_dir, img_name)
        img = Image.open(img_item_path)
        img = self.transforms(img)
        fruit_label_tensor = self.fruit_classes_dict[self.fruit_class_dir]
        return img, fruit_label_tensor

    def __len__(self):
        """
        返回有多少个水果类别
        """
        return len(self.img_path)
