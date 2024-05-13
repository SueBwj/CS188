import sys
import os
parent_dir = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))  # 获取父文件夹路径
sys.path.append(parent_dir)  # 添加父文件夹路径
from FruitData import FruitData
from PIL import Image
import unittest


class TestFruitDataClass(unittest.TestCase):

    def testGetitem(self):
        root_dir = r'C:\Users\王洁\Desktop\cs188\AI_lesson\小组大作业\dataset\Test'
        fruit_class_dir = r'C:\Users\王洁\Desktop\cs188\AI_lesson\小组大作业\dataset\Test\Apple Braeburn'
        apple_dataset = FruitData(root_dir, fruit_class_dir)
        test_img, test_path = apple_dataset[0]
        print(test_path)
        path = r'C:\Users\王洁\Desktop\cs188\AI_lesson\小组大作业\dataset\Test\Apple Braeburn'
        self.assertEqual(test_path, path)


if __name__ == "__main__":
    unittest.main()
