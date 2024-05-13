from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from FruitData import FruitData
from NetModule import *
from torch.optim import SGD

# 自定义transform,将图像转化为tensor形式
dataset_transform = transforms.Compose([transforms.ToTensor()])
# 将tensor转化为PIL
to_pil = transforms.ToPILImage()


# 构造训练数据
train_classes = os.listdir('./dataset/Training')
train_dir = './dataset/Training'
training_datasets = None
for label_dir in train_classes:
    training_dataset = FruitData(
        root_dir=train_dir, fruit_class_dir=label_dir, transforms=dataset_transform)
    if training_datasets == None:
        training_datasets = training_dataset
    else:
        training_datasets += training_dataset

# 构造测试数据
test_classes = os.listdir('./dataset/Test')
test_datasets = None
test_dir = './dataset/Test'
for label_dir in test_classes:
    test_dataset = FruitData(
        root_dir=test_dir, fruit_class_dir=label_dir, transforms=dataset_transform)
    if test_datasets == None:
        test_datasets = test_dataset
    else:
        test_datasets += test_dataset


print("训练数据集的长度为:{}".format(len(training_datasets)))  # ctrl+d复制当前行
print("测试数据集的长度为:{}".format(len(test_datasets)))


# 加载训练数据集
train_loader = DataLoader(dataset=training_datasets, batch_size=64,
                          shuffle=True, num_workers=0, drop_last=False)

# 加载测试数据集
test_loader = DataLoader(dataset=test_datasets, batch_size=64,
                         shuffle=True, num_workers=0, drop_last=False)

# 创建网络模型
PredictModule = FruieClassifier()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01
optimizer = SGD(PredictModule.parameters(), lr=learning_rate)


# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 100  # 10

for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i + 1))

    # 训练步骤开始
    PredictModule.train()
    for data in train_loader:
        imgs, targets = data
        outputs = PredictModule(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        # 梯度清零，因为上次循环的梯度数据对这次循环没有用
        optimizer.zero_grad()
        # 反向传播，求出每个结点的梯度
        loss.backward()
        # 对模型参数调优
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            # .item的作用是把tensor数据类型转为纯数字
            print("训练次数:{}, Loss:{}".format(total_train_step, loss.item()))

    # 测试步骤开始
