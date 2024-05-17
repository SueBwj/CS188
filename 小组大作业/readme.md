## Fruit Classification
该项目为使用CNN对131中水果和蔬菜进行分类，准确率为92%。dataset来源于fruite-360

#### CNN架构：
1. CONV2d
2. ReLU ：增加函数的表达能力
3. MaxPool2d : 对图片进行压缩，减小计算量
4. CONV2d
5. ReLU
6. MaxPool2d
7. CONV2d
8. ReLU
9. MaxPool2d
10. Flatten
11. Linear
12.  Relu
13.  Linear


#### 文件
1. FruitData.py -- 将数据载入，转化为符合DataLoader所需格式
2. NetModule.py -- 为神经网路架构
3. predict.py -- 载入数据并训练和评估模型
