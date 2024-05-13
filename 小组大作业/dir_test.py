import os
from FruitData import FruitData
from torchvision import transforms
fruit_classes = os.listdir('./dataset/Training')
mytransform = transforms.Compose([transforms.ToTensor()])
print(len(fruit_classes))

root_dir = './dataset/Training'
training_datasets = None

for label_dir in fruit_classes:
    training_dataset = FruitData(
        root_dir=root_dir, fruit_class_dir=label_dir, transforms=mytransform)
    if training_datasets == None:
        training_datasets = training_dataset
    else:
        training_datasets += training_dataset

print(len(training_datasets))
