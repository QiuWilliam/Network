import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from torch.utils.data import random_split
import timm

# 定义超参数
batch_size = 32
num_epochs = 50
learning_rate = 0.001

# 拆分训练集
fn_train = "train.txt"
fn_valid = "valid.txt"
with open("train_list.txt", "rb") as f:
    all_data = f.readlines()
train_num = int(len(all_data) * 0.7)
train_data, valid_data = random_split(all_data, [train_num , len(all_data) - train_num])

with open(fn_train, "wb") as f:
    f.writelines(train_data)

with open(fn_valid, "wb") as f:
    f.writelines(valid_data)

class CatDataset(Dataset):
    """ 
    猫咪分类数据集定义
    图像增强方法可扩展
    """
    def __init__(self, mode="train"):
        self.data = []
        if mode in ("train", "valid"):
            with open("{}.txt".format(mode)) as f:
                for line in f.readlines():
                    info = line.strip().split("\t")
                    if len(info) == 2:
                        # [[path, label], [path, label]]
                        self.data.append([info[0].strip(), info[1].strip()])
        else:
            base_file_path = "cat_12_test"
            files = os.listdir(base_file_path)
            for info in files:
                file_path = os.path.join(base_file_path, info)
                self.data.append([os.path.join("cat_12_test", info), -1])
        
        if mode == "train":
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __getitem__(self, index):
        """
        根据索引获取样本
        return: 图像（rgb）, 所属分类
        """
        image_file, lable = self.data[index]
        image = Image.open(image_file)
        
        # 图像格式转化
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # 图像增强
        image = self.transforms(image)
        return image, np.array(lable, dtype="int64")

    def __len__(self):
        """获取样本总数"""
        return len(self.data)
    
class CustomViT(nn.Module):
    def __init__(self, num_classes=12, pretrained=True, model_name='vit_base_patch16_224'):
        super(CustomViT, self).__init__()
        # 加载预训练的ViT模型
        self.vit = timm.create_model(model_name, pretrained=pretrained)

        # 替换最后的分类层以适应我们的分类任务（12种猫）
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

train_dataset = CatDataset(mode="train")
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=0)

val_dataset = CatDataset(mode="valid")
val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=0)

# 模型初始化
model = CustomViT(num_classes=12, pretrained=True)

device = torch.device("cpu")
print(f"Using device: {device}")

model.to(device)

# 加载预训练的权重
model_path = 't_best_model.pth'
model_weights = torch.load(model_path, map_location=device)

# 将权重应用到模型上
model.load_state_dict(model_weights)


# 训练过程
for epoch in range(num_epochs):

    # 验证过程
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in val_loader:

            images, labels = images.to(device), labels.to(device)  # 移动数据到 GPU

            outputs = model(images)
            _, predicted = torch.max(outputs, axis=1)  # 获取预测的类别
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)  # 注意这里应该是 labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.4f}")

