import paddle
import paddle.io.dataloader
import paddle.vision.transforms as T
import numpy as np
import os
from PIL import Image
import paddle.nn.functional as F
import paddle.nn as nn

# 定义超参数
batch_size = 32
epoches = 50


# 拆分训练集
fn_train = "train.txt"
fn_valid = "valid.txt"
with open("train_list.txt", "rb") as f:
    all_data = f.readlines()
train_num = int(len(all_data) * 0.8)
train_data, valid_data = paddle.io.random_split(all_data, [train_num , len(all_data) - train_num])

with open(fn_train, "wb") as f:
    f.writelines(train_data)

with open(fn_valid, "wb") as f:
    f.writelines(valid_data)

class CatDataset(paddle.io.Dataset):
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
            self.transforms = T.Compose([
                T.RandomResizedCrop((96,96)),  # 随机裁剪
                T.ContrastTransform(0.4),
                T.RandomHorizontalFlip(0.5),  # 随机水平翻转
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像归一化
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(256),                 # 图像大小修改
                T.ContrastTransform(0.4),
                T.RandomCrop((96,96)),      # 随机裁剪
                T.ToTensor(),                  # 数据的格式转换和标准化 HWC => CHW
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # 图像归一化
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

train_dataset = CatDataset(mode="train")
train_loader = paddle.io.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=0)

val_dataset = CatDataset(mode="valid")
val_loader = paddle.io.DataLoader(val_dataset,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=0)

#generator如下
class Generator(nn.Layer):

    def __init__(self, nz=100):
        super().__init__()
        self.layer1=nn.Sequential(
            nn.Conv2DTranspose(nz,1024,[4,4],data_format="NCHW"),
            nn.BatchNorm2D(1024,0.9,1e-05),
            nn.ReLU()
        )
        self.layer2=nn.Sequential(
            nn.Conv2DTranspose(1024,512,[4,4],[2,2],1,data_format="NCHW"),
            nn.BatchNorm2D(512,0.9,1e-05),
            nn.ReLU()
        )
        self.layer3=nn.Sequential(
            nn.Conv2DTranspose(512,256,[4,4],[2,2],1,data_format="NCHW"),
            nn.BatchNorm2D(256,0.9,1e-05),
            nn.ReLU()
        )
        self.layer4=nn.Sequential(
            nn.Conv2DTranspose(256,128,[4,4],[2,2],1,data_format="NCHW"),
            nn.BatchNorm2D(128,0.9,1e-05),
            nn.ReLU()
        )
        self.layer5=nn.Sequential(
            nn.Conv2DTranspose(128,3,[5,5],[3,3],1,data_format="NCHW"),
            nn.Tanh()
        )
    def forward(self, x):
        y=self.layer1(x)
        y=self.layer2(y)
        y=self.layer3(y)
        y=self.layer4(y)
        return self.layer5(y)

#discriminator如下
class Discriminator(nn.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2D(3,64,[5,5],[3,3],1,data_format="NCHW"),
            nn.BatchNorm2D(64,0.9,1e-05),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer2=nn.Sequential(
            nn.Conv2D(64,128,[4,4],[2,2],1,data_format="NCHW"),
            nn.BatchNorm2D(128,0.9,1e-05),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer3=nn.Sequential(
            nn.Conv2D(128,256,[4,4],[2,2],1,data_format="NCHW"),
            nn.BatchNorm2D(256,0.9,1e-05),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer4=nn.Sequential(
            nn.Conv2D(256,512,[4,4],[2,2],1,data_format="NCHW"),
            nn.BatchNorm2D(512,0.9,1e-05),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer5=nn.Sequential(
            nn.Conv2D(512,1,[4,4],data_format="NCHW"),
            nn.Sigmoid()
        )

    def forward(self, x):
        y=self.layer1(x)
        y=self.layer2(y)
        y=self.layer3(y)
        y=self.layer4(y)
        return self.layer5(y)
    
class DCGAN:
    def __init__(self, lr=0.0002, batch_size=64, episodes=50):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.optim_for_gen = paddle.optimizer.Adam(lr, parameters=self.generator.parameters(), beta1=0.5)
        self.optim_for_dis = paddle.optimizer.Adam(lr, parameters=self.discriminator.parameters(), beta1=0.5)
        self.loss_for_gen = nn.BCELoss()
        self.loss_for_dis = nn.BCELoss()
        self.true_label = 1
        self.false_label = 0
        self.batch_size = batch_size
        self.episodes = episodes
        self.label = paddle.to_tensor([1]*self.batch_size)

    def train(self):
        for epoch in range(1, self.episodes+1):
            for i, img in enumerate(self.data_loader):
                # train discriminator
                # 使用真实数据集
                img = img[0]
                self.optim_for_dis.clear_grad()
                output = self.discriminator(img).squeeze()
                self.label = paddle.to_tensor([self.true_label]*img.shape[0], dtype='float32')
                err1 = self.loss_for_dis(output, self.label)
                err1.backward()

                # 使用生成数据集
                self.label = paddle.to_tensor([self.false_label]*img.shape[0], dtype='float32')
                noise = paddle.randn((img.shape[0], 100, 1, 1))
                fake = self.generator(noise)
                output = self.discriminator(fake.detach()).squeeze()
                err2 = self.loss_for_dis(output, self.label)
                err2.backward()
                err = err1+err2
                self.optim_for_dis.step()


                # train generator
                self.optim_for_gen.clear_grad()
                self.label = paddle.to_tensor([self.true_label]*img.shape[0], dtype='float32')
                output = self.discriminator(fake).squeeze()
                err3 = self.loss_for_gen(output, self.label)
                err3.backward()
                self.optim_for_gen.step()


                if i%50==0:
                    print('epoch/Epoch {}/{} iter/Iter {}/{} lossD{:.4f}, lossG{:.4f}'.format(epoch, self.episodes, i+1, len(self.data_loader), err.item(), err3.item()))
                
                if epoch%5 == 0:
                    paddle.save(self.generator.state_dict(), os.path.join('model', 'generator{}.pdmodel'.format(str(epoch).zfill(4))))
                    paddle.save(self.discriminator.state_dict(), os.path.join('model', 'discriminator{}.pdmodel'.format(str(epoch).zfill(4))))
            
            self.plot_and_save('result', 'result{}.jpg'.format(str(epoch).zfill(4)))
        
        if not os.path.exists('model'):
            os.mkdir('model')
        paddle.save(self.generator.state_dict(), os.path.join('model', 'generator.pdmodel'))
        paddle.save(self.discriminator.state_dict(), os.path.join('model', 'discriminator.pdmodel'))
    
    def test(self, noise):
        fake = self.generator(noise)
        return fake
        

    def lode_model(self, gen_path, dis_path):
        gen_path_dict = paddle.load(gen_path)
        self.generator.set_state_dict(gen_path_dict)
        dis_path_dict = paddle.load(dis_path)
        self.discriminator.set_state_dict(dis_path_dict)


paddle.device.set_device('cpu')
dcgan = DCGAN()
# 模型载入
dcgan.lode_model('model/generator0025.pdmodel', 'model/discriminator0025.pdmodel')

class Net(nn.Layer):
    def __init__(self, discriminator):
        super(Net, self).__init__()
        # 复制预训练模型的层（除了最后一层）
        self.layer1 = discriminator.layer1
        self.layer2 = discriminator.layer2
        self.layer3 = discriminator.layer3
        self.layer4 = discriminator.layer4
        # 替换最后一层为适用于新任务的全连接层
        self.fc = nn.Linear(512*4*4, 15)
        self.softmax = nn.Softmax(axis=1)  # 使用 Softmax 进行多分类
    
    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        # 全连接层
        y = y.reshape([y.shape[0], -1])  # 展平特征
        y = self.fc(y)
        return self.softmax(y)
    

def train(model, optimizer, train_loader, val_loader):
    # 定义交叉熵损失函数
    loss_function = paddle.nn.CrossEntropyLoss()

    for epoch in range(epoches):
        model.train()  # 设置模型为训练模式
        train_loss, train_correct = 0, 0

        for data, target in train_loader:
            optimizer.clear_grad()  # 清除梯度
            output = model(data)  # 前向传播
            loss = loss_function(output, target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            train_loss += loss.item()
            train_correct += (output.argmax(axis=1) == target).astype(paddle.float32).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / len(train_loader.dataset)

        model.eval()  # 设置模型为评估模式
        val_loss, val_correct = 0, 0
        with paddle.no_grad():  # 不跟踪梯度
            for data, target in val_loader:
                output = model(data)
                val_loss += loss_function(output, target).item()
                val_correct += (output.argmax(axis=1) == target).astype(paddle.float32).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / len(val_loader.dataset)

        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

mynet = Net(dcgan.discriminator)

# 选择要更新的参数，即最后的全连接层
# params_to_update = mynet.fc.parameters()

opt = paddle.optimizer.Adam(learning_rate=0.0001, parameters=mynet.parameters())

train(mynet,opt,train_loader,val_loader)
