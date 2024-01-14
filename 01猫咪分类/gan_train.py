import paddle.vision
from paddle.io import DataLoader
import paddle.optimizer
import paddle.vision.transforms as transforms
import matplotlib.pyplot as plt
import paddle
import paddle.nn as nn
import numpy as np
import paddle.vision.transforms as T
import os


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
    
dis_net = Discriminator()
# paddle.summary(dis_net, (batchsize, 3, 96, 96))
print(dis_net)

gen_net = Generator()
# paddle.summary(gen_net, (batchsize, 100, 1, 1))
print(gen_net)

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

    def load_data(self, root = 'cat_12_train'):
        transform = T.Compose([
                T.RandomResizedCrop((96,96)),  # 随机裁剪
                T.ContrastTransform(0.4),
                T.RandomHorizontalFlip(0.5),  # 随机水平翻转
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像归一化
            ])
        dataset = paddle.vision.datasets.ImageFolder(root, transform=transform)
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

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
        
    def plot_and_save(self, path:str, name:str):
        if not os.path.exists(path):
            os.mkdir(path)
        noise = paddle.randn((64, 100, 1, 1))
        t = transforms.Transpose(order=(1, 2, 0))
        to_plot = self.generator(noise).numpy()
        plt.figure(figsize=(8, 8))
        for i, img in enumerate(to_plot):
            img = t(img)
            plt.subplot(8, 8, i+1)
            plt.imshow(img)
            plt.axis('off')
        plt.savefig(os.path.join(path, name))

    def lode_model(self, gen_path, dis_path):
        gen_path_dict = paddle.load(gen_path)
        self.generator.set_state_dict(gen_path_dict)
        dis_path_dict = paddle.load(dis_path)
        self.discriminator.set_state_dict(dis_path_dict)


if __name__ == '__main__':
    paddle.device.set_device('cpu')
    dcgan = DCGAN()
    dcgan.load_data()
    dcgan.lode_model('model/generator0020.pdmodel', 'model/discriminator0020.pdmodel')
    dcgan.train()