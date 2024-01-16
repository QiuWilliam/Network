import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import jieba
import numpy as np
import torch.optim as optim
from tqdm import tqdm

# 假设这是你的数据集路径
train_file = 'data//train.txt'
dev_file = 'data//dev.txt'

# 数据预处理
def load_data(file):
    texts, labels = [], []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            text, label = line.strip().split('\t')
            texts.append(text)
            labels.append(label)
    return texts, labels

def preprocess(texts):
    # 中文分词
    texts = [' '.join(jieba.cut(text)) for text in texts]
    return texts

# 构建词汇表
def build_vocab(texts):
    vocab = set()
    for text in texts:
        vocab.update(text.split())
    word_to_idx = {word: i+1 for i, word in enumerate(vocab)}
    word_to_idx['<PAD>'] = 0
    return word_to_idx

# 文本编码
def encode_texts(texts, word_to_idx, max_length):
    encoded = np.zeros((len(texts), max_length), dtype=int)
    for i, text in enumerate(texts):
        for j, word in enumerate(text.split()):
            if j >= max_length:
                break
            encoded[i, j] = word_to_idx.get(word, 0)
    return encoded

Y_dict = {
    0: '财经',
    1: '彩票',
    2: '房产',
    3: '股票',
    4: '家居',
    5: '教育',
    6: '科技',
    7: '社会',
    8: '时尚',
    9: '时政',
    10: '体育',
    11: '星座',
    12: '游戏',
    13: '娱乐'
}

index_to_label = Y_dict
label_to_index = {v: k for k, v in index_to_label.items()}

# 修改数据集类以使用标签索引
class TextDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx, max_length=100):
        self.texts = encode_texts(texts, word_to_idx, max_length)
        self.labels = [label_to_index[label] for label in labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

# CNN模型
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes, num_filters):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1,num_filters, (k, embed_dim)) for k in filter_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # 添加一个维度，表示通道数
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# 训练函数
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for texts, labels in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 验证函数
def evaluate(model, dev_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for texts, labels in tqdm(dev_loader):
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dev_loader)

# 主函数
def main():
    # 加载和预处理数据
    train_texts, train_labels = load_data(train_file)
    dev_texts, dev_labels = load_data(dev_file)
    train_texts = preprocess(train_texts)
    dev_texts = preprocess(dev_texts)

    # 构建词汇表和数据集
    word_to_idx = build_vocab(train_texts + dev_texts)
    train_dataset = TextDataset(train_texts, train_labels, word_to_idx)
    dev_dataset = TextDataset(dev_texts, dev_labels, word_to_idx)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

    # 创建模型实例
    vocab_size = len(word_to_idx)
    embed_dim = 128  # 嵌入维度
    num_classes = 14  # 类别数
    filter_sizes = [3, 4, 5]  # 卷积核尺寸
    num_filters = 100  # 卷积核数量
    model = TextCNN(vocab_size, embed_dim, num_classes, filter_sizes, num_filters)

    # 训练模型 
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练和验证模型
    num_epochs = 10  # 迭代次数
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        dev_loss = evaluate(model, dev_loader, criterion)
        print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {dev_loss:.4f}')


# 运行主函数
if __name__ == '__main__':
    main()
