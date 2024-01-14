import torch
import torch.autograd as autograd
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
from tqdm import tqdm

CUDA = False
VOCAB_SIZE = 5000
MAX_SEQ_LEN = 20
START_LETTER = 0
BATCH_SIZE = 32
MLE_TRAIN_EPOCHS = 100
ADV_TRAIN_EPOCHS = 50
POS_NEG_SAMPLES = 10000

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

oracle_samples_path = './oracle_samples.trc'
oracle_state_dict_path = './oracle_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_gen_path = './gen_MLEtrain_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_dis_path = './dis_pretrain_EMBDIM_64_HIDDENDIM64_VOCAB5000_MAXSEQLEN20.trc'

class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, dropout=0.2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2*2*hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, input, hidden):
        # input dim                                                # batch_size x seq_len
        emb = self.embeddings(input)                               # batch_size x seq_len x embedding_dim
        emb = emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim
        _, hidden = self.gru(emb, hidden)                          # 4 x batch_size x hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
        out = self.gru2hidden(hidden.view(-1, 4*self.hidden_dim))  # batch_size x 4*hidden_dim
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)                                 # batch_size x 1
        out = torch.sigmoid(out)
        return out

    def batchClassify(self, inp):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """

        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return out.view(-1)

    def batchBCELoss(self, inp, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.

         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """

        loss_fn = nn.BCELoss()
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return loss_fn(out, target)

class TextClassifier(nn.Module):
    def __init__(self, discriminator: Discriminator):
        super(TextClassifier, self).__init__()
        # 复制鉴别器的网络结构
        self.hidden_dim = discriminator.hidden_dim
        self.embeddings = discriminator.embeddings
        self.gru = discriminator.gru
        self.gru2hidden = discriminator.gru2hidden
        self.dropout_linear = discriminator.dropout_linear

        # 替换最后一层为适用于14类分类任务的全连接层
        self.hidden2out = nn.Linear(discriminator.hidden_dim, 14)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim))
        if discriminator.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, input):
        batch_size = input.size(0)  # 获取当前批次的大小
        hidden = self.init_hidden(batch_size)  # 动态初始化隐藏状态
        emb = self.embeddings(input)
        emb = emb.permute(1, 0, 2)
        _, hidden = self.gru(emb, hidden)
        hidden = hidden.permute(1, 0, 2).contiguous()
        out = self.gru2hidden(hidden.view(-1, 4*self.hidden_dim))
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)  # 使用新的全连接层
        return torch.softmax(out, dim=1)


# 数据处理函数
def process_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    texts, labels = [], []
    for line in lines:
        text, label = line.strip().split('\t')
        texts.append(text)
        labels.append(label)
    return texts, labels

# 构建词汇表
def build_vocab(texts, vocab_size):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = ['<PAD>', '<UNK>'] + [word for word, _ in counter.most_common(vocab_size - 2)]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    return word_to_idx

# 文本编码
def encode_texts(texts, word_to_idx, max_seq_len):
    encoded_texts = []
    for text in texts:
        encoded_text = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in text.split()]
        padded_encoded_text = encoded_text[:max_seq_len] + [word_to_idx['<PAD>']] * (max_seq_len - len(encoded_text))
        encoded_texts.append(padded_encoded_text)
    return np.array(encoded_texts)

# 标签编码
def encode_labels(labels, label_to_idx):
    return np.array([label_to_idx[label] for label in labels])

# 定义数据集类
class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])

# 加载和处理数据
train_texts, train_labels = process_data('data//train.txt')
dev_texts, dev_labels = process_data('data//dev.txt')

# 构建词汇表和标签索引
vocab_size = 5000
max_seq_len = 20
word_to_idx = build_vocab(train_texts, vocab_size)
label_to_idx = {label: idx for idx, label in enumerate(['财经', '科技', '时政', '房产', '社会', '游戏', '家居', '时尚', '股票', '彩票', '娱乐', '教育', '星座', '体育'])}

# 编码文本和标签
encoded_train_texts = encode_texts(train_texts, word_to_idx, max_seq_len)
encoded_dev_texts = encode_texts(dev_texts, word_to_idx, max_seq_len)

encoded_train_labels = encode_labels(train_labels, label_to_idx)
encoded_dev_labels = encode_labels(dev_labels, label_to_idx)

batch_size = 32
train_dataset = NewsDataset(encoded_train_texts, encoded_train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataset = NewsDataset(encoded_dev_texts, encoded_dev_labels)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

gpu = False

discriminator = Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
discriminator.load_state_dict(torch.load(pretrained_dis_path))
text_classifier = TextClassifier(discriminator)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(text_classifier.parameters(), lr=0.0001)


# 训练循环
num_epochs = 10  # 根据需要调整epoch数量
for epoch in range(num_epochs):
    text_classifier.train()
    total_loss = 0
    total = 0
    correct = 0

    for texts, labels in tqdm(train_loader):
        if gpu:
            texts, labels = texts.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = text_classifier(texts)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total
    print(f'Epoch {epoch}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')

    # 模型验证
    text_classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in tqdm(dev_loader):
            if gpu:
                texts, labels = texts.cuda(), labels.cuda()
            outputs = text_classifier(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    dev_acc = 100 * correct / total
    print(f'Accuracy on dev set: {dev_acc:.2f}%')


