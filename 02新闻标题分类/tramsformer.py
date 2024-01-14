import torch
from torch.utils.data import Dataset, DataLoader
# Load model directly
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

class NewsTitleDataset(Dataset):
    def __init__(self, filename, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sentences = []
        self.labels = []

        # 加载数据并进行预处理
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                sentence, label = line.strip().split('\t')
                self.sentences.append(sentence)
                self.labels.append(label)

        # 将标签转换为数字
        self.label_dict = {'财经': 0, '科技': 1, '时政': 2, '房产': 3, '社会': 4, '游戏': 5, '家居': 6, '时尚': 7, '股票': 8, '彩票': 9, '娱乐': 10, '教育': 11, '星座': 12, '体育': 13}
        self.labels = [self.label_dict[label] for label in self.labels]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

train_dataset = NewsTitleDataset('data\\train.txt', tokenizer, max_len=256)
dev_dataset = NewsTitleDataset('data\\dev.txt', tokenizer, max_len=256)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

class BertNewsClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertNewsClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.drop = nn.Dropout(p=0.3)
        # 修正为 self.bert.config.hidden_size
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes) 

    def forward(self, input_ids, attention_mask):
        # 直接获取pooler_output
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)


model = BertNewsClassifier(n_classes=14)  # 14个类别

def train_and_evaluate(model, train_loader, dev_loader, loss_fn, optimizer, device, n_epochs):
    best_accuracy = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        
        for batch_idx, d in enumerate(tqdm(train_loader)):
            input_ids = d["ids"].to(device)
            attention_mask = d["mask"].to(device)
            labels = d["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_correct += torch.sum(preds == labels)

        train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct.double() / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{n_epochs} - Training loss: {train_loss}, accuracy: {train_accuracy}")

        model.eval()
        dev_loss = 0
        dev_correct = 0

        with torch.no_grad():
            for batch_idx, d in enumerate(tqdm(dev_loader)):
                input_ids = d["ids"].to(device)
                attention_mask = d["mask"].to(device)
                labels = d["labels"].to(device)

                outputs = model(input_ids, attention_mask)
                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, labels)

                dev_loss += loss.item()
                dev_correct += torch.sum(preds == labels)

        dev_loss = dev_loss / len(dev_loader)
        dev_accuracy = dev_correct.double() / len(dev_loader.dataset)
        print(f"Epoch {epoch + 1}/{n_epochs} - Validation loss: {dev_loss}, accuracy: {dev_accuracy}")

        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            torch.save(model.state_dict(), 'best_model_state.bin')


device = torch.device( "cpu")

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)


train_and_evaluate(model, train_loader, dev_loader, loss_fn, optimizer, device, n_epochs=3)
