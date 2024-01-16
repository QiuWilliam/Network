import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from datasets import load_metric
from datasets import load_metric
import numpy as np

# from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

from transformers import BartTokenizer, BartForConditionalGeneration

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# 定义了数据文件的路径
train_chinese_path = 'data\\train_dev_test\\casia2015_ch_train.txt'
train_english_path = 'data\\train_dev_test\\casia2015_en_train.txt'
dev_chinese_path = 'data\\train_dev_test\\casia2015_ch_dev.txt'
dev_english_path = 'data\\train_dev_test\\casia2015_en_dev.txt'


# 加载训练数据
def load_data(chinese_path, english_path):
    with open(chinese_path, 'r', encoding='utf-8') as chinese_file:
        chinese_data = [line.strip() for line in chinese_file.readlines()]
    with open(english_path, 'r', encoding='utf-8') as english_file:
        english_data = [line.strip() for line in english_file.readlines()]

    # 确保中文数据和英文数据的行数相同
    assert len(chinese_data) == len(english_data), "Data lines must be equal."

    return chinese_data, english_data


train_chinese_data, train_english_data = load_data(train_chinese_path, train_english_path)
dev_chinese_data, dev_english_data = load_data(dev_chinese_path, dev_english_path)


class TranslationDataset(Dataset):
    def __init__(self, chinese_data, english_data, tokenizer, max_length=128):
        self.chinese_data = chinese_data
        self.english_data = english_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.chinese_data)

    def __getitem__(self, idx):
        chinese = self.chinese_data[idx]
        english = self.english_data[idx]

        source_encoding = self.tokenizer(
            chinese,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        target_encoding = self.tokenizer(
            english,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return source_encoding.input_ids[0], target_encoding.input_ids[0]



train_dataset = TranslationDataset(train_chinese_data, train_english_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

dev_dataset = TranslationDataset(dev_chinese_data, dev_english_data, tokenizer)
dev_loader = DataLoader(dev_dataset, batch_size=16)


optimizer = AdamW(model.parameters(), lr=5e-5)
device = torch.device("cpu")
model.to(device)

# 加载预训练的权重
model_path = 'best_model.pth'
model_weights = torch.load(model_path, map_location=device)

# 将权重应用到模型上
model.load_state_dict(model_weights)

# 初始化BLEU指标
bleu_metric = load_metric("sacrebleu")


def validate(model, dataloader, tokenizer):
    model.eval()
    total_bleu_score = 0

    for batch in dataloader:
        input_ids, labels = batch
        outputs = model.generate(input_ids)

        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        references = [[ref] for ref in tokenizer.batch_decode(labels, skip_special_tokens=True)]

        bleu_score = bleu_metric.compute(predictions=predictions, references=references)
        total_bleu_score += bleu_score["score"]

    average_bleu_score = total_bleu_score / len(dataloader)
    return average_bleu_score

epochs = 3
for epoch in range(1, epochs + 1):

    avg_bleu_score = validate(model, dev_loader, tokenizer)
    print(f"Epoch [{epoch}/{epochs}]  Average BLEU Score: {avg_bleu_score:.2f}")
