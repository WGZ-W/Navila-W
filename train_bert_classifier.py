import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# ---------------------------- 配置参数 ----------------------------
class Config:
    model_name = 'bert-base-uncased'      # 预训练模型名称（当 local_model_path 为 None 时使用）
    local_model_path = './bert'  # 本地预训练模型目录，若为 None 则从 HuggingFace 在线下载
    max_seq_len = 32                      # 名词短语通常较短，可设小值
    batch_size = 16
    epochs = 5
    learning_rate = 2e-5
    warmup_ratio = 0.1                    # 预热步数比例
    weight_decay = 0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    data_path = './noun_phrases.csv'      # 数据文件路径，包含 text 和 label 列
    model_save_path = './bert_scene_object_classifier.pt'

cfg = Config()

# 固定随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(cfg.seed)

# ---------------------------- 自定义数据集 ----------------------------
class NounPhraseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ---------------------------- 模型定义（BERT + 两层MLP） ----------------------------
class BertSceneObjectClassifier(nn.Module):
    def __init__(self, bert_model_name, local_model_path=None, num_classes=2, dropout=0.1):
        super().__init__()
        # 优先使用本地模型路径
        if local_model_path is not None and os.path.exists(local_model_path):
            self.bert = BertModel.from_pretrained(local_model_path)
        else:
            self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        # 两层 MLP: 768 -> 256 -> 2
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取 [CLS]  token 的隐状态
        cls_embedding = outputs.last_hidden_state[:, 0, :]   # (batch, hidden)
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)
        return logits

# ---------------------------- 训练与评估函数 ----------------------------
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    prec, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    return avg_loss, acc, prec, recall, f1

# ---------------------------- 主程序 ----------------------------
def main():
    # 1. 加载数据（假设 CSV 文件包含 'text' 和 'label' 两列，label 为 0/1）
    if not os.path.exists(cfg.data_path):
        raise FileNotFoundError(f"数据文件 {cfg.data_path} 不存在，请准备数据。")
    df = pd.read_csv(cfg.data_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()   # 0: object, 1: scene   (可根据需要调换)

    print(f"加载数据：{len(texts)} 条样本，类别分布：\n{df['label'].value_counts()}")

    # 2. 划分训练集/验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=cfg.seed, stratify=labels
    )

    # 3. 初始化 tokenizer（优先本地）
    if cfg.local_model_path is not None and os.path.exists(cfg.local_model_path):
        tokenizer = BertTokenizer.from_pretrained(cfg.local_model_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(cfg.model_name)

    # 4. 创建 Dataset 和 DataLoader
    train_dataset = NounPhraseDataset(train_texts, train_labels, tokenizer, cfg.max_seq_len)
    val_dataset = NounPhraseDataset(val_texts, val_labels, tokenizer, cfg.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # 5. 初始化模型、优化器、学习率调度器、损失函数
    model = BertSceneObjectClassifier(cfg.model_name, cfg.local_model_path).to(cfg.device)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    criterion = nn.CrossEntropyLoss()

    # 6. 训练循环
    best_f1 = 0.0
    for epoch in range(1, cfg.epochs + 1):
        print(f"\n===== Epoch {epoch}/{cfg.epochs} =====")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, cfg.device)
        val_loss, val_acc, val_prec, val_recall, val_f1 = evaluate(model, val_loader, criterion, cfg.device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | P: {val_prec:.4f} R: {val_recall:.4f} F1: {val_f1:.4f}")

        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), cfg.model_save_path)
            print(f"模型已保存至 {cfg.model_save_path} (F1={best_f1:.4f})")

    print("\n训练完成。")

    # 7. 可选：在验证集上输出最终结果（加载最佳模型）
    model.load_state_dict(torch.load(cfg.model_save_path))
    final_loss, final_acc, final_prec, final_recall, final_f1 = evaluate(model, val_loader, criterion, cfg.device)
    print("\n最佳模型在验证集上的表现：")
    print(f"Loss: {final_loss:.4f}, Acc: {final_acc:.4f}, P: {final_prec:.4f}, R: {final_recall:.4f}, F1: {final_f1:.4f}")

# ---------------------------- 推理示例 ----------------------------
def predict_single_phrase(phrase, model, tokenizer, device, max_len=32):
    """对单个名词短语进行预测，返回类别名称和置信度"""
    model.eval()
    encoding = tokenizer(
        phrase,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = np.argmax(probs)
    label_map = {0: 'object-level', 1: 'scene-level'}
    return label_map[pred], probs[pred]

if __name__ == '__main__':
    main()

    # 推理测试（需要先训练好模型）
    # 示例：假设模型已存在
    # device = cfg.device
    # 优先使用本地 tokenizer
    # if cfg.local_model_path and os.path.exists(cfg.local_model_path):
    #     tokenizer = BertTokenizer.from_pretrained(cfg.local_model_path)
    # else:
    #     tokenizer = BertTokenizer.from_pretrained(cfg.model_name)
    # model = BertSceneObjectClassifier(cfg.model_name, cfg.local_model_path).to(device)
    # model.load_state_dict(torch.load(cfg.model_save_path, map_location=device))
    # phrase = "the wooden table"
    # label, conf = predict_single_phrase(phrase, model, tokenizer, device)
    # print(f"'{phrase}' -> {label} (confidence: {conf:.4f}")