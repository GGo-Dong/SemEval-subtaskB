import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# [수정 1] AdamW를 transformers가 아닌 torch.optim에서 가져옴
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import f1_score

# ================= 설정 =================
MODEL_NAME = "microsoft/graphcodebert-base"
SAVE_PATH = "models/best_graphcodebert.pth"
DATA_DIR = "data"
# GraphCodeBERT는 메모리를 많이 먹으므로 OOM(Out of Memory) 에러가 나면 8로 줄이세요
BATCH_SIZE = 16   
EPOCHS = 4
LR = 2e-5
MAX_LEN = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Device: {device}")

# 모델 저장 폴더가 없으면 생성
if not os.path.exists("models"):
    os.makedirs("models")

# ================= 데이터셋 =================
class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.codes = df['code'].fillna("").astype(str).tolist()
        self.labels = df['label'].values
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(self.codes[idx], truncation=True, padding='max_length', 
                             max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ================= 모델 =================
class GraphCodeBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] 토큰 사용 (index 0)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# ================= 실행 로직 =================
# 1. 데이터 로드
print("📂 Loading Data...")
train_df = pd.read_parquet(os.path.join(DATA_DIR, "task_b_training_set.parquet"))
val_df = pd.read_parquet(os.path.join(DATA_DIR, "task_b_validation_set.parquet"))

# 컬럼명 방어 로직
if 'code' not in train_df.columns and 'text' in train_df.columns:
    train_df['code'] = train_df['text']
    val_df['code'] = val_df['text']

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_dataset = CodeDataset(train_df, tokenizer, MAX_LEN)
val_dataset = CodeDataset(val_df, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# 2. 모델 초기화
# 클래스 개수 11개 확인 (데이터셋에 따라 다를 수 있음)
model = GraphCodeBERTClassifier(MODEL_NAME, num_classes=11).to(device)
optimizer = AdamW(model.parameters(), lr=LR)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*EPOCHS)
criterion = nn.CrossEntropyLoss()

# 3. 학습 루프
best_f1 = 0.0

print("🚀 Start Training GraphCodeBERT...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    
    # tqdm 진행바
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
    # 검증
    model.eval()
    preds_list = []
    labels_list = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, mask)
            preds = torch.argmax(logits, dim=1)
            
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            
    val_f1 = f1_score(labels_list, preds_list, average='macro')
    avg_loss = train_loss / len(train_loader)
    
    print(f"\nEpoch {epoch+1} Result:")
    print(f"  - Train Loss: {avg_loss:.4f}")
    print(f"  - Val F1:     {val_f1:.4f}")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  💾 Best Model Saved! ({SAVE_PATH})")
    print("-" * 50)

print(f"🏆 Final Best Val F1: {best_f1}")