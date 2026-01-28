import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import numpy as np

# ================= 설정 =================
MODEL_NAME = "microsoft/graphcodebert-base"
MODEL_PATH = "models/best_graphcodebert.pth"
DATA_DIR = "data"
SAVE_DIR = "features"
BATCH_SIZE = 32  # 추론(Inference) 때는 학습보다 메모리를 덜 먹으니 키워도 됩니다
MAX_LEN = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 클래스 정의 (학습 때와 동일) =================
class GraphCodeBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits

class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.codes = df['code'].fillna("").astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(self.codes[idx], truncation=True, padding='max_length', 
                             max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0)
        }

# ================= 추출 함수 =================
def extract_and_save(loader, df, filename, model):
    print(f"🚀 Extracting features for {filename}...")
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, mask)
            probs = torch.softmax(logits, dim=1) # 확률값으로 변환
            all_probs.extend(probs.cpu().numpy())
            
    # 데이터프레임으로 저장 (gcb_prob_0, gcb_prob_1 ... 형태)
    prob_cols = [f'gcb_prob_{i}' for i in range(len(all_probs[0]))]
    out_df = pd.DataFrame(all_probs, columns=prob_cols)
    
    # ID가 있으면 같이 저장 (나중에 합칠 때 안전장치)
    if 'id' in df.columns:
        out_df['id'] = df['id'].values
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    save_path = os.path.join(SAVE_DIR, f"gcb_{filename}")
    out_df.to_parquet(save_path)
    print(f"✅ Saved to {save_path}")

# ================= 메인 실행 =================
# 1. 모델 로드
print("📂 Loading Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = GraphCodeBERTClassifier(MODEL_NAME, num_classes=11).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# 2. 데이터 로드 함수
def load_data(fname):
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        # 테스트셋 이름이 다를 경우 대비
        path = os.path.join(DATA_DIR, "task_b_test_set_sample.parquet") if "test" in fname else path
    df = pd.read_parquet(path)
    if 'code' not in df.columns and 'text' in df.columns:
        df['code'] = df['text']
    return df

# Train / Val / Test 로드
train_df = load_data("task_b_training_set.parquet")
val_df = load_data("task_b_validation_set.parquet")
test_df = load_data("test.parquet") # 파일명 확인 필요

# 데이터셋 & 로더 생성
train_loader = DataLoader(CodeDataset(train_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
val_loader = DataLoader(CodeDataset(val_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(CodeDataset(test_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 3. 추출 실행
extract_and_save(train_loader, train_df, "train.parquet", model)
extract_and_save(val_loader, val_df, "val.parquet", model)
extract_and_save(test_loader, test_df, "test.parquet", model)