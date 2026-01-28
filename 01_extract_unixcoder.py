import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

# ================= 설정 =================
MODEL_PATH = "models/best_model_epoch_4.pth"  # 사용자가 가진 pth 파일 경로
BASE_MODEL = "microsoft/unixcoder-base"
DATA_DIR = "data"
SAVE_DIR = "features"
BATCH_SIZE = 128
NUM_CLASSES = 11  # 데이터셋 클래스 개수

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Current Device: {device}")
os.makedirs(SAVE_DIR, exist_ok=True)

# ================= 모델 정의 (기존과 동일해야 함) =================
class UniXcoderClassifier(nn.Module):
    def __init__(self, base_model_name, num_classes):
        super().__init__()
        self.unixcoder = AutoModel.from_pretrained(base_model_name)
        self.classifier = nn.Linear(self.unixcoder.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3) # 학습때 썼던 드롭아웃 (Inference엔 영향 X)
        self.layer_norm = nn.LayerNorm(self.unixcoder.config.hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.unixcoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.layer_norm(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# ================= 데이터셋 정의 =================
class SimpleCodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.codes = df['code'].fillna("").astype(str).tolist() # 컬럼명 확인 필요
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(self.codes[idx], truncation=True, padding='max_length', 
                             max_length=self.max_len, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0), 
                'attention_mask': enc['attention_mask'].squeeze(0)}

# ================= 추출 함수 =================
def extract_probs(df, model, tokenizer, filename):
    dataset = SimpleCodeDataset(df, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    all_probs = []
    model.eval()
    
    print(f"🔄 Extracting from {filename}...")
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, mask)
            probs = torch.softmax(logits, dim=1) # Logits -> Probabilities 변환
            all_probs.append(probs.cpu().numpy())
            
    # 결과를 DataFrame으로 저장
    import numpy as np
    all_probs = np.concatenate(all_probs, axis=0)
    cols = [f"unix_prob_{i}" for i in range(NUM_CLASSES)]
    prob_df = pd.DataFrame(all_probs, columns=cols)
    
    # 원본 라벨이 있다면 붙여주기 (학습용)
    if 'label' in df.columns:
        prob_df['label'] = df['label'].values
        
    save_path = os.path.join(SAVE_DIR, f"unix_{filename}")
    prob_df.to_parquet(save_path)
    print(f"✅ Saved to {save_path}")

# [01_extract_unixcoder.py 의 맨 마지막 부분 수정]

# ... (위쪽 함수 정의 등은 그대로 두세요) ...

# ================= 실행 =================
# 1. 모델 로드
print("Loading Model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = UniXcoderClassifier(BASE_MODEL, NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# 2. 데이터 로드
# (Train, Val은 이미 features 폴더에 있으니 주석 처리해서 건너뜁니다!)
# train_df = pd.read_parquet(os.path.join(DATA_DIR, "task_b_training_set.parquet"))
# val_df = pd.read_parquet(os.path.join(DATA_DIR, "task_b_validation_set.parquet"))

# extract_probs(train_df, model, tokenizer, "train.parquet")
# extract_probs(val_df, model, tokenizer, "val.parquet")

# [중요] 누락되었던 Test 셋만 추출 실행!
print("🚀 Processing Test Set...")
test_path = os.path.join(DATA_DIR, "test.parquet")

if os.path.exists(test_path):
    test_df = pd.read_parquet(test_path)
    extract_probs(test_df, model, tokenizer, "test.parquet") # -> features/unix_test.parquet 생성됨
else:
    print(f"❌ '{test_path}' 파일이 없습니다. 경로를 확인하세요.")