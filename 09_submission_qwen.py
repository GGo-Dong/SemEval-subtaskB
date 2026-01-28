import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import os
import numpy as np

# ================= 설정 =================
BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH = "models/qwen_qlora_final"
DATA_DIR = "data"
SAVE_DIR = "features"   # 여기에 저장됩니다
BATCH_SIZE = 8
MAX_LEN = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 데이터셋 =================
class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.texts = df['code'].fillna("").astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', 
                             max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0)
        }

# ================= 메인 실행 =================
print("📂 Loading Data...")
test_df = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
if 'code' not in test_df.columns: test_df['code'] = test_df['text']

print("🤖 Loading Qwen Model & Adapter...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
# [중요] 앙상블용 확률 추출 시에는 패딩 설정 유지
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_ID,
    num_labels=11,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
base_model.config.pad_token_id = tokenizer.pad_token_id

# 학습된 LoRA 어댑터 장착
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# ================= 확률 추출 및 저장 =================
print("🚀 Extracting Probabilities for Ensemble...")
test_loader = DataLoader(CodeDataset(test_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

all_probs = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        
        logits = model(input_ids, attention_mask=mask).logits
        probs = torch.softmax(logits, dim=1) # 확률값으로 변환
        all_probs.extend(probs.cpu().numpy().astype(np.float16)) # 용량 절약

# 저장
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

prob_cols = [f'qwen_prob_{i}' for i in range(len(all_probs[0]))]
out_df = pd.DataFrame(all_probs, columns=prob_cols)

# 나중에 순서 섞임 방지를 위해 ID도 함께 저장
id_col = 'id' if 'id' in test_df.columns else test_df.columns[0]
out_df['id'] = test_df[id_col].values

save_path = os.path.join(SAVE_DIR, "qwen_test.parquet")
out_df.to_parquet(save_path)

print(f"✅ Saved Features to: {save_path}")
print("👉 Now you can run '10_final_ensemble.py'!")