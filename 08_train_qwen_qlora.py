import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    BitsAndBytesConfig,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# ================= 설정 =================
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct" 
DATA_DIR = "data"
OUTPUT_DIR = "qwen_qlora_checkpoints"

# 하이퍼파라미터
MAX_LEN = 512
BATCH_SIZE = 4
GRAD_ACCUM = 4
EPOCHS = 1        
LR = 2e-4

# 이어하기를 위한 체크포인트 경로
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint-1000")

# ================= 데이터 로드 및 정제 =================
print("📂 Loading Data...")
full_train_df = pd.read_parquet(os.path.join(DATA_DIR, "task_b_training_set.parquet"))

# 컬럼명 통일
if 'code' not in full_train_df.columns: full_train_df['code'] = full_train_df['text']

# [정제] 125글자 미만 제거 (노이즈 제거)
print("🧹 Cleaning Data: Removing codes shorter than 125 chars...")
clean_mask = (full_train_df['code'].fillna("").str.strip().str.len() >= 125)
train_df_clean = full_train_df[clean_mask]

# [학습 데이터 샘플링] 10만 개
if len(train_df_clean) > 100000:
    train_df = train_df_clean.sample(n=100000, random_state=42).reset_index(drop=True)
else:
    train_df = train_df_clean.reset_index(drop=True)

print(f"📉 Final Training Data Size: {len(train_df)} samples")

# [검증 데이터 로드 및 축소] - 여기가 수정된 핵심 부분!
print("📉 Loading Validation Data...")
full_val_df = pd.read_parquet(os.path.join(DATA_DIR, "task_b_validation_set.parquet"))

# [속도 향상] 검증 데이터 2만 개로 샘플링 (2시간 -> 15분 단축)
val_df = full_val_df.sample(n=20000, random_state=42).reset_index(drop=True)

if 'code' not in val_df.columns: val_df['code'] = val_df['text']
print(f"✅ Validation Data Downsampled: {len(full_val_df)} -> {len(val_df)} (Speed Up!)")

# ================= 토크나이저 & 데이터셋 =================
print("⚙️ Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# HuggingFace Dataset 변환
from datasets import Dataset as HFDataset

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=MAX_LEN)

train_ds = HFDataset.from_pandas(train_df[['code', 'label']].rename(columns={'code': 'text'}))
val_ds = HFDataset.from_pandas(val_df[['code', 'label']].rename(columns={'code': 'text'}))

train_tokenized = train_ds.map(preprocess_function, batched=True)
val_tokenized = val_ds.map(preprocess_function, batched=True)

# ================= 모델 준비 (QLoRA) =================
print("🤖 Loading Model with 4-bit Quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=11,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ================= 학습 설정 =================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro")
    }

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    
    num_train_epochs=EPOCHS,     
    
    weight_decay=0.01,
    eval_strategy="steps",       
    
    eval_steps=1000,             
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    logging_steps=50,
    report_to="none",
    dataloader_num_workers=4
)

# ================= 트레이너 실행 (Resume Logic 포함) =================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

# 체크포인트 존재 여부 확인 후 이어하기
if os.path.exists(CHECKPOINT_PATH):
    print(f"🚀 Resuming Training from {CHECKPOINT_PATH}...")
    trainer.train(resume_from_checkpoint=CHECKPOINT_PATH)
else:
    print("🚀 Start Training Qwen-7B (QLoRA) from Scratch...")
    trainer.train()

print("💾 Saving Final LoRA Adapter...")
trainer.save_model("models/qwen_qlora_final")