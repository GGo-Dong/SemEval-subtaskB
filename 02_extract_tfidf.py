import pandas as pd
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# ================= 설정 =================
DATA_DIR = "data"
SAVE_DIR = "features"

# [수정] 확인된 실제 'generator' 이름들을 정확히 리스트로 정의
MISTRAL_LIST = [
    'mistralai/Devstral-Small-2505',
    'mistralai/Mistral-7B-Instruct-v0.3'
]

GRANITE_LIST = [
    'ibm-granite/granite-3.3-8b-instruct',
    'ibm-granite/granite-3.2-2b-instruct',
    'ibm-granite/granite-3.3-8b-base'
]

# 두 리스트를 합쳐서 "관심 대상" 정의
TARGET_GENERATORS = MISTRAL_LIST + GRANITE_LIST

def load_data(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 파일이 없습니다: {path}")
    df = pd.read_parquet(path)
    # 컬럼 이름 방어 로직
    if 'code' not in df.columns and 'text' in df.columns:
        df['code'] = df['text']
    return df

# 1. 데이터 로드
print("📂 Loading Data...")
train_df = load_data("task_b_training_set.parquet")
val_df = load_data("task_b_validation_set.parquet")

# 2. Specialist 학습 데이터 필터링
print("🛠️ Training TF-IDF Specialist...")
print(f"   - Target Mistral models: {len(MISTRAL_LIST)} types")
print(f"   - Target Granite models: {len(GRANITE_LIST)} types")

# generator 컬럼이 타겟 리스트에 포함된 행만 추출
mask = train_df['generator'].isin(TARGET_GENERATORS)
spec_train = train_df[mask].copy()

print(f"   - Selected {len(spec_train)} samples for specialist training.")

if len(spec_train) == 0:
    print("❌ 오류: 데이터를 찾지 못했습니다! generator 컬럼 내용을 다시 확인하세요.")
    exit()

# 3. 라벨 생성 (Granite=0, Mistral=1)
# generator 이름이 MISTRAL_LIST에 있으면 1, 아니면(GRANITE면) 0
y_train = spec_train['generator'].apply(lambda x: 1 if x in MISTRAL_LIST else 0)

# 4. 파이프라인 학습 (TF-IDF + Random Forest)
pipeline = Pipeline([
    # max_features를 늘려서 더 미세한 특징까지 잡도록 함
    ('tfidf', TfidfVectorizer(max_features=5000, token_pattern=r'\b\w+\b')),
    ('clf', RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42))
])

pipeline.fit(spec_train['code'].fillna(""), y_train)
print("✅ Training Complete!")

# 5. 전체 데이터에 대해 확률 추출 및 저장 함수
def save_tfidf_probs(df, filename):
    print(f"💉 Injecting features for {filename}...")
    texts = df['code'].fillna("")
    
    # 확률 예측 (Class 0: Granite, Class 1: Mistral)
    probs = pipeline.predict_proba(texts)
    
    out_df = pd.DataFrame()
    out_df['tfidf_prob_granit'] = probs[:, 0]
    out_df['tfidf_prob_mistral'] = probs[:, 1]
    
    # 저장
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    save_path = os.path.join(SAVE_DIR, f"tfidf_{filename}")
    out_df.to_parquet(save_path)
    print(f"✅ Saved to {save_path}")

# 실행: Train / Val / Test
save_tfidf_probs(train_df, "train.parquet")
save_tfidf_probs(val_df, "val.parquet")

if os.path.exists(os.path.join(DATA_DIR, "test.parquet")):
    test_df = load_data("test.parquet")
    save_tfidf_probs(test_df, "test.parquet")