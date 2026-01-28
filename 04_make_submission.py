import pandas as pd
import xgboost as xgb
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ================= 설정 =================
FEATURE_DIR = "features"
DATA_DIR = "data"
SUBMISSION_FILE = "submission.csv"

# 1. Feature 로드 함수 (UnixCoder + TF-IDF 병합)
def load_merged_features(split_name):
    # 파일 경로 설정
    unix_path = os.path.join(FEATURE_DIR, f"unix_{split_name}.parquet")
    tfidf_path = os.path.join(FEATURE_DIR, f"tfidf_{split_name}.parquet")
    
    # 파일 존재 여부 확인
    if not os.path.exists(unix_path):
        raise FileNotFoundError(f"❌ {unix_path} 파일이 없습니다. 01번 코드를 실행했나요?")
    if not os.path.exists(tfidf_path):
        raise FileNotFoundError(f"❌ {tfidf_path} 파일이 없습니다. 02번 코드를 실행했나요?")

    print(f"🔄 Merging features for {split_name}...")
    unix_df = pd.read_parquet(unix_path)
    tfidf_df = pd.read_parquet(tfidf_path)
    
    # 옆으로 합치기 (Concat)
    merged_df = pd.concat([unix_df, tfidf_df], axis=1)
    return merged_df

# ================= 메인 실행 로직 =================

# 1. 데이터 로드 (Train + Val을 모두 합쳐서 학습 데이터로 사용)
print("📂 Loading All Features...")
train_feat = load_merged_features("train")
val_feat = load_merged_features("val")
full_train_df = pd.concat([train_feat, val_feat], axis=0).reset_index(drop=True)

# Test 데이터 로드
test_df = load_merged_features("test")

# 학습에 사용할 Feature 컬럼 정의 (label 제외)
feature_cols = [c for c in full_train_df.columns if c != 'label']
print(f"📌 Training on {len(full_train_df)} samples with {len(feature_cols)} features.")

# 2. 라벨 인코딩 (문자열 -> 숫자)
# XGBoost는 숫자 라벨만 먹으므로 변환 필요
le = LabelEncoder()
y_train = le.fit_transform(full_train_df['label'])

# 3. XGBoost 전체 데이터 재학습
print("🚀 Retraining XGBoost on Full Data (Train + Val)...")
dtrain = xgb.DMatrix(full_train_df[feature_cols], label=y_train)
dtest = xgb.DMatrix(test_df[feature_cols])

params = {
    'objective': 'multi:softmax',
    'num_class': len(le.classes_),
    'max_depth': 6,          # 03번에서 검증된 파라미터
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'mlogloss',
    # [수정] CPU 스레드 설정 제거하고 GPU 설정 추가
    # 'nthread': -1,  <-- 제거 또는 주석 처리
    'device': 'cuda', # <-- 최신 XGBoost 버전일 경우
    'tree_method': 'hist', # <-- GPU 가속 히스토그램 방식
}

# Epoch 500회 (아까 로그 보니 100~500 사이에서 충분히 수렴함)
model = xgb.train(params, dtrain, num_boost_round=500, verbose_eval=50)

# 4. 예측 및 복원
print("🔮 Predicting on Test Set...")
preds = model.predict(dtest)
pred_labels = le.inverse_transform(preds.astype(int)) # 숫자(0,1,..) -> 원래 문자열(Mistral,..)

# 5. Submission CSV 생성
print("💾 Creating Submission File...")

# 원본 Test 파일에서 ID 가져오기
origin_test_path = os.path.join(DATA_DIR, "test.parquet")
if not os.path.exists(origin_test_path):
     # 혹시 파일명이 다를 경우를 대비해 sample 파일 체크
    origin_test_path = os.path.join(DATA_DIR, "task_b_test_set_sample.parquet")

origin_test = pd.read_parquet(origin_test_path)

# ID 컬럼 확인 (보통 'id'임)
id_col = 'id' if 'id' in origin_test.columns else origin_test.columns[0]
print(f"   - Using ID column: {id_col}")

submission = pd.DataFrame({
    id_col: origin_test[id_col],
    'label': pred_labels
})

submission.to_csv(SUBMISSION_FILE, index=False)
print(f"✅ Submission file saved: {SUBMISSION_FILE}")
print(submission.head())