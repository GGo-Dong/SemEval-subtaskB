import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# ================= 설정 =================
FEATURE_DIR = "features"
DATA_DIR = "data"
SUBMISSION_FILE = "submission_ensemble.csv"

# 1. 데이터 로드 (Test Set 기준)
print("📂 Loading Probabilities...")

# UniXcoder 확률 로드
unix_test = pd.read_parquet(os.path.join(FEATURE_DIR, "unix_test.parquet"))
unix_probs = unix_test[[c for c in unix_test.columns if 'unix_prob' in c]].values

# GraphCodeBERT 확률 로드
gcb_test = pd.read_parquet(os.path.join(FEATURE_DIR, "gcb_test.parquet"))
gcb_probs = gcb_test[[c for c in gcb_test.columns if 'gcb_prob' in c]].values

print(f"   - UniXcoder Shape: {unix_probs.shape}")
print(f"   - GraphCodeBERT Shape: {gcb_probs.shape}")

# 2. 앙상블 (Soft Voting: 단순 평균)
print("⚖️ Averaging Probabilities...")
final_probs = (unix_probs + gcb_probs) / 2

# 3. 최종 클래스 결정
final_preds_idx = np.argmax(final_probs, axis=1)

# 4. 라벨 복원 (숫자 -> 문자열)
# 라벨 인코더를 다시 fit해서 순서를 맞춰야 합니다.
print("🔄 Restoring Labels...")
train_df = pd.read_parquet(os.path.join(DATA_DIR, "task_b_training_set.parquet"))
le = LabelEncoder()
le.fit(train_df['label']) # 학습 데이터의 라벨 순서대로 인코딩

pred_labels = le.inverse_transform(final_preds_idx)

# 5. 제출 파일 생성
print("💾 Saving Submission...")
# 원본 Test 파일에서 ID 가져오기
test_origin = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
id_col = 'id' if 'id' in test_origin.columns else test_origin.columns[0]

submission = pd.DataFrame({
    id_col: test_origin[id_col],
    'label': pred_labels
})

submission.to_csv(SUBMISSION_FILE, index=False)
print(f"✅ Submission Saved: {SUBMISSION_FILE}")
print(submission.head())