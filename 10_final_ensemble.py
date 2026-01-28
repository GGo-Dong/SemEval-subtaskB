import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# ================= 설정 =================
FEATURE_DIR = "features"
DATA_DIR = "data"
SUBMISSION_FILE = "submission_final_ensemble.csv"

# ================= 데이터 로드 =================
print("📂 Loading Probabilities...")

# 1. UniXcoder (0.36~0.38 예상)
unix_df = pd.read_parquet(os.path.join(FEATURE_DIR, "unix_test.parquet"))
unix_probs = unix_df[[c for c in unix_df.columns if 'unix_prob' in c]].values

# 2. GraphCodeBERT (0.38 예상)
gcb_df = pd.read_parquet(os.path.join(FEATURE_DIR, "gcb_test.parquet"))
gcb_probs = gcb_df[[c for c in gcb_df.columns if 'gcb_prob' in c]].values

# 3. Qwen-7B (0.33 - 하지만 관점이 다름)
qwen_df = pd.read_parquet(os.path.join(FEATURE_DIR, "qwen_test.parquet"))
qwen_probs = qwen_df[[c for c in qwen_df.columns if 'qwen_prob' in c]].values

print(f"Shapes -> Unix: {unix_probs.shape}, GCB: {gcb_probs.shape}, Qwen: {qwen_probs.shape}")

# ================= 앙상블 전략 (Weighted Voting) =================
print("⚖️ Calculating Weighted Average...")

# Qwen 점수가 낮게 나왔으므로 가중치를 조정합니다.
# 독단적으로 결정하지 못하게 하고, 셋의 의견을 골고루 듣습니다.
w_unix = 0.4
w_gcb = 0.4
w_qwen = 0.2  # 점수가 낮으니 비중을 줄여서 '보조 조언자' 역할로 씁니다.

final_probs = (unix_probs * w_unix) + (gcb_probs * w_gcb) + (qwen_probs * w_qwen)

# ================= 제출 파일 생성 =================
print("🔄 Generating Submission...")

# 라벨 인코더 복원
train_df = pd.read_parquet(os.path.join(DATA_DIR, "task_b_training_set.parquet"))
le = LabelEncoder()
le.fit(train_df['label'])

# 확률 -> 클래스
final_preds_idx = np.argmax(final_probs, axis=1)
pred_labels = le.inverse_transform(final_preds_idx)

# 저장
test_origin = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
id_col = 'id' if 'id' in test_origin.columns else test_origin.columns[0]

submission = pd.DataFrame({
    id_col: test_origin[id_col],
    'label': pred_labels
})

submission.to_csv(SUBMISSION_FILE, index=False)
print(f"✅ Final Submission Saved: {SUBMISSION_FILE}")