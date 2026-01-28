import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 설정 =================
FEATURE_DIR = "features"
DATA_DIR = "data" # 원본 라벨 확인용

# 1. Feature 파일들 로드 및 병합
def load_features(split_name):
    # UnixCoder Features
    unix_df = pd.read_parquet(os.path.join(FEATURE_DIR, f"unix_{split_name}.parquet"))
    # TF-IDF Features
    tfidf_df = pd.read_parquet(os.path.join(FEATURE_DIR, f"tfidf_{split_name}.parquet"))
    
    # 옆으로 합치기 (Concat)
    # unix_df에는 이미 'label'이 포함되어 있음
    merged_df = pd.concat([unix_df, tfidf_df], axis=1)
    return merged_df

print("🔄 Merging Features...")
train_df = load_features("train")
val_df = load_features("val")

# 학습에 사용할 Feature 컬럼 정의 (label 제외)
feature_cols = [c for c in train_df.columns if c != 'label']
print(f"📌 Using {len(feature_cols)} features.")

# 2. XGBoost 학습
print("🚀 Training XGBoost Meta-Learner...")

# 라벨이 문자열이라면 숫자로 변환 필요 (이미 숫자인 경우 생략 가능)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(train_df['label'])
y_val = le.transform(val_df['label'])

dtrain = xgb.DMatrix(train_df[feature_cols], label=y_train)
dval = xgb.DMatrix(val_df[feature_cols], label=y_val)

params = {
    'objective': 'multi:softmax',
    'num_class': len(le.classes_),
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'mlogloss'
}

model = xgb.train(
    params, 
    dtrain, 
    num_boost_round=500, 
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=50,
    verbose_eval=50
)

print("\n📊 Evaluating...")
preds = model.predict(dval)
f1 = f1_score(y_val, preds, average='macro')
print(f"🏆 Final Macro F1 Score: {f1:.4f}")

# [수정] target_names를 문자열로 변환 (에러 해결 핵심!)
target_names_str = le.classes_.astype(str)

print(classification_report(y_val, preds, target_names=target_names_str))

# Feature Importance 시각화 (이게 진짜 중요합니다!)
plt.figure(figsize=(10, 8)) # 그림 크기 키움
xgb.plot_importance(model, max_num_features=20, height=0.5)
plt.title("Feature Importance (TF-IDF vs UniXcoder)")
plt.tight_layout()
plt.savefig("xgb_feature_importance.png")
print("🖼️ Saved feature importance plot: xgb_feature_importance.png")