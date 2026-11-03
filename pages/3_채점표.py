import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. 페이지 설정
st.set_page_config(
    page_title="뇌졸중 모델 채점표",
    page_icon="📋",
    layout="wide"
)

st.title("📋 뇌졸중 예측 모델 채점표 및 오류 분석실")
st.write("분류 모델의 예측 결과를 혼동행렬과 평가지표(정확도, 재현율, 정밀도, F1 스코어)로 자세히 채점하고, 잘못 예측한 환자 목록을 분석하는 페이지입니다.")

# 2. 데이터 불러오기 ('분류 모델' 페이지와 동일)
DATA_URL = "[https://raw.githubusercontent.com/TENDO-Sj/data/refs/heads/main/stroke.csv](https://raw.githubusercontent.com/TENDO-Sj/data/refs/heads/main/stroke.csv)"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df = load_data()

st.markdown("---")

# 3. 속성 선택 UI 설정
st.subheader("⚙️ 모델 입력 속성(특징) 선택")
st.write("입력으로 사용할 속성을 고르세요. (처음에는 체질량지수를 뺀 넷이 선택되어 있습니다)")

feature_name_map = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

reverse_feature_map = {v: k for k, v in feature_name_map.items()}

default_korean_features = ["나이", "평균 혈당", "고혈압", "심장병"]
all_korean_features = ["나이", "평균 혈당", "체질량지수", "고혈압", "심장병"]

selected_korean_features = st.multiselect(
    "사용할 속성 선택 (2개 이상)",
    options=all_korean_features,
    default=default_korean_features,
    key="score_feature_select"
)

if len(selected_korean_features) < 2:
    st.warning("⚠️ 속성을 2개 이상 선택해 주세요!")
    st.stop()

selected_features = [reverse_feature_map[k] for k in selected_korean_features]

st.markdown("---")

# 4. 데이터 전처리 및 훈련/테스트셋 분할 (동일 조건)
df_sorted = df.sort_values("id").reset_index(drop=True)

test_indices = []
train_indices = []

for i in range(0, len(df_sorted), 10):
    chunk = df_sorted.iloc[i:i+10]
    if len(chunk) >= 3:
        test_indices.extend(chunk.iloc[:3].index.tolist())
        train_indices.extend(chunk.iloc[3:].index.tolist())
    else:
        test_indices.extend(chunk.index.tolist())

df_train = df_sorted.loc[train_indices].copy()
df_test = df_sorted.loc[test_indices].copy()

# 체질량지수(bmi) 결측치 처리 (고른 경우에만 훈련용 중앙값)
if "bmi" in selected_features:
    bmi_median = df_train["bmi"].median()
    df_train["bmi"] = df_train["bmi"].fillna(bmi_median)
    df_test["bmi"] = df_test["bmi"].fillna(bmi_median)

X_train = df_train[selected_features]
y_train = df_train["stroke"]

X_test = df_test[selected_features]
y_test = df_test["stroke"]

test_total_count = len(y_test)
test_stroke_count = int((y_test == 1).sum())

st.info(f"📊 **테스트 데이터 현황**: 테스트용 사람 수는 **{test_total_count:,}명**, 그중 실제 뇌졸중인 사람은 **{test_stroke_count:,}명**입니다.")

# 5. 모델 학습 및 예측
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

dt_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

majority_class = y_train.mode()[0]
maj_preds = np.array([majority_class] * len(y_test))

# 6. 혼동행렬 카운트 함수
def get_cm_counts(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    return tn, fp, fn, tp

lr_tn, lr_fp, lr_fn, lr_tp = get_cm_counts(y_test, lr_preds)
dt_tn, dt_fp, dt_fn, dt_tp = get_cm_counts(y_test, dt_preds)
maj_tn, maj_fp, maj_fn, maj_tp = get_cm_counts(y_test, maj_preds)

st.markdown("---")

# 7. 2×2 혼동행렬 표 출력
st.subheader("🔲 모델별 혼동행렬 (Confusion Matrix)")
st.write("행은 **실제 값**, 열은 **예측 값**을 나타냅니다. (양성 = 뇌졸중 1)")

col_cm1, col_cm2, col_cm3 = st.columns(3)

def make_cm_dataframe(tp, fp, fn, tn):
    df_cm = pd.DataFrame(
        [
            [f"{tn:,} (TN)", f"{fp:,} (FP)", f"{tn + fp:,}"],
            [f"{fn:,} (FN)", f"{tp:,} (TP)", f"{fn + tp:,}"],
            [f"{tn + fn:,}", f"{fp + tp:,}", f"{tn + fp + fn + tp:,}"]
        ],
        index=["실제: 아님(0)", "실제: 뇌졸중(1)", "합계"],
        columns=["예측: 아님(0)", "예측: 뇌졸중(1)", "합계"]
    )
    return df_cm

with col_cm1:
    st.markdown("**로지스틱 회귀(확률로 답하는 모델)**")
    st.table(make_cm_dataframe(lr_tp, lr_fp, lr_fn, lr_tn))

with col_cm2:
    st.markdown("**의사결정트리(질문으로 답하는 모델)**")
    st.table(make_cm_dataframe(dt_tp, dt_fp, dt_fn, dt_tn))

with col_cm3:
    st.markdown("**다수결 기준 모델**")
    st.table(make_cm_dataframe(maj_tp, maj_fp, maj_fn, maj_tn))

st.markdown("---")

# 8. 평가지표 계산 및 비교표
st.subheader("📈 모델별 성능 평가지표 비교")
st.write("정확도, 재현율, 정밀도, F1 스코어를 한눈에 비교합니다.")

def calculate_metrics(tp, fp, fn, tn, y_true, preds):
    acc = accuracy_score(y_true, preds)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
            f1_str = f"{f1:.4f}"
        else:
            f1_str = "0.0000"
        precision_str = f"{precision:.4f}"
    else:
        precision_str = "계산할 수 없음"
        f1_str = "계산할 수 없음"
        
    return acc, recall, precision_str, f1_str

lr_acc, lr_rec, lr_prec_str, lr_f1_str = calculate_metrics(lr_tp, lr_fp, lr_fn, lr_tn, y_test, lr_preds)
dt_acc, dt_rec, dt_prec_str, dt_f1_str = calculate_metrics(dt_tp, dt_fp, dt_fn, dt_tn, y_test, dt_preds)
maj_acc, maj_rec, maj_prec_str, maj_f1_str = calculate_metrics(maj_tp, maj_fp, maj_fn, maj_tn, y_test, maj_preds)

metrics_data = [
    {
        "모델 이름": "로지스틱 회귀(확률로 답하는 모델)",
        "정확도": f"{lr_acc:.4f}",
        "재현율": f"{lr_rec:.4f}",
        "정밀도": lr_prec_str,
        "F1 스코어": lr_f1_str
    },
    {
        "모델 이름": "의사결정트리(질문으로 답하는 모델)",
        "정확도": f"{dt_acc:.4f}",
        "재현율": f"{dt_rec:.4f}",
        "정밀도": dt_prec_str,
        "F1 스코어": dt_f1_str
    },
    {
        "모델 이름": "훈련용에서 많은 쪽으로만 답하는 모델",
        "정확도": f"{maj_acc:.4f}",
        "재현율": f"{maj_rec:.4f}",
        "정밀도": maj_prec_str,
        "F1 스코어": maj_f1_str
    }
]

st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)

st.markdown("---")

# 9. 오류 예측 환자 상세 분석
st.subheader("🔍 오류 예측 환자 상세 분석")
st.write("어떤 모델의 오류 목록을 확인할 것인지 선택해 주세요.")

model_choice = st.selectbox(
    "모델 선택",
    options=["로지스틱 회귀(확률로 답하는 모델)", "의사결정트리(질문으로 답하는 모델)",
