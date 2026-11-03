import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. 페이지 설정
st.set_page_config(
    page_title="모델 성능 높이기",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ 클래스 가중치 조정을 통한 모델 성능 높이기 실습실")
st.write("뇌졸중 데이터는 양성(1) 데이터가 매우 드물기 때문에 발생하는 불균형 문제를 해결하기 위해, 가중치 조정을 적용하기 전과 후의 결과를 비교하는 페이지입니다.")

# 2. 데이터 불러오기 ('채점표' 페이지와 동일)
DATA_URL = "https://raw.githubusercontent.com/TENDO-Sj/data/refs/heads/main/stroke.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df = load_data()

st.markdown("---")

# 3. 속성 선택 UI 설정
st.subheader("⚙️ 모델 입력 속성(특징) 선택")
st.write("입력으로 사용할 속성을 고르세요. (처음에는 나이, 평균 혈당, 고혈압, 심장병 넷이 선택되어 있습니다)")

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
    key="perf_feature_select"
)

if len(selected_korean_features) < 2:
    st.warning("⚠️ 속성을 2개 이상 선택해 주세요!")
    st.stop()

selected_features = [reverse_feature_map[k] for k in selected_korean_features]

st.markdown("---")

# 4. 데이터 전처리 및 훈련/테스트셋 분할
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

# 5. 모델 학습 (가중치 조정 전 vs 후)
# 1) 로지스틱 회귀
lr_before = LogisticRegression(random_state=42, max_iter=1000, class_weight=None)
lr_before.fit(X_train, y_train)
lr_preds_before = lr_before.predict(X_test)

lr_after = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_after.fit(X_train, y_train)
lr_preds_after = lr_after.predict(X_test)

# 2) 의사결정트리
dt_before = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42, class_weight=None)
dt_before.fit(X_train, y_train)
dt_preds_before = dt_before.predict(X_test)

dt_after = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42, class_weight='balanced')
dt_after.fit(X_train, y_train)
dt_preds_after = dt_after.predict(X_test)

# 혼동행렬 및 평가지표 계산 함수
def get_cm_metrics(y_true, preds):
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    acc = accuracy_score(y_true, preds)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return tn, fp, fn, tp, acc, recall, precision, f1

lr_b_tn, lr_b_fp, lr_b_fn, lr_b_tp, lr_b_acc, lr_b_rec, lr_b_prec, lr_b_f1 = get_cm_metrics(y_test, lr_preds_before)
lr_a_tn, lr_a_fp, lr_a_fn, lr_a_tp, lr_a_acc, lr_a_rec, lr_a_prec, lr_a_f1 = get_cm_metrics(y_test, lr_preds_after)

dt_b_tn, dt_b_fp, dt_b_fn, dt_b_tp, dt_b_acc, dt_b_rec, dt_b_prec, dt_b_f1 = get_cm_metrics(y_test, dt_preds_before)
dt_a_tn, dt_a_fp, dt_a_fn, dt_a_tp, dt_a_acc, dt_a_rec, dt_a_prec, dt_a_f1 = get_cm_metrics(y_test, dt_preds_after)

# 6. 결과 표 비교 출력
st.subheader("📊 클래스 가중치 조정 전·후 모델 성능 비교")
st.write("가중치 조정을 적용하기 전과 맞춘 후의 혼동행렬 및 평가지표를 나란히 비교합니다.")

def make_comparison_dataframe(b_tp, b_fp, b_fn, b_tn, b_acc, b_rec, b_prec, b_f1,
                              a_tp, a_fp, a_fn, a_tn, a_acc, a_rec, a_prec, a_f1):
    data = {
        "평가 항목": ["TN (참음성)", "FP (거짓양성)", "FN (거짓음성)", "TP (참양성)", "정확도 (Accuracy)", "재현율 (Recall)", "정밀도 (Precision)", "F1 스코어"],
        "가중치 조정 전 (Before)": [
            f"{b_tn:,}", f"{b_fp:,}", f"{b_fn:,}", f"{b_tp:,}",
            f"{b_acc:.4f}", f"{b_rec:.4f}", f"{b_prec:.4f}", f"{b_f1:.4f}"
        ],
        "가중치 조정 후 (After)": [
            f"{a_tn:,}", f"{a_fp:,}", f"{a_fn:,}", f"{a_tp:,}",
            f"{a_acc:.4f}", f"{a_rec:.4f}", f"{a_prec:.4f}", f"{a_f1:.4f}"
        ]
    }
    return pd.DataFrame(data)

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.markdown("**로지스틱 회귀(확률로 답하는 모델)**")
    lr_df_comp = make_comparison_dataframe(
        lr_b_tp, lr_b_fp, lr_b_fn, lr_b_tn, lr_b_acc, lr_b_rec, lr_b_prec, lr_b_f1,
        lr_a_tp, lr_a_fp, lr_a_fn, lr_a_tn, lr_a_acc, lr_a_rec, lr_a_prec, lr_a_f1
    )
    st.dataframe(lr_df_comp, hide_index=True, use_container_width=True)

with col_t2:
    st.markdown("**의사결정트리(질문으로 답하는 모델)**")
    dt_df_comp = make_comparison_dataframe(
        dt_b_tp, dt_b_fp, dt_b_fn, dt_b_tn, dt_b_acc, dt_b_rec, dt_b_prec, dt_b_f1,
        dt_a_tp, dt_a_fp, dt_a_fn, dt_a_tn, dt_a_acc, dt_a_rec, dt_a_prec, dt_a_f1
    )
    st.dataframe(dt_df_comp, hide_index=True, use_container_width=True)

st.markdown("---")

# 7. 변화 요약 줄글 작성 (로지스틱 회귀 기준) - 삼중 따옴표 오류 방지 분리 작성
st.subheader("📝 성능 변화 요약")
st.markdown(f"- **찾아낸 환자 수 (TP)**: {lr_b_tp}명에서 **{lr_a_tp}명**으로 바뀌었습니다.")
st.markdown(f"- **놓친 환자 수 (FN)**: {lr_b_fn}명에서 **{lr_a_fn}명**으로 바뀌었습니다.")
st.markdown(f"- **헛짚은 사람 수 (FP)**: {lr_b_fp}명에서 **{lr_a_fp}명**으로 바뀌었습니다.")

st.markdown("---")

# 8. 산점도 및 결정 경계 시각화 (2차원 또는 3차원)
st.subheader("📈 결정 경계 변화 시각화 (가중치 조정 전 vs 후)")

try:
    if len(selected_korean_features) == 2:
        st.write("선택한 2개의 속성을 가로축과 세로축으로 하는 2차원 산점도와 전후 결정 경계선을 그립니다.")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            x_axis_korean = st.selectbox("가로축 속성", options=selected_korean_features, index=0, key="perf_x_2d")
        with col_f2:
            y_axis_korean = st.selectbox("세로축 속성", options=selected_korean_features, index=1, key="perf_y_2d")
            
        x_col = reverse_feature_map[x_axis_korean]
        y_col = reverse_feature_map[y_axis_korean]
        
        df_test_vis = df_test.copy()
        df_test_vis["stroke_label"] = df_test_vis["stroke"].map({0: "아님(0)", 1: "뇌졸중(1)"})
        
        other_features = [f for f in selected_features if f not in [x_col, y_col]]
        fixed_values = {f: df_train[f].median() for f in other_features}
        
        fig = px.scatter(
            df_test_vis,
            x=x_col,
            y=y_col,
            color="stroke_label",
            labels={x_col: x_axis_korean, y_col: y_axis_korean, "stroke_label": "실제 뇌졸중 여부"},
            title="2차원 산점도 및 가중치 조정 전후 결정 경계 (로지스틱 회귀)"
        )
        
        x_min, x_max = df_test[x_col].min(), df_test[x_col].max()
        y_min, y_max = df_test[y_col].min(), df_test[y_col].max()
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        x_min_plot, x_max_plot = x_min - x_range * 0.1, x_max + x_range * 0.1
        y_min_plot, y_max_plot = y_min - y_range * 0.1, y_max + y_range * 0.1
        
        xx, yy = np.meshgrid(
            np.linspace(x_min_plot, x_max_plot, 100),
            np.linspace(y_min_plot, y_max_plot, 100)
        )
        
        grid_df = pd.DataFrame(index=range(xx.ravel().shape[0]))
        for feat in selected_features:
            if feat == x_col:
                grid_df[feat] = xx.ravel()
            elif feat == y_col:
                grid_df[feat] = yy.ravel()
            else:
                grid_df[feat] = fixed_values[feat]
                
        # Before 경계선 (파란색)
        probs_b = lr_before.predict_proba(grid_df)[:, 1].reshape(xx.shape)
        fig.add_trace(
            go.Contour(
                x=np.linspace(x_min_plot, x_max_plot, 100),
                y=np.linspace(y_min_plot, y_max_plot, 100),
                z=probs_b,
                contours=dict(coloring='lines', showlabels=False),
                line=dict(color='blue', width=2),
                name="조정 전 경계(0.5)",
                showscale=False
            )
        )
        
        # After 경계선 (주황색)
        probs_a = lr_after.predict_proba(grid_df)[:, 1].reshape(xx.shape)
        fig.add_trace(
            go.Contour(
                x=np.linspace(x_min_plot, x_max_plot, 100),
                y=np.linspace(y_min_plot, y_max_plot, 100),
                z=probs_a,
                contours=dict(coloring='lines', showlabels=False),
                line=dict(color='orange', width=2),
                name="조정 후 경계(0.5)",
                showscale=False
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        if (probs_b.min() > 0.5 or probs_b.max() < 0.5) and (probs_a.min() > 0.5 or probs_a.max() < 0.5):
            st.caption("ℹ️ 안내: 조정 전과 후의 0.5 경계선 모두 현재 그림 범위 밖에 있습니다.")
        elif probs_b.min() > 0.5 or probs_b.max() < 0.5:
            st.caption("ℹ️ 안내: 가중치 조정 전(파란색) 0.5 경계선이 현재 그림 범위 밖에 있습니다.")
        elif probs_a.min() > 0.5 or probs_a.max() < 0.5:
            st.caption("ℹ️ 안내: 가중치 조정 후(주황색) 0.5 경계선이 현재 그림 범위 밖에 있습니다.")
        else:
            st.caption("ℹ️ 파란색 실선은 가중치 조정 전(Before) 0.5 경계선, 주황색 실선은 가중치 조정 후(After) 0.5 경계선입니다.")

    else:
        st.write("선택한 속성이 3개 이상이므로, 3가지 축을 골라 3차원 산점도를 보여드립니다.")
        
        col_31, col_32, col_33 = st.columns(3)
        with col_31:
            x_axis_korean = st.selectbox("X축 속성", options=selected_korean_features, index=0, key="perf_x_3d")
        with col_32:
            y_axis_korean = st.selectbox("Y축 속성", options=selected_korean_features, index=1, key="perf_y_3d")
        with col_33:
            z_axis_korean = st.selectbox("Z축 속성", options=selected_korean_features, index=min(2, len(selected_korean_features)-1), key="perf_z_3d")
            
        x_col = reverse_feature_map[x_axis_korean]
        y_col = reverse_feature_map[y_axis_korean]
        z_col = reverse_feature_map[z_axis_korean]
        
        df_test_vis = df_test.copy()
        df_test_vis["stroke_label"] = df_test_vis["stroke"].map({0: "아님(0)", 1: "뇌졸중(1)"})
        
        fig = px.scatter_3d(
            df_test_vis,
            x=x_col,
            y=y_col,
            z=z_col,
            color="stroke_label",
            labels={x_col: x_axis_korean, y_col: y_axis_korean, z_col: z_axis_korean, "stroke_label": "실제 뇌졸중 여부"},
            title="3차원 산점도 (로지스틱 회귀 가중치 조정 전후 비교)"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("ℹ️ 선택한 3개의 속성 축에 따른 3차원 산점도입니다. 마우스로 회전하여 살펴보세요.")
except Exception as e:
    st.info("ℹ️ 시각화 렌더링 중 오류가 발생하여 표와 요약 결과만 우선 표시됩니다.")
