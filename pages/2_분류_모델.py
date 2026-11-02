import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. 페이지 설정
st.set_page_config(
    page_title="뇌졸중 분류 모델 실습실",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 뇌졸중 예측 분류 모델 실습실")
st.write("로지스틱 회귀와 의사결정트리 모델을 활용하여 뇌졸중 발생 여부를 예측하고 모델의 성능과 규칙을 시각적으로 탐색하는 페이지입니다.")

# 2. 데이터 불러오기
DATA_URL = "https://raw.githubusercontent.com/TENDO-Sj/data/refs/heads/main/stroke.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df = load_data()

st.markdown("---")

# 3. 속성 선택 UI 설정
st.subheader("⚙️ 모델 입력 속성(특징) 선택")
st.write("입력으로 사용할 속성을 고르세요. (처음에는 체질량지수를 뺀 넷이 선택되어 있습니다)")

# 매핑 사전 정의
feature_name_map = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

# 역방향 매핑 (우리말 -> 원본 열 이름)
reverse_feature_map = {v: k for k, v in feature_name_map.items()}

default_korean_features = ["나이", "평균 혈당", "고혈압", "심장병"]
all_korean_features = ["나이", "평균 혈당", "체질량지수", "고혈압", "심장병"]

selected_korean_features = st.multiselect(
    "사용할 속성 선택 (2개 이상)",
    options=all_korean_features,
    default=default_korean_features
)

# 두 개보다 적게 고르면 안내문을 표시하고 멈춤
if len(selected_korean_features) < 2:
    st.warning("⚠️ 속성을 2개 이상 선택해 주세요!")
    st.stop()

# 선택된 우리말 속성을 원본 열 이름으로 변환
selected_features = [reverse_feature_map[k] for k in selected_korean_features]

st.markdown("---")

# 4. 데이터 전처리 및 훈련/테스트셋 분할 (조건 반영)
# 번호(id) 순으로 정렬
df_sorted = df.sort_values("id").reset_index(drop=True)

test_indices = []
train_indices = []

# 열 명씩 묶어 각 묶음의 앞 세 명을 테스트용으로 고정, 나머지 일곱 명은 훈련용
for i in range(0, len(df_sorted), 10):
    chunk = df_sorted.iloc[i:i+10]
    if len(chunk) >= 3:
        test_indices.extend(chunk.iloc[:3].index.tolist())
        train_indices.extend(chunk.iloc[3:].index.tolist())
    else:
        # 10명이 안 되는 마지막 조 처리 예외
        test_indices.extend(chunk.index.tolist())

df_train = df_sorted.loc[train_indices].copy()
df_test = df_sorted.loc[test_indices].copy()

# 체질량지수(bmi)는 비어 있는 사람이 있으니 고른 경우에만 훈련용의 중앙값으로 채움
if "bmi" in selected_features:
    bmi_median = df_train["bmi"].median()
    df_train["bmi"] = df_train["bmi"].fillna(bmi_median)
    df_test["bmi"] = df_test["bmi"].fillna(bmi_median)

# X, y 분리 (stroke가 1이면 양성/뇌졸중, 0이면 아님)
X_train = df_train[selected_features]
y_train = df_train["stroke"]

X_test = df_test[selected_features]
y_test = df_test["stroke"]

# 5. 모델 학습 및 평가
# 1) 로지스틱 회귀 모델 (확률로 답하는 모델)
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

lr_train_acc = accuracy_score(y_train, lr_model.predict(X_train))
lr_test_acc = accuracy_score(y_test, lr_model.predict(X_test))

# 2) 의사결정트리 모델 (질문으로 답하는 모델)
# 질문 세 번까지(max_depth=3), 마지막 마디에 사람이 다섯 명 미만이면 더 나누지 않음(min_samples_leaf=5), 난수 고정(random_state=42)
dt_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_model.fit(X_train, y_train)

dt_train_acc = accuracy_score(y_train, dt_model.predict(X_train))
dt_test_acc = accuracy_score(y_test, dt_model.predict(X_test))

# 3) 다수결 모델 ("입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델")
majority_class = y_train.mode()[0]
maj_train_preds = [majority_class] * len(y_train)
maj_test_preds = [majority_class] * len(y_test)
maj_train_acc = accuracy_score(y_train, maj_train_preds)
maj_test_acc = accuracy_score(y_test, maj_test_preds)

# 6. 정확도 큰 숫자 카드 표시
st.subheader("📊 모델별 정확도 비교")

col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.markdown(
        f"""
        <div style="padding: 15px; border-radius: 10px; background-color: #f0f2f6; text-align: center; border-top: 5px solid #1f77b4;">
            <p style="margin-bottom: 5px; font-weight: bold; color: #333;">로지스틱 회귀(확률로 답하는 모델)</p>
            <h1 style="color: #1f77b4; font-size: 38px; margin: 5px 0;">{lr_test_acc * 100:.1f}%</h1>
            <p style="color: #666; font-size: 13px; margin: 0;">훈련 정확도: {lr_train_acc * 100:.1f}% | 테스트 정확도: {lr_test_acc * 100:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_m2:
    st.markdown(
        f"""
        <div style="padding: 15px; border-radius: 10px; background-color: #f0f2f6; text-align: center; border-top: 5px solid #2ca02c;">
            <p style="margin-bottom: 5px; font-weight: bold; color: #333;">의사결정트리(질문으로 답하는 모델)</p>
            <h1 style="color: #2ca02c; font-size: 38px; margin: 5px 0;">{dt_test_acc * 100:.1f}%</h1>
            <p style="color: #666; font-size: 13px; margin: 0;">훈련 정확도: {dt_train_acc * 100:.1f}% | 테스트 정확도: {dt_test_acc * 100:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_m3:
    st.markdown(
        f"""
        <div style="padding: 15px; border-radius: 10px; background-color: #f0f2f6; text-align: center; border-top: 5px solid #7f7f7f;">
            <p style="margin-bottom: 5px; font-weight: bold; color: #333;">다수결 기준 모델 (기준선)</p>
            <h1 style="color: #7f7f7f; font-size: 38px; margin: 5px 0;">{maj_test_acc * 100:.1f}%</h1>
            <p style="color: #666; font-size: 13px; margin: 0;">훈련 정확도: {maj_train_acc * 100:.1f}% | 테스트 정확도: {maj_test_acc * 100:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# 7. 산점도 및 결정 경계 시각화
st.subheader("📈 2차원 산점도 및 결정 경계 시각화")
st.write("고른 속성 중 가로축과 세로축으로 삼을 두 가지를 선택해 주세요.")

col_f1, col_f2 = st.columns(2)
with col_f1:
    x_axis_korean = st.selectbox("가로축 속성", options=selected_korean_features, index=0)
with col_f2:
    y_axis_korean = st.selectbox("세로축 속성", options=selected_korean_features, index=min(1, len(selected_korean_features)-1))

x_col = reverse_feature_map[x_axis_korean]
y_col = reverse_feature_map[y_axis_korean]

# 테스트 데이터 복사본 생성 및 라벨링
df_test_vis = df_test.copy()
df_test_vis["stroke_label"] = df_test_vis["stroke"].map({0: "아님(0)", 1: "뇌졸중(1)"})

# 두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산
other_features = [f for f in selected_features if f not in [x_col, y_col]]
fixed_values = {}
fixed_info_texts = []

for feat in other_features:
    med_val = df_train[feat].median()
    fixed_values[feat] = med_val
    fixed_info_texts.append(f"**{feature_name_map[feat]}**: 훈련 데이터 중앙값({med_val:.2f})")

st.info(f"📌 **고정된 속성 값 안내**: 두 축 외의 속성들은 다음과 같이 고정하여 계산되었습니다: " + ", ".join(fixed_info_texts))

# 산점도 그리기 (Plotly)
fig = px.scatter(
    df_test_vis,
    x=x_col,
    y=y_col,
    color="stroke_label",
    labels={x_col: x_axis_korean, y_col: y_axis_korean, "stroke_label": "실제 뇌졸중 여부"},
    title="테스트 데이터 산점도 및 결정 경계"
)

# 로지스틱 회귀가 0.5로 가르는 선(결정 경계) 및 의사결정트리 영역 색칠 추가
# 그리드 생성
x_min, x_max = df_test[x_col].min(), df_test[x_col].max()
y_min, y_max = df_test[y_col].min(), df_test[y_col].max()

# 여유 공간 확보
x_range = x_max - x_min if x_max != x_min else 1
y_range = y_max - y_min if y_max != y_max else 1
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

# 1) 로지스틱 회귀 0.5 확률 등고선 (가르는 선) 추가
try:
    probs = lr_model.predict_proba(grid_df)[:, 1]
    zz_lr = probs.reshape(xx.shape)
    
    fig.add_contour(
        x=np.linspace(x_min_plot, x_max_plot, 100),
        y=np.linspace(y_min_plot, y_max_plot, 100),
        z=zz_lr,
        contours=dict(coloring='lines', showlabels=True, labelfont=dict(size=12, color='red')),
        line=dict(color='red', width=2),
        name="로지스틱 경계(확률 0.5)",
        showscale=False
    )
    
    # 선이 그림 범위 내에 있는지 확인
    if zz_lr.min() > 0.5 or zz_lr.max() < 0.5:
        st.caption("ℹ️ 참고: 로지스틱 회귀 모델이 0.5로 가르는 경계선이 현재 보고 있는 그림 범위 밖에 있습니다.")
    else:
        st.caption("ℹ️ 빨간색 실선은 로지스틱 회귀 모델이 확률 0.5로 예측하는 경계선입니다.")
except Exception as e:
    st.warning("로지스틱 경계선 그리기 중 오류가 발생했습니다.")

# 2) 의사결정트리가 나눈 칸 옅은 색 칠하기 (만약 선택된 두 축이 의사결정트리에 포함되어 있다면 영역 표시)
if x_col in selected_features and y_col in selected_features:
    zz_dt = dt_model.predict(grid_df).reshape(xx.shape)
    fig.add_contour(
        x=np.linspace(x_min_plot, x_max_plot, 100),
        y=np.linspace(y_min_plot, y_max_plot, 100),
        z=zz_dt,
        contours=dict(coloring='fill'),
        opacity=0.15,
        colorscale=[[0, 'blue'], [1, 'orange']],
        showscale=False,
        name="의사결정트리 영역"
    )

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 8. 의사결정트리 가지 그림 (Graphviz DOT 문자열 이용)
st.subheader("🌳 의사결정트리 질문 구조 (가지 그림)")
st.write("의사결정트리 모델이 학습한 질문과 마디별 데이터 분포입니다.")

# DOT 문자열 생성
n_nodes = dt_model.tree_.node_count
children_left = dt_model.tree_.children_left
children_right = dt_model.tree_.children_right
feature_indices = dt_model.tree_.feature
thresholds = dt_model.tree_.threshold
values = dt_model.tree_.value

dot_lines = ["digraph Tree {", "node [shape=box, style=\"filled, rounded\", fontname=\"sans-serif\"];"]

# 실제로 나무가 물은 속성들 수집
asked_features_set = set()

for i in range(n_nodes):
    val = values[i][0]
    total_samples = int(val.sum())
    stroke_samples = int(val[1]) if len(val) > 1 else 0
    ratio = (stroke_samples / total_samples * 100) if total_samples > 0 else 0
    
    if children_left[i] != children_right[i]: # 분기 마디 (질문 마디)
        feat_idx = feature_indices[i]
        orig_feat_name = selected_features[feat_idx]
        korean_feat_name = feature_name_map[orig_feat_name]
        asked_features_set.add(korean_feat_name)
        
.        thresh = thresholds[i]
        label_text = f"{korean_feat_name} <= {thresh:.2f}\\n인원: {total_samples명, 뇌졸중: {stroke_samples}명\\n비율: {ratio:.1f}%"
        dot_lines.append(f'  {i} [label="{label_text}", fillcolor="#e5f5f9"];')
    else: # 리프 마디 (답을 내는 마디)
        # 답에 따라 색을 달리 함 (뇌졸중 비율이 높으면 주황 계열, 낮으면 파랑 계열)
        fill_color = "#fee6ce" if ratio > 20 else "#deebf7"
        label_text = f"결과 답변\\n인원: {total_samples}명\\n뇌졸중: {stroke_samples}명\\n비율: {ratio:.1f}%"
        dot_lines.append(f'  {i} [label="{label_text}", fillcolor="{fill_color}"];')
    
    # 간선 연결
    if children_left[i] != children_right[i]:
        dot_lines.append(f'  {i} -> {children_left[i]} [label="예"];')
        dot_lines.append(f'  {i} -> {children_right[i]} [label="아니요"];')

dot_lines.append("}")
dot_string = "\n".join(dot_lines)

# st.graphviz_chart를 통해 그림 표시
st.graphviz_chart(dot_string)

# 9. 나무 해석 요약 한 줄씩 적기
st.markdown("### 📝 의사결정트리 분석 요약")

# 리프 마디 개수 및 아님(0)이라고 답하는 칸 개수 계산
leaf_nodes_count = 0
not_stroke_leaf_count = 0

for i in range(n_nodes):
    if children_left[i] == children_right[i]:
        leaf_nodes_count += 1
        val = values[i][0]
        # 다수파 클래스가 0(아님)인 경우
        if len(val) > 1 and val[0] >= val[1]:
            not_stroke_leaf_count.ى += 1
        elif len(val) == 1:
            not_stroke_leaf_count += 1

asked_features_list_str = ", ".join(list(asked_features_set)) if asked_features_set else "없음"

st.markdown(f"* **답을 내는 마디 개수**: 총 **{leaf_nodes_count}칸**이며, 그중 **{not_stroke_leaf_count}칸**이 뇌졸중 '아님'으로 답하고 있습니다.")
st.markdown(f"* **실제로 물은 속성**: 고른 속성 가운데 이 나무가 실제로 질문에 사용한 것은 **[{asked_features_list_str}]**입니다.")
