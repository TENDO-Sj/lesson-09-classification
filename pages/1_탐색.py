import streamlit as st
import pandas as pd
import plotly.express as px

# 1. 페이지 설정
st.set_page_config(
    page_title="뇌졸중 데이터 탐색",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 뇌졸중 데이터 탐색 실습실")
st.write("뇌졸중 데이터셋을 바탕으로 다양한 변수 간의 관계와 분포를 탐색하는 페이지입니다.")

# 2. 데이터 불러오기 (첫 화면과 같은 주소, UTF-8 인코딩)
DATA_URL = "https://raw.githubusercontent.com/TENDO-Sj/data/refs/heads/main/stroke.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df = load_data()

st.markdown("---")

# 3. 나이와 평균 혈당의 분포 히스토그램 두 개 나란히 배치
st.subheader("📊 나이와 평균 혈당의 분포")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**[나이(age) 분포]**")
    fig_age = px.histogram(df, x="age", nbins=30, labels={"age": "나이"}, color_discrete_sequence=["skyblue"])
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    st.markdown("**[평균 혈당(avg_glucose_level) 분포]**")
    fig_glucose = px.histogram(df, x="avg_glucose_level", nbins=30, labels={"avg_glucose_level": "평균 혈당"}, color_discrete_sequence=["orange"])
    st.plotly_chart(fig_glucose, use_container_width=True)

st.markdown("---")

# 4. 뇌졸중 여부에 따른 나이와 평균 혈당 상자그림 비교 및 평균값 표
st.subheader("📦 뇌졸중 여부에 따른 나이 및 평균 혈당 비교")
df["stroke_label"] = df["stroke"].map({0: "정상(0)", 1: "뇌졸중(1)"})

col3, col4 = st.columns(2)

with col3:
    st.markdown("**[뇌졸중 여부별 나이 상자그림]**")
    fig_box_age = px.box(df, x="stroke_label", y="age", color="stroke_label", labels={"stroke_label": "뇌졸중 여부", "age": "나이"})
    st.plotly_chart(fig_box_age, use_container_width=True)

with col4:
    st.markdown("**[뇌졸중 여부별 평균 혈당 상자그림]**")
    fig_box_glucose = px.box(df, x="stroke_label", y="avg_glucose_level", color="stroke_label", labels={"stroke_label": "뇌졸중 여부", "avg_glucose_level": "평균 혈당"})
    st.plotly_chart(fig_box_glucose, use_container_width=True)

# 두 그룹의 평균값 표 함께 제공
st.markdown("**[두 그룹의 평균값 비교 표]**")
mean_table = df.groupby("stroke_label")[["age", "avg_glucose_level"]].mean().reset_index()
mean_table.columns = ["뇌졸중 여부", "나이 평균", "평균 혈당 평균"]
st.dataframe(mean_table, hide_index=True)

st.markdown("---")

# 5. 고혈압 및 심장병 여부에 따른 뇌졸중 비율 막대그래프
st.subheader("📈 질환(고혈압, 심장병)에 따른 뇌졸중 비율")

col5, col6 = st.columns(2)

with col5:
    st.markdown("**[고혈압(hypertension)에 따른 뇌졸중 비율]**")
    hyp_stroke = df.groupby("hypertension", as_index=False)["stroke"].mean()
    hyp_stroke["hypertension_label"] = hyp_stroke["hypertension"].map({0: "고혈압 없음", 1: "고혈압 있음"})
    hyp_stroke["stroke_percent"] = hyp_stroke["stroke"] * 100
    fig_hyp = px.bar(hyp_stroke, x="hypertension_label", y="stroke_percent", text="stroke_percent", 
                     labels={"hypertension_label": "고혈압 여부", "stroke_percent": "뇌졸중 비율 (%)"},
                     color="hypertension_label")
    fig_hyp.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig_hyp, use_container_width=True)

with col6:
    st.markdown("**[심장병(heart_disease)에 따른 뇌졸중 비율]**")
    heart_stroke = df.groupby("heart_disease", as_index=False)["stroke"].mean()
    heart_stroke["heart_label"] = heart_stroke["heart_disease"].map({0: "심장병 없음", 1: "심장병 있음"})
    heart_stroke["stroke_percent"] = heart_stroke["stroke"] * 100
    fig_heart = px.bar(heart_stroke, x="heart_label", y="stroke_percent", text="stroke_percent", 
                       labels={"heart_label": "심장병 여부", "stroke_percent": "뇌졸중 비율 (%)"},
                       color="heart_label")
    fig_heart.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig_heart, use_container_width=True)

st.markdown("---")

# 6. 체질량지수(bmi) 결측치 그룹 vs 전체 뇌졸중 비율 비교 표
st.subheader("📋 체질량지수(bmi) 결측치 그룹과 전체 뇌졸중 비율 비교")

bmi_null_df = df[df["bmi"].isnull()]
null_total = len(bmi_null_df)
null_stroke_ratio = (bmi_null_df["stroke"].sum() / null_total * 100) if null_total > 0 else 0

total_count = len(df)
total_stroke_ratio = (df["stroke"].sum() / total_count * 100) if total_count > 0 else 0

comparison_data = [
    {"구분": "BMI 비어있는 사람들", "인원수": null_total, "뇌졸중 비율(%)": round(null_stroke_ratio, 2)},
    {"구분": "전체 사람들", "인원수": total_count, "뇌졸중 비율(%)": round(total_stroke_ratio, 2)}
]
comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, hide_index=True)

st.markdown("---")

# 7. 흡연 상태(smoking_status) 값마다 사람 수 세기 표
st.subheader("🚬 흡연 상태별 인원수")
smoking_counts = df["smoking_status"].value_counts().reset_index()
smoking_counts.columns = ["흡연 상태 (smoking_status)", "사람 수"]
st.dataframe(smoking_counts, hide_index=True)
