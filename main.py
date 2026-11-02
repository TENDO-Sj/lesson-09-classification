import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px

# 1. 브라우저 탭 설정 및 페이지 구성
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# 2. 화면 맨 위 제목과 아이콘
st.title("🧠 뇌졸중 예측 실습실")
st.write("뇌졸중 데이터셋을 탐색하고 각 열의 의미를 정리하는 실습 공간입니다.")

# 3. 데이터 불러오기 (요청하신 새로운 주소 적용)
DATA_URL = "https://raw.githubusercontent.com/TENDO-Sj/lesson09-distinguish/refs/heads/main/stroke.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df = load_data()

# 4. 상단 숫자 카드 네 개 (전체 사람 수, 열 개수, stroke=1 인원, 비율)
total_people = len(df)
num_columns = len(df.columns)
stroke_count = int((df['stroke'] == 1).sum())
stroke_ratio = (stroke_count / total_people) * 100 if total_people > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("전체 사람 수", f"{total_people:,}명")
col2.metric("열 개수", f"{num_columns}개")
col3.metric("뇌졸중(1) 인원", f"{stroke_count:,}명")
col4.metric("뇌졸중 비율", f"{stroke_ratio:.2f}%")

st.markdown("---")

# 5. '내_기록' 폴더 및 JSON 파일 처리 (저장 및 불러오기)
os.makedirs("내_기록", exist_ok=True)
meta_path = os.path.join("내_기록", "stroke_meta.json")

# 기존 저장된 내용 불러오기
meta_data = {"meanings": {}, "source_text": ""}
if os.path.exists(meta_path):
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
    except Exception:
        pass

# 6. 열 정보 요약 표 생성 (우리말 뜻은 저장된 값 또는 빈 칸)
summary_data = []
for col in df.columns:
    dtype = str(df[col].dtype)
    missing_count = int(df[col].isnull().sum())
    saved_meaning = meta_data.get("meanings", {}).get(col, "")
    summary_data.append({
        "열 이름": col,
        "우리말 뜻": saved_meaning,
        "값의 종류": dtype,
        "빈 값 개수": missing_count
    })

summary_df = pd.DataFrame(summary_data)

st.subheader("📋 데이터셋 열 정보 확인 및 우리말 뜻 작성")
st.write("교재를 참고하여 '우리말 뜻' 칸을 직접 채우고 아래의 **[내용 저장하기]** 버튼을 눌러주세요.")

# 표 편집 기능 (st.data_editor 활용)
edited_summary = st.data_editor(
    summary_df,
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름", disabled=True),
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻 (여기에 입력하세요)", required=False),
        "값의 종류": st.column_config.TextColumn("값의 종류", disabled=True),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", disabled=True)
    },
    hide_index=True,
    key="summary_editor"
)

st.markdown("---")

# 7. 데이터 출처 작성 자리
st.subheader("📚 데이터 출처")
saved_source = meta_data.get("source_text", "")
source_text_input = st.text_area("교재를 참고하여 데이터 출처를 입력하세요:", value=saved_source, key="source_input")

# 저장 버튼
if st.button("내용 저장하기"):
    new_meanings = {}
    for idx, row in edited_summary.iterrows():
        new_meanings[row["열 이름"]] = row["우리말 뜻"] if row["우리말 뜻"] is not None else ""
    
    updated_meta = {
        "meanings": new_meanings,
        "source_text": source_text_input
    }
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(updated_meta, f, ensure_ascii=False, indent=4)
    st.success("내용이 '내_기록/stroke_meta.json' 파일에 성공적으로 저장되었습니다! (새로고침해도 유지됩니다)")

st.markdown("---")

# 8. 데이터를 처음 다섯 줄만 그대로 표시
st.subheader("👀 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(5))
