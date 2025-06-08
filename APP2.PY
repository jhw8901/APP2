import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 모델 및 스케일러 로드
model = joblib.load("mental_health_model.pkl")

# 점수 계산 함수들
def calculate_emotional_loneliness(eq1, eq2, eq3):
    return eq1 + eq2 + eq3

def calculate_social_loneliness(sq1, sq2, sq3):
    return sq1 + sq2 + sq3

def calculate_total_loneliness(eq1, eq2, eq3, sq1, sq2, sq3):
    return eq1 + eq2 + eq3 + sq1 + sq2 + sq3

def calculate_smcq(sm_list):
    return sum(sm_list)

def calculate_sgdsk(sgd_list):
    return sum(sgd_list)

def calculate_mood_scores(m1, n1, t1, i1, f1, g1, l1, s1):
    return {
        'melanco': m1,
        'noninterest': n1,
        'loostaste': t1,
        'irritate': i1,
        'fatigue': f1,
        'guilty': g1,
        'lessfocus': l1,
        'suicidal': s1
    }

st.set_page_config(page_title="Depression Risk Checker", layout="centered")
st.title("우울증 위험 예측 시스템")
st.markdown("설문 문항에 응답해 주세요.")

# 감정적 외로움 문항
st.markdown("### 감정적 외로움")
eq1 = st.radio("1. 대체로 공허감을 경험한다.", [1, 2, 3], format_func=lambda x: {1: "아니다", 2: "보통이다", 3: "그렇다"}[x])
eq2 = st.radio("2. 내 주변에 사람이 있을 때가 그립다.", [1, 2, 3], format_func=lambda x: {1: "아니다", 2: "보통이다", 3: "그렇다"}[x])
eq3 = st.radio("3. 가끔씩 버림받았다는 느낌이 든다.", [1, 2, 3], format_func=lambda x: {1: "아니다", 2: "보통이다", 3: "그렇다"}[x])

# 사회적 외로움 문항
st.markdown("### 사회적 외로움")
sq1 = st.radio("4. 내가 곤란한 경우에 처했을 때 의지할 수 있는 사람이 많다.", [1, 2, 3], format_func=lambda x: {1: "아니다", 2: "보통이다", 3: "그렇다"}[x])
sq2 = st.radio("5. 내가 완전히 믿을 수 있는 사람이 많다.", [1, 2, 3], format_func=lambda x: {1: "아니다", 2: "보통이다", 3: "그렇다"}[x])
sq3 = st.radio("6. 내가 친하게 느끼는 사람은 충분히 있다.", [1, 2, 3], format_func=lambda x: {1: "아니다", 2: "보통이다", 3: "그렇다"}[x])

# SMCQ 13문항
st.markdown("### 인지기능 저하 자가평가 (SMCQ)")
smcq_questions = [
    "1. 당신은 기억력에 문제가 있습니까?",
    "2. 당신의 기억력은 10년 전에 비해 저하되었습니까?",
    "3. 당신은 기억력이 동년의 다른 사람들에 비해 나쁘다고 생각합니까?",
    "4. 당신은 기억력 저하로 일상생활에 불편을 느끼십니까?",
    "5. 당신은 최근에 일어난 일을 기억하는 것이 어렵습니까?",
    "6. 당신은 며칠 전에 나눈 대화 내용을 기억하는 것이 어렵습니까?",
    "7. 당신은 친한 사람의 이름을 기억하기 어렵습니까?",
    "8. 당신은 물건 둔 곳을 기억하기 어렵습니까?",
    "9. 당신은 이전에 비해 물건을 자주 잃어버립니까?",
    "10. 당신은 집 근처에서 길을 잃은 적이 있습니까?",
    "11. 당신은 가게에서 사려고 하는 두세 가지 물건 이름을 기억하기 어렵습니까?",
    "12. 당신은 가스불이나 전깃불 끄는 것을 기억하기 어렵습니까?",
    "13. 당신은 자주 사용하는 전화번호를 기억하기 어렵습니까?"
]
smcq = [st.radio(q, [0, 1], horizontal=True) for q in smcq_questions]

# SGDS-K 15문항
st.markdown("### 노인 우울척도 (SGDS-K)")
sgdsk_questions = [
    "1. 현재의 생활에 대체로 만족하십니까?",
    "2. 요즘 들어 활동량이나 의욕이 많이 떨어지셨습니까?",
    "3. 자신이 헛되이 살고 있다고 느끼십니까?",
    "4. 생활이 지루하게 느껴질 때가 많습니까?",
    "5. 평소에 기분은 상쾌한 편이십니까?",
    "6. 자신에게 불길한 일이 닥칠 것 같아 불안하십니까?",
    "7. 대체로 마음이 즐거운 편이십니까?",
    "8. 절망적이라는 느낌이 자주 드십니까?",
    "9. 바깥에 나가기가 싫고 집에만 있고 싶습니까?",
    "10. 또래 노인들보다 기억력이 더 나쁘다고 느끼십니까?",
    "11. 현재 살아 있다는 것이 즐겁게 생각되십니까?",
    "12. 지금의 내 자신이 아무 쓸모 없는 사람이라고 느끼십니까?",
    "13. 기력이 좋은 편이십니까?",
    "14. 지금 자신의 처지가 아무런 희망도 없다고 느끼십니까?",
    "15. 자신이 다른 사람들의 처지보다 더 못하다고 생각하십니까?"
]
sgdsk = [st.radio(q, [0, 1], horizontal=True) for q in sgdsk_questions]

# 기분장애 과거력
mod1 = st.radio("기분장애 과거력 있음?", ["아니오", "예"])

# 최근 2주간 기분 상태
st.markdown("### 최근 2주간 기분 상태")
mood = {
    'melanco': st.radio("우울한 기분", [0, 1]),
    'noninterest': st.radio("흥미 저하", [0, 1]),
    'loostaste': st.radio("입맛 감소", [0, 1]),
    'irritate': st.radio("짜증", [0, 1]),
    'fatigue': st.radio("피로", [0, 1]),
    'guilty': st.radio("죄책감", [0, 1]),
    'lessfocus': st.radio("집중력 저하", [0, 1]),
    'suicidal': st.radio("자살 생각", [0, 1]),
}

if st.button("우울증 위험 예측하기"):
    emotional = calculate_emotional_loneliness(eq1, eq2, eq3)
    social = calculate_social_loneliness(sq1, sq2, sq3)
    total = calculate_total_loneliness(eq1, eq2, eq3, sq1, sq2, sq3)
    sm_score = calculate_smcq(smcq)
    sgd_score = calculate_sgdsk(sgdsk)
    mod_value = 1 if mod1 == "예" else 0

    input_df = pd.DataFrame({
        "emotionalloneliness": [emotional],
        "socialloneliness": [social],
        "totalloneliness": [total],
        "SMCQ": [sm_score],
        "SGDSK": [sgd_score],
        "moodhistory": [mod_value],
        **{k: [v] for k, v in mood.items()}
    })

    # 전처리 포함 모델이므로 scaler 없이 바로 예측
    prediction = model.predict_proba(input_df)[0][1]

    st.subheader("📊 예측 결과")
    st.metric("우울증 위험도", f"{prediction * 100:.2f} %")

    if prediction >= 0.5:
        st.warning("⚠️ 우울 위험이 높습니다. 전문가의 상담이 필요할 수 있습니다.")
    else:
        st.success("😊 현재는 우울 위험이 낮은 상태입니다.")
