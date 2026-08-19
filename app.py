"""NextPitch — 투구 전략 분석 대시보드.

실행:  streamlit run app.py

페이지 구성
  투구 추천   학습된 모델로 (구종 × 코스) 후보를 평가해 기대 실점이 가장 낮은
              선택을 찾는다. 모델 산출물이 있어야 동작한다.
  타자 분석   코스별·구종별 약점. 데이터 집계만으로 동작한다.
  투수 분석   레퍼토리, 무브먼트, 카운트별 구종 선택.
  팀 분석     팀 단위 투수진 지표와 소속 선수.
  모델 성능   테스트 시즌 지표, 학습 곡선, 캘리브레이션, 어블레이션.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(page_title="NextPitch — 투구 전략 분석",
                   page_icon="⚾", layout="wide",
                   initial_sidebar_state="expanded")

from src.dashboard import page_advisor, page_model, page_players
from src.dashboard.loaders import load_resources
from src.dashboard.theme import CSS

st.markdown(CSS, unsafe_allow_html=True)

res, warnings = load_resources()

# ---------------------------------------------------------------- 사이드바
with st.sidebar:
    st.markdown("### ⚾ NextPitch")
    st.markdown('<p class="caption">MLB Statcast 기반 투구 전략 분석</p>',
                unsafe_allow_html=True)
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    PAGES = {
        "투구 추천": ("advisor", res.has_model),
        "타자 분석": ("batter", res.has_profiles),
        "투수 분석": ("pitcher", res.has_profiles),
        "팀 분석": ("team", res.has_profiles),
        "모델 성능": ("model", res.metadata is not None),
    }
    choice = st.radio("메뉴", list(PAGES),
                      format_func=lambda k: k if PAGES[k][1] else f"{k}  (준비 안 됨)",
                      label_visibility="collapsed")

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    ready = [k for k, (_, ok) in PAGES.items() if ok]
    st.markdown(f'<p class="caption">사용 가능: {len(ready)}/{len(PAGES)} 페이지</p>',
                unsafe_allow_html=True)

    if warnings:
        with st.expander(f"준비 안 된 항목 {len(warnings)}건"):
            for w in warnings:
                st.markdown(f'<p class="caption">· {w}</p>', unsafe_allow_html=True)

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="caption">기대 실점은 <b>공격 팀 관점</b>입니다.<br>'
        '양수(빨강) = 타자 유리 · 음수(파랑) = 투수 유리</p>',
        unsafe_allow_html=True)

# ---------------------------------------------------------------- 본문
key = PAGES[choice][0]

if key == "advisor":
    page_advisor.render(res)
elif key == "batter":
    if res.has_profiles:
        page_players.render_batter(res)
    else:
        st.warning("프로파일이 없습니다. `python scripts/build_profiles.py "
                   "--raw data/raw --out models/profiles.pkl` 를 실행하세요.")
elif key == "pitcher":
    if res.has_profiles:
        page_players.render_pitcher(res)
    else:
        st.warning("프로파일이 없습니다.")
elif key == "team":
    if res.has_profiles:
        page_players.render_team(res)
    else:
        st.warning("프로파일이 없습니다.")
elif key == "model":
    page_model.render(res)
