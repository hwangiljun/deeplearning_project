"""투구 추천 페이지 — 앱의 본체.

이전 앱과의 차이
----------------
* 후보 한 구를 8번 복사해 시퀀스를 만들던 것을 **이번 타석의 실제 투구 이력**
  으로 바꿨다. 학습이 실제 시퀀스로 이뤄졌으므로 그렇게 넣어야 맞다.
* 존 안 9칸이 아니라 **존 밖을 포함한 격자 전체**를 평가한다. 유인구가 추천
  후보에 들어온다.
* 점수가 임의 가중치가 아니라 **기대 실점 변화량**이다. 단위가 실점이라
  "이 공을 던지면 기대 실점이 0.03 줄어든다" 처럼 해석된다.
* 결과를 구종별 히트맵 소형 다중으로 함께 보여준다. Top-3 만 보면 "왜 그 공인가"
  를 알 수 없다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ..data import OUTCOMES
from ..recommend import GameState, apply_command_window, score_candidates, top_k
from . import charts
from .loaders import player_options, seasons_available, team_options
from .theme import (INK, INK_2, MUTED, STATUS, SUCCESS_TEXT, pitch_color,
                    pitch_label, tiles)

PITCH_ORDER = ["FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS"]


def _pick_player(profiles, role: str, season: int, key: str, label: str):
    c1, c2 = st.columns([1, 2])
    team = c1.selectbox(f"{label} 팀", team_options(profiles, season), key=f"{key}_team")
    opts = player_options(profiles, role, season, team)
    if not opts:
        c2.warning("해당 조건의 선수가 없습니다.")
        return None, None
    name = c2.selectbox(label, [o[0] for o in opts], key=f"{key}_p")
    return dict(opts)[name], name


def _hand_of(profiles, pid: str, season: int, role: str, default: str) -> str:
    d = profiles.directory
    m = d[(d["player_id"] == str(pid)) & (d["season"] == season) & (d["role"] == role)]
    return m["hand"].iat[0] if len(m) else default


def render(res):
    st.markdown("## 투구 전략 추천")

    if not res.has_model:
        st.warning(
            "학습된 모델이 아직 없습니다. `notebooks/02_train.ipynb` 를 실행해 "
            "`artifacts.zip` 을 받은 뒤 `models/main/` 에 풀어 넣으면 이 페이지가 활성화됩니다.\n\n"
            "그동안 **타자 분석 · 투수 분석 · 팀 분석** 페이지는 데이터 집계만으로 "
            "동작하므로 바로 사용할 수 있습니다.")
        return

    p = res.profiles
    seasons = seasons_available(p) if p is not None else [2025]

    # ---------------- 필터 (한 줄에 모아 모든 결과를 같은 슬라이스로) ----------------
    with st.container():
        top = st.columns([0.7, 2.6, 2.6])
        season = top[0].selectbox("시즌", seasons, key="adv_season")
        with top[1]:
            pitcher_id, pitcher_name = _pick_player(p, "pitcher", season, "adv_pit", "투수")
        with top[2]:
            batter_id, batter_name = _pick_player(p, "batter", season, "adv_bat", "타자")

    if pitcher_id is None or batter_id is None:
        return

    stand = _hand_of(p, batter_id, season, "batter", "R")
    throws = _hand_of(p, pitcher_id, season, "pitcher", "R")

    # ---------------- 상황 ----------------
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    s = st.columns([0.8, 0.8, 0.8, 2.0, 0.9, 1.0, 1.4])
    balls = s[0].number_input("볼", 0, 3, 0)
    strikes = s[1].number_input("스트라이크", 0, 2, 0)
    outs = s[2].number_input("아웃", 0, 2, 0)
    runners = s[3].multiselect("주자", ["1루", "2루", "3루"])
    inning = s[4].number_input("이닝", 1, 15, 1)
    score_diff = s[5].number_input("점수차", -10, 10, 0,
                                   help="투수 팀 기준. 양수면 이기고 있는 상황")
    command = s[6].select_slider(
        "투수 제구", options=["정밀", "보통", "거침"], value="보통",
        help="주변 칸까지 평균내는 범위. 제구가 나쁠수록 넓은 영역의 평균으로 평가합니다")
    window = {"정밀": 1, "보통": 3, "거침": 5}[command]

    st.caption(f"타석 {stand}타 · 투수 {throws}투 · "
               f"{'초' if True else ''}{inning}회 · {balls}-{strikes}, {outs}아웃")

    # ---------------- 이번 타석 이력 ----------------
    with st.expander("이번 타석에서 이미 던진 공 (선택)", expanded=False):
        st.markdown('<p class="caption">모델은 타석 안의 투구 순서를 보고 판단합니다. '
                    '실제 이력을 넣으면 배합을 반영한 추천이 나옵니다. '
                    '비워 두면 초구 상황으로 계산합니다.</p>', unsafe_allow_html=True)
        default = pd.DataFrame({"구종": pd.Series(dtype="str"),
                                "몸쪽(-1)↔바깥쪽(+1)": pd.Series(dtype="float"),
                                "높이 (0=존아래, 1=존위)": pd.Series(dtype="float")})
        hist_df = st.data_editor(
            default, num_rows="dynamic", width="stretch", key="adv_hist",
            column_config={
                "구종": st.column_config.SelectboxColumn(options=PITCH_ORDER, width="small"),
                "몸쪽(-1)↔바깥쪽(+1)": st.column_config.NumberColumn(
                    min_value=-1.5, max_value=1.5, step=0.1, format="%.2f"),
                "높이 (0=존아래, 1=존위)": st.column_config.NumberColumn(
                    min_value=-0.5, max_value=1.5, step=0.1, format="%.2f"),
            })

    history = []
    for _, row in hist_df.iterrows():
        if pd.notna(row.get("구종")):
            history.append({
                "pitch_type": row["구종"],
                "loc_x": float(row.get("몸쪽(-1)↔바깥쪽(+1)") or 0.0),
                "loc_z": float(row.get("높이 (0=존아래, 1=존위)") or 0.5),
            })

    if not st.button("전략 계산", type="primary", width="stretch"):
        return

    # ---------------- 계산 ----------------
    state = GameState(
        batter=batter_id, pitcher=pitcher_id, stand=stand, p_throws=throws,
        balls=balls, strikes=strikes, outs=outs,
        on_1b="1루" in runners, on_2b="2루" in runners, on_3b="3루" in runners,
        inning=inning, score_diff=score_diff, season=int(season), history=history,
    )

    with st.spinner("후보 시나리오 평가 중..."):
        cands = score_candidates(res.model, res.encoders, res.tables, state,
                                 batter_stats=res.batter_stats)
        cands = apply_command_window(cands, res.tables, window=window)
        best = top_k(cands, k=3)

    if not best:
        st.error("후보를 만들지 못했습니다.")
        return

    baseline = float(np.mean([c.window_value for c in cands]))
    st.markdown(tiles([
        ("평가한 후보", f"{len(cands):,}",
         f"구종 {len({c.pitch_type for c in cands})}종 × 코스 {len(res.tables.grid_x)*len(res.tables.grid_z)}"),
        ("최선의 기대 실점", f"{best[0].window_value:+.4f}", "낮을수록 투수 유리"),
        ("평균 대비 이득", f"{best[0].window_value - baseline:+.4f}",
         "무작위 선택 대비"),
        ("제구 가정", command, f"윈도우 {window}×{window}"),
    ]), unsafe_allow_html=True)

    # ---------------- Top-3 ----------------
    left, right = st.columns([1, 1.35])

    with left:
        st.markdown("### 추천 Top 3")
        for i, c in enumerate(best, 1):
            side = "바깥쪽" if c.loc_x > 0.15 else ("몸쪽" if c.loc_x < -0.15 else "가운데")
            hgt = "높게" if c.loc_z > 0.66 else ("낮게" if c.loc_z < 0.33 else "중간")
            zone = "존 안" if c.in_zone else "존 밖(유인구)"
            top3 = " · ".join(f"{n} {v*100:.0f}%" for n, v in c.summary(3))
            color = pitch_color(c.pitch_type)
            val_color = SUCCESS_TEXT if c.window_value < 0 else STATUS["critical"]
            st.markdown(f"""
            <div class="rec" style="--accent:{color}">
              <div class="rank">#{i}</div>
              <div class="name">{pitch_label(c.pitch_type)}</div>
              <div class="loc">{side} · {hgt} <span style="color:{MUTED}">({zone})</span></div>
              <div style="margin-top:6px">
                <span class="val" style="color:{val_color}">{c.window_value:+.4f}</span>
                <span class="sub"> 기대 실점</span>
              </div>
              <div class="sub" style="margin-top:4px">{top3}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### 1순위 결과 분포")
        st.plotly_chart(charts.outcome_bars(best[0].probs, OUTCOMES, height=290),
                        width="stretch")

    # ---------------- 구종별 히트맵 (소형 다중) ----------------
    with right:
        st.markdown("### 구종별 코스 지도")
        st.markdown('<p class="caption">모든 그림이 <b>같은 색 범위</b>를 씁니다. '
                    '파랑이 짙을수록 투수에게 유리한 코스입니다. 검은 사각형이 스트라이크존.</p>',
                    unsafe_allow_html=True)

        gx, gz = res.tables.grid_x, res.tables.grid_z
        by_pt: dict[str, np.ndarray] = {}
        for c in cands:
            g = by_pt.setdefault(c.pitch_type, np.full((len(gz), len(gx)), np.nan, np.float32))
            xi = int(np.argmin(np.abs(np.asarray(gx) - c.loc_x)))
            zi = int(np.argmin(np.abs(np.asarray(gz) - c.loc_z)))
            g[zi, xi] = c.window_value

        lim = float(np.nanmax([np.nanmax(np.abs(g)) for g in by_pt.values()]))
        order = [pt for pt in PITCH_ORDER if pt in by_pt] + \
                [pt for pt in by_pt if pt not in PITCH_ORDER]

        for i in range(0, len(order), 2):
            cols = st.columns(2)
            for col, pt in zip(cols, order[i:i + 2]):
                with col:
                    st.plotly_chart(
                        charts.zone_heatmap(by_pt[pt], gx, gz,
                                            title=pitch_label(pt), symmetric=lim,
                                            height=270),
                        width="stretch")

    # ---------------- 표 뷰 ----------------
    with st.expander("전체 후보 표로 보기"):
        rows = [{
            "구종": pitch_label(c.pitch_type),
            "몸쪽↔바깥쪽": round(c.loc_x, 2),
            "높이": round(c.loc_z, 2),
            "존": "안" if c.in_zone else "밖",
            "기대실점": round(c.window_value, 4),
            **{f"P({n})": round(float(c.probs[j]), 3)
               for j, n in enumerate(OUTCOMES)},
        } for c in sorted(cands, key=lambda x: x.window_value)]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                     height=420)
