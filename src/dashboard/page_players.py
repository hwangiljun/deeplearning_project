"""선수·팀 분석 페이지.

값의 부호 규약은 앱 전체에서 하나다.
**delta_run_exp 는 공격 팀 관점이다. 양수 = 타자에게 유리, 음수 = 투수에게 유리.**
그래서 히트맵의 빨강은 언제나 "투수에게 위험한 구역", 파랑은 "투수에게 안전한
구역" 을 뜻한다. 페이지가 바뀌어도 색의 의미는 바뀌지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from . import charts
from .loaders import (min_pitches_setting, player_options_counted,
                      seasons_available, team_options)
from .theme import INK_2, MUTED, SUCCESS_TEXT, STATUS, pitch_label, tiles

PCT = lambda v: "—" if pd.isna(v) else f"{v*100:.1f}%"          # noqa: E731
NUM = lambda v, d=2: "—" if pd.isna(v) else f"{v:+.{d}f}"       # noqa: E731


def _scope_controls(profiles, key: str, role: str):
    """필터 한 줄. 여러 차트를 한 슬라이스로 묶어 준다 (차트별 필터 금지)."""
    seasons = seasons_available(profiles)
    c1, c2, c3 = st.columns([1, 1.2, 3])
    season = c1.selectbox("시즌", seasons, key=f"{key}_season")
    team = c2.selectbox("팀", team_options(profiles, season), key=f"{key}_team")

    opts, total = player_options_counted(profiles, role, season, team)
    if not opts:
        c3.warning(
            f"조건에 맞는 선수가 없습니다. 이 범위에 {total}명이 있지만 모두 "
            f"최소 표본({min_pitches_setting():,}구) 미만입니다. "
            "사이드바에서 최소 표본을 낮추거나 시즌을 바꿔 보세요.")
        return season, team, None, None
    label = c3.selectbox(f"{'타자' if role == 'batter' else '투수'} ({len(opts)}/{total}명)",
                         [o[0] for o in opts], key=f"{key}_player")
    pid = dict(opts)[label]
    return season, team, pid, label


def _league_row(summary: pd.DataFrame, season: int) -> pd.Series:
    sub = summary[summary["season"] == season]
    return sub.mean(numeric_only=True)


def _delta_note(v, ref, higher_is_better: bool, pct: bool = True) -> str:
    if pd.isna(v) or pd.isna(ref):
        return ""
    d = v - ref
    good = (d > 0) if higher_is_better else (d < 0)
    color = SUCCESS_TEXT if good else STATUS["critical"]
    txt = f"{d*100:+.1f}%p" if pct else f"{d:+.2f}"
    return f'<span style="color:{color}">리그比 {txt}</span>'


# --------------------------------------------------------------------------
def render_batter(res):
    st.markdown("## 타자 분석")
    st.markdown('<p class="caption">공격 팀 관점 기대 실점 기준입니다. '
                '<b>양수(빨강) = 타자에게 유리</b>, 음수(파랑) = 투수에게 유리. '
                '표본이 적은 칸은 리그 평균 쪽으로 축소해 우연한 극단값을 걸러냅니다.</p>',
                unsafe_allow_html=True)

    p = res.profiles
    season, team, pid, label = _scope_controls(p, "bat", "batter")
    if pid is None:
        return

    row = p.batter_summary[(p.batter_summary["season"] == season)
                           & (p.batter_summary["batter"] == pid)]
    if row.empty:
        st.info("이 선수의 표본이 부족합니다.")
        return
    r = row.iloc[0]
    lg = _league_row(p.batter_summary, season)

    st.markdown(tiles([
        ("투구 수", f"{int(r.pitches):,}", f"{int(r.pa):,} 타석"),
        ("기대 실점 / 100구", f"{r.run_value_per_100:+.2f}",
         _delta_note(r.run_value_per_100, lg.run_value_per_100, True, pct=False)),
        ("헛스윙률", PCT(r.whiff_rate), _delta_note(r.whiff_rate, lg.whiff_rate, False)),
        ("체이스율", PCT(r.chase_rate), _delta_note(r.chase_rate, lg.chase_rate, False)),
        ("삼진율", PCT(r.k_rate), _delta_note(r.k_rate, lg.k_rate, False)),
        ("볼넷율", PCT(r.bb_rate), _delta_note(r.bb_rate, lg.bb_rate, True)),
    ]), unsafe_allow_html=True)

    left, right = st.columns([1.05, 1])

    with left:
        key = (int(season), str(pid))
        if key in p.batter_zone:
            grid, cnt = p.batter_zone[key]
            lim = float(np.nanpercentile(np.abs(grid), 97)) or 0.05
            st.plotly_chart(
                charts.zone_heatmap(grid, p.grid_x, p.grid_z,
                                    title=f"{label.split('·')[0].strip()} — 코스별 기대 실점",
                                    counts=cnt, symmetric=lim, height=400),
                width="stretch")
            with st.expander("표로 보기"):
                st.dataframe(charts.zone_table(grid, p.grid_x, p.grid_z, cnt),
                             width="stretch", hide_index=True)
        else:
            st.info("존 히트맵을 그릴 표본이 부족합니다.")

    with right:
        bp = p.batter_by_pitch
        bp = bp[(bp["season"] == season) & (bp["batter"] == pid)]
        if len(bp):
            bp = bp.sort_values("n", ascending=False)
            st.plotly_chart(
                charts.signed_bars([pitch_label(t) for t in bp["pitch_type"]],
                                   bp["run_value_per_100"].tolist(),
                                   height=max(220, 42 * len(bp)),
                                   unit="실점/100구",
                                   title="구종별 기대 실점 (100구당)"),
                width="stretch")
            show = bp[["pitch_type", "n", "run_value_per_100", "whiff_rate"]].copy()
            show.columns = ["구종", "표본", "기대실점/100구", "헛스윙률"]
            show["헛스윙률"] = show["헛스윙률"].map(PCT)
            st.dataframe(show, width="stretch", hide_index=True)
        else:
            st.info("구종별 표본이 부족합니다.")


# --------------------------------------------------------------------------
def render_pitcher(res):
    st.markdown("## 투수 분석")
    st.markdown('<p class="caption">구종 색은 구종에 고정되어 있어 필터를 바꿔도 '
                '같은 구종은 같은 색을 유지합니다.</p>', unsafe_allow_html=True)

    p = res.profiles
    season, team, pid, label = _scope_controls(p, "pit", "pitcher")
    if pid is None:
        return

    row = p.pitcher_summary[(p.pitcher_summary["season"] == season)
                            & (p.pitcher_summary["pitcher"] == pid)]
    if row.empty:
        st.info("이 선수의 표본이 부족합니다.")
        return
    r = row.iloc[0]
    lg = _league_row(p.pitcher_summary, season)

    st.markdown(tiles([
        ("투구 수", f"{int(r.pitches):,}", f"{int(r.pa):,} 타석 상대"),
        ("기대 실점 / 100구", f"{r.run_value_per_100:+.2f}",
         _delta_note(r.run_value_per_100, lg.run_value_per_100, False, pct=False)),
        ("헛스윙률", PCT(r.whiff_rate), _delta_note(r.whiff_rate, lg.whiff_rate, True)),
        ("존 투구율", PCT(r.zone_rate), _delta_note(r.zone_rate, lg.zone_rate, True)),
        ("삼진율", PCT(r.k_rate), _delta_note(r.k_rate, lg.k_rate, True)),
        ("볼넷율", PCT(r.bb_rate), _delta_note(r.bb_rate, lg.bb_rate, False)),
    ]), unsafe_allow_html=True)

    # 레퍼토리 — 참조 테이블이 있으면 실제 평균 물리량을 함께 보여준다
    rep = None
    if res.tables is not None:
        t = res.tables.repertoire
        rep = t[t["pitcher"].astype(str) == str(pid)].copy()

    c1, c2 = st.columns([1, 1])
    with c1:
        if rep is not None and len(rep):
            st.plotly_chart(
                charts.usage_bars(rep["pitch_type"].tolist(), rep["usage"].tolist(),
                                  title="구종 구사율"),
                width="stretch")
        else:
            st.info("레퍼토리 테이블이 없습니다 (`models/tables.pkl`).")
    with c2:
        if rep is not None and len(rep):
            st.plotly_chart(charts.movement_scatter(rep, title="구종별 무브먼트"),
                            width="stretch")

    if rep is not None and len(rep):
        show = rep[["pitch_type", "n", "usage", "speed", "spin_rate",
                    "pfx_x", "pfx_z"]].copy()
        show["usage"] = (show["usage"] * 100).round(1)
        show["pfx_x"] = (show["pfx_x"] * 12).round(1)
        show["pfx_z"] = (show["pfx_z"] * 12).round(1)
        show.columns = ["구종", "표본", "구사율(%)", "구속(mph)", "회전(rpm)",
                        "수평무브(in)", "수직무브(in)"]
        st.dataframe(show.round(1), width="stretch", hide_index=True)

    c3, c4 = st.columns([1, 1])
    with c3:
        u = p.pitcher_usage
        u = u[(u["season"] == season) & (u["pitcher"] == pid)]
        if len(u):
            piv = u.pivot_table(index="count_state", columns="pitch_type",
                                values="usage").reindex(range(12)).fillna(0)
            st.plotly_chart(charts.count_usage_heatmap(piv, title="카운트별 구종 선택"),
                            width="stretch")
    with c4:
        key = (int(season), str(pid))
        if key in p.pitcher_zone:
            grid, cnt = p.pitcher_zone[key]
            lim = float(np.nanpercentile(np.abs(grid), 97)) or 0.05
            st.plotly_chart(
                charts.zone_heatmap(grid, p.grid_x, p.grid_z,
                                    title="코스별 결과 (양수=피해)", counts=cnt,
                                    symmetric=lim, height=330),
                width="stretch")


# --------------------------------------------------------------------------
def render_team(res):
    st.markdown("## 팀 분석")
    st.markdown('<p class="caption">투수진 기준 집계입니다. 기대 실점이 낮을수록 '
                '좋은 투수진입니다.</p>', unsafe_allow_html=True)

    p = res.profiles
    seasons = seasons_available(p)
    season = st.selectbox("시즌", seasons, key="team_season")

    t = p.team_summary[p.team_summary["season"] == season].copy()
    if t.empty:
        st.info("데이터가 없습니다.")
        return

    t = t.sort_values("run_value_per_100")
    st.plotly_chart(
        charts.signed_bars(t["team"].tolist(), t["run_value_per_100"].tolist(),
                           height=max(320, 22 * len(t)), unit="실점/100구",
                           title="팀 투수진 기대 실점 (100구당, 낮을수록 우수)"),
        width="stretch")

    show = t[["team", "pitches", "run_value_per_100", "whiff_rate",
              "zone_rate", "k_rate", "bb_rate"]].copy()
    show["run_value_per_100"] = show["run_value_per_100"].round(3)
    for c in ["whiff_rate", "zone_rate", "k_rate", "bb_rate"]:
        show[c] = show[c].map(PCT)
    show.columns = ["팀", "투구 수", "기대실점/100구", "헛스윙률", "존 투구율",
                    "삼진율", "볼넷율"]
    st.dataframe(show, width="stretch", hide_index=True)

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    st.markdown("### 팀 소속 선수")
    team = st.selectbox("팀 선택", [x for x in team_options(p, season) if x != "전체"],
                        key="team_roster")
    c1, c2 = st.columns(2)
    for col, role, title in [(c1, "pitcher", "투수"), (c2, "batter", "타자")]:
        with col:
            st.markdown(f"**{title}**")
            d = p.directory
            d = d[(d["role"] == role) & (d["season"] == season)
                  & (d["team"] == team)].nlargest(30, "pitches")
            out = d[["name", "hand", "pitches"]].copy()
            out.columns = ["선수", "손", "투구 수"]
            st.dataframe(out, width="stretch", hide_index=True, height=380)
