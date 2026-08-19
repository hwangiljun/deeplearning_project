"""모델 성능 페이지.

이전 보고서의 약점을 화면에서 그대로 드러내려고 만든 페이지다.

* 주 지표를 정확도 하나로 두지 않는다. 이 시스템은 확률로 순위를 매기므로
  **log-loss 와 캘리브레이션**이 정확도보다 본질적이다.
* 검증셋 최고 에폭 값을 성능으로 보고하지 않는다. 학습 곡선을 그대로 보여주고,
  성능은 한 번도 쓰지 않은 **테스트 시즌** 값으로 적는다.
* 어블레이션 표를 싣는다. 무엇이 성능에 기여했는지 분리해서 보여준다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..data import OUTCOMES
from . import charts
from .loaders import MODEL_DIR, MODELS
from .theme import AXIS, GRID, INK, INK_2, MUTED, plotly_layout, tiles

KO = {"ball": "볼", "called_strike": "루킹", "swinging_strike": "헛스윙",
      "foul": "파울", "hit_by_pitch": "사구", "field_out": "인플레이 아웃",
      "single": "단타", "double": "2루타", "triple": "3루타", "home_run": "홈런"}


def _curve(history: list[dict]) -> go.Figure:
    ep = [h["epoch"] for h in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ep, y=[h["log_loss"] for h in history], mode="lines+markers",
        line=dict(color="#2a78d6", width=2),
        marker=dict(size=6, color="#2a78d6", line=dict(width=2, color="#fcfcfb")),
        name="검증 log-loss",
        hovertemplate="epoch %{x}<br>log-loss %{y:.4f}<extra></extra>"))
    best = int(np.argmin([h["log_loss"] for h in history]))
    fig.add_annotation(x=ep[best], y=history[best]["log_loss"],
                       text=f"최저 {history[best]['log_loss']:.4f}",
                       showarrow=True, arrowhead=0, arrowcolor=AXIS,
                       font=dict(size=11, color=INK_2), ay=-28)
    fig.update_layout(**plotly_layout(
        height=300,
        xaxis=dict(showgrid=False, zeroline=False, linecolor=AXIS,
                   title=dict(text="에폭", font=dict(size=11, color=MUTED)),
                   tickfont=dict(color=MUTED, size=10)),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, showline=False,
                   title=dict(text="검증 log-loss", font=dict(size=11, color=MUTED)),
                   tickfont=dict(color=MUTED, size=10)),
        margin=dict(l=58, r=8, t=16, b=44)))
    return fig


def _ablation_table() -> pd.DataFrame | None:
    """어블레이션 결과를 표로. ``ablation.json`` 을 우선 읽는다.

    노트북의 요약 표 셀(csv 생성)을 실행하지 않아도 되도록, 루프가 저장한
    원본 json 에서 직접 만든다.
    """
    for path in (MODELS / "ablation.json", MODEL_DIR / "ablation.json"):
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            rows = []
            for name, v in raw.items():
                test = v.get("test", {})
                rows.append({
                    "구성": name,
                    "log-loss": round(test.get("log_loss", float("nan")), 4),
                    "macro-AUC": round(test.get("macro_auc", float("nan")), 4),
                    "정확도(%)": round(test.get("accuracy", float("nan")) * 100, 2),
                    "macro-F1": round(test.get("macro_f1", float("nan")), 4),
                    "에폭": v.get("epochs"),
                })
            t = pd.DataFrame(rows).sort_values("log-loss").reset_index(drop=True)
            # 기준선(full) 대비 차이를 함께 보여준다
            if "full" in t["구성"].values:
                base = t.loc[t["구성"] == "full", "log-loss"].iat[0]
                t["full 대비"] = (t["log-loss"] - base).round(4)
            return t

    for path in (MODELS / "ablation_table.csv", MODEL_DIR / "ablation_table.csv"):
        if path.exists():
            return pd.read_csv(path)
    return None


def render(res):
    st.markdown("## 모델 성능")

    if res.metadata is None:
        st.warning("`models/main/metadata.json` 이 없습니다. 학습 후 산출물을 넣어 주세요.")
        return

    meta = res.metadata
    tm = meta.get("test_metrics") or {}
    bv = meta.get("best_val") or {}

    st.markdown('<p class="caption">테스트 지표는 학습·조기종료·모델 선택에 '
                '<b>한 번도 쓰지 않은 시즌</b>에서 측정한 값입니다.</p>',
                unsafe_allow_html=True)

    if tm:
        st.markdown(tiles([
            ("log-loss", f"{tm.get('log_loss', float('nan')):.4f}", "낮을수록 좋음 · 주 지표"),
            ("macro AUC", f"{tm.get('macro_auc', float('nan')):.4f}", "클래스 균등 가중"),
            ("macro F1", f"{tm.get('macro_f1', float('nan')):.4f}", "희귀 클래스 반영"),
            ("정확도", f"{tm.get('accuracy', 0)*100:.2f}%", "참고 지표"),
        ]), unsafe_allow_html=True)

    st.markdown(
        f'<p class="caption">학습 시즌 {meta.get("train_seasons", "?")} · '
        f'테스트 시즌 {meta.get("test_seasons", "?")}</p>', unsafe_allow_html=True)

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### 학습 곡선")
        hist = meta.get("history")
        if hist:
            st.plotly_chart(_curve(hist), width="stretch")
            st.markdown('<p class="caption">조기 종료 기준은 검증 log-loss 입니다. '
                        '정확도의 우연한 최고점을 성능으로 삼지 않습니다.</p>',
                        unsafe_allow_html=True)
        else:
            st.info("학습 이력이 없습니다.")

    with c2:
        st.markdown("### 캘리브레이션")
        ev = MODEL_DIR / "evaluation.pkl"
        if ev.exists():
            import joblib
            e = joblib.load(ev)
            st.plotly_chart(
                charts.reliability_diagram(e["prob_true"], e["prob_pred"]),
                width="stretch")
            st.markdown(f'<p class="caption">ECE {e.get("ece", float("nan")):.4f} — '
                        '대각선에 붙을수록 표시되는 확률이 실제 빈도와 일치합니다.</p>',
                        unsafe_allow_html=True)
        else:
            st.info("`models/main/evaluation.pkl` 이 없습니다. "
                    "`scripts/evaluate_model.py` 를 실행하면 생성됩니다.")

    # ---- 혼동 행렬 / 클래스별 지표 ----
    ev = MODEL_DIR / "evaluation.pkl"
    if ev.exists():
        import joblib
        e = joblib.load(ev)
        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
        c3, c4 = st.columns([1.1, 1])
        with c3:
            st.markdown("### 혼동 행렬 (행 정규화)")
            st.plotly_chart(
                charts.confusion_heatmap(e["confusion"], [KO[o] for o in OUTCOMES]),
                width="stretch")
        with c4:
            st.markdown("### 클래스별 지표")
            per = pd.DataFrame(e["per_class"])
            per["클래스"] = [KO[o] for o in OUTCOMES]
            show = per[["클래스", "support", "precision", "recall", "f1", "auc"]].copy()
            show.columns = ["클래스", "표본", "정밀도", "재현율", "F1", "AUC"]
            st.dataframe(show.round(3), width="stretch", hide_index=True,
                         height=400)

    # ---- 어블레이션 ----
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    st.markdown("### 어블레이션")
    st.markdown('<p class="caption">한 번에 한 요소만 바꿔 기여를 분리합니다. '
                '특히 <code>seq_len_1</code> 은 직전 투구 정보를 완전히 제거한 조건이라, '
                '전체 모델이 이를 이기지 못하면 "투구 배합이 중요하다"는 전제가 성립하지 않습니다.</p>',
                unsafe_allow_html=True)
    t = _ablation_table()
    if t is not None:
        st.dataframe(t, width="stretch", hide_index=True)
    else:
        st.info("어블레이션 결과가 없습니다 (`models/ablation.json`). "
                "`notebooks/02_train.ipynb` 의 어블레이션 셀을 실행하세요.")
