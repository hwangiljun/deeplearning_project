"""보고서에 넣을 그림을 실제 산출물에서 렌더한다.

    python scripts/build_report_figures.py

모든 그림은 models/ 의 산출물과 data 에서 계산된 값만 쓴다. 손으로 그린
수치는 없다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dashboard import charts
from src.dashboard.theme import (AXIS, GRID, INK, INK_2, MUTED, SEQUENTIAL,
                                 SURFACE, plotly_layout)
from src.data import OUTCOMES
from src.features import ReferenceTables
from src.profiles import Profiles

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

KO = {"ball": "볼", "called_strike": "루킹", "swinging_strike": "헛스윙",
      "foul": "파울", "hit_by_pitch": "사구", "field_out": "인플레이 아웃",
      "single": "단타", "double": "2루타", "triple": "3루타", "home_run": "홈런"}

BLUE, RED, GRAY = "#2a78d6", "#e34948", "#898781"


def save(fig, name, w, h):
    p = OUT / name
    fig.write_image(p, width=w, height=h, scale=2)
    print(f"  {name}  ({p.stat().st_size/1024:.0f}KB)")


# --------------------------------------------------------------------------
def fig_leak():
    """그림 1. 시간 역순 시퀀스가 만들어 낸 누수 구조."""
    fig = go.Figure()

    def box(x, y, w, h, text, fill, line=INK, tc=INK):
        fig.add_shape(type="rect", x0=x, x1=x + w, y0=y, y1=y + h,
                      fillcolor=fill, line=dict(color=line, width=1.2))
        fig.add_annotation(x=x + w / 2, y=y + h / 2, text=text, showarrow=False,
                           font=dict(size=11, color=tc))

    # 실제 시간 흐름
    fig.add_annotation(x=0.5, y=3.62, text="<b>실제 시간 흐름</b>", showarrow=False,
                       xref="x", font=dict(size=12, color=INK_2))
    for i, lab in enumerate(["1구", "2구", "3구", "4구", "5구", "6구", "7구", "8구"]):
        box(i, 3.0, 0.9, 0.5, lab, "#eef3fa")
    fig.add_annotation(x=8.35, y=3.25, text="→", showarrow=False,
                       font=dict(size=20, color=MUTED))

    # 원본 파일 순서 (역순)
    fig.add_annotation(x=0.5, y=2.40, text="<b>원본 파일에 담긴 순서 (역시간순)</b>",
                       showarrow=False, xref="x", font=dict(size=12, color=INK_2))
    for i, lab in enumerate(["8구", "7구", "6구", "5구", "4구", "3구", "2구", "1구"]):
        box(i, 1.6, 0.9, 0.5, lab, "#fdeeee")

    # 정렬 없이 잘라낸 윈도우
    fig.add_shape(type="rect", x0=-0.08, x1=8.0, y0=1.5, y1=2.2,
                  line=dict(color=RED, width=2, dash="solid"),
                  fillcolor="rgba(0,0,0,0)")
    fig.add_annotation(x=7.45, y=1.28, text="<b>타깃</b> = 윈도우의 마지막 = 1구",
                       showarrow=False, font=dict(size=11, color=RED))
    fig.add_annotation(x=3.0, y=1.28,
                       text="입력으로 들어간 7개는 모두 <b>타깃보다 나중</b>에 던진 공",
                       showarrow=False, font=dict(size=11, color=RED))

    # 누수 경로
    fig.add_annotation(
        x=4.0, y=0.55,
        text=("카운트(balls·strikes)가 모든 타임스텝에 포함되어 있었으므로,<br>"
              "<b>2구의 카운트와 1구의 카운트를 비교하면 1구의 결과를 그대로 읽을 수 있다.</b><br>"
              "볼이 1 늘었으면 볼, 스트라이크가 1 늘었으면 스트라이크."),
        showarrow=False, font=dict(size=11, color=INK),
        align="center", bgcolor="#fdf4f4",
        bordercolor=RED, borderwidth=1, borderpad=8)

    fig.update_layout(**plotly_layout(
        height=380,
        xaxis=dict(visible=False, range=[-0.4, 9.0]),
        yaxis=dict(visible=False, range=[0.0, 3.9]),
        margin=dict(l=10, r=10, t=10, b=10)))
    save(fig, "fig1_leak.png", 900, 380)


def fig_architecture():
    """그림 2. 제안 모델 구조."""
    fig = go.Figure()

    def box(x, y, w, h, title, sub="", fill="#ffffff", edge=INK):
        fig.add_shape(type="rect", x0=x, x1=x + w, y0=y, y1=y + h,
                      fillcolor=fill, line=dict(color=edge, width=1.3))
        ty = y + h - 0.28 if h > 1.6 else y + h * 0.62
        fig.add_annotation(x=x + w / 2, y=ty, text=f"<b>{title}</b>",
                           showarrow=False, font=dict(size=11, color=INK))
        if sub:
            sy = y + h - 0.55 if h > 1.6 else y + h * 0.26
            fig.add_annotation(x=x + w / 2, y=sy, text=sub,
                               showarrow=False, font=dict(size=9.5, color=MUTED))

    def arrow(x0, y0, x1, y1):
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True, arrowhead=2,
                           arrowsize=1, arrowwidth=1.2, arrowcolor=MUTED, text="")

    # 시퀀스 경로 (왼쪽)
    box(0.1, 8.2, 3.4, 0.9, "투구 시퀀스", "타석 내 최근 6구 · 물리량 8개", "#eef3fa")
    box(0.1, 6.9, 1.6, 0.9, "엔티티 임베딩", "타자·투수·구종·좌우", "#ffffff")
    box(1.9, 6.9, 1.6, 0.9, "물리량", "위치·구속·회전·무브", "#ffffff")
    box(0.1, 5.6, 3.4, 0.9, "선형 사영 + 위치 임베딩", "d_model = 128")
    box(0.1, 4.0, 3.4, 1.2, "Transformer Encoder", "2층 · 4헤드 · FFN 512 · Pre-LN + 최종 LayerNorm", "#eef3fa")
    box(0.1, 2.7, 3.4, 0.9, "마스크드 평균 풀링", "패딩 위치 제외")

    # 맥락 경로 (오른쪽)
    box(4.3, 5.9, 3.0, 3.2, "맥락 정보", "", "#fdf1e9")
    box(4.3, 4.0, 3.0, 1.2, "Dense + ReLU", "32차원", "#fdf1e9")

    # 결합
    box(1.6, 1.4, 4.2, 0.9, "결합 (concat)", "128 + 32 = 160차원")
    box(1.6, 0.1, 4.2, 0.9, "출력 헤드", "160 → 128 → 64 → 10 클래스")

    arrow(1.8, 8.2, 0.9, 7.8)
    arrow(1.8, 8.2, 2.7, 7.8)
    arrow(0.9, 6.9, 1.5, 6.5)
    arrow(2.7, 6.9, 2.1, 6.5)
    arrow(1.8, 5.6, 1.8, 5.2)
    arrow(1.8, 4.0, 1.8, 3.6)
    arrow(1.8, 2.7, 2.6, 2.3)
    arrow(5.8, 5.9, 5.8, 5.2)
    arrow(5.8, 4.0, 4.8, 2.3)
    arrow(3.7, 1.4, 3.7, 1.0)

    fig.add_annotation(x=5.8, y=7.05, align="center",
                       text=("볼 · 스트라이크 · 아웃<br>주자 1·2·3루<br>이닝 · 초말<br>"
                             "점수차 · 타순 회차<br>타자 직전시즌 5개 지표"),
                       showarrow=False, font=dict(size=9.5, color=MUTED))
    fig.add_annotation(x=6.9, y=2.9, text="<b>스킵 연결</b><br>어텐션을 거치지 않고<br>출력층에 직접 결합",
                       showarrow=False, font=dict(size=9.5, color="#c25a1e"), align="left")

    fig.update_layout(**plotly_layout(
        height=620,
        xaxis=dict(visible=False, range=[-0.1, 8.2]),
        yaxis=dict(visible=False, range=[-0.1, 9.4]),
        margin=dict(l=10, r=10, t=10, b=10)))
    save(fig, "fig2_architecture.png", 820, 620)


def fig_curve(meta):
    """그림 3. 학습 곡선."""
    h = meta["history"]
    ep = [x["epoch"] for x in h]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ep, y=[x["log_loss"] for x in h], mode="lines+markers",
                             line=dict(color=BLUE, width=2),
                             marker=dict(size=5, color=BLUE,
                                         line=dict(width=1.5, color=SURFACE)),
                             name="검증 log-loss"))
    b = int(np.argmin([x["log_loss"] for x in h]))
    fig.add_annotation(x=ep[b], y=h[b]["log_loss"], text=f"최저 {h[b]['log_loss']:.4f} (epoch {ep[b]})",
                       showarrow=True, arrowhead=0, arrowcolor=AXIS, ay=-30,
                       font=dict(size=10, color=INK_2))
    fig.update_layout(**plotly_layout(
        height=320,
        xaxis=dict(showgrid=False, zeroline=False, linecolor=AXIS,
                   title=dict(text="에폭", font=dict(size=11, color=MUTED)),
                   tickfont=dict(color=MUTED, size=10)),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, showline=False,
                   title=dict(text="검증 log-loss", font=dict(size=11, color=MUTED)),
                   tickfont=dict(color=MUTED, size=10)),
        margin=dict(l=60, r=16, t=16, b=46)))
    save(fig, "fig3_curve.png", 760, 320)


def fig_ablation(ab):
    """그림 4. 어블레이션 — full 대비 log-loss 증가량."""
    base = ab["full"]["test"]["log_loss"]
    names = {"full": "제안 모델 (full)", "no_context_skip": "맥락 스킵 연결 제거",
             "seq_len_1": "시퀀스 길이 1 (직전 투구 제거)",
             "no_state_embed": "상태 임베딩 제거", "last_token": "마지막 토큰 풀링",
             "seq_len_3": "시퀀스 길이 3", "seq_len_10": "시퀀스 길이 10"}
    items = [(names.get(k, k), v["test"]["log_loss"] - base)
             for k, v in ab.items() if k != "full"]
    items.sort(key=lambda x: x[1])
    labels = [i[0] for i in items]
    vals = [i[1] for i in items]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=[RED if v > 0 else BLUE for v in vals], line=dict(width=0)),
        text=[f"{v:+.4f}" for v in vals], textposition="outside", cliponaxis=False,
        textfont=dict(size=10, color=INK_2)))
    span = max(abs(min(vals)), abs(max(vals))) * 1.45
    fig.update_layout(**plotly_layout(
        height=300,
        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=INK,
                   zerolinewidth=1.2, showline=False, range=[-span, span],
                   title=dict(text="제안 모델 대비 log-loss 증가량 (오른쪽일수록 성능 악화)",
                              font=dict(size=10.5, color=MUTED)),
                   tickfont=dict(color=MUTED, size=10)),
        yaxis=dict(showgrid=False, zeroline=False, linecolor=AXIS,
                   tickfont=dict(color=INK_2, size=10.5)),
        margin=dict(l=190, r=60, t=12, b=52)))
    save(fig, "fig4_ablation.png", 820, 300)


def fig_calibration(ev):
    """그림 5. 캘리브레이션 곡선."""
    fig = charts.reliability_diagram(ev["prob_pred"], ev["prob_true"], height=340)
    fig.add_annotation(x=0.05, y=0.95, xanchor="left", showarrow=False, align="left",
                       text=(f"ECE {ev['ece']:.4f}<br>"
                             f"temperature {ev.get('temperature', float('nan')):.3f}"),
                       font=dict(size=11, color=INK_2),
                       bgcolor="#ffffff", bordercolor=AXIS, borderwidth=1, borderpad=6)
    save(fig, "fig5_calibration.png", 520, 340)


def fig_confusion(ev):
    """그림 6. 혼동 행렬."""
    fig = charts.confusion_heatmap(ev["confusion"], [KO[o] for o in OUTCOMES], height=460)
    save(fig, "fig6_confusion.png", 700, 460)


def fig_count_value(tables):
    """그림 7. 카운트별 결과 가치."""
    cv = tables.count_value
    counts = [f"{b}-{s}" for b in range(4) for s in range(3)]
    show = ["ball", "called_strike", "swinging_strike", "foul", "field_out",
            "single", "home_run"]
    idx = [OUTCOMES.index(o) for o in show]
    z = cv[:, idx]

    fig = go.Figure(go.Heatmap(
        z=z, x=[KO[o] for o in show], y=counts,
        colorscale=charts.DIVERGING, zmid=0,
        zmin=-float(np.abs(z).max()), zmax=float(np.abs(z).max()),
        xgap=2, ygap=2,
        colorbar=dict(title=dict(text="기대 실점", side="top",
                                 font=dict(size=10, color=MUTED)),
                      thickness=10, len=0.72, outlinewidth=0,
                      tickfont=dict(size=9, color=MUTED))))
    fig.update_layout(**plotly_layout(
        height=380,
        annotations=charts._cell_labels(np.abs(z), [KO[o] for o in show], counts,
                                        fmt="", min_show=1e9),
        xaxis=dict(showgrid=False, zeroline=False, showline=False, side="top",
                   tickfont=dict(color=INK_2, size=10.5)),
        yaxis=dict(showgrid=False, zeroline=False, showline=False,
                   autorange="reversed", tickfont=dict(color=INK_2, size=10.5),
                   title=dict(text="볼-스트라이크", font=dict(size=10.5, color=MUTED))),
        margin=dict(l=76, r=16, t=54, b=12)))
    # 숫자를 직접 얹는다 (셀 밝기 기준이 아니라 값 그대로)
    ann = []
    for i, c in enumerate(counts):
        for j, o in enumerate(show):
            ann.append(dict(x=KO[o], y=c, text=f"{z[i, j]:+.2f}", showarrow=False,
                            font=dict(size=9, color=INK_2)))
    fig.update_layout(annotations=ann)
    save(fig, "fig7_count_value.png", 700, 380)


def fig_zone(profiles):
    """그림 8. 타자 코스별 기대 실점 예시."""
    key = max(profiles.batter_zone, key=lambda k: profiles.batter_zone[k][1].sum())
    grid, cnt = profiles.batter_zone[key]
    d = profiles.directory
    m = d[(d.season == key[0]) & (d.player_id == key[1]) & (d.role == "batter")]
    name = m.name.iat[0] if len(m) else key[1]
    lim = float(np.nanpercentile(np.abs(grid), 97))
    fig = charts.zone_heatmap(grid, profiles.grid_x, profiles.grid_z,
                              title=f"{name} ({key[0]}) — 코스별 기대 실점",
                              counts=cnt, symmetric=lim, height=420)
    save(fig, "fig8_zone.png", 620, 420)
    return name, key[0], int(cnt.sum())


# --------------------------------------------------------------------------
def main():
    print("보고서 그림 렌더")
    M = ROOT / "models"
    meta = json.loads((M / "main" / "metadata.json").read_text(encoding="utf-8"))
    ev = joblib.load(M / "main" / "evaluation.pkl")
    ab = json.loads((M / "ablation.json").read_text(encoding="utf-8"))
    tables = ReferenceTables.load(M / "tables.pkl")
    profiles = Profiles.load(M / "profiles.pkl")

    fig_leak()
    fig_architecture()
    fig_curve(meta)
    fig_ablation(ab)
    fig_calibration(ev)
    fig_confusion(ev)
    fig_count_value(tables)
    who = fig_zone(profiles)
    print(f"\n완료 → {OUT}")
    print(f"  (그림 8 예시 선수: {who[0]}, {who[1]}시즌, {who[2]:,}구)")


if __name__ == "__main__":
    main()
