"""재사용 차트.

모든 차트는 (1) 얇은 마크와 후퇴하는 격자, (2) 값 해석을 색에만 의존하지 않도록
직접 라벨 또는 짝이 되는 표, (3) 마크보다 큰 호버 영역을 갖는다.

스트라이크존 히트맵의 가로축은 '몸쪽 ↔ 바깥쪽' 이다. 데이터 단계에서 좌타자
좌표를 미러링했기 때문에, 좌우 타자를 한 화면에서 같은 의미로 비교할 수 있다.
이전 앱은 우타자 기준 포수 시점으로만 그려서 좌타자 화면이 좌우가 뒤집혀 있었다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .theme import (AXIS, BORDER, CARD, DIVERGING, GRID, INK, INK_2, MUTED,
                    OTHER, SEQUENTIAL, SURFACE, pitch_color, pitch_label,
                    plotly_layout)

ZONE_X = 0.83   # 스트라이크존 반폭 (피트)


def _zone_shapes(color: str = INK, width: float = 1.6) -> list:
    """스트라이크존 테두리와 3×3 보조선."""
    shapes = [dict(type="rect", x0=-ZONE_X, x1=ZONE_X, y0=0.0, y1=1.0,
                   line=dict(color=color, width=width), fillcolor="rgba(0,0,0,0)",
                   layer="above")]
    for x in (-ZONE_X / 3, ZONE_X / 3):
        shapes.append(dict(type="line", x0=x, x1=x, y0=0.0, y1=1.0,
                           line=dict(color=color, width=0.6), layer="above"))
    for z in (1 / 3, 2 / 3):
        shapes.append(dict(type="line", x0=-ZONE_X, x1=ZONE_X, y0=z, y1=z,
                           line=dict(color=color, width=0.6), layer="above"))
    return shapes


def zone_heatmap(grid: np.ndarray, grid_x: np.ndarray, grid_z: np.ndarray,
                 title: str = "", counts: np.ndarray | None = None,
                 symmetric: float | None = None, height: int = 380,
                 unit: str = "기대 실점", reverse_good: bool = False) -> go.Figure:
    """존 격자 히트맵. 부호가 의미를 가지므로 발산형으로 칠한다.

    ``symmetric`` 을 주면 색 범위를 ±그 값으로 고정한다. 여러 차트를 나란히
    비교할 때 색 의미가 흔들리지 않도록 반드시 고정해서 쓴다.
    """
    lim = symmetric if symmetric is not None else float(np.nanmax(np.abs(grid))) or 0.05

    hover = []
    for zi in range(len(grid_z)):
        row = []
        for xi in range(len(grid_x)):
            side = "바깥쪽" if grid_x[xi] > 0.1 else ("몸쪽" if grid_x[xi] < -0.1 else "가운데")
            hgt = "높음" if grid_z[zi] > 0.66 else ("낮음" if grid_z[zi] < 0.33 else "중간")
            n = f"<br>표본 {int(counts[zi, xi]):,}구" if counts is not None else ""
            row.append(f"<b>{side} · {hgt}</b><br>{unit} {grid[zi, xi]:+.4f}{n}")
        hover.append(row)

    fig = go.Figure(go.Heatmap(
        z=grid, x=list(range(len(grid_x))), y=list(range(len(grid_z))),
        colorscale=DIVERGING, zmid=0, zmin=-lim, zmax=lim,
        xgap=2, ygap=2,                    # 셀 사이 2px 표면 간격
        hoverinfo="text", text=hover,
        colorbar=dict(title=dict(text=unit, side="top", font=dict(size=10, color=MUTED)),
                      thickness=10, len=0.72, outlinewidth=0,
                      tickfont=dict(size=10, color=MUTED)),
    ))

    # 존 테두리는 '존 안 셀' 집합을 감싸도록 셀 경계(±0.5)에 맞춘다.
    # 격자점 좌표에 그대로 그리면 테두리가 셀 한가운데를 관통해 읽기 어렵다.
    xin = np.where(np.abs(np.asarray(grid_x)) <= ZONE_X + 1e-9)[0]
    zin = np.where((np.asarray(grid_z) >= -1e-9) & (np.asarray(grid_z) <= 1 + 1e-9))[0]
    shapes = [dict(type="rect",
                   x0=xin.min() - 0.5, x1=xin.max() + 0.5,
                   y0=zin.min() - 0.5, y1=zin.max() + 0.5,
                   line=dict(color=INK, width=1.6), fillcolor="rgba(0,0,0,0)")]

    fig.update_layout(**plotly_layout(
        height=height,
        title=dict(text=title, font=dict(size=13, color=INK), x=0,
                   xanchor="left", xref="paper", yref="container", y=0.97),
        shapes=shapes,
        xaxis=dict(showgrid=False, zeroline=False, showline=False,
                   tickmode="array", tickvals=[0, len(grid_x) // 2, len(grid_x) - 1],
                   ticktext=["몸쪽", "가운데", "바깥쪽"],
                   tickfont=dict(color=MUTED, size=11)),
        yaxis=dict(showgrid=False, zeroline=False, showline=False,
                   tickmode="array", tickvals=[0, len(grid_z) // 2, len(grid_z) - 1],
                   ticktext=["낮음", "중간", "높음"],
                   tickfont=dict(color=MUTED, size=11)),
        margin=dict(l=44, r=8, t=34, b=28),
    ))
    return fig


def zone_table(grid: np.ndarray, grid_x, grid_z, counts=None) -> pd.DataFrame:
    """히트맵의 표 뷰. 색만으로 값을 읽지 않아도 되게 항상 함께 제공한다."""
    rows = []
    for zi in range(len(grid_z) - 1, -1, -1):
        r = {"높이": f"{grid_z[zi]:+.2f}"}
        for xi in range(len(grid_x)):
            key = f"{grid_x[xi]:+.2f}"
            r[key] = round(float(grid[zi, xi]), 4)
            if counts is not None:
                r[key] = f"{grid[zi, xi]:+.4f} (n={int(counts[zi, xi])})"
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def _cell_labels(z: np.ndarray, xs: list, ys: list, fmt="{:.0f}%",
                 min_show: float = 5.0, light_above: float = 0.55,
                 vmax: float | None = None) -> list[dict]:
    """히트맵 셀 위 숫자 라벨.

    plotly 의 ``texttemplate`` 은 셀마다 글자색을 바꿀 수 없어서, 짙은 셀에
    어두운 글씨가 얹혀 읽히지 않는다. 주석으로 직접 그리면서 셀 밝기에 따라
    글자색을 뒤집는다.
    """
    top = vmax if vmax is not None else float(np.nanmax(z)) or 1.0
    out = []
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            v = z[i, j]
            if not np.isfinite(v) or v < min_show:
                continue
            light = (v / top) > light_above
            out.append(dict(x=xv, y=yv, text=fmt.format(v), showarrow=False,
                            font=dict(size=10, color=SURFACE if light else INK_2)))
    return out


def outcome_bars(probs: np.ndarray, outcomes: list[str],
                 height: int = 300) -> go.Figure:
    """결과별 확률. 단일 계열이므로 색은 하나만 쓴다."""
    ko = {"ball": "볼", "called_strike": "루킹 스트라이크",
          "swinging_strike": "헛스윙", "foul": "파울", "hit_by_pitch": "사구",
          "field_out": "인플레이 아웃", "single": "단타", "double": "2루타",
          "triple": "3루타", "home_run": "홈런"}
    order = np.argsort(probs)
    labels = [ko.get(outcomes[i], outcomes[i]) for i in order]
    vals = probs[order] * 100

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color="#2a78d6", line=dict(width=0)),
        text=[f"{v:.1f}%" for v in vals],       # 직접 라벨 (툴팁에만 의존하지 않음)
        textposition="outside", cliponaxis=False,
        textfont=dict(size=11, color=INK_2),
        hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(**plotly_layout(
        height=height,
        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, showline=False,
                   ticksuffix="%", tickfont=dict(color=MUTED, size=11),
                   range=[0, max(vals) * 1.25]),
        yaxis=dict(showgrid=False, zeroline=False, linecolor=AXIS,
                   tickfont=dict(color=INK_2, size=11)),
        margin=dict(l=8, r=40, t=10, b=8),
        bargap=0.28,
    ))
    return fig


def signed_bars(labels: list[str], values: list[float], height: int = 300,
                unit: str = "", title: str = "") -> go.Figure:
    """부호가 의미를 갖는 값의 가로 막대. 발산형 두 극색만 쓴다(크기 램프 아님)."""
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    colors = ["#2a78d6" if v < 0 else "#e34948" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:+.2f}" for v in values],
        textposition="outside", cliponaxis=False,
        textfont=dict(size=11, color=INK_2),
        hovertemplate="%{y}<br>%{x:+.3f} " + unit + "<extra></extra>",
    ))
    span = max(abs(min(values)), abs(max(values))) * 1.35 or 1
    fig.update_layout(**plotly_layout(
        height=height,
        title=dict(text=title, font=dict(size=13, color=INK), x=0,
                   xanchor="left", xref="paper", yref="container", y=0.97),
        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=AXIS,
                   zerolinewidth=1, showline=False, range=[-span, span],
                   tickfont=dict(color=MUTED, size=11)),
        yaxis=dict(showgrid=False, zeroline=False, linecolor=AXIS,
                   tickfont=dict(color=INK_2, size=11)),
        margin=dict(l=8, r=44, t=34, b=8),
        bargap=0.3,
    ))
    return fig


def usage_bars(pitch_types: list[str], usage: list[float],
               height: int = 260, title: str = "") -> go.Figure:
    """구종별 구사율. 색은 구종에 고정 배정되어 있어 필터에도 바뀌지 않는다."""
    order = np.argsort(usage)[::-1]
    pts = [pitch_types[i] for i in order]
    vals = [usage[i] * 100 for i in order]

    fig = go.Figure(go.Bar(
        x=[pitch_label(p) for p in pts], y=vals,
        marker=dict(color=[pitch_color(p) for p in pts], line=dict(width=0)),
        text=[f"{v:.1f}%" for v in vals], textposition="outside",
        cliponaxis=False, textfont=dict(size=11, color=INK_2),
        hovertemplate="%{x}<br>구사율 %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(**plotly_layout(
        height=height,
        title=dict(text=title, font=dict(size=13, color=INK), x=0,
                   xanchor="left", xref="paper", yref="container", y=0.97),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=AXIS,
                   tickfont=dict(color=INK_2, size=11)),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, showline=False,
                   ticksuffix="%", tickfont=dict(color=MUTED, size=11),
                   range=[0, max(vals) * 1.2]),
        margin=dict(l=8, r=8, t=34, b=8), bargap=0.34,
    ))
    return fig


def count_usage_heatmap(pivot: pd.DataFrame, height: int = 330,
                        title: str = "") -> go.Figure:
    """카운트 × 구종 구사율. 크기만 나타내므로 단일 색조 순차형."""
    z = pivot.to_numpy() * 100
    counts = [f"{i//3}-{i%3}" for i in pivot.index]

    fig = go.Figure(go.Heatmap(
        z=z, x=[pitch_label(c) for c in pivot.columns], y=counts,
        colorscale=SEQUENTIAL, xgap=2, ygap=2, zmin=0,
        hovertemplate="카운트 %{y} · %{x}<br>구사율 %{z:.1f}%<extra></extra>",
        colorbar=dict(title=dict(text="구사율", side="top",
                                 font=dict(size=10, color=MUTED)),
                      thickness=10, len=0.72, outlinewidth=0, ticksuffix="%",
                      tickfont=dict(size=10, color=MUTED)),
    ))
    labels = _cell_labels(z, [pitch_label(c) for c in pivot.columns], counts,
                          vmax=float(np.nanmax(z)) or 100.0)
    fig.update_layout(**plotly_layout(
        height=height,
        title=dict(text=title, font=dict(size=13, color=INK), x=0,
                   xanchor="left", xref="paper", yref="container", y=0.97),
        xaxis=dict(showgrid=False, zeroline=False, showline=False, side="top",
                   tickfont=dict(color=INK_2, size=11)),
        yaxis=dict(showgrid=False, zeroline=False, showline=False, autorange="reversed",
                   tickfont=dict(color=INK_2, size=11)),
        margin=dict(l=42, r=8, t=58, b=8),
        annotations=labels,
    ))
    return fig


def movement_scatter(rep: pd.DataFrame, height: int = 360,
                     title: str = "") -> go.Figure:
    """구종별 무브먼트 지도. 색만으로 구분하지 않도록 전 구종에 직접 라벨을 단다.

    (범주 8색은 인접쌍 기준으로 검증됐지만, 산점도는 모든 쌍이 동시에 보이므로
    색 분리에만 기대면 안 된다. 라벨이 그 보조 부호 역할을 한다.)
    """
    pts = [(r["pfx_x"] * 12, r["pfx_z"] * 12, r) for _, r in rep.iterrows()]

    # 라벨 충돌 회피. 이미 놓인 라벨과 가까우면 다음 위치로 돌린다.
    # (구종이 적어 완전 배치가 가능하므로 단순 탐욕법으로 충분하다)
    span_x = max(1e-6, max(p[0] for p in pts) - min(p[0] for p in pts))
    span_y = max(1e-6, max(p[1] for p in pts) - min(p[1] for p in pts))
    POSITIONS = ["top center", "bottom center", "middle right", "middle left"]
    placed: list[tuple[float, float, str]] = []

    def _place(x, y):
        for pos in POSITIONS:
            if all(abs(x - px) / span_x > 0.16 or abs(y - py) / span_y > 0.10
                   or pos != ppos for px, py, ppos in placed):
                placed.append((x, y, pos))
                return pos
        placed.append((x, y, POSITIONS[0]))
        return POSITIONS[0]

    fig = go.Figure()
    for x, y, r in pts:
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=max(12, min(38, r["usage"] * 110)),
                        color=pitch_color(r["pitch_type"]), opacity=0.85,
                        line=dict(width=2, color=SURFACE)),   # 겹침 대비 표면 링
            text=[pitch_label(r["pitch_type"])], textposition=_place(x, y),
            textfont=dict(size=10, color=INK_2),
            hovertemplate=(f"<b>{pitch_label(r['pitch_type'])}</b><br>"
                           f"구속 {r['speed']:.1f} mph<br>"
                           f"회전 {r['spin_rate']:.0f} rpm<br>"
                           f"수평 {r['pfx_x']*12:+.1f} in · 수직 {r['pfx_z']*12:+.1f} in<br>"
                           f"구사율 {r['usage']*100:.1f}%<extra></extra>"),
        ))
    fig.update_layout(**plotly_layout(
        height=height,
        title=dict(text=title, font=dict(size=13, color=INK), x=0,
                   xanchor="left", xref="paper", yref="container", y=0.97),
        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=AXIS,
                   showline=False, title=dict(text="수평 무브먼트 (인치, 포수 시점)",
                                              font=dict(size=11, color=MUTED)),
                   tickfont=dict(color=MUTED, size=10)),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=AXIS,
                   showline=False, title=dict(text="수직 무브먼트 (인치)",
                                              font=dict(size=11, color=MUTED)),
                   tickfont=dict(color=MUTED, size=10)),
        margin=dict(l=66, r=28, t=38, b=48),
    ))
    return fig


def reliability_diagram(prob_true, prob_pred, height: int = 320) -> go.Figure:
    """캘리브레이션 곡선. 대각선에 붙을수록 확률이 정직하다."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color=AXIS, width=1),
                             hoverinfo="skip", name="완전 보정"))
    fig.add_trace(go.Scatter(
        x=prob_pred, y=prob_true, mode="lines+markers",
        line=dict(color="#2a78d6", width=2),
        marker=dict(size=8, color="#2a78d6", line=dict(width=2, color=SURFACE)),
        hovertemplate="예측 %{x:.3f}<br>실제 %{y:.3f}<extra></extra>", name="모델",
    ))
    fig.update_layout(**plotly_layout(
        height=height,
        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, showline=False,
                   title=dict(text="예측 확률", font=dict(size=11, color=MUTED)),
                   range=[0, 1], tickfont=dict(color=MUTED, size=10)),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, showline=False,
                   title=dict(text="실제 빈도", font=dict(size=11, color=MUTED)),
                   range=[0, 1], tickfont=dict(color=MUTED, size=10)),
        margin=dict(l=52, r=8, t=16, b=44),
    ))
    return fig


def confusion_heatmap(cm: np.ndarray, labels: list[str],
                      height: int = 420) -> go.Figure:
    """혼동 행렬 (행 정규화). 크기만 나타내므로 순차형."""
    with np.errstate(invalid="ignore"):
        norm = cm / cm.sum(axis=1, keepdims=True)
    norm = np.nan_to_num(norm)

    fig = go.Figure(go.Heatmap(
        z=norm * 100, x=labels, y=labels, colorscale=SEQUENTIAL, zmin=0, zmax=100,
        xgap=2, ygap=2,
        hovertemplate="실제 %{y} → 예측 %{x}<br>%{z:.1f}%<extra></extra>",
        colorbar=dict(thickness=10, len=0.75, outlinewidth=0, ticksuffix="%",
                      tickfont=dict(size=10, color=MUTED)),
    ))
    labels = _cell_labels(norm * 100, labels, labels, fmt="{:.0f}", min_show=2.0,
                          vmax=100.0)
    fig.update_layout(**plotly_layout(
        height=height,
        annotations=labels,
        xaxis=dict(showgrid=False, zeroline=False, showline=False, side="top",
                   title=dict(text="예측", font=dict(size=11, color=MUTED)),
                   tickangle=-40, tickfont=dict(color=INK_2, size=10)),
        yaxis=dict(showgrid=False, zeroline=False, showline=False, autorange="reversed",
                   title=dict(text="실제", font=dict(size=11, color=MUTED)),
                   tickfont=dict(color=INK_2, size=10)),
        margin=dict(l=110, r=8, t=80, b=8),
    ))
    return fig
