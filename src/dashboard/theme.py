"""대시보드 색·타이포·Plotly 템플릿.

색은 역할로만 참조한다. 하드코딩된 헥스가 화면 곳곳에 흩어지면 나중에
바꿀 수가 없다.

설계 규칙
---------
* **구종 색은 구종에 고정**된다. 필터로 구종이 빠져도 남은 구종의 색은
  그대로다. 순위나 등장 순서로 색을 배정하지 않는다.
* **기대 실점처럼 부호가 의미를 갖는 값은 발산형(diverging)** 으로 칠한다.
  파랑(투수 유리) ↔ 회색(중립) ↔ 빨강(타자 유리). 중간값이 '아무것도 아님'
  으로 읽혀야 하므로 중앙은 유채색이 아닌 회색이다.
* **크기만 나타내는 값은 단일 색조 순차형**(파랑 연→진)이다. 무지개 램프는
  쓰지 않는다.
* 축과 격자는 표면에서 한 단계만 떨어진 얇은 실선이다. 점선은 쓰지 않는다.

팔레트는 dataviz 검증기로 확인했다(light, surface #fcfcfb):
명도대·채도·색각 분리·정상시야 분리 모두 통과. 대비 경고가 있는 세 색은
"직접 라벨 또는 표 뷰" 로 완화하므로, 모든 차트에 라벨과 표를 함께 제공한다.
"""

from __future__ import annotations

# ---------------------------------------------------------------- 표면과 잉크
SURFACE = "#fcfcfb"       # 차트 표면
PAGE = "#f9f9f7"          # 페이지 바탕
CARD = "#ffffff"
INK = "#0b0b0b"           # 본문
INK_2 = "#52514e"         # 보조
MUTED = "#898781"         # 축·라벨
GRID = "#e1e0d9"          # 격자 (얇은 실선)
AXIS = "#c3c2b7"
BORDER = "rgba(11,11,11,0.10)"

# ---------------------------------------------------------------- 범주형 8슬롯
# dataviz 기본 순서. 이 순서 자체가 색각 안전성 장치라 임의로 바꾸지 않는다.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
OTHER = "#898781"

# 구종 → 색 슬롯 고정 배정. 구사 빈도 상위 8종에 1~8번을 준다.
PITCH_COLOR = {
    "FF": CATEGORICAL[0],  # 포심
    "SI": CATEGORICAL[1],  # 싱커
    "SL": CATEGORICAL[2],  # 슬라이더
    "CH": CATEGORICAL[3],  # 체인지업
    "FC": CATEGORICAL[4],  # 커터
    "ST": CATEGORICAL[5],  # 스위퍼
    "CU": CATEGORICAL[6],  # 커브
    "FS": CATEGORICAL[7],  # 스플리터
}
PITCH_NAME_KO = {
    "FF": "포심", "SI": "싱커", "SL": "슬라이더", "CH": "체인지업",
    "FC": "커터", "ST": "스위퍼", "CU": "커브", "FS": "스플리터",
    "KC": "너클커브", "SV": "슬러브", "KN": "너클볼", "EP": "이퍼스",
    "FA": "패스트볼", "FO": "포크", "CS": "슬로커브", "SC": "스크류",
}


def pitch_color(code: str) -> str:
    """구종 코드 → 고정 색. 미배정 구종은 중립 회색."""
    return PITCH_COLOR.get(code, OTHER)


def pitch_label(code: str) -> str:
    ko = PITCH_NAME_KO.get(code)
    return f"{code} {ko}" if ko else str(code)


# ---------------------------------------------------------------- 상태색
STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}
SUCCESS_TEXT = "#006300"

# ---------------------------------------------------------------- 연속형 스케일
# 순차형: 파랑 한 색조, 연 → 진
SEQUENTIAL = [
    [0.00, "#cde2fb"], [0.20, "#9ec5f4"], [0.40, "#6da7ec"],
    [0.60, "#3987e5"], [0.80, "#256abf"], [1.00, "#0d366b"],
]

# 발산형: 파랑(투수 유리, 음수) ↔ 중립 회색 ↔ 빨강(타자 유리, 양수).
# 중앙이 회색이라 '변화 없음' 이 색으로도 읽힌다.
DIVERGING = [
    [0.00, "#0d366b"], [0.18, "#2a78d6"], [0.38, "#9ec5f4"],
    [0.50, "#f0efec"],
    [0.62, "#f3b9b8"], [0.82, "#e34948"], [1.00, "#8f2020"],
]

FONT = 'system-ui, -apple-system, "Segoe UI", "Malgun Gothic", sans-serif'


def plotly_layout(height: int = 340, **kw) -> dict:
    """모든 차트가 공유하는 레이아웃. 얇은 마크, 후퇴하는 크롬."""
    base = dict(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK_2),
        margin=dict(l=8, r=8, t=28, b=8),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=AXIS,
                   linewidth=1, tickfont=dict(color=MUTED, size=11)),
        yaxis=dict(gridcolor=GRID, gridwidth=1, zeroline=False,
                   linecolor=AXIS, linewidth=1,
                   tickfont=dict(color=MUTED, size=11)),
        hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER,
                        font=dict(family=FONT, size=12, color=INK)),
        showlegend=False,
    )
    base.update(kw)
    return base


# ---------------------------------------------------------------- 페이지 CSS
CSS = f"""
<style>
  .stApp {{ background: {PAGE}; }}
  html, body, [class*="css"] {{ font-family: {FONT}; }}

  /* 상단 여백 축소 — 화면을 데이터에 쓴다 */
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }}

  h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.01em; }}
  h1 {{ font-size: 1.6rem; font-weight: 650; }}
  h2 {{ font-size: 1.15rem; font-weight: 620; margin-top: 0.4rem; }}
  h3 {{ font-size: 0.98rem; font-weight: 600; }}

  /* 카드 */
  .card {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 14px 16px; margin-bottom: 10px;
  }}

  /* 지표 타일 */
  .tiles {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; }}
  .tile {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 12px 16px; min-width: 116px; flex: 1;
  }}
  .tile .k {{ font-size: 0.72rem; color: {MUTED}; text-transform: uppercase;
              letter-spacing: 0.04em; margin-bottom: 4px; }}
  .tile .v {{ font-size: 1.5rem; font-weight: 620; color: {INK}; line-height: 1.1; }}
  .tile .s {{ font-size: 0.75rem; color: {INK_2}; margin-top: 2px; }}

  /* 추천 카드 */
  .rec {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 12px 14px; margin-bottom: 8px; border-left: 3px solid var(--accent);
  }}
  .rec .rank {{ font-size: 0.72rem; color: {MUTED}; letter-spacing: 0.06em; }}
  .rec .name {{ font-size: 1.05rem; font-weight: 620; color: {INK}; }}
  .rec .loc {{ font-size: 0.86rem; color: {INK_2}; margin-top: 1px; }}
  .rec .val {{ font-size: 1.24rem; font-weight: 620; }}
  .rec .sub {{ font-size: 0.74rem; color: {MUTED}; }}

  /* 표 */
  .stDataFrame {{ border: 1px solid {BORDER}; border-radius: 8px; }}
  [data-testid="stMetricValue"] {{ font-size: 1.4rem; }}

  /* 사이드바 */
  section[data-testid="stSidebar"] {{ background: {CARD};
      border-right: 1px solid {BORDER}; }}

  .caption {{ font-size: 0.78rem; color: {MUTED}; line-height: 1.5; }}
  .sep {{ height: 1px; background: {GRID}; margin: 14px 0; }}
</style>
"""


def tile(label: str, value: str, sub: str = "") -> str:
    s = f'<div class="s">{sub}</div>' if sub else ""
    return f'<div class="tile"><div class="k">{label}</div><div class="v">{value}</div>{s}</div>'


def tiles(items: list[tuple[str, str, str]]) -> str:
    """(라벨, 값, 보조설명) 목록 → 지표 타일 행."""
    return '<div class="tiles">' + "".join(tile(*i) for i in items) + "</div>"


def value_color(v: float, invert: bool = False) -> str:
    """부호에 따른 텍스트 색. 투수 관점에서 음수(실점 감소)가 좋다."""
    good = v < 0 if not invert else v > 0
    return SUCCESS_TEXT if good else STATUS["critical"]
