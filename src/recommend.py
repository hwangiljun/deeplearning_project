"""투구 추천 엔진.

주어진 상황에서 (구종 × 코스) 후보를 만들고, 각 후보에 대해 모델이 낸 결과
확률 분포를 **기대 실점 변화량**으로 환산해 순위를 매긴다.

이전 앱과 달라진 점
-------------------
* **점수 함수**: ``p_success - 1.5*p_fail - penalty`` 였다. good/bad 두 집합이
  전체를 정확히 이분하므로 ``p_fail = 1 - p_success`` 였고, 결국
  ``2.5*p_success - 1.5`` 라서 가중치 1.5 도, 시커에만 붙던 0.05 페널티도
  거의 의미가 없었다. → ``Σ P(결과) × 기대실점(카운트, 결과)`` 로 교체.
  카운트를 반영하고, 단위가 '실점' 이라 해석도 된다.
* **후보 범위**: 존 안 9칸뿐이라 유인구를 추천할 수 없었다. → 존 밖까지 격자 확장.
* **구위**: 구종별 리그 평균 상수를 썼다. → 투수×구종 실제 평균.
* **시퀀스**: 후보 한 구를 8번 복사해 넣었다(학습은 실제 8구 시퀀스였으므로
  train/serve 불일치). → 실제 타석 이력을 앞에 붙이고 후보를 마지막에 둔다.
* **제구**: 정확히 한 칸을 찍어 추천했다. → Takamido & Nakamoto (2026) 의
  command window 를 적용해, 주변 칸까지 평균낸 '제구 오차에 강건한 영역'으로
  평가한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .data import (CAT_FEATURES, CONTEXT_FEATURES, OUTCOMES, PITCH_FEATURES,
                   build_context_features, build_pitch_features, build_sequences)
from .features import ReferenceTables, in_strike_zone
from .prepare import Encoders, apply_encoders


@dataclass
class GameState:
    """추천을 요청하는 시점의 상황."""

    batter: str
    pitcher: str
    stand: str = "R"
    p_throws: str = "R"
    balls: int = 0
    strikes: int = 0
    outs: int = 0
    on_1b: bool = False
    on_2b: bool = False
    on_3b: bool = False
    inning: int = 1
    is_bottom: bool = False
    score_diff: int = 0          # 투수 팀 기준
    thru_order: int = 1
    season: int = 2025
    # 이번 타석에서 이미 던진 공들 (오래된 것부터). 각 원소는
    # {"pitch_type", "loc_x", "loc_z"} 를 갖는 dict.
    history: list = field(default_factory=list)


@dataclass
class Candidate:
    pitch_type: str
    loc_x: float
    loc_z: float
    run_value: float        # 기대 실점 변화량 (낮을수록 투수에게 유리)
    window_value: float     # command window 평균 (실제 순위 기준)
    probs: np.ndarray       # (10,) 결과별 확률
    in_zone: bool

    def summary(self, top_n: int = 3) -> list[tuple[str, float]]:
        order = np.argsort(-self.probs)[:top_n]
        return [(OUTCOMES[i], float(self.probs[i])) for i in order]


# --------------------------------------------------------------------------
def available_pitches(tables: ReferenceTables, pitcher: str,
                      min_usage: float = 0.03) -> pd.DataFrame:
    """이 투수가 실제로 던지는 구종과 그 평균 물리량.

    기록이 없는 투수(신인·미등록)는 리그 평균 레퍼토리로 대체한다. 이전 코드처럼
    조용히 0번 투수로 바꿔치기하지 않는다.
    """
    rep = tables.repertoire
    mine = rep[(rep["pitcher"].astype(str) == str(pitcher))
               & (rep["usage"] >= min_usage)]
    if len(mine) >= 2:
        return mine.copy()

    lg = tables.league_repertoire
    lg = lg[lg["usage"] >= 0.02].copy()
    lg["pitcher"] = str(pitcher)
    return lg


def candidate_grid(tables: ReferenceTables, include_out_of_zone: bool = True):
    """(loc_x, loc_z) 후보 격자. 존 밖을 포함하면 유인구도 추천 대상이 된다."""
    xs, zs = tables.grid_x, tables.grid_z
    pts = [(x, z) for z in zs for x in xs]
    if not include_out_of_zone:
        pts = [(x, z) for x, z in pts if in_strike_zone(x, z)]
    return np.array(pts, dtype=np.float32)


# --------------------------------------------------------------------------
def _build_frame(state: GameState, cands: pd.DataFrame) -> pd.DataFrame:
    """후보마다 '가상의 타석' 을 만들어 하나의 DataFrame 으로 쌓는다.

    학습과 완전히 같은 전처리 함수를 태우기 위해서다. 후보 i 의 타석은
    ``at_bat_number = i`` 를 갖고, 이번 타석의 실제 이력 뒤에 후보 투구가
    마지막 행으로 붙는다. ``build_sequences`` 가 타석 경계를 지키므로
    후보끼리 섞이지 않는다.
    """
    hist = state.history
    rows = []
    for i, c in enumerate(cands.itertuples(index=False)):
        for j, h in enumerate(hist):
            rows.append({
                "at_bat_number": i, "pitch_number": j + 1,
                "pitch_type": h["pitch_type"],
                "plate_x": h["loc_x"] * (-1.0 if state.stand == "L" else 1.0),
                "plate_z_norm": h["loc_z"],
                "speed": h.get("speed", np.nan),
                "spin_rate": h.get("spin_rate", np.nan),
                "spin_sin": h.get("spin_sin", np.nan),
                "spin_cos": h.get("spin_cos", np.nan),
                "pfx_x": h.get("pfx_x", np.nan),
                "pfx_z": h.get("pfx_z", np.nan),
            })
        rows.append({
            "at_bat_number": i, "pitch_number": len(hist) + 1,
            "pitch_type": c.pitch_type,
            "plate_x": c.loc_x * (-1.0 if state.stand == "L" else 1.0),
            "plate_z_norm": c.loc_z,
            "speed": c.speed, "spin_rate": c.spin_rate,
            "spin_sin": c.spin_sin, "spin_cos": c.spin_cos,
            "pfx_x": c.pfx_x, "pfx_z": c.pfx_z,
        })

    df = pd.DataFrame(rows)
    # 미러링은 build_pitch_features 가 다시 하므로 원본 좌표계로 되돌려 둔다.
    df["loc_x"] = df["plate_x"] * (-1.0 if state.stand == "L" else 1.0)
    df["loc_z"] = df["plate_z_norm"]

    # 상황 정보는 모든 행에 동일하게 채운다 (타깃 시점 기준).
    df["game_pk"] = 0
    df["batter"] = str(state.batter)
    df["pitcher"] = str(state.pitcher)
    df["stand"] = state.stand
    df["p_throws"] = state.p_throws
    df["balls"] = state.balls
    df["strikes"] = state.strikes
    df["outs_when_up"] = state.outs
    df["on_1b"] = 1.0 if state.on_1b else np.nan
    df["on_2b"] = 1.0 if state.on_2b else np.nan
    df["on_3b"] = 1.0 if state.on_3b else np.nan
    df["inning"] = state.inning
    df["inning_topbot"] = "Bot" if state.is_bottom else "Top"
    df["bat_score"] = 0
    df["fld_score"] = state.score_diff
    df["n_thruorder_pitcher"] = state.thru_order
    df["season"] = state.season
    df["delta_run_exp"] = 0.0
    df["outcome_idx"] = 0
    df["game_date"] = pd.Timestamp(f"{state.season}-06-01")
    return df


@torch.no_grad()
def score_candidates(model, enc: Encoders, tables: ReferenceTables,
                     state: GameState, batter_stats: pd.DataFrame | None = None,
                     include_out_of_zone: bool = True,
                     device: str = "cpu") -> list[Candidate]:
    """모든 (구종 × 코스) 후보의 기대 실점 변화량을 계산한다."""
    rep = available_pitches(tables, state.pitcher)
    grid = candidate_grid(tables, include_out_of_zone)

    # 구종 × 코스 데카르트 곱
    cands = rep.loc[rep.index.repeat(len(grid))].reset_index(drop=True)
    cands["loc_x"] = np.tile(grid[:, 0], len(rep))
    cands["loc_z"] = np.tile(grid[:, 1], len(rep))

    df = _build_frame(state, cands)

    # ---- 학습과 동일한 전처리 ----
    # loc_x/loc_z 는 이미 만들어 뒀으므로 build_pitch_features 가 덮어쓰지 않도록
    # 필요한 원본 열만 채워서 통과시킨다.
    df["sz_bot"], df["sz_top"] = 0.0, 1.0          # loc_z 가 이미 정규화 값
    df["plate_z"] = df["plate_z_norm"]
    df["release_speed"] = df["speed"]
    df["effective_speed"] = df["speed"]
    df["release_spin_rate"] = df["spin_rate"]
    # pfx_x 는 이미 원본(미러링 전)이다. 타자별 미러링은 build_pitch_features 가 한다.
    df["spin_axis"] = np.degrees(np.arctan2(df["spin_sin"], df["spin_cos"])) % 360

    df = build_pitch_features(df)
    df = build_context_features(df, batter_stats)
    df = apply_encoders(df, enc)

    seq = build_sequences(df, seq_len=model.cfg.seq_len)

    # 각 가상 타석의 마지막 행(=후보 투구)만 남긴다
    last = df.groupby("at_bat_number")["pitch_number"].transform("max")
    keep = np.where(df["pitch_number"].to_numpy() == last.to_numpy())[0]

    model.eval()
    logits, _ = model(
        torch.from_numpy(seq.cat[keep]).to(device),
        torch.from_numpy(seq.num[keep]).to(device),
        torch.from_numpy(seq.mask[keep]).to(device),
        torch.from_numpy(seq.ctx[keep]).to(device),
        torch.from_numpy(seq.state[keep]).to(device),
    )
    probs = torch.softmax(logits, dim=1).cpu().numpy()

    # 기대 실점 = Σ P(결과) × 카운트별 결과 가치
    count_state = min(state.balls, 3) * 3 + min(state.strikes, 2)
    values = tables.count_value[count_state]          # (10,)
    run_values = probs @ values

    out = []
    for i, c in enumerate(cands.itertuples(index=False)):
        out.append(Candidate(
            pitch_type=c.pitch_type,
            loc_x=float(c.loc_x), loc_z=float(c.loc_z),
            run_value=float(run_values[i]), window_value=float(run_values[i]),
            probs=probs[i],
            in_zone=bool(in_strike_zone(c.loc_x, c.loc_z)),
        ))
    return out


# --------------------------------------------------------------------------
def apply_command_window(cands: list[Candidate], tables: ReferenceTables,
                         window: int = 3) -> list[Candidate]:
    """제구 오차를 반영해 주변 칸까지 평균낸 값으로 순위를 매긴다.

    Takamido & Nakamoto (2026) 의 command window. 투수는 점 하나를 정확히
    찍을 수 없으므로, 노린 지점 주변 window×window 칸의 평균 결과를 그 지점의
    대표값으로 삼는다. window 가 클수록 제구가 나쁜 투수를 가정한 것이다.

    이전 앱은 9칸 중 정확히 한 칸을 추천해, 옆 칸이 실투 구간이어도 알 수 없었다.
    """
    gx = np.asarray(tables.grid_x, dtype=np.float64)
    gz = np.asarray(tables.grid_z, dtype=np.float64)
    nx, nz = len(gx), len(gz)

    # 격자 좌표는 float32 를 거치며 미세하게 어긋나므로 최근접 격자점을 찾는다.
    def _ix(v: float) -> int:
        return int(np.argmin(np.abs(gx - v)))

    def _iz(v: float) -> int:
        return int(np.argmin(np.abs(gz - v)))

    by_pitch: dict[str, np.ndarray] = {}
    for c in cands:
        g = by_pitch.setdefault(c.pitch_type, np.full((nz, nx), np.nan, np.float32))
        g[_iz(c.loc_z), _ix(c.loc_x)] = c.run_value

    half = window // 2
    smoothed = {}
    for pt, g in by_pitch.items():
        out = np.full_like(g, np.nan)
        for z in range(nz):
            for x in range(nx):
                z0, z1 = max(0, z - half), min(nz, z + half + 1)
                x0, x1 = max(0, x - half), min(nx, x + half + 1)
                patch = g[z0:z1, x0:x1]
                out[z, x] = np.nanmean(patch)
        smoothed[pt] = out

    for c in cands:
        c.window_value = float(smoothed[c.pitch_type][_iz(c.loc_z), _ix(c.loc_x)])
    return cands


def top_k(cands: list[Candidate], k: int = 3, min_separation: float = 0.30) -> list[Candidate]:
    """상위 k개. 서로 너무 가까운 코스는 하나만 남겨 다양성을 확보한다."""
    ordered = sorted(cands, key=lambda c: c.window_value)
    picked: list[Candidate] = []
    for c in ordered:
        if any(c.pitch_type == p.pitch_type
               and abs(c.loc_x - p.loc_x) < min_separation
               and abs(c.loc_z - p.loc_z) < min_separation for p in picked):
            continue
        picked.append(c)
        if len(picked) == k:
            break
    return picked
