"""추천에 필요한 참조 테이블을 데이터에서 만든다.

이전 앱은 이 세 가지를 전부 상수로 박아놨다.

* ``get_pitch_specs()`` 가 구종별 리그 평균 상수를 돌려줬다. 투수를 골라도
  슬라이더는 항상 84.5마일/2450rpm 이었다. → 투수×구종 실제 평균으로 교체.
* 결과를 good/bad 로 이분해 ``p_success - 1.5*p_fail`` 로 점수를 냈다.
  두 집합이 전체를 정확히 이분하므로 ``p_fail = 1 - p_success`` 였고, 결국
  점수는 ``2.5*p_success - 1.5`` 라서 **가중치 1.5 가 순위에 아무 영향이 없었다.**
  → 결과별·카운트별 기대 실점 변화량(delta_run_exp)으로 교체.
* 존 안 9칸만 후보로 삼아 유인구를 추천할 수 없었다. → 존 밖까지 격자 확장.

위치별 가치 테이블은 Takamido & Nakamoto (2026) 의 SLG 페널티를 일반화한
것이다. 논문은 인플레이 확률에 위치별 SLG 를 더해 "헛스윙은 잘 나오지만
장타 위험이 큰 코스"에 벌점을 줬다. 여기서는 SLG 대신 실제 기대 실점
변화량을 쓰므로 페널티를 따로 더할 필요 없이 목적함수에 이미 반영된다.
(비교를 위해 SLG 테이블도 같이 만든다.)

주의: 이 테이블들은 **표준화 전** 원본 스케일의 loc_x / loc_z 를 쓴다.
``build_pitch_features()`` 직후, ``apply_encoders()`` 전에 호출할 것.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import OUTCOMES, OUTCOME_TO_IDX

# 구속 3분위. 논문의 임계값과 같다 (학습 데이터 3분위 경계에서 유도됨).
VELOCITY_BANDS = [(-np.inf, 86.7, "low"), (86.7, 92.5, "mid"), (92.5, np.inf, "high")]
BAND_NAMES = ["low", "mid", "high"]

# 후보 격자. loc_x 는 피트(미러링 후 + = 바깥쪽), loc_z 는 타자별 존 정규화 높이.
# 스트라이크존은 loc_x 약 ±0.83ft, loc_z 0~1 이므로 격자는 존 밖까지 덮는다.
GRID_X = np.array([-1.15, -0.83, -0.55, -0.28, 0.0, 0.28, 0.55, 0.83, 1.15])
GRID_Z = np.array([-0.30, 0.0, 0.17, 0.33, 0.50, 0.67, 0.83, 1.0, 1.30])


def velocity_band(speed: np.ndarray | pd.Series) -> np.ndarray:
    """구속 → 0(low) / 1(mid) / 2(high)."""
    s = np.asarray(speed, dtype=float)
    return np.where(s < 86.7, 0, np.where(s < 92.5, 1, 2)).astype(np.int64)


def in_strike_zone(loc_x, loc_z) -> np.ndarray:
    """스트라이크존 안인가. 존 폭은 홈플레이트 + 공 반지름 ≈ ±0.83ft."""
    return (np.abs(loc_x) <= 0.83) & (loc_z >= 0.0) & (loc_z <= 1.0)


# --------------------------------------------------------------------------
@dataclass
class ReferenceTables:
    """추천기가 참조하는 데이터 유래 테이블 묶음."""

    repertoire: pd.DataFrame     # (pitcher, pitch_type) → 평균 물리량 + 구사율
    league_repertoire: pd.DataFrame  # pitch_type → 리그 평균 (미등록 투수 대비)
    count_value: np.ndarray      # (12, 10) 카운트상태 × 결과 → 평균 기대실점
    location_value: np.ndarray   # (3, nz, nx) 구속밴드 × 격자 → 평균 기대실점
    location_slg: np.ndarray     # (3, nz, nx) 구속밴드 × 격자 → 평균 SLG
    grid_x: np.ndarray
    grid_z: np.ndarray
    seasons: list

    def save(self, path) -> Path:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path) -> "ReferenceTables":
        import joblib
        return joblib.load(path)


# --------------------------------------------------------------------------
def build_repertoire(df: pd.DataFrame, min_pitches: int = 25) -> pd.DataFrame:
    """투수×구종별 실제 평균 물리량과 구사율.

    반사실 분석에서 "이 투수가 이 구종을 던진다면" 의 물리량으로 쓰인다.
    ``min_pitches`` 미만으로 던진 구종은 그 투수의 레퍼토리로 보지 않는다
    (한두 번 실험한 구종을 추천하면 현장에서 못 쓴다).
    """
    # 무브먼트는 **미러링 전 원본**(pfx_x)으로 집계한다. mov_x 는 타자 좌우에
    # 따라 부호가 뒤집히므로 좌우 타자를 섞어 평균내면 상쇄되어 무의미해진다.
    # (예: Pablo López 체인지업 실제 -16.1in → 미러링 평균 +6.2in)
    cols = ["speed", "spin_rate", "spin_sin", "spin_cos", "pfx_x", "pfx_z"]
    g = df.groupby(["pitcher", "pitch_type"], observed=True)

    rep = g[cols].mean()
    rep["n"] = g.size()
    rep = rep.reset_index()

    total = rep.groupby("pitcher")["n"].transform("sum")
    rep["usage"] = rep["n"] / total
    rep = rep[rep["n"] >= min_pitches].reset_index(drop=True)
    rep["velocity_band"] = velocity_band(rep["speed"])
    return rep


def build_league_repertoire(df: pd.DataFrame) -> pd.DataFrame:
    """구종별 리그 평균. 처음 보는 투수를 골랐을 때의 대체값."""
    cols = ["speed", "spin_rate", "spin_sin", "spin_cos", "pfx_x", "pfx_z"]
    g = df.groupby("pitch_type", observed=True)
    rep = g[cols].mean()
    rep["n"] = g.size()
    rep = rep.reset_index()
    rep["usage"] = rep["n"] / rep["n"].sum()
    rep["velocity_band"] = velocity_band(rep["speed"])
    return rep


def build_count_value(df: pd.DataFrame) -> np.ndarray:
    """(카운트상태 12, 결과 10) → 평균 기대 실점 변화량.

    같은 '볼' 이라도 0-0 에서는 거의 무해하지만 3-2 에서는 볼넷이다.
    이전 점수 함수는 이 차이를 전혀 반영하지 못했다.
    투수는 이 값을 **최소화**한다 (양수 = 타자에게 유리).
    """
    table = np.full((12, len(OUTCOMES)), np.nan, dtype=np.float32)

    grp = df.groupby(["count_state", "outcome_idx"], observed=True)["delta_run_exp"].mean()
    for (cs, oi), v in grp.items():
        if 0 <= cs < 12 and 0 <= oi < len(OUTCOMES):
            table[int(cs), int(oi)] = v

    # 관측이 없는 칸(예: 3-2 에서의 hit_by_pitch 조합 일부)은 결과별 전체 평균으로
    overall = df.groupby("outcome_idx", observed=True)["delta_run_exp"].mean()
    for oi, v in overall.items():
        col = table[:, int(oi)]
        col[np.isnan(col)] = v
    table[np.isnan(table)] = 0.0
    return table


def _grid_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """연속값을 격자 셀 인덱스로. 격자점 사이 중점을 경계로 쓴다."""
    bounds = (edges[:-1] + edges[1:]) / 2
    return np.clip(np.digitize(values, bounds), 0, len(edges) - 1)


def build_location_tables(df: pd.DataFrame,
                          grid_x: np.ndarray = GRID_X,
                          grid_z: np.ndarray = GRID_Z):
    """(구속밴드 3, 높이칸, 좌우칸) → 평균 기대실점 / 평균 SLG.

    논문 Figure 3(a) 의 '구속밴드별 위치 SLG' 에 해당한다. 우리는 여기에
    기대 실점 변화량 판을 하나 더 만든다. 앱의 히트맵 배경으로도 쓴다.
    """
    d = df.dropna(subset=["loc_x", "loc_z", "speed"]).copy()

    band = velocity_band(d["speed"])
    xi = _grid_index(d["loc_x"].to_numpy(), grid_x)
    zi = _grid_index(d["loc_z"].to_numpy(), grid_z)

    # 루타수(SLG 분자). 인플레이 결과에만 값이 있다.
    tb = d["outcome_idx"].map({
        OUTCOME_TO_IDX["single"]: 1.0,
        OUTCOME_TO_IDX["double"]: 2.0,
        OUTCOME_TO_IDX["triple"]: 3.0,
        OUTCOME_TO_IDX["home_run"]: 4.0,
        OUTCOME_TO_IDX["field_out"]: 0.0,
    })
    is_ab = tb.notna()

    shape = (3, len(grid_z), len(grid_x))
    run_sum = np.zeros(shape, np.float64)
    run_cnt = np.zeros(shape, np.float64)
    tb_sum = np.zeros(shape, np.float64)
    tb_cnt = np.zeros(shape, np.float64)

    np.add.at(run_sum, (band, zi, xi), d["delta_run_exp"].fillna(0.0).to_numpy())
    np.add.at(run_cnt, (band, zi, xi), 1.0)
    m = is_ab.to_numpy()
    np.add.at(tb_sum, (band[m], zi[m], xi[m]), tb[m].to_numpy())
    np.add.at(tb_cnt, (band[m], zi[m], xi[m]), 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        run_val = np.where(run_cnt > 0, run_sum / run_cnt, np.nan)
        slg = np.where(tb_cnt > 0, tb_sum / tb_cnt, np.nan)

    # 표본이 없는 칸은 해당 밴드 평균으로 채운다
    for b in range(3):
        for arr in (run_val, slg):
            fill = np.nanmean(arr[b]) if np.isfinite(arr[b]).any() else 0.0
            arr[b] = np.where(np.isnan(arr[b]), fill, arr[b])

    return run_val.astype(np.float32), slg.astype(np.float32)


def build_reference_tables(df: pd.DataFrame,
                           seasons: list | None = None,
                           min_pitches: int = 25) -> ReferenceTables:
    """참조 테이블 일괄 생성.

    ``df`` 는 ``build_pitch_features`` + ``build_context_features`` 를 거치고
    ``outcome_idx`` 가 붙어 있으며 **아직 표준화되지 않은** 상태여야 한다.

    ``seasons`` 를 주면 그 시즌만 써서 만든다. 2025 를 테스트로 남겨두고
    반사실 평가를 할 때는 2024 로만 만들어야 테이블 자체가 미래 정보를
    끌어오지 않는다.
    """
    if seasons is not None:
        df = df[df["season"].isin(seasons)]

    run_val, slg = build_location_tables(df)
    return ReferenceTables(
        repertoire=build_repertoire(df, min_pitches),
        league_repertoire=build_league_repertoire(df),
        count_value=build_count_value(df),
        location_value=run_val,
        location_slg=slg,
        grid_x=GRID_X,
        grid_z=GRID_Z,
        seasons=sorted(df["season"].unique().tolist()),
    )
