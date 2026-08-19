"""대시보드가 쓰는 선수·팀 집계.

``features.py`` 가 추천 엔진의 입력을 만든다면, 여기서는 **사람이 읽는 화면**을
위한 집계를 만든다. 선수 명부, 팀 소속, 타자 약점 존, 투수 레퍼토리와 카운트별
구사 성향 등이다.

표본이 적은 칸은 리그 평균 쪽으로 당긴다(경험적 베이즈 축소). 어떤 타자의
특정 존에 공이 3개밖에 안 왔는데 그 3개가 우연히 홈런이었다고 해서 "이 선수의
약점" 이라고 표시하면 안 되기 때문이다. 이전 앱에는 이런 화면 자체가 없었다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import OUTCOMES, OUTCOME_TO_IDX
from .features import GRID_X, GRID_Z, _grid_index, in_strike_zone, velocity_band

# 스윙으로 간주하는 결과
SWING_OUTCOMES = {"swinging_strike", "foul", "field_out",
                  "single", "double", "triple", "home_run"}
WHIFF_OUTCOMES = {"swinging_strike"}
HIT_OUTCOMES = {"single", "double", "triple", "home_run"}

SHRINK_K = 40.0   # 존 칸별 축소 강도. 이 표본수에서 리그 평균과 반반이 된다.


def _derive_flags(df: pd.DataFrame) -> pd.DataFrame:
    """스윙/헛스윙/존 여부 등 집계용 파생 열."""
    df = df.copy()
    oc = df["outcome"]
    df["is_swing"] = oc.isin(SWING_OUTCOMES)
    df["is_whiff"] = oc.isin(WHIFF_OUTCOMES)
    df["is_hit"] = oc.isin(HIT_OUTCOMES)
    df["is_in_zone"] = in_strike_zone(df["loc_x"].to_numpy(), df["loc_z"].to_numpy())
    df["is_chase"] = df["is_swing"] & ~df["is_in_zone"]
    df["is_contact"] = df["is_swing"] & ~df["is_whiff"]
    # 타석을 끝낸 투구
    df["ends_pa"] = df["events"].notna() & df["events"].astype(str).ne("")
    ev = df["events"].astype(str)
    df["pa_k"] = df["ends_pa"] & ev.str.startswith("strikeout")
    df["pa_bb"] = df["ends_pa"] & ev.isin(["walk", "intent_walk"])
    return df


# --------------------------------------------------------------------------
def build_player_directory(df: pd.DataFrame,
                           name_map: dict | None = None) -> pd.DataFrame:
    """선수 명부. 투수/타자를 한 표에 담고 팀·손·표본수를 붙인다.

    투수 이름은 Statcast 의 ``player_name`` 에 그대로 들어 있다(투구 단위
    데이터에서 이 열은 투수를 가리킨다). 타자 이름은 별도 매핑이 필요하다.
    """
    name_map = {str(k): v for k, v in (name_map or {}).items()}

    # --- 투수 ---
    p = df.groupby(["season", "pitcher"], observed=True).agg(
        team=("pitch_team", lambda s: s.mode().iat[0] if len(s.mode()) else ""),
        hand=("p_throws", lambda s: s.mode().iat[0] if len(s.mode()) else ""),
        pitches=("pitcher", "size"),
        name=("player_name", lambda s: s.mode().iat[0] if len(s.mode()) else ""),
    ).reset_index().rename(columns={"pitcher": "player_id"})
    p["role"] = "pitcher"

    # --- 타자 ---
    b = df.groupby(["season", "batter"], observed=True).agg(
        team=("bat_team", lambda s: s.mode().iat[0] if len(s.mode()) else ""),
        hand=("stand", lambda s: s.mode().iat[0] if len(s.mode()) else ""),
        pitches=("batter", "size"),
        pa=("ends_pa", "sum"),
    ).reset_index().rename(columns={"batter": "player_id"})
    b["role"] = "batter"
    b["name"] = b["player_id"].astype(str).map(name_map)

    out = pd.concat([p, b], ignore_index=True)
    out["player_id"] = out["player_id"].astype(str)
    # 이름을 모르면 ID 를 노출한다. 이전 앱처럼 목록에서 조용히 빼지 않는다.
    out["name"] = out["name"].fillna("").replace("", np.nan)
    out["name"] = out["name"].fillna(out["player_id"].map(lambda i: f"ID {i}"))
    out["label"] = out["name"] + " (" + out["team"].fillna("?") + ")"
    return out.sort_values(["role", "season", "name"]).reset_index(drop=True)


def normalize_name(name: str) -> str:
    """'Gray, Jon' → 'Jon Gray'. Statcast 는 성, 이름 순으로 준다."""
    if isinstance(name, str) and ", " in name:
        last, first = name.split(", ", 1)
        return f"{first} {last}"
    return name


# --------------------------------------------------------------------------
def _zone_grid(sub: pd.DataFrame, league: np.ndarray,
               grid_x=GRID_X, grid_z=GRID_Z, k: float = SHRINK_K):
    """(높이칸, 좌우칸) 평균 기대 실점. 표본이 적은 칸은 리그 평균으로 축소."""
    nz, nx = len(grid_z), len(grid_x)
    xi = _grid_index(sub["loc_x"].to_numpy(), grid_x)
    zi = _grid_index(sub["loc_z"].to_numpy(), grid_z)

    s = np.zeros((nz, nx)); n = np.zeros((nz, nx))
    np.add.at(s, (zi, xi), sub["delta_run_exp"].fillna(0.0).to_numpy())
    np.add.at(n, (zi, xi), 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(n > 0, s / n, 0.0)
    # 경험적 베이즈: 표본이 적을수록 리그 평균에 가깝게
    shrunk = (n * raw + k * league) / (n + k)
    return shrunk.astype(np.float32), n.astype(np.int32)


def league_zone_grid(df: pd.DataFrame) -> np.ndarray:
    nz, nx = len(GRID_Z), len(GRID_X)
    xi = _grid_index(df["loc_x"].to_numpy(), GRID_X)
    zi = _grid_index(df["loc_z"].to_numpy(), GRID_Z)
    s = np.zeros((nz, nx)); n = np.zeros((nz, nx))
    np.add.at(s, (zi, xi), df["delta_run_exp"].fillna(0.0).to_numpy())
    np.add.at(n, (zi, xi), 1.0)
    overall = df["delta_run_exp"].mean()
    with np.errstate(invalid="ignore"):
        g = np.where(n > 0, s / n, overall)
    return g.astype(np.float32)


@dataclass
class Profiles:
    directory: pd.DataFrame
    batter_summary: pd.DataFrame      # 타자별 요약 지표
    batter_zone: dict                 # (season, batter) → (grid, count)
    batter_by_pitch: pd.DataFrame     # 타자 × 구종
    pitcher_summary: pd.DataFrame
    pitcher_zone: dict
    pitcher_usage: pd.DataFrame       # 투수 × 구종 × 카운트상태 구사율
    team_summary: pd.DataFrame
    league_zone: np.ndarray
    grid_x: np.ndarray
    grid_z: np.ndarray

    def save(self, path) -> Path:
        import joblib
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path) -> "Profiles":
        import joblib
        return joblib.load(path)


# --------------------------------------------------------------------------
def _summary(g: pd.DataFrame) -> pd.Series:
    n = len(g)
    swings = g["is_swing"].sum()
    out_zone = (~g["is_in_zone"]).sum()
    return pd.Series({
        "pitches": n,
        "pa": int(g["ends_pa"].sum()),
        "run_value_per_100": 100.0 * g["delta_run_exp"].mean(),
        "whiff_rate": g["is_whiff"].sum() / swings if swings else np.nan,
        "swing_rate": swings / n,
        "chase_rate": g["is_chase"].sum() / out_zone if out_zone else np.nan,
        "zone_rate": g["is_in_zone"].sum() / n,
        "k_rate": g["pa_k"].sum() / max(g["ends_pa"].sum(), 1),
        "bb_rate": g["pa_bb"].sum() / max(g["ends_pa"].sum(), 1),
        "hard_hit_rate": g["is_hit"].sum() / max(g["is_contact"].sum(), 1),
    })


def build_profiles(df: pd.DataFrame, name_map: dict | None = None,
                   min_pitches: int = 200) -> Profiles:
    """대시보드용 집계를 한 번에 만든다.

    ``df`` 는 ``build_pitch_features`` 를 거치고 ``outcome`` 이 붙은,
    **표준화 전** 상태여야 한다.
    """
    df = _derive_flags(df)
    league = league_zone_grid(df)

    directory = build_player_directory(df, name_map)
    directory["name"] = directory["name"].map(normalize_name)
    directory["label"] = directory["name"] + " · " + directory["team"].fillna("?")

    # ---- 타자 ----
    bat = df.groupby(["season", "batter"], observed=True).apply(
        _summary, include_groups=False).reset_index()
    bat = bat[bat["pitches"] >= min_pitches].reset_index(drop=True)
    bat["batter"] = bat["batter"].astype(str)

    batter_zone = {}
    for (season, bid), sub in df.groupby(["season", "batter"], observed=True):
        if len(sub) >= min_pitches:
            batter_zone[(int(season), str(bid))] = _zone_grid(sub, league)

    bbp = df.groupby(["season", "batter", "pitch_type"], observed=True).agg(
        n=("delta_run_exp", "size"),
        run_value=("delta_run_exp", "mean"),
        whiff=("is_whiff", "sum"),
        swings=("is_swing", "sum"),
    ).reset_index()
    bbp = bbp[bbp["n"] >= 25].reset_index(drop=True)
    bbp["whiff_rate"] = bbp["whiff"] / bbp["swings"].replace(0, np.nan)
    bbp["run_value_per_100"] = 100 * bbp["run_value"]
    bbp["batter"] = bbp["batter"].astype(str)

    # ---- 투수 ----
    pit = df.groupby(["season", "pitcher"], observed=True).apply(
        _summary, include_groups=False).reset_index()
    pit = pit[pit["pitches"] >= min_pitches].reset_index(drop=True)
    pit["pitcher"] = pit["pitcher"].astype(str)

    pitcher_zone = {}
    for (season, pid), sub in df.groupby(["season", "pitcher"], observed=True):
        if len(sub) >= min_pitches:
            pitcher_zone[(int(season), str(pid))] = _zone_grid(sub, league)

    usage = df.groupby(["season", "pitcher", "count_state", "pitch_type"],
                       observed=True).size().rename("n").reset_index()
    total = usage.groupby(["season", "pitcher", "count_state"])["n"].transform("sum")
    usage["usage"] = usage["n"] / total
    usage["pitcher"] = usage["pitcher"].astype(str)

    # ---- 팀 ----
    team = df.groupby(["season", "pitch_team"], observed=True).apply(
        _summary, include_groups=False).reset_index().rename(
        columns={"pitch_team": "team"})

    return Profiles(
        directory=directory,
        batter_summary=bat, batter_zone=batter_zone, batter_by_pitch=bbp,
        pitcher_summary=pit, pitcher_zone=pitcher_zone, pitcher_usage=usage,
        team_summary=team, league_zone=league,
        grid_x=GRID_X, grid_z=GRID_Z,
    )
