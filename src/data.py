"""Statcast 원본 → 학습용 시퀀스 텐서.

이 모듈은 이전 파이프라인의 세 가지 치명적 결함을 바로잡는다.

1. **시간 정렬**: pybaseball 이 돌려주는 원본은 역시간순이다(한 타석 안에서
   pitch_number 가 7,6,5...1). 이전 코드는 정렬 없이 그대로 슬라이딩 윈도우를
   만들어, 결과적으로 "나중에 던진 공들로 먼저 던진 공의 결과를 맞히는" 문제를
   풀고 있었다. 카운트(balls/strikes)가 모든 타임스텝에 들어가므로 타깃 다음
   투구의 카운트만 보면 볼/스트라이크를 그대로 읽을 수 있었다.
2. **타석 경계**: 이전에는 타자별로 시즌 전체를 이어붙여 윈도우를 잘라서, 한
   시퀀스가 서로 다른 경기·투수·월을 넘나들었다. 여기서는 시퀀스를 타석(PA)
   안으로 제한하고 부족한 앞부분은 패딩 + 마스크로 처리한다.
3. **분할**: 겹치는 윈도우를 무작위로 나누면 학습/검증이 8개 중 6개를 공유한다.
   시즌(또는 날짜) 단위로 자른다.

좌표계는 Takamido & Nakamoto (2026) 를 따른다. 좌타자는 좌우를 미러링해
"몸쪽/바깥쪽"을 좌우 타석에 관계없이 같은 부호로 만들고, 높이는 타자별
스트라이크존(sz_bot~sz_top)으로 정규화한다.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------
# 결과(outcome) 정의
# --------------------------------------------------------------------------
# 투구 단위로 실제 관측되는 사건만 클래스로 둔다.
# strikeout / walk 는 투구 결과가 아니라 카운트에서 파생되는 타석 결과이므로
# 클래스에서 뺀다(삼진 = 2스트라이크에서의 스트라이크, 볼넷 = 3볼에서의 볼).
# 이전 정의는 strike 와 strikeout 을 동시에 클래스로 둬서 서로 중복이었고,
# 카운트가 입력에 있으니 모델이 쉽게 맞히는 라벨이었다.
OUTCOMES = [
    "ball",
    "called_strike",
    "swinging_strike",
    "foul",
    "hit_by_pitch",
    "field_out",
    "single",
    "double",
    "triple",
    "home_run",
]
OUTCOME_TO_IDX = {o: i for i, o in enumerate(OUTCOMES)}

_DESC_MAP = {
    "ball": "ball",
    "blocked_ball": "ball",
    "automatic_ball": "ball",
    "pitchout": "ball",
    "called_strike": "called_strike",
    "automatic_strike": "called_strike",
    "swinging_strike": "swinging_strike",
    "swinging_strike_blocked": "swinging_strike",
    "missed_bunt": "swinging_strike",
    "foul": "foul",
    "foul_tip": "foul",
    "foul_bunt": "foul",
    "bunt_foul_tip": "foul",
    "hit_by_pitch": "hit_by_pitch",
}

# 인플레이 타구는 events 로 결과를 정한다. 이전 코드는 화이트리스트에 없는
# force_out / GIDP / sac_fly / triple / field_error 등을 통째로 버려
# 3루타가 클래스에서 사라졌었다.
_EVENT_MAP = {
    "single": "single",
    "double": "double",
    "triple": "triple",
    "home_run": "home_run",
    "field_out": "field_out",
    "force_out": "field_out",
    "grounded_into_double_play": "field_out",
    "double_play": "field_out",
    "triple_play": "field_out",
    "sac_fly": "field_out",
    "sac_bunt": "field_out",
    "sac_fly_double_play": "field_out",
    "sac_bunt_double_play": "field_out",
    "fielders_choice_out": "field_out",
    "fielders_choice": "single",  # 타자는 살아나감
    "field_error": "single",
}

# 시퀀스 경로에 들어가는 투구 물리량
PITCH_FEATURES = [
    "loc_x",      # 미러링된 좌우 위치 (+ = 타자 기준 바깥쪽)
    "loc_z",      # 타자별 존으로 정규화한 높이 (0 = 존 하단, 1 = 존 상단)
    "speed",      # effective_speed
    "spin_rate",  # release_spin_rate
    "spin_sin",   # spin_axis 를 원형으로 인코딩
    "spin_cos",
    "mov_x",      # 미러링된 수평 무브먼트
    "mov_z",      # 수직 무브먼트
]

# 맥락 경로(스킵 연결)에 들어가는 현재 상황
CONTEXT_FEATURES = [
    "balls",
    "strikes",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "inning",
    "is_bottom",
    "score_diff",           # 투수 팀 기준 점수차
    "n_thruorder_pitcher",  # 타순 몇 바퀴째인가
    "bat_k_rate",           # 아래 5개는 상대 타자의 '직전 시즌' 성적
    "bat_hr_rate",
    "bat_iso",
    "bat_avg",
    "bat_ops",
]

CAT_FEATURES = ["batter", "pitcher", "pitch_type", "stand", "p_throws"]

# 정렬 키. 이 순서가 곧 시간순이다.
SORT_KEYS = ["game_date", "game_pk", "at_bat_number", "pitch_number"]

# 원본에서 실제로 읽어야 하는 열
RAW_COLUMNS = [
    "game_date", "game_pk", "game_type", "at_bat_number", "pitch_number",
    "inning", "inning_topbot",
    "batter", "pitcher", "stand", "p_throws", "pitch_type",
    "plate_x", "plate_z", "sz_top", "sz_bot",
    "effective_speed", "release_speed", "release_spin_rate", "spin_axis",
    "pfx_x", "pfx_z",
    "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b",
    "bat_score", "fld_score", "n_thruorder_pitcher",
    "events", "description", "delta_run_exp",
    # 아래 3개는 모델 피처가 아니다. 대시보드의 팀/선수 분류에만 쓴다.
    "home_team", "away_team", "player_name",
]


# --------------------------------------------------------------------------
# 원본 적재
# --------------------------------------------------------------------------
def load_raw(paths, columns: list[str] | None = RAW_COLUMNS) -> pd.DataFrame:
    """parquet/csv 원본을 읽어 **시간순으로 정렬해** 돌려준다."""
    if isinstance(paths, (str, Path)):
        paths = [paths]

    frames = []
    for p in paths:
        p = Path(p)
        if p.suffix == ".parquet":
            frames.append(pd.read_parquet(p, columns=columns))
        else:
            frames.append(pd.read_csv(p, usecols=columns, low_memory=False))
    df = pd.concat(frames, ignore_index=True)

    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"].drop(columns=["game_type"])

    df["game_date"] = pd.to_datetime(df["game_date"])

    # ★ 핵심 수정. 원본은 역시간순이라 정렬 없이는 시퀀스가 거꾸로 만들어진다.
    df = df.sort_values(SORT_KEYS, kind="mergesort").reset_index(drop=True)
    df["season"] = df["game_date"].dt.year

    # 초공격이면 원정팀이 타격, 홈팀이 수비. 투수는 수비팀, 타자는 공격팀 소속이다.
    if {"home_team", "away_team", "inning_topbot"} <= set(df.columns):
        top = df["inning_topbot"].eq("Top")
        df["pitch_team"] = np.where(top, df["home_team"], df["away_team"])
        df["bat_team"] = np.where(top, df["away_team"], df["home_team"])
    return df


def define_outcome(df: pd.DataFrame) -> pd.Series:
    """description/events 로부터 투구 단위 결과 라벨을 만든다."""
    desc = df["description"].astype(str)
    out = desc.map(_DESC_MAP)

    in_play = desc.eq("hit_into_play")
    out.loc[in_play] = df.loc[in_play, "events"].astype(str).map(_EVENT_MAP)

    return out  # 매핑 실패분은 NaN → 호출부에서 제거


# --------------------------------------------------------------------------
# 피처 생성
# --------------------------------------------------------------------------
def build_pitch_features(df: pd.DataFrame) -> pd.DataFrame:
    """좌표 미러링 + 존 정규화 + 회전축 원형 인코딩."""
    df = df.copy()

    # 좌타자는 좌우를 뒤집어 '몸쪽/바깥쪽'의 부호를 통일한다.
    # 미러링 후 양수 = 타자 기준 바깥쪽.
    side = np.where(df["stand"].eq("L"), -1.0, 1.0)
    df["loc_x"] = df["plate_x"] * side
    df["mov_x"] = df["pfx_x"] * side

    # 타자별 스트라이크존으로 높이를 정규화한다. 0 = 존 하단, 1 = 존 상단.
    zone_height = (df["sz_top"] - df["sz_bot"]).replace(0, np.nan)
    df["loc_z"] = (df["plate_z"] - df["sz_bot"]) / zone_height

    df["speed"] = df["effective_speed"].fillna(df["release_speed"])
    df["spin_rate"] = df["release_spin_rate"]

    # 회전축은 각도(0~360)라 그대로 쓰면 0도와 359도가 멀리 떨어진 값이 된다.
    rad = np.deg2rad(df["spin_axis"])
    df["spin_sin"] = np.sin(rad)
    df["spin_cos"] = np.cos(rad)

    df["mov_z"] = df["pfx_z"]
    return df


def compute_batter_stats(df: pd.DataFrame) -> pd.DataFrame:
    """타자별 시즌 성적을 계산하고, 붙일 대상 시즌을 +1 해서 돌려준다.

    반환된 ``season`` 열은 '이 성적을 참조해야 하는 시즌'을 뜻한다.
    (2024 성적 → 2025 경기의 맥락 피처). 이렇게 하면 같은 시즌 성적을
    그 시즌 예측에 쓰는 누수가 생기지 않는다.
    """
    pa = df[df["events"].notna() & df["events"].astype(str).ne("")].copy()

    ev = pa["events"].astype(str)
    pa["is_ab"] = ~ev.isin(
        ["walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt",
         "catcher_interf", "truncated_pa", "sac_fly_double_play"]
    )
    pa["is_hit"] = ev.isin(["single", "double", "triple", "home_run"])
    pa["is_k"] = ev.str.startswith("strikeout")
    pa["is_hr"] = ev.eq("home_run")
    pa["is_bb"] = ev.isin(["walk", "intent_walk"])
    pa["tb"] = ev.map({"single": 1, "double": 2, "triple": 3, "home_run": 4}).fillna(0)

    stats = pa.groupby(["season", "batter"]).agg(
        pa_count=("events", "size"),
        ab=("is_ab", "sum"),
        hits=("is_hit", "sum"),
        k=("is_k", "sum"),
        hr=("is_hr", "sum"),
        bb=("is_bb", "sum"),
        tb=("tb", "sum"),
    ).reset_index()

    stats = stats[stats["pa_count"] >= 50]  # 표본이 너무 적은 타자는 제외

    ab = stats["ab"].replace(0, np.nan)
    stats["bat_avg"] = stats["hits"] / ab
    stats["bat_iso"] = stats["tb"] / ab - stats["bat_avg"]
    stats["bat_k_rate"] = stats["k"] / stats["pa_count"]
    stats["bat_hr_rate"] = stats["hr"] / stats["pa_count"]
    obp = (stats["hits"] + stats["bb"]) / stats["pa_count"]
    stats["bat_ops"] = obp + stats["tb"] / ab

    stats["season"] = stats["season"] + 1  # 다음 시즌에 붙는다
    cols = ["bat_k_rate", "bat_hr_rate", "bat_iso", "bat_avg", "bat_ops"]
    return stats[["season", "batter"] + cols]


def build_context_features(df: pd.DataFrame,
                           batter_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """현재 상황 + 상대 타자의 직전 시즌 성적."""
    df = df.copy()

    for base in ["on_1b", "on_2b", "on_3b"]:
        df[base] = df[base].notna().astype(np.float32)

    df["is_bottom"] = df["inning_topbot"].eq("Bot").astype(np.float32)
    # 투수 팀 기준 점수차 (양수 = 투수 팀이 앞섬)
    df["score_diff"] = (df["fld_score"] - df["bat_score"]).astype(np.float32)

    # 상태 임베딩용 인덱스. 표준화 대상이 아니므로 여기서 원시값으로 만들어 둔다.
    # 카운트 12상태(볼 0-3 x 스트라이크 0-2), 베이스-아웃 24상태(주자 8 x 아웃 3).
    df["count_state"] = (df["balls"].clip(0, 3) * 3
                         + df["strikes"].clip(0, 2)).astype(np.int64)
    bases = df["on_1b"] + df["on_2b"] * 2 + df["on_3b"] * 4
    df["baseout_state"] = (bases.clip(0, 7) * 3
                           + df["outs_when_up"].clip(0, 2)).astype(np.int64)

    stat_cols = ["bat_k_rate", "bat_hr_rate", "bat_iso", "bat_avg", "bat_ops"]
    if batter_stats is None:
        for c in stat_cols:
            df[c] = np.nan
    else:
        # batter 는 학습 경로에서는 정수, 앱 경로에서는 문자열로 들어온다.
        # 문자열로 통일해서 붙이지 않으면 merge 가 통째로 실패해 모든 타자가
        # 기본값으로 채워진다.
        bs = batter_stats.copy()
        bs["_bkey"] = bs["batter"].astype(str)
        bs = bs.drop(columns=["batter"])
        df["_bkey"] = df["batter"].astype(str)
        df = df.merge(bs, on=["season", "_bkey"], how="left").drop(columns=["_bkey"])

    # 직전 시즌 기록이 없는 타자(신인 등)는 평균보다 10% 나쁜 값으로 채운다.
    # 논문과 동일한 처리. 타자에게 나쁜 쪽 = 삼진율↑, 나머지 지표↓.
    for c in stat_cols:
        mean = df[c].mean()
        if not np.isfinite(mean):
            mean = 0.0
        fill = mean * (1.10 if c == "bat_k_rate" else 0.90)
        df[c] = df[c].fillna(fill).astype(np.float32)
    return df


# --------------------------------------------------------------------------
# 시퀀스 생성
# --------------------------------------------------------------------------
@dataclass
class SequenceData:
    """모델 입력 묶음. n = 샘플(투구) 수, L = 시퀀스 길이."""

    cat: np.ndarray      # (n, L, 5)   범주형 인덱스
    num: np.ndarray      # (n, L, 8)   투구 물리량
    mask: np.ndarray     # (n, L)      True = 유효한 투구, False = 패딩
    ctx: np.ndarray      # (n, 15)     현재 상황 (표준화됨)
    state: np.ndarray    # (n, 2)      [카운트 상태 0-11, 베이스-아웃 상태 0-23]
    y: np.ndarray        # (n,)        결과 클래스
    run_exp: np.ndarray  # (n,)        delta_run_exp (기대 실점 변화량)
    meta: pd.DataFrame   # 샘플별 원본 식별자 (분할·분석용)

    def __len__(self) -> int:
        return len(self.y)


def build_sequences(df: pd.DataFrame, seq_len: int = 6) -> SequenceData:
    """각 투구를 타깃으로, 같은 타석 안의 직전 투구들을 시퀀스로 붙인다.

    시퀀스의 **마지막 원소가 타깃 투구 자신**이다. 그 공의 물리량(구종·위치·
    구속)을 알고 결과를 묻는 구조라, 앱에서 "이 공을 던지면?" 을 그대로 물을 수
    있다. 타석 첫 공처럼 앞이 부족하면 0 으로 채우고 mask 를 False 로 둔다.
    타석 경계를 넘지 않으므로 다른 타자·투수의 공이 섞이지 않는다.
    """
    n = len(df)
    cat_vals = df[[f"{c}_idx" for c in CAT_FEATURES]].to_numpy(np.int64)
    num_vals = df[PITCH_FEATURES].to_numpy(np.float32)

    cat = np.zeros((n, seq_len, len(CAT_FEATURES)), dtype=np.int64)
    num = np.zeros((n, seq_len, len(PITCH_FEATURES)), dtype=np.float32)
    mask = np.zeros((n, seq_len), dtype=bool)

    # 각 행이 속한 타석의 시작 위치. 정렬돼 있으므로 같은 타석은 연속 구간이다.
    pa_key = (df["game_pk"].to_numpy(np.int64) * 1000
              + df["at_bat_number"].to_numpy(np.int64))
    new_pa = np.empty(n, dtype=bool)
    new_pa[0] = True
    new_pa[1:] = pa_key[1:] != pa_key[:-1]
    pa_start = np.maximum.accumulate(np.where(new_pa, np.arange(n), 0))

    idx = np.arange(n)
    for offset in range(seq_len):
        # 시퀀스 뒤에서 offset 번째 자리에 (타깃 - offset) 번째 투구를 넣는다.
        src = idx - offset
        valid = src >= pa_start  # 같은 타석 안에 머무는가
        pos = seq_len - 1 - offset
        rows = idx[valid]
        cat[rows, pos] = cat_vals[src[valid]]
        num[rows, pos] = num_vals[src[valid]]
        mask[rows, pos] = True

    return SequenceData(
        cat=cat,
        num=num,
        mask=mask,
        ctx=df[CONTEXT_FEATURES].to_numpy(np.float32),
        state=df[["count_state", "baseout_state"]].to_numpy(np.int64),
        y=df["outcome_idx"].to_numpy(np.int64),
        run_exp=df["delta_run_exp"].fillna(0.0).to_numpy(np.float32),
        meta=df[["game_date", "game_pk", "at_bat_number", "pitch_number",
                 "batter", "pitcher", "season"]].reset_index(drop=True),
    )
