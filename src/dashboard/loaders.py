"""산출물 로딩과 캐싱.

학습 결과물이 아직 없어도 앱이 죽지 않는다. 모델이 없으면 추천 페이지만
비활성화하고, 데이터 집계로 돌아가는 선수·팀 분석 페이지는 그대로 쓴다.
(이전 앱은 파일 하나만 없어도 ``st.stop()`` 으로 전체가 멈췄다.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"

MODEL_DIR = MODELS / "main"
TABLES_PATH = MODELS / "tables.pkl"
PROFILES_PATH = MODELS / "profiles.pkl"


@dataclass
class Resources:
    model = None
    encoders = None
    tables = None
    profiles = None
    metadata: dict | None = None
    batter_stats = None

    @property
    def has_model(self) -> bool:
        return self.model is not None and self.tables is not None

    @property
    def has_profiles(self) -> bool:
        return self.profiles is not None


@st.cache_resource(show_spinner="산출물 불러오는 중...")
def load_resources() -> tuple[Resources, list[str]]:
    """모델·테이블·프로파일을 읽는다. 없는 것은 경고만 남기고 넘어간다."""
    import joblib

    res = Resources()
    warnings: list[str] = []

    # --- 프로파일 (선수·팀 분석) ---
    if PROFILES_PATH.exists():
        try:
            from ..profiles import Profiles
            res.profiles = Profiles.load(PROFILES_PATH)
        except Exception as e:                     # noqa: BLE001
            warnings.append(f"프로파일 로드 실패: {e}")
    else:
        warnings.append(
            f"`{PROFILES_PATH.relative_to(ROOT)}` 가 없습니다. "
            "`python scripts/build_profiles.py --raw data/raw --out models/profiles.pkl` 를 실행하세요."
        )

    # --- 참조 테이블 (추천) ---
    if TABLES_PATH.exists():
        try:
            from ..features import ReferenceTables
            res.tables = ReferenceTables.load(TABLES_PATH)
        except Exception as e:                     # noqa: BLE001
            warnings.append(f"참조 테이블 로드 실패: {e}")
    else:
        warnings.append(
            f"`{TABLES_PATH.relative_to(ROOT)}` 가 없습니다. "
            "`python scripts/build_tables.py --raw data/raw --out models/tables.pkl` 를 실행하세요."
        )

    # --- 학습된 모델 ---
    cfg_path = MODEL_DIR / "model_config.json"
    if cfg_path.exists():
        try:
            import torch
            from ..model import ContextAwareTransformer, ModelConfig

            cfg = ModelConfig(**json.loads(cfg_path.read_text(encoding="utf-8")))
            model = ContextAwareTransformer(cfg)
            model.load_state_dict(torch.load(MODEL_DIR / "model.pth",
                                             map_location="cpu"))
            model.eval()
            res.model = model
            res.encoders = joblib.load(MODEL_DIR / "encoders.pkl")

            meta = MODEL_DIR / "metadata.json"
            if meta.exists():
                res.metadata = json.loads(meta.read_text(encoding="utf-8"))
        except Exception as e:                     # noqa: BLE001
            warnings.append(f"모델 로드 실패: {e}")
    else:
        warnings.append(
            "학습된 모델이 없습니다 (`models/main/`). "
            "`notebooks/02_train.ipynb` 실행 후 `artifacts.zip` 을 풀어 넣으세요."
        )

    bs = MODELS / "batter_stats.pkl"
    if bs.exists():
        try:
            res.batter_stats = joblib.load(bs)
        except Exception:                          # noqa: BLE001
            pass

    return res, warnings


# --------------------------------------------------------------------------
def player_options(profiles, role: str, season: int | None = None,
                   team: str | None = None, min_pitches: int = 200):
    """(표시 이름, 선수 ID) 목록. 팀·시즌으로 좁힌다."""
    d = profiles.directory
    d = d[d["role"] == role]
    if season is not None:
        d = d[d["season"] == season]
    if team and team != "전체":
        d = d[d["team"] == team]
    d = d[d["pitches"] >= min_pitches]
    d = d.sort_values("pitches", ascending=False)
    return [(f"{r['name']}  ·  {r['team']}", str(r["player_id"]))
            for _, r in d.iterrows()]


def team_options(profiles, season: int | None = None) -> list[str]:
    d = profiles.directory
    if season is not None:
        d = d[d["season"] == season]
    teams = sorted(t for t in d["team"].dropna().unique() if t)
    return ["전체"] + teams


def seasons_available(profiles) -> list[int]:
    return sorted(profiles.directory["season"].unique().tolist(), reverse=True)
