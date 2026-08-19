"""인코더·스케일러 적합과 데이터셋 조립.

이전 파이프라인은 인코더와 MinMaxScaler 를 **전체 데이터에 fit** 한 뒤 나눴다.
검증셋의 분포 정보가 스케일링을 통해 학습에 새어 들어가는 누수다. 여기서는
학습 구간에만 fit 하고 그 통계를 검증/테스트에 그대로 적용한다.

범주형은 인덱스 0 을 UNK 로 예약한다. 이전 코드의
``try: idx = le.transform([x]) except: idx = 0`` 은 처음 보는 선수를 조용히
0번 선수로 바꿔치기했다. UNK 를 따로 두고 학습 중 무작위로 마스킹하면
모델이 "모르는 선수" 표현을 실제로 학습한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    CAT_FEATURES,
    CONTEXT_FEATURES,
    OUTCOME_TO_IDX,
    PITCH_FEATURES,
    build_context_features,
    build_pitch_features,
    build_sequences,
    compute_batter_stats,
    define_outcome,
    load_raw,
)

UNK = 0  # 모든 범주형에서 0번은 "처음 보는 값"


@dataclass
class Encoders:
    """학습 구간에서만 추정한 전처리 파라미터."""

    cat_maps: dict[str, dict] = field(default_factory=dict)   # 원본값 → 인덱스
    cat_sizes: dict[str, int] = field(default_factory=dict)   # 임베딩 테이블 크기
    pitch_lo: np.ndarray | None = None    # 투구 물리량 클리핑 하한 (1퍼센타일)
    pitch_hi: np.ndarray | None = None    # 상한 (99퍼센타일)
    pitch_mean: np.ndarray | None = None
    pitch_std: np.ndarray | None = None
    pitch_median: np.ndarray | None = None  # 결측 대체값
    ctx_mean: np.ndarray | None = None
    ctx_std: np.ndarray | None = None
    outcomes: list[str] = field(default_factory=list)

    def cat_index(self, col: str, values) -> np.ndarray:
        """원본값 배열을 인덱스로. 사전에 없으면 UNK."""
        m = self.cat_maps[col]
        return np.array([m.get(v, UNK) for v in values], dtype=np.int64)


def fit_encoders(train: pd.DataFrame, min_count: int = 30) -> Encoders:
    """학습 구간에서 범주 사전과 정규화 통계를 만든다.

    ``min_count`` 미만으로 등장한 선수는 사전에 넣지 않고 UNK 로 보낸다.
    한두 번 나온 선수의 임베딩은 어차피 학습되지 않고, 그런 선수를 UNK 로
    보내야 학습 중에 UNK 표현이 실제로 훈련된다.
    """
    enc = Encoders(outcomes=list(OUTCOME_TO_IDX))

    for col in CAT_FEATURES:
        vc = train[col].astype(str).value_counts()
        keep = vc[vc >= min_count].index if col in ("batter", "pitcher") else vc.index
        # 0 은 UNK 이므로 1 부터 부여
        enc.cat_maps[col] = {v: i + 1 for i, v in enumerate(sorted(keep))}
        enc.cat_sizes[col] = len(enc.cat_maps[col]) + 1

    pitch = train[PITCH_FEATURES].to_numpy(np.float32)
    enc.pitch_lo = np.nanpercentile(pitch, 1, axis=0).astype(np.float32)
    enc.pitch_hi = np.nanpercentile(pitch, 99, axis=0).astype(np.float32)
    clipped = np.clip(pitch, enc.pitch_lo, enc.pitch_hi)
    enc.pitch_median = np.nanmedian(clipped, axis=0).astype(np.float32)
    enc.pitch_mean = np.nanmean(clipped, axis=0).astype(np.float32)
    enc.pitch_std = np.nanstd(clipped, axis=0).astype(np.float32)
    enc.pitch_std[enc.pitch_std < 1e-6] = 1.0

    ctx = train[CONTEXT_FEATURES].to_numpy(np.float32)
    enc.ctx_mean = np.nanmean(ctx, axis=0).astype(np.float32)
    enc.ctx_std = np.nanstd(ctx, axis=0).astype(np.float32)
    enc.ctx_std[enc.ctx_std < 1e-6] = 1.0
    return enc


def apply_encoders(df: pd.DataFrame, enc: Encoders) -> pd.DataFrame:
    """범주 인덱싱 + 수치 클리핑/표준화를 적용한다 (통계는 학습 구간 것)."""
    df = df.copy()

    for col in CAT_FEATURES:
        df[f"{col}_idx"] = enc.cat_index(col, df[col].astype(str).to_numpy())

    pitch = df[PITCH_FEATURES].to_numpy(np.float32)
    pitch = np.clip(pitch, enc.pitch_lo, enc.pitch_hi)
    # 결측은 학습 구간 중앙값으로 채운다 (구속·회전수 등이 2~3% 비어 있다)
    nan = np.isnan(pitch)
    pitch[nan] = np.take(enc.pitch_median, np.where(nan)[1])
    df[PITCH_FEATURES] = (pitch - enc.pitch_mean) / enc.pitch_std

    ctx = df[CONTEXT_FEATURES].to_numpy(np.float32)
    ctx = np.nan_to_num(ctx, nan=0.0)
    df[CONTEXT_FEATURES] = (ctx - enc.ctx_mean) / enc.ctx_std
    return df


def prepare(raw_paths,
            train_seasons: list[int],
            test_seasons: list[int],
            seq_len: int = 6,
            min_count: int = 30):
    """원본 경로 목록 → (학습 SequenceData, 테스트 SequenceData, Encoders).

    분할은 **시즌 단위**다. 겹치는 윈도우가 학습/테스트로 갈리는 일이 없고,
    학습 시점 이후의 정보가 전혀 들어가지 않는다.
    """
    df = load_raw(raw_paths)

    # 결과 라벨. 매핑되지 않는 투구(경기 중단 등)는 버린다.
    df["outcome"] = define_outcome(df)
    df = df[df["outcome"].notna()].reset_index(drop=True)
    df["outcome_idx"] = df["outcome"].map(OUTCOME_TO_IDX).astype(np.int64)

    df = build_pitch_features(df)

    # 타자 성적은 '직전 시즌' 것만 쓴다 (compute_batter_stats 가 season+1 처리).
    batter_stats = compute_batter_stats(df)
    df = build_context_features(df, batter_stats)

    train_df = df[df["season"].isin(train_seasons)].reset_index(drop=True)
    test_df = df[df["season"].isin(test_seasons)].reset_index(drop=True)
    if train_df.empty:
        raise ValueError(f"학습 시즌 {train_seasons} 에 해당하는 데이터가 없다")

    enc = fit_encoders(train_df, min_count=min_count)

    train_seq = build_sequences(apply_encoders(train_df, enc), seq_len)
    test_seq = (build_sequences(apply_encoders(test_df, enc), seq_len)
                if not test_df.empty else None)
    return train_seq, test_seq, enc


def prepare_by_date(raw_paths, cutoff: str, seq_len: int = 6, min_count: int = 30):
    """시즌이 하나뿐일 때 쓰는 날짜 기준 분할 (cutoff 이전=학습, 이후=테스트)."""
    df = load_raw(raw_paths)
    df["outcome"] = define_outcome(df)
    df = df[df["outcome"].notna()].reset_index(drop=True)
    df["outcome_idx"] = df["outcome"].map(OUTCOME_TO_IDX).astype(np.int64)

    df = build_pitch_features(df)
    df = build_context_features(df, None)  # 단일 시즌이면 직전 시즌 성적이 없다

    split = pd.Timestamp(cutoff)
    train_df = df[df["game_date"] < split].reset_index(drop=True)
    test_df = df[df["game_date"] >= split].reset_index(drop=True)

    enc = fit_encoders(train_df, min_count=min_count)
    train_seq = build_sequences(apply_encoders(train_df, enc), seq_len)
    test_seq = (build_sequences(apply_encoders(test_df, enc), seq_len)
                if not test_df.empty else None)
    return train_seq, test_seq, enc
