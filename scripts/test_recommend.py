"""추천 엔진 배선 검증. 랜덤 가중치 모델로 경로 전체를 통과시켜 본다.

성능이 아니라 '학습과 동일한 전처리로 후보가 만들어지는가' 를 본다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (OUTCOME_TO_IDX, build_context_features, build_pitch_features,
                      compute_batter_stats, define_outcome, load_raw)
from src.features import build_reference_tables
from src.model import ContextAwareTransformer, ModelConfig
from src.prepare import prepare
from src.recommend import (GameState, apply_command_window, score_candidates, top_k)

SAMPLE = Path("data/sample.parquet")

print("[1] 인코더 준비")
train, test, enc = prepare(SAMPLE, train_seasons=[2024], test_seasons=[2025], seq_len=6)

print("[2] 참조 테이블 준비")
df = load_raw(SAMPLE)
df["outcome"] = define_outcome(df)
df = df[df["outcome"].notna()].reset_index(drop=True)
df["outcome_idx"] = df["outcome"].map(OUTCOME_TO_IDX).astype(np.int64)
df = build_pitch_features(df)
bstats = compute_batter_stats(df)
df = build_context_features(df, bstats)
tables = build_reference_tables(df, seasons=[2024])

print("[3] 랜덤 가중치 모델")
model = ContextAwareTransformer(ModelConfig(cat_sizes=enc.cat_sizes, seq_len=6))

# 실제로 데이터에 있는 투수/타자를 하나 고른다
rep = tables.repertoire
busiest = rep.groupby("pitcher")["n"].sum().idxmax()
some_batter = df["batter"].value_counts().index[0]
print(f"    투수 {busiest} / 타자 {some_batter}")

for label, st in [
    ("0-0 무사 주자없음", GameState(batter=str(some_batter), pitcher=str(busiest),
                                    balls=0, strikes=0, season=2025)),
    ("0-2 (승부구 상황)", GameState(batter=str(some_batter), pitcher=str(busiest),
                                    balls=0, strikes=2, season=2025)),
    ("3-2 만루 2아웃", GameState(batter=str(some_batter), pitcher=str(busiest),
                                 balls=3, strikes=2, outs=2,
                                 on_1b=True, on_2b=True, on_3b=True, season=2025)),
    ("이력 3구 있음", GameState(batter=str(some_batter), pitcher=str(busiest),
                                balls=1, strikes=1, season=2025,
                                history=[{"pitch_type": "FF", "loc_x": -0.5, "loc_z": 0.8},
                                         {"pitch_type": "SL", "loc_x": 0.6, "loc_z": 0.2},
                                         {"pitch_type": "FF", "loc_x": 0.1, "loc_z": 0.5}])),
]:
    cands = score_candidates(model, enc, tables, st, batter_stats=bstats)
    cands = apply_command_window(cands, tables, window=3)
    best = top_k(cands, k=3)

    zone_n = sum(c.in_zone for c in cands)
    print(f"\n=== {label} ===")
    print(f"    후보 {len(cands)}개 (존 안 {zone_n} / 존 밖 {len(cands)-zone_n})")
    for i, c in enumerate(best, 1):
        top = ", ".join(f"{n} {p*100:.0f}%" for n, p in c.summary(3))
        print(f"    #{i} {c.pitch_type:3s} x={c.loc_x:+5.2f} z={c.loc_z:+5.2f} "
              f"{'존안' if c.in_zone else '존밖'} | 기대실점 {c.window_value:+.4f} | {top}")

print("\n[4] 정합성 점검")
cands = score_candidates(model, enc, tables,
                         GameState(batter=str(some_batter), pitcher=str(busiest), season=2025),
                         batter_stats=bstats)
P = np.array([c.probs for c in cands])
print(f"    확률 합이 1인가: {np.allclose(P.sum(1), 1.0)}")
print(f"    NaN 있나: {np.isnan(P).any()}")
print(f"    후보별 기대실점 범위: {min(c.run_value for c in cands):+.4f} ~ "
      f"{max(c.run_value for c in cands):+.4f}")
print("\n배선 검증 통과 (랜덤 가중치이므로 추천 내용 자체는 의미 없음)")
