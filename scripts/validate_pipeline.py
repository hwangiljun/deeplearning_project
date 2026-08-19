"""파이프라인이 실제로 누수를 잡았는지 로컬 데이터로 검증한다."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (CONTEXT_FEATURES, OUTCOMES, PITCH_FEATURES, define_outcome,
                      load_raw)
from src.prepare import prepare

# 400경기 전체를 담은 샘플. 타석이 온전해 시퀀스 검증에 그대로 쓸 수 있다.
SAMPLE = Path("data/sample.parquet")

print("=" * 70)
print("1. 정렬 검증 — 원본은 역시간순, load_raw 이후엔 시간순이어야 한다")
print("=" * 70)
df = load_raw(SAMPLE)
print(f"   총 {len(df):,}구")

pa = df[(df.game_pk == df.game_pk.iloc[0])].head(12)
print("   첫 경기 앞 12구 (at_bat_number, pitch_number):")
print("   ", list(zip(pa.at_bat_number.tolist(), pa.pitch_number.tolist())))

g = df.groupby(["game_pk", "at_bat_number"])["pitch_number"]
bad = (~g.apply(lambda s: s.is_monotonic_increasing)).sum()
print(f"   타석 내 pitch_number 가 오름차순이 아닌 타석 수: {bad}  (0 이어야 정상)")

print()
print("=" * 70)
print("2. 결과 라벨 — 버려지는 투구가 얼마나 되나")
print("=" * 70)
oc = define_outcome(df)
print(f"   매핑 실패(제거 대상): {oc.isna().sum():,}구 ({100*oc.isna().mean():.2f}%)")
print("   클래스 분포:")
vc = oc.value_counts()
for k, v in vc.items():
    print(f"     {k:18s} {v:8,}  {100*v/len(oc):5.2f}%")

print()
print("=" * 70)
print("3. 시퀀스 조립 (seq_len=6, 2024 학습 / 2025 테스트)")
print("=" * 70)
train, test, enc = prepare(SAMPLE, train_seasons=[2024],
                           test_seasons=[2025], seq_len=6)
print(f"   학습 {len(train):,}구 / 테스트 {len(test):,}구")
print(f"   cat {train.cat.shape}  num {train.num.shape}  mask {train.mask.shape}  ctx {train.ctx.shape}")
print(f"   범주 사전 크기: {enc.cat_sizes}")
print(f"   유효 시퀀스 길이 분포: {np.bincount(train.mask.sum(1))[1:]}")
print(f"   (1구째 타석이 많으므로 길이 1 이 가장 많은 게 정상)")

print()
print("=" * 70)
print("4. 누수 점검 — 시퀀스에 '미래'가 섞이지 않았는가")
print("=" * 70)
m = train.meta
# 시퀀스 마지막 원소는 타깃 자신이어야 한다
print(f"   mask 의 마지막 칸이 항상 True 인가: {bool(train.mask[:, -1].all())}")
# 같은 타석 안에서만 참조하는지: pitch_number 가 k 인 투구의 유효 길이는 min(k, 6)
expected = np.minimum(m.pitch_number.to_numpy(), 6)
actual = train.mask.sum(1)
print(f"   유효 길이 == min(pitch_number, 6) 인 비율: {100*(expected == actual).mean():.2f}%  (100% 여야 정상)")
# 카운트는 시퀀스가 아니라 ctx 에만 있다 → 카운트 기반 누수 채널 자체가 없다
print(f"   시퀀스 피처 목록: {PITCH_FEATURES}")
print(f"   -> balls/strikes 가 시퀀스에 없으므로, 다음 투구의 카운트로 정답을 읽는 경로가 차단됨")

print()
print("=" * 70)
print("5. 스케일 점검")
print("=" * 70)
valid = train.mask.reshape(-1)
flat = train.num.reshape(-1, len(PITCH_FEATURES))[valid]
print("   투구 물리량 (학습 구간 표준화 후):")
for i, name in enumerate(PITCH_FEATURES):
    print(f"     {name:10s} mean={flat[:, i].mean():+.3f} std={flat[:, i].std():.3f}"
          f" min={flat[:, i].min():+.2f} max={flat[:, i].max():+.2f}")
print(f"   NaN 남아있나: num={np.isnan(train.num).any()} ctx={np.isnan(train.ctx).any()}")

print()
print("=" * 70)
print("6. delta_run_exp (추천 점수의 기반)")
print("=" * 70)
re = pd.Series(train.run_exp)
y = pd.Series(train.y)
print("   결과별 평균 기대 실점 변화량 (투수 관점, 음수가 투수에게 유리):")
for i, name in enumerate(OUTCOMES):
    sel = re[y == i]
    if len(sel):
        print(f"     {name:18s} n={len(sel):8,}  mean={sel.mean():+.4f}")
