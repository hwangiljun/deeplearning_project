"""참조 테이블을 만들고 내용을 점검한다.

    python scripts/build_tables.py [--seasons 2024] [--out models/tables.pkl]
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (OUTCOMES, OUTCOME_TO_IDX, build_context_features,
                      build_pitch_features, compute_batter_stats, define_outcome,
                      load_raw)
from src.features import build_reference_tables, in_strike_zone

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/sample.parquet")
ap.add_argument("--seasons", type=int, nargs="*", default=None)
ap.add_argument("--out", default=None)
args = ap.parse_args()

paths = sorted(Path(args.raw).glob("*.parquet")) if Path(args.raw).is_dir() else [Path(args.raw)]
print(f"원본: {[p.name for p in paths]}")

df = load_raw(paths)
df["outcome"] = define_outcome(df)
df = df[df["outcome"].notna()].reset_index(drop=True)
df["outcome_idx"] = df["outcome"].map(OUTCOME_TO_IDX).astype(np.int64)
df = build_pitch_features(df)
df = build_context_features(df, compute_batter_stats(df))
print(f"{len(df):,}구 / 시즌 {sorted(df.season.unique().tolist())}")

tables = build_reference_tables(df, seasons=args.seasons)

print("\n" + "=" * 72)
print("1. 투수 레퍼토리 — 하드코딩 상수를 대체한다")
print("=" * 72)
rep = tables.repertoire
print(f"   (투수, 구종) 조합 {len(rep):,}개 / 투수 {rep.pitcher.nunique():,}명")
print(f"   투수당 평균 구종 수: {rep.groupby('pitcher').size().mean():.1f}")
print("\n   같은 슬라이더라도 투수마다 이렇게 다르다 (SL 구사 상위 6명):")
sl = rep[rep.pitch_type == "SL"].nlargest(6, "n")
for _, r in sl.iterrows():
    print(f"     투수 {int(r.pitcher)}  {r.speed:5.1f}mph  {r.spin_rate:6.0f}rpm  "
          f"수평무브 {r.pfx_x:+.2f}  구사율 {r.usage*100:4.1f}%")
print("\n   (이전 앱은 모든 투수에게 SL = 84.5mph / 2450rpm 을 썼다)")
print("\n   리그 평균 (미등록 투수 대체값):")
for _, r in tables.league_repertoire.nlargest(8, "n").iterrows():
    print(f"     {r.pitch_type:4s} {r.speed:5.1f}mph {r.spin_rate:6.0f}rpm  구사율 {r.usage*100:5.2f}%")

print("\n" + "=" * 72)
print("2. 카운트별 결과 가치 (기대 실점 변화량, 투수는 최소화)")
print("=" * 72)
cv = tables.count_value
hdr = "   count |" + "".join(f"{o[:9]:>10s}" for o in OUTCOMES[:6])
print(hdr)
for b in range(4):
    for s in range(3):
        cs = b * 3 + s
        row = "".join(f"{cv[cs, i]:+10.3f}" for i in range(6))
        print(f"   {b}-{s}   |{row}")
print("\n   같은 'ball' 인데 0-0 에서 %+.3f, 3-2 에서 %+.3f  (볼넷이 되므로)"
      % (cv[0, OUTCOME_TO_IDX['ball']], cv[11, OUTCOME_TO_IDX['ball']]))
print("   같은 'called_strike' 인데 0-0 에서 %+.3f, 0-2 에서 %+.3f  (삼진이 되므로)"
      % (cv[0, OUTCOME_TO_IDX['called_strike']], cv[2, OUTCOME_TO_IDX['called_strike']]))
print("\n   -> 이전의 good/bad 이분법은 이 차이를 전혀 담지 못했다")

print("\n" + "=" * 72)
print("3. 위치별 기대 실점 (구속밴드 × 격자). 투수에게 음수가 유리")
print("=" * 72)
lv = tables.location_value
gx, gz = tables.grid_x, tables.grid_z
for b, name in enumerate(["low (<86.7)", "mid (86.7-92.5)", "high (>=92.5)"]):
    print(f"\n   [{name}]   행=높이(위→아래), 열=좌우(몸쪽→바깥쪽)")
    for zi in range(len(gz) - 1, -1, -1):
        cells = "".join(f"{lv[b, zi, xi]:+7.3f}" for xi in range(len(gx)))
        zone = "존" if 0 <= gz[zi] <= 1 else "  "
        print(f"     z={gz[zi]:+5.2f} {zone} {cells}")

best = np.unravel_index(np.argmin(lv), lv.shape)
worst = np.unravel_index(np.argmax(lv), lv.shape)
print(f"\n   가장 유리한 칸: {['low','mid','high'][best[0]]} 구속, "
      f"x={gx[best[2]]:+.2f} z={gz[best[1]]:+.2f}  ({lv[best]:+.3f})")
print(f"   가장 불리한 칸: {['low','mid','high'][worst[0]]} 구속, "
      f"x={gx[worst[2]]:+.2f} z={gz[worst[1]]:+.2f}  ({lv[worst]:+.3f})")

if args.out:
    p = tables.save(args.out)
    print(f"\n저장: {p}")
