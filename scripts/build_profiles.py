"""대시보드용 선수·팀 집계를 만들고 점검한다.

    python scripts/build_profiles.py --raw data/raw --out models/profiles.pkl
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (OUTCOME_TO_IDX, build_context_features, build_pitch_features,
                      compute_batter_stats, define_outcome, load_raw)
from src.profiles import build_profiles

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/sample.parquet")
ap.add_argument("--names", default="models/player_mapping.pkl",
                help="없으면 models/legacy/player_mapping.pkl 을 쓴다")
ap.add_argument("--min-pitches", type=int, default=200)
ap.add_argument("--out", default=None)
a = ap.parse_args()

paths = sorted(Path(a.raw).glob("*.parquet")) if Path(a.raw).is_dir() else [Path(a.raw)]
df = load_raw(paths)
df["outcome"] = define_outcome(df)
df = df[df["outcome"].notna()].reset_index(drop=True)
df["outcome_idx"] = df["outcome"].map(OUTCOME_TO_IDX).astype(np.int64)
df = build_pitch_features(df)
df = build_context_features(df, compute_batter_stats(df))

name_map = {}
names_path = Path(a.names)
if not names_path.exists():
    names_path = Path("models/legacy/player_mapping.pkl")
a.names = str(names_path)
if Path(a.names).exists():
    import joblib
    name_map = joblib.load(a.names)
    print(f"이름 매핑 {len(name_map):,}건 로드")

pr = build_profiles(df, name_map, min_pitches=a.min_pitches)

print("\n" + "=" * 70)
print("선수 명부")
print("=" * 70)
d = pr.directory
print(f"  총 {len(d):,}행  (투수 {(d.role=='pitcher').sum():,} / 타자 {(d.role=='batter').sum():,})")
print(f"  이름 미확인: {d.name.str.startswith('ID ').sum():,}건")
print("\n  샘플:")
for _, r in d[d.role == 'pitcher'].nlargest(4, 'pitches').iterrows():
    print(f"    투수 {r.label:32s} {r.hand}투  {r.pitches:,}구  ({r.season})")
for _, r in d[d.role == 'batter'].nlargest(4, 'pitches').iterrows():
    print(f"    타자 {r.label:32s} {r.hand}타  {r.pitches:,}구  ({r.season})")

print("\n" + "=" * 70)
print("타자 요약 (표본 상위 5)")
print("=" * 70)
b = pr.batter_summary.merge(
    d[d.role == 'batter'][['season', 'player_id', 'name', 'team']],
    left_on=['season', 'batter'], right_on=['season', 'player_id'], how='left')
cols = ['name', 'team', 'pitches', 'run_value_per_100', 'whiff_rate',
        'chase_rate', 'k_rate', 'bb_rate']
print(b.nlargest(5, 'pitches')[cols].to_string(index=False,
      float_format=lambda v: f"{v:6.3f}"))

print("\n" + "=" * 70)
print("타자 존 히트맵 — 축소 효과 확인")
print("=" * 70)
key = max(pr.batter_zone, key=lambda k: pr.batter_zone[k][1].sum())
grid, cnt = pr.batter_zone[key]
nm = d[(d.season == key[0]) & (d.player_id == key[1]) & (d.role == 'batter')]
print(f"  {nm.name.iat[0] if len(nm) else key[1]} ({key[0]}) — 총 {cnt.sum():,}구")
print("  값 = 기대 실점(타자 관점, 양수가 타자에게 유리) / 괄호 = 표본수")
for zi in range(len(pr.grid_z) - 1, -1, -1):
    cells = "".join(f"{grid[zi,xi]:+6.3f}({cnt[zi,xi]:>3d})" for xi in range(len(pr.grid_x)))
    zone = "존" if 0 <= pr.grid_z[zi] <= 1 else "  "
    print(f"    z={pr.grid_z[zi]:+5.2f} {zone} {cells}")
print(f"\n  리그 평균 대비 최대 편차: {np.abs(grid - pr.league_zone).max():+.4f}")
print(f"  표본 0인 칸 개수: {(cnt==0).sum()} → 리그 평균으로 축소됨")

print("\n" + "=" * 70)
print("투수 카운트별 구사율 (예시)")
print("=" * 70)
u = pr.pitcher_usage
pid = u.groupby('pitcher')['n'].sum().idxmax()
sub = u[(u.pitcher == pid) & (u.season == u.season.max())]
piv = sub.pivot_table(index='count_state', columns='pitch_type', values='usage').fillna(0)
nm = d[(d.player_id == pid) & (d.role == 'pitcher')]
print(f"  {nm.name.iat[0] if len(nm) else pid}")
print("  카운트   " + "".join(f"{c:>7s}" for c in piv.columns))
for cs in piv.index:
    print(f"    {cs//3}-{cs%3}   " + "".join(f"{piv.loc[cs,c]*100:6.1f}%" for c in piv.columns))

print("\n" + "=" * 70)
print("팀 요약")
print("=" * 70)
t = pr.team_summary
t = t[t.season == t.season.max()].nsmallest(5, 'run_value_per_100')
print("  투수진 기대실점 상위 5팀 (100구당, 낮을수록 좋음):")
print(t[['team', 'pitches', 'run_value_per_100', 'whiff_rate', 'k_rate']].to_string(
    index=False, float_format=lambda v: f"{v:6.3f}"))

if a.out:
    print(f"\n저장: {pr.save(a.out)}")
