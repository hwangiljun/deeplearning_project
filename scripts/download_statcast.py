"""Statcast 원본 데이터를 월 단위로 내려받아 data/raw/ 에 저장한다.

월별 parquet 으로 나눠 저장하므로 중간에 끊겨도 이어받을 수 있다.
사용법:
    python scripts/download_statcast.py 2025-03 2025-10
    python scripts/download_statcast.py 2024-09 2024-10
"""
import sys
import calendar
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
from pybaseball import statcast

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def months(start: str, end: str):
    """'YYYY-MM' 범위를 (연, 월) 목록으로 펼친다."""
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def fetch(year: int, month: int) -> Path | None:
    out = RAW / f"statcast_{year}_{month:02d}.parquet"
    if out.exists():
        print(f"[skip] {out.name} (이미 존재)", flush=True)
        return out

    last = calendar.monthrange(year, month)[1]
    start, end = f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last}"
    print(f"[받는 중] {start} ~ {end}", flush=True)

    df = statcast(start_dt=start, end_dt=end)
    if df is None or df.empty:
        print("  -> 데이터 없음 (비시즌)", flush=True)
        return None

    # 정규시즌만. game_type: R=정규, F/D/L/W=포스트시즌, S=시범경기
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"]
    if df.empty:
        print("  -> 정규시즌 경기 없음", flush=True)
        return None

    RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"  -> {len(df):,}구 저장 ({out.name}, {out.stat().st_size/1e6:.1f}MB)", flush=True)
    return out


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(1)
    total = 0
    for y, m in months(sys.argv[1], sys.argv[2]):
        p = fetch(y, m)
        if p is not None:
            total += len(pd.read_parquet(p, columns=["game_pk"]))
    print(f"\n합계 {total:,}구", flush=True)


if __name__ == "__main__":
    main()
