"""코랩에서 받은 산출물 zip 을 models/ 에 설치한다.

    python scripts/install_artifacts.py                    # 다운로드 폴더에서 자동 탐색
    python scripts/install_artifacts.py path/to/file.zip   # 직접 지정

압축 안에 들어 있는 구조를 그대로 ``models/`` 아래에 편다.
02_train.ipynb 산출물이면 ``main/`` 이, 03_build_artifacts.ipynb 산출물이면
``tables.pkl`` / ``profiles.pkl`` 등이 들어 있다. 둘 다 순서에 상관없이
같은 자리에 겹쳐 설치하면 된다.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

# 앱이 실제로 찾는 파일들
EXPECTED = {
    "main/model.pth": "학습된 모델",
    "main/encoders.pkl": "인코더",
    "main/model_config.json": "모델 설정",
    "main/metadata.json": "학습 이력·테스트 지표",
    "main/evaluation.pkl": "혼동행렬·캘리브레이션 (03번 노트북)",
    "tables.pkl": "참조 테이블 (03번 노트북)",
    "profiles.pkl": "선수·팀 프로파일 (03번 노트북)",
    "batter_stats.pkl": "타자 직전시즌 성적 (03번 노트북)",
    "ablation_table.csv": "어블레이션 표",
}


def find_zip() -> Path | None:
    """다운로드 폴더에서 가장 최근 산출물 zip 을 찾는다."""
    candidates: list[Path] = []
    for d in [Path.home() / "Downloads", Path.home() / "다운로드", ROOT]:
        if d.exists():
            candidates += [p for p in d.glob("*.zip")
                           if "artifact" in p.name.lower()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else find_zip()
    if src is None or not src.exists():
        print("산출물 zip 을 찾지 못했습니다.")
        print("  코랩에서 받은 artifacts.zip / dashboard_artifacts.zip 경로를 직접 지정하세요:")
        print("  python scripts/install_artifacts.py C:/Users/.../Downloads/artifacts.zip")
        raise SystemExit(1)

    print(f"설치 원본: {src}  ({src.stat().st_size/1e6:.1f} MB)")
    MODELS.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(src) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        print(f"압축 내용 {len(names)}개:")
        for n in sorted(names):
            info = z.getinfo(n)
            print(f"  {info.file_size/1e6:8.2f} MB  {n}")
        z.extractall(MODELS)

    print(f"\n{MODELS} 에 설치 완료\n")

    print("앱이 찾는 파일 점검:")
    missing = []
    for rel, desc in EXPECTED.items():
        p = MODELS / rel
        if p.exists():
            print(f"  [있음]  {rel:26s} {p.stat().st_size/1e6:7.2f} MB  {desc}")
        else:
            print(f"  [없음]  {rel:26s} {'':7s}     {desc}")
            missing.append(rel)

    print()
    if not missing:
        print("모든 산출물이 준비됐습니다. `streamlit run app.py` 로 확인하세요.")
    else:
        need_train = [m for m in missing if m.startswith("main/") and "evaluation" not in m]
        need_build = [m for m in missing if not m.startswith("main/") or "evaluation" in m]
        if need_train:
            print("→ 02_train.ipynb 의 산출물이 아직 없습니다:", ", ".join(need_train))
        if need_build:
            print("→ 03_build_artifacts.ipynb 의 산출물이 아직 없습니다:", ", ".join(need_build))
        print("  (없어도 앱은 실행됩니다. 해당 페이지만 비활성화됩니다.)")

    print("\n앱이 이미 떠 있다면 우상단 메뉴 → Clear cache 후 새로고침하세요.")


if __name__ == "__main__":
    main()
