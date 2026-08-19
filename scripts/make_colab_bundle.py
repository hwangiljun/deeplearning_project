"""코랩에 올릴 코드 묶음(colab_src.zip)을 만든다.

저장소를 아직 푸시하지 않았을 때, notebooks/02_train.ipynb 의 첫 셀에서
이 zip 을 업로드하면 로컬과 완전히 같은 src/ 코드로 학습할 수 있다.

    python scripts/make_colab_bundle.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab_src.zip"

INCLUDE = ["src", "scripts"]
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for top in INCLUDE:
        for p in (ROOT / top).rglob("*.py"):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            z.write(p, p.relative_to(ROOT))
    for extra in ["requirements.txt"]:
        f = ROOT / extra
        if f.exists():
            z.write(f, extra)

print(f"{OUT}  ({OUT.stat().st_size/1024:.1f}KB)")
for n in zipfile.ZipFile(OUT).namelist():
    print("  ", n)
