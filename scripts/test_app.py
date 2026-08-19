"""모든 페이지를 헤드리스로 렌더해 예외가 없는지 확인한다."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

PAGES = ["투구 추천", "타자 분석", "투수 분석", "팀 분석", "모델 성능"]

fails = 0
for page in PAGES:
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=180)
    at.run()
    if at.exception:
        print(f"[FAIL] 초기 로드: {at.exception[0].message}")
        fails += 1
        break

    at.sidebar.radio[0].set_value(page).run()
    if at.exception:
        print(f"[FAIL] {page}")
        for e in at.exception:
            print("   ", e.message.strip().splitlines()[-1] if e.message else e)
            if e.stack_trace:
                print("   ", "\n    ".join(e.stack_trace[-6:]))
        fails += 1
    else:
        n_warn = len(at.warning)
        n_info = len(at.info)
        print(f"[OK]   {page:10s}  차트 {len(at.get('plotly_chart'))} · "
              f"표 {len(at.dataframe)} · 경고 {n_warn} · 안내 {n_info}")

print()
print("전체 통과" if fails == 0 else f"{fails}개 페이지 실패")
sys.exit(1 if fails else 0)
