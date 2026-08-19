"""데이터 → 모델 → 학습 한 바퀴가 실제로 도는지 작은 표본으로 확인한다."""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import OUTCOMES
from src.prepare import prepare
from src.model import ModelConfig
from src.train import TrainConfig, train, evaluate, to_tensors
import torch
from torch.utils.data import DataLoader

SAMPLE = Path("data/sample.parquet")

print("[1/4] 전처리")
t0 = time.time()
tr, te, enc = prepare(SAMPLE, train_seasons=[2024],
                      test_seasons=[2025], seq_len=6)
print(f"      학습 {len(tr):,} / 테스트 {len(te):,}  ({time.time()-t0:.0f}s)")
print(f"      state 범위: count {tr.state[:,0].min()}~{tr.state[:,0].max()}, "
      f"baseout {tr.state[:,1].min()}~{tr.state[:,1].max()}")

print("[2/4] 표본 축소 (스모크 테스트용)")
def head(d, n):
    from src.data import SequenceData
    return SequenceData(d.cat[:n], d.num[:n], d.mask[:n], d.ctx[:n],
                        d.state[:n], d.y[:n], d.run_exp[:n], d.meta.iloc[:n])
tr_s, te_s = head(tr, 60000), head(te, 20000)

print("[3/4] 모델 구성")
mcfg = ModelConfig(cat_sizes=enc.cat_sizes, d_model=128, num_layers=2,
                   dim_feedforward=512, nhead=4, seq_len=6,
                   n_classes=len(OUTCOMES))
tcfg = TrainConfig(epochs=2, batch_size=512, patience=2)

print("[4/4] 2에폭 학습")
model, hist, best = train(tr_s, mcfg, tcfg, device="cpu")

print("\n--- 홀드아웃 평가 ---")
loader = DataLoader(to_tensors(te_s), batch_size=512)
m, p, t = evaluate(model, loader, "cpu", len(OUTCOMES))
for k, v in m.items():
    print(f"   {k:12s} {v:.4f}")

print("\n--- 예측 확률 분포 (다수 클래스로 붕괴하지 않았는지) ---")
print("   평균 예측 확률:")
for i, name in enumerate(OUTCOMES):
    print(f"     {name:18s} pred={p[:,i].mean():.4f}  actual={(t==i).mean():.4f}")
print("\n스모크 테스트 통과")
