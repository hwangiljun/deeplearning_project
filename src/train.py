"""학습 루프와 어블레이션.

이전 학습의 문제와 대응
-----------------------
* **무작위 분할**: ``train_test_split(..., stratify=y)`` 로 셔플 분할했고,
  stride 2 슬라이딩 윈도우라 학습/검증 표본이 8개 중 6개를 공유했다.
  → 시즌(또는 날짜) 단위로 자른다. 검증셋도 학습 구간의 **뒷부분 날짜**에서 뗀다.
* **불균형 이중 보정**: ``WeightedRandomSampler`` 와 Focal Loss 를 동시에 썼다.
  출력 확률이 실제 분포가 아니라 균등화된 분포를 가리키게 되어 캘리브레이션이
  망가진다. → 기본은 순수 CrossEntropy. 샘플러는 쓰지 않는다.
* **최고 에폭 체크포인트를 그대로 보고**: 30에폭 중 9에폭의 67.52% 를 성능으로
  보고했으나 이후 21에폭은 한 번도 그 값을 넘지 못했다(노이즈 피크).
  → 조기 종료 기준을 검증 log-loss 로 두고, 테스트는 별도 시즌에서 한 번만 잰다.
* **지표가 Accuracy 하나**: 불균형 다중분류에서 정확도는 다수 클래스가 지배한다.
  → macro-F1, log-loss, macro AUC 를 함께 본다. 이 시스템은 확률로 순위를
    매기므로 log-loss/캘리브레이션이 정확도보다 본질적이다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .data import OUTCOMES, SequenceData
from .model import ContextAwareTransformer, FocalLoss, ModelConfig


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 6              # 검증 log-loss 기준 조기 종료
    loss: str = "ce"               # "ce" | "focal"
    focal_gamma: float = 2.0
    run_value_weight: float = 0.3  # 기대 실점 보조 회귀 손실 가중
    unk_dropout: float = 0.05      # 학습 중 선수 ID 를 UNK 로 바꿀 확률
    val_fraction: float = 0.15     # 학습 구간 뒷부분에서 검증셋을 뗀다
    seed: int = 42
    grad_clip: float = 1.0
    auc_every_epoch: bool = False   # macro AUC 는 비싸다. 에폭마다 재지 않는다.


def to_tensors(d: SequenceData, idx=None) -> TensorDataset:
    """SequenceData → TensorDataset.

    ``idx`` 를 주면 **그 인덱스로 미리 잘라서** 텐서를 만든다.
    ``torch.utils.data.Subset`` 으로 감싸면 DataLoader 가 샘플을 하나씩
    파이썬 레벨로 꺼내므로, 수십만 건에서는 CPU 가 병목이 되어 GPU 가 논다.
    미리 자르면 배치 슬라이싱이 C 레벨에서 끝난다.
    """
    arrays = [d.cat, d.num, d.mask, d.ctx, d.state, d.y, d.run_exp]
    if idx is not None:
        idx = np.asarray(idx)
        arrays = [a[idx] for a in arrays]
    return TensorDataset(*[torch.from_numpy(np.ascontiguousarray(a)) for a in arrays])


def temporal_val_split(d: SequenceData, fraction: float):
    """학습 구간을 날짜 기준으로 잘라 (앞=학습, 뒤=검증) 인덱스를 만든다.

    무작위 분할과 달리 같은 타석의 이웃 투구가 양쪽으로 갈리지 않는다.
    """
    dates = d.meta["game_date"].to_numpy()
    order = np.argsort(dates, kind="mergesort")
    cut = int(len(order) * (1 - fraction))
    # 경계 날짜가 양쪽에 걸치지 않도록 날짜 단위로 자른다
    boundary = dates[order[cut]]
    train_idx = np.where(dates < boundary)[0]
    val_idx = np.where(dates >= boundary)[0]
    return train_idx, val_idx


@torch.no_grad()
def evaluate(model, loader, device, n_classes: int, with_auc: bool = True) -> dict:
    model.eval()
    probs, targets = [], []
    for batch in loader:
        cat, num, mask, ctx, state, y, _ = [b.to(device) for b in batch]
        logits, _ = model(cat, num, mask, ctx, state)
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
        targets.append(y.cpu().numpy())

    p = np.concatenate(probs)
    t = np.concatenate(targets)
    pred = p.argmax(1)

    labels = list(range(n_classes))
    out = {
        "accuracy": float((pred == t).mean()),
        "macro_f1": float(f1_score(t, pred, average="macro", labels=labels,
                                   zero_division=0)),
        "log_loss": float(log_loss(t, p, labels=labels)),
    }
    if with_auc:
        try:
            out["macro_auc"] = float(roc_auc_score(t, p, multi_class="ovr",
                                                   average="macro", labels=labels))
        except ValueError:
            out["macro_auc"] = float("nan")
    else:
        out["macro_auc"] = float("nan")
    return out, p, t


def train(train_data: SequenceData,
          model_cfg: ModelConfig,
          train_cfg: TrainConfig,
          device: str | None = None,
          verbose: bool = True):
    """학습 후 (모델, 학습 이력, 최종 검증 지표) 를 돌려준다."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    tr_idx, va_idx = temporal_val_split(train_data, train_cfg.val_fraction)
    tr_loader = DataLoader(to_tensors(train_data, tr_idx),
                           batch_size=train_cfg.batch_size, shuffle=True,
                           drop_last=True)
    va_loader = DataLoader(to_tensors(train_data, va_idx),
                           batch_size=train_cfg.batch_size * 2, shuffle=False)

    model = ContextAwareTransformer(model_cfg).to(device)

    if train_cfg.loss == "focal":
        criterion = FocalLoss(gamma=train_cfg.focal_gamma).to(device)
    else:
        criterion = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                            weight_decay=train_cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=2)

    history = []
    best = {"log_loss": float("inf")}
    best_state = None
    bad_epochs = 0

    if verbose:
        print(f"학습 {len(tr_idx):,} / 검증 {len(va_idx):,} | device={device}")
        print(f"파라미터 {sum(p.numel() for p in model.parameters()):,}개")

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0

        for batch in tr_loader:
            cat, num, mask, ctx, state, y, rv = [b.to(device) for b in batch]

            # 학습 중 일부 선수 ID 를 UNK(0) 으로 바꿔 UNK 임베딩을 실제로 훈련시킨다.
            # 이렇게 해야 앱에서 처음 보는 선수가 들어와도 무너지지 않는다.
            if train_cfg.unk_dropout > 0:
                drop = torch.rand(cat.shape[0], 1, 2, device=device) < train_cfg.unk_dropout
                cat[:, :, :2] = torch.where(drop.expand(-1, cat.shape[1], -1),
                                            torch.zeros_like(cat[:, :, :2]),
                                            cat[:, :, :2])

            opt.zero_grad()
            logits, pred_rv = model(cat, num, mask, ctx, state)
            loss = criterion(logits, y)
            if pred_rv is not None and train_cfg.run_value_weight > 0:
                loss = loss + train_cfg.run_value_weight * F.mse_loss(pred_rv, rv)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()

            total += loss.item() * y.size(0)
            n += y.size(0)

        metrics, _, _ = evaluate(model, va_loader, device, model_cfg.n_classes,
                                 with_auc=train_cfg.auc_every_epoch)
        sched.step(metrics["log_loss"])
        metrics["epoch"] = epoch
        metrics["train_loss"] = total / max(n, 1)
        metrics["seconds"] = time.time() - t0
        history.append(metrics)

        line = (f"  epoch {epoch:2d} | loss {metrics['train_loss']:.4f} "
                f"| val logloss {metrics['log_loss']:.4f} "
                f"acc {metrics['accuracy']*100:.2f}% "
                f"macroF1 {metrics['macro_f1']:.4f} "
                f"({metrics['seconds']:.0f}s)")
        if verbose:
            print(line, flush=True)
        else:
            # 조용한 모드에서도 살아 있다는 신호는 남긴다. 어블레이션을 돌릴 때
            # 30분 넘게 출력이 없으면 멈춘 것처럼 보인다.
            print(f"    · epoch {epoch} logloss {metrics['log_loss']:.4f} "
                  f"({metrics['seconds']:.0f}s)", flush=True)

        if metrics["log_loss"] < best["log_loss"] - 1e-5:
            best = metrics
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= train_cfg.patience:
                if verbose:
                    print(f"  조기 종료 (검증 log-loss 가 {train_cfg.patience}에폭 개선 없음)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best


def save_artifacts(out_dir, model, model_cfg: ModelConfig, encoders,
                   history=None, metrics=None, extra=None) -> Path:
    """앱과 평가가 쓰는 산출물을 한 폴더에 모아 저장한다."""
    import joblib

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out / "model.pth")
    joblib.dump(encoders, out / "encoders.pkl")
    (out / "model_config.json").write_text(
        json.dumps(model_cfg.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8")

    meta = {"outcomes": OUTCOMES}
    if history is not None:
        meta["history"] = history
    if metrics is not None:
        meta["best_val"] = metrics
    if extra:
        meta.update(extra)
    (out / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    return out


# --------------------------------------------------------------------------
# 어블레이션
# --------------------------------------------------------------------------
def ablation_grid(base_cat_sizes: dict) -> dict[str, ModelConfig]:
    """보고서 표에 넣을 어블레이션 구성.

    이전 보고서는 제안 모델과 베이스라인의 d_model·레이어 수·손실함수·에폭·
    학습률이 전부 달라서 어느 요소가 기여했는지 분리할 수 없었다.
    여기서는 **한 번에 하나씩만** 바꾼다.
    """
    def cfg(**kw):
        return ModelConfig(cat_sizes=base_cat_sizes, **kw)

    # 중요도 순으로 둔다. 시간이 부족해 중간에 끊더라도 보고서에 꼭 필요한
    # 세 줄(full / no_context_skip / seq_len_1)은 먼저 확보되도록.
    return {
        # 제안 모델 (전체)
        "full":            cfg(),
        # 맥락 스킵 연결 제거 → 보고서의 핵심 주장 검증
        "no_context_skip": cfg(use_context_skip=False),
        # 시퀀스 길이 1 → 직전 투구 정보를 완전히 제거.
        # full 이 이걸 이기지 못하면 "투구 배합이 중요하다" 는 전제가 무너진다.
        "seq_len_1":       cfg(seq_len=1),
        # 상태 임베딩 제거 → 카운트/베이스아웃 임베딩의 기여
        "no_state_embed":  cfg(use_state_embed=False),
        # 마지막 토큰만 사용 (이전 구현 방식)
        "last_token":      cfg(pooling="last"),
        # 시퀀스 길이 민감도
        "seq_len_3":       cfg(seq_len=3),
        "seq_len_10":      cfg(seq_len=10),
    }
