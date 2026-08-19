"""모델 평가 — 보고서 그림과 대시보드 '모델 성능' 페이지에 들어갈 수치.

이전 보고서는 Top-1 정확도 하나로 성능을 보고했다. 11개 불균형 클래스에서
정확도는 다수 클래스가 지배하므로, Focal Loss 로 희귀 클래스를 살렸다는
주장을 **정확도로는 보여줄 수가 없다.** 게다가 이 시스템은 확률로 후보 순위를
매기므로 확률의 질(캘리브레이션)이 정확도보다 본질적이다.

여기서 만드는 것
  * 클래스별 precision / recall / F1 / AUC / 표본수
  * 혼동 행렬
  * 캘리브레이션 곡선과 ECE
  * 검증셋에서 적합한 temperature (필요할 때만 쓰는 보정 계수)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (confusion_matrix, log_loss, precision_recall_fscore_support,
                             roc_auc_score)

from .data import OUTCOMES


@torch.no_grad()
def predict(model, loader, device: str = "cuda"):
    """(확률 (n,C), 정답 (n,), 로짓 (n,C)) 을 돌려준다."""
    model.eval()
    logits, targets = [], []
    for batch in loader:
        cat, num, mask, ctx, state, y, _ = [b.to(device) for b in batch]
        out, _ = model(cat, num, mask, ctx, state)
        logits.append(out.cpu())
        targets.append(y.cpu())
    logits = torch.cat(logits)
    targets = torch.cat(targets)
    return F.softmax(logits, dim=1).numpy(), targets.numpy(), logits.numpy()


def per_class_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    """클래스별 지표. 희귀 클래스가 실제로 잡히는지 여기서만 보인다."""
    pred = probs.argmax(1)
    labels = list(range(len(OUTCOMES)))
    p, r, f1, sup = precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=0)

    auc = []
    for i in labels:
        pos = (y == i).astype(int)
        # 한 클래스만 존재하면 AUC 가 정의되지 않는다
        auc.append(roc_auc_score(pos, probs[:, i]) if 0 < pos.sum() < len(pos)
                   else float("nan"))

    return {"outcome": list(OUTCOMES), "precision": p.tolist(), "recall": r.tolist(),
            "f1": f1.tolist(), "support": sup.tolist(), "auc": auc}


def calibration(probs: np.ndarray, y: np.ndarray, n_bins: int = 15):
    """상위 예측 기준 신뢰도 보정 곡선과 ECE.

    각 구간에서 '모델이 말한 확신' 과 '실제 맞힌 비율' 을 비교한다.
    둘이 어긋나면 화면에 표시되는 확률이 거짓말을 하고 있다는 뜻이다.
    """
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)

    prob_pred, prob_true, weights = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() < 20:          # 표본이 너무 적은 구간은 곡선에서 뺀다
            continue
        prob_pred.append(float(conf[m].mean()))
        prob_true.append(float(correct[m].mean()))
        weights.append(float(m.sum()))

    prob_pred = np.array(prob_pred)
    prob_true = np.array(prob_true)
    w = np.array(weights)
    ece = float((w * np.abs(prob_true - prob_pred)).sum() / w.sum()) if len(w) else float("nan")
    return prob_true, prob_pred, ece


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """검증셋에서 NLL 을 최소화하는 temperature 를 찾는다 (Guo et al., 2017).

    이전 앱은 ``temperature = 5.0`` 을 손으로 박아 넣어 분포를 균등에 가깝게
    뭉갰다. 보정이 필요하다면 이렇게 **데이터에서 적합**해야 하고, 필요 없으면
    (T ≈ 1) 손대지 않는 것이 맞다.
    """
    lg = torch.tensor(logits, dtype=torch.float32)
    t = torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)      # T = exp(log_t), 양수 보장

    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def evaluate_full(model, test_loader, val_loader=None, device: str = "cuda") -> dict:
    """평가 일괄 실행. 결과 dict 는 그대로 evaluation.pkl 로 저장한다."""
    probs, y, logits = predict(model, test_loader, device)
    labels = list(range(len(OUTCOMES)))

    prob_true, prob_pred, ece = calibration(probs, y)
    result = {
        "probs_mean": probs.mean(0).tolist(),
        "actual_freq": np.bincount(y, minlength=len(OUTCOMES)).astype(float).tolist(),
        "confusion": confusion_matrix(y, probs.argmax(1), labels=labels),
        "per_class": per_class_metrics(probs, y),
        "prob_true": prob_true, "prob_pred": prob_pred, "ece": ece,
        "log_loss": float(log_loss(y, probs, labels=labels)),
        "accuracy": float((probs.argmax(1) == y).mean()),
        "n_test": int(len(y)),
    }

    if val_loader is not None:
        _, yv, lv = predict(model, val_loader, device)
        T = fit_temperature(lv, yv)
        result["temperature"] = T
        # 보정을 적용했을 때 실제로 나아지는지 확인한다
        scaled = F.softmax(torch.tensor(logits) / T, dim=1).numpy()
        result["log_loss_scaled"] = float(log_loss(y, scaled, labels=labels))
        pt2, pp2, ece2 = calibration(scaled, y)
        result["ece_scaled"] = ece2

    return result


def summary_text(result: dict) -> str:
    """콘솔에 찍을 요약."""
    lines = [
        f"테스트 표본 {result['n_test']:,}",
        f"log-loss {result['log_loss']:.4f} · 정확도 {result['accuracy']*100:.2f}%",
        f"ECE {result['ece']:.4f}  (0 에 가까울수록 확률이 정직하다)",
    ]
    if "temperature" in result:
        lines.append(
            f"temperature {result['temperature']:.3f} → "
            f"보정 후 log-loss {result['log_loss_scaled']:.4f} · ECE {result['ece_scaled']:.4f}")
        if abs(result["temperature"] - 1.0) < 0.05:
            lines.append("  T가 1에 가깝다 = 이미 보정되어 있다. 별도 보정 불필요.")

    pc = result["per_class"]
    lines.append("")
    lines.append(f"{'클래스':18s} {'표본':>8s} {'정밀도':>7s} {'재현율':>7s} {'F1':>7s} {'AUC':>7s}")
    for i, name in enumerate(pc["outcome"]):
        lines.append(f"{name:18s} {pc['support'][i]:8,} {pc['precision'][i]:7.3f} "
                     f"{pc['recall'][i]:7.3f} {pc['f1'][i]:7.3f} {pc['auc'][i]:7.3f}")
    return "\n".join(lines)
