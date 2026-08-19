"""상황 인식형 Transformer (Context-Aware Transformer).

기본 골격은 이전 모델과 같다. 시퀀스는 Transformer 로, 현재 상황은 별도 경로로
처리한 뒤 출력 직전에 합친다. Takamido & Nakamoto (2026) 도 같은 이원 경로
구조를 쓰므로, 설계 방향 자체는 유지하고 아래 결함만 고쳤다.

이전 구현 대비 수정 사항
------------------------
* **최종 LayerNorm 추가**: ``norm_first=True`` (Pre-LN) 인데 마지막 정규화가
  없었다. 잔차 스트림이 정규화 없이 헤드로 들어가 학습이 불안정해진다.
* **패딩 마스크**: 타석 앞부분이 패딩인 시퀀스에 마스크를 씌운다. 이전에는
  마스크가 없어 존재하지 않는 공에도 어텐션이 걸렸다.
* **마스크드 평균 풀링**: 마지막 토큰만 쓰던 것을 유효 구간 평균으로 바꿨다
  (논문과 동일). 짧은 타석에서 특히 안정적이다.
* **맥락 경로 표현력**: 이전에는 원시 정수 6개를 ``Linear(6,32)+ReLU`` 에
  통과시켜서, 0-0 카운트·무사·주자없음이면 입력이 전부 0 이 되어
  ``ReLU(bias)`` 라는 상수만 남았다. 즉 가장 흔한 상황에서 스킵 연결의
  정보량이 0 이었다. 지금은 (a) 맥락 피처를 표준화하고 (b) 카운트 12상태와
  베이스-아웃 24상태를 임베딩으로 따로 넣는다.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """모델 형상. 어블레이션은 이 값만 바꿔 가며 돌린다."""

    cat_sizes: dict           # 범주형별 사전 크기 (UNK 포함)
    n_pitch_features: int = 8
    n_context_features: int = 15
    n_classes: int = 10

    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2       # 논문 튜닝 결과도 2개가 최적이었다
    dim_feedforward: int = 512
    dropout: float = 0.2
    seq_len: int = 6

    embed_dims: tuple = (32, 32, 8, 2, 2)  # batter, pitcher, pitch_type, stand, p_throws
    context_dim: int = 32

    # --- 어블레이션 스위치 ---
    use_context_skip: bool = True   # 맥락 경로(스킵 연결) 사용 여부
    use_state_embed: bool = True    # 카운트/베이스-아웃 상태 임베딩 사용 여부
    pooling: str = "mean"           # "mean" (마스크드 평균) | "last" (마지막 토큰)
    predict_run_value: bool = True  # 기대 실점 변화량 보조 회귀 헤드

    def to_dict(self) -> dict:
        return asdict(self)


CAT_ORDER = ["batter", "pitcher", "pitch_type", "stand", "p_throws"]


class ContextAwareTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # --- 엔티티 임베딩 ---
        self.embeddings = nn.ModuleDict()
        total_embed = 0
        for name, dim in zip(CAT_ORDER, cfg.embed_dims):
            # padding_idx=0 은 UNK. 처음 보는 선수는 여기로 온다.
            self.embeddings[name] = nn.Embedding(cfg.cat_sizes[name], dim)
            total_embed += dim

        # --- 시퀀스 경로 ---
        self.input_proj = nn.Linear(total_embed + cfg.n_pitch_features, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        # Pre-LN 에서는 마지막 LayerNorm 이 반드시 필요하다 (이전 구현엔 없었다).
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=cfg.num_layers, norm=nn.LayerNorm(cfg.d_model)
        )

        # --- 맥락 경로 ---
        ctx_in = cfg.n_context_features
        if cfg.use_state_embed:
            # 카운트 12상태(볼 0-3 × 스트라이크 0-2), 베이스-아웃 24상태(8×3)
            self.count_embed = nn.Embedding(12, 16)
            self.baseout_embed = nn.Embedding(24, 16)
            ctx_in += 32
        self.context_proj = nn.Sequential(
            nn.Linear(ctx_in, cfg.context_dim),
            nn.ReLU(),
        )

        # --- 출력 헤드 ---
        head_in = cfg.d_model + (cfg.context_dim if cfg.use_context_skip else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.LayerNorm(128),   # BatchNorm 은 배치 구성에 민감해 LayerNorm 으로 교체
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, cfg.n_classes)
        # 기대 실점 변화량을 직접 맞히는 보조 헤드. 추천 점수와 직결된다.
        self.run_value = nn.Linear(64, 1) if cfg.predict_run_value else None

    # ------------------------------------------------------------------
    def forward(self, x_cat, x_num, mask, x_ctx, state_idx=None):
        """
        Parameters
        ----------
        x_cat : (B, L, 5) long      범주형 인덱스
        x_num : (B, L, 8) float     투구 물리량
        mask  : (B, L) bool         True = 유효한 투구
        x_ctx : (B, 15) float       현재 상황 (표준화됨)
        state_idx : (B, 2) long     [카운트 상태, 베이스-아웃 상태]
        """
        embs = [self.embeddings[name](x_cat[:, :, i])
                for i, name in enumerate(CAT_ORDER)]
        x = torch.cat(embs + [x_num], dim=2)
        x = self.input_proj(x) + self.pos_embed[:, : x.size(1)]

        # 패딩 자리에는 어텐션이 걸리지 않게 한다.
        x = self.transformer(x, src_key_padding_mask=~mask)

        if self.cfg.pooling == "mean":
            m = mask.unsqueeze(-1).float()
            seq_vec = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        else:
            seq_vec = x[:, -1, :]

        if self.cfg.use_context_skip:
            ctx = x_ctx
            if self.cfg.use_state_embed:
                if state_idx is None:
                    raise ValueError("use_state_embed=True 이면 state_idx 가 필요하다")
                ctx = torch.cat([
                    ctx,
                    self.count_embed(state_idx[:, 0]),
                    self.baseout_embed(state_idx[:, 1]),
                ], dim=1)
            vec = torch.cat([seq_vec, self.context_proj(ctx)], dim=1)
        else:
            vec = seq_vec

        h = self.head(vec)
        logits = self.classifier(h)
        rv = self.run_value(h).squeeze(-1) if self.run_value is not None else None
        return logits, rv


class FocalLoss(nn.Module):
    """희귀 클래스에 가중을 주는 손실 (Lin et al., 2017).

    주의: 이전 학습은 Focal Loss 와 ``WeightedRandomSampler`` 를 **동시에** 썼다.
    불균형을 두 번 보정한 셈이라 출력 확률이 실제 분포가 아니라 균등화된 분포를
    가리키게 되고, 그래서 앱에서 확률이 이상하게 나왔다. 둘 중 하나만 쓴다.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def count_state(balls: torch.Tensor, strikes: torch.Tensor) -> torch.Tensor:
    """(볼, 스트라이크) → 0..11 상태 인덱스."""
    return (balls.clamp(0, 3) * 3 + strikes.clamp(0, 2)).long()


def baseout_state(on1, on2, on3, outs) -> torch.Tensor:
    """(주자, 아웃) → 0..23 상태 인덱스. 세이버메트릭스 표준 상태공간."""
    bases = (on1.long() + on2.long() * 2 + on3.long() * 4).clamp(0, 7)
    return (bases * 3 + outs.clamp(0, 2).long()).long()
