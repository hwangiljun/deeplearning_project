# NextPitch — 상황 인식형 트랜스포머 기반 투구 전략 분석

MLB Statcast 데이터로 투구 결과를 예측하고, 주어진 상황에서 기대 실점이 가장 낮은
구종·코스를 찾아 주는 시스템. Streamlit 대시보드로 제공한다.

> **이 저장소는 기존 프로젝트의 전면 재작업본이다.** 초기 버전에서 학습 데이터
> 파이프라인에 심각한 결함이 발견되어, 데이터 구성부터 평가 방법까지 다시 만들었다.
> 무엇이 잘못됐고 어떻게 고쳤는지는 아래 [무엇을 고쳤나](#무엇을-고쳤나)에 정리했다.

---

## 대시보드

```bash
pip install -r requirements.txt
streamlit run app.py
```

| 페이지 | 내용 |
|---|---|
| **투구 추천** | 매치업·상황·타석 이력을 입력하면 (구종 × 코스) 후보 전체를 평가해 기대 실점이 가장 낮은 Top-3 를 제시. 구종별 코스 지도를 함께 표시 |
| **타자 분석** | 코스별·구종별 약점, 헛스윙률·체이스율 등 지표 (리그 대비 증감 포함) |
| **투수 분석** | 구종 레퍼토리와 무브먼트, 카운트별 구종 선택 성향, 코스 성향 |
| **팀 분석** | 30개 팀 투수진 지표 순위와 소속 선수 |
| **모델 성능** | 테스트 시즌 지표, 학습 곡선, 캘리브레이션, 혼동 행렬, 어블레이션 |

학습 산출물이 없어도 **타자·투수·팀 분석 페이지는 데이터 집계만으로 동작한다.**

---

## 무엇을 고쳤나

초기 버전은 검증 정확도 67.52% 를 보고했다. 재현 과정에서 **그 수치가 미래 정보
누수로 만들어진 값**임이 확인됐다.

### 1. 시퀀스가 시간 역순이었다 (가장 치명적)

pybaseball 이 돌려주는 Statcast 원본은 **역시간순**이다. 한 타석 안에서
`pitch_number` 가 7, 6, 5 … 1 로 내려간다. 초기 코드에는 정렬이 없었다.

```python
grouped = df.groupby('batter')        # 역순 그대로 유지됨
sequences.append(data[i : i+8])
targets.append(target_vals[i+7])      # 윈도우의 '마지막' = 시간상 가장 과거
```

결과적으로 **나중에 던져진 7개의 공으로 그보다 먼저 던져진 1개 공의 결과를
맞히는 문제**를 풀고 있었다. `balls`/`strikes` 가 모든 타임스텝에 들어 있었으므로,
타깃 다음 투구의 카운트만 비교하면 볼/스트라이크를 그대로 읽을 수 있었다
(볼이 1 늘었으면 볼, 스트라이크가 1 늘었으면 스트라이크).

**수정**: `game_date → game_pk → at_bat_number → pitch_number` 로 정렬하고,
시퀀스를 **타석 안으로 제한**했다. 카운트는 시퀀스가 아니라 맥락 경로에만 넣어
누수 경로 자체를 없앴다.

### 2. 무작위 분할 + 겹치는 윈도우

보고서에는 "시간 순서에 따라 8:2 분할" 이라고 적혀 있으나 실제 코드는
`train_test_split(..., random_state=42, stratify=y)` — 셔플 분할이었다.
게다가 stride 2 슬라이딩이라 `[i..i+7]` 과 `[i+2..i+9]` 가 8개 중 6개를 공유했다.

**수정**: **시즌 단위 분할** (2024 학습 / 2025 테스트). 검증셋도 학습 구간의
뒷부분 날짜에서 뗀다. 인코더·스케일러는 학습 구간에만 적합한다.

### 3. 불균형 이중 보정으로 확률이 왜곡됨

`WeightedRandomSampler`(역빈도 오버샘플링)와 Focal Loss 를 **동시에** 썼다.
출력 확률이 실제 분포가 아니라 균등화된 분포를 가리키게 되어, 앱에서
`temperature = 5.0` 을 손으로 박아 넣어 가리고 있었다.

**수정**: 샘플러 제거, 기본 CrossEntropy. 예측 주변 확률이 실제 빈도와 일치한다.
보정이 필요하면 검증셋에서 temperature 를 적합한다(Guo et al., 2017).

### 4. 그 밖의 데이터 문제

- 시범경기 35,482구가 학습에 섞여 있었다 → `game_type == 'R'` 필터
- `triple`, `sac_fly`, GIDP, `force_out` 등 인플레이 결과 10,224구가 삭제되고 있었다
  → 결과 매핑 재정의 (버려지는 투구 10,224 → 6)
- 2024년 3~8월만 수집되어 있었다 (보고서는 "정규 시즌 전체") → 2024·2025 전체

### 5. 추천 점수가 수학적으로 무의미했다

```python
score = p_success - (p_fail * 1.5) - penalty
```

good/bad 두 집합이 11개 클래스 전체를 정확히 이분하므로 `p_fail = 1 - p_success`,
따라서 `score = 2.5 * p_success - 1.5`. **가중치 1.5 는 순위에 아무 영향이 없었다.**
또한 2스트라이크에서의 파울(사실상 무가치)과 범타를 똑같이 "성공" 으로 묶었다.

**수정**: `Σ P(결과) × 기대 실점(카운트, 결과)`. Statcast `delta_run_exp` 에서
카운트별 결과 가치를 직접 추정한다.

| 결과 | 0-0 | 0-2 | 3-2 |
|---|---|---|---|
| ball | +0.035 | +0.021 | +0.287 |
| called_strike | −0.040 | −0.168 | −0.323 |
| foul | −0.040 | −0.010 | −0.018 |
| field_out | −0.253 | −0.166 | −0.319 |

### 6. 앱의 나머지 결함

- 모델 파일 경로가 어긋나 **앱이 실행조차 되지 않았다**
- 후보 한 구를 8번 복사해 시퀀스를 만들었다 (학습은 실제 시퀀스) → 실제 타석 이력 사용
- 구종별 물리량이 리그 평균 상수였다 (투수를 골라도 슬라이더는 항상 84.5mph)
  → 투수×구종 실측 평균
- 존 안 9칸만 후보라 **유인구를 추천할 수 없었다** → 존 밖 포함 81칸 격자
- 미등록 선수를 조용히 0번 선수로 대체했다 → UNK 인덱스 + 학습 중 무작위 마스킹
- 무브먼트를 타자 좌우로 미러링한 값의 평균을 표시해 부호가 상쇄되고 있었다
  → 표시·반사실 후보에는 원본 `pfx_x`, 미러링은 모델 입력 단계에서만

---

## 모델

이원 경로 구조. 시퀀스는 Transformer 로, 현재 상황은 별도 경로로 처리해 출력
직전에 결합한다. 이 설계는 Takamido & Nakamoto (2026) 가 쓴 구조와 일치한다.

```
투구 시퀀스 (타석 내 최근 6구, 물리량 8개)
  → 임베딩 + 선형 사영 → Transformer Encoder (2층, 4헤드) → 마스크드 평균 풀링
                                                                    ↓
맥락 15개 (카운트·주자·이닝·점수차·타순·타자 직전시즌 성적)              결합 → MLP → 10 클래스
  → Dense + ReLU ──────────────────────────────────────────────────↑
```

초기 구현에서 함께 고친 점:

- **최종 LayerNorm 추가** — `norm_first=True`(Pre-LN)인데 마지막 정규화가 없었다
- **패딩 마스크 추가** — 타석 앞부분 패딩에도 어텐션이 걸리고 있었다
- **마스크드 평균 풀링** — 마지막 토큰만 쓰던 것을 교체
- **맥락 경로 표현력** — 원시 정수 6개를 `Linear(6,32)+ReLU` 에 통과시켜서,
  0-0 카운트·무사·주자없음이면 입력이 전부 0 이 되어 `ReLU(bias)` 라는 상수만
  남았다. 가장 흔한 상황에서 스킵 연결의 정보량이 0 이었다

**출력 클래스 재정의**: `strikeout` / `walk` 는 투구 결과가 아니라 카운트에서
파생되는 타석 결과이므로 뺐다(삼진 = 2스트라이크에서의 스트라이크). 초기 정의는
`strike` 와 `strikeout` 을 동시에 클래스로 둬 서로 중복이었고, 카운트가 입력에
있으니 모델이 공짜로 맞히는 라벨이었다.

```
ball · called_strike · swinging_strike · foul · hit_by_pitch
field_out · single · double · triple · home_run
```

---

## 결과

2024 시즌 학습 / **2025 시즌 테스트** (학습·조기종료·모델 선택에 한 번도 쓰지 않음).

| 구성 | log-loss | macro AUC | 정확도 | macro-F1 |
|---|---|---|---|---|
| **full** | **1.1259** | **0.8410** | 57.48% | 0.2633 |
| no_context_skip | 1.1423 | 0.8364 | 56.91% | 0.2571 |
| no_state_embed | 1.1244 | 0.8412 | 57.53% | 0.2573 |
| last_token | 1.1269 | 0.8400 | 57.42% | 0.2683 |
| seq_len_1 | 1.1302 | 0.8393 | 57.36% | 0.2627 |

한 번에 한 요소만 바꿔 기여를 분리했다. 나머지 조건(폭·깊이·손실함수·에폭·학습률)은
모두 동일하다.

**① 맥락 스킵 연결이 실제로 기여한다.** 제거 시 log-loss +0.0164, AUC −0.0046.
표에서 가장 큰 효과이며, 유일하게 노이즈 범위 밖으로 보이는 차이다.

**② 시퀀스의 기여는 작다.** `seq_len_1`(직전 투구 정보 완전 제거) 대비
log-loss 차이가 +0.0043 에 그친다. 예측력의 대부분은 그 공 자체의 물리량과
현재 상황·타자 특성에서 나오며, **"투구 배합의 시계열 맥락이 결정적" 이라는
초기 전제는 이 데이터에서 지지되지 않는다.** 셋업 피치 최적화 효과가 최종구
최적화보다 작았던 Takamido & Nakamoto (2026) 의 보고와도 방향이 일치한다.

**③ 카운트/베이스-아웃 상태 임베딩은 효과가 없다.** 제거해도 동등하거나 미세하게
낫다. 맥락 경로의 이득은 연속 맥락 피처(특히 타자 직전시즌 성적)에서 나온다.

### 정확도를 주 지표로 쓰지 않는 이유

초기 67.52% 와 여기의 57.48% 는 **비교 대상이 아니다.** 전자는 미래 누수가 있고,
카운트에서 공짜로 파생되는 `strikeout`/`walk` 를 클래스로 포함하며, 시범경기가
섞여 있다. 그리고 이 시스템은 확률로 후보 순위를 매기므로 확률의 질이 본질이다.
주 지표는 **log-loss 와 캘리브레이션**, 보조로 macro AUC · macro-F1 을 본다.

> **한계**: 위 표는 시드 1개의 단일 실행이다. ②③ 의 차이(0.001~0.004)는 노이즈일
> 수 있다. 핵심 3개 구성은 시드를 늘려 평균±표준편차로 보고할 필요가 있다.

---

## 구조

```
app.py                    Streamlit 진입점
src/
├── data.py               원본 → 시퀀스 (정렬·타석경계·미러링·존 정규화)
├── prepare.py            인코더/스케일러 (학습 구간에만 적합)
├── model.py              이원 경로 Transformer
├── train.py              학습 루프 + 어블레이션 격자
├── evaluate.py           클래스별 지표·혼동행렬·캘리브레이션·temperature
├── features.py           투수 레퍼토리·카운트별 결과 가치·위치별 가치
├── profiles.py           선수/팀 집계 (경험적 베이즈 축소 적용)
├── recommend.py          후보 생성 · 기대 실점 점수 · command window
└── dashboard/            테마·차트·페이지
notebooks/
├── 01_download_statcast.ipynb   데이터 수집 (Colab)
├── 02_train.ipynb               학습 + 어블레이션 (Colab, GPU)
└── 03_build_artifacts.ipynb     테이블·프로파일·평가 (Colab, CPU)
scripts/                  로컬 실행용 CLI
models/                   학습 산출물 (아래 참조)
docs/DATA_SCHEMA.md       Statcast 컬럼 명세와 누수 주의 컬럼
```

### 산출물 배치

```
models/
├── main/          02_train.ipynb → model.pth, encoders.pkl, model_config.json,
│                                   metadata.json, evaluation.pkl
├── tables.pkl     03_build_artifacts.ipynb
├── profiles.pkl   03_build_artifacts.ipynb
├── batter_stats.pkl
└── legacy/        초기 버전 산출물 (현재 코드와 호환되지 않음, 기록용 보존)
```

데이터(`data/`)와 아카이브는 용량 때문에 저장소에 포함하지 않는다.
`notebooks/01_download_statcast.ipynb` 로 재생성할 수 있다.

---

## 재현

```bash
# 로컬 (샘플 데이터로 파이프라인 점검)
python scripts/validate_pipeline.py     # 누수 차단 검증
python scripts/build_tables.py   --raw data/raw --out models/tables.pkl
python scripts/build_profiles.py --raw data/raw --out models/profiles.pkl
python scripts/test_app.py              # 전 페이지 렌더 검증

# Colab
notebooks/01 → 02 → 03 순서로 실행
python scripts/make_colab_bundle.py     # src/ 를 zip 으로 묶어 Colab 에 업로드
```

---

## 참고문헌

1. R. Takamido and H. Nakamoto, "Counterfactual Optimization of Baseball Pitch
   Sequences and Estimation of Its Impact on Season-Level Statistics,"
   arXiv:2606.17345, 2026. — 타석 단위 시퀀스 구성, 좌표 미러링과 타자별 존
   정규화, 투수×구종 실측 평균을 이용한 반사실 대체, command window
2. T. Lin, P. Goyal, R. Girshick, K. He, P. Dollár, "Focal Loss for Dense Object
   Detection," ICCV, 2017.
3. C. Guo and F. Berkhahn, "Entity Embeddings of Categorical Variables,"
   arXiv:1604.06737, 2016.
4. C. Guo, G. Pleiss, Y. Sun, K. Weinberger, "On Calibration of Modern Neural
   Networks," ICML, 2017. — temperature scaling
5. MLB Advanced Media, "Statcast Data," Baseball Savant.
   https://baseballsavant.mlb.com
