# 작업 기록 — 테스트베드 인수 및 Flow Matching 확장

원저자(Jaewook)의 ESP32 모터 제어 테스트베드를 인수하여 재구축하고,
기존 확산(diffusion) 게인 청크 방식에 **flow matching**을 추가한 작업 기록.

작성일: 2026-07-31

---

## 0. 한눈에 보기

| 영역 | 상태 |
|---|---|
| 하드웨어 배선·펌웨어 | ✅ 완료 (재배선 + 신규 펌웨어) |
| RPM 스케일 검증 | ✅ 명판과 97% 일치 |
| Kafka 파이프라인 | ✅ 동작 (브로커 자체 구축) |
| DDIM 추론 30배 가속 | ✅ 적용 |
| PyTorch 전면 포팅 | ✅ 10개 모델 |
| 실물 데이터 수집 | 🔄 진행 중 (1000궤적) |
| 게인 스윕 | ⏸ 대기 |

---

## 1. 하드웨어 인수

### 1.1 모터 사양 (확정)

| 항목 | 값 | 출처 |
|---|---|---|
| 모델 | JGB37-520 | — |
| 정격 | 12V / 178 RPM | 제품 사양 |
| 감속비 | **1:56** | 제조사 + 실측 검증 |
| 엔코더 PPR | 11 (모터축 기준) | 제조사 |
| 출력축 1회전 카운트 | **2,464** (11 × 4체배 × 56) | 계산 |

### 1.2 배선 (데이터시트 확정)

원래 인수 상태에서 **6선 중 5선이 잘못 연결**돼 있었다.

| 색 | 실제 역할 | 연결 위치 |
|---|---|---|
| 🔴 빨강 | 모터 + | TB6612 **AO1** |
| ⚪ 흰색 | 모터 − | TB6612 **AO2** |
| 🔵 파랑 | 엔코더 VCC | ESP32 **3V3** |
| ⬛ 검정 | 엔코더 GND | ESP32 **GND** |
| 🟡 노랑 | 신호 A | ESP32 **D18** |
| 🟢 초록 | 신호 B | ESP32 **D19** |

> ⚠️ 검정=엔코더 GND, 흰색=모터 −. 일반적인 직관(검정=모터−)과 반대다.
> 잘못 연결하면 모터 권선이 GPIO에 물려 ESP32가 파손된다.

**드라이버 (TB6612FNG)**

```
PWMA → D25    AIN1 → D26    AIN2 → D27
VCC  → 3V3    STBY → 3V3    VM → 12V 외부
GND  → 공통 (ESP32 GND + PSU− + 엔코더 검정)
```

### 1.3 펌웨어 (신규 작성)

기존 펌웨어는 **소스가 없고 핀 맵을 알 수 없는 블랙박스**였다. 배선을 고쳐도
엔코더가 0을 반환했는데, 원인은 기존 펌웨어가 D18/D19가 아닌 다른 핀을
읽고 있었기 때문. 원본은 `firmware/backup/`에 4MB 전체 백업.

`firmware/motor_controller/motor_controller.ino`

기존 프로토콜을 그대로 구현하여 `motor_interface.py` 수정 불필요:

```
PING        → PONG
SET_PWM <v> → OK PWM <v>
GET_STATE   → STATE rpm=<r> pwm=<p> encoder=<c>
STOP        → OK STOP
```

추가한 진단·안전 기능:

| 명령 | 용도 |
|---|---|
| `VERSION` / `INFO` | **핀 맵·PWM 상한·기어비·엔코더 부호 출력** |
| `RAW` | A/B 핀 레벨 (배선 진단) |
| `RESET_ENCODER` | 카운트 초기화 |
| `SET_GEAR <r>` | 기어비 런타임 변경 |
| `SET_LIMIT <v>` | PWM 상한 런타임 변경 |
| `SET_ENCODER_SIGN ±1` | 엔코더 방향 반전 |
| `SET_DRIVE_MODE BRAKE\|COAST` | 감쇠 모드 |

안전장치 3겹:
1. 디바이스 측 PWM 상한 (호스트 버그 무력화)
2. 커맨드 워치독 — 1초 무통신 시 자동 정지
3. 정지 시 코스트 (급제동 아님)

**핵심 교훈**: `VERSION`으로 핀 맵을 노출시킨 것이 가장 큰 개선. 블랙박스
상태에서는 배선 문제를 원리적으로 진단할 수 없었다.

### 1.4 검증 결과

```
pwm    rpm    rpm/count
 40  22.07      0.552
 60  37.37      0.623
 80  52.46      0.656
100  67.78      0.678     ← 명판 상한 0.698의 97%
-80 -48.46      역방향 정상
```

- 부호 규약 정상 (양수 PWM → 양수 RPM)
- 정상상태 변동 1.5%
- ESP32 리셋 없음 (약 1W 소비, 발열 무시 가능)

---

## 2. PWM 상한 — 가장 큰 성능 요인

### 2.1 문제

`config.py`의 `REAL_PWM_MAX = 140`은 원저자가 *"Conservative real motor
limits for initial test"* 주석과 함께 임의로 잡은 값이지 하드웨어 한계가
아니다 (100% = 255).

우리 모터는 같은 RPM에 **원저자보다 1.4배 많은 PWM**이 필요하다.
그래서 140에서는 95 RPM 목표가 정확히 포화 지점에 걸렸다.

| | 원저자 | 우리 (140) | 우리 (200) |
|---|---:|---:|---:|
| 실사용 최대 PWM | 104~109 (78%) | **140 (100%)** | 154 (77%) |
| 포화 비율 | 0% | 9% | **0%** |

### 2.2 조치

`REAL_PWM_MAX: 140 → 200`, `SOFT_LIMIT: 120 → 170`
(펌웨어 `pwmLimit`도 200으로 동기화)

전기적 안전: 모터 정격 0.45~1A, TB6612 연속 1.2A. PWM 200 = 78% 듀티.

### 2.3 효과

| 지표 | PWM 140 | PWM 200 |
|---|---:|---:|
| 게인 DB IAE | 95.89 ± 1.96 | **92.98 ± 0.54** |
| 확산 DDIM20 IAE | 114.33 | **82.10** |
| 재현성 (std) | 1.96 | **0.54** (3.6배 개선) |

포화 구간에서는 제어기 명령이 잘려서 실행마다 편차가 컸다.

---

## 3. 원저자 결과와의 관계

### 3.1 절대 수치는 비교 불가

원저자 DDIM20: IAE 53.73, mean_pwm 85.2 → **0.963 rpm/count**

이 값은 12V·178RPM·1:56 모터의 **물리적 한계(0.698)를 38% 초과**한다.
우리 실측 0.679는 명판의 97%로 정상. 즉 원저자 테스트베드는
전압이 달랐거나(≈16.6V), 펌웨어가 RPM을 부풀려 보고했거나
(유효 기어비 ≈43.6), 다른 모터였다.

**따라서 53.73을 재현 대상으로 삼지 않는다.** 대신 우리 테스트베드 내부
비교로 전환했고, flow matching 평가에는 이쪽이 오히려 타당하다.

### 3.2 원저자와 맞춘 항목

```
sampling-mode   balanced       ✅
num-runs        1000           🔄 진행 중
run-time        12 s           ✅
obs/horizon     10 / 20        ✅
target range    65~105 RPM     ✅
label profile   tracking_first ✅
label quality   top_k          ✅
diffusion_steps 100            ✅  ← 기본값 1000이었음, 발견하여 수정
ddim_steps      20             ✅
sample_count    4              ✅
```

> `diffusion_steps=100`을 놓쳤다면 코사인 노이즈 스케줄이 완전히 달라
> 비교 자체가 무의미했을 것이다. 물려받은 payload에서 발견.

---

## 4. 소프트웨어 환경

### 4.1 conda 환경

| 환경 | 내용 | 용도 |
|---|---|---|
| `lgcontrol` | TF 2.15 / Keras 2.15 (CPU) | 물려받은 확산 모델 실행 |
| `lgcontrol-torch` | PyTorch 2.11+cu128, tinygrad 0.13 | **주력** |
| `lgcontrol-gpu` | TF 2.20 GPU 시도 | ❌ 사용 불가, 삭제 가능 |

### 4.2 GPU 문제 (RTX 5070 Ti, sm_120)

| 스택 | Blackwell | Keras 2 | 결과 |
|---|---|---|---|
| TF 2.15 | ✗ CUDA 12.2 | ✓ | GPU 불가 |
| TF 2.20 + tf-keras | ✗ `CUDA_ERROR_INVALID_PTX` | ✓ | GPU 불가 |
| **PyTorch 2.11+cu128** | **✓ 네이티브** | — | **동작** |
| **tinygrad 0.13** | **✓ 런타임 컴파일** | — | **동작** |

TF는 sm_120 큐빈이 없고 번들 PTX도 오래돼 JIT조차 실패. PyTorch로 전면 전환한 이유.

### 4.3 WSL2 시리얼 접근

```powershell
# Windows 관리자 PowerShell (USB 재연결마다 attach 필요)
usbipd list
usbipd bind   --busid 3-3      # 최초 1회
usbipd attach --wsl --busid 3-3
```
```bash
sudo usermod -aG dialout $USER   # 최초 1회, wsl --shutdown 후 반영
```

### 4.4 Kafka (자체 구축)

Docker/Java 모두 없어 conda로 직접 구축:

```bash
conda install -c conda-forge openjdk=17
# kafka_2.13-3.9.0, KRaft 모드 (ZooKeeper 불필요)
bin/kafka-storage.sh format -t <uuid> -c config/kraft/server.properties
bin/kafka-server-start.sh config/kraft/server.properties
```

토픽: `motor_state`, `motor_gain_command`, `motor_schedule_chunk`, `motor_event_log`

---

## 5. 발견하고 수정한 버그

### 5.1 컨슈머 그룹 조인 실패 (치명적)

kafka-python 3.0.9 + Kafka 3.9 조합에서 **컨슈머 그룹 조인이 완료되지 않는다.**
40초간 400회 폴을 해도 파티션 할당이 비어 있었고, 컨트롤러가 스케줄 청크를
**0개** 수신했다.

`local_kafka_controller.py`를 수동 파티션 할당으로 전환:

```python
consumer.assign([TopicPartition(topic, p) for p in sorted(partitions)])
consumer.seek_to_end()
consumer.poll(timeout_ms=1)   # 프라이밍 폴 — 없으면 이후 1ms 폴이 영원히 빈 값
```

실시간 루프에는 오히려 이쪽이 낫다 (리밸런스 정지 없음).

### 5.2 모델 가중치 로드 실패

- payload의 `weights_path`가 원저자 Windows 절대경로
- 파일명이 `.weights.h5`인데 내용은 **Keras 2.10 레거시 형식**.
  Keras 2.15는 그 확장자를 신형 로더(`vars` 그룹 기대)로 라우팅해 실패.
  → `.legacy.h5`로 복사하여 해결

### 5.3 데이터셋 선택 버그

저장소의 `latest_label_path()`가 **이름순** 정렬:

```
chunk_labels_real_pilot50_...   'r'
chunk_labels_sim400_...         's' ← 사전순 뒤라 선택됨
```

타임스탬프가 라벨 뒤에 있어 최신 파일이 안 잡힌다. mtime 기준으로 변경.

### 5.4 `plt.show()` 블로킹

저장소 스크립트들이 `plt.show()`를 호출하는데 WSL에 디스플레이가 없어
무한 대기. **`MPLBACKEND=Agg` 필수.**

---

## 6. DDIM 추론 30배 가속

### 6.1 발견

병목은 연산이 아니라 **eager 디스패치 오버헤드**였다. 1.5M 파라미터
모델을 batch=1로 **20번 순차 호출**하니 오버헤드가 20배 누적.

| 방식 | p90 | 가속 |
|---|---:|---:|
| eager (기존) | 815 ms | 1.0× |
| `tf.function` | 59 ms | 13.7× |
| **`tf.function` + XLA** | **27 ms** | **30.4×** |

### 6.2 적용

`schedule_generators.py`의 `DiffusionUnetGainChunkGenerator`에
`graph_sampler=True`(기본값) 추가. 트레이싱 실패 시 eager 폴백이라
기존 동작을 깨지 않는다.

실 파이프라인 효과: 생성기 p90 **0.79s → 0.031s** (25배)

### 6.3 함의

- 저장소의 경량화 실험(light32/light16)이 실패한 이유가 설명된다.
  파라미터를 줄여도 **순차 스텝 수가 그대로면 지연이 안 줄어든다.**
- 이 최적화만으로 DDIM30을 쓸 여유가 생긴다 — 원저자가 DDIM20을
  택해야 했던 제약이 사라짐.
- **flow matching이 NFE를 20 → 2로 줄이는 접근이 구조적으로 옳다.**

---

## 7. PyTorch 전면 포팅

원저자 Keras 구조를 층 단위로 이식 (causal conv 패딩, residual skip,
MultiHeadAttention 4헤드/key_dim 24 등 동일).

### 7.1 게인 청크 계열 — `chunk_labels` 데이터셋

| 모델 | 파라미터 | 종류 | 파일 |
|---|---:|---|---|
| MLP | 399K | 결정론적 | `torch_gain_chunk_baselines.py` |
| CNN | 464K | 결정론적 | 〃 |
| CNN-residual | 636K | 결정론적 | 〃 |
| CNN-attention | 609K | 결정론적 | 〃 |
| Diffusion (DDIM) | 1.46M | 생성 | `torch_gain_chunk_common.py` |
| Flow matching | 1.46M | 생성 | 〃 |

### 7.2 호라이즌 코스트 계열 — `esp32_horizon_cost_dataset`

| 모델 | 구조 | 출력 |
|---|---|---|
| RandomForest | sklearn 400 trees | horizon_cost |
| MLP-cost | 128-128-64-32 | horizon_cost |
| Multi-task | 160-160-96-48 | 7개 지표 |
| Direct policy | 256-192-128 | 게인 3개 |

설계 판단:
- **Huber 손실 유지** — `horizon_iae`는 오른쪽 꼬리가 길어 제곱오차를 쓰면
  소수의 나쁜 후보가 학습을 지배
- **Direct policy 입력에서 kp/ki/kd 제외** — 예측 대상을 입력으로 주면 안 됨

### 7.3 서버 배선

```bash
--generator torch_diffusion_gain_chunk     --torch-model-path <..> --diffusion-ddim-steps 20
--generator torch_flow_matching_gain_chunk --torch-model-path <..> --flow-steps 2 --flow-solver midpoint
```

`torch_schedule_generators.py`는 지연 임포트라 TF 모드만 쓸 때 torch 불필요.

---

## 8. 실험 결과

### 8.1 실물 베이스라인 (README 시나리오 70→95→90→73)

| 방법 | IAE | 전환 후 IAE | 포화 |
|---|---:|---:|---:|
| 게인 DB (3회) | 92.98 ± 0.54 | 28.42 ± 0.98 | 0% |
| 확산 DDIM20 (Kafka, TF) | 82.10 | 32.76 | 0% |

전체 IAE는 확산이 이기지만 **전환 구간에서는 게인 DB가 우세**.

### 8.2 파일럿 모델 비교 (50궤적, 예비)

| 모델 | NFE | MAE | best-of-4 | 다양성 | p90 |
|---|---:|---:|---:|---:|---:|
| **flow (4스텝)** | 8 | **0.0691** | 0.0599 | 0.032 | 67ms |
| flow (2스텝) | 4 | 0.0699 | 0.0623 | 0.031 | 85ms |
| diffusion DDIM30 | 30 | 0.0738 | **0.0576** | 0.049 | 374ms |
| diffusion DDIM20 | 20 | 0.0744 | 0.0579 | 0.049 | 133ms |
| flow (1스텝) | 2 | 0.0746 | 0.0727 | 0.039 | **41ms** |
| baseline CNN | 1 | 0.0871 | 0.0871 | **0** | **1.0ms** |
| baseline MLP | 1 | 0.0967 | 0.0967 | **0** | 0.9ms |

관찰:
- **생성 모델이 지도학습 베이스라인을 전부 이긴다** (0.069~0.077 vs 0.087~0.098)
- **flow가 확산보다 정확하면서 빠르다** (0.0691 vs 0.0744, 지연 절반)
- **best-of-N에서는 확산이 우세** — 다양성 0.049 vs 0.032.
  응답 대리모델로 후보를 고르는 구조에서는 확산이 유리할 수 있다
- **베이스라인은 다양성 0** — 후보 선택 자체가 불가능

> ⚠️ 파일럿(top-k 2,725행)이라 절대 수치 신뢰도 낮음. 1000궤적 재학습 필요.

### 8.3 시뮬레이션 실험 (참고)

`simulate_generator_closed_loop.py`로 배포 비용 배율을 스윕한 결과,
**flow matching이 배포 비용 증가에 면역**임을 확인:

| 배율 | ddim20 IAE | ddim20 폴백 | flow1 IAE | flow1 폴백 |
|---|---:|---:|---:|---:|
| x1 | 69.92 | 2.5% | 68.83 | 2.5% |
| x15 | 69.18 | 4.2% | 67.79 | 2.5% |
| x30 | **70.36** | **9.3%** | **68.05** | **2.5%** |

x30에서 DDIM10이 DDIM20을 이긴다 — 원저자가 DDIM20으로 DDIM30을 이긴
것과 **정확히 같은 메커니즘**.

### 8.4 오프라인 지표의 함정

시뮬레이션 폐루프에서 **오프라인 MAE 순위와 제어 성능 순위가 반대**로
나왔다 (flow1이 MAE는 최악인데 IAE는 최고). 청크 MAE는 top-k 휴리스틱으로
고른 라벨을 얼마나 잘 모방했는지를 잴 뿐, 라벨 자체가 최적해가 아니다.

**모델 선택을 MAE로 하면 안 된다.** 저장소가 응답 대리모델로 후보를
고르는 설계를 택한 것이 옳았다는 방증.

---

## 9. tinygrad — 오버헤드 해부

`tf.function` 30배의 정체를 커널 단위로 분해.

### 9.1 결과

| 스텝 | 커널 | eager | JIT | 가속 | eager µs/커널 |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 5.39 ms | 0.163 ms | 33× | 2,697 |
| 20 | 40 | 45.33 ms | 0.284 ms | 159× | 1,133 |
| 40 | 80 | 86.64 ms | 0.311 ms | 279× | 1,083 |

### 9.2 확인된 사실

**커널 수 = 13 + 2 × steps** — 스텝당 정확히 2개.
`(x@w1).relu()`와 `(h@w2+b).tanh()`가 각각 **한 커널로 융합**됐다.

**eager는 커널당 ~1,100µs로 일정, JIT는 전체가 거의 상수.**
스텝을 늘리면 eager만 선형으로 늘어난다.

`DEBUG=2`에서 동일 커널의 시간 차이:
```
*** CUDA  5  E_64_32_4   tm   4.03us  (590 GFLOPS)  ← 실제 연산
*** CUDA  8  E_64_32_4   tm  69.38us  ( 34 GFLOPS)  ← 디스패치 포함
```
**17배 차이.** GPU 계산은 4µs, 나머지는 오버헤드.

### 9.3 결론

커널당 오버헤드가 일정하므로 **NFE를 1/10로 줄이면 지연도 1/10**.
파라미터 감소로는 얻을 수 없는 이득이며, flow matching 접근의 근거.

도구:
```bash
python src/tinygrad_overhead_probe.py --steps 20             # 타이밍
DEBUG=2 python src/tinygrad_overhead_probe.py --steps 3 --skip-timing  # 커널
DEBUG=5 ...                                                   # PTX 어셈블리
```

---

## 10. 틀렸던 진단 (기록)

정직한 기록을 위해 남긴다.

| 가설 | 결과 |
|---|---|
| GPU가 CPU보다 빠를 것 | ❌ batch=1 소형 모델은 **CPU가 2~3배 빠름** |
| 코스트 감쇠 모드가 원인 | ❌ **정반대** — 슬로우 디케이(브레이크)가 평균 전류가 높아 더 빠름 |
| PWM 140→200이 스케일 일치 근거 | ❌ 원저자는 PWM 140을 쓴 적 없음 (최대 109). 우연의 일치 |
| 지연이 IAE 병목 | ⚠️ **부분적** — 25배 가속했으나 IAE 변화 없음. 실제 주범은 포화 |

**교훈**: 시간만 재고 원인을 추정하면 틀린다. tinygrad `DEBUG=2`처럼
개수를 셀 수 있는 도구가 있으면 추정이 사실로 바뀐다.

---

## 11. 추가한 파일

### 하드웨어·진단
| 파일 | 용도 |
|---|---|
| `firmware/motor_controller/` | 신규 ESP32 펌웨어 |
| `firmware/backup/` | 원본 펌웨어 4MB 백업 |
| `src/check_encoder.py` | 엔코더 카운트 검사 |
| `src/encoder_live.py` | A/B 핀 실시간 + 실패 모드 판별 |
| `src/calibrate_gear.py` | 기어비 실측 |
| `src/spin_test.py` | 회전·부호·RPM 검증 |
| `src/probe_esp32_firmware.py` | 펌웨어 조사 |

### 실험
| 파일 | 용도 |
|---|---|
| `src/run_esp32_scenario.py` | 다중 목표 시나리오 실행 |
| `src/simulate_generator_closed_loop.py` | 실측 지연 폐루프 (`--latency-scale`) |
| `src/collect_sim_gain_chunk_db.py` | 하드웨어 없이 동일 스키마 수집 |
| `src/merge_chunk_datasets.py` | 여러 수집 병합 |

### 모델
| 파일 | 용도 |
|---|---|
| `src/torch_gain_chunk_common.py` | GainChunkUNet + 데이터 파이프라인 |
| `src/torch_gain_chunk_baselines.py` | MLP/CNN/residual/attention |
| `src/torch_horizon_models.py` | 호라이즌 코스트 계열 |
| `src/train_torch_gain_chunk.py` | 확산/flow 학습 |
| `src/train_torch_gain_chunk_baseline.py` | 베이스라인 학습 |
| `src/train_torch_horizon_models.py` | 호라이즌 학습 |
| `src/train_all_gain_chunk_models.py` | 6개 일괄 학습 + 비교표 |
| `src/torch_schedule_generators.py` | 서버용 PyTorch 생성기 |
| `src/benchmark_ddim_speedups.py` | DDIM 가속 벤치마크 |
| `src/tinygrad_overhead_probe.py` | 커널 오버헤드 해부 |

### 수정한 기존 파일
| 파일 | 변경 |
|---|---|
| `config.py` | 포트, PWM 상한 140→200 |
| `schedule_generators.py` | `_sample_chunks()` 훅, graph sampler, FlowMatching |
| `gain_recommender_server.py` | flow/torch 생성기 배선 |
| `local_kafka_controller.py` | 수동 파티션 할당 |

---

## 12. 실행 방법

### 공통 프리픽스
```bash
source /home/dw/miniforge3/etc/profile.d/conda.sh
conda activate lgcontrol-torch
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg
```

### 하드웨어 확인
```bash
python src/probe_esp32_firmware.py     # 펌웨어·핀 맵
python src/encoder_live.py             # 엔코더 (축을 손으로 돌릴 것)
python src/spin_test.py --reverse      # 회전 시험
```

### 데이터 수집 → 학습
```bash
python src/collect_diffusion_gain_chunk_db.py --num-runs 1000 \
    --run-time 12 --pwm-max 200 --sampling-mode balanced --run-label real1000

python src/merge_chunk_datasets.py --label real1000
python src/label_diffusion_gain_chunks.py --chunk-path <merged.csv> \
    --profile tracking_first --top-k 5 --label real1000

python src/train_all_gain_chunk_models.py --dataset <labels.csv> --run-label real1000
```

### Kafka 파이프라인
```bash
# 브로커
cd /home/dw/kafka/kafka_2.13-3.9.0 && bin/kafka-server-start.sh config/kraft/server.properties

# 서버 (flow matching)
python src/gain_recommender_server.py --generator torch_flow_matching_gain_chunk \
    --torch-model-path <model.joblib> --flow-steps 2 --flow-solver midpoint \
    --inference-delay 0.5 --disable-artificial-inference-sleep

# 컨트롤러 (실물 모터)
python src/local_kafka_controller.py --schedule-apply-mode delay_aware --sim-time 20 \
    --target-sequence "70,95,90,73" --target-change-times "5,10,15"
```

---

## 13. 남은 작업

| 작업 | 상태 |
|---|---|
| 청크 수집 1000궤적 | 🔄 진행 중 |
| 게인 스윕 420 케이스 (~2.6h) | ⏸ |
| 호라이즌 데이터셋 재구축 | ⏸ 스윕 후 |
| 10개 모델 본 학습 | ⏸ |
| 실물 폐루프 최종 비교 | ⏸ |
| tinygrad U-Net 포팅 | ⏸ 선택 |

### 알려진 이슈

- **서버 콜드 스타트 1.8초** — 첫 생성만 느림(모델 로드). 실험 전 워밍업 권장
- **호라이즌 포화 타깃 R²=0** — PWM 200에서 포화가 없어 상수.
  스윕 데이터로 해소 예상, 안 되면 멀티태스크에서 제외 필요
- **`usbipd attach`** — USB 재연결마다 필요
- **`lgcontrol-gpu` 환경** — 사용 불가, 삭제 가능 (~5GB)

---

## 14. 검증된 지연 (참고)

| 경로 | 지연 |
|---|---|
| torch flow(2스텝) 단독, CPU | 13ms |
| **서버 내 정상상태** | **13.7ms (p90 15.1ms)** |
| 서버 콜드 스타트 | 1.8s |
| TF DDIM20 eager | 815ms |
| TF DDIM20 graph+XLA | 27ms |

서버 가정치 `--inference-delay 0.5`는 flow matching 기준 **33배 여유**.
