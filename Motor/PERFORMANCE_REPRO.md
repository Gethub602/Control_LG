# 추론 가속과 tinygrad 직접 재현 가이드

이 문서는 결과 숫자만 믿는 대신, 같은 장비에서 다음 사실을 직접 확인하기 위한 절차다.

tinygrad 공식 문서도 `TinyJit`이 Python 실행을 건너뛰고 캡처한 커널을 재생한다고
설명하며, 환경 변수 문서에서 `DEBUG=1..7`과 `VIZ=1`을 제공한다.

- https://docs.tinygrad.org/
- https://docs.tinygrad.org/env_vars/
- https://docs.tinygrad.org/mnist/

1. TF DDIM20의 약 30배 가속이 모델 축소가 아니라 실행 방식 변경에서 나오는가?
2. 동일 입력과 동일 초기 노이즈에서 eager, graph, XLA의 출력이 같은가?
3. 작은 순차 연산에서 실제 GPU 계산보다 스케줄·디스패치 비용이 큰가?
4. TinyJit이 무엇을 제거하는지 tinygrad의 텍스트 출력과 웹 UI에서 볼 수 있는가?

## 1. 환경

저장소 루트에서 시작한다.

```bash
cd /home/dw/LGControl/Motor
source /home/dw/miniforge3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg
```

TensorFlow 원본 모델은 `lgcontrol`, tinygrad는 `lgcontrol-torch` 환경을 쓴다.

## 2. 현재 데이터 수집 확인

```bash
conda activate lgcontrol-torch
python src/summarize_gain_collection.py --target-runs 950
pgrep -af collect_diffusion_gain_chunk_db
tail -40 /home/dw/kafka/collect950.log
```

`progress`는 메타데이터에 저장된 완료 궤적 수다. `aborted`가 모두 `False`인지,
각 scenario와 gain profile 수가 균형적인지, 모든 궤적이 120행인지 확인한다.
`max dt`와 `gaps > 0.2 sec`는 10 Hz 수집 루프가 간헐적으로 밀린 정도를 보여준다.

2026-07-31의 최초 수집에서는 WSL realtime clock이 약 33초마다 1초 앞으로
보정되는 현상이 확인됐다. 실제 sleep/serial 지연은 아니었지만 당시 수집기가
`time.time()`으로 목표 전환 시각을 정했으므로 해당 궤적은 제외해야 한다.
수집기는 이후 elapsed time을 `time.monotonic()`으로 기록하도록 수정됐다.
구형 데이터와 신규 데이터를 합칠 때는 다음 필터를 사용한다.

```bash
python src/merge_chunk_datasets.py <old_chunks.csv> <new_chunks.csv> \
  --max-time-gap 0.2 --label real1000_clean
```

## 3. 실제 TF DDIM20 가속 재현

```bash
conda activate lgcontrol
PYTHONNOUSERSITE=1 python src/benchmark_ddim_speedups.py \
  --model-path artifacts/final_ddim20/models/diffusion_gain_chunk_unet_balanced1000_global_topk_full_20260508_193250_linux.joblib \
  --ddim-steps 20 --repeats 15
```

표의 의미:

- `eager (current)`: 기존 저장소 경로. Python이 모델을 20번 순차 호출한다.
- `eager, fixed noise`: 동일 노이즈 수치 검증용 eager 구현이다.
- `tf.function`: 20스텝 전체를 TensorFlow graph 한 번으로 실행한다.
- `tf.function + XLA`: 같은 graph를 XLA가 컴파일·융합한다.
- `cold ms`: 최초 trace/compile이 포함된 한 번의 비용이다.
- `p90 ms`: 워밍업 후 정상상태 지연이며 실제 30배 주장에 쓰는 값이다.

마지막 `numerical parity`에서 `allclose=True`를 확인한다. 이 검사는 세 경로에
관측값, 정적 특성, 초기 노이즈를 완전히 동일하게 넣는다. XLA는 부동소수점 연산
재배치 때문에 마지막 몇 비트가 다를 수 있으므로 `max_abs`도 함께 본다.

핵심 해석은 다음과 같다. 모델 파라미터와 DDIM 스텝 수는 세 경로 모두 같다.
따라서 정상상태 지연 차이는 정확도를 희생한 경량화가 아니라 Python 호출,
그래프 재구축, 개별 연산 디스패치가 graph/XLA 안으로 이동한 결과다.

## 4. tinygrad 텍스트 실험

```bash
conda activate lgcontrol-torch

# 스텝 수에 따른 eager와 JIT 지연
for n in 1 5 10 20 40; do
  python src/tinygrad_overhead_probe.py --steps "$n" --repeats 30
done

# 초기화를 제외한 3개 순차 스텝의 커널을 직접 출력
DEBUG=2 python src/tinygrad_overhead_probe.py --steps 3 --skip-timing
```

`=== STEADY SAMPLER PASS ===` 뒤만 센다. 이 프로브는 한 스텝에 작은 matmul 두
개가 있고 tinygrad가 활성화 연산을 융합하므로, 3스텝에서 6개 GPU 커널이 나온다.
각 커널 줄의 `tm`은 GPU 실행 시간이다. 바로 위 `scheduled ... kernels in ... ms`는
Python 그래프 구성과 스케줄링 시간을 포함한다. 작은 batch-1 연산에서는 수십
마이크로초의 계산보다 이 호스트 측 비용이 훨씬 큰 것을 직접 비교할 수 있다.

`DEBUG=3`은 IR/연산을, `DEBUG=4`는 생성된 커널 소스까지 보여준다. 출력이 매우
길기 때문에 먼저 `--steps 1 --skip-timing`으로 보는 것이 좋다.

## 5. tinygrad 내장 VIZ 웹 UI

재현 결과를 저장소 아래에 보존하려면 임시 디렉터리를 명시한다.

```bash
conda activate lgcontrol-torch
mkdir -p artifacts/tinygrad_viz
TMPDIR="$PWD/artifacts/tinygrad_viz" VIZ=1 \
  python src/tinygrad_overhead_probe.py --steps 3 --skip-timing
```

이 실행은 보통 다음 두 파일을 만든다. 사용자명이 접미사로 붙을 수 있다.

```text
artifacts/tinygrad_viz/profile.pkl.<user>
artifacts/tinygrad_viz/rewrites.pkl.<user>
```

먼저 터미널에서 내용을 확인한다.

```bash
python -m tinygrad.viz.cli \
  --profile-path artifacts/tinygrad_viz/profile.pkl.$USER \
  --rewrites-path artifacts/tinygrad_viz/rewrites.pkl.$USER --list

python -m tinygrad.viz.cli \
  --profile-path artifacts/tinygrad_viz/profile.pkl.$USER \
  --rewrites-path artifacts/tinygrad_viz/rewrites.pkl.$USER -t 20
```

웹 UI를 실행한다.

```bash
PORT=8000 python -m tinygrad.viz.serve \
  --profile-path artifacts/tinygrad_viz/profile.pkl.$USER \
  --rewrites-path artifacts/tinygrad_viz/rewrites.pkl.$USER
```

Windows 브라우저에서 `http://localhost:8000`을 연다. WSL2의 localhost 전달이
꺼져 있다면 WSL의 IP를 확인해 `http://<WSL_IP>:8000`으로 접속한다.

UI에서는 다음 순서로 본다.

1. 프로파일 타임라인에서 `USER`, `TINY`, `CUDA`를 비교한다.
2. `Schedule ... Kernels`를 열어 스케줄 생성 시간을 확인한다.
3. CUDA 커널 하나를 선택해 `View Kernel Graph`에서 융합된 연산을 본다.
4. `View Source`에서 실제 생성된 CUDA 커널 소스를 본다.
5. JIT capture 관련 항목에서 여러 커널이 재사용 가능한 실행으로 묶이는지 본다.

종료는 서버 터미널에서 `Ctrl-C`다. VIZ는 컴파일/프로파일 자체의 오버헤드를
추가하므로 VIZ가 켜진 실행 시간을 성능 숫자로 사용하지 않는다. 성능은 4절의
VIZ 없는 실행으로 재고, VIZ는 구조와 원인을 보는 용도로 사용한다.

## 6. 이 실험이 증명하는 범위

tinygrad 프로브는 TF U-Net 자체를 포팅한 것이 아니라, batch 1의 작은 모델을
여러 번 순차 호출할 때 생기는 메커니즘을 분리한 최소 실험이다. 따라서
tinygrad의 200배 같은 배율을 TF의 28~30배와 같은 숫자로 해석하면 안 된다.
두 실험이 공통으로 보여주는 것은 스텝마다 작은 계산을 개별 dispatch하는 구조가
느리고, 전체 루프를 capture/compile하면 계산량을 바꾸지 않고도 큰 폭으로
빨라진다는 점이다.

그리고 flow matching은 이 최적화와 별개의 축이다. graph/XLA/TinyJit은 같은
20스텝을 효율적으로 실행하고, flow matching은 필요한 함수 평가 횟수 자체를
20에서 2 수준으로 줄인다. 둘은 경쟁 관계가 아니라 함께 적용할 수 있다.
