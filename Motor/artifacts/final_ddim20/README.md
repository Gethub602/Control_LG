# Final DDIM20 Diffusion Gain-Chunk Artifacts

This folder keeps the compact artifacts used for the final motor-control result.
Large raw logs and processed training datasets are intentionally excluded from
git because they are generated experiment data.

## Final Method

- Controller: asynchronous server-assisted PID gain scheduling
- Generator: conditional diffusion U-Net
- Sampling: DDIM 20
- Chunk: 20 gain steps, 0.1 s per step, 2.0 s horizon
- Real system: ESP32 local PID controller + encoder DC motor + Kafka

## Key Result

Scenario: `70 -> 95 -> 90 -> 73 RPM`, transitions at `5, 10, 15 s`.

| Method | Runs | IAE | After-change IAE | Generator p90 |
| --- | ---: | ---: | ---: | ---: |
| Full Diffusion DDIM20 | 5 | 53.73 +/- 0.81 | 21.49 +/- 0.69 | 0.611 s |
| Light32 DDIM20 | 5 | 58.56 +/- 1.94 | 26.42 +/- 1.87 | 0.667 s |

The final selected model is the full DDIM20 U-Net because it gives the best
tracking performance while staying within the asynchronous chunk update window.

## Main Files

- `models/diffusion_gain_chunk_unet_balanced1000_global_topk_full_20260508_193250.joblib`
- `models/diffusion_gain_chunk_unet_balanced1000_global_topk_full_20260508_193250.weights.h5`
- `summary/full_diffusion_ddim20_30_40_50_comparison_20260511_1354.csv`
- `summary/full_diffusion_ddim20_5run_repeats_20260511_1354.csv`
- `summary/full_diffusion_ddim20_30_50_segment_compare_20260511_1354.csv`
- `summary/full_diffusion_ddim20_30_50_decel_gain_compare_20260511_1354.csv`

## Example Runtime Command

Start the recommender server:

```bat
call C:\Users\Jaewook\anaconda3\Scripts\activate.bat tensorflow
python src\gain_recommender_server.py ^
  --generator diffusion_unet_gain_chunk ^
  --diffusion-unet-model-path artifacts\final_ddim20\models\diffusion_gain_chunk_unet_balanced1000_global_topk_full_20260508_193250.joblib ^
  --diffusion-ddim-steps 20 ^
  --diffusion-sample-count 1 ^
  --inference-delay 0.5 ^
  --disable-artificial-inference-sleep ^
  --chunk-horizon-steps 20 ^
  --command-min-interval 0.5
```

Run the local ESP32 controller:

```bat
python src\local_kafka_controller.py ^
  --schedule-apply-mode delay_aware ^
  --sim-time 20 ^
  --target-sequence "70,95,90,73" ^
  --target-change-times "5,10,15" ^
  --run-label final_ddim20
```
