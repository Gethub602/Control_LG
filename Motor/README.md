# Motor Diffusion Gain-Chunk Control

This project implements and validates an asynchronous server-assisted PID gain
scheduling method for an ESP32-controlled DC motor.

The final research direction is a conditional diffusion U-Net that generates a
time-varying PID gain chunk. The ESP32 keeps the low-level PID loop local, while
the PC/Kafka server asynchronously publishes future gain schedules.

## Final Method

- Local controller: ESP32 PID loop with encoder RPM feedback
- Server: Kafka consumer/producer running the gain scheduler
- Policy: conditional diffusion U-Net
- Sampler: DDIM 20
- Chunk: 20 steps, 0.1 s per step, 2.0 s horizon
- Application: delay-aware schedule buffer on the local controller

## Key Result

Final validation scenario:

```text
70 -> 95 -> 90 -> 73 RPM
target changes at 5, 10, and 15 s
```

The final DDIM20 diffusion model achieved the best real-motor tracking result
among the tested DB/RF/DL/direct-policy/diffusion variants.

Compact final artifacts are stored in:

```text
artifacts/final_ddim20/
```

Large raw logs, processed datasets, and generated intermediate results are
ignored by git. They can be regenerated from the scripts under `src/`.

## Main Runtime Commands

Start the Kafka-based gain recommender:

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

Run the ESP32 local controller:

```bat
python src\local_kafka_controller.py ^
  --schedule-apply-mode delay_aware ^
  --sim-time 20 ^
  --target-sequence "70,95,90,73" ^
  --target-change-times "5,10,15" ^
  --run-label final_ddim20
```

## Kafka Topics

- `motor_state`
- `motor_schedule_chunk`
- `motor_gain_command`

## Notes

This architecture does not replace the low-level real-time controller with a
network loop. Kafka and the PC server only update gain chunks; if communication
is delayed, the ESP32 continues with the currently available local PID gains or
the latest valid schedule.
