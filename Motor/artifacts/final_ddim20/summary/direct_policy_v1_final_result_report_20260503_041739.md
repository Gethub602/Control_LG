# Surrogate-Objective Direct Policy V1 Final Result Report

## Summary

- Selected model: `esp32_direct_policy_mlp_20260502_153840`
- Online policy: Direct MLP only, 28 state features -> `Kp, Ki, Kd`
- Training method: Direct policy optimized through a frozen MT-DL horizon surrogate objective
- Runtime setting: inference delay `0.5 s`, chunk horizon `20` steps, chunk duration about `2.0 s`
- Final validation: `30` runs across `6` scenarios
- Scenario-average IAE: `54.2682`
- Scenario-average after-change IAE: `17.4433`
- Mean max PWM: `118.4435`
- Peak max PWM over runs: `127.9664`
- Clean-baseline wins where available: IAE `4/4`, after-change IAE `4/4`

## Model Structure

```text
Online inference:
  controller state (28 features)
    -> Direct Policy MLP
    -> normalized [kp, ki, kd]
    -> gain bounds: Kp 0.6~1.4, Ki 0.7~2.5, Kd 0.0~0.1
    -> 20-step gain chunk
    -> Kafka / delay-aware controller

Training:
  state -> Direct Policy MLP -> predicted gain
    -> frozen MT-DL horizon surrogate
    -> predicted horizon metrics
    -> loss: horizon IAE + PWM/risk + gain anchor + smoothness
```

## Timing Structure

```text
t = 0.0 s: target/state is published
t ~= 0.008 s: Direct MLP chunk generation p90 completes
t = 0.5 s: delay-aware chunk starts applying
t = 0.5~2.5 s: 20-step chunk coverage
Every ~1 s: subsequent chunks refresh future gain commands
```

## Scenario Set

| scenario               | category             | target_sequence   | target_change_times   |   sim_time |   repeats | description                                                                 |
|:-----------------------|:---------------------|:------------------|:----------------------|-----------:|----------:|:----------------------------------------------------------------------------|
| Seen 70->85->100       | seen interpolation   | 70,85,100         | 5,10                  |         15 |         5 | Seen DB targets with acceleration steps.                                    |
| Unseen 70->81->100     | unseen interpolation | 70,81,100         | 5,10                  |         15 |         5 | Intermediate 81 rpm target not directly in ESP32_REAL_PID_GAIN_DB.          |
| Dynamic 70->95->90->73 | dynamic mixed        | 70,95,90,73       | 5,10,15               |         20 |         5 | Acceleration and deceleration mixed across unseen/intermediate targets.     |
| Reverse 100->85->70    | reverse deceleration | 100,85,70         | 5,10                  |         15 |         5 | Deceleration-focused transition from high to low rpm.                       |
| Large Step 70->100->70 | large step           | 70,100,70         | 5,10                  |         15 |         5 | Large acceleration followed by large deceleration.                          |
| Mixed 85->70->100->81  | mixed robust         | 85,70,100,81      | 5,10,15               |         20 |         5 | Starts from 85 rpm, includes deceleration, acceleration, and unseen 81 rpm. |

## Final 5-Repeat Results

| scenario               | category             |   n |     IAE |   IAE_std |   after_change_IAE |   after_change_IAE_std |   mean_pwm |   max_pwm |   max_pwm_std |   delta_IAE_vs_clean |   delta_after_change_IAE_vs_clean |   delta_max_pwm_vs_clean |
|:-----------------------|:---------------------|----:|--------:|----------:|-------------------:|-----------------------:|-----------:|----------:|--------------:|---------------------:|----------------------------------:|-------------------------:|
| Seen 70->85->100       | seen interpolation   |   5 | 45.0958 |    1.0374 |            12.3773 |                 1.0605 |    88.2394 |  117.1290 |        1.1086 |              -1.9716 |                           -1.5054 |                  -2.1901 |
| Unseen 70->81->100     | unseen interpolation |   5 | 43.5787 |    0.5901 |            11.4370 |                 0.4546 |    86.1687 |  118.5536 |        0.7559 |              -3.2309 |                           -2.1700 |                  -0.9832 |
| Dynamic 70->95->90->73 | dynamic mixed        |   5 | 50.9102 |    0.6816 |            18.2655 |                 0.6134 |    85.2498 |  113.4740 |        9.2655 |              -3.5400 |                           -3.2401 |                   1.6911 |
| Reverse 100->85->70    | reverse deceleration |   5 | 68.2083 |    0.3681 |            14.5294 |                 0.4441 |    86.2694 |  120.8004 |        1.5794 |              -1.7092 |                           -1.3186 |                  -5.9976 |
| Large Step 70->100->70 | large step           |   5 | 54.2818 |    0.5549 |            22.3482 |                 0.3762 |    81.8490 |  122.5113 |        0.6574 |             nan      |                          nan      |                 nan      |
| Mixed 85->70->100->81  | mixed robust         |   5 | 63.5347 |    0.4316 |            25.7021 |                 0.5330 |    86.2562 |  118.1928 |        7.6273 |             nan      |                          nan      |                 nan      |

## Ablation Interpretation

| Trial | Result | Decision |
|---|---|---|
| Clean Direct MLP | Strong supervised baseline, but not trajectory-objective trained | Reference baseline |
| IAE/PWM label weighting | Did not beat clean Direct in real control | Rejected |
| DB-reference / residual DB | Discrete gain prior alone did not improve closed-loop IAE | Rejected |
| Rank-weighted labels | Offline gain MAE improved, real IAE did not | Rejected |
| Surrogate-objective V1 | Won all clean-baseline scenarios in 5-repeat validation | Selected |
| Surrogate-objective V2 PWM penalty | Did not improve average IAE or max PWM over V1 | Rejected |

## Representative Figures

- Unseen 70->81->100: `results\figures\final_v1\clean_vs_v1_unseen_70_to_81_to_100.png`
- Dynamic 70->95->90->73: `results\figures\final_v1\clean_vs_v1_dynamic_70_to_95_to_90_to_73.png`
- Reverse 100->85->70: `results\figures\final_v1\clean_vs_v1_reverse_100_to_85_to_70.png`

## Key Conclusion

The main improvement came from optimizing the Direct policy against a closed-loop horizon objective, not from better gain-label regression. The frozen surrogate is used only during training; online inference remains low-latency because only the Direct MLP runs in the control loop.