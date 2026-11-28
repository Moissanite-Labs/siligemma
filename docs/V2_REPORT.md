# Track S187 — `SiliGemma V2` Verification Report (`VERDICT: PASS`)

- **Provenance / Energy Control:** `UNCONTROLLED_LAPTOP_NOT_CLAIMED` — UNCONTROLLED_LAPTOP_NOT_CLAIMED: RTX 3080 Laptop thermal/power cap unconstrained; NVML 100 Hz readings are empirical observations rather than controlled power-efficiency claims.
- **Immutable Anchors:** `PREREG=9d94fee5e38fb2ce...`, `ARCH=b047f14809cdadd4...`, `SLGB=111a672f1d52ba83...` (`2.09 GB`), `repack_triggered=False` (`pack_time_s=0.0s`)
- **SASS/PTX Audit (`bench` + `.so`):** `bench_kernels=47`, `so_kernels=47`, `max_regs=48` (`<=48`), `stack_bytes=0`, `local_bytes=0`, `sass_ldl_stl=0`

## Pillar 1 — B=1 & B=16 Kernel Throughput & Memory Safety vs SiliGemma 1.0

| Arm | V1 Baseline (`tok/s` / `ms`) | V2 Measured (`tok/s` / `ms`) | Delta vs V1 | NVML Avg Power | J/tok | tok/J | Wh/run | Hz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B=1 K=1 g=32` | 136.22 / 7.341 | **151.41** / 6.604 | **+11.16%** | 79.77 W | 0.5319 | 1.88 | 0.03369 | 100.0 |
| `B=1 K=2 g=32` | 107.84 / 9.273 | **118.99** / 8.404 | +10.34% | 79.80 W | 0.6733 | 1.49 | 0.03348 | 100.0 |
| `B=1 Adaptive (live)` (`mean K=1.642`) | — | **128.65** / 7.773 | hardware | 79.82 W | 0.6204 | 1.61 | 0.03326 | 100.0 |
| `B=16 K=1 g=32` | 883.45 / 18.111 | **940.50** / 17.012 | **+6.46%** | 79.87 W | 0.0853 | 11.73 | 0.03373 | 100.0 |
| `B=16 K=2 g=32` | 709.23 / 22.560 | **741.72** / 21.572 | +4.58% | 79.85 W | 0.1084 | 9.23 | 0.03372 | 100.0 |

- **VRAM, Adv Decode & SECDED(72,64):** `dynamic_vram_delta_bytes_1k=0 B`, `b1_k1_adv=152.13 tok/s`, `b1_k3=98.21 tok/s`, `exact_128k_alloc_mib=2154.83 MiB` (`<= 2457.6`, `device_mem_info=2334.94 MiB`), `secded_sec=0`, `ded=0`, `iris_poisoned=0`, `ple_dma_stall=+0.79%`, `woodbury_overhead=+2.24%`, `rho_cuda=0.0021`.
- **FP32 .slgb Parity:** `B=1 K=1 max_err=3.042e-04 (6/6)`, `B=1 K=2 U4+LoopCD max_err=3.295e-04 (6/6, delta_vs_k1=9.759e+00)`, `B=1 K=3 DGS max_err=4.902e-04 (6/6)`, `B=16 K=1 max_err=3.196e-04 (48/48)`.

## Pillar 2 — Goodfire Outlier Guard + U-4 ES + LoopCD-Hidden + P233c Adaptive + K=3 Double-GS QR

- **Goodfire Top-8 Outlier Channels:** `[{'absmax': 141.0, 'dim': 438}, {'absmax': 96.0, 'dim': 1295}, {'absmax': 84.5, 'dim': 221}, {'absmax': 67.0, 'dim': 850}, {'absmax': 65.5, 'dim': 195}, {'absmax': 49.25, 'dim': 1297}, {'absmax': 45.75, 'dim': 393}, {'absmax': 41.5, 'dim': 59}]` (top-4 `[438, 1295, 221, 850]` kept FP32-invariant inside CUDA `LoopCD-Hidden` & `P233c` kernels).
- **Calibrated Parameters (disjoint `wt2_val`/`cd_train`/`gsm_train`):** `loopcd_w*=-0.500`, `tau_sr=1.0016`, `tau_cos=0.3036`.
- **Held-Out NLL (`K=1` -> `K=2+U4+LoopCD` -> `Adaptive` -> `K=3 DGS-QR`):**
  - `WT2 test` (`N=16`): `K=1=5.8654`, `K=2_raw=7.2333`, `K=2+U4+LoopCD=5.9246` (`delta=+0.0592`, `95% CI=[+0.0163, +0.1022]`, `boot_p=0.0046`, `perm_p=0.0229`), `Adaptive=5.8625`, `K=3_DGS=7.5606`.
  - `Countdown test` (`N=128`): `K=1=1.0764`, `K=2+U4+LoopCD=1.0982` (`delta=+0.0219`, `perm_p=0.0179`), `Adaptive=1.0965`, `K=3_DGS=2.2611`.
  - `GSM8K test` (`N=128`): `K=1=2.1174`, `K=2+U4+LoopCD=2.0391` (`delta=-0.0863`, `perm_p=9.999e-05`), `Adaptive=2.2224`, `K=3_DGS=3.0641`.
- **K=3 Double-GS QR Stability:** `sgs_leak=2.831e-07`, `dgs_leak=9.686e-08` (`<= 1e-6`), `rho_raw=0.7282`, `rho_damped=0.3249` (`< 0.618`).

## Pillar 3 — Pure-Q Hadamard (`Block-H4` vs `Sylvester H_16` vs `Sloane Had.16.3`) & `#15550` Pareto-BPW

- **Pure-Q Orthogonality:** `3/3`; **Pure-Q `#15550` Pareto-BPW (`277` tensors):** `4.5000 bpw` (`1282203648 B <= 1282203648 B`, MSE delta `+0.0000%`).
- **277-Tensor Weight SQNR (`dB` / `l_inf`):** `Block-H4=20.4050 dB` (`0.1123`), `Sylvester H_16=20.4980 dB` (`0.1243`), `Sloane Had.16.3=20.4972 dB` (`0.1406`).
- **Live Held-Out `WT2 test` PPL / `CD` / `GSM` NLL:** `Block-H4=397.41` / `1.0616` / `2.1692` vs `Sylvester=316.73` / `0.9628` / `2.1366` vs `Sloane Had.16.3=276.06` / `1.0645` / `2.0677`.
- **Scope Note (`repack_triggered=False`):** Immutable SiliGemma 1.0 archive gemma4_formb_g32.slgb uses pinned Block-H4 wiring; 16x16 runtime FWHT kernels and offline Sylvester/Sloane re-packing are deferred out of V2 scope (best_rot_live=sloane_had16_3, sylvester_beats_all3=True, sloane_beats_all3=False).

## Kill Gates
`{"arena_128k_le_cap": true, "b16_k1_ge_870": true, "b16_k1_ge_920_v2": true, "b1_k1_ge_135": true, "b1_k1_ge_150_kg_gfb3": true, "non_vacuous_k2_u4_loopcd": true, "nvml_100hz_all_arms_present": true, "pareto_15550_within_cap": true, "parity_b16_top1_and_err": true, "parity_b1_k2_loopcd_top1_and_err": true, "parity_b1_k3_dgs_top1_and_err": true, "parity_b1_top1_and_err": true, "pure_q_had_3of3": true, "riccati_rho_damped_lt_0618": true, "s97_double_gs_leak_le_1e6": true, "sass_regs_le_48": true, "sass_zero_stack_spill_ldl_stl": true, "secded_iris_zero_poison": true, "v1_slgb_unmodified": true, "zero_dynamic_alloc_1k": true}`

