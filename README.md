# SiliGemma (`Moissanite-Labs/siligemma`)

**SiliGemma** is the bare-metal Ampere `sm_86` (`Form B`) inference engine, `[16, 11, 4]` & Hamming `(72, 64)` SECDED lattice toolkit, and zero-backprop recurrent inference suite for [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it).

- **GitHub Repository:** [`Moissanite-Labs/siligemma`](https://github.com/Moissanite-Labs/siligemma)
- **Flagship Hugging Face Release (`SiliGemma V2`, `47`-Kernel `sm_86`):** [`ahrazzle/SiliGemma-2.0`](https://huggingface.co/ahrazzle/SiliGemma-2.0) · [**`SiliQuant Flagships`**](https://huggingface.co/collections/ahrazzle/siliquant-flagships-6b09fd348ec8b911eb469008)
- **Prior Stage-2 Release (`SiliGemma 1.0`):** [`ahrazzle/SiliGemma-1.0`](https://huggingface.co/ahrazzle/SiliGemma-1.0)
- **Form A Hugging Face Model (Standard-Stack Drop-In):** [`ahrazzle/SiliGemma-4-E2B-FormA`](https://huggingface.co/ahrazzle/SiliGemma-4-E2B-FormA)
- **Form B Hugging Face Model (Stage-1 Baseline):** [`ahrazzle/SiliGemma-4-E2B-FormB-Native`](https://huggingface.co/ahrazzle/SiliGemma-4-E2B-FormB-Native)
- **License:** Apache License 2.0 (inherited from `google/gemma-4-E2B-it`; see [`LICENSE`](LICENSE) and [`NOTICE_UPSTREAM_LICENSE.txt`](NOTICE_UPSTREAM_LICENSE.txt))

---

## Core Architectural Breakthroughs (`SiliGemma V2` — Verified `PASS`, `20/20` Gates)

1. **Zero-Anchor Bare-Metal C++20/CUDA `sm_86` `47`-Kernel Engine (`kernels/`):**
   - `siligemma_bench_sm86` (`namespace siligemma` + `namespace s184b` + `namespace v2`) loads both the shipped bit-identical `slgb-v1` archive (`gemma4_formb_g32.slgb`, `2,094,062,080 B`, IEEE CRC-32 `flags=3`, `0.0 s` repack reuse) and U-4 sidecars/archives, linking strictly standard system libraries (`libc`, `libstdc++`, `libm`, `libdl`, `libpthread`, `libgcc_s`) with **zero Python, PyTorch, `transformers`, `cuBLAS`, or Form-A CSR dependencies**.
   - **`47` compiled `sm_86` kernels** (verified across BOTH the standalone binary and `libsiligemma_kernels.so`) enforce **`<= 48` registers** (`max_regs = 48`), **`0 B` stack**, **`0 B` spill loads/stores**, and **`0` `LDL`/`STL`** SASS instructions.
   - `tree_reduce_128_sm86` achieves **`0`-ULP** bit-identical binary reduction across CPU and GPU; full 64-bit **`Hamming(72, 64)` `SECDED`** (`secded72_check32_heal_dev`, repairing 1-bit `SEC` faults in-place cleanly without poisoning) alongside `HoloSphereRazorCanary64` + golden-trace `IRIS` traps passed **`3/3` LSB SEC repairs (`0.0` heal diff)**, **`1/1` full 64-bit ECC SEC repair (`0.0` diff)**, **`3/3` DED traps**, and **`1/1` IRIS golden-mismatch traps** in `--selftest-gpu`.
2. **`151.41 tok/s` `B=1` & `940.50 tok/s` `B=16` Decode, Exact `FP32` Parity, & `2,154.83 MiB` Static `128K` Arena:**
   - Exact `int32_t` 3-limb radix-128 **`dp4a22`** GEMVs (`B=1`), `SPLIT_T = 16` occupancy-scaled attention (`attention_kv_split_sm86<16>`), parallel `stageA+stageB` RMSNorm/`H_4`/quantization (`norm_b1_v2`), SMEM-cached GeGLU + 2-row `Split-K=8` `down_proj`, and FP16-pair **`mma.sync.m16n8k16`** Tensor Core GEMMs (`B=16`) push decode to **`151.41 tok/s`** (`B=1 K=1`, **`+11.16%`** vs `1.0`), **`118.99 tok/s`** (`B=1 K=2`, `+10.34%`), **`128.65 tok/s`** (`B=1` live Adaptive `mean K=1.642`), **`940.50 tok/s`** (`B=16 K=1`, **`+6.46%`**), and **`741.72 tok/s`** (`B=16 K=2`, `+4.58%`).
   - Achieves exact **`FP32` `.slgb` parity** (`bos = 2` fail-closed across all batch rows): **`3.042e-4`** (`6/6`) at `B=1 K=1`, **`3.295e-4`** (`6/6`, non-vacuous `max_delta_vs_k1 = 9.759`) at `B=1 K=2 U4+LoopCD`, **`4.902e-4`** (`6/6`) at `B=1 K=3 DGS-QR`, and **`3.196e-4`** (`48/48`) at `B=16 K=1`.
   - Exact static `128K` VRAM allocation is **`2,154.83 MiB`** (`2,259,499,660 B <= 2,457.6 MiB`; `device_mem_info_128k_mib = 2,334.94 MiB`) tested at live high context index (`pos = 131071`), with **`0 B` dynamic allocation** over `1,000` decode steps, **`+0.79%`** async `17,920 B/tok` PLE DMA stall, and **`+2.24%`** hoisted `256 B` Woodbury–Cayley steering overhead (`wb_err = 1.192e-7`, `norm_drift = 2.299e-9`).
3. **Outlier Guard (`{438, 1295, 221, 850}`), `LoopCD-Hidden`, Double-GS QR (`K=3`), & Pure-$\mathbb{Q}$ Hadamard Ablation:**
   - Top-4 empirically profiled residual outlier channels (`dim 438: 141.0`, `1295: 96.0`, `221: 84.5`, `850: 67.0`) are masked in `FP32` inside `LoopCD-Hidden` (`w* = -0.500` calibrated on disjoint validation/train splits, persisted live into CUDA with `rho_cuda = 0.00207 > 0`) and clean-band Adaptive `K in {1, 2}` halting (`tau_sr = 1.0016`, `tau_cos = 0.3036`), lowering `GSM8K test` NLL from `2.1174` (`K=1`) to **`2.0391`** (`delta = -0.0863`, `perm_p = 9.999e-5`) and `WT2 test` NLL (`Adaptive`) to **`5.8625`**.
   - Full-matrix Double Gram–Schmidt QR on `K=3` (`53L`, `0 B` extra KV) cuts orthogonal projection leak from `2.831e-7` to **`9.686e-8 <= 1e-6`** and contracts unclamped `rho_raw = 0.7282` to **`rho_damped = 0.3249 < 0.6180`**.
   - Exact rational (`fractions.Fraction`) `3/3` orthogonal checks, real measured per-tensor Pareto-BPW across `{3.125, 4.125, 4.500, 16.000}` bpw (`277` tensors @ uniform `4.500 bpw` = `1,282,203,648 B`, `mse_delta = +0.0000%`), and `g=32` INT4 Hadamard ablation establishing **`Sloane Had.16.3`** improves `WT2 test` PPL by **`-30.53%`** (`397.41 -> 276.06`) while keeping `gemma4_formb_g32.slgb` bit-identical (`repack_triggered = false`, `0.0 s` repack).

---

## Measured RTX 3080 (`sm_86`) `SiliGemma V2` Scorecard & NVML Energy Profile

All GPU timings and `100 Hz` NVML (`libnvidia-ml.so.1` steady-clock trapezoidal integration) power measurements were recorded live on an `NVIDIA GeForce RTX 3080 Laptop GPU (8,192 MiB, sm_86)` with **`provenance_tag = UNCONTROLLED_LAPTOP_NOT_CLAIMED`** (live D2D `cudaMemcpy` peak = **`357.63 GB/s`**; `0` uninjected SECDED/IRIS trips across all runs).

| Batch | Depth (`K`) | `lm_head` Mode | Ctx (`T`) | Latency (`ms`) | Throughput (`tok/s`) | Speedup vs. V1 / Notes | NVML (`W` / `J/tok` / `tok/J` / `Wh` / `peak °C` / `Hz`) |
|---:|---:|---|---:|---:|---:|:---|:---|
| `B=1` | `K=1` (`35L`) | `4.5bpw_shadow` | `128` | 6.604 | **151.41** | **`+11.16%`** vs V1 `136.22` (`215.72 GB/s`; adv `152.13`) | `79.77 W` · `0.5319 J/tok` · `1.880 tok/J` · `0.03369 Wh` · `68 °C` · `100.0 Hz` |
| `B=1` | `K=1` (`35L`) | `bf16_tied` | `128` | 8.034 | **124.47** | `+11.90%` vs `111.23` (`260.83 GB/s`) | `79.90 W` · `0.6447 J/tok` · `1.551 tok/J` · `0.02686 Wh` · `69 °C` · `100.0 Hz` |
| `B=1` | `K=1` (`35L`) | `4.5bpw_shadow` | `1024` | 7.272 | **137.52** | Full-window context (`113.98 -> 137.52`) | `—` |
| `B=1` | `K=2` (`44L`) | `4.5bpw_shadow` | `128` | 8.404 | **118.99** | **`+10.34%`** vs V1 `107.84` (`0 B` extra KV) | `79.80 W` · `0.6733 J/tok` · `1.485 tok/J` · `0.03348 Wh` · `70 °C` · `100.0 Hz` |
| `B=1` | `K=2` (`44L`) | `bf16_tied` | `128` | 9.814 | **101.90** | `0 B` extra KV (`91.50 -> 101.90`) | `—` |
| `B=1` | `Adaptive` (live) | `4.5bpw_shadow` | `128` | 7.773 | **128.65** | Live CUDA adaptive (`mean K=1.642`) | `79.82 W` · `0.6204 J/tok` · `1.612 tok/J` · `0.03326 Wh` · `71 °C` · `100.0 Hz` |
| `B=1` | `K=3` (`53L`) | `4.5bpw_shadow` | `128` | 10.183 | **98.21** | Double-GS QR (`0 B` extra KV) | `—` |
| `B=16` | `K=1` (`35L`) | `4.5bpw_shadow` | `128` | 17.012 | **940.50** | **`+6.46%`** vs V1 `883.45` (`mma.m16n8k16`) | `79.87 W` · **`0.0853 J/tok`** · **`11.727 tok/J`** · `0.03373 Wh` · `72 °C` · `100.0 Hz` |
| `B=16` | `K=1` (`35L`) | `bf16_tied` | `128` | 18.274 | **875.55** | `+8.68%` vs `805.62` (`mma.m16n8k16`) | `79.86 W` · `0.0915 J/tok` · `10.926 tok/J` · `0.02685 Wh` · `73 °C` · `100.0 Hz` |
| `B=16` | `K=2` (`44L`) | `4.5bpw_shadow` | `128` | 21.572 | **741.72** | **`+4.58%`** vs V1 `709.23` (`0 B` extra KV) | `79.85 W` · `0.1084 J/tok` · `9.226 tok/J` · `0.03372 Wh` · `73 °C` · `100.0 Hz` |

> [!NOTE]
> Energy readings are not guaranteed to be consistent due to uncontrollable variables (thermal state, ambient temperature, fan curve, and dynamic power limits).

### Logit Parity vs. Independent `FP32` `.slgb` Dequantization Reference

| Inference Path | Reference | Max Logit Error | Mean Logit Error | Top-1 Exact | Max Delta vs. `K=1` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `B=1` `K=1` `dp4a22` (`4.5 bpw`) | `FP32` `.slgb` ref | **`3.042e-4`** | `6.365e-5` | **`6/6`** | `—` |
| `B=1` `K=2` `U-4 + LoopCD` (`w=-0.50`) | `FP32` `.slgb` ref | **`3.295e-4`** | `7.371e-5` | **`6/6`** | `9.759` (non-vacuous) |
| `B=1` `K=3` `DGS-QR` (`53L`) | `FP32` `.slgb` ref | **`4.902e-4`** | `1.061e-4` | **`6/6`** | `25.152` |
| `B=16` `K=1` `mma.m16n8k16` (`4.5 bpw`) | `FP32` `.slgb` ref | **`3.196e-4`** | `1.148e-5` | **`48/48`** | `—` |

---

## Build, Verification, & CPU Packing

```bash
# 1. Pure-CPU C++20 reference parity & negative controls (0 GPU, < 200 MiB RAM)
make test-cpu && make clean

# 2. 3x -I isolated execution parity
CUDA_VISIBLE_DEVICES="" python3 -I -B tests/test_siligemma_cpu.py
CUDA_VISIBLE_DEVICES="" python3 -I -O -B tests/test_siligemma_cpu.py
CUDA_VISIBLE_DEVICES="" python3 -I -B -W error tests/test_siligemma_cpu.py

# 3. Compile sm_86 CUDA bench + shared .so & verify <= 48 regs, 0 B stack/spill, 0 LDL/STL in SASS
make audit-sass && make clean

# 4. Stream-pack google/gemma-4-E2B-it safetensors -> Form B .slgb on CPU (< 250 MiB RAM)
python3 -B tools/pack_siligemma_formb.py pack --safetensors /path/to/model.safetensors --out ./gemma4_formb_g32.slgb
python3 -B tools/pack_siligemma_formb.py verify ./gemma4_formb_g32.slgb
```

---

## Custom Infrastructure Stack & Source Repositories

| Stack Layer | Component | Role in SiliGemma | Public Repository |
| :--- | :--- | :--- | :--- |
| **SiliGemma Engine** | **SiliGemma** | Bare-metal `sm_86` Gemma-4 Form B kernels, `.slgb` packer, & Form A/B verification | [`Moissanite-Labs/siligemma`](https://github.com/Moissanite-Labs/siligemma) |
| **Lattice Quantization & ECC** | **SiliQuant** | `[16, 11, 4]` Extended Hamming + `Hamming(72, 64)` SECDED `A_16` + `BlockH4` quantization | [`Moissanite-Labs/siliquant`](https://github.com/Moissanite-Labs/siliquant) |
| **Bare-Metal Silicon Kernels** | **SiliKernels** | Zero-spill Ampere `sm_86` & Intel `AVX-512` primitives | [`Moissanite-Labs/silikernels`](https://github.com/Moissanite-Labs/silikernels) |
| **Verified Zero-Spill Compiler** | **HoloEmit** | Pure-`Init` `Lean 4.36.0` compiler targeting zero-spill `sm_86` PTX & `AVX-512` | [`Moissanite-Labs/holoemit`](https://github.com/Moissanite-Labs/holoemit) |
| **Metrology & Telemetry** | **SiliSense** | Pure-stdlib silicon physics metrology & fault-canary telemetry | [`Moissanite-Labs/silisense`](https://github.com/Moissanite-Labs/silisense) |
| **Bounded Adaptation** | **ML Weft / Weft-Loom** | Deterministic weight-space bounds & zero-drift rollback | [`Moissanite-Labs/weft`](https://github.com/Moissanite-Labs/weft) · [`Moissanite-Labs/weft-loom`](https://github.com/Moissanite-Labs/weft-loom) |
| **Workbench OS & Exoskeleton** | **ML-OS / Aletheion** | Research workbench OS & host hardware exoskeleton | [`Moissanite-Labs/ml-os`](https://github.com/Moissanite-Labs/ml-os) · [`Foundation-0/aletheion`](https://github.com/Foundation-0/aletheion) |
| **Verification & Audit Harness** | **Aletheia** | 3-mode `-I` parity, negative-control gates, & leak scanning | [`Foundation-0/aletheia`](https://github.com/Foundation-0/aletheia) |
| **Five-Layer Stack Blueprint** | **Siliconcordat** | Open L1–L5 stack specification | [`siliconcordat/.github`](https://github.com/siliconcordat/.github) |

---

## Verified Inventory & Documentation Artifacts

`docs/` preserves the V2 live verification receipt ([`docs/V2_VERDICT.json`](docs/V2_VERDICT.json)) and markdown report ([`docs/V2_REPORT.md`](docs/V2_REPORT.md)) alongside the historical Stage-1 (`STAGE1_VERDICT.json`) and Stage-2 (`STAGE2_VERDICT.json`) artifacts verified in CI via `EXPECTED_PINNED_SHA256` (`tests/test_siligemma_cpu.py`).

| Item | Historical Stage-1 Doc Value | Verified Code / Artifact Value & Clarification |
|---|---|---|
| **FP32 Norm & Scalar Tensor Count** | `247` (Table 2.1) | **`262`** tensors (`15 x 8 + 20 x 7 + 2 = 262`; `285,987` FP32 params already included the `15` `k_norm` vectors). |
| **Checkpoint Text Param Total** | `4,666,330,403` (Table 2.1) | **`4,647,449,891`** (`4,628,569,379` active + `18,880,512` dead KV; `4,666,330,403` double-added dead KV). |
| **Static `128K` VRAM Footprint** | `2,744.94 MiB` (`2.68 GiB`) | V2 exact device allocation = **`2,154.83 MiB`** (`2,259,499,660 B <= 2,457.6 MiB`; whole-device `cudaMemGetInfo = 2,334.94 MiB`, `pos = 131071` verified). |
| **`SECDED` Protection & IRIS** | "`3/3` SEC / `3/3` DED / `0` stale logits" | Upgraded in V2 to full 64-bit **`Hamming(72, 64)`** (`secded72_check32_heal_dev`, healing 1-bit SEC cleanly in-place without poisoning, halting only on 2-bit DED) plus golden-trace `IRIS` mismatch detection (`3/3` + `1/1` SEC healed `0.0` diff, `3/3` DED traps, `1/1` IRIS trap verified live). |
| **EOS Tokens & Bench Mode** | `{1, 106}` in `.slgb` header | `generation_config.json` includes `[1, 106, 50]` (turn stop `50`); `.slgb` stores `{1, 106}` and `siligemma_bench_sm86` enforces `bos=2` across all batch rows. |
