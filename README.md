# SiliGemma (`Moissanite-Labs/siligemma`)

**SiliGemma** is the bare-metal Ampere `sm_86` (`Form B`) inference engine, `[16, 11, 4]` Extended Hamming SECDED lattice toolkit, and zero-backprop adaptation suite for [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it).

- **GitHub Repository:** [`Moissanite-Labs/siligemma`](https://github.com/Moissanite-Labs/siligemma)
- **Flagship Hugging Face Release (`SiliGemma 1.0`, Stage-2 `33`-Kernel):** [`ahrazzle/SiliGemma-1.0`](https://huggingface.co/ahrazzle/SiliGemma-1.0) · [**`SiliQuant Flagships`**](https://huggingface.co/collections/ahrazzle/siliquant-flagships-6b09fd348ec8b911eb469008)
- **Form A Hugging Face Model (Standard-Stack Drop-In):** [`ahrazzle/SiliGemma-4-E2B-FormA`](https://huggingface.co/ahrazzle/SiliGemma-4-E2B-FormA)
- **Form B Hugging Face Model (Stage-1 Baseline):** [`ahrazzle/SiliGemma-4-E2B-FormB-Native`](https://huggingface.co/ahrazzle/SiliGemma-4-E2B-FormB-Native)
- **License:** Apache License 2.0 (inherited from `google/gemma-4-E2B-it`; see [`LICENSE`](LICENSE) and [`NOTICE_UPSTREAM_LICENSE.txt`](NOTICE_UPSTREAM_LICENSE.txt))

---

## Core Architectural Breakthroughs (`Stage-2` `SiliGemma 1.0` & `Track S184 / S184b`)

1. **Zero-Anchor Bare-Metal C++20/CUDA `sm_86` `33`-Kernel Engine (`kernels/`):**
   - `siligemma_bench_sm86` (`namespace s184b` + R6-hardened `namespace siligemma`) loads both the shipped `slgb-v1` archive (`gemma4_formb_g32.slgb`, `2,094,062,080 B`, IEEE CRC-32 `flags=3`) and Stage-2 `slgb-v2`, linking strictly standard system libraries (`libc`, `libstdc++`, `libm`, `libgcc_s`) with **zero Python, PyTorch, `transformers`, `cuBLAS`, or Form-A CSR dependencies**.
   - **`33` compiled `sm_86` kernels** (`57` total `ptxas` entries across `kernels/siligemma_kernels_sm86.cu` and `kernels/siligemma_bench_sm86.cu`) enforce **`<= 48` registers**, **`0 B` stack**, **`0 B` spill loads/stores**, and **`0` `LDL`/`STL`** SASS instructions.
   - `tree_reduce_128_sm86` achieves **`0`-ULP** bit-identical binary reduction across CPU/GPU; `[16, 11, 4]` `1-LOP3 + 5-POPC` `SECDED` protects the `A_16` LSB parity plane (`0x11111111`), and `HoloSphereRazorCanary64` + `IRIS` passed **`3/3` LSB SEC repairs**, **`3/3` LSB DED traps**, and **`1/1` IRIS Trojan traps** in `run_gpu_selftest` (`--selftest-gpu`).
2. **Stage-2 Exact `FP32` Logit Parity (`KG-GFB1 PASS`: `6.509e-5` / `cos = 0.9999999999993`):**
   - Replaces Stage-1's 8-bit `dp4a` activation truncation (`0.8598` drift) and `BF16` KV/scale drift (`1.349` drift) with **3-limb radix-128 exact `dp4a22`** (`B=1`, `max_err = 6.509e-5` `4.5 bpw` / `6.104e-5` `BF16` `lm_head`, `cos = 0.9999999999993`, `6/6` top-1 exact) and **FP16-pair `mma.sync.m16n8k16`** Tensor Cores (`B=16`, `max_err = 1.125e-4`, `cos = 0.9999999999987`, `48/48` top-1 exact) against `FP32` `Gemma4ForCausalLM`, storing exact `FP32` KV inside preserved `kArena64KShadow`.
   - Enforces `bos_token_id = 2` fail-closed, quantizes `276` live 2D text linears (`1,055,711,232 B` @ `4.5 bpw g=32`), omits **`60` dead `L15..34` `(k_proj, v_proj, k_norm)` tensors** (`18,880,512` params), packs `262` FP32 norm/scalar vectors (`285,987` params), allocates **`0 B` dynamic VRAM/heap** over `1,000` decode steps (**`2,388.06 MiB` = `2.33 GiB` @ `128K` ctx** live VRAM in `STAGE2_VERDICT.json`, vs `2,574.93 MiB` `kArena128K` theoretical arena / `2,744.94 MiB` Stage-1), and overlaps `17,920 B/tok` pinned host-DMA PLE with **`0.802%` stall**.
3. **`3.87x` Batched Throughput, Hoisted `256 B` Woodbury (`1.89%`), & Joint `H1 × L25..33 K=2` Synergy:**
   - Coalesced `uint4` `dp4a22` GEMV, fused `QKV`, fused `GeGLU+H_4+quant22`, 2-row `Split-K=8` `mlp_down`, and `mma.m16n8k16` lift throughput (`B=16 K=1`: `225.81 -> 873.20 tok/s`, **`3.87x`**; `B=1 K=1`: `89.50 -> 135.02 tok/s`, **`1.51x`**).
   - Hoisted `c_wb_inv_256b` `8x8` Cayley–Woodbury steering on `L10..L14` incurs **`1.89%` overhead** (`wb_err = 1.192e-7`, `norm_drift = 2.299e-9`, `0` KV writes in `L15..34`).
   - Joint **`H1` (`L10..L14` hub) × `L25..L33` `K=2` (`44L`)** contracts (`rho_hat = 0.1720`, `rho_cuda = 0.000517 << 0.6180`, **`0 B` extra KV**, **`0 B` extra bulk weights**) and improves held-out splits (`H1 K=1 -> H1+U4 K=2 tuned`: `WT2_test` `140.702 -> 128.413`, NLL `-0.0914`, `p=9.999e-5`; `Countdown_test` NLL `-0.0444`, `p=9.999e-5`; `GSM8K_OOD` NLL **`-0.2559`**, `p=9.999e-5`; Base `K=1 -> K=2`: `WT2` `213.835 -> 198.273`, `GSM8K Acc` `15.63% -> 18.75%`).

---

## Measured RTX 3080 (`sm_86`) Stage-2 Scorecard, NVML Energy & Honest Limitations

All GPU timings and NVML power measurements were run on an `NVIDIA GeForce RTX 3080 Laptop GPU (8,192 MiB, sm_86)` with **`energy_power_status = UNCONTROLLED_LAPTOP_NOT_CLAIMED`** (live D2D `cudaMemcpy` peak = **`357.78 GB/s`**).

| Batch | Depth (`K`) | `lm_head` Mode | Ctx (`T`) | Latency (`ms`) | Throughput (`tok/s`) | Speedup vs. Stage-1 / Notes | NVML (`W` / `J/tok` / `tok/J` / `peak °C`) |
|---:|---:|---|---:|---:|---:|:---|:---|
| `B=1` | `K=1` (`35L`) | `4.5bpw_shadow` | `128` | 7.406 | **135.02** | `1.51x` (`89.50 -> 135.02`, `173.12 GB/s`) | `73.49 W` · `0.5462 J/tok` · `1.831 tok/J` · `70 °C` |
| `B=1` | `K=1` (`35L`) | `bf16_tied` | `128` | 8.991 | **111.23** | `1.42x` vs Stage-1 `78.56` (`206.99 GB/s`) | `—` |
| `B=1` | `K=1` (`35L`) | `4.5bpw_shadow` | `1024` | 8.773 | **113.98** | Full-window context | `—` |
| `B=1` | `K=2` (`44L`) | `4.5bpw_shadow` | `128` | 9.421 | **106.15** | `1.82x` vs Stage-1 `58.46` (`0 B` extra KV) | `79.80 W` · `0.7520 J/tok` · `1.330 tok/J` · `71 °C` |
| `B=1` | `K=2` (`44L`) | `bf16_tied` | `128` | 10.929 | **91.50** | `1.71x` vs Stage-1 `53.60` (`0 B` extra KV) | `—` |
| `B=16` | `K=1` (`35L`) | `4.5bpw_shadow` | `128` | 18.324 | **873.20** | **`3.87x`** (`225.81 -> 873.20`) | `79.87 W` · **`0.0919 J/tok`** · **`10.879 tok/J`** · `72 °C` |
| `B=16` | `K=1` (`35L`) | `bf16_tied` | `128` | 19.861 | **805.62** | **`3.83x`** (`210.41 -> 805.62`) | `—` |
| `B=16` | `K=2` (`44L`) | `4.5bpw_shadow` | `128` | 22.882 | **699.23** | **`3.99x`** (`175.29 -> 699.23`, `0 B` extra KV) | `79.90 W` · `0.1151 J/tok` · `8.689 tok/J` · `72 °C` |

> [!NOTE]
> Energy readings are not guaranteed to be consistent due to uncontrollable variables (thermal state, ambient temperature, fan curve, and dynamic power limits). *(Measured via `100 Hz` NVML INA3221 rail trapezoidal integration cross-checked within `<= 0.95 W` by `100 ms` `nvidia-smi`; see `nvmlDeviceGetTotalEnergyConsumption` drift disclosure in [`ahrazzle/SiliGemma-1.0`](https://huggingface.co/ahrazzle/SiliGemma-1.0).)*

- **Honest Stage-2 Gate Disposition (`STAGE2_VERDICT.json`: `overall_verdict = FAIL_GATE` — `KG-GFB1=PASS`, `KG-GFB2=PASS`, `KG-GFB3=FAIL`, `KG-GFB4=PASS`, `provenance_tag = UNCONTROLLED_LAPTOP_NOT_CLAIMED`):**
  - Stage-2 resolved Stage-1's parity (`KG-GFB1 PASS`: `0.8598 -> 6.509e-5`) and batched Tensor Core (`B=16`: `873.20 / 805.62 / 699.23 tok/s`) targets.
  - `KG-GFB3` stays honestly `FAIL` solely because single-stream canonical `0`-ULP `B=1` hit `135.02 / 111.23 / 106.15 / 91.50 tok/s` (`173.12–206.99 GB/s`) below the strict `150 / 122 / 120 / 98 tok/s` floor (bounded by serial per-layer norm synchronization and small-matrix occupancy relative to the `357.78 GB/s` D2D stream peak).
  - *Note on `slgb-v1` vs `slgb-v2`:* The shipped standalone `gemma4_formb_g32.slgb` archive (`slgb-v1`, `flags=3`) initializes the `U-4` slot to `K=1` identity (`wf=0, wg=0, mq=0`); `siligemma_bench_sm86` loads both `slgb-v1` and Stage-2 `slgb-v2` directly.

---

## Build, Verification, & CPU Packing

```bash
# 1. Pure-CPU C++20 reference parity & R5 negative controls (0 GPU, < 200 MiB RAM)
make test-cpu && make clean

# 2. 3x -I isolated execution parity
CUDA_VISIBLE_DEVICES="" python3 -I -B tests/test_siligemma_cpu.py
CUDA_VISIBLE_DEVICES="" python3 -I -O -B tests/test_siligemma_cpu.py
CUDA_VISIBLE_DEVICES="" python3 -I -B -W error tests/test_siligemma_cpu.py

# 3. Compile sm_86 CUDA kernels & verify 0 B stack/spill + 0 LDL/STL in SASS (requires CUDA toolkit)
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
| **Lattice Quantization & ECC** | **SiliQuant** | `[16, 11, 4]` Extended Hamming `A_16` + `BlockH4` quantization | [`Moissanite-Labs/siliquant`](https://github.com/Moissanite-Labs/siliquant) |
| **Bare-Metal Silicon Kernels** | **SiliKernels** | Zero-spill Ampere `sm_86` & Intel `AVX-512` primitives | [`Moissanite-Labs/silikernels`](https://github.com/Moissanite-Labs/silikernels) |
| **Verified Zero-Spill Compiler** | **HoloEmit** | Pure-`Init` `Lean 4.36.0` compiler targeting zero-spill `sm_86` PTX & `AVX-512` | [`Moissanite-Labs/holoemit`](https://github.com/Moissanite-Labs/holoemit) |
| **Metrology & Telemetry** | **SiliSense** | Pure-stdlib silicon physics metrology & fault-canary telemetry | [`Moissanite-Labs/silisense`](https://github.com/Moissanite-Labs/silisense) |
| **Bounded Adaptation** | **ML Weft / Weft-Loom** | Deterministic weight-space bounds & zero-drift rollback | [`Moissanite-Labs/weft`](https://github.com/Moissanite-Labs/weft) · [`Moissanite-Labs/weft-loom`](https://github.com/Moissanite-Labs/weft-loom) |
| **Workbench OS & Exoskeleton** | **ML-OS / Aletheion** | Research workbench OS & host hardware exoskeleton | [`Moissanite-Labs/ml-os`](https://github.com/Moissanite-Labs/ml-os) · [`Foundation-0/aletheion`](https://github.com/Foundation-0/aletheion) |
| **Verification & Audit Harness** | **Aletheia** | 3-mode `-I` parity, R1–R5 fail-closed gates, & leak scanning | [`Foundation-0/aletheia`](https://github.com/Foundation-0/aletheia) |
| **Five-Layer Stack Blueprint** | **Siliconcordat** | Open L1–L5 stack specification | [`siliconcordat/.github`](https://github.com/siliconcordat/.github) |

---

## Verified Inventory & Errata vs. Immutable Track S184 / S184b Docs

The historical Stage-1 files in `docs/` (`STAGE1_VERDICT.json`, `STAGE1_REPORT.md`, `PREREG_W22_GEMMA4_FORMB.json`, `ARCHITECTURE_FORMB_SILIGEMMA.md` SHA-256 `b7410d0c...`) and Stage-2 files (`STAGE2_VERDICT.json`, `STAGE2_REPORT.md`, `PREREG_W22_GEMMA4_FORMB_S2.json`, `ARCHITECTURE_FORMB_S2.md` SHA-256 `2c0be6c7...`) are preserved byte-identical to their executed RTX 3080 runs (`EXPECTED_PINNED_SHA256` in `tests/test_siligemma_cpu.py`). `kernels/` carries the Stage-2 `33`-kernel upgrade (`namespace s184b` + R6-hardened `namespace siligemma`) with direct `slgb-v1` + `slgb-v2` loading. Post-Stage-1 hardening and doc errata are recorded transparently below:

| Item | Historical S184 Doc Value | Verified Code / Artifact Value & Clarification |
|---|---|---|
| **FP32 Norm & Scalar Tensor Count** | `247` (`ARCHITECTURE` Table 2.1) | **`262`** tensors (`15 x 8 + 20 x 7 + 2 = 262`; `285,987` FP32 params already included the `15` `k_norm` vectors). |
| **Checkpoint Text Param Total** | `4,666,330,403` (`ARCHITECTURE` Table 2.1) | **`4,647,449,891`** (`4,628,569,379` active + `18,880,512` dead KV; `4,666,330,403` double-added dead KV). |
| **Partition 7 & `kArena128K` Sizes** | `19.34 MiB` (`ARCHITECTURE` §3.1) | **`19.28 MiB`** (`20,217,856 B`); `kArena128K` (`2,700,007,680 B`) total VRAM = **`2,744.94 MiB` (`2.68 GiB`)**. |
| **`[16, 11, 4]` SECDED Protection Scope** | "`3/3` SEC / `3/3` DED / `0` stale logits" | Protects the **`A_16` LSB parity plane (`0x11111111`)** (`run_selftest_gpu` verified injected LSB SEC/DED + IRIS); release hardening added `siligemma_iris_canary_finalize_sm86` + host `evaluate_canary_watchdog` + `.slgb` IEEE CRC-32 + token/position bounds + PLE header verification in `siligemma_arena.hpp`/`siligemma_bench_sm86.cu` (`nvcc` `sm_86` zero-spill/`<= 48`-reg compile & C++20 `make test-cpu` verified; host harness changes not re-executed on RTX 3080). |
| **Form A Cross-Track Reference Values** | `S182b2` `1,203,561,445 B`, `S184 G0` `5.118627` | Pruned `S182b4` `H1` `model.safetensors` linears = **`1,203,427,606 B`**; `S182b3/b4 H0` `WT2 = 5.116801` vs `S184 G0` `5.118627`. |
| **EOS Tokens & Bench Mode** | `{1, 106}` in `.slgb` header | `generation_config.json` includes `[1, 106, 50]` (turn stop `50`); `.slgb` stores `{1, 106}` and `siligemma_bench_sm86` runs fixed-length timing/WT2 eval without an EOS loop. |
