# SiliGemma (`Moissanite-Labs/siligemma`)

**SiliGemma** is the bare-metal Ampere `sm_86` (`Form B`) inference engine, `[16, 11, 4]` Extended Hamming SECDED lattice toolkit, and zero-backprop adaptation suite for [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it).

- **GitHub Repository:** [`Moissanite-Labs/siligemma`](https://github.com/Moissanite-Labs/siligemma)
- **Form A Hugging Face Model (Standard-Stack Drop-In):** [`ahrazzle/SiliGemma-4-E2B-FormA`](https://huggingface.co/ahrazzle/SiliGemma-4-E2B-FormA)
- **Form B Hugging Face Model (Anchor-Free Bare-Metal `sm_86`):** [`ahrazzle/SiliGemma-4-E2B-FormB-Native`](https://huggingface.co/ahrazzle/SiliGemma-4-E2B-FormB-Native)
- **License:** Apache License 2.0 (inherited from `google/gemma-4-E2B-it`; see [`LICENSE`](LICENSE) and [`NOTICE_UPSTREAM_LICENSE.txt`](NOTICE_UPSTREAM_LICENSE.txt))

---

## Core Architectural Breakthroughs (`Track S182b3/b4` & `Track S184`)

1. **Zero-Anchor Bare-Metal C++20/CUDA `sm_86` Engine (`kernels/`):**
   - `siligemma_bench_sm86` links strictly standard system libraries (`libc`, `libstdc++`, `libm`, `libgcc_s`) with **zero Python, PyTorch, `transformers`, `cuBLAS`, or Form-A CSR dependencies**.
   - `20/20` compiled `sm_86` kernels enforce **`<= 48` registers**, **`0 B` stack**, **`0 B` spill loads/stores**, and **`0` `LDL`/`STL`** SASS instructions.
   - `tree_reduce_128_sm86` achieves **`0`-ULP** bit-identical binary reduction across CPU/GPU; `[16, 11, 4]` `1-LOP3 + 5-POPC` `SECDED` protects the `A_16` LSB parity plane (`0x11111111`; bits 1–3 carry quotient indices without ECC), and `HoloSphereRazorCanary64` + `IRIS` passed **`3/3` LSB SEC repairs**, **`3/3` LSB DED traps**, and **`1/1` IRIS Trojan traps** in `run_selftest_gpu`.
2. **Exact Topology & `bos=2` Tokenizer Fix:**
   - Enforces `bos_token_id = 2` fail-closed in `.slgb` headers and token buffers (`Gemma4Tokenizer` omission caused `WT2_test[0]` to collapse from `4.6484 -> 11.1014` nats in Form A and `4.6075 -> 11.2901` in Form B).
   - Quantizes all `276` live 2D text linears (`1,055,711,232 B` @ `4.5 bpw g=32`), physically omits **`60` dead `L15..34` `(k_proj, v_proj, k_norm)` tensors** (`18,880,512` params), packs `262` FP32 norm/scalar vectors (`285,987` params) + IEEE CRC-32 verified `.slgb` headers (`2,094,062,080 B`), allocates **`0 B` dynamic VRAM/heap** over `1,000` decode steps (`2,744.94 MiB` = **`2.68 GiB`** @ `128K` ctx), and overlaps `17,920 B/tok` pinned host-DMA PLE with **`+0.01%` stall**.
3. **Hoisted `256 B` Woodbury & `L10..L14` `20x` `shared_KV` Hub Steering:**
   - Single-update `c_wb_inv_256b` constant-memory `8x8` Cayley–Woodbury steering preserves row norms (`3.67e-9` drift) at `+13.11%` AB/BA overhead (`D=1536`); steering `L10..L14` added `+0.19%` end-to-end latency (`11.180 -> 11.201 ms`), `1.308e-9` CUDA norm drift, `WT2_test -0.0380` nats, and `0` KV writes in `L15..34`.
4. **`KG-GFB4 PASS` — `0 B`-KV `L25..L33` `K=2` (`44L`) Recurrent Loop (`T_{Inkling2 -> Gemma4}`):**
   - Looping `L25..L33` with `2,695,680` (`5.14 MiB`) `r=16` `BF16` `(q, mlp, ple)` controllers reserves **`0 B` extra KV cache** (`K=1` zero-init diff = `0.0`), contracts with **`rho_hat = 0.1062` CUDA / `0.1228` Torch** (`<< 0.6180`), and lifts held-out accuracy across all 3 benchmarks (`K=1 -> K=2`) in `Track S184`'s calibrated Torch/CUDA evaluation (`docs/STAGE1_VERDICT.json`; note: the shipped standalone `.slgb` release initializes the `U-4` slot to `K=1` identity `wf=0, wg=0, mq=0` — mathematically `h_out = h_in` — and the C++ bench executes segment re-execution / WT2 timing without an in-binary CD/GSM eval loop):
     - **`WT2_test` (`16x256`):** Torch `5.410329 -> 5.202459` (`-0.207870` nats); CUDA `5.412680 -> 5.205190`
     - **`Countdown_test` (`128`):** `1.376065 -> 1.195397` (**`+0.180667` nats**, 95% CI `[+0.1638, +0.1973]`, `p=9.999e-5`)
     - **`GSM8K_OOD` (`64`):** `1.779174 -> 1.620899` (**`+0.158275` nats**, 95% CI `[+0.1399, +0.1774]`, `p=9.999e-5`)

---

## Verified RTX 3080 (`sm_86`) Scorecard & Honest Law-4 Scope

All GPU measurements were run on an `NVIDIA GeForce RTX 3080 Laptop GPU (8,192 MiB, sm_86)` with **`energy_power_status = UNCONTROLLED_LAPTOP_NOT_CLAIMED`**.

| Batch | Depth (`K`) | `lm_head` Mode | Median (ms) | p95 (ms) | Throughput (`tok/s`) |
|---:|---:|---|---:|---:|---:|
| `B=1` | `K=1` (`35L`) | `4.5bpw_shadow` | 11.173 | 11.184 | **89.50** |
| `B=1` | `K=1` (`35L`) | `bf16_tied` | 12.729 | 12.789 | **78.56** |
| `B=1` | `K=2` (`44L`) | `4.5bpw_shadow` | 17.105 | 17.119 | **58.46** |
| `B=1` | `K=2` (`44L`) | `bf16_tied` | 18.656 | 18.669 | **53.60** |
| `B=1` | `K=3` (`53L`) | `4.5bpw_shadow` | 23.036 | 23.053 | **43.41** |
| `B=1` | `K=3` (`53L`) | `bf16_tied` | 24.588 | 24.603 | **40.67** |
| `B=16` | `K=1` (`35L`) | `4.5bpw_shadow` | 70.855 | 71.761 | **225.81** |
| `B=16` | `K=1` (`35L`) | `bf16_tied` | 76.040 | 76.618 | **210.41** |
| `B=16` | `K=2` (`44L`) | `4.5bpw_shadow` | 91.279 | 91.691 | **175.29** |
| `B=16` | `K=2` (`44L`) | `bf16_tied` | 96.270 | 96.536 | **166.20** |
| `B=16` | `K=3` (`53L`) | `4.5bpw_shadow` | 110.836 | 111.828 | **144.36** |
| `B=16` | `K=3` (`53L`) | `bf16_tied` | 116.292 | 116.563 | **137.59** |

- **Honest Stage-1 `KG-GFB1` / `KG-GFB3` Disposition (`overall_disposition = KILL_FALSIFIED`):**
  - `KG-GFB2` (`PASS`) and `KG-GFB4` (`PASS`) passed cleanly.
  - `KG-GFB1` logged `KILL_FALSIFIED` because Stage-1 8-bit `dp4a` activation quantization + `g=32` Form B vs `g=128` Form A `G0_Base_A16` differed by `max_abs=0.8598` (`cos=0.99966`) vs PyTorch Form B and `5.1771` (`cos=0.98820`) vs Form A `G0` (exceeding the `<= 1e-3` gate).
  - `KG-GFB3` logged `KILL_FALSIFIED` because Stage-1 scalar-load `dp4a` GEMV hit `89.50 tok/s` (`B=1`, `K=1`) and `58.46 tok/s` (`B=1`, `K=2`) vs the preregistered `>= 190 / >= 155 tok/s` roofline floor.
  - Both targets are isolated to Stage-1 `8`-bit activation quantization and unvectorized GEMV loads; multi-limb exact `dp4a` + `uint4` coalesced GEMV + `mma.m16n8k16` Stage-2 optimization is actively tracked in follow-up work.

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

## Verified Inventory & Errata vs. Immutable Track S184 Docs

The files in `docs/` (`STAGE1_VERDICT.json`, `STAGE1_REPORT.md`, `PREREG_W22_GEMMA4_FORMB.json`, `ARCHITECTURE_FORMB_SILIGEMMA.md` SHA-256 `b7410d0c...`) and `kernels/siligemma_kernels_sm86.cu` (`5ee2e547...`) are preserved byte-identical to the executed **Track S184** RTX 3080 run (`EXPECTED_PINNED_SHA256` in `tests/test_siligemma_cpu.py`). Post-Stage-1 hardening and doc errata are recorded transparently below:

| Item | Historical S184 Doc Value | Verified Code / Artifact Value & Clarification |
|---|---|---|
| **FP32 Norm & Scalar Tensor Count** | `247` (`ARCHITECTURE` Table 2.1) | **`262`** tensors (`15 x 8 + 20 x 7 + 2 = 262`; `285,987` FP32 params already included the `15` `k_norm` vectors). |
| **Checkpoint Text Param Total** | `4,666,330,403` (`ARCHITECTURE` Table 2.1) | **`4,647,449,891`** (`4,628,569,379` active + `18,880,512` dead KV; `4,666,330,403` double-added dead KV). |
| **Partition 7 & `kArena128K` Sizes** | `19.34 MiB` (`ARCHITECTURE` §3.1) | **`19.28 MiB`** (`20,217,856 B`); `kArena128K` (`2,700,007,680 B`) total VRAM = **`2,744.94 MiB` (`2.68 GiB`)**. |
| **`[16, 11, 4]` SECDED Protection Scope** | "`3/3` SEC / `3/3` DED / `0` stale logits" | Protects the **`A_16` LSB parity plane (`0x11111111`)** (`run_selftest_gpu` verified injected LSB SEC/DED + IRIS); release hardening added `siligemma_iris_canary_finalize_sm86` + host `evaluate_canary_watchdog` + `.slgb` IEEE CRC-32 + token/position bounds + PLE header verification in `siligemma_arena.hpp`/`siligemma_bench_sm86.cu` (`nvcc` `sm_86` zero-spill/`<= 48`-reg compile & C++20 `make test-cpu` verified; host harness changes not re-executed on RTX 3080). |
| **Form A Cross-Track Reference Values** | `S182b2` `1,203,561,445 B`, `S184 G0` `5.118627` | Pruned `S182b4` `H1` `model.safetensors` linears = **`1,203,427,606 B`**; `S182b3/b4 H0` `WT2 = 5.116801` vs `S184 G0` `5.118627`. |
| **EOS Tokens & Bench Mode** | `{1, 106}` in `.slgb` header | `generation_config.json` includes `[1, 106, 50]` (turn stop `50`); `.slgb` stores `{1, 106}` and `siligemma_bench_sm86` runs fixed-length timing/WT2 eval without an EOS loop. |
