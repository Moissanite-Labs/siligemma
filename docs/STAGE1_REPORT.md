# Track S184 (`EXP-Q81`) Stage-1 Report: SiliGemma-Native `sm_86` Form B Live RTX 3080 Verification

- **Overall Stage-1 Verdict**: `KILL_FALSIFIED`
- **Target Hardware / Energy**: `NVIDIA GeForce RTX 3080 Laptop GPU (8192 MiB, sm_86)`, `UNCONTROLLED_LAPTOP_NOT_CLAIMED`
- **Zero-Anchor Compliance**: bare-metal `siligemma_bench_sm86` links `linux-vdso.so.1, libstdc++.so.6, libm.so.6, libgcc_s.so.1, libc.so.6, /lib64/ld-linux-x86-64.so.2` (`0` Python/PyTorch/transformers/cuBLAS/Form-A CSR anchors); `nvcc --Werror all-warnings` compiled `20` kernels (`max_registers=48`, `0 B` stack/spill, `0` LDL/`0` STL)
- **Governance**: `upload_authorized=False`, `weights_stored_in_expedition=0`, `ELF=0`, `$HOME`-only paths, `no_claude_on_rl_or_inference=True`

## `KG-GFB1`: Zero-Anchor Wire Arena, `bos=2`, Async PLE DMA, & Logit Parity

- **Verdict**: `KILL_FALSIFIED`
- **Static Arena & Zero-Alloc (`1,000` decode steps + `K=2,3`)**: `276` live linears (`60` dead KV omitted), `wire_bytes=1055711232` B (`4.5 bpw g=32`), `kArena128K=2700007680` B (`2744.94 MiB <= 3072 MiB`), `alloc_vram_delta=0 B`, `alloc_heap_delta=0 B`
- **Async Double-Buffered PLE DMA (`17,920 B/tok`, `4.698 GB` host `cudaHostAlloc` pinned=True)**: pre-resident `11.1712 ms` vs overlapped async DMA `11.1729 ms` -> **DMA stall = `+0.01%`** (cap `<= +2.0%`)
- **`bos=2` Sanity (`WT2_test[0]`)**: `no_bos=11.290077` -> `with_bos=2=4.607528` nats
- **Logit & NLL Parity (Real `google/gemma-4-E2B-it`)**:
  - Bare-metal CUDA Form B (`BF16 lm_head`) vs PyTorch Form B (`BF16 lm_head`): `max_abs=8.5982e-01`, `cos=0.999657`, `argmax_match=True`
  - Bare-metal CUDA Form B (`4.5bpw lm_head`) vs PyTorch Form B (`4.5bpw lm_head`): `max_abs=9.0006e-01`, `cos=0.999658`, `argmax_match=True`
  - Bare-metal CUDA Form B (`BF16 lm_head`) vs **`S182b2` Form A `G0_Base_A16`** (`g=128 + CSR`): `max_abs=5.1771e+00`, `cos=0.988201` (prereg cap `<= 1.0e-3`: `False`)
  - Bare-metal CUDA `WT2_test` NLL (`B=16`): `K=1 BF16_lm=5.412680`, `K=1 4.5bpw_lm=5.423810` vs Torch Form B `5.410329` & Form A `G0_Base_A16` `5.118627`

## `KG-GFB2`: `sm_86` Kernel Hygiene, `0`-ULP Tree, `IRIS+SECDED`, & Woodbury Hoist Overhead

- **Verdict**: `PASS`
- **`tree_reduce_128_sm86` (`D in {256,512,1536,6144,12288}`, `B in {1,16}`)**: `max_ulp=0`, fused `dp4a` GEMV max abs diff vs CPU FP32 ref = `9.9021e-02`
- **Live `IRIS + [16,11,4]` `SECDED`**: SEC repaired=`3/3` (`err=0.000e+00`), DED SafeHalt traps=`3/3` (`watchdog=True`), IRIS Trojan catch=`1/1` (`watchdog=True`), uninjected runtime DED=`0`
- **Live Hoisted `256 B` `c_wb_inv_256b` Woodbury (`10,000` vecs + `4` corners)**: `max_rel_norm_drift=3.672e-09`, AB/BA overhead (`D=1536`): Euclidean `65.13 us` vs Cayley-Woodbury `73.67 us` -> **`+13.11%`** (cap `<= +25.0%`)

## `KG-GFB3`: Live RTX 3080 CUDA Graph Throughput (`K=1,2,3`) & `L10..L14` Hub Steering

- **Verdict**: `KILL_FALSIFIED`

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

- **`L10..L14` `20x` Cayley-Woodbury Hub Steering**: CUDA 5-slot `norm_drift=1.308e-09`, end-to-end `B=1 K=1` hub latency `11.180 -> 11.201 ms` (`+0.19%`), PyTorch `row_norm_drift=2.058e-07`, `WT2_test drift=-0.038026` nats (`<= +0.03`), `kv_writes_l15_34=0`

## `KG-GFB4`: `U-4` `0 B`-KV `L25..L33` `K=1,2,3` Contraction & Accuracy Probe

- **Verdict**: `PASS`
- **Controller & KV Growth**: `2695680` `r=16` `BF16` `(q, mlp, ple)` params (`5391360` B), `extra_kv_bytes=0 B`, `K=1` zero-init diff vs `35L` = `0.000e+00`
- **Contraction & Trajectory Geometry**: Torch `K=3 rho_hat=0.1228`, `K=2 step_ratio=0.1474`, `exit_cos=0.9953`; Bare-metal CUDA `WT2` `K=3 rho_hat=0.1062`, `K=2 step_ratio=0.1797`, `exit_cos=0.9929`
- **Disjoint Held-Out NLLs (`K=1 -> K=2`)**:
  - `WT2_test` (`16x256`): Torch `5.410329 -> 5.202459` (K3: `5.199883`, drift `-0.207870` nats); Bare-metal CUDA `K=1=5.412680 -> K=2=5.205190 -> K=3=5.217070`
  - `Countdown_test` (`128` items): `1.376065 -> 1.195397` (`gain=+0.180667` nats, 95% CI `[+0.163776, +0.197342]`, `p=9.999e-05`)
  - `GSM8K_OOD` (`64` items): `1.779174 -> 1.620899` (`gain=+0.158275` nats, 95% CI `[+0.139919, +0.177410]`, `p=9.999e-05`)
