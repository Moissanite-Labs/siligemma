# Track S184 (`EXP-Q81`): End-to-End Bare-Metal `Gemma-4` Form B Architecture (`SiliGemma-Native sm_86`)

- **Track / Queue / Preregistration:** `TRACK-S184` / `EXP-Q81` / [`PREREG_W22_GEMMA4_FORMB.json`]($HOME/Projects/expedition/track-s184-gemma4-formb-native/PREREG_W22_GEMMA4_FORMB.json)
- **Target Model Snapshot:** `google/gemma-4-E2B-it` (HF snapshot `4adf15748de9e1a0097b6274cee9c15836c41c37`, `config.json` SHA-256 `1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330`)
- **Governance & Execution Scope:** Strictly CPU-only architectural & analytical design (`CUDA_VISIBLE_DEVICES=""`, `< 400 MiB` RAM, `0` model weight loads, `WEIGHTS=0`, `ELF=0`, `$HOME`-only paths, `NO-CLAUDE`, `UNCONTROLLED_LAPTOP_NOT_CLAIMED`). Blueprint targets bare-metal **C++20 / CUDA `sm_86`** (`NVIDIA GeForce RTX 3080 Laptop GPU`, `8,192 MiB` VRAM, `6.5 GiB = 6,979,321,856 B` usable cap, `356.68 GB/s` STREAM Triad, `16 GB/s` PCIe Gen4 pinned H2D DMA) dropping **100% of `transformers`, `PEFT`, PyTorch `nn.Module`, `cuBLAS`, and FP16/BF16 Form A CSR anchors**.

---

## 1. Upstream Pinned Verification Anchors (`SHA-256`)

| Anchor Component | Artifact (`$HOME`-Rooted Path) | Verified SHA-256 | Core Contribution to `Form B` |
|---|---|---|---|
| **S180c Hoisted `sm_86` Report** | [`STAGE1_REPORT.md`]($HOME/Projects/expedition/track-s180c-hoist-optimization-s1/STAGE1_REPORT.md) | `db4695698add0675d6791f69907fa8364c4b72bb6d259a193962f89c6bbdd216` | `25/25` zero-spill kernels (`44`/`48` regs), `1,894.91 tok/s` `6L` graph |
| **S180c `sm_86` CUDA Kernels** | [`s180c_hoist_bench.cu`]($HOME/Projects/expedition/track-s180c-hoist-optimization-s1/s180c_hoist_bench.cu) | `b08477c047d2efd91a8ee8741ecba6aaa21d5c980b3e81258cc47b9bf2fd8c41` | `c_wb_inv_256b`, `u[4]`/`x_rot` Q8 hoisting, fused `dp4a` GEMV |
| **S180 `HoloSphere` Kernel Base** | [`silinative_a16_lora4_sm86.cu`]($HOME/Projects/expedition/track-s180-silinative-holosphere-s1/silinative_a16_lora4_sm86.cu) | `a62ea47a83efc024afcc00340360de2382bc5381dbca1a4e6b9d216f6888ef12` | `HoloSphereRazorCanary64`, `WatchdogAction`, `1-LOP3+5-POPC` SECDED |
| **Canonical Tree Header (`P223c`)** | [`canonical_hoist_sm86.cuh`]($HOME/Projects/publish/siliqwen/kernels/canonical_hoist_sm86.cuh) | `8f72138ab4990edc98bd1300acde0456e583676b9c913b89478b4b4de2cd9a85` | `tree_reduce_128_sm86` (`0`-ULP `7`-stage binary reduction) |
| **S182 Functor (`Qwen->Gemma4`)** | [`QWEN_TO_GEMMA4_FUNCTOR_FULL.md`]($HOME/Projects/expedition/track-s182-gemma4-siliquant-s1/QWEN_TO_GEMMA4_FUNCTOR_FULL.md) | `6fd701588a7b6e9f8b6f35f20e878605eda6707ea87d7a83e85e3d57a5e1bf9b` | `0` needles (`cos_max_r4 <= 0.2367`), `L13/14` `20x` KV wall, `L25..33` valley |
| **S182b2 `bos=2` Form A Report** | [`STAGE1_REPORT.md`]($HOME/Projects/expedition/track-s182b2-gemma4-bos-fix/STAGE1_REPORT.md) | `7ec9c1299ad5bcf0c017325f046575b016bf586b60ecc0ea696e0a6302f3cec7` | `bos=2` fix (`11.1014 -> 4.6484`), `G0` Form A `2772.0 MiB`, `0` SECDED |
| **S182d Closed 3-Way Loop** | [`DESIGN_S182D_SILILOOP.md`]($HOME/Projects/expedition/track-s182d-siliqwen-loop/DESIGN_S182D_SILILOOP.md) | `8c9e0b60e94490a28e371f1ee3a4e1644106d39ad35b96badd2a0df3c2e2fbd8` | `U-4` `T_{Inkling2->Gemma4}` (`L25..33` `0 B` KV loop, `2.70M` `r=16` gates) |
| **S183 Roofline Constants** | [`verify_s183_roofline.py`]($HOME/Projects/expedition/track-s183-inkling2-mini-s1/verify_s183_roofline.py) | `c81f748286a262d397cf5141294bed2218daba88b085d61a82fcd735f4168dfd` | RTX 3080 (`356.68 GB/s`, `16 GB/s` PCIe Gen4, `6.5 GiB` cap) |
| **P223/P224 Unlocks Synthesis** | [`FORWARD_UNLOCKS_P223_P224_INKLING_SILIQWEN.md`]($HOME/Projects/expedition/scratch/FORWARD_UNLOCKS_P223_P224_INKLING_SILIQWEN.md) | `1ee3463abe95199fbf54a246306bb103a3abbcc075f7dfa4595cea825bcf5f04` | `zed` static frame collapse, `P224a` IRIS CRC, `P224c` `RiccatiDecoupling` |
| **Gemma-4 `config.json`** | [`config.json`]($HOME/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/4adf15748de9e1a0097b6274cee9c15836c41c37/config.json) | `1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330` | `35L`, `H=1536`, `bos=2`, `eos=[1,106]`, `softcap=30.0` |

---

## 2. Exact `Gemma-4-E2B-it` Topology & Wire-Format Surgery (Zero Anchors)

Form B compiles `google/gemma-4-E2B-it` directly into a flat `.slgb` (`SiliGemma` Bare-Metal) memory image, eliminating all Python/PyTorch objects, all `siliquant-hf-v1` CSR indirection (`qquot + qham11 + qov_*`, which consumed `1,203,561,445 B = 1,147.80 MiB` in Form A `S182b2`), and all structurally dead checkpoint weights.

### 2.1 Verified Parameter, Linear, and Pruned-Dead-Tensor Inventory

Derived directly from `text_config` (`config.json`, `H=1536`, `L=35`, `num_kv_shared_layers=20`, `num_attention_heads=8`, `num_key_value_heads=1`, `head_dim=256`, `global_head_dim=512`, `sliding_window=512`, `intermediate_size=6144` on `L0..14` and `12288` on `L15..34`, `hidden_size_per_layer_input=256`, `vocab_size=262144`):

| Tensor Group | Tensor Count | Exact Param Count | Form B Disposition & Exact Byte Size |
|---|---:|---:|---|
| **Live 2D Linears (`276/276`: `K%128==0, M%16==0`)** | `276` | `1,876,819,968` | **GPU Static Arena:** `1,055,711,232 B` (`1,006.80 MiB`, `g=32`, `4.5 bpw`) or `967,735,296 B` (`922.90 MiB`, `g=128`, `4.125 bpw`) |
| — Breakdown: `L0..34` Attn (`Q,O` x `35` + `K,V` x `15`) | `100` | `278,396,928` | `4.5 bpw` `A_16-BlockH4` packed `uint64_t` + FP16 scale |
| — Breakdown: `L0..34` GeGLU MLP (`gate,up,down` x `35`) | `105` | `1,557,135,360` | `4.5 bpw` `A_16-BlockH4` (`6144` on `L0..14`, `12288` on `L15..34`) |
| — Breakdown: PLE Projections (`70` in-layer + `1` global) | `71` | `41,287,680` | `2 * 1536 * 256` x `35` (`27.53M`) + `per_layer_model_projection` (`13.76M`) |
| **Pruned Dead KV Tensors (`L15..34`)** | **`60` (`0` stored)** | `18,880,512` (`0` stored) | **Physically absent from wire format:** `20 x (k_proj, v_proj, k_norm)` omitted (`v_norm` has `with_scale=False`) |
| **Norms & Layer Scalars (`FP32` canonical)** | `247` vectors/scalars | `285,987` | **GPU Static Arena:** `1,143,948 B` (`1.0910 MiB` `FP32` — `5*1536 + q_norm + opt k_norm + c_l` per layer + global norms) |
| **Tied `embed_tokens` / `lm_head`** | `1` (tied) | `402,653,184` | **GPU Static Arena:** `805,306,368 B` (`768.00 MiB` `BF16` exact tying; optional `4.5 bpw` `lm_head` fast-read shadow `226,492,416 B = 216.00 MiB`) |
| **`embed_tokens_per_layer` (PLE Table)** | `1` | `2,348,810,240` | **CPU Page-Locked Pinned RAM (`cudaHostAlloc`):** `4,697,620,480 B` (`4,480.00 MiB = 4.3750 GiB` `BF16`); **`17,920 B/tok` (`17.50 KiB/tok`) async DMA** |
| **Check Total (`excl. tied lm_head`)** | `276` live + norms + tables | `4,647,449,891` | With `60` dead checkpoint tensors: `4,666,330,403` (+ `embed_tokens` untied alias = HF `5.123B`) |

### 2.2 Fail-Closed Header, Tokenizer (`bos=2`), and Wire Encoding Rules

1. **Tokenizer & Binary Header Gate (`bos=2` Fail-Closed):**
   - `S182b2` proved omitting `bos_token_id = 2` collapses Gemma-4 `WT2` (`11.101388 -> 4.648381` nats, `-6.4530`) and `Countdown` (`2.376753 -> 1.325589` nats, `-1.0512`).
   - Every `.slgb` binary header encodes `uint32_t magic = 0x534C4734` (`"SLG4"`), `bos_token_id = 2u`, `eos_token_ids[2] = {1u, 106u}`, `pad_token_id = 0u`, `vocab_size = 262144u`, `num_layers = 35u`, `num_kv_shared_layers = 20u`, `final_logit_softcapping = 30.0f`, and `rms_norm_eps = 1e-6f`.
   - The C++20 runtime `SiliGemmaEngine::decode_step()` refuses execution (`WatchdogAction::SAFE_HALT_DED_ROLLBACK` / error exit `-1`) if `token_buf[0] != 2u` on fresh context creation or if any `L15..34` layer descriptor carries non-null `k_proj`/`v_proj`/`k_norm` offsets.
2. **Exact `A_16-BlockH4` Wire Layout (`4.5 bpw` Baseline / `4.125 bpw` High-Density):**
   - **Baseline (`g=32`, `4.5 bpw = 18 B / 32 weights`):** Two `uint64_t` `[16,11,4]` Plotkin RM(2,4)-protected nibble words (`16 B`, `16` nibbles/word, `64`-bit alignment matching `a16_secded_lop3_popc_syndrome`) + one `__half` scale (`2 B`), packed in pure Structure-of-Arrays (`uint64_t words[]` contiguous, followed by `uint16_t scales[]` contiguous) so weight loads coalesce across `128 B` cache lines with **`0` unaligned reads**. Exact byte budget for `276` linears: `1,876,819,968 * 18 / 32 = 1,055,711,232 B` (`1,006.80 MiB`).
   - **Alternative (`g=128`, `4.125 bpw = 66 B / 128 weights` matching `S182` `R^2=5213`):** Eight `uint64_t` SECDED words (`64 B`) + one `__half` scale (`2 B`) -> `967,735,296 B` (`922.90 MiB`).
3. **Exact Gemma-4 Forward Math Invariants:**
   - **Attention Scaling:** Per `modeling_gemma4.py`, `Gemma4TextAttention` applies `scaling = 1.0` (no $1/\sqrt{d_h}$), `Q <- q_norm(W_q x)`, `K <- k_norm(W_k x)` (only `L0..14`), and **parameter-free `v_norm`** (`V_hat = V / sqrt(mean(V^2) + 1e-6)`, `with_scale=False`).
   - **Sliding vs. Proportional RoPE:** Sliding layers (`L % 5 != 4`, `d_h=256`) apply standard RoPE (`rope_theta = 10000.0`, `128` pairs); full-attention layers (`L in {4,9,14,19,24,29,34}`, `d_h=512`) apply **`proportional` RoPE** (`modeling_rope_utils.py:194-268`: `partial_rotary_factor = 0.25`, `rope_theta = 1000000.0` -> `rope_angles = int(0.25 * 512 // 2) = 64` rotated pairs = `128` rotated dims, concatenated with `nope_angles = 192` zero frequencies = `384` pass-through dims).
   - **Softcap Exit:** `logits = 30.0f * tanhf((W_emb @ h_norm) / 30.0f)`.

---

## 3. Zero-Alloc Static CUDA Graph Arena, Async PLE DMA, & `zed` KV Ring Buffer

`SiliGemma` pre-allocates **one contiguous `cudaMalloc` device arena** (`cuda_free_vram_delta_bytes == 0` during warmup/decode/looping, matching `S180c` `KG-H2`'s `0 B` delta), executing the entire token step inside a single captured **CUDA Graph** (or `K in {1,2,3}` pre-captured static graph variants).

### 3.1 `SiliGemmaArenaLayout` Static Byte Budget (`~2.75 GiB` Envelope)

| Arena Partition | Alignment | `16K` Compact (`T=16384`) | `64K` Target + `4.5bpw` `lm_head` (`T=65536`) | `128K` Max (`T=131072`) | Dynamic Alloc |
|---|---:|---:|---:|---:|---:|
| **1. `276` Live `A_16-BlockH4` Linears (`g=32`)** | `256 B` | `1,006.80 MiB` | `1,006.80 MiB` | `1,006.80 MiB` | `0 B` |
| **2. Tied `embed_tokens` (`BF16`) [+ Fast `4.5bpw` `lm_head`]** | `256 B` | `768.00 MiB` (`+0`) | `768.00 MiB` (`+216.00 MiB`) | `768.00 MiB` (`+0`) | `0 B` |
| **3. Norms & `layer_scalar` (`FP32`, `285,987` params)** | `256 B` | `1.09 MiB` | `1.09 MiB` | `1.09 MiB` | `0 B` |
| **4. `U-4` `r=16` `BF16` Loop Controllers (`2,695,680` params)** | `256 B` | `5.14 MiB` | `5.14 MiB` | `5.14 MiB` | `0 B` |
| **5. `zed` Static KV Ring (`L0..14` only; `L15..34` = `0 B`)** | `4096 B` | `102.00 MiB` | `390.00 MiB` | `774.00 MiB` | **`0 B`** |
| **6. Double-Buffered PLE Staging (`2 x 16 x 8960 x 2B`)** | `256 B` | `0.55 MiB` | `0.55 MiB` | `0.55 MiB` | `0 B` |
| **7. Activations, `u[4]`/`x_q8`, `U-4` Frames, & Logits (`B=16`)** | `256 B` | `19.34 MiB` | `19.34 MiB` | `19.34 MiB` | `0 B` |
| **8. `HoloSphereRazorCanary64` + Telemetry Slab** | `64 B` | `0.06 MiB` | `0.06 MiB` | `0.06 MiB` | `0 B` |
| **Total GPU Tensor Arena (`cudaMalloc` once)** | — | **`1,902.93 MiB` (`1.858 GiB`)** | **`2,406.93 MiB` (`2.351 GiB`)** | **`2,574.93 MiB` (`2.515 GiB`)** | **`0 B`** |
| **Total RTX 3080 VRAM incl. CUDA Driver/Graph Headroom (`~400–420 MiB`)** | — | **`~2.25 GiB`** | **`~2.76 GiB` (Target `~2.75–2.8 GiB`)** | **`~2.91 GiB`** | `<< 6.5 GiB` cap |

*High-density `g=128` (`4.125 bpw`) reduces Partition 1 by `-83.90 MiB` (`922.90 MiB`), dropping `16K` arena tensors to `1,819.03 MiB` (`~2.17 GiB` total VRAM).*

### 3.2 Async Double-Buffered Pinned Host-DMA (`embed_tokens_per_layer`)

1. **Pinned Host Residency:** `embed_tokens_per_layer` (`262,144 x 8,960` `BF16` = `4,697,620,480 B = 4.3750 GiB`, well inside RTX 3080 host `25.0 GB` budget) resides in page-locked host memory (`cudaHostAlloc`).
2. **Double-Buffered Ping-Pong DMA Overlap (`stream_dma` vs. `stream_compute`):**
   - For token $t$, embedding lookup `embed_tokens[id_t]` (`3,072 B`) executes on GPU while host gathers `raw_ple[t]` (`35 x 256 x 2 B = 17,920 B = 17.50 KiB`, i.e., `~18 KiB/tok`; `512 B` per layer slice) into pinned staging slot `buf[t & 1]`.
   - `cudaMemcpyAsync(d_ple_stage[t & 1], h_ple_pin[t & 1], 17920 * B, cudaMemcpyHostToDevice, stream_dma)` fires in parallel with `L0` compute (`per_layer_model_projection(embed_t)` + `L0` Attention/MLP, plus `cudaEventRecord(ev_ple_ready)`).
   - **Exact Overlap Proof:** At verified `S137`/`S183` PCIe Gen4 H2D DMA (`16.0 GB/s`; conservative `12.0 GB/s`), transferring `17,920 B` (`B=1`) takes **`1.120 us`** (`1.493 us` conservative) — or **`17.92 us`** at `B=16` (`286,720 B`) — while `L0` Attention + MLP execution (`~85 us` at `B=1` on `35L`) and global `per_layer_model_projection` (`[1, 1536] -> [35, 256]`) run cleanly ahead of `L0`'s terminal PLE branch (`stream_compute` calls `cudaStreamWaitEvent(ev_ple_ready)` right before `L0` `per_layer_input_gate`). Result: **`~0` host-DMA pipeline stall** (`<= 1.5%` jitter tolerance).

### 3.3 `zed` Static `[0..12]` / `[13..34]` Shared-KV Ring Buffer (`0` Dynamic KV Allocation)

Following `P223c` (`zed`) static acyclic lifetime elimination:
- **Physical Allocator Elimination:** Only `L0..L14` store KV pairs (`1` KV head, `BF16`):
  - `12` sliding producers (`L0..3, L5..8, L10..13`, `d_h=256`): fixed `512`-slot circular ring per layer = `512 * (2 * 1 * 256 * 2 B) = 524,288 B = 512 KiB/layer` (`6.00 MiB` total regardless of $T_{\max} \ge 512$).
  - `3` full-attention producers (`L4, L9, L14`, `d_h=512`): static ring of `T_max` slots = `T_max * (2 * 1 * 512 * 2 B) = 2,048 B/tok/layer` (`30.00 MiB` total KV at `T_max=4096`, `102.00 MiB` at `T_max=16384`, `390.00 MiB` at `T_max=65536`, `774.00 MiB` at `T_max=131072`).
- **`[13..34]` Read-Only Lifetime:** Every upper layer `L15..34` (`16` sliding + `4` full `L19,24,29,34`) reads `L13` or `L14` ring slots via compile-time pointer aliases (`const __nv_bfloat16* kv_ring_ptr[l] = (l >= 15) ? (is_full(l) ? arena.kv_L14 : arena.kv_L13) : arena.kv[l]`). **`L15..34` and all `K=2,3` loop iterations over `L25..33` reserve `0 B` of extra KV cache**.

---

## 4. Fused `sm_86` `0 B`-Spill Kernels, `P223c` Canonical Trees, & `P224a` IRIS `Canary64`

Every kernel compiles under `nvcc -O3 -std=c++20 -arch=sm_86 --ptxas-options=-v --Werror all-warnings` with strict **`0 B` stack / `0 B` spill loads / `0 B` spill stores / `0` `LDL` / `0` `STL`**, bounded directly by `S180c`'s `25/25` zero-spill receipt (`44`/`48` regs for hoisted `dp4a`/`fp32` `A_16-BlockH4` GEMV).

### 4.1 Register, Shared-Memory, & Cycle Budget Table (`sm_86`, RTX 3080)

| Kernel / Device Primitive | Source Heritage | Target Regs | Max SMEM | Stack / Spill | Key Algorithmic / Hardware Invariant |
|---|---|---:|---:|---:|---|
| `siligemma_precompute_woodbury_inv8_sm86` | `S180c` (`35` reg, `512 B`) | `<= 36` | `512 B` | `0 B` | Computes $M^{-1}_{8\times 8}$ once/update into **`__constant__ float c_wb_inv_256b[64]` (`256 B`)** |
| `siligemma_sandwich_norm_u4_q8_sm86` | `S180c` `preq8_multi_u4` (`47` reg) + `resid_rmsnorm` (`44` reg) | **`<= 48`** | `512 B` | `0 B` | Fused `RMSNorm + layer_scalar + h4_rot_f4` + symmetric Q8 quantization + `u[4]` Cayley precompute |
| `siligemma_a16_h4_dp4a_gemv_sm86` | `S180c` `hoisted_u4_dp4a` (`44` reg) / `preq8_u4_dp4a` (`48` reg) | **`<= 48`** | `1,152 B` | `0 B` | Coalesced `uint64_t` SoA fetch, `a16_secded_lop3_popc_syndrome`, `dp4a.s32.s32`, `tree_reduce_128_sm86` |
| `siligemma_fused_sandwich_h4_dp4a_gemv_sm86` | `SiliGemma` Single-Pass Norm-to-GEMV (`4`-warp CTA, `128` thd) | **`<= 48`** | `1,664 B` | `0 B` | Inline `Sandwich-RMSNorm -> H_4 (h4_rot_f4) -> Q8 -> A_16 dp4a -> tree_reduce_128_sm86` |
| `siligemma_cayley_woodbury_r4_hoisted_sm86` | `S180c` (`56` reg, `160 B`, **`-19.57%`/`+3.44%`** vs Euclidean) | `<= 56` | `160 B` | `0 B` | Norm-preserving $S^3$/Cayley $r=4$ steering via `c_wb_inv_256b` (`2.368 us` `B=16`/`D=512`, `5.296 us` `G`/`D=1536`) |
| `siligemma_fused_geglu_ple_sandwich_sm86` | `S180c` `gate_up_swiglu` (`59` reg) adapted to Gemma-4 `GeGLU` | `<= 59` | `1,152 B` | `0 B` | `gelu_pytorch_tanh` + `post_feedforward_layernorm` + PLE residual + `layer_scalar` $c_\ell$ |
| `siligemma_iris_canary_finalize_sm86` | `S180` `razor_canary_emit_sys` + `P224a` IRIS | `<= 24` | `64 B` | `0 B` | Folds per-layer `Q` CRC32C/0x9E3779B9 digest into `HoloSphereRazorCanary64` + atomic release |

- **Occupancy & Cycle Verification (`<= 48` regs):** On `sm_86` (`65,536` registers/SM, `1,536` max threads/SM), `<= 48` registers with `128`-thread (`4`-warp) CTAs allows up to `10` CTAs (`1,280` threads/SM = `83.3%` warp occupancy) with **`0` local-memory traffic** — eliminating `S180`'s `100/107`-reg register cliff (`31.26%` speedup already observed in `S180c` `6L` graph: `1,443.55 -> 1,894.91 tok/s`).

### 4.2 Exact Canonical Binary Reduction Tree (`P223c`, `0`-ULP Invariant)

Every inner-product reduction in `siligemma_*` replaces non-associative warp/CTA atomic summation with `canonical_hoist_sm86.cuh`'s `tree_reduce_128_sm86(float val, float* smem_128)`:
1. For `g=32` / `g=128` tiles across Gemma-4 hidden widths (`256`, `512`, `1536`, `6144`, `12288`), each CTA of `128` threads maintains fixed-stride thread-lane assignment `k_lane = tid`, zero-padding out-of-range lanes (`val = 0.0f` when `k >= K_rem`).
2. `smem_128[tid] = val; __syncthreads();` followed by the fixed `7`-stage pairwise binary tree (`stride = 64, 32, 16, 8, 4, 2, 1`), guaranteeing **`0` ULP divergence** across decode (`B=1`), batched decode (`B=16`), hoisted prefix checkpoints, and cross-validation against Form A/CPU reference trees.

### 4.3 `P224a` `IRIS` Runtime+Scan CRC + `[16,11,4]` `1-LOP3+5-POPC` `SECDED` `SafeHalt`

`SiliGemma` binds physical bit-error protection (`[16,11,4]` Hamming) with `P224a`'s stateful runtime+scan defense inside the unmodified **`64 B` `HoloSphereRazorCanary64`** ABI (`silinative_a16_lora4_sm86.cu:56-68`):
1. **Inline `[16,11,4]` SECDED Decode:** Every `uint64_t` weight word in the `dp4a` GEMV passes through `a16_secded_lop3_popc_syndrome(w64)` (`lop3.b32 ..., 0x28` + `5x __popc()`). Single-bit SEC triggers atomic increment on `sec_count_u32`; double-bit DED triggers `ded_count_u32`, forces **`WatchdogAction::SAFE_HALT_DED_ROLLBACK`**, and halts the token graph before output commit (`0` stale logits emitted).
2. **`IRIS` Stateful Runtime+Scan Accumulator:** Per `P224a` `SN-T2/SN-O2`, static single-word parity alone cannot detect semantic/state-level Trojan perturbations spanning groups. During each layer `l = 0..34` and loop pass `p`, each CTA XOR-folds quantized activation tile digests (`0x9E3779B9u` golden-ratio Weyl mix) into `last_syndrome_u32` (high 16 bits = rolling layer IRIS CRC, low 16 bits = SECDED syndrome) alongside boot contract `epoch_u32` (`boot_contract_median_ms = 0.5110 ms` verified in `S180c`), failing closed (`SAFE_HALT_DED_ROLLBACK`) if runtime CRC mismatches the deterministic golden trace on probe/verification passes (`EXP-P224a`).

### 4.4 Hoisted `256 B` Constant-Memory `8x8` Cayley–Woodbury Updates (`S180c`)

For norm-preserving `r=4` rank steering on any linear projection (`Q`, `O`, `down_proj`, `per_layer_projection`):
- Updating weights triggers a single precompute call (`siligemma_precompute_woodbury_inv8_sm86`) populating `__constant__ float c_wb_inv_256b[64]` (`256 B` = one `8x8` inverse matrix $M^{-1}$).
- Inference computes `u[4]` (`8` floats/CTA) via `precompute_u4_h4` and applies `siligemma_cayley_woodbury_r4_hoisted_sm86` directly from constant broadcast cache (`0` global `M^{-1}` reads per row), achieving `S180c`'s verified **`-19.57%` latency reduction at `B=16, D=512` (`2.368 us` vs `2.944 us` Euclidean LoRA)** and **`+3.44%` overhead at `D=1536` (`5.296 us` vs `5.120 us`)** with exact $S^3$/Cayley norm preservation (`max err <= 1.28e-7` FP32).

---

## 5. Architectural Surgery: `L10..L14` `20x` `shared_KV` Hub + `U-4` `Inkling-2 -> Gemma-4` `0 B`-KV Loop

`S182` measured Gemma-4's exact `35L` laminar trajectory (`gemma4_cpu_prep_measurements.json`):
- **Mid-Stack `20x` Shared-KV Hub (`L10..L14`):** `step_norm_rel` climbs across `L10=1.0382`, `L11=1.0136`, `L12=1.3445`, `L13=1.1969` (`turn=0.9412`, sliding `shared_KV` producer) to the internal `90°` compiler wall at **`L14=2.0466` (`turn=0.9978`, full `shared_KV` producer)** — feeding `20` upper consumer layers (`L15..L34`, i.e., `20x` fan-out).
- **Upper Ultra-Laminar Sanctuary (`L25..L33`):** Above `L24` (`0.7720`), residual turbulence drops monotonically through **`L25=0.3690` (`turn=0.0687`) -> `L29=0.2496` (`turn=0.0318`) -> `L30=0.1189` (`turn=0.0074`, CD `0.0073`) -> `L33=0.3188` (`turn=0.0388`)** before the terminal `L34` exit cliff (`2.5414`). `L30` is `5.6x–6x` calmer than Qwen2.5's best basin (`L12..18`, `turn=0.0412`).

### 5.1 Interface 1 — Direct Runtime Steering on `L10..L14` `20x` Norm-Clamped `shared_KV` Hub

Form B exposes `SiliGemmaHubSteerDesc` directly over `L10..L14` (`100`% separated from downstream `L15..34` read-only execution):
- **Norm-Clamped Target Surface:** Exposes Cayley–Woodbury `c_wb_inv_256b` `r=4` steering hooks (`u[4]` + `c_l` scalars) on `L10..L14` `{q_proj, k_proj, v_proj, o_proj, down_proj}` + continuous `q_norm` / `k_norm` / `4` sandwich `RMSNorm` gains (`v_norm` stays scale-free).
- **Invariance Guarantee (`20x` Fan-Out Safety):** Because `(q_norm, k_norm, v_norm)` project every `L13`/`L14` KV head back onto a fixed-radius hypersphere (`0` needles, `cos_max_r4 <= 0.2367`) and Cayley–Woodbury steering preserves row norms (`||W'_r|| == ||W_r||`), steering `L10..L14` cleanly redirects all `20` downstream consumers **before** locking `shared_KV_{13,14}` into the `zed` read-only lifetime (`L15..34` never write KV).

### 5.2 Interface 2 — `U-4` `T_{Inkling2 -> Gemma4}` Adaptive `K in {1,2,3}` Looping on `L25..L33` (`0 B` Extra KV)

Mapping `Inkling-2-Mini`'s `[6, 18) x 3` looped reasoning core onto Gemma-4's `L25..L33` (`9` physical layers: `L25..28, L30..33` sliding `d_h=256`, `L29` full `d_h=512`, `579,600,384` live linear params = `310.92 MiB` @ `4.5 bpw`, `285.01 MiB` @ `4.125 bpw`):
1. **Effective Depth & Zero Extra KV:** Executing `L0..24` once, looping `L25..L33` `K in {1,2,3}` times, and executing `L34` once yields **`35L` (`K=1`), `44L` (`K=2`), or `53L` (`K=3`) effective depth** inside the `~2.75–2.8 GiB` VRAM envelope with **`0 B` extra KV-cache allocation** (`L25..33` have `0` `k_proj/v_proj` weights and read static `shared_KV_{13,14}`).
2. **`2.70M` (`5.14 MiB`) `r=16` `BF16` `tanh` Pass Controllers (`3-Gate: q, mlp, ple`):**
   - Exact parameter count (`S182d` parity): Across `9` layers (`L25..33`) x `3` gates (`q, mlp, ple`, collapsing Inkling-2's dead `k, v` gates) at `r=16, H=1536`: `9 * 3 * (r*2H + 2*H*r) + 9 * 3 * H = 9 * 3 * (4 * 16 * 1536) + 41,472 = 2,654,208 + 41,472 = 2,695,680` params = **`5,391,360 B` (`5.1416 MiB` `BF16`)**.
   - **Zero-Init Exact Identity at `K=1`:** Gates (`forget_gate_proj, gain_gate_proj`) zero-initialize so `K=1` reduces bit-identically (`<= 1e-5` max abs logit diff) to unlooped `35L` Gemma-4 (`P223c` O(1) static frame collapse holds single rotating `(h_24, shared_KV_{13,14})` buffer across passes `p=1..K`, `196,608 B` @ `B=16`).
3. **`P224c` `RiccatiDecoupling` + `S96`/`S164` Coercivity & Adaptive Exit:**
   - Adaptive token gating halts early after pass $p \in \{1,2\}$ when relative inter-pass step residual $s_p = \|h^{(p)}_{33} - h^{(p-1)}_{33}\|_2 / (\|h^{(p)}_{33}\|_2 + 10^{-6}) \le \tau_{\mathrm{exit}}$.
   - **Lyapunov Contraction Guard:** Enforces `P224c` `RiccatiDecoupling` coercivity (`unbounded_P_lyapunov_coercivity`) + `S96` loop-Schur spectral proxy $\hat{\rho}(M_{\mathrm{loop}}) < (\sqrt{5}-1)/2 \approx 0.6180$ ($G_3(\rho) = 1+\rho+\rho^2 \le 2$) + `S164` hard residual clamp $s_p < 1.5$ (clamping/exiting cleanly to `L34` if exceeded), alongside exact `4` sandwich `RMSNorm`s and `layer_scalar` $c_\ell$.

---

## 6. RTX 3080 Roofline Performance Model (`356.68 GB/s` DRAM / `16 GB/s` PCIe Gen4)

Computed from verified `config.json` tensor dimensions and `S183` measured STREAM Triad (`356.68 GB/s`) + PCIe Gen4 (`16.0 GB/s`) baselines (`g=32` `4.5 bpw`, `276` live linears = `1,006.80 MiB`, `L25..33` loop body = `310.92 MiB/pass`, `4.5 bpw` `lm_head` shadow = `216.00 MiB`, tied `BF16` `lm_head` = `768.00 MiB`; `75%–80%` effective DRAM efficiency range modeled from `S180c` `dp4a` graph execution):

| Configuration (`g=32`, `4.5 bpw`) | Streamed Weights / Step | FLOPs / Tok | `B=1` OI | `B=1` Ideal `tok/s` (`ms`) | `B=1` Realistic `75%–80%` `tok/s` | `B=16` OI (`T=1024`) | `B=16` `80%` `tok/s` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`K=1` (`35L`) Body-Only (`276` linears)** | `1,006.80 MiB` | `3.75 GF` | `3.55 F/B` | `337.86 tok/s` (`2.960 ms`) | **`253.39 – 270.29 tok/s`** | — | — |
| **`K=1` (`35L`) + `4.5bpw` `lm_head` (`216 MiB`)** | `1,222.80 MiB` | `4.56 GF` | `3.56 F/B` | `278.18 tok/s` (`3.595 ms`) | **`208.63 – 222.54 tok/s`** | `49.17 F/B` | **`3,077.5 tok/s`** |
| **`K=2` (`44L`) + `4.5bpw` `lm_head` (`216 MiB`)** | `1,533.73 MiB` | `5.72 GF` | `3.56 F/B` | `221.78 tok/s` (`4.509 ms`) | **`166.34 – 177.43 tok/s`** | `50.56 F/B` | **`2,523.0 tok/s`** |
| **`K=3` (`53L`) + `4.5bpw` `lm_head` (`216 MiB`)** | `1,844.65 MiB` | `6.88 GF` | `3.56 F/B` | `184.40 tok/s` (`5.423 ms`) | **`138.30 – 147.52 tok/s`** | `51.53 F/B` | **`2,137.8 tok/s`** |
| **`K=1` (`35L`) + Tied `BF16` `lm_head` (`768 MiB`)** | `1,774.80 MiB` | `4.56 GF` | `2.45 F/B` | `191.66 tok/s` (`5.218 ms`) | **`143.74 – 153.33 tok/s`** | — | — |
| **`K=2` (`44L`) + Tied `BF16` `lm_head` (`768 MiB`)** | `2,085.73 MiB` | `5.72 GF` | `2.62 F/B` | `163.09 tok/s` (`6.132 ms`) | **`122.32 – 130.47 tok/s`** | — | — |
| **`K=3` (`53L`) + Tied `BF16` `lm_head` (`768 MiB`)** | `2,396.65 MiB` | `6.88 GF` | `2.74 F/B` | `141.93 tok/s` (`7.046 ms`) | **`106.45 – 113.54 tok/s`** | — | — |

*High-density `g=128` (`4.125 bpw`) raises `K=1` Body-Only ideal to `368.57 tok/s` (`276.43–294.86` realistic) and `K=1` + Tied `BF16` `lm_head` ideal to `201.17 tok/s` (`150.88–160.93` realistic). Unique `T=1024` KV cache reads (`12 x 512 x 1024 B + 3 x 1024 x 2048 B`) total `12.00 MiB/tok` (`1.19%` of `1,006.80 MiB` `K=1` body weights at `B=1`), while `17,920 B/tok` `PLE` PCIe DMA completes in `1.120 us` (`> 70x` below the `~85 us` `L0` compute window).*

---

## 7. Falsifiable Kill Gates (`KG-GFB1..KG-GFB4`) & Honest Scope Limits (`Law 5`)

1. **`KG-GFB1` (Zero-Anchor Wire Parity, `bos=2` Gate, & Static `0 B`-Alloc Arena):**
   - `SiliGemma` binary `.slgb` (`g=32` `4.5 bpw`, `276` live linears = `1,006.80 MiB`, `0` dead `k/v/k_norm`, `bos=2` header lock) executes `K=1` `35L` decode with **`0` Python/PyTorch/`cuBLAS` dependencies**, `cuda_free_vram_delta_bytes == 0`, `host_heap_delta_bytes == 0`, async `17,920 B/tok` PLE DMA stall `<= 2%` of step latency, `0 B` extra KV allocation across `[13..34]`, peak VRAM `<= 3.0 GiB` (up to `128K` ctx), and max abs logit drift `<= 1e-3` (with `BF16` `lm_head` & `fp32` norms) vs `G0_Base_A16` Form A parity traces (`FAIL` $\implies$ `KILL_FALSIFIED`).
2. **`KG-GFB2` (`0 B`-Spill `sm_86` Kernel Hygiene, `0`-ULP Tree, `IRIS`+SECDED, & Hoist Overhead):**
   - All compiled `sm_86` kernels pass `nvcc --ptxas-options=-v --Werror all-warnings` with **`0 B` stack / `0 B` spill loads / `0 B` spill stores / `0` `LDL`/`STL`**, bulk `Sandwich-RMSNorm + H4 + dp4a` GEMV uses **`<= 48` registers**, `tree_reduce_128_sm86` delivers **`0` ULP drift** across CTA/batch configs, `a16_secded_lop3_popc_syndrome` + `IRIS` `HoloSphereRazorCanary64` yields `0` clean syndromes and `100%` `SafeHalt` kill on injected bit-flip / multi-layer Trojan mutants, and hoisted `c_wb_inv_256b` `8x8` Woodbury overhead stays `<= +25%` (`target -19.57%..+3.44%`) (`FAIL` $\implies$ `KILL_FALSIFIED`).
3. **`KG-GFB3` (Roofline Decode Floor & `L10..L14` `20x` Norm-Clamped Hub Steering):**
   - On `RTX 3080 Laptop` CUDA Graph execution (`B=1`), `K=1` (`35L`) sustains **`>= 190 tok/s` (`4.5bpw` `lm_head`) / `>= 135 tok/s` (tied `BF16` `lm_head`)**, and `K=2` (`44L`) sustains **`>= 155 tok/s` / `>= 115 tok/s`** (`>= 70%` DRAM roofline), while `L10..L14` Cayley–Woodbury steering preserves exact row norms (`| ||W'_r||/||W_r|| - 1 | <= 1e-6`), holds `WT2` drift `<= +0.03` nats, and mutates `0` KV entries inside `[15..34]` (`FAIL` $\implies$ `KILL_FALSIFIED`).
4. **`KG-GFB4` (`U-4` `L25..L33` `0 B`-KV Loop Contraction & Non-Circular Reasoning Lift):**
   - Adaptive $K \in \{1,2,3\}$ looping of `L25..L33` (`44L/53L`) with `2,695,680` (`5.14 MiB`) `r=16` `BF16` `tanh` gates allocates **`0 B` extra KV cache**, satisfies `RiccatiDecoupling` / `S96` $\hat{\rho}(M_{\mathrm{loop}}) < 0.6180$ and `S164` $s_p < 1.5$ across all passes, holds `K=2` `WT2` drift `<= +0.05` nats vs `K=1`, and achieves `>= +0.015` nats Countdown/OOD improvement over `K=1` on disjoint splits (`FAIL` $\implies$ `KILL_FALSIFIED`).
5. **Honest Scope Limits (`Law 5`):**
   - `SiliGemma` `T_{Inkling2 -> Gemma4}` `L25..L33` looping strictly increases iterative refinement depth at `+0 B` bulk weights and `+0 B` KV cache; it does **not** supply `26.39B` `128`-expert parametric recall, cannot modify frozen `shared_KV_{13,14}` key/value tensors inside `[15..34]`, and Stage-1 `sm_86` kernel timings reported here are extrapolated roofline & `S180c` kernel measurements until a post-lock GPU build executes `KG-GFB1..4` live.
