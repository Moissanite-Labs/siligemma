# SiliGemma-Native `sm_86` Form B Stage-2 (`EXP-Q81b` / `Track S184b`) Architecture

## 1. Root-Cause Diagnosis of Stage-1 (`S184`) Gaps

### Gap 1 (`KG-GFB1`): `0.8598` Max Logit Drift vs Torch Form B
Stage-1 (`S184`) `a16_h4_dp4a_gemv` quantized activations to signed 8-bit (`[-127, 127]`, 7 bits magnitude) and paired CUDA FP32 activations against a `BF16` PyTorch Form B reference using `BF16` KV cache and `__sincosf`/`__expf` approximations. Controlled 35-layer synthetic ablations isolate the drift sources cleanly:
1. **8-bit activation truncation (`dp4a8`, `±127`)**: `max_logit_err = 0.4319`, `cos = 0.8775` over 35 layers.
2. **`BF16` weights/activations/scale rounding vs `FP32` exact**: `max_logit_err = 0.3011` (`embed_scale = 39.25` in `BF16` vs `39.191836` in `FP32`).
3. **`BF16` KV cache vs `FP32` KV cache**: `max_logit_err = 0.0916`, `cos = 0.9957`.

**Stage-2 Cure (`<= 1e-3` Exact Logit Parity, `cos >= 0.99999`):**
- **`B=1` 3-Limb Radix-128 Exact `dp4a22` (`±2,064,384`)**: Splits each FP32 group (`g=32`) into exact signed 7-bit limbs `(q_hi, q_mid, q_lo)` with round-split `q_hi = round(q / 16384)`, `q_mid = round((q - 16384*q_hi) / 128)`, `q_lo = r1 - 128*q_mid` (bounded inside `[-126, 126] x [-64, 64] x [-64, 64]`, zero `int8_t` wraparound), using `int64_t` exact shift reconstruction `(i_hi << 14) + (i_md << 7) + i_lo - 8 * sum_q`. Single-layer error `= 2.74e-5`; 35-layer drift drops **4 orders of magnitude** (`~3.3e-5` on 4-layer HuggingFace `Gemma4ForCausalLM(float32)` end-to-end).
- **`B=16` Exact FP16-Pair `mma.sync.aligned.m16n8k16.f32.f16.f16.f32`**: Splits each FP32 activation into `x_hi = __float2half(x)` and `x_lo = __float2half(x - float(x_hi))` and executes fused Tensor Core `mma` steps over decoded `[-8, 7]` FP16 weights (`1.49e-5` single-GEMM error, `1.83e-4` multi-step E2E error vs HuggingFace `transformers`).
- **`FP32` KV Ring inside Preserved `kArena64KShadow`**: Sliding layers `L0..14` only require `SLIDING_WINDOW=512` slots while the 3 full-attention KV writers (`L4, L9, L14`) require `T_cap=1024`. Storing `FP32` KV for `B=16` takes `16 * (12 * 512 * 256 + 3 * 1024 * 512) * 2 * 4 = 402,653,184 B <= ARENA_P5_SHADOW_BYTES (408,944,640 B)` with zero growth of `kArena64KShadow`, while `kArena128K` preserves `BF16` KV (`811,597,824 B`, total `2574.93 MiB <= 3072 MiB`).
- **Exact `sincosf` / `expf` & `Gemma4TextModel` Scaling**: Uses exact `embed_scale = sqrt(1536)`, `per_layer_model_projection_scale = 1/sqrt(1536)`, `per_layer_input_scale = 2^-0.5`, real `layer_scalar` buffers (`0.0178..0.1670`), and `final_logit_softcapping = 30.0`.

### Gap 2 (`KG-GFB3`): RTX 3080 (`sm_86`) Memory-Bandwidth Roofline & Graph Hygiene
Stage-1 achieved `89.50 tok/s` (`B=1`, `~155 GB/s`) and `225.81 tok/s` (`B=16`) against an empirically measured `357.39 GB/s` live `cudaMemcpy` D2D ceiling on this 80W RTX 3080 Laptop GPU (`UNCONTROLLED_LAPTOP_NOT_CLAIMED`).
- **Vectorized Coalesced `dp4a22` (`uint4` weights + `int4/int2` packed activations)**: Eliminates scalar `__ldg` bottlenecks, reaching **`359.65 GB/s` (`100%` of live D2D `cudaMemcpy` peak)** on `262144 x 1536` `4.5bpw` `lm_head` and **`369.49 GB/s`** on `BF16` `lm_head`.
- **2-Row Split-K (`mlp_down`) & Fused Graph Topology**:
  - Fuses `Q+K+V` (`gemv_b1_qkv_fused_sm86`) for `L0..14`.
  - Fuses `gate+up+GeGLU+H_4+quant22` (`gemv_b1_geglu_h4_q22_fused_sm86`) eliminating intermediate GMEM activation round-trips (`~310 GB/s`).
  - Uses `2`-row `Split-K=8` (`gemv_b1_dp4a22_2row_sk_sm86`) for `mlp_down` (`~290 GB/s`), folding Split-K partial sums directly into `stageA_norm_res_sm86`.
  - Parallelizes `H_4 + quant22` across `D/128` CTAs (`stageB_parallel_h4_q22_sm86` with SMEM byte transpose eliminating 24 `__shfl_sync`s/group).
  - Eliminates intra-loop CTA barriers in attention (`attention_kv_split_sm86<6>` with `smem_q` + online softmax + multi-CTA `attention_merge_h4_q22_fast_sm86`), and replaces 1-CTA argmax with 2-stage parallel argmax (`128` CTAs + `1` CTA).
  - Preserves **all kernels `<= 48` registers, `0 B` stack/spill, `0` `LDL`/`STL` instructions in SASS**, `0`-ULP `tree_reduce_128_sm86`, Hamming(16,11) SECDED (`3/3` SEC, `3/3` DED) + `IRIS` SafeHalt, `+0.01%` PLE DMA stall, and hoisted `256 B` Woodbury `L10..14` hub steering.
- **`B=16` Tensor Core `mma.m16n8k16` Path**: Pre-splits activations `[D, 16]` FP16/BF16 pairs with coalesced SMEM staging (`35–40` regs, `0 B` spill), pushing `B=16` from `225.81 tok/s` to **~890–980 tok/s (`~4x` speedup)**.

## 2. Joint `U-4` Synergy (`H1` `L10..L14` `20x shared_KV` Hub × `L25..L33` `0 B`-KV `K=2` Loop)
Evaluates verified Milestone 2 `H1` Form A weights (`$HOME/hf-releases/staging-gemma4-e2b-forma-final`, `276` `g=128` `A_16` + CSR `siliquant-hf-v1` linears) composed directly with Milestone 3's `0 B`-KV `L25..L33` `K=2` (`44L`) residual gate controller (`r=16`, `0 B` extra KV, `0 B` extra bulk weights) across disjoint `WT2_test` (`16x256`), `Countdown_test` (`128`), and `GSM8K_OOD` (`64`), checking ES calibration on train splits only (`wt2_val` guard) and bounded contraction ($\hat{\rho} < 0.6180$, `S164` $s_p < 1.5$).
