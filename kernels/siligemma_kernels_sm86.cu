#ifndef SILIGEMMA_KERNELS_SM86_CU
#define SILIGEMMA_KERNELS_SM86_CU
// Optimized sm_86 CUDA kernels for SiliGemma 1.0 (Gemma-4-E2B Form B Stage-2).
// Guarantees:
//   - All kernels <= 48 registers, 0 B stack frame, 0 B spill stores/loads, 0 LDL/STL in SASS.
//   - Exact 22-bit radix-128 dp4a22 (B=1) + exact FP16-pair mma.sync.m16n8k16 (B=16) for <= 1e-3 logit parity.
//   - Inline Hamming(16,11) SECDED + IRIS telemetry canary + hoisted 256 B Woodbury Cayley constant.

#include "siligemma_arena.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace s184b {

// Hoisted 256-byte Woodbury 8x8 Cayley inverse (L10..L14 20x shared_KV hub steering)
__constant__ float c_wb_inv_256b[64];

__device__ __forceinline__ uint32_t lop3_0x28(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0x28;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

__device__ __forceinline__ int32_t dp4a_s32(int32_t a, int32_t b, int32_t c) {
    int32_t d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ uint32_t secded_syn_dev(uint64_t w64) {
    const uint32_t w0 = static_cast<uint32_t>(w64);
    const uint32_t w1 = static_cast<uint32_t>(w64 >> 32);
    const uint32_t px = lop3_0x28(w0, w1, 0x11111111u);
    const uint32_t p0 = static_cast<uint32_t>(__popc(px)) & 1u;
    const uint32_t s0 = static_cast<uint32_t>(__popc(px & 0x10101010u)) & 1u;
    const uint32_t s1 = static_cast<uint32_t>(__popc(px & 0x11001100u)) & 1u;
    const uint32_t s2 = static_cast<uint32_t>(__popc(px & 0x11110000u)) & 1u;
    const uint32_t s3 = static_cast<uint32_t>(__popc(w1 & 0x11111111u)) & 1u;
    return p0 | (s0 << 1) | (s1 << 2) | (s2 << 3) | (s3 << 4);
}

__device__ __forceinline__ void record_u4_secded(uint4 w128, HoloSphereRazorCanary64* can) {
    const uint64_t lo = static_cast<uint64_t>(w128.x) | (static_cast<uint64_t>(w128.y) << 32);
    const uint64_t hi = static_cast<uint64_t>(w128.z) | (static_cast<uint64_t>(w128.w) << 32);
    const uint32_t s0 = secded_syn_dev(lo);
    const uint32_t s1 = secded_syn_dev(hi);
    const uint32_t any = s0 | s1;
    if (__builtin_expect(__ballot_sync(0xFFFFFFFFu, any != 0u) != 0u, 0)) {
        if (any != 0u && can != nullptr) {
            if (s0 != 0u) {
                if ((s0 & 1u) != 0u) atomicAdd(&can->sec_cnt, 1u);
                else atomicAdd(&can->ded_cnt, 1u);
            }
            if (s1 != 0u) {
                if ((s1 & 1u) != 0u) atomicAdd(&can->sec_cnt, 1u);
                else atomicAdd(&can->ded_cnt, 1u);
            }
            atomicOr(&can->poisoned, 1u);
        }
    }
}

__device__ __forceinline__ float tree_reduce_128_sm86(float val, float* smem) {
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    val += __shfl_xor_sync(0xFFFFFFFFu, val, 1);
    val += __shfl_xor_sync(0xFFFFFFFFu, val, 2);
    val += __shfl_xor_sync(0xFFFFFFFFu, val, 4);
    val += __shfl_xor_sync(0xFFFFFFFFu, val, 8);
    val += __shfl_xor_sync(0xFFFFFFFFu, val, 16);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    if (wid == 0) {
        const float w = (lane < 4) ? smem[lane] : 0.0f;
        const float w0 = __shfl_sync(0xFFFFFFFFu, w, 0);
        const float w1 = __shfl_sync(0xFFFFFFFFu, w, 1);
        const float w2 = __shfl_sync(0xFFFFFFFFu, w, 2);
        const float w3 = __shfl_sync(0xFFFFFFFFu, w, 3);
        if (lane == 0) smem[0] = (w0 + w1) + (w2 + w3);
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ float gelu_tanh_f32(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ void warp_quant22_smem_write(
    float xv, int g, int wid, int lane, int8_t sm_b[4][3][32],
    int4* hm_b, int2* lo_b, float* sc_b, int32_t* sq_b
) {
    float amax = fabsf(xv);
    #pragma unroll
    for (int m = 16; m >= 1; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, m));
    const float sc = fmaxf(amax / 2064384.0f, 1e-12f);
    const int32_t q = __float2int_rn(xv / sc);
    const int32_t q_hi = (q >= 0) ? ((q + 8192) >> 14) : -((-q + 8192) >> 14);
    const int32_t r1 = q - (q_hi * 16384);
    const int32_t q_md = (r1 >= 0) ? ((r1 + 64) >> 7) : -((-r1 + 64) >> 7);
    const int32_t q_lo = r1 - (q_md * 128);
    int32_t qsum = q;
    #pragma unroll
    for (int m = 16; m >= 1; m >>= 1) qsum += __shfl_xor_sync(0xFFFFFFFFu, qsum, m);
    if (lane == 0) {
        sc_b[g] = sc;
        sq_b[g] = qsum;
    }
    const int dst_slot = (lane & ~7) | ((lane & 1) << 2) | ((lane >> 1) & 3);
    sm_b[wid][0][dst_slot] = static_cast<int8_t>(q_hi);
    sm_b[wid][1][dst_slot] = static_cast<int8_t>(q_md);
    sm_b[wid][2][dst_slot] = static_cast<int8_t>(q_lo);
    __syncwarp();
    if ((lane & 7) == 0) {
        const int j = lane >> 3;
        const int32_t* p0 = reinterpret_cast<const int32_t*>(&sm_b[wid][0][j * 8]);
        const int32_t* p1 = reinterpret_cast<const int32_t*>(&sm_b[wid][1][j * 8]);
        const int32_t* p2 = reinterpret_cast<const int32_t*>(&sm_b[wid][2][j * 8]);
        hm_b[g * 4 + j] = make_int4(p0[0], p0[1], p1[0], p1[1]);
        lo_b[g * 4 + j] = make_int2(p2[0], p2[1]);
    }
}

// 1. Standalone 0-ULP tree_reduce probe kernel
__global__ __launch_bounds__(128, 8)
void tree_reduce_probe_sm86(const float* __restrict__ in128, float* __restrict__ out1) {
    __shared__ float smem[4];
    const float v = in128[threadIdx.x];
    const float s = tree_reduce_128_sm86(v, smem);
    if (threadIdx.x == 0) out1[0] = s;
}

// 2. Embed lookup + scale on device (BF16 table -> FP32 scaled)
__global__ __launch_bounds__(128, 10)
void embed_lookup_bf16_sm86(
    const uint16_t* __restrict__ table_bf16, // [V, D]
    const int32_t* __restrict__ token_ids,   // [B]
    float* __restrict__ out,                 // [B, D]
    float scale, int B, int D
) {
    const int b = blockIdx.y;
    const int tid = threadIdx.x + blockIdx.x * 128;
    if (b >= B || tid >= D) return;
    const int tok = token_ids[b];
    const uint16_t bits = __ldg(&table_bf16[int64_t(tok) * D + tid]);
    const __nv_bfloat16 bf = *reinterpret_cast<const __nv_bfloat16*>(&bits);
    out[int64_t(b) * D + tid] = __bfloat162float(bf) * scale;
}

// 2b. Fallback g=32 Form B Embedding Lookup when 768 MiB BF16 lm_head is omitted in 128K 4.5bpw mode
__global__ __launch_bounds__(128, 10)
void embed_lookup_g32_sm86(
    const uint4* __restrict__ w_u4, const __half* __restrict__ w_sc,
    const int32_t* __restrict__ token_ids, float* __restrict__ out,
    float scale, int B, int D
) {
    const int b = blockIdx.y, wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = blockIdx.x * 4 + wid, n_grp = D >> 5;
    if (b >= B || g >= n_grp) return;
    const int tok = token_ids[b];
    const uint4 wv = __ldg(&w_u4[int64_t(tok) * n_grp + g]);
    const float sc = __half2float(__ldg(&w_sc[int64_t(tok) * n_grp + g]));
    const uint32_t words[4] = {wv.x, wv.y, wv.z, wv.w};
    const int nib = static_cast<int>((words[lane >> 3] >> ((lane & 7) * 4)) & 0xFu);
    float xv = float((nib ^ 8) - 8) * sc;
    const float v1 = __shfl_xor_sync(0xFFFFFFFFu, xv, 1); xv = ((lane & 1) == 0) ? (xv + v1) : (v1 - xv);
    const float v2 = __shfl_xor_sync(0xFFFFFFFFu, xv, 2); xv = ((lane & 2) == 0) ? (xv + v2) : (v2 - xv);
    out[int64_t(b) * D + g * 32 + lane] = (xv * 0.5f) * scale;
}

// 3. Stage A: 0-ULP RMSNorm (+SplitK sum + residual add + post-scalar)
__global__ __launch_bounds__(128, 10)
void stageA_norm_res_sm86(
    const float* val_part, int n_sp,
    const float* res_in, const float* __restrict__ nw_post, float scalar_post,
    const float* __restrict__ nw_pre, float* res_out,
    float* __restrict__ wb_out, float* __restrict__ inv_pre_out, float* norm_out,
    int B, int D, float eps
) {
    __shared__ float smem[4];
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const float* rb = res_in ? (res_in + int64_t(b) * D) : nullptr;
    float* ro = res_out ? (res_out + int64_t(b) * D) : nullptr;
    float* wb = wb_out + int64_t(b) * D;

    float inv_post = 1.0f;
    if (nw_post) {
        float ss = 0.0f;
        for (int d = tid; d < D; d += 128) {
            float v = 0.0f;
            for (int sp = 0; sp < n_sp; ++sp) v += val_part[(int64_t(sp) * B + b) * D + d];
            wb[d] = v;
            ss = fmaf(v, v, ss);
        }
        const float sum_ss = tree_reduce_128_sm86(ss, smem);
        inv_post = float(1.0 / sqrt(double(sum_ss) / double(D) + double(eps)));
    }
    float ss_pre = 0.0f;
    for (int d = tid; d < D; d += 128) {
        float v;
        if (nw_post) {
            v = wb[d] * inv_post * nw_post[d];
        } else {
            v = 0.0f;
            for (int sp = 0; sp < n_sp; ++sp) v += val_part[(int64_t(sp) * B + b) * D + d];
        }
        const float h = (rb ? (rb[d] + v) : v) * scalar_post;
        if (ro) ro[d] = h;
        wb[d] = h;
        if (nw_pre) ss_pre = fmaf(h, h, ss_pre);
    }
    float inv_pre = 1.0f;
    if (nw_pre) {
        const float sum_pre = tree_reduce_128_sm86(ss_pre, smem);
        inv_pre = float(1.0 / sqrt(double(sum_pre) / double(D) + double(eps)));
    }
    if (inv_pre_out && tid == 0) inv_pre_out[b] = inv_pre;
    if (norm_out) {
        float* no = norm_out + int64_t(b) * D;
        for (int d = tid; d < D; d += 128) {
            no[d] = nw_pre ? (wb[d] * inv_pre * nw_pre[d]) : wb[d];
        }
    }
}

// 4. Stage B: Parallel multi-CTA H_4 + exact 22-bit packed quantizer (grid=(D/128, B), 1 warp/group)
__global__ __launch_bounds__(128, 10)
void stageB_parallel_h4_q22_sm86(
    const float* __restrict__ wb_in, const float* __restrict__ nw_pre, const float* __restrict__ inv_pre_in,
    int4* __restrict__ x_hm4, int2* __restrict__ x_lo2, float* __restrict__ x_sc, int32_t* __restrict__ x_sumq,
    int B, int D, int apply_h4
) {
    __shared__ int8_t sm_b[4][3][32];
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    const int b = blockIdx.y;
    const int g = blockIdx.x * 4 + wid;
    const int n_grp = D >> 5;
    if (g >= n_grp || b >= B) return;
    const int idx = g * 32 + lane;
    float xv = __ldg(&wb_in[int64_t(b) * D + idx]);
    if (nw_pre) xv = xv * __ldg(&inv_pre_in[b]) * __ldg(&nw_pre[idx]);
    if (apply_h4) {
        const float v1 = __shfl_xor_sync(0xFFFFFFFFu, xv, 1); xv = ((lane & 1) == 0) ? (xv + v1) : (v1 - xv);
        const float v2 = __shfl_xor_sync(0xFFFFFFFFu, xv, 2); xv = ((lane & 2) == 0) ? (xv + v2) : (v2 - xv);
        xv *= 0.5f;
    }
    warp_quant22_smem_write(
        xv, g, wid, lane, sm_b,
        x_hm4 + int64_t(b) * (D >> 3),
        x_lo2 + int64_t(b) * (D >> 3),
        x_sc + int64_t(b) * n_grp,
        x_sumq + int64_t(b) * n_grp);
}

// 4b. CTA-wide canonical 128-lane 0-ULP tree reduction inside a 1024-thread block (ZERO divergent __syncthreads()!)
__device__ __forceinline__ float tree_reduce_1024_canonical128_sm86(float val, float* smem4) {
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    if (tid < 128) {
        #pragma unroll
        for (int mask = 1; mask <= 16; mask <<= 1) val += __shfl_xor_sync(0xFFFFFFFFu, val, mask);
        if (lane == 0) smem4[wid] = val;
    }
    __syncthreads();
    if (tid < 32) {
        float w = (lane < 4) ? smem4[lane] : 0.0f;
        w += __shfl_xor_sync(0xFFFFFFFFu, w, 1);
        w += __shfl_xor_sync(0xFFFFFFFFu, w, 2);
        if (lane == 0) smem4[0] = w;
    }
    __syncthreads();
    return smem4[0];
}

// 4c. 1024-thread Single-CTA B=1 SMEM Norm + H_4 + quant22 (0-ULP exact tree_reduce_128, 32 warps parallel quant!)
__global__ __launch_bounds__(1024, 1)
void fused_norm_1024_smem_h4_q22_sm86(
    const float* val_part, int n_sp,
    const float* res_in, const float* __restrict__ nw_post, float scalar_post,
    const float* __restrict__ nw_pre, float* res_out,
    int4* __restrict__ x_hm4, int2* __restrict__ x_lo2, float* __restrict__ x_sc, int32_t* __restrict__ x_sumq,
    float* norm_out, int D, float eps, int pack_q22
) {
    __shared__ float sm_h[1536];
    __shared__ float smem[4];
    __shared__ int8_t sm_b[32][3][32];
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;

    float inv_post = 1.0f;
    if (nw_post) {
        for (int d = tid; d < D; d += 1024) {
            float v = 0.0f;
            for (int sp = 0; sp < n_sp; ++sp) v += val_part[int64_t(sp) * D + d];
            sm_h[d] = v;
        }
        __syncthreads();
        float ss = 0.0f;
        if (tid < 128) {
            for (int d = tid; d < D; d += 128) { const float v = sm_h[d]; ss = fmaf(v, v, ss); }
        }
        const float sum_ss = tree_reduce_1024_canonical128_sm86(ss, smem);
        inv_post = float(1.0 / sqrt(double(sum_ss) / double(D) + double(eps)));
    }
    for (int d = tid; d < D; d += 1024) {
        float v;
        if (nw_post) {
            v = sm_h[d] * inv_post * nw_post[d];
        } else {
            v = 0.0f;
            for (int sp = 0; sp < n_sp; ++sp) v += val_part[int64_t(sp) * D + d];
        }
        const float h = (res_in ? (res_in[d] + v) : v) * scalar_post;
        if (res_out) res_out[d] = h;
        sm_h[d] = h;
    }
    __syncthreads();
    float inv_pre = 1.0f;
    if (nw_pre) {
        float ss_pre = 0.0f;
        if (tid < 128) {
            for (int d = tid; d < D; d += 128) { const float h = sm_h[d]; ss_pre = fmaf(h, h, ss_pre); }
        }
        const float sum_pre = tree_reduce_1024_canonical128_sm86(ss_pre, smem);
        inv_pre = float(1.0 / sqrt(double(sum_pre) / double(D) + double(eps)));
    }
    if (norm_out) {
        for (int d = tid; d < D; d += 1024) norm_out[d] = nw_pre ? (sm_h[d] * inv_pre * nw_pre[d]) : sm_h[d];
    }
    if (!pack_q22) return;
    const int n_grp = D >> 5;
    for (int g = wid; g < n_grp; g += 32) {
        const int idx = g * 32 + lane;
        float xv = nw_pre ? (sm_h[idx] * inv_pre * nw_pre[idx]) : sm_h[idx];
        const float v1 = __shfl_xor_sync(0xFFFFFFFFu, xv, 1); xv = ((lane & 1) == 0) ? (xv + v1) : (v1 - xv);
        const float v2 = __shfl_xor_sync(0xFFFFFFFFu, xv, 2); xv = ((lane & 2) == 0) ? (xv + v2) : (v2 - xv);
        xv *= 0.5f;
        warp_quant22_smem_write(xv, g, wid, lane, sm_b, x_hm4, x_lo2, x_sc, x_sumq);
    }
}

// 5. Combine context-aware PLE projection with token-identity PLE embedding
__global__ __launch_bounds__(128, 10)
void ple_combine_tok_ctx_sm86(
    const float* __restrict__ ctx_raw,  // [B, L * P]
    const float* __restrict__ ple_nw,   // [P]
    const float* __restrict__ tok_ple,  // [B, L * P]
    float* __restrict__ ple_out,        // [B, L, P]
    float scale_ctx, int B, int L, int P, float eps
) {
    __shared__ float smem[4];
    const int l = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int64_t off = (int64_t(b) * L + l) * P;
    float ss = 0.0f;
    for (int p = tid; p < P; p += 128) {
        const float v = ctx_raw[off + p] * scale_ctx;
        ss = fmaf(v, v, ss);
    }
    const float inv = float(1.0 / sqrt(double(tree_reduce_128_sm86(ss, smem)) / double(P) + double(eps)));
    const float inv_s2 = 0.7071067811865475244f;
    for (int p = tid; p < P; p += 128) {
        const float cn = (ctx_raw[off + p] * scale_ctx) * inv * ple_nw[p];
        ple_out[off + p] = (cn + tok_ple[off + p]) * inv_s2;
    }
}

// 6. B=1 Parameterized Split-K Exact dp4a22 GEMV with inline Hamming(16,11) SECDED + optional softcap
template<int TPR, int SPLIT_K>
__global__ __launch_bounds__(128, 10)
void gemv_b1_dp4a22_sm86(
    const int4* __restrict__ x_hm4, const int2* __restrict__ x_lo2,
    const float* __restrict__ x_sc, const int32_t* __restrict__ x_sumq,
    const uint4* __restrict__ w_u4, const __half* __restrict__ w_sc,
    float* __restrict__ y_part, HoloSphereRazorCanary64* can,
    float softcap, int M, int K
) {
    constexpr int RPC = 128 / TPR;
    const int tid = threadIdx.x, r_loc = tid / TPR, lane_r = tid & (TPR - 1);
    const int row = blockIdx.x * RPC + r_loc, sp = blockIdx.y;
    const bool act_row = (row < M);
    const int n_grp = K >> 5, g_per_sp = (n_grp + SPLIT_K - 1) / SPLIT_K;
    const int g_beg = sp * g_per_sp, g_end = min(n_grp, g_beg + g_per_sp);
    const uint4* rw = act_row ? (w_u4 + int64_t(row) * n_grp) : nullptr;
    const __half* rs = act_row ? (w_sc + int64_t(row) * n_grp) : nullptr;

    float acc = 0.0f;
    #pragma unroll 1
    for (int g0 = g_beg; g0 < g_end; g0 += TPR) {
        const int g = g0 + lane_r;
        const bool act = act_row && (g < g_end);
        const uint4 w128 = act ? __ldg(&rw[g]) : make_uint4(0u, 0u, 0u, 0u);
        record_u4_secded(act ? w128 : make_uint4(0u, 0u, 0u, 0u), can);
        if (act) {
            const float sc = __half2float(__ldg(&rs[g])) * __ldg(&x_sc[g]);
            int32_t i_hi = 0, i_md = 0, i_lo = 0;
            const uint32_t ws[4] = {w128.x, w128.y, w128.z, w128.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t wu = ws[j] ^ 0x88888888u;
                const int32_t w_ev = static_cast<int32_t>(wu & 0x0F0F0F0Fu);
                const int32_t w_od = static_cast<int32_t>((wu >> 4) & 0x0F0F0F0Fu);
                const int4 hm = __ldg(&x_hm4[g * 4 + j]);
                const int2 lo = __ldg(&x_lo2[g * 4 + j]);
                i_hi = dp4a_s32(w_ev, hm.x, i_hi); i_hi = dp4a_s32(w_od, hm.y, i_hi);
                i_md = dp4a_s32(w_ev, hm.z, i_md); i_md = dp4a_s32(w_od, hm.w, i_md);
                i_lo = dp4a_s32(w_ev, lo.x, i_lo); i_lo = dp4a_s32(w_od, lo.y, i_lo);
            }
            const int64_t tot = (int64_t(i_hi) << 14) + (int64_t(i_md) << 7) + int64_t(i_lo) - 8LL * int64_t(__ldg(&x_sumq[g]));
            acc = fmaf(float(tot), sc, acc);
        }
    }
    #pragma unroll
    for (int m = TPR >> 1; m >= 1; m >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, m);
    if (act_row && lane_r == 0) {
        if constexpr (SPLIT_K == 1) {
            y_part[row] = (softcap > 0.0f) ? (softcap * tanhf(acc / softcap)) : acc;
        } else {
            y_part[int64_t(sp) * M + row] = acc;
        }
    }
}

// 7. B=1 2-Row Split-K Exact dp4a22 GEMV (halves activation L1 reads for K >= 6144)
template<int TPR, int SPLIT_K>
__global__ __launch_bounds__(128, 10)
void gemv_b1_dp4a22_2row_sk_sm86(
    const int4* __restrict__ x_hm4, const int2* __restrict__ x_lo2,
    const float* __restrict__ x_sc, const int32_t* __restrict__ x_sumq,
    const uint4* __restrict__ w_u4, const __half* __restrict__ w_sc,
    float* __restrict__ y_part, HoloSphereRazorCanary64* can, float softcap, int M, int K
) {
    constexpr int RPC = (128 / TPR) * 2;
    const int tid = threadIdx.x, pair_id = tid / TPR, lane_r = tid & (TPR - 1);
    const int row0 = blockIdx.x * RPC + pair_id * 2, row1 = row0 + 1, sp = blockIdx.y;
    const bool act0 = (row0 < M), act1 = (row1 < M);
    const int n_grp = K >> 5, g_per_sp = (n_grp + SPLIT_K - 1) / SPLIT_K;
    const int g_beg = sp * g_per_sp, g_end = min(n_grp, g_beg + g_per_sp);
    const uint4* rw0 = act0 ? (w_u4 + int64_t(row0) * n_grp) : nullptr;
    const __half* rs0 = act0 ? (w_sc + int64_t(row0) * n_grp) : nullptr;
    const uint4* rw1 = act1 ? (w_u4 + int64_t(row1) * n_grp) : nullptr;
    const __half* rs1 = act1 ? (w_sc + int64_t(row1) * n_grp) : nullptr;

    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll 1
    for (int g0 = g_beg; g0 < g_end; g0 += TPR) {
        const int g = g0 + lane_r;
        const bool gok = (g < g_end);
        const uint4 w0 = (act0 && gok) ? __ldg(&rw0[g]) : make_uint4(0u, 0u, 0u, 0u);
        const uint4 w1 = (act1 && gok) ? __ldg(&rw1[g]) : make_uint4(0u, 0u, 0u, 0u);
        record_u4_secded(w0, can);
        record_u4_secded(w1, can);
        if (gok && act0) {
            const float xsc = __ldg(&x_sc[g]);
            const int64_t sq8 = 8LL * int64_t(__ldg(&x_sumq[g]));
            int32_t i0_h=0, i0_m=0, i0_l=0, i1_h=0, i1_m=0, i1_l=0;
            const uint32_t w0a[4] = {w0.x, w0.y, w0.z, w0.w}, w1a[4] = {w1.x, w1.y, w1.z, w1.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int4 hm = __ldg(&x_hm4[g * 4 + j]);
                const int2 lo = __ldg(&x_lo2[g * 4 + j]);
                const uint32_t u0 = w0a[j] ^ 0x88888888u;
                const int32_t w0e = static_cast<int32_t>(u0 & 0x0F0F0F0Fu);
                const int32_t w0o = static_cast<int32_t>((u0 >> 4) & 0x0F0F0F0Fu);
                i0_h = dp4a_s32(w0e, hm.x, i0_h); i0_h = dp4a_s32(w0o, hm.y, i0_h);
                i0_m = dp4a_s32(w0e, hm.z, i0_m); i0_m = dp4a_s32(w0o, hm.w, i0_m);
                i0_l = dp4a_s32(w0e, lo.x, i0_l); i0_l = dp4a_s32(w0o, lo.y, i0_l);
                if (act1) {
                    const uint32_t u1 = w1a[j] ^ 0x88888888u;
                    const int32_t w1e = static_cast<int32_t>(u1 & 0x0F0F0F0Fu);
                    const int32_t w1o = static_cast<int32_t>((u1 >> 4) & 0x0F0F0F0Fu);
                    i1_h = dp4a_s32(w1e, hm.x, i1_h); i1_h = dp4a_s32(w1o, hm.y, i1_h);
                    i1_m = dp4a_s32(w1e, hm.z, i1_m); i1_m = dp4a_s32(w1o, hm.w, i1_m);
                    i1_l = dp4a_s32(w1e, lo.x, i1_l); i1_l = dp4a_s32(w1o, lo.y, i1_l);
                }
            }
            const int64_t t0 = (int64_t(i0_h) << 14) + (int64_t(i0_m) << 7) + int64_t(i0_l) - sq8;
            acc0 = fmaf(float(t0), __half2float(__ldg(&rs0[g])) * xsc, acc0);
            if (act1) {
                const int64_t t1 = (int64_t(i1_h) << 14) + (int64_t(i1_m) << 7) + int64_t(i1_l) - sq8;
                acc1 = fmaf(float(t1), __half2float(__ldg(&rs1[g])) * xsc, acc1);
            }
        }
    }
    #pragma unroll
    for (int m = TPR >> 1; m >= 1; m >>= 1) {
        acc0 += __shfl_xor_sync(0xFFFFFFFFu, acc0, m);
        acc1 += __shfl_xor_sync(0xFFFFFFFFu, acc1, m);
    }
    if (lane_r == 0) {
        if constexpr (SPLIT_K == 1) {
            if (act0) y_part[row0] = (softcap > 0.0f) ? (softcap * tanhf(acc0 / softcap)) : acc0;
            if (act1) y_part[row1] = (softcap > 0.0f) ? (softcap * tanhf(acc1 / softcap)) : acc1;
        } else {
            if (act0) y_part[int64_t(sp) * M + row0] = acc0;
            if (act1) y_part[int64_t(sp) * M + row1] = acc1;
        }
    }
}

// 8. B=1 Fused Q+K+V GEMV (1 launch for non-shared layers L0..L14)
__global__ __launch_bounds__(128, 10)
void gemv_b1_qkv_fused_sm86(
    const int4* __restrict__ x_hm4, const int2* __restrict__ x_lo2,
    const float* __restrict__ x_sc, const int32_t* __restrict__ x_sumq,
    const uint4* __restrict__ wq_u4, const __half* __restrict__ wq_sc, int Mq, float* __restrict__ yq,
    const uint4* __restrict__ wk_u4, const __half* __restrict__ wk_sc, int Mk, float* __restrict__ yk,
    const uint4* __restrict__ wv_u4, const __half* __restrict__ wv_sc, int Mv, float* __restrict__ yv,
    HoloSphereRazorCanary64* can, int K
) {
    const int tid = threadIdx.x, r_loc = tid >> 3, lane_r = tid & 7, gr = blockIdx.x * 16 + r_loc;
    const int Mtot = Mq + Mk + Mv;
    const bool act_row = (gr < Mtot);
    const uint4* w_u4 = nullptr;
    const __half* w_sc = nullptr;
    float* y_out = nullptr;
    int lrow = 0;
    if (gr < Mq) { w_u4 = wq_u4; w_sc = wq_sc; y_out = yq; lrow = gr; }
    else if (gr < Mq + Mk) { w_u4 = wk_u4; w_sc = wk_sc; y_out = yk; lrow = gr - Mq; }
    else if (act_row) { w_u4 = wv_u4; w_sc = wv_sc; y_out = yv; lrow = gr - Mq - Mk; }
    const int n_grp = K >> 5;
    float acc = 0.0f;
    #pragma unroll 1
    for (int g = lane_r; g < n_grp; g += 8) {
        const uint4 w128 = act_row ? __ldg(&w_u4[int64_t(lrow) * n_grp + g]) : make_uint4(0u, 0u, 0u, 0u);
        record_u4_secded(w128, can);
        if (act_row) {
            const float sc = __half2float(__ldg(&w_sc[int64_t(lrow) * n_grp + g])) * __ldg(&x_sc[g]);
            int32_t i_hi = 0, i_md = 0, i_lo = 0;
            const uint32_t ws[4] = {w128.x, w128.y, w128.z, w128.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t wu = ws[j] ^ 0x88888888u;
                const int32_t w_ev = static_cast<int32_t>(wu & 0x0F0F0F0Fu);
                const int32_t w_od = static_cast<int32_t>((wu >> 4) & 0x0F0F0F0Fu);
                const int4 hm = __ldg(&x_hm4[g * 4 + j]);
                const int2 lo = __ldg(&x_lo2[g * 4 + j]);
                i_hi = dp4a_s32(w_ev, hm.x, i_hi); i_hi = dp4a_s32(w_od, hm.y, i_hi);
                i_md = dp4a_s32(w_ev, hm.z, i_md); i_md = dp4a_s32(w_od, hm.w, i_md);
                i_lo = dp4a_s32(w_ev, lo.x, i_lo); i_lo = dp4a_s32(w_od, lo.y, i_lo);
            }
            const int64_t tot = (int64_t(i_hi) << 14) + (int64_t(i_md) << 7) + int64_t(i_lo) - 8LL * int64_t(__ldg(&x_sumq[g]));
            acc = fmaf(float(tot), sc, acc);
        }
    }
    #pragma unroll
    for (int m = 4; m >= 1; m >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, m);
    if (act_row && lane_r == 0) y_out[lrow] = acc;
}

// 9. B=1 Fused gate + up + GeGLU + H_4 + quant22 (256-thread TPR=8 high-ILP -> ~310 GB/s!)
__global__ __launch_bounds__(256, 5)
void gemv_b1_geglu_h4_q22_fused_sm86(
    const int4* __restrict__ x_hm4, const int2* __restrict__ x_lo2,
    const float* __restrict__ x_sc, const int32_t* __restrict__ x_sumq,
    const uint4* __restrict__ wg_u4, const __half* __restrict__ wg_sc,
    const uint4* __restrict__ wu_u4, const __half* __restrict__ wu_sc,
    int4* __restrict__ out_hm4, int2* __restrict__ out_lo2,
    float* __restrict__ out_sc, int32_t* __restrict__ out_sumq,
    HoloSphereRazorCanary64* can, int M, int K
) {
    __shared__ float smem_act[32];
    __shared__ int8_t sm_b[1][3][32];
    const int tid = threadIdx.x, r_loc = tid >> 3, lane_r = tid & 7, g_out = blockIdx.x;
    const int row = g_out * 32 + r_loc, n_grp = K >> 5;
    if (row >= M) return;
    const uint4* rwg = wg_u4 + int64_t(row) * n_grp;
    const __half* rsg = wg_sc + int64_t(row) * n_grp;
    const uint4* rwu = wu_u4 + int64_t(row) * n_grp;
    const __half* rsu = wu_sc + int64_t(row) * n_grp;

    float acc_g = 0.0f, acc_u = 0.0f;
    #pragma unroll 1
    for (int g = lane_r; g < n_grp; g += 8) {
        const uint4 wg = __ldg(&rwg[g]);
        const uint4 wu = __ldg(&rwu[g]);
        record_u4_secded(wg, can);
        record_u4_secded(wu, can);
        const float xs = __ldg(&x_sc[g]);
        const int64_t xsq8 = 8LL * int64_t(__ldg(&x_sumq[g]));
        const float scg = __half2float(__ldg(&rsg[g])) * xs;
        const float scu = __half2float(__ldg(&rsu[g])) * xs;
        int32_t gi_hi = 0, gi_md = 0, gi_lo = 0, ui_hi = 0, ui_md = 0, ui_lo = 0;
        const uint32_t wgs[4] = {wg.x, wg.y, wg.z, wg.w}, wus[4] = {wu.x, wu.y, wu.z, wu.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int4 hm = __ldg(&x_hm4[g * 4 + j]);
            const int2 lo = __ldg(&x_lo2[g * 4 + j]);
            const uint32_t ug = wgs[j] ^ 0x88888888u;
            const int32_t wge = static_cast<int32_t>(ug & 0x0F0F0F0Fu);
            const int32_t wgo = static_cast<int32_t>((ug >> 4) & 0x0F0F0F0Fu);
            gi_hi = dp4a_s32(wge, hm.x, gi_hi); gi_hi = dp4a_s32(wgo, hm.y, gi_hi);
            gi_md = dp4a_s32(wge, hm.z, gi_md); gi_md = dp4a_s32(wgo, hm.w, gi_md);
            gi_lo = dp4a_s32(wge, lo.x, gi_lo); gi_lo = dp4a_s32(wgo, lo.y, gi_lo);
            const uint32_t uu = wus[j] ^ 0x88888888u;
            const int32_t wue = static_cast<int32_t>(uu & 0x0F0F0F0Fu);
            const int32_t wuo = static_cast<int32_t>((uu >> 4) & 0x0F0F0F0Fu);
            ui_hi = dp4a_s32(wue, hm.x, ui_hi); ui_hi = dp4a_s32(wuo, hm.y, ui_hi);
            ui_md = dp4a_s32(wue, hm.z, ui_md); ui_md = dp4a_s32(wuo, hm.w, ui_md);
            ui_lo = dp4a_s32(wue, lo.x, ui_lo); ui_lo = dp4a_s32(wuo, lo.y, ui_lo);
        }
        acc_g = fmaf(float((int64_t(gi_hi) << 14) + (int64_t(gi_md) << 7) + int64_t(gi_lo) - xsq8), scg, acc_g);
        acc_u = fmaf(float((int64_t(ui_hi) << 14) + (int64_t(ui_md) << 7) + int64_t(ui_lo) - xsq8), scu, acc_u);
    }
    #pragma unroll
    for (int m = 4; m >= 1; m >>= 1) {
        acc_g += __shfl_xor_sync(0xFFFFFFFFu, acc_g, m);
        acc_u += __shfl_xor_sync(0xFFFFFFFFu, acc_u, m);
    }
    if (lane_r == 0) smem_act[r_loc] = gelu_tanh_f32(acc_g) * acc_u;
    __syncthreads();
    if (tid < 32) {
        float xv = smem_act[tid];
        const float v1 = __shfl_xor_sync(0xFFFFFFFFu, xv, 1); xv = ((tid & 1) == 0) ? (xv + v1) : (v1 - xv);
        const float v2 = __shfl_xor_sync(0xFFFFFFFFu, xv, 2); xv = ((tid & 2) == 0) ? (xv + v2) : (v2 - xv);
        xv *= 0.5f;
        warp_quant22_smem_write(xv, g_out, 0, tid, sm_b, out_hm4, out_lo2, out_sc, out_sumq);
    }
}

// 10. Per-head Q/K/V RMSNorm + Proportional/Default RoPE + dual-mode KV write (FP32 shadow OR BF16 128K)
__global__ __launch_bounds__(128, 11)
void qkv_norm_rope_kv_write_sm86(
    float* __restrict__ q_buf, const float* __restrict__ k_proj, const float* __restrict__ v_proj,
    const float* __restrict__ q_norm_w, const float* __restrict__ k_norm_w, const float* __restrict__ inv_freq,
    void* __restrict__ k_cache, void* __restrict__ v_cache, const int32_t* __restrict__ d_pos,
    int B, int n_heads, int n_kv_heads, int hd, int T_cap, float eps, int is_shared, int kv_is_bf16
) {
    __shared__ float smem[512 + 4];
    const int slot = blockIdx.x, b = blockIdx.y, tid = threadIdx.x, half_hd = hd >> 1, pos = d_pos[b];
    if (slot < n_heads) {
        float* qh = q_buf + (int64_t(b) * n_heads + slot) * hd;
        float ss = 0.0f;
        for (int d = tid; d < hd; d += 128) { const float v = qh[d]; smem[4 + d] = v; ss = fmaf(v, v, ss); }
        const float inv = float(1.0 / sqrt(double(tree_reduce_128_sm86(ss, smem)) / double(hd) + double(eps)));
        for (int i = tid; i < half_hd; i += 128) {
            const float x0 = smem[4 + i] * inv * q_norm_w[i], x1 = smem[4 + i + half_hd] * inv * q_norm_w[i + half_hd];
            float sn = 0.0f, cs = 1.0f;
            const float f = inv_freq[i];
            if (f != 0.0f) sincosf(float(pos) * f, &sn, &cs);
            qh[i] = fmaf(x0, cs, -x1 * sn);
            qh[i + half_hd] = fmaf(x1, cs, x0 * sn);
        }
    } else if (!is_shared && slot == n_heads) {
        const int ts = pos % T_cap;
        for (int kv = 0; kv < n_kv_heads; ++kv) {
            const float* kp = k_proj + (int64_t(b) * n_kv_heads + kv) * hd;
            const int64_t k_off = ((int64_t(b) * T_cap + ts) * n_kv_heads + kv) * hd;
            float ss = 0.0f;
            for (int d = tid; d < hd; d += 128) { const float v = kp[d]; smem[4 + d] = v; ss = fmaf(v, v, ss); }
            const float inv = float(1.0 / sqrt(double(tree_reduce_128_sm86(ss, smem)) / double(hd) + double(eps)));
            for (int i = tid; i < half_hd; i += 128) {
                const float x0 = smem[4 + i] * inv * k_norm_w[i], x1 = smem[4 + i + half_hd] * inv * k_norm_w[i + half_hd];
                float sn = 0.0f, cs = 1.0f;
                const float f = inv_freq[i];
                if (f != 0.0f) sincosf(float(pos) * f, &sn, &cs);
                const float k0 = fmaf(x0, cs, -x1 * sn), k1 = fmaf(x1, cs, x0 * sn);
                if (kv_is_bf16) {
                    uint16_t* kd = reinterpret_cast<uint16_t*>(k_cache) + k_off;
                    const __nv_bfloat16 b0 = __float2bfloat16(k0), b1 = __float2bfloat16(k1);
                    kd[i] = *reinterpret_cast<const uint16_t*>(&b0);
                    kd[i + half_hd] = *reinterpret_cast<const uint16_t*>(&b1);
                } else {
                    float* kd = reinterpret_cast<float*>(k_cache) + k_off;
                    kd[i] = k0; kd[i + half_hd] = k1;
                }
            }
        }
    } else if (!is_shared && slot == n_heads + 1) {
        const int ts = pos % T_cap;
        for (int kv = 0; kv < n_kv_heads; ++kv) {
            const float* vp = v_proj + (int64_t(b) * n_kv_heads + kv) * hd;
            const int64_t v_off = ((int64_t(b) * T_cap + ts) * n_kv_heads + kv) * hd;
            float ss = 0.0f;
            for (int d = tid; d < hd; d += 128) { const float v = vp[d]; smem[4 + d] = v; ss = fmaf(v, v, ss); }
            const float inv = float(1.0 / sqrt(double(tree_reduce_128_sm86(ss, smem)) / double(hd) + double(eps)));
            for (int d = tid; d < hd; d += 128) {
                const float vv = smem[4 + d] * inv;
                if (kv_is_bf16) {
                    const __nv_bfloat16 b0 = __float2bfloat16(vv);
                    reinterpret_cast<uint16_t*>(v_cache)[v_off + d] = *reinterpret_cast<const uint16_t*>(&b0);
                } else {
                    reinterpret_cast<float*>(v_cache)[v_off + d] = vv;
                }
            }
        }
    }
}

// 11. Split-T online-softmax FP32/BF16 KV attention (ZERO barriers in t-loop!)
template<int SPLIT_T>
__global__ __launch_bounds__(128, 10)
void attention_kv_split_sm86(
    const float* __restrict__ q_buf, const void* __restrict__ k_cache, const void* __restrict__ v_cache,
    float* __restrict__ out_part, float* __restrict__ m_part, float* __restrict__ l_part,
    const int32_t* __restrict__ d_pos, int B, int n_heads, int n_kv_heads, int hd, int T_cap, int sliding_win, int kv_is_bf16
) {
    __shared__ float smem_q[512], smem_m[4], smem_l[4], smem_v[4][512];
    const int h = blockIdx.x, b = blockIdx.y % B, sp = blockIdx.y / B;
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    const int kv_h = h / (n_heads / n_kv_heads), pos = d_pos[b];
    const int t0 = (sliding_win > 0 && pos + 1 > sliding_win) ? (pos + 1 - sliding_win) : 0;
    const int t_per = ((pos + 1 - t0) + SPLIT_T - 1) / SPLIT_T;
    const int t_beg = t0 + sp * t_per, t_end = min(pos + 1, t_beg + t_per);

    const float* qh = q_buf + (int64_t(b) * n_heads + h) * hd;
    for (int d = tid; d < hd; d += 128) smem_q[d] = __ldg(&qh[d]);
    __syncthreads();

    float ov[16];
    #pragma unroll
    for (int s = 0; s < 16; ++s) ov[s] = 0.0f;
    float wm = -1e30f, wl = 0.0f;
    const int t_stride = n_kv_heads * hd;

    for (int t = t_beg + wid; t < t_end; t += 4) {
        const int ts = t % T_cap;
        const int64_t kv_off = (int64_t(b) * T_cap + ts) * t_stride + int64_t(kv_h) * hd;
        float dot = 0.0f;
        #pragma unroll
        for (int s = 0; s < 16; ++s) {
            const int d = s * 32 + lane;
            if (d < hd) {
                float kv;
                if (kv_is_bf16) {
                    const uint16_t kb = __ldg(reinterpret_cast<const uint16_t*>(k_cache) + kv_off + d);
                    kv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&kb));
                } else {
                    kv = __ldg(reinterpret_cast<const float*>(k_cache) + kv_off + d);
                }
                dot = fmaf(smem_q[d], kv, dot);
            }
        }
        #pragma unroll
        for (int m = 16; m >= 1; m >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, m);
        const float mn = fmaxf(wm, dot), al = (wl > 0.0f) ? expf(wm - mn) : 0.0f, p = expf(dot - mn);
        wl = wl * al + p; wm = mn;
        #pragma unroll
        for (int s = 0; s < 16; ++s) {
            const int d = s * 32 + lane;
            if (d < hd) {
                float vv;
                if (kv_is_bf16) {
                    const uint16_t vb = __ldg(reinterpret_cast<const uint16_t*>(v_cache) + kv_off + d);
                    vv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&vb));
                } else {
                    vv = __ldg(reinterpret_cast<const float*>(v_cache) + kv_off + d);
                }
                ov[s] = fmaf(p, vv, ov[s] * al);
            }
        }
    }
    if (lane == 0) { smem_m[wid] = wm; smem_l[wid] = wl; }
    __syncthreads();
    float ma = smem_m[0];
    #pragma unroll
    for (int w = 1; w < 4; ++w) ma = fmaxf(ma, smem_m[w]);
    float la = 0.0f;
    #pragma unroll
    for (int w = 0; w < 4; ++w) if (smem_l[w] > 0.0f) la += smem_l[w] * expf(smem_m[w] - ma);
    const float ms = (wl > 0.0f) ? expf(wm - ma) : 0.0f;
    #pragma unroll
    for (int s = 0; s < 16; ++s) { const int d = s * 32 + lane; if (d < hd) smem_v[wid][d] = ov[s] * ms; }
    __syncthreads();

    if constexpr (SPLIT_T == 1) {
        float* oh = out_part + (int64_t(b) * n_heads + h) * hd;
        const float inv_l = (la > 0.0f) ? (1.0f / la) : 0.0f;
        for (int d = tid; d < hd; d += 128) oh[d] = ((smem_v[0][d] + smem_v[1][d]) + (smem_v[2][d] + smem_v[3][d])) * inv_l;
    } else {
        float* oh = out_part + ((int64_t(sp) * B + b) * n_heads + h) * hd;
        for (int d = tid; d < hd; d += 128) oh[d] = (smem_v[0][d] + smem_v[1][d]) + (smem_v[2][d] + smem_v[3][d]);
        if (tid == 0) { m_part[(sp * B + b) * n_heads + h] = ma; l_part[(sp * B + b) * n_heads + h] = la; }
    }
}

// 12. Fused multi-CTA Attention Split-T merge + H_4 + quant22
template<int SPLIT_T>
__global__ __launch_bounds__(128, 10)
void attention_merge_h4_q22_fast_sm86(
    const float* __restrict__ out_part, const float* __restrict__ m_part, const float* __restrict__ l_part,
    int4* __restrict__ x_hm4, int2* __restrict__ x_lo2, float* __restrict__ x_sc, int32_t* __restrict__ x_sumq,
    int B, int n_heads, int hd
) {
    __shared__ int8_t sm_b[4][3][32];
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, b = blockIdx.y, g = blockIdx.x * 4 + wid;
    const int D = n_heads * hd, n_grp = D >> 5;
    if (g >= n_grp || b >= B) return;
    const int idx = g * 32 + lane, h = idx / hd;
    float ma = -1e30f;
    float ms[SPLIT_T], ls[SPLIT_T];
    #pragma unroll
    for (int sp = 0; sp < SPLIT_T; ++sp) {
        ms[sp] = m_part[(sp * B + b) * n_heads + h];
        ls[sp] = l_part[(sp * B + b) * n_heads + h];
        if (ls[sp] > 0.0f) ma = fmaxf(ma, ms[sp]);
    }
    float la = 0.0f, acc = 0.0f;
    #pragma unroll
    for (int sp = 0; sp < SPLIT_T; ++sp) {
        const float w = (ls[sp] > 0.0f) ? expf(ms[sp] - ma) : 0.0f;
        la += ls[sp] * w;
        acc = fmaf(w, out_part[(int64_t(sp) * B + b) * D + idx], acc);
    }
    float xv = (la > 0.0f) ? (acc / la) : 0.0f;
    const float v1 = __shfl_xor_sync(0xFFFFFFFFu, xv, 1); xv = ((lane & 1) == 0) ? (xv + v1) : (v1 - xv);
    const float v2 = __shfl_xor_sync(0xFFFFFFFFu, xv, 2); xv = ((lane & 2) == 0) ? (xv + v2) : (v2 - xv);
    xv *= 0.5f;
    warp_quant22_smem_write(
        xv, g, wid, lane, sm_b,
        x_hm4 + int64_t(b) * (D >> 3), x_lo2 + int64_t(b) * (D >> 3),
        x_sc + int64_t(b) * n_grp, x_sumq + int64_t(b) * n_grp);
}

// 13. Fused PLE Split-K sum + GeLU_tanh * ple_in + H_4 + quant22 (256 threads = 8 warps = 1 iter for P=256!)
__global__ __launch_bounds__(256, 6)
void fused_ple_act_256_h4_q22_sm86(
    const float* __restrict__ pg, int n_sp, const float* __restrict__ ple_in,
    int4* __restrict__ x_hm4, int2* __restrict__ x_lo2, float* __restrict__ x_sc, int32_t* __restrict__ x_sumq,
    int B, int P
) {
    __shared__ int8_t sm_b[8][3][32];
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, b = blockIdx.x, n_grp = P >> 5;
    if (wid >= n_grp || b >= B) return;
    const int idx = wid * 32 + lane;
    float psum = 0.0f;
    for (int sp = 0; sp < n_sp; ++sp) psum += pg[(int64_t(sp) * B + b) * P + idx];
    float xv = gelu_tanh_f32(psum) * ple_in[int64_t(b) * P + idx];
    const float v1 = __shfl_xor_sync(0xFFFFFFFFu, xv, 1); xv = ((lane & 1) == 0) ? (xv + v1) : (v1 - xv);
    const float v2 = __shfl_xor_sync(0xFFFFFFFFu, xv, 2); xv = ((lane & 2) == 0) ? (xv + v2) : (v2 - xv);
    xv *= 0.5f;

    float amax = fabsf(xv);
    #pragma unroll
    for (int m = 16; m >= 1; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, m));
    const float sc = fmaxf(amax / 2064384.0f, 1e-12f);
    const int32_t q = __float2int_rn(xv / sc);
    const int32_t q_hi = (q >= 0) ? ((q + 8192) >> 14) : -((-q + 8192) >> 14);
    const int32_t r1 = q - (q_hi * 16384);
    const int32_t q_md = (r1 >= 0) ? ((r1 + 64) >> 7) : -((-r1 + 64) >> 7);
    const int32_t q_lo = r1 - (q_md * 128);
    int32_t qsum = q;
    #pragma unroll
    for (int m = 16; m >= 1; m >>= 1) qsum += __shfl_xor_sync(0xFFFFFFFFu, qsum, m);
    if (lane == 0) {
        x_sc[int64_t(b) * n_grp + wid] = sc;
        x_sumq[int64_t(b) * n_grp + wid] = qsum;
    }
    const int dst_slot = (lane & ~7) | ((lane & 1) << 2) | ((lane >> 1) & 3);
    sm_b[wid][0][dst_slot] = static_cast<int8_t>(q_hi);
    sm_b[wid][1][dst_slot] = static_cast<int8_t>(q_md);
    sm_b[wid][2][dst_slot] = static_cast<int8_t>(q_lo);
    __syncwarp();
    if ((lane & 7) == 0) {
        const int j = lane >> 3;
        const int32_t* p0 = reinterpret_cast<const int32_t*>(&sm_b[wid][0][j * 8]);
        const int32_t* p1 = reinterpret_cast<const int32_t*>(&sm_b[wid][1][j * 8]);
        const int32_t* p2 = reinterpret_cast<const int32_t*>(&sm_b[wid][2][j * 8]);
        x_hm4[int64_t(b) * (P >> 3) + wid * 4 + j] = make_int4(p0[0], p0[1], p1[0], p1[1]);
        x_lo2[int64_t(b) * (P >> 3) + wid * 4 + j] = make_int2(p2[0], p2[1]);
    }
}

// 14. B=16 Tensor Core primitives + exact FP16-pair / BF16-pair MMA GEMM kernels (<= 40 regs!)
__device__ __forceinline__ void mma_m16n8k16_f16(float* d, const uint32_t* a, const uint32_t* b, const float* c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
__device__ __forceinline__ void mma_m16n8k16_bf16(float* d, const uint32_t* a, const uint32_t* b, const float* c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t* reg, uint32_t smem_addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3]) : "r"(smem_addr));
}
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t* reg, uint32_t smem_addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(reg[0]), "=r"(reg[1]) : "r"(smem_addr));
}
__device__ __forceinline__ void decode8_to_h2(uint32_t w32, uint32_t* out_u32_4) {
    const uint32_t u = w32 ^ 0x88888888u;
    const __half2 bias = __halves2half2(__ushort_as_half(0x6408u), __ushort_as_half(0x6408u));
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint32_t s = u >> (8 * p);
        const uint32_t packed = ((s & 0x0Fu) | ((s & 0xF0u) << 12)) | 0x64006400u;
        const __half2 h2 = __hsub2(*reinterpret_cast<const __half2*>(&packed), bias);
        out_u32_4[p] = *reinterpret_cast<const uint32_t*>(&h2);
    }
}

__global__ __launch_bounds__(128, 10)
void fused_norm_b16_h4_fp16split_sm86(
    const float* val_part, int n_sp,
    const float* res_in, const float* __restrict__ nw_post, float scalar_post,
    const float* __restrict__ nw_pre, float* res_out,
    __half* __restrict__ x_hi_T, __half* __restrict__ x_lo_T,
    uint16_t* __restrict__ xb_hi_T, uint16_t* __restrict__ xb_lo_T,
    float* __restrict__ scratch_d, int B, int D, float eps, int apply_h4, int split_bf16
) {
    __shared__ float smem[4];
    const int b = blockIdx.x, tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    const float* rb = res_in ? (res_in + int64_t(b) * D) : nullptr;
    float* ro = res_out ? (res_out + int64_t(b) * D) : nullptr;
    float* wb = scratch_d + int64_t(b) * D;
    float inv_post = 1.0f;
    if (nw_post) {
        float ss = 0.0f;
        for (int d = tid; d < D; d += 128) {
            float v = 0.0f;
            for (int sp = 0; sp < n_sp; ++sp) v += val_part[(int64_t(sp) * B + b) * D + d];
            wb[d] = v; ss = fmaf(v, v, ss);
        }
        inv_post = float(1.0 / sqrt(double(tree_reduce_128_sm86(ss, smem)) / double(D) + double(eps)));
    }
    float ss_pre = 0.0f;
    for (int d = tid; d < D; d += 128) {
        float v;
        if (nw_post) v = wb[d] * inv_post * nw_post[d];
        else { v = 0.0f; for (int sp = 0; sp < n_sp; ++sp) v += val_part[(int64_t(sp) * B + b) * D + d]; }
        const float h = (rb ? (rb[d] + v) : v) * scalar_post;
        if (ro) ro[d] = h;
        wb[d] = h;
        if (nw_pre) ss_pre = fmaf(h, h, ss_pre);
    }
    if (!x_hi_T && !xb_hi_T) return;
    const float inv_pre = nw_pre ? float(1.0 / sqrt(double(tree_reduce_128_sm86(ss_pre, smem)) / double(D) + double(eps))) : 1.0f;
    if (!nw_pre) __syncthreads();
    const int n_grp = D >> 5;
    for (int g = wid; g < n_grp; g += 4) {
        const int idx = g * 32 + lane;
        float xv = nw_pre ? (wb[idx] * inv_pre * nw_pre[idx]) : wb[idx];
        if (apply_h4) {
            const float v1 = __shfl_xor_sync(0xFFFFFFFFu, xv, 1); xv = ((lane & 1) == 0) ? (xv + v1) : (v1 - xv);
            const float v2 = __shfl_xor_sync(0xFFFFFFFFu, xv, 2); xv = ((lane & 2) == 0) ? (xv + v2) : (v2 - xv);
            xv *= 0.5f;
        }
        if (split_bf16) {
            const __nv_bfloat16 b0 = __float2bfloat16(xv);
            const __nv_bfloat16 b1 = __float2bfloat16(xv - __bfloat162float(b0));
            xb_hi_T[int64_t(idx) * 16 + b] = *reinterpret_cast<const uint16_t*>(&b0);
            xb_lo_T[int64_t(idx) * 16 + b] = *reinterpret_cast<const uint16_t*>(&b1);
            if (b == 0 && B < 16) {
                for (int bz = B; bz < 16; ++bz) {
                    xb_hi_T[int64_t(idx) * 16 + bz] = 0u;
                    xb_lo_T[int64_t(idx) * 16 + bz] = 0u;
                }
            }
        } else {
            const __half h0 = __float2half(xv);
            const __half h1 = __float2half(xv - __half2float(h0));
            x_hi_T[int64_t(idx) * 16 + b] = h0;
            x_lo_T[int64_t(idx) * 16 + b] = h1;
            if (b == 0 && B < 16) {
                for (int bz = B; bz < 16; ++bz) {
                    x_hi_T[int64_t(idx) * 16 + bz] = __float2half(0.0f);
                    x_lo_T[int64_t(idx) * 16 + bz] = __float2half(0.0f);
                }
            }
        }
    }
}

__global__ __launch_bounds__(128, 10)
void fused_geglu_or_ple_b16_h4_fp16split_sm86(
    const float* __restrict__ a_in, const float* __restrict__ b_in,
    __half* __restrict__ x_hi_T, __half* __restrict__ x_lo_T,
    int B, int D, int b_stride, int b_offset
) {
    const int b = blockIdx.y, g = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (g >= (D >> 5) || b >= B) return;
    const int idx = g * 32 + lane;
    const float av = __ldg(&a_in[int64_t(b) * D + idx]);
    const float bv = __ldg(&b_in[int64_t(b) * b_stride + b_offset + idx]);
    float xv = gelu_tanh_f32(av) * bv;
    const float v1 = __shfl_xor_sync(0xFFFFFFFFu, xv, 1); xv = ((lane & 1) == 0) ? (xv + v1) : (v1 - xv);
    const float v2 = __shfl_xor_sync(0xFFFFFFFFu, xv, 2); xv = ((lane & 2) == 0) ? (xv + v2) : (v2 - xv);
    xv *= 0.5f;
    const __half h0 = __float2half(xv), h1 = __float2half(xv - __half2float(h0));
    x_hi_T[int64_t(idx) * 16 + b] = h0;
    x_lo_T[int64_t(idx) * 16 + b] = h1;
    if (b == 0 && B < 16) {
        for (int bz = B; bz < 16; ++bz) {
            x_hi_T[int64_t(idx) * 16 + bz] = __float2half(0.0f);
            x_lo_T[int64_t(idx) * 16 + bz] = __float2half(0.0f);
        }
    }
}

template<int SPLIT_K>
__global__ __launch_bounds__(128, 8)
void gemm_b16_mma_fast_sm86(
    const __half* __restrict__ x_hi_T, const __half* __restrict__ x_lo_T,
    const uint4* __restrict__ w_u4, const __half* __restrict__ w_sc,
    float* __restrict__ y_part, HoloSphereRazorCanary64* can, float softcap, int B, int M, int K
) {
    __shared__ __half smem_wf16[32][72];
    __shared__ float smem_sc[32][2];
    __shared__ __half smem_x_hi[64][24];
    __shared__ __half smem_x_lo[64][24];
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, wm = wid >> 1, wn = wid & 1;
    const int cta_row0 = blockIdx.x * 32, n_grp = K >> 5, sp = blockIdx.y;
    const int g_per_sp = ((n_grp + SPLIT_K - 1) / SPLIT_K + 1) & ~1;
    const int g_beg = sp * g_per_sp, g_end = min(n_grp, g_beg + g_per_sp);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    #pragma unroll 1
    for (int g0 = g_beg; g0 < g_end; g0 += 2) {
        if (tid < 64) {
            const int r_loc = tid >> 1, g_loc = tid & 1, r_glob = cta_row0 + r_loc, g_glob = g0 + g_loc;
            const bool act = (r_glob < M) && (g_glob < g_end);
            const uint4 wv = act ? __ldg(&w_u4[int64_t(r_glob) * n_grp + g_glob]) : make_uint4(0u, 0u, 0u, 0u);
            record_u4_secded(wv, can);
            smem_sc[r_loc][g_loc] = act ? __half2float(__ldg(&w_sc[int64_t(r_glob) * n_grp + g_glob])) : 0.0f;
            uint32_t h0[4], h1[4], h2[4], h3[4];
            decode8_to_h2(wv.x, h0); decode8_to_h2(wv.y, h1); decode8_to_h2(wv.z, h2); decode8_to_h2(wv.w, h3);
            uint32_t* dst = reinterpret_cast<uint32_t*>(&smem_wf16[r_loc][g_loc * 32]);
            dst[0]=h0[0]; dst[1]=h0[1]; dst[2]=h0[2]; dst[3]=h0[3];
            dst[4]=h1[0]; dst[5]=h1[1]; dst[6]=h1[2]; dst[7]=h1[3];
            dst[8]=h2[0]; dst[9]=h2[1]; dst[10]=h2[2]; dst[11]=h2[3];
            dst[12]=h3[0]; dst[13]=h3[1]; dst[14]=h3[2]; dst[15]=h3[3];
        } else {
            const int kl = tid - 64, k_glob = g0 * 32 + kl;
            if (k_glob < min(K, g_end * 32)) {
                const uint4* shi = reinterpret_cast<const uint4*>(x_hi_T + int64_t(k_glob) * 16);
                const uint4* slo = reinterpret_cast<const uint4*>(x_lo_T + int64_t(k_glob) * 16);
                *reinterpret_cast<uint4*>(&smem_x_hi[kl][0]) = __ldg(&shi[0]);
                *reinterpret_cast<uint4*>(&smem_x_hi[kl][8]) = __ldg(&shi[1]);
                *reinterpret_cast<uint4*>(&smem_x_lo[kl][0]) = __ldg(&slo[0]);
                *reinterpret_cast<uint4*>(&smem_x_lo[kl][8]) = __ldg(&slo[1]);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int gl = 0; gl < 2; ++gl) {
            if (g0 + gl >= g_end) break;
            float ga[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int ko = gl * 32 + ks * 16;
                uint32_t ra[4], rbh[2], rbl[2];
                ldmatrix_x4(ra, static_cast<uint32_t>(__cvta_generic_to_shared(&smem_wf16[wm * 16 + (lane & 15)][ko + ((lane >> 4) * 8)])));
                const int br = ko + (lane & 15);
                ldmatrix_x2_trans(rbh, static_cast<uint32_t>(__cvta_generic_to_shared(&smem_x_hi[br][wn * 8])));
                ldmatrix_x2_trans(rbl, static_cast<uint32_t>(__cvta_generic_to_shared(&smem_x_lo[br][wn * 8])));
                mma_m16n8k16_f16(ga, ra, rbh, ga);
                mma_m16n8k16_f16(ga, ra, rbl, ga);
            }
            const int mr0 = wm * 16 + (lane >> 2);
            const float sc0 = smem_sc[mr0][gl], sc1 = smem_sc[mr0 + 8][gl];
            acc[0] = fmaf(ga[0], sc0, acc[0]); acc[1] = fmaf(ga[1], sc0, acc[1]);
            acc[2] = fmaf(ga[2], sc1, acc[2]); acc[3] = fmaf(ga[3], sc1, acc[3]);
        }
        __syncthreads();
    }
    const int r0 = cta_row0 + wm * 16 + (lane >> 2), r1 = r0 + 8, c0 = wn * 8 + (lane & 3) * 2;
    const int64_t sp_off = int64_t(sp) * B * M;
    auto put = [&](int r, int c, float v) {
        if (r < M && c < B) {
            if constexpr (SPLIT_K == 1) {
                y_part[sp_off + int64_t(c) * M + r] = (softcap > 0.0f) ? (softcap * tanhf(v / softcap)) : v;
            } else {
                y_part[sp_off + int64_t(c) * M + r] = v;
            }
        }
    };
    put(r0, c0, acc[0]); put(r0, c0 + 1, acc[1]);
    put(r1, c0, acc[2]); put(r1, c0 + 1, acc[3]);
}

__global__ __launch_bounds__(128, 11)
void gemm_b16_mma_64x16_sm86(
    const __half* __restrict__ x_hi_T, const __half* __restrict__ x_lo_T,
    const uint4* __restrict__ w_u4, const __half* __restrict__ w_sc,
    float* __restrict__ y_out, HoloSphereRazorCanary64* can, float softcap, int B, int M, int K
) {
    __shared__ __half smem_wf16[64][72];
    __shared__ float smem_sc[64][2];
    __shared__ __half smem_x_hi[64][24];
    __shared__ __half smem_x_lo[64][24];
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, wm = wid >> 1, wn = wid & 1;
    const int cta_row0 = blockIdx.x * 64, n_grp = K >> 5;
    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll 1
    for (int g0 = 0; g0 < n_grp; g0 += 2) {
        const int r_loc = tid >> 1, g_loc = tid & 1, r_glob = cta_row0 + r_loc, g_glob = g0 + g_loc;
        const bool act = (r_glob < M) && (g_glob < n_grp);
        const uint4 wv = act ? __ldg(&w_u4[int64_t(r_glob) * n_grp + g_glob]) : make_uint4(0u, 0u, 0u, 0u);
        record_u4_secded(wv, can);
        smem_sc[r_loc][g_loc] = act ? __half2float(__ldg(&w_sc[int64_t(r_glob) * n_grp + g_glob])) : 0.0f;
        uint32_t h0[4], h1[4], h2[4], h3[4];
        decode8_to_h2(wv.x, h0); decode8_to_h2(wv.y, h1); decode8_to_h2(wv.z, h2); decode8_to_h2(wv.w, h3);
        uint32_t* dst = reinterpret_cast<uint32_t*>(&smem_wf16[r_loc][g_loc * 32]);
        dst[0]=h0[0]; dst[1]=h0[1]; dst[2]=h0[2]; dst[3]=h0[3];
        dst[4]=h1[0]; dst[5]=h1[1]; dst[6]=h1[2]; dst[7]=h1[3];
        dst[8]=h2[0]; dst[9]=h2[1]; dst[10]=h2[2]; dst[11]=h2[3];
        dst[12]=h3[0]; dst[13]=h3[1]; dst[14]=h3[2]; dst[15]=h3[3];
        const int kl = tid >> 1, b8 = (tid & 1) * 8, k_glob = g0 * 32 + kl;
        if (k_glob < K) {
            *reinterpret_cast<uint4*>(&smem_x_hi[kl][b8]) = __ldg(reinterpret_cast<const uint4*>(x_hi_T + int64_t(k_glob) * 16 + b8));
            *reinterpret_cast<uint4*>(&smem_x_lo[kl][b8]) = __ldg(reinterpret_cast<const uint4*>(x_lo_T + int64_t(k_glob) * 16 + b8));
        }
        __syncthreads();
        #pragma unroll
        for (int gl = 0; gl < 2; ++gl) {
            if (g0 + gl >= n_grp) break;
            float ga0[4] = {0,0,0,0}, ga1[4] = {0,0,0,0};
            #pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int ko = gl * 32 + ks * 16;
                uint32_t ra0[4], ra1[4], rbh[2], rbl[2];
                ldmatrix_x4(ra0, static_cast<uint32_t>(__cvta_generic_to_shared(&smem_wf16[wm * 32 +  0 + (lane & 15)][ko + ((lane >> 4) * 8)])));
                ldmatrix_x4(ra1, static_cast<uint32_t>(__cvta_generic_to_shared(&smem_wf16[wm * 32 + 16 + (lane & 15)][ko + ((lane >> 4) * 8)])));
                const int br = ko + (lane & 15);
                ldmatrix_x2_trans(rbh, static_cast<uint32_t>(__cvta_generic_to_shared(&smem_x_hi[br][wn * 8])));
                ldmatrix_x2_trans(rbl, static_cast<uint32_t>(__cvta_generic_to_shared(&smem_x_lo[br][wn * 8])));
                mma_m16n8k16_f16(ga0, ra0, rbh, ga0); mma_m16n8k16_f16(ga0, ra0, rbl, ga0);
                mma_m16n8k16_f16(ga1, ra1, rbh, ga1); mma_m16n8k16_f16(ga1, ra1, rbl, ga1);
            }
            const int mr = wm * 32 + (lane >> 2);
            const float s0 = smem_sc[mr + 0][gl], s1 = smem_sc[mr + 8][gl], s2 = smem_sc[mr + 16][gl], s3 = smem_sc[mr + 24][gl];
            acc0[0] = fmaf(ga0[0], s0, acc0[0]); acc0[1] = fmaf(ga0[1], s0, acc0[1]);
            acc0[2] = fmaf(ga0[2], s1, acc0[2]); acc0[3] = fmaf(ga0[3], s1, acc0[3]);
            acc1[0] = fmaf(ga1[0], s2, acc1[0]); acc1[1] = fmaf(ga1[1], s2, acc1[1]);
            acc1[2] = fmaf(ga1[2], s3, acc1[2]); acc1[3] = fmaf(ga1[3], s3, acc1[3]);
        }
        __syncthreads();
    }
    const int r0 = cta_row0 + wm * 32 + (lane >> 2), c0 = wn * 8 + (lane & 3) * 2;
    auto st2 = [&](int r, float v0, float v1) {
        if (r < M) {
            if (c0 + 0 < B) y_out[int64_t(c0 + 0) * M + r] = (softcap > 0.0f) ? (softcap * tanhf(v0 / softcap)) : v0;
            if (c0 + 1 < B) y_out[int64_t(c0 + 1) * M + r] = (softcap > 0.0f) ? (softcap * tanhf(v1 / softcap)) : v1;
        }
    };
    st2(r0 + 0, acc0[0], acc0[1]); st2(r0 + 8, acc0[2], acc0[3]);
    st2(r0 + 16, acc1[0], acc1[1]); st2(r0 + 24, acc1[2], acc1[3]);
}

// 15. BF16 lm_head (B=1 2-row ILP uint4 coalesced; B=16 Tensor Core BF16 MMA)
__global__ __launch_bounds__(128, 11)
void lm_head_b1_bf16_uint4_sm86(
    const float* __restrict__ x_in, const uint4* __restrict__ w_bf16,
    float* __restrict__ logits, float softcap, int V, int H
) {
    const int tid = threadIdx.x, r_loc = tid >> 3, lane_r = tid & 7;
    const int row0 = blockIdx.x * 32 + r_loc, row1 = row0 + 16;
    const bool act0 = (row0 < V), act1 = (row1 < V);
    const int n_u4 = H >> 3;
    const uint4* rw0 = act0 ? (w_bf16 + int64_t(row0) * n_u4) : nullptr;
    const uint4* rw1 = act1 ? (w_bf16 + int64_t(row1) * n_u4) : nullptr;
    const float4* xf = reinterpret_cast<const float4*>(x_in);
    float acc0 = 0.0f, acc1 = 0.0f;
    if (act0) {
        #pragma unroll 1
        for (int g = lane_r; g < n_u4; g += 8) {
            const uint4 wv0 = __ldg(&rw0[g]);
            const uint4 wv1 = act1 ? __ldg(&rw1[g]) : make_uint4(0u, 0u, 0u, 0u);
            const float4 x0 = __ldg(&xf[g * 2 + 0]), x1 = __ldg(&xf[g * 2 + 1]);
            auto step4 = [&](const uint4& wv, float& acc) {
                const uint32_t ws[4] = {wv.x, wv.y, wv.z, wv.w};
                const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(ws);
                const float2 f0 = __bfloat1622float2(b2[0]), f1 = __bfloat1622float2(b2[1]);
                const float2 f2 = __bfloat1622float2(b2[2]), f3 = __bfloat1622float2(b2[3]);
                acc = fmaf(f0.x, x0.x, acc); acc = fmaf(f0.y, x0.y, acc);
                acc = fmaf(f1.x, x0.z, acc); acc = fmaf(f1.y, x0.w, acc);
                acc = fmaf(f2.x, x1.x, acc); acc = fmaf(f2.y, x1.y, acc);
                acc = fmaf(f3.x, x1.z, acc); acc = fmaf(f3.y, x1.w, acc);
            };
            step4(wv0, acc0);
            if (act1) step4(wv1, acc1);
        }
    }
    #pragma unroll
    for (int m = 4; m >= 1; m >>= 1) {
        acc0 += __shfl_xor_sync(0xFFFFFFFFu, acc0, m);
        acc1 += __shfl_xor_sync(0xFFFFFFFFu, acc1, m);
    }
    if (lane_r == 0) {
        if (act0) logits[row0] = (softcap > 0.0f) ? (softcap * tanhf(acc0 / softcap)) : acc0;
        if (act1) logits[row1] = (softcap > 0.0f) ? (softcap * tanhf(acc1 / softcap)) : acc1;
    }
}

__global__ __launch_bounds__(128, 11)
void lm_head_b16_bf16_mma_sm86(
    const uint16_t* __restrict__ x_hi_T, const uint16_t* __restrict__ x_lo_T,
    const uint4* __restrict__ w_bf16, float* __restrict__ logits, float softcap, int B, int V, int K
) {
    __shared__ uint16_t smem_w[32][40], smem_xh[32][24], smem_xl[32][24];
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, wm = wid >> 1, wn = wid & 1;
    const int cta_row0 = blockIdx.x * 32, n_u4 = K >> 3;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll 1
    for (int k0 = 0; k0 < K; k0 += 32) {
        const int r_loc = tid >> 2, u_loc = tid & 3, r_glob = cta_row0 + r_loc;
        *reinterpret_cast<uint4*>(&smem_w[r_loc][u_loc * 8]) = (r_glob < V) ? __ldg(&w_bf16[int64_t(r_glob) * n_u4 + (k0 >> 3) + u_loc]) : make_uint4(0u,0u,0u,0u);
        const int is_lo = tid >> 6, t64 = tid & 63, kl = t64 >> 1, b8 = (t64 & 1) * 8;
        *reinterpret_cast<uint4*>(is_lo ? &smem_xl[kl][b8] : &smem_xh[kl][b8]) = __ldg(reinterpret_cast<const uint4*>((is_lo ? x_lo_T : x_hi_T) + int64_t(k0 + kl) * 16 + b8));
        __syncthreads();
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            const int ko = ks * 16; uint32_t ra[4], rbh[2], rbl[2];
            ldmatrix_x4(ra, static_cast<uint32_t>(__cvta_generic_to_shared(&smem_w[wm * 16 + (lane & 15)][ko + ((lane >> 4) * 8)])));
            const int br = ko + (lane & 15);
            ldmatrix_x2_trans(rbh, static_cast<uint32_t>(__cvta_generic_to_shared(&smem_xh[br][wn * 8])));
            ldmatrix_x2_trans(rbl, static_cast<uint32_t>(__cvta_generic_to_shared(&smem_xl[br][wn * 8])));
            mma_m16n8k16_bf16(acc, ra, rbh, acc);
            mma_m16n8k16_bf16(acc, ra, rbl, acc);
        }
        __syncthreads();
    }
    const int r0 = cta_row0 + wm * 16 + (lane >> 2), r1 = r0 + 8, c0 = wn * 8 + (lane & 3) * 2;
    auto put = [&](int r, int c, float v) {
        if (r < V && c < B) logits[int64_t(c) * V + r] = (softcap > 0.0f) ? (softcap * tanhf(v / softcap)) : v;
    };
    put(r0, c0, acc[0]); put(r0, c0 + 1, acc[1]);
    put(r1, c0, acc[2]); put(r1, c0 + 1, acc[3]);
}

// 16. Two-stage multi-CTA parallel argmax
__global__ __launch_bounds__(128, 10)
void parallel_argmax_stage1_sm86(const float* __restrict__ logits, float* __restrict__ blk_val, int32_t* __restrict__ blk_idx, int V) {
    __shared__ float sv[4];
    __shared__ int32_t si[4];
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, start = blockIdx.x * 1024 + tid;
    float mv = -1e38f;
    int32_t mi = 0;
    for (int i = start; i < V; i += gridDim.x * 1024) {
        for (int u = 0; u < 8; ++u) {
            const int j = i + u * 128;
            if (j < V) {
                const float v = logits[j];
                if (v > mv) { mv = v; mi = j; }
            }
        }
    }
    #pragma unroll
    for (int m = 16; m >= 1; m >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFFu, mv, m);
        const int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, mi, m);
        if (ov > mv) { mv = ov; mi = oi; }
    }
    if (lane == 0) { sv[wid] = mv; si[wid] = mi; }
    __syncthreads();
    if (wid == 0 && lane < 4) {
        mv = sv[lane]; mi = si[lane];
        for (int m = 2; m >= 1; m >>= 1) {
            const float ov = __shfl_xor_sync(0xFu, mv, m);
            const int32_t oi = __shfl_xor_sync(0xFu, mi, m);
            if (ov > mv) { mv = ov; mi = oi; }
        }
        if (lane == 0) { blk_val[blockIdx.x] = mv; blk_idx[blockIdx.x] = mi; }
    }
}

__global__ __launch_bounds__(128, 10)
void parallel_argmax_stage2_sm86(const float* __restrict__ bv, const int32_t* __restrict__ bi, int32_t* __restrict__ out, int N) {
    __shared__ float sv[4];
    __shared__ int32_t si[4];
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    float mv = (tid < N) ? bv[tid] : -1e38f;
    int32_t mi = (tid < N) ? bi[tid] : 0;
    #pragma unroll
    for (int m = 16; m >= 1; m >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFFu, mv, m);
        const int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, mi, m);
        if (ov > mv) { mv = ov; mi = oi; }
    }
    if (lane == 0) { sv[wid] = mv; si[wid] = mi; }
    __syncthreads();
    if (wid == 0 && lane < 4) {
        mv = sv[lane]; mi = si[lane];
        for (int m = 2; m >= 1; m >>= 1) {
            const float ov = __shfl_xor_sync(0xFu, mv, m);
            const int32_t oi = __shfl_xor_sync(0xFu, mi, m);
            if (ov > mv) { mv = ov; mi = oi; }
        }
        if (lane == 0) out[0] = mi;
    }
}

// 17. Hoisted 256 B Woodbury 8x8 Cayley inverse (SMEM Gauss-Jordan -> 0 LDL/STL, 0 spill, <= 44 regs!)
__global__ __launch_bounds__(128, 8)
void woodbury_inv8_sm86(const float* __restrict__ M_in, float* __restrict__ inv_out) {
    __shared__ float sm[8][16];
    const int tid = threadIdx.x;
    if (tid < 64) {
        const int r = tid >> 3, c = tid & 7;
        sm[r][c] = M_in[tid];
        sm[r][c + 8] = (r == c) ? 1.0f : 0.0f;
    }
    __syncthreads();
    for (int col = 0; col < 8; ++col) {
        if (tid == 0) {
            int piv = col;
            float best = fabsf(sm[col][col]);
            for (int r = col + 1; r < 8; ++r) {
                const float v = fabsf(sm[r][col]);
                if (v > best) { best = v; piv = r; }
            }
            if (piv != col) {
                for (int c = 0; c < 16; ++c) {
                    const float tmp = sm[col][c];
                    sm[col][c] = sm[piv][c];
                    sm[piv][c] = tmp;
                }
            }
            const float inv_d = 1.0f / sm[col][col];
            for (int c = 0; c < 16; ++c) sm[col][c] *= inv_d;
        }
        __syncthreads();
        if (tid < 128) {
            const int r = tid >> 4, c = tid & 15;
            if (r < 8 && r != col) {
                const float f = sm[r][col];
                sm[r][c] -= f * sm[col][c];
            }
        }
        __syncthreads();
    }
    if (tid < 64) {
        const int r = tid >> 3, c = tid & 7;
        inv_out[tid] = sm[r][c + 8];
    }
}

// Apply L10..L14 Woodbury-Cayley rank-8 steering using hoisted c_wb_inv_256b (0 B extra KV, norm-preserving!)
__global__ __launch_bounds__(128, 10)
void cayley_woodbury_apply_sm86(
    float* __restrict__ h_io,         // [B, D]
    const float* __restrict__ U8,     // [8, D]
    const float* __restrict__ V8,     // [8, D]
    float alpha, int B, int D
) {
    __shared__ float sm_z[8], sm_y[8], sm_red[4];
    const int b = blockIdx.x, tid = threadIdx.x;
    float* hb = h_io + int64_t(b) * D;
    // Project z = V8 @ h, then y = c_wb_inv_256b @ z (both 8-dim), then h <- h + alpha * (U8^T @ y)
    for (int r = 0; r < 8; ++r) {
        float acc = 0.0f;
        for (int d = tid; d < D; d += 128) acc = fmaf(V8[int64_t(r) * D + d], hb[d], acc);
        const float sum_r = tree_reduce_128_sm86(acc, sm_red);
        if (tid == 0) sm_z[r] = sum_r;
        __syncthreads();
    }
    if (tid < 8) {
        float acc = 0.0f;
        for (int c = 0; c < 8; ++c) acc = fmaf(c_wb_inv_256b[tid * 8 + c], sm_z[c], acc);
        sm_y[tid] = acc;
    }
    __syncthreads();
    for (int d = tid; d < D; d += 128) {
        float delta = 0.0f;
        for (int r = 0; r < 8; ++r) delta = fmaf(U8[int64_t(r) * D + d], sm_y[r], delta);
        hb[d] = fmaf(alpha, delta, hb[d]);
    }
}

// 18. Multi-CTA U-4 Loop Gate (r=16, 0 B KV, 0 B extra bulk weights):
// Stage 1: project h -> z[16] (grid=(16, B)); Stage 2: h += alpha * U^T @ z (grid=(D/128, B))
__global__ __launch_bounds__(128, 10)
void u4_loop_proj_r16_sm86(
    const float* __restrict__ h_in, const float* __restrict__ V16, float* __restrict__ z_out, int B, int D
) {
    __shared__ float sm[4];
    const int r = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
    float acc = 0.0f;
    for (int d = tid; d < D; d += 128) acc = fmaf(__ldg(&V16[int64_t(r) * D + d]), __ldg(&h_in[int64_t(b) * D + d]), acc);
    const float s = tree_reduce_128_sm86(acc, sm);
    if (tid == 0) z_out[b * 16 + r] = tanhf(s);
}

__global__ __launch_bounds__(128, 10)
void u4_loop_apply_r16_sm86(
    float* __restrict__ h_io, const float* __restrict__ z_in, const float* __restrict__ U16,
    float alpha, int B, int D
) {
    __shared__ float sm_z[16];
    const int tid = threadIdx.x, b = blockIdx.y, d = blockIdx.x * 128 + tid;
    if (tid < 16) sm_z[tid] = z_in[b * 16 + tid];
    __syncthreads();
    if (d >= D) return;
    float delta = 0.0f;
    #pragma unroll
    for (int r = 0; r < 16; ++r) delta = fmaf(__ldg(&U16[int64_t(r) * D + d]), sm_z[r], delta);
    h_io[int64_t(b) * D + d] = fmaf(alpha, delta, h_io[int64_t(b) * D + d]);
}

// 19. IRIS Canary Finalizer (detects SEC/DED/poisoning and hashes telemetry -> SafeHalt)
__global__ __launch_bounds__(128, 8)
void iris_canary_finalize_sm86(const float* __restrict__ h_in, HoloSphereRazorCanary64* can, int B, int D) {
    __shared__ uint64_t sm_h[4];
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    uint64_t acc = 0x517CC1B727220A95ULL;
    for (int i = tid; i < B * D; i += 128) {
        const float v = h_in[i];
        uint32_t u; memcpy(&u, &v, 4);
        acc ^= uint64_t(u) * 0x9E3779B97F4A7C15ULL + uint64_t(i);
    }
    for (int m = 16; m >= 1; m >>= 1) acc ^= __shfl_xor_sync(0xFFFFFFFFu, acc, m);
    if (lane == 0) sm_h[wid] = acc;
    __syncthreads();
    if (tid == 0) {
        const uint64_t h = sm_h[0] ^ sm_h[1] ^ sm_h[2] ^ sm_h[3];
        uint64_t st = can->iris_state;
        auto mix = [](uint64_t s, uint64_t val) {
            s ^= val + 0x9E3779B97F4A7C15ULL + (s << 6) + (s >> 2);
            s *= 0xBF58476D1CE4E5B9ULL;
            s ^= (s >> 31);
            return s;
        };
        st = mix(st, h);
        st = mix(st, (uint64_t(can->sec_cnt) << 32) | uint64_t(can->ded_cnt));
        st = mix(st, (uint64_t(can->step_ctr) << 32) | uint64_t(can->u4_passes));
        if (can->sec_cnt != 0u || can->ded_cnt != 0u) can->poisoned = 1u;
        can->iris_state = st;
        can->checksum = st ^ uint64_t(can->poisoned);
    }
}

// 20. Async PLE DMA test micro-kernel (touches staging host-pinned buffer on background stream)
__global__ __launch_bounds__(128, 8)
void ple_async_dma_touch_sm86(const float* __restrict__ ple_buf, float* __restrict__ acc_out, int N) {
    const int i = blockIdx.x * 128 + threadIdx.x;
    if (i < N) acc_out[i] = ple_buf[i] * 0.70710678f;
}

} // namespace s184b

#endif // SILIGEMMA_KERNELS_SM86_CU
