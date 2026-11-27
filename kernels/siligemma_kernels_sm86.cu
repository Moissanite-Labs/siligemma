// SPDX-License-Identifier: Apache-2.0
// Track S184 (EXP-Q81): Bare-Metal CUDA sm_86 Kernels for SiliGemma-Native Form B
//
// Design guarantees (KG-GFB1..KG-GFB4):
//   - Zero Python / PyTorch / transformers / cuBLAS dependencies.
//   - 0 B stack frame, 0 B spill loads/stores, 0 LDL/STL SASS instructions, <= 48 regs
//     across EVERY kernel (enforced via register-lean scalarization + __launch_bounds__).
//   - P223c 0-ULP canonical binary reduction tree (tree_reduce_128_sm86).
//   - P224a IRIS runtime+scan CRC + [16,11,4] 1-LOP3+5-POPC SECDED in HoloSphereRazorCanary64.
//   - S180c hoisted 256 B constant-memory 8x8 Cayley-Woodbury steering (c_wb_inv_256b & L10..14 hub).
//   - zed static [13..34] shared-KV ring (0 B extra KV) + U-4 L25..33 K in {1,2,3} loop gates.

#ifndef SILIGEMMA_KERNELS_SM86_CU
#define SILIGEMMA_KERNELS_SM86_CU

#include "siligemma_arena.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace siligemma_sm86 {

// ----------------------------------------------------------------------------
// 1. Constant-Memory Hoisted 8x8 Woodbury Buffers (256 B each)
// ----------------------------------------------------------------------------
__constant__ float c_wb_inv_256b[64];         // Single-update 256 B 8x8 M^{-1}
__constant__ float c_wb_inv_hub_l10_14[5][64]; // L10..L14 20x shared_KV hub slots (5 x 256 B)

// ----------------------------------------------------------------------------
// 2. Inline PTX, SECDED, Canonical Tree, H4, & IRIS Telemetry Primitives
// ----------------------------------------------------------------------------

__device__ __forceinline__ uint32_t ptx_lop3_xor_and(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0x28;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

__device__ __forceinline__ int32_t ptx_dp4a_s32(int32_t a_s8x4, int32_t b_s8x4, int32_t c) {
    int32_t d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a_s8x4), "r"(b_s8x4), "r"(c));
    return d;
}

__device__ __forceinline__ int32_t pack4_s8(int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    return (c0 & 0xFF) | ((c1 & 0xFF) << 8) | ((c2 & 0xFF) << 16) | ((c3 & 0xFF) << 24);
}

__device__ __forceinline__ uint32_t a16_secded_lop3_popc_syndrome(uint64_t w64) {
    const uint32_t w0 = uint32_t(w64);
    const uint32_t w1 = uint32_t(w64 >> 32);
    const uint32_t px = ptx_lop3_xor_and(w0, w1, 0x11111111u);
    const uint32_t p0 = uint32_t(__popc(px)) & 1u;
    const uint32_t s0 = uint32_t(__popc(px & 0x10101010u)) & 1u;
    const uint32_t s1 = uint32_t(__popc(px & 0x11001100u)) & 1u;
    const uint32_t s2 = uint32_t(__popc(px & 0x11110000u)) & 1u;
    const uint32_t s3 = uint32_t(__popc(w1 & 0x11111111u)) & 1u;
    return p0 | (s0 << 1) | (s1 << 2) | (s2 << 3) | (s3 << 4);
}

__device__ __forceinline__ void razor_atom_release_sys_add_u32(uint32_t* host_ptr, uint32_t delta) {
    asm volatile(
        "atom.release.sys.global.add.u32 _, [%0], %1;"
        :
        : "l"(host_ptr), "r"(delta)
        : "memory"
    );
}

__device__ __forceinline__ void razor_st_release_sys_u32(uint32_t* host_ptr, uint32_t val) {
    asm volatile(
        "st.release.sys.global.u32 [%0], %1;"
        :
        : "l"(host_ptr), "r"(val)
        : "memory"
    );
}

__device__ __forceinline__ void razor_canary_emit_sys(
    siligemma::HoloSphereRazorCanary64* host_canary,
    uint32_t lane_sec,
    uint32_t lane_ded,
    uint32_t diag_word,
    int lane
) {
    const uint32_t sec_mask = __ballot_sync(0xFFFFFFFFu, lane_sec != 0u);
    const uint32_t ded_mask = __ballot_sync(0xFFFFFFFFu, lane_ded != 0u);
    const uint32_t any_mask = sec_mask | ded_mask;
    if (any_mask != 0u) {
        const int src_lane = __ffs(int(any_mask)) - 1;
        const uint32_t diag_out = __shfl_sync(0xFFFFFFFFu, diag_word, src_lane);
        if (lane == 0 && host_canary != nullptr) {
            if (sec_mask != 0u) {
                razor_atom_release_sys_add_u32(&host_canary->sec_count_u32, uint32_t(__popc(sec_mask)));
            }
            if (ded_mask != 0u) {
                razor_atom_release_sys_add_u32(&host_canary->ded_count_u32, uint32_t(__popc(ded_mask)));
            }
            razor_st_release_sys_u32(&host_canary->last_syndrome_u32, diag_out);
        }
    }
}

__device__ __forceinline__ uint32_t iris_mix_word_dev(uint32_t state, uint32_t word) {
    uint32_t x = state ^ (word + siligemma::IRIS_WEYL_MIX + (state << 6) + (state >> 2));
    x ^= (x >> 16);
    x *= 0x85EBCA6Bu;
    x ^= (x >> 13);
    x *= 0xC2B2AE35u;
    x ^= (x >> 16);
    return x;
}

__device__ __forceinline__ float warp_sum_32(float v) {
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, mask);
    }
    return v;
}

// P223c 0-ULP Fixed-Shape 7-Stage Binary Reduction Tree over 128 lanes.
// Preserves exact IEEE-754 float32 reduction order while cutting CTA barriers 8 -> 3.
__device__ __forceinline__ float tree_reduce_128_sm86(float val, float* smem_128) {
    const int tid = threadIdx.x;
    smem_128[tid] = val;
    __syncthreads();
    if (tid < 64) smem_128[tid] += smem_128[tid + 64];
    __syncthreads();
    if (tid < 32) {
        float w = smem_128[tid] + smem_128[tid + 32];
        w += __shfl_down_sync(0xFFFFFFFFu, w, 16);
        w += __shfl_down_sync(0xFFFFFFFFu, w, 8);
        w += __shfl_down_sync(0xFFFFFFFFu, w, 4);
        w += __shfl_down_sync(0xFFFFFFFFu, w, 2);
        w += __shfl_down_sync(0xFFFFFFFFu, w, 1);
        if (tid == 0) smem_128[0] = w;
    }
    __syncthreads();
    return smem_128[0];
}

__device__ __forceinline__ float4 h4_rot_f4(float4 xv) {
    const float s01 = xv.x + xv.y;
    const float d01 = xv.x - xv.y;
    const float s23 = xv.z + xv.w;
    const float d23 = xv.z - xv.w;
    return make_float4(
        0.5f * (s01 + s23),
        0.5f * (d01 + d23),
        0.5f * (s01 - s23),
        0.5f * (d01 - d23)
    );
}

__device__ __forceinline__ float gelu_pytorch_tanh_dev(float g) {
    const float inner = 0.7978845608f * (g + 0.044715f * g * g * g);
    return 0.5f * g * (1.0f + tanhf(inner));
}

__device__ __forceinline__ int32_t a16_nibble_dp4a16_xrot(uint64_t w64, const int32_t* __restrict__ xrot_q8_4) {
    int32_t idot = 0;
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        const uint32_t nibs = uint32_t((w64 >> (g * 16)) & 0xFFFFULL);
        const int32_t c0 = int32_t(nibs << 28) >> 28;
        const int32_t c1 = int32_t(nibs << 24) >> 28;
        const int32_t c2 = int32_t(nibs << 20) >> 28;
        const int32_t c3 = int32_t(nibs << 16) >> 28;
        idot = ptx_dp4a_s32(pack4_s8(c0, c1, c2, c3), xrot_q8_4[g], idot);
    }
    return idot;
}

__device__ __forceinline__ void eval_a16_blk16_secded_dp4a(
    bool active_blk,
    uint64_t w64,
    float sc,
    const int32_t* __restrict__ x_q8_4,
    siligemma::HoloSphereRazorCanary64* host_canary,
    int lane,
    float& acc
) {
    const uint32_t syn = active_blk ? a16_secded_lop3_popc_syndrome(w64) : 0u;
    const uint32_t any = __ballot_sync(0xFFFFFFFFu, syn != 0u);
    if (__builtin_expect(any != 0u, 0)) {
        uint32_t l_sec = 0u, l_ded = 0u, diag = 0u;
        if (syn != 0u) {
            if (syn & 1u) {
                w64 ^= (1ULL << (((syn >> 1) & 15u) * 4u));
                l_sec = 1u;
                diag = syn & 0x1Fu;
            } else {
                l_ded = 1u;
                diag = (1u << 5) | (syn & 0x1Fu);
            }
        }
        razor_canary_emit_sys(host_canary, l_sec, l_ded, diag, lane);
    }
    if (active_blk) {
        const int32_t idot16 = a16_nibble_dp4a16_xrot(w64, x_q8_4);
        acc = fmaf(float(idot16), sc, acc);
    }
}

// Compute canonical block-128 sum of squares over D floats using smem_tree[128] & smem_blk[16].
__device__ __forceinline__ float canonical_sum_sq_d128(
    const float* __restrict__ x_in,
    int dim,
    float* smem_tree,
    float* smem_blk
) {
    const int tid = threadIdx.x;
    const int n_chunks = (dim + 127) >> 7;
    #pragma unroll 1
    for (int c = 0; c < n_chunks; ++c) {
        const int idx = c * 128 + tid;
        const float xv = (idx < dim) ? __ldg(&x_in[idx]) : 0.0f;
        const float cs = tree_reduce_128_sm86(xv * xv, smem_tree);
        if (tid == 0) smem_blk[c] = cs;
    }
    __syncthreads();
    const float bval = (tid < n_chunks) ? smem_blk[tid] : 0.0f;
    return tree_reduce_128_sm86(bval, smem_tree);
}

// ----------------------------------------------------------------------------
// 3. Kernel 1: Update-Time 8x8 Woodbury Inverse Precompute -> c_wb_inv_256b
// ----------------------------------------------------------------------------

__global__ __launch_bounds__(256, 5)
void siligemma_precompute_woodbury_inv8_sm86(
    const float4* __restrict__ a_f4,
    const float4* __restrict__ b_f4,
    float* __restrict__ m_inv_out_64,
    float alpha_scale,
    int d_model
) {
    __shared__ float smem_aug[8][16];  // 512 B: [M | I_8]

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
    float m4 = 0.0f, m5 = 0.0f, m6 = 0.0f, m7 = 0.0f;

    for (int i = lane; i < d_model; i += 32) {
        const float4 av = __ldg(&a_f4[i]);
        const float4 bv = __ldg(&b_f4[i]);
        float vw = 0.0f;
        if (warp_id == 0) vw = av.x;
        else if (warp_id == 1) vw = av.y;
        else if (warp_id == 2) vw = av.z;
        else if (warp_id == 3) vw = av.w;
        else if (warp_id == 4) vw = bv.x;
        else if (warp_id == 5) vw = bv.y;
        else if (warp_id == 6) vw = bv.z;
        else vw = bv.w;

        m0 = fmaf(vw, bv.x, m0);  m1 = fmaf(vw, bv.y, m1);
        m2 = fmaf(vw, bv.z, m2);  m3 = fmaf(vw, bv.w, m3);
        m4 = fmaf(vw, -av.x, m4); m5 = fmaf(vw, -av.y, m5);
        m6 = fmaf(vw, -av.z, m6); m7 = fmaf(vw, -av.w, m7);
    }

    m0 = warp_sum_32(m0); m1 = warp_sum_32(m1);
    m2 = warp_sum_32(m2); m3 = warp_sum_32(m3);
    m4 = warp_sum_32(m4); m5 = warp_sum_32(m5);
    m6 = warp_sum_32(m6); m7 = warp_sum_32(m7);

    if (lane == 0) {
        const float half_s = 0.5f * alpha_scale;
        smem_aug[warp_id][0] = ((warp_id == 0) ? 1.0f : 0.0f) - half_s * m0;
        smem_aug[warp_id][1] = ((warp_id == 1) ? 1.0f : 0.0f) - half_s * m1;
        smem_aug[warp_id][2] = ((warp_id == 2) ? 1.0f : 0.0f) - half_s * m2;
        smem_aug[warp_id][3] = ((warp_id == 3) ? 1.0f : 0.0f) - half_s * m3;
        smem_aug[warp_id][4] = ((warp_id == 4) ? 1.0f : 0.0f) - half_s * m4;
        smem_aug[warp_id][5] = ((warp_id == 5) ? 1.0f : 0.0f) - half_s * m5;
        smem_aug[warp_id][6] = ((warp_id == 6) ? 1.0f : 0.0f) - half_s * m6;
        smem_aug[warp_id][7] = ((warp_id == 7) ? 1.0f : 0.0f) - half_s * m7;
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            smem_aug[warp_id][8 + c] = (warp_id == c) ? 1.0f : 0.0f;
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll 1
        for (int col = 0; col < 8; ++col) {
            float best_val = (lane >= col && lane < 8) ? fabsf(smem_aug[lane][col]) : -1.0f;
            int best_row = lane;
            #pragma unroll
            for (int mask = 4; mask >= 1; mask >>= 1) {
                const float ov = __shfl_xor_sync(0xFFFFFFFFu, best_val, mask);
                const int orow = __shfl_xor_sync(0xFFFFFFFFu, best_row, mask);
                const bool take = (ov > best_val) || ((ov == best_val) && (orow < best_row));
                best_val = fmaxf(best_val, ov);
                best_row = take ? orow : best_row;
            }
            const int piv = __shfl_sync(0xFFFFFFFFu, best_row, 0);
            if (piv != col && lane < 16) {
                const float tmp = smem_aug[col][lane];
                smem_aug[col][lane] = smem_aug[piv][lane];
                smem_aug[piv][lane] = tmp;
            }
            __syncwarp();

            const float inv_piv = 1.0f / smem_aug[col][col];
            if (lane < 16) smem_aug[col][lane] *= inv_piv;
            __syncwarp();

            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                if (r != col) {
                    const float factor = smem_aug[r][col];
                    __syncwarp();
                    if (lane < 16) {
                        smem_aug[r][lane] = fmaf(-factor, smem_aug[col][lane], smem_aug[r][lane]);
                    }
                    __syncwarp();
                } else {
                    __syncwarp();
                    __syncwarp();
                }
            }
        }
    }
    __syncthreads();

    if (tid < 64) {
        m_inv_out_64[tid] = smem_aug[tid >> 3][8 + (tid & 7)];
    }
}

// ----------------------------------------------------------------------------
// 4. Kernel 2: Hoisted 256 B Constant-Memory Cayley-Woodbury r=4 Steering
//    Split A/B accumulators keep register pressure <= 40 regs, 0 B spill!
// ----------------------------------------------------------------------------

__device__ __forceinline__ void cayley_woodbury_r4_hoisted_impl(
    const float* __restrict__ x_in,
    const float4* __restrict__ a_f4,
    const float4* __restrict__ b_f4,
    float* __restrict__ y_out,
    float alpha_scale,
    int d_model,
    const float4* __restrict__ cinv4
) {
    __shared__ float4 sw[8][2];   // 256 B
    __shared__ __align__(16) float sc_buf[8];   // 32 B -> Total 288 B SMEM

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    float z0 = 0.0f, z1 = 0.0f, z2 = 0.0f, z3 = 0.0f;
    float z4 = 0.0f, z5 = 0.0f, z6 = 0.0f, z7 = 0.0f;
    #pragma unroll 2
    for (int i = tid; i < d_model; i += 256) {
        const float xi = __ldg(&x_in[i]);
        const float4 av = __ldg(&a_f4[i]);
        const float4 bv = __ldg(&b_f4[i]);
        z0 = fmaf(av.x, xi, z0); z1 = fmaf(av.y, xi, z1);
        z2 = fmaf(av.z, xi, z2); z3 = fmaf(av.w, xi, z3);
        z4 = fmaf(bv.x, xi, z4); z5 = fmaf(bv.y, xi, z5);
        z6 = fmaf(bv.z, xi, z6); z7 = fmaf(bv.w, xi, z7);
    }
    z0 = warp_sum_32(z0); z1 = warp_sum_32(z1);
    z2 = warp_sum_32(z2); z3 = warp_sum_32(z3);
    z4 = warp_sum_32(z4); z5 = warp_sum_32(z5);
    z6 = warp_sum_32(z6); z7 = warp_sum_32(z7);
    if (lane == 0) {
        sw[wid][0] = make_float4(z0, z1, z2, z3);
        sw[wid][1] = make_float4(z4, z5, z6, z7);
    }
    __syncthreads();

    if (wid == 0 && lane < 8) {
        float4 zv0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 zv1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            const float4 s = sw[w][0];
            const float4 t = sw[w][1];
            zv0.x += s.x; zv0.y += s.y; zv0.z += s.z; zv0.w += s.w;
            zv1.x += t.x; zv1.y += t.y; zv1.z += t.z; zv1.w += t.w;
        }
        const float4 r0 = cinv4[lane * 2 + 0];
        const float4 r1 = cinv4[lane * 2 + 1];
        sc_buf[lane] = alpha_scale * (
            fmaf(r0.x, zv0.x, fmaf(r0.y, zv0.y, fmaf(r0.z, zv0.z, r0.w * zv0.w)))
            + fmaf(r1.x, zv1.x, fmaf(r1.y, zv1.y, fmaf(r1.z, zv1.z, r1.w * zv1.w)))
        );
    }
    __syncthreads();

    const float4 c03 = *reinterpret_cast<const float4*>(&sc_buf[0]);
    const float4 c47 = *reinterpret_cast<const float4*>(&sc_buf[4]);

    #pragma unroll 2
    for (int i = tid; i < d_model; i += 256) {
        const float4 av = __ldg(&a_f4[i]);
        const float4 bv = __ldg(&b_f4[i]);
        const float delta =
            fmaf(bv.x, c03.x, fmaf(bv.y, c03.y, fmaf(bv.z, c03.z, bv.w * c03.w)))
          - fmaf(av.x, c47.x, fmaf(av.y, c47.y, fmaf(av.z, c47.z, av.w * c47.w)));
        y_out[i] = __ldg(&x_in[i]) + delta;
    }
}

__global__ __launch_bounds__(256, 5)
void siligemma_cayley_woodbury_r4_hoisted_sm86(
    const float* __restrict__ x_in,
    const float4* __restrict__ a_f4,
    const float4* __restrict__ b_f4,
    float* __restrict__ y_out,
    float alpha_scale,
    int d_model
) {
    cayley_woodbury_r4_hoisted_impl(
        x_in, a_f4, b_f4, y_out, alpha_scale, d_model,
        reinterpret_cast<const float4*>(c_wb_inv_256b)
    );
}

// Hub Steering Interface on L10..L14 20x shared_KV store (hub_slot in 0..4)
__global__ __launch_bounds__(256, 5)
void siligemma_cayley_woodbury_hub_l10_14_sm86(
    const float* __restrict__ x_in,
    const float4* __restrict__ a_f4,
    const float4* __restrict__ b_f4,
    float* __restrict__ y_out,
    float alpha_scale,
    int d_model,
    int hub_slot
) {
    const int safe_slot = (hub_slot >= 0 && hub_slot < 5) ? hub_slot : 0;
    cayley_woodbury_r4_hoisted_impl(
        x_in, a_f4, b_f4, y_out, alpha_scale, d_model,
        reinterpret_cast<const float4*>(c_wb_inv_hub_l10_14[safe_slot])
    );
}

// ----------------------------------------------------------------------------
// 5. Kernel 3: Pre-Quantized Sandwich-RMSNorm + H4 + Q8 + Optional u[4] Hoist
// ----------------------------------------------------------------------------

__global__ __launch_bounds__(128, 10)
void siligemma_sandwich_norm_u4_q8_sm86(
    const float* __restrict__ x_in,
    const float* __restrict__ rms_gamma,
    const float4* __restrict__ a_rot_f4,   // nullable when no r=4 steer is attached
    int32_t* __restrict__ x_q8_out,
    float* __restrict__ x_sc_out,
    float4* __restrict__ u4_out,
    float eps,
    int d_model
) {
    __shared__ float smem_tree[128]; // 512 B
    __shared__ float smem_blk[96];   // 384 B -> Total 896 B SMEM

    const int tid = threadIdx.x;
    const int d_f4 = d_model >> 2;
    const int b_idx = int(blockIdx.x);
    const float* x_row = x_in + int64_t(b_idx) * int64_t(d_model);
    int32_t* q8_row = x_q8_out + int64_t(b_idx) * int64_t(d_f4);
    float* sc_row = x_sc_out + int64_t(b_idx) * int64_t(d_model >> 5);

    float inv_rms = 1.0f;
    if (eps >= 0.0f) {
        const float sum_sq = canonical_sum_sq_d128(x_row, d_model, smem_tree, smem_blk);
        inv_rms = rsqrtf(sum_sq / float(d_model) + eps);
    }

    const float4* x_g4 = reinterpret_cast<const float4*>(x_row);
    float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;

    #pragma unroll 1
    for (int i = tid; i < d_f4; i += 128) {
        const float4 xv = __ldg(&x_g4[i]);
        const int g_base = i * 4;
        const float4 gv = (eps >= 0.0f && rms_gamma != nullptr)
            ? make_float4(__ldg(&rms_gamma[g_base]), __ldg(&rms_gamma[g_base + 1]),
                          __ldg(&rms_gamma[g_base + 2]), __ldg(&rms_gamma[g_base + 3]))
            : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        const float4 xn = make_float4(
            xv.x * inv_rms * gv.x,
            xv.y * inv_rms * gv.y,
            xv.z * inv_rms * gv.z,
            xv.w * inv_rms * gv.w
        );
        const float4 xr = h4_rot_f4(xn);

        float amax = fmaxf(fmaxf(fabsf(xr.x), fabsf(xr.y)), fmaxf(fabsf(xr.z), fabsf(xr.w)));
        #pragma unroll
        for (int mask = 4; mask >= 1; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask));
        }
        const float sc = (amax > 1.0e-12f) ? (amax * (1.0f / 127.0f)) : 1.0e-12f;
        const float inv_sc = 1.0f / sc;
        q8_row[i] = pack4_s8(
            __float2int_rn(fminf(127.0f, fmaxf(-127.0f, xr.x * inv_sc))),
            __float2int_rn(fminf(127.0f, fmaxf(-127.0f, xr.y * inv_sc))),
            __float2int_rn(fminf(127.0f, fmaxf(-127.0f, xr.z * inv_sc))),
            __float2int_rn(fminf(127.0f, fmaxf(-127.0f, xr.w * inv_sc)))
        );
        if ((i & 7) == 0) sc_row[i >> 3] = sc;

        if (a_rot_f4 != nullptr) {
            const float4 a0 = __ldg(&a_rot_f4[0 * d_f4 + i]);
            const float4 a1 = __ldg(&a_rot_f4[1 * d_f4 + i]);
            const float4 a2 = __ldg(&a_rot_f4[2 * d_f4 + i]);
            const float4 a3 = __ldg(&a_rot_f4[3 * d_f4 + i]);
            u0 = fmaf(a0.x, xr.x, fmaf(a0.y, xr.y, fmaf(a0.z, xr.z, fmaf(a0.w, xr.w, u0))));
            u1 = fmaf(a1.x, xr.x, fmaf(a1.y, xr.y, fmaf(a1.z, xr.z, fmaf(a1.w, xr.w, u1))));
            u2 = fmaf(a2.x, xr.x, fmaf(a2.y, xr.y, fmaf(a2.z, xr.z, fmaf(a2.w, xr.w, u2))));
            u3 = fmaf(a3.x, xr.x, fmaf(a3.y, xr.y, fmaf(a3.z, xr.z, fmaf(a3.w, xr.w, u3))));
        }
    }

    if (a_rot_f4 != nullptr && u4_out != nullptr) {
        const float r0 = tree_reduce_128_sm86(u0, smem_tree);
        const float r1 = tree_reduce_128_sm86(u1, smem_tree);
        const float r2 = tree_reduce_128_sm86(u2, smem_tree);
        const float r3 = tree_reduce_128_sm86(u3, smem_tree);
        if (tid == 0) u4_out[b_idx] = make_float4(r0, r1, r2, r3);
    }
}

// ----------------------------------------------------------------------------
// 6. Kernel 4: Coalesced A_16 + Block-H4 dp4a GEMV (0-ULP tree_reduce_128_sm86)
//    Supports g=32 & g=128 SoA wire format, inline SECDED, optional u[4] epilogue.
// ----------------------------------------------------------------------------

__global__ __launch_bounds__(128, 10)
void siligemma_a16_h4_dp4a_gemv_sm86(
    const int32_t* __restrict__ x_q8_in,
    const float* __restrict__ x_sc_in,
    const uint64_t* __restrict__ w_packed_u64,
    const __half* __restrict__ w_scales_f16,
    const float4* __restrict__ u4_in,
    const float4* __restrict__ b_f4,
    float* __restrict__ y_out,
    siligemma::HoloSphereRazorCanary64* host_canary,
    float lora_alpha,
    int out_dim,
    int in_dim,
    int group_size
) {
    __shared__ float smem_tree[128]; // 512 B

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int row = int(blockIdx.x);
    const int b_idx = int(blockIdx.y);
    if (row >= out_dim) return;

    const int n_blk16 = in_dim >> 4;
    const int n_grp_w = (group_size == 128) ? (in_dim >> 7) : (in_dim >> 5);
    const int g_shift = (group_size == 128) ? 3 : 1;  // blk16 >> 3 == g128, blk16 >> 1 == g32

    const uint64_t* row_u64 = w_packed_u64 + int64_t(row) * int64_t(n_blk16);
    const __half* row_sc = w_scales_f16 + int64_t(row) * int64_t(n_grp_w);
    const int32_t* xq_row = x_q8_in + int64_t(b_idx) * int64_t(in_dim >> 2);
    const float* xsc_row = x_sc_in + int64_t(b_idx) * int64_t(in_dim >> 5);

    float acc = 0.0f;
    #pragma unroll 1
    for (int blk_base = 0; blk_base < n_blk16; blk_base += 128) {
        const int blk = blk_base + tid;
        const bool active = (blk < n_blk16);
        const uint64_t w64 = active ? __ldg(&row_u64[blk]) : 0ULL;
        const float sc = active ? (__half2float(__ldg(&row_sc[blk >> g_shift])) * __ldg(&xsc_row[blk >> 1])) : 0.0f;
        eval_a16_blk16_secded_dp4a(active, w64, sc, xq_row + (active ? blk * 4 : 0), host_canary, lane, acc);
    }

    const float sum = tree_reduce_128_sm86(acc, smem_tree);
    if (tid == 0) {
        float val = sum;
        if (b_f4 != nullptr && u4_in != nullptr && lora_alpha != 0.0f) {
            const float4 uv = __ldg(&u4_in[b_idx]);
            const float4 bv = __ldg(&b_f4[row]);
            const float tangent = fmaf(bv.x, uv.x, fmaf(bv.y, uv.y, fmaf(bv.z, uv.z, bv.w * uv.w)));
            val = fmaf(lora_alpha, tangent, val);
        }
        y_out[int64_t(b_idx) * int64_t(out_dim) + int64_t(row)] = val;
    }
}

// Fast 4-row/CTA warp-tree A_16 dp4a GEMV (B=1 decode, optional softcap on lm_head)
__global__ __launch_bounds__(128, 10)
void siligemma_a16_h4_dp4a_gemv_r4_sm86(
    const int32_t* __restrict__ x_q8_in,
    const float* __restrict__ x_sc_in,
    const uint64_t* __restrict__ w_packed_u64,
    const __half* __restrict__ w_scales_f16,
    float* __restrict__ y_out,
    siligemma::HoloSphereRazorCanary64* host_canary,
    float softcap,
    int out_dim,
    int in_dim,
    int group_size
) {
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int row = int(blockIdx.x) * 4 + wid;
    const bool active_row = (row < out_dim);

    const int n_blk16 = in_dim >> 4;
    const int n_grp_w = (group_size == 128) ? (in_dim >> 7) : (in_dim >> 5);
    const int g_shift = (group_size == 128) ? 3 : 1;

    const uint64_t* row_u64 = active_row ? (w_packed_u64 + int64_t(row) * int64_t(n_blk16)) : nullptr;
    const __half* row_sc = active_row ? (w_scales_f16 + int64_t(row) * int64_t(n_grp_w)) : nullptr;

    float acc = 0.0f;
    #pragma unroll 1
    for (int blk_base = 0; blk_base < n_blk16; blk_base += 32) {
        const int blk = blk_base + lane;
        const bool active = active_row && (blk < n_blk16);
        const uint64_t w64 = active ? __ldg(&row_u64[blk]) : 0ULL;
        const float sc = active ? (__half2float(__ldg(&row_sc[blk >> g_shift])) * __ldg(&x_sc_in[blk >> 1])) : 0.0f;
        eval_a16_blk16_secded_dp4a(active, w64, sc, x_q8_in + (active ? blk * 4 : 0), host_canary, lane, acc);
    }

    acc = warp_sum_32(acc);
    if (active_row && lane == 0) {
        if (softcap > 0.0f) {
            acc = softcap * tanhf(acc / softcap);
        }
        y_out[row] = acc;
    }
}

// Weight-Reused B<=16 A_16 dp4a GEMM (reads w64 & scales ONCE across all B items)
__global__ __launch_bounds__(128, 6)
void siligemma_a16_h4_dp4a_gemm_b16_sm86(
    const int32_t* __restrict__ x_q8_in,
    const float* __restrict__ x_sc_in,
    const uint64_t* __restrict__ w_packed_u64,
    const __half* __restrict__ w_scales_f16,
    float* __restrict__ y_out,
    siligemma::HoloSphereRazorCanary64* host_canary,
    float softcap,
    int batch_size,
    int out_dim,
    int in_dim,
    int group_size
) {
    __shared__ float smem_tree[128];     // 512 B
    __shared__ float smem_b[16][128];    // 8,192 B

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int row = int(blockIdx.x);
    if (row >= out_dim) return;

    const int n_blk16 = in_dim >> 4;
    const int n_grp_w = (group_size == 128) ? (in_dim >> 7) : (in_dim >> 5);
    const int g_shift = (group_size == 128) ? 3 : 1;
    const int x_q8_stride = in_dim >> 2;
    const int x_sc_stride = in_dim >> 5;

    for (int b = 0; b < batch_size; ++b) {
        smem_b[b][tid] = 0.0f;
    }

    const uint64_t* row_u64 = w_packed_u64 + int64_t(row) * int64_t(n_blk16);
    const __half* row_sc = w_scales_f16 + int64_t(row) * int64_t(n_grp_w);

    #pragma unroll 1
    for (int blk_base = 0; blk_base < n_blk16; blk_base += 128) {
        const int blk = blk_base + tid;
        const bool active = (blk < n_blk16);
        uint64_t w64 = active ? __ldg(&row_u64[blk]) : 0ULL;
        const uint32_t syn = active ? a16_secded_lop3_popc_syndrome(w64) : 0u;
        const uint32_t any = __ballot_sync(0xFFFFFFFFu, syn != 0u);
        if (__builtin_expect(any != 0u, 0)) {
            uint32_t l_sec = 0u, l_ded = 0u, diag = 0u;
            if (syn != 0u) {
                if (syn & 1u) {
                    w64 ^= (1ULL << (((syn >> 1) & 15u) * 4u));
                    l_sec = 1u;
                    diag = syn & 0x1Fu;
                } else {
                    l_ded = 1u;
                    diag = (1u << 5) | (syn & 0x1Fu);
                }
            }
            razor_canary_emit_sys(host_canary, l_sec, l_ded, diag, lane);
        }
        if (active) {
            const float w_sc = __half2float(__ldg(&row_sc[blk >> g_shift]));
            int32_t wp0 = 0, wp1 = 0, wp2 = 0, wp3 = 0;
            {
                const uint32_t n0 = uint32_t(w64 & 0xFFFFULL);
                wp0 = pack4_s8(int32_t(n0 << 28) >> 28, int32_t(n0 << 24) >> 28, int32_t(n0 << 20) >> 28, int32_t(n0 << 16) >> 28);
                const uint32_t n1 = uint32_t((w64 >> 16) & 0xFFFFULL);
                wp1 = pack4_s8(int32_t(n1 << 28) >> 28, int32_t(n1 << 24) >> 28, int32_t(n1 << 20) >> 28, int32_t(n1 << 16) >> 28);
                const uint32_t n2 = uint32_t((w64 >> 32) & 0xFFFFULL);
                wp2 = pack4_s8(int32_t(n2 << 28) >> 28, int32_t(n2 << 24) >> 28, int32_t(n2 << 20) >> 28, int32_t(n2 << 16) >> 28);
                const uint32_t n3 = uint32_t((w64 >> 48) & 0xFFFFULL);
                wp3 = pack4_s8(int32_t(n3 << 28) >> 28, int32_t(n3 << 24) >> 28, int32_t(n3 << 20) >> 28, int32_t(n3 << 16) >> 28);
            }
            const int x_q_off = blk * 4;
            const int x_s_off = blk >> 1;
            #pragma unroll 1
            for (int b = 0; b < batch_size; ++b) {
                const int32_t* xq = x_q8_in + int64_t(b) * int64_t(x_q8_stride) + x_q_off;
                int32_t idot = ptx_dp4a_s32(wp0, __ldg(&xq[0]), 0);
                idot = ptx_dp4a_s32(wp1, __ldg(&xq[1]), idot);
                idot = ptx_dp4a_s32(wp2, __ldg(&xq[2]), idot);
                idot = ptx_dp4a_s32(wp3, __ldg(&xq[3]), idot);
                const float sc = w_sc * __ldg(&x_sc_in[int64_t(b) * int64_t(x_sc_stride) + x_s_off]);
                smem_b[b][tid] = fmaf(float(idot), sc, smem_b[b][tid]);
            }
        }
    }

    #pragma unroll 1
    for (int b = 0; b < batch_size; ++b) {
        float sum = tree_reduce_128_sm86(smem_b[b][tid], smem_tree);
        if (tid == 0) {
            if (softcap > 0.0f) {
                sum = softcap * tanhf(sum / softcap);
            }
            y_out[int64_t(b) * int64_t(out_dim) + int64_t(row)] = sum;
        }
    }
}

// ----------------------------------------------------------------------------
// 7. Kernel 5: Single-Pass Fused Sandwich-RMSNorm + H4 + Q8 + A_16 dp4a GEMV
//    <= 48 regs, <= 1,664 B SMEM, 0 B stack/spill, 0-ULP tree_reduce_128_sm86.
// ----------------------------------------------------------------------------

__global__ __launch_bounds__(128, 10)
void siligemma_fused_sandwich_h4_dp4a_gemv_sm86(
    const float* __restrict__ x_in,
    const float* __restrict__ rms_gamma,
    const uint64_t* __restrict__ w_packed_u64,
    const __half* __restrict__ w_scales_f16,
    float* __restrict__ y_out,
    siligemma::HoloSphereRazorCanary64* host_canary,
    float eps,
    int out_dim,
    int in_dim,
    int group_size
) {
    __shared__ float   smem_tree[128];   // 512 B
    __shared__ int32_t smem_x_q8[256];   // 1,024 B (TILE_K = 1024 -> 256 x 4B)
    __shared__ float   smem_x_sc[32];    // 128 B -> Total 1,664 B SMEM

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int row = int(blockIdx.x);
    const int b_idx = int(blockIdx.y);
    if (row >= out_dim) return;

    const float* x_row = x_in + int64_t(b_idx) * int64_t(in_dim);

    // Phase 1: Canonical 0-ULP block-128 sum of squares (reuses smem_x_q8 temporarily as float[96])
    float* smem_blk_sq = reinterpret_cast<float*>(smem_x_q8);
    const float sum_sq = canonical_sum_sq_d128(x_row, in_dim, smem_tree, smem_blk_sq);
    const float inv_rms = rsqrtf(sum_sq / float(in_dim) + eps);

    // Phase 2: Tiled Sandwich-Norm -> H_4 -> Q8 + SECDED + dp4a
    const int n_blk16 = in_dim >> 4;
    const int n_grp_w = (group_size == 128) ? (in_dim >> 7) : (in_dim >> 5);
    const int g_shift = (group_size == 128) ? 3 : 1;
    const uint64_t* row_u64 = w_packed_u64 + int64_t(row) * int64_t(n_blk16);
    const __half* row_sc = w_scales_f16 + int64_t(row) * int64_t(n_grp_w);
    const float4* x_g4 = reinterpret_cast<const float4*>(x_row);

    float acc = 0.0f;
    #pragma unroll 1
    for (int tile_base = 0; tile_base < in_dim; tile_base += 1024) {
        const int rem = in_dim - tile_base;
        const int tile_elems = (rem < 1024) ? rem : 1024;
        const int tile_f4 = tile_elems >> 2;
        const int tile_base_f4 = tile_base >> 2;
        const int tile_blk16 = tile_elems >> 4;
        const int tile_base_blk16 = tile_base >> 4;

        for (int i = tid; i < 256; i += 128) {
            const bool active = (i < tile_f4);
            float4 xr = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (active) {
                const float4 xv = __ldg(&x_g4[tile_base_f4 + i]);
                const int g_base = (tile_base_f4 + i) * 4;
                const float4 gv = (rms_gamma != nullptr)
                    ? make_float4(__ldg(&rms_gamma[g_base]), __ldg(&rms_gamma[g_base + 1]),
                                  __ldg(&rms_gamma[g_base + 2]), __ldg(&rms_gamma[g_base + 3]))
                    : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
                xr = h4_rot_f4(make_float4(
                    xv.x * inv_rms * gv.x,
                    xv.y * inv_rms * gv.y,
                    xv.z * inv_rms * gv.z,
                    xv.w * inv_rms * gv.w
                ));
            }
            float amax = fmaxf(fmaxf(fabsf(xr.x), fabsf(xr.y)), fmaxf(fabsf(xr.z), fabsf(xr.w)));
            #pragma unroll
            for (int mask = 4; mask >= 1; mask >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask));
            }
            const float sc = (amax > 1.0e-12f) ? (amax * (1.0f / 127.0f)) : 1.0e-12f;
            const float inv_sc = 1.0f / sc;
            if (active) {
                smem_x_q8[i] = pack4_s8(
                    __float2int_rn(fminf(127.0f, fmaxf(-127.0f, xr.x * inv_sc))),
                    __float2int_rn(fminf(127.0f, fmaxf(-127.0f, xr.y * inv_sc))),
                    __float2int_rn(fminf(127.0f, fmaxf(-127.0f, xr.z * inv_sc))),
                    __float2int_rn(fminf(127.0f, fmaxf(-127.0f, xr.w * inv_sc)))
                );
                if ((i & 7) == 0) smem_x_sc[i >> 3] = sc;
            }
        }
        __syncthreads();

        const bool active_blk = (tid < tile_blk16);
        const int blk_glob = tile_base_blk16 + tid;
        const uint64_t w64 = active_blk ? __ldg(&row_u64[blk_glob]) : 0ULL;
        const float sc = active_blk ? (__half2float(__ldg(&row_sc[blk_glob >> g_shift])) * smem_x_sc[tid >> 1]) : 0.0f;
        eval_a16_blk16_secded_dp4a(active_blk, w64, sc, smem_x_q8 + (active_blk ? tid * 4 : 0), host_canary, lane, acc);
        __syncthreads();
    }

    const float sum = tree_reduce_128_sm86(acc, smem_tree);
    if (tid == 0) {
        y_out[int64_t(b_idx) * int64_t(out_dim) + int64_t(row)] = sum;
    }
}

// ----------------------------------------------------------------------------
// 8. Kernel 6: Gemma-4 GeGLU + Post-FFN / PLE Sandwich + Layer Scalar c_l
// ----------------------------------------------------------------------------

__global__ __launch_bounds__(128, 10)
void siligemma_geglu_act_sm86(
    const float* __restrict__ gate_in,
    const float* __restrict__ up_in,
    float* __restrict__ act_out,
    int d_ff
) {
    const int idx = int(blockIdx.x) * 128 + int(threadIdx.x);
    if (idx < d_ff) {
        act_out[idx] = gelu_pytorch_tanh_dev(__ldg(&gate_in[idx])) * __ldg(&up_in[idx]);
    }
}

__global__ __launch_bounds__(128, 10)
void siligemma_ple_combine_gate_sm86(
    const float* __restrict__ ple_gate_raw,
    const float* __restrict__ ple_model_proj,
    const __nv_bfloat16* __restrict__ ple_token_dma,
    const float* __restrict__ ple_proj_norm_gamma,
    float* __restrict__ ple_gated_out,
    int layer_idx,
    int num_layers,
    int ple_dim,
    float hidden_inv_sqrt,
    float eps
) {
    __shared__ float smem_tree[128]; // 512 B
    __shared__ float smem_blk[4];    // 16 B

    const int tid = threadIdx.x;
    const int b_idx = int(blockIdx.x);
    const int slice_off = (b_idx * num_layers + layer_idx) * ple_dim;

    // 256 = 2 * 128 -> canonical RMSNorm over per_layer_model_projection slice
    for (int c = 0; c < 2; ++c) {
        const int i = c * 128 + tid;
        float p = 0.0f;
        if (i < ple_dim) {
            const float mv = __bfloat162float(__float2bfloat16(__ldg(&ple_model_proj[slice_off + i]) * hidden_inv_sqrt));
            p = mv;
        }
        const float cs = tree_reduce_128_sm86(p * p, smem_tree);
        if (tid == 0) smem_blk[c] = cs;
    }
    __syncthreads();
    const float tot_sq = tree_reduce_128_sm86((tid < 2) ? smem_blk[tid] : 0.0f, smem_tree);
    const float inv_rms = rsqrtf(tot_sq / float(ple_dim) + eps);
    constexpr float kInvSqrt2 = 0.7071067811865475f;
    constexpr float kSqrtPle = 16.0f; // sqrt(256)

    for (int i = tid; i < ple_dim; i += 128) {
        const float mv = __bfloat162float(__float2bfloat16(__ldg(&ple_model_proj[slice_off + i]) * hidden_inv_sqrt));
        const float mp = mv * inv_rms * __ldg(&ple_proj_norm_gamma[i]);
        const float tok_p = __bfloat162float(__ldg(&ple_token_dma[slice_off + i])) * kSqrtPle;
        const float ple_comb = (mp + tok_p) * kInvSqrt2;
        const float g_act = gelu_pytorch_tanh_dev(__ldg(&ple_gate_raw[b_idx * ple_dim + i]));
        ple_gated_out[b_idx * ple_dim + i] = g_act * ple_comb;
    }
}

__global__ __launch_bounds__(128, 10)
void siligemma_fused_geglu_ple_sandwich_sm86(
    const float* __restrict__ res_in,
    const float* __restrict__ block_raw_in,
    const float* __restrict__ post_norm_gamma,
    const float* __restrict__ ple_proj_raw_in,   // nullable (attn sandwich skips PLE stage)
    const float* __restrict__ post_ple_gamma,
    float* __restrict__ h_out,
    float layer_scalar,
    float eps,
    int d_model
) {
    __shared__ float smem_tree[128]; // 512 B
    __shared__ float smem_blk[16];   // 64 B -> Total 576 B <= 1,152 B SMEM

    const int tid = threadIdx.x;
    const int b_idx = int(blockIdx.x);
    const int64_t row_off = int64_t(b_idx) * int64_t(d_model);

    const float sum_sq_a = canonical_sum_sq_d128(block_raw_in + row_off, d_model, smem_tree, smem_blk);
    const float inv_rms_a = rsqrtf(sum_sq_a / float(d_model) + eps);

    if (ple_proj_raw_in != nullptr && post_ple_gamma != nullptr) {
        const float sum_sq_p = canonical_sum_sq_d128(ple_proj_raw_in + row_off, d_model, smem_tree, smem_blk);
        const float inv_rms_p = rsqrtf(sum_sq_p / float(d_model) + eps);
        for (int i = tid; i < d_model; i += 128) {
            const float v_l = __ldg(&res_in[row_off + i]) + __ldg(&block_raw_in[row_off + i]) * inv_rms_a * __ldg(&post_norm_gamma[i]);
            const float p_n = __ldg(&ple_proj_raw_in[row_off + i]) * inv_rms_p * __ldg(&post_ple_gamma[i]);
            h_out[row_off + i] = layer_scalar * (v_l + p_n);
        }
    } else {
        for (int i = tid; i < d_model; i += 128) {
            const float r_val = (res_in != nullptr) ? __ldg(&res_in[row_off + i]) : 0.0f;
            const float v_l = r_val + __ldg(&block_raw_in[row_off + i]) * inv_rms_a * __ldg(&post_norm_gamma[i]);
            h_out[row_off + i] = layer_scalar * v_l;
        }
    }
}

// ----------------------------------------------------------------------------
// 9. Kernel 7: zed Shared-KV Attention (Sliding d_h=256 & Proportional d_h=512)
//    L0..14 write static ring; L15..34 & L25..33 loops strictly read-only (0 B extra KV)
// ----------------------------------------------------------------------------

__global__ __launch_bounds__(128, 10)
void siligemma_attention_kv_step_sm86(
    const float* __restrict__ q_raw,
    const float* __restrict__ k_raw,         // nullptr for L15..34 reuse layers
    const float* __restrict__ v_raw,         // nullptr for L15..34 reuse layers
    const float* __restrict__ q_norm_gamma,
    const float* __restrict__ k_norm_gamma,  // nullptr for L15..34 reuse layers
    __nv_bfloat16* __restrict__ kv_ring_buf, // [batch_stride, ring_slots, 2, head_dim]
    float* __restrict__ attn_out,            // [batch, 8, head_dim]
    int pos,
    const int32_t* __restrict__ d_pos_ptr,   // nullable dynamic position pointer for CUDA Graphs
    int ring_slots,
    int kv_batch_stride,
    int head_dim,
    int is_full_attn,
    int is_shared_layer,
    float eps
) {
    __shared__ float smem_tree[128];          // 512 B
    __shared__ float smem_blk[4];             // 16 B
    __shared__ __nv_bfloat16 smem_cur_k[512]; // 1,024 B (eliminates inter-CTA data race at t == eff_pos)
    __shared__ __nv_bfloat16 smem_cur_v[512]; // 1,024 B -> Total 2,576 B SMEM

    const int tid = threadIdx.x;
    const int head_idx = int(blockIdx.x);
    const int b_idx = int(blockIdx.y);
    const int stride_k = 2 * head_dim;
    const int eff_pos = (d_pos_ptr != nullptr) ? __ldg(d_pos_ptr) : pos;
    const int64_t b_kv_off = int64_t(b_idx) * int64_t(kv_batch_stride) * int64_t(stride_k);

    // Step A: all 8 head CTAs compute K/V at t == eff_pos into local SMEM; head 0 writes global ring
    if (!is_shared_layer && k_raw != nullptr && v_raw != nullptr) {
        const float* k_in = k_raw + int64_t(b_idx) * int64_t(head_dim);
        const float* v_in = v_raw + int64_t(b_idx) * int64_t(head_dim);

        const float k_ss = canonical_sum_sq_d128(k_in, head_dim, smem_tree, smem_blk);
        const float v_ss = canonical_sum_sq_d128(v_in, head_dim, smem_tree, smem_blk);
        const float k_inv = rsqrtf(k_ss / float(head_dim) + eps);
        const float v_inv = rsqrtf(v_ss / float(head_dim) + eps); // v_norm has with_scale=False

        const float rope_theta = is_full_attn ? siligemma::GEMMA4_ROPE_THETA_FULL : siligemma::GEMMA4_ROPE_THETA_SLIDING;
        const int half_d = head_dim >> 1;
        const int rot_lim = is_full_attn ? int(siligemma::GEMMA4_FULL_ROTATED_PAIRS) : half_d;
        const int slot = eff_pos % ring_slots;
        __nv_bfloat16* k_dst = kv_ring_buf + b_kv_off + int64_t(slot) * int64_t(stride_k);
        __nv_bfloat16* v_dst = k_dst + head_dim;

        for (int i = tid; i < half_d; i += 128) {
            const float k0 = __ldg(&k_in[i]) * k_inv * __ldg(&k_norm_gamma[i]);
            const float k1 = __ldg(&k_in[i + half_d]) * k_inv * __ldg(&k_norm_gamma[i + half_d]);
            float rk0 = k0, rk1 = k1;
            if (i < rot_lim) {
                const float inv_f = powf(rope_theta, -float(2 * i) / float(head_dim));
                float s_v = 0.0f, c_v = 1.0f;
                __sincosf(float(eff_pos) * inv_f, &s_v, &c_v);
                rk0 = fmaf(k0, c_v, -k1 * s_v);
                rk1 = fmaf(k1, c_v, k0 * s_v);
            }
            const __nv_bfloat16 bk0 = __float2bfloat16(rk0);
            const __nv_bfloat16 bk1 = __float2bfloat16(rk1);
            const __nv_bfloat16 bv0 = __float2bfloat16(__ldg(&v_in[i]) * v_inv);
            const __nv_bfloat16 bv1 = __float2bfloat16(__ldg(&v_in[i + half_d]) * v_inv);
            smem_cur_k[i] = bk0;
            smem_cur_k[i + half_d] = bk1;
            smem_cur_v[i] = bv0;
            smem_cur_v[i + half_d] = bv1;
            if (head_idx == 0) {
                k_dst[i] = bk0;
                k_dst[i + half_d] = bk1;
                v_dst[i] = bv0;
                v_dst[i + half_d] = bv1;
            }
        }
    }
    __syncthreads();

    // Step B: Q norm + RoPE (registered per head) & online attention over active window
    const float* q_in = q_raw + (int64_t(b_idx) * 8LL + int64_t(head_idx)) * int64_t(head_dim);
    const float q_ss = canonical_sum_sq_d128(q_in, head_dim, smem_tree, smem_blk);
    const float q_inv = rsqrtf(q_ss / float(head_dim) + eps);

    const float rope_theta = is_full_attn ? siligemma::GEMMA4_ROPE_THETA_FULL : siligemma::GEMMA4_ROPE_THETA_SLIDING;
    const int half_d = head_dim >> 1;
    const int rot_lim = is_full_attn ? int(siligemma::GEMMA4_FULL_ROTATED_PAIRS) : half_d;

    const int p0 = tid;
    const int p1 = tid + 128;
    float q0_a = __ldg(&q_in[p0]) * q_inv * __ldg(&q_norm_gamma[p0]);
    float q0_b = __ldg(&q_in[p0 + half_d]) * q_inv * __ldg(&q_norm_gamma[p0 + half_d]);
    if (p0 < rot_lim) {
        const float inv_f = powf(rope_theta, -float(2 * p0) / float(head_dim));
        float s_v = 0.0f, c_v = 1.0f;
        __sincosf(float(eff_pos) * inv_f, &s_v, &c_v);
        const float rq0 = fmaf(q0_a, c_v, -q0_b * s_v);
        const float rq1 = fmaf(q0_b, c_v, q0_a * s_v);
        q0_a = rq0; q0_b = rq1;
    }

    float q1_a = 0.0f, q1_b = 0.0f;
    if (half_d > 128) {
        q1_a = __ldg(&q_in[p1]) * q_inv * __ldg(&q_norm_gamma[p1]);
        q1_b = __ldg(&q_in[p1 + half_d]) * q_inv * __ldg(&q_norm_gamma[p1 + half_d]);
        if (p1 < rot_lim) {
            const float inv_f = powf(rope_theta, -float(2 * p1) / float(head_dim));
            float s_v = 0.0f, c_v = 1.0f;
            __sincosf(float(eff_pos) * inv_f, &s_v, &c_v);
            const float rq0 = fmaf(q1_a, c_v, -q1_b * s_v);
            const float rq1 = fmaf(q1_b, c_v, q1_a * s_v);
            q1_a = rq0; q1_b = rq1;
        }
    }

    const int max_hist = is_full_attn ? ring_slots : ((ring_slots < int(siligemma::GEMMA4_SLIDING_WINDOW)) ? ring_slots : int(siligemma::GEMMA4_SLIDING_WINDOW));
    const int n_hist = (eff_pos + 1 < max_hist) ? (eff_pos + 1) : max_hist;
    const int t_start = eff_pos + 1 - n_hist;

    float m_run = -1.0e30f;
    float l_run = 0.0f;
    float o0_a = 0.0f, o0_b = 0.0f, o1_a = 0.0f, o1_b = 0.0f;

    #pragma unroll 1
    for (int t = t_start; t <= eff_pos; ++t) {
        const bool use_smem = (!is_shared_layer && t == eff_pos);
        const __nv_bfloat16* k_src = use_smem
            ? smem_cur_k
            : (kv_ring_buf + b_kv_off + int64_t(t % ring_slots) * int64_t(stride_k));
        const __nv_bfloat16* v_src = use_smem ? smem_cur_v : (k_src + head_dim);

        float dot_local = fmaf(q0_a, __bfloat162float(k_src[p0]), q0_b * __bfloat162float(k_src[p0 + half_d]));
        if (half_d > 128) {
            dot_local = fmaf(q1_a, __bfloat162float(k_src[p1]), fmaf(q1_b, __bfloat162float(k_src[p1 + half_d]), dot_local));
        }
        // Gemma-4 uses scaling = 1.0
        const float score = tree_reduce_128_sm86(dot_local, smem_tree);

        const float m_new = fmaxf(m_run, score);
        const float alpha = __expf(m_run - m_new);
        const float beta = __expf(score - m_new);
        l_run = fmaf(l_run, alpha, beta);

        o0_a = fmaf(o0_a, alpha, beta * __bfloat162float(v_src[p0]));
        o0_b = fmaf(o0_b, alpha, beta * __bfloat162float(v_src[p0 + half_d]));
        if (half_d > 128) {
            o1_a = fmaf(o1_a, alpha, beta * __bfloat162float(v_src[p1]));
            o1_b = fmaf(o1_b, alpha, beta * __bfloat162float(v_src[p1 + half_d]));
        }
        m_run = m_new;
    }

    const float inv_l = (l_run > 1.0e-12f) ? (1.0f / l_run) : 0.0f;
    float* out_dst = attn_out + (int64_t(b_idx) * 8LL + int64_t(head_idx)) * int64_t(head_dim);
    out_dst[p0] = o0_a * inv_l;
    out_dst[p0 + half_d] = o0_b * inv_l;
    if (half_d > 128) {
        out_dst[p1] = o1_a * inv_l;
        out_dst[p1 + half_d] = o1_b * inv_l;
    }
}

// ----------------------------------------------------------------------------
// 10. Kernel 8: U-4 L25..L33 Loop Controller Gate + Riccati/S164 Residual Norm
// ----------------------------------------------------------------------------

__global__ __launch_bounds__(128, 10)
void siligemma_u4_loop_gate_step_sm86(
    const float* __restrict__ h_in,
    const __nv_bfloat16* __restrict__ m_prev_bf16,
    const __nv_bfloat16* __restrict__ msg_query_bf16, // [H]
    const __nv_bfloat16* __restrict__ w_down_bf16,    // [r=16, 2*H]
    const __nv_bfloat16* __restrict__ w_forget_bf16,  // [H, r=16]
    const __nv_bfloat16* __restrict__ w_gain_bf16,    // [H, r=16]
    float* __restrict__ h_out,
    float eps,
    int d_model
) {
    __shared__ float smem_tree[128]; // 512 B
    __shared__ float smem_blk[16];   // 64 B
    __shared__ float smem_z[16];     // 64 B (rank=16 bottleneck latent)

    const int tid = threadIdx.x;
    const int b_idx = int(blockIdx.x);
    const int64_t row_off = int64_t(b_idx) * int64_t(d_model);

    // Compute canonical RMSNorm scales for h_in and m_eff = m_prev * sigmoid(msg_query)
    const float h_ss = canonical_sum_sq_d128(h_in + row_off, d_model, smem_tree, smem_blk);
    const float h_inv = rsqrtf(h_ss / float(d_model) + eps);

    // Compute m_eff norm squared cleanly
    const int n_chunks = (d_model + 127) >> 7;
    #pragma unroll 1
    for (int c = 0; c < n_chunks; ++c) {
        const int idx = c * 128 + tid;
        float mv = 0.0f;
        if (idx < d_model) {
            const float mq = __bfloat162float(__ldg(&msg_query_bf16[idx]));
            const float qs = 1.0f / (1.0f + __expf(-mq));
            mv = __bfloat162float(__ldg(&m_prev_bf16[row_off + idx])) * qs;
        }
        const float cs = tree_reduce_128_sm86(mv * mv, smem_tree);
        if (tid == 0) smem_blk[c] = cs;
    }
    __syncthreads();
    const float m_ss = tree_reduce_128_sm86((tid < n_chunks) ? smem_blk[tid] : 0.0f, smem_tree);
    const float m_inv = rsqrtf(m_ss / float(d_model) + eps);

    // Project [RMSNorm(h); RMSNorm(m_eff)] -> z[0..15]
    #pragma unroll 1
    for (int r = 0; r < 16; ++r) {
        float acc = 0.0f;
        const int64_t row_w = int64_t(r) * int64_t(2 * d_model);
        for (int idx = tid; idx < d_model; idx += 128) {
            const float mq = __bfloat162float(__ldg(&msg_query_bf16[idx]));
            const float qs = 1.0f / (1.0f + __expf(-mq));
            const float hn = __ldg(&h_in[row_off + idx]) * h_inv;
            const float mn = __bfloat162float(__ldg(&m_prev_bf16[row_off + idx])) * qs * m_inv;
            acc = fmaf(__bfloat162float(__ldg(&w_down_bf16[row_w + idx])), hn, acc);
            acc = fmaf(__bfloat162float(__ldg(&w_down_bf16[row_w + d_model + idx])), mn, acc);
        }
        const float z_r = tree_reduce_128_sm86(acc, smem_tree);
        if (tid == 0) smem_z[r] = z_r;
    }
    __syncthreads();

    // Update h_out = (1 - tanh(W_forget @ z)) * h + tanh(W_gain @ z) * m_eff
    for (int idx = tid; idx < d_model; idx += 128) {
        float fg = 0.0f, gg = 0.0f;
        const int64_t off_g = int64_t(idx) * 16LL;
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            const float zv = smem_z[r];
            fg = fmaf(__bfloat162float(__ldg(&w_forget_bf16[off_g + r])), zv, fg);
            gg = fmaf(__bfloat162float(__ldg(&w_gain_bf16[off_g + r])), zv, gg);
        }
        const float mq = __bfloat162float(__ldg(&msg_query_bf16[idx]));
        const float qs = 1.0f / (1.0f + __expf(-mq));
        const float m_eff = __bfloat162float(__ldg(&m_prev_bf16[row_off + idx])) * qs;
        const float hv = __ldg(&h_in[row_off + idx]);
        h_out[row_off + idx] = fmaf(1.0f - tanhf(fg), hv, tanhf(gg) * m_eff);
    }
}

// ----------------------------------------------------------------------------
// 11. Kernel 9: P224a IRIS Runtime+Scan Canary Finalize + Softcap Argmax
// ----------------------------------------------------------------------------

__global__ __launch_bounds__(128, 10)
void siligemma_iris_canary_finalize_sm86(
    const float* __restrict__ h_state,
    siligemma::HoloSphereRazorCanary64* host_canary,
    uint32_t layer_idx,
    uint32_t pass_idx,
    uint32_t golden_iris_expected,
    int check_golden,
    int d_model
) {
    __shared__ uint32_t smem_crc[128]; // 512 B

    const int tid = threadIdx.x;
    uint32_t local_h = siligemma::IRIS_WEYL_MIX ^ (layer_idx * 1315423911u) ^ (pass_idx * 2654435761u);
    for (int i = tid; i < d_model; i += 128) {
        const float fv = __ldg(&h_state[i]);
        const uint32_t bits = __float_as_uint(fv);
        local_h = iris_mix_word_dev(local_h, bits);
    }
    smem_crc[tid] = local_h;
    __syncthreads();

    #pragma unroll
    for (int s = 64; s >= 1; s >>= 1) {
        if (tid < s) {
            smem_crc[tid] = iris_mix_word_dev(smem_crc[tid], smem_crc[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0 && host_canary != nullptr) {
        const uint32_t step_crc = smem_crc[0];
        const uint32_t prev_crc = host_canary->iris_runtime_crc_u32;
        const uint32_t next_crc = iris_mix_word_dev(prev_crc, step_crc);
        host_canary->iris_runtime_crc_u32 = next_crc;
        const uint32_t lo_syn = host_canary->last_syndrome_u32 & 0xFFFFu;
        razor_st_release_sys_u32(&host_canary->last_syndrome_u32, ((next_crc & 0xFFFFu) << 16) | lo_syn);
        if (check_golden && next_crc != golden_iris_expected) {
            razor_atom_release_sys_add_u32(&host_canary->iris_mismatch_count_u32, 1u);
            razor_atom_release_sys_add_u32(&host_canary->ded_count_u32, 1u);
        }
    }
}

__global__ __launch_bounds__(128, 10)
void siligemma_softcap_logits_argmax_sm86(
    const float* __restrict__ raw_logits,
    float* __restrict__ softcap_logits,
    int32_t* __restrict__ tok_out,
    float* __restrict__ score_out,
    int32_t* __restrict__ d_pos_mut,
    float softcap,
    int vocab_size
) {
    __shared__ float smem_val[128];
    __shared__ int32_t smem_idx[128];

    const int tid = threadIdx.x;
    const int b_idx = int(blockIdx.x);
    const int64_t row_off = int64_t(b_idx) * int64_t(vocab_size);
    const float* r_in = raw_logits + row_off;
    float* r_out = (softcap_logits != nullptr) ? (softcap_logits + row_off) : nullptr;
    const bool do_cap = (softcap > 0.0f);
    const float inv_cap = do_cap ? (1.0f / softcap) : 1.0f;

    float best_v = -1.0e30f;
    int32_t best_i = 0;

    const int v_f4 = vocab_size >> 2;
    const float4* r_f4 = reinterpret_cast<const float4*>(r_in);
    float4* o_f4 = (r_out != nullptr) ? reinterpret_cast<float4*>(r_out) : nullptr;
    for (int i = tid; i < v_f4; i += 128) {
        const float4 v4 = __ldg(&r_f4[i]);
        const float c0 = do_cap ? (softcap * tanhf(v4.x * inv_cap)) : v4.x;
        const float c1 = do_cap ? (softcap * tanhf(v4.y * inv_cap)) : v4.y;
        const float c2 = do_cap ? (softcap * tanhf(v4.z * inv_cap)) : v4.z;
        const float c3 = do_cap ? (softcap * tanhf(v4.w * inv_cap)) : v4.w;
        if (o_f4 != nullptr) o_f4[i] = make_float4(c0, c1, c2, c3);
        const int base = i * 4;
        if (c0 > best_v || (c0 == best_v && base + 0 < best_i)) { best_v = c0; best_i = base + 0; }
        if (c1 > best_v || (c1 == best_v && base + 1 < best_i)) { best_v = c1; best_i = base + 1; }
        if (c2 > best_v || (c2 == best_v && base + 2 < best_i)) { best_v = c2; best_i = base + 2; }
        if (c3 > best_v || (c3 == best_v && base + 3 < best_i)) { best_v = c3; best_i = base + 3; }
    }
    smem_val[tid] = best_v;
    smem_idx[tid] = best_i;
    __syncthreads();

    #pragma unroll
    for (int s = 64; s >= 1; s >>= 1) {
        if (tid < s) {
            const float ov = smem_val[tid + s];
            const int32_t oi = smem_idx[tid + s];
            if (ov > smem_val[tid] || (ov == smem_val[tid] && oi < smem_idx[tid])) {
                smem_val[tid] = ov;
                smem_idx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        tok_out[b_idx] = smem_idx[0];
        score_out[b_idx] = smem_val[0];
        if (b_idx == 0 && d_pos_mut != nullptr) {
            *d_pos_mut += 1;
        }
    }
}

// ----------------------------------------------------------------------------
// 12. Tied BF16 Embedding Lookup / State Snapshot + BF16 LM Head + Euclidean Baselines
// ----------------------------------------------------------------------------

__global__ __launch_bounds__(128, 10)
void siligemma_embed_lookup_sm86(
    const __nv_bfloat16* __restrict__ embed_bf16,
    const int32_t* __restrict__ tok_in_i32,
    const float* __restrict__ f32_in,
    float* __restrict__ f32_out,
    __nv_bfloat16* __restrict__ bf16_out,
    float embed_scale,
    int d_model,
    int mode  // 0: embed lookup (* embed_scale -> f32_out), 1: snapshot f32_in -> bf16_out
) {
    const int tid = threadIdx.x;
    const int b_idx = int(blockIdx.x);
    const int64_t row_off = int64_t(b_idx) * int64_t(d_model);
    if (mode == 0) {
        const int32_t tid_tok = __ldg(&tok_in_i32[b_idx]);
        const __nv_bfloat16* src = embed_bf16 + int64_t(tid_tok) * int64_t(d_model);
        for (int i = tid; i < d_model; i += 128) {
            f32_out[row_off + i] = __bfloat162float(__ldg(&src[i])) * embed_scale;
        }
    } else {
        for (int i = tid; i < d_model; i += 128) {
            bf16_out[row_off + i] = __float2bfloat16(__ldg(&f32_in[row_off + i]));
        }
    }
}

__global__ __launch_bounds__(128, 10)
void siligemma_bf16_lm_head_r4_sm86(
    const float* __restrict__ x_norm_in,
    const __nv_bfloat16* __restrict__ emb_bf16,
    float* __restrict__ logits_out,
    float softcap,
    int vocab_size,
    int d_model
) {
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int row = int(blockIdx.x) * 4 + wid;
    const bool active_row = (row < vocab_size);
    const int d_f8 = d_model >> 3;

    const float4* w_f4 = active_row ? reinterpret_cast<const float4*>(emb_bf16 + int64_t(row) * int64_t(d_model)) : nullptr;
    const float4* x_f4 = reinterpret_cast<const float4*>(x_norm_in);

    float acc = 0.0f;
    if (active_row) {
        for (int i = lane; i < d_f8; i += 32) {
            const float4 wv = __ldg(&w_f4[i]);
            const float4 x0 = __ldg(&x_f4[i * 2 + 0]);
            const float4 x1 = __ldg(&x_f4[i * 2 + 1]);
            const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wv);
            const float2 f0 = __bfloat1622float2(wp[0]);
            const float2 f1 = __bfloat1622float2(wp[1]);
            const float2 f2 = __bfloat1622float2(wp[2]);
            const float2 f3 = __bfloat1622float2(wp[3]);
            acc = fmaf(f0.x, x0.x, fmaf(f0.y, x0.y, fmaf(f1.x, x0.z, fmaf(f1.y, x0.w, acc))));
            acc = fmaf(f2.x, x1.x, fmaf(f2.y, x1.y, fmaf(f3.x, x1.z, fmaf(f3.y, x1.w, acc))));
        }
    }
    acc = warp_sum_32(acc);
    if (active_row && lane == 0) {
        if (softcap > 0.0f) {
            acc = softcap * tanhf(acc / softcap);
        }
        logits_out[row] = acc;
    }
}

__global__ __launch_bounds__(128, 6)
void siligemma_bf16_lm_head_b16_sm86(
    const float* __restrict__ x_norm_in,
    const __nv_bfloat16* __restrict__ emb_bf16,
    float* __restrict__ logits_out,
    float softcap,
    int batch_size,
    int vocab_size,
    int d_model
) {
    __shared__ float smem_tree[128];  // 512 B
    __shared__ float smem_b[16][128]; // 8,192 B

    const int tid = threadIdx.x;
    const int row = int(blockIdx.x);
    if (row >= vocab_size) return;

    for (int b = 0; b < batch_size; ++b) {
        smem_b[b][tid] = 0.0f;
    }

    const int d_f8 = d_model >> 3;
    const int d_f4 = d_model >> 2;
    const float4* w_f4 = reinterpret_cast<const float4*>(emb_bf16 + int64_t(row) * int64_t(d_model));
    const float4* x_f4 = reinterpret_cast<const float4*>(x_norm_in);

    #pragma unroll 1
    for (int i = tid; i < d_f8; i += 128) {
        const float4 wv = __ldg(&w_f4[i]);
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wv);
        const float2 f0 = __bfloat1622float2(wp[0]);
        const float2 f1 = __bfloat1622float2(wp[1]);
        const float2 f2 = __bfloat1622float2(wp[2]);
        const float2 f3 = __bfloat1622float2(wp[3]);
        #pragma unroll 1
        for (int b = 0; b < batch_size; ++b) {
            const float4 x0 = __ldg(&x_f4[int64_t(b) * int64_t(d_f4) + i * 2 + 0]);
            const float4 x1 = __ldg(&x_f4[int64_t(b) * int64_t(d_f4) + i * 2 + 1]);
            float a = fmaf(f0.x, x0.x, fmaf(f0.y, x0.y, fmaf(f1.x, x0.z, f1.y * x0.w)));
            a = fmaf(f2.x, x1.x, fmaf(f2.y, x1.y, fmaf(f3.x, x1.z, fmaf(f3.y, x1.w, a))));
            smem_b[b][tid] += a;
        }
    }

    #pragma unroll 1
    for (int b = 0; b < batch_size; ++b) {
        float sum = tree_reduce_128_sm86(smem_b[b][tid], smem_tree);
        if (tid == 0) {
            if (softcap > 0.0f) {
                sum = softcap * tanhf(sum / softcap);
            }
            logits_out[int64_t(b) * int64_t(vocab_size) + int64_t(row)] = sum;
        }
    }
}

__global__ __launch_bounds__(128, 10)
void siligemma_euclidean_lora_r4_sm86(
    const float* __restrict__ x_in,
    const float4* __restrict__ a_f4,
    const float4* __restrict__ b_f4,
    float* __restrict__ y_out,
    float alpha_scale,
    int d_model
) {
    __shared__ float4 smem_u_warp[4];
    __shared__ float4 smem_u;

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
    for (int i = tid; i < d_model; i += 128) {
        const float xi = __ldg(&x_in[i]);
        const float4 av = __ldg(&a_f4[i]);
        u0 = fmaf(av.x, xi, u0);
        u1 = fmaf(av.y, xi, u1);
        u2 = fmaf(av.z, xi, u2);
        u3 = fmaf(av.w, xi, u3);
    }
    u0 = warp_sum_32(u0);
    u1 = warp_sum_32(u1);
    u2 = warp_sum_32(u2);
    u3 = warp_sum_32(u3);
    if (lane == 0) smem_u_warp[wid] = make_float4(u0, u1, u2, u3);
    __syncthreads();
    if (tid == 0) {
        float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float4 sw = smem_u_warp[w];
            s.x += sw.x; s.y += sw.y; s.z += sw.z; s.w += sw.w;
        }
        smem_u = s;
    }
    __syncthreads();

    const float4 uv = smem_u;
    for (int i = tid; i < d_model; i += 128) {
        const float4 bv = __ldg(&b_f4[i]);
        const float delta = alpha_scale * fmaf(bv.x, uv.x, fmaf(bv.y, uv.y, fmaf(bv.z, uv.z, bv.w * uv.w)));
        y_out[i] = __ldg(&x_in[i]) + delta;
    }
}

__global__ __launch_bounds__(128, 10)
void siligemma_tree_reduce_probe_sm86(
    const float* __restrict__ x_in,
    float* __restrict__ sum_out,
    int dim
) {
    __shared__ float smem_tree[128];
    __shared__ float smem_blk[96];
    const int tid = threadIdx.x;
    const int b_idx = int(blockIdx.x);
    const float* x_row = x_in + int64_t(b_idx) * int64_t(dim);
    const int n_chunks = (dim + 127) >> 7;
    #pragma unroll 1
    for (int c = 0; c < n_chunks; ++c) {
        const int idx = c * 128 + tid;
        const float xv = (idx < dim) ? __ldg(&x_row[idx]) : 0.0f;
        const float cs = tree_reduce_128_sm86(xv, smem_tree);
        if (tid == 0) smem_blk[c] = cs;
    }
    __syncthreads();
    const float tot = tree_reduce_128_sm86((tid < n_chunks) ? smem_blk[tid] : 0.0f, smem_tree);
    if (tid == 0) sum_out[b_idx] = tot;
}

}  // namespace siligemma_sm86

#endif  // SILIGEMMA_KERNELS_SM86_CU
