// SPDX-License-Identifier: Apache-2.0
// Track S184 (EXP-Q81): SiliGemma-Native Form B Static Arena, Zero-Anchor Wire
// Layout, Canonical P223c Reduction Trees, P224a IRIS/SECDED Telemetry, & CPU Reference.
//
// Pure C++20 header (compiles standalone under g++ -std=c++20 AND under nvcc -arch=sm_86).
// Zero third-party Python/PyTorch/transformers/cuBLAS anchors.

#ifndef SILIGEMMA_ARENA_HPP
#define SILIGEMMA_ARENA_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace siligemma {

// ----------------------------------------------------------------------------
// 1. Pinned Gemma-4-E2B-it Form B Architectural & Wire Constants
// ----------------------------------------------------------------------------

static constexpr uint32_t SLGB_MAGIC                 = 0x534C4734u;  // "SLG4"
static constexpr uint32_t SLGB_VERSION               = 1u;
static constexpr uint32_t GEMMA4_BOS_TOKEN_ID        = 2u;
static constexpr uint32_t GEMMA4_EOS_TOKEN_ID_0      = 1u;
static constexpr uint32_t GEMMA4_EOS_TOKEN_ID_1      = 106u;
static constexpr uint32_t GEMMA4_PAD_TOKEN_ID        = 0u;
static constexpr uint32_t GEMMA4_VOCAB_SIZE          = 262144u;
static constexpr uint32_t GEMMA4_HIDDEN_SIZE         = 1536u;
static constexpr uint32_t GEMMA4_NUM_LAYERS          = 35u;
static constexpr uint32_t GEMMA4_NUM_KV_SHARED       = 20u;
static constexpr uint32_t GEMMA4_FIRST_SHARED_LAYER  = 15u;          // L15..L34 reuse L13/L14
static constexpr uint32_t GEMMA4_KV_STORE_SLIDING    = 13u;
static constexpr uint32_t GEMMA4_KV_STORE_FULL       = 14u;
static constexpr uint32_t GEMMA4_NUM_ATTN_HEADS      = 8u;
static constexpr uint32_t GEMMA4_NUM_KV_HEADS        = 1u;
static constexpr uint32_t GEMMA4_SLIDING_HEAD_DIM    = 256u;
static constexpr uint32_t GEMMA4_GLOBAL_HEAD_DIM     = 512u;
static constexpr uint32_t GEMMA4_SLIDING_WINDOW      = 512u;
static constexpr uint32_t GEMMA4_INTERMEDIATE_LO     = 6144u;        // L0..14
static constexpr uint32_t GEMMA4_INTERMEDIATE_HI     = 12288u;       // L15..34
static constexpr uint32_t GEMMA4_PLE_DIM             = 256u;
static constexpr float    GEMMA4_FINAL_SOFTCAP       = 30.0f;
static constexpr float    GEMMA4_RMS_NORM_EPS        = 1.0e-6f;
static constexpr float    GEMMA4_ROPE_THETA_SLIDING  = 10000.0f;
static constexpr float    GEMMA4_ROPE_THETA_FULL     = 1000000.0f;
static constexpr float    GEMMA4_FULL_PARTIAL_ROTARY = 0.25f;
static constexpr uint32_t GEMMA4_FULL_ROTATED_PAIRS  = 64u;
static constexpr uint32_t GEMMA4_FULL_NOPE_PAIRS     = 192u;
static constexpr uint32_t GEMMA4_SLID_ROTATED_PAIRS  = 128u;

static constexpr uint32_t LIVE_LINEAR_COUNT          = 276u;
static constexpr uint32_t DEAD_KV_TENSORS_OMITTED    = 60u;          // 20 x (k_proj, v_proj, k_norm)
static constexpr uint64_t LIVE_LINEAR_PARAMS         = 1876819968ULL;
static constexpr uint64_t DEAD_KV_PARAMS_ABSENT      = 18880512ULL;
static constexpr uint32_t NORM_SCALAR_PARAMS_FP32    = 285987u;
static constexpr uint64_t EMBED_TOKENS_PARAMS        = 402653184ULL;
static constexpr uint64_t EMBED_TOKENS_BF16_BYTES    = 805306368ULL;
static constexpr uint64_t LM_HEAD_SHADOW_G32_BYTES   = 226492416ULL;
static constexpr uint64_t PLE_HOST_PINNED_PARAMS     = 2348810240ULL;
static constexpr uint64_t PLE_HOST_PINNED_BYTES      = 4697620480ULL;
static constexpr uint32_t PLE_DMA_BYTES_PER_TOKEN    = 17920u;       // 35 * 256 * 2 B

static constexpr uint64_t WIRE_BYTES_G32_45BPW       = 1055711232ULL;
static constexpr uint64_t WIRE_BYTES_G128_4125BPW    = 967735296ULL;

// Hub & U-4 Loop Constants
static constexpr uint32_t HUB_STEER_START_LAYER      = 10u;
static constexpr uint32_t HUB_STEER_END_LAYER        = 14u;
static constexpr uint32_t LOOP_START_LAYER           = 25u;
static constexpr uint32_t LOOP_END_LAYER             = 33u;
static constexpr uint32_t LOOP_NUM_LAYERS            = 9u;           // L25..L33
static constexpr uint32_t LOOP_NUM_GATES             = 3u;           // q, mlp, ple
static constexpr uint32_t LOOP_GATE_RANK             = 16u;
static constexpr uint32_t LOOP_CONTROLLER_PARAMS     = 2695680u;
static constexpr uint64_t LOOP_CONTROLLER_BF16_BYTES = 5391360ULL;
static constexpr uint64_t LOOP_SEGMENT_LINEAR_PARAMS = 579600384ULL;
static constexpr uint64_t LOOP_SEGMENT_G32_BYTES     = 326025216ULL;
static constexpr uint64_t LOOP_SEGMENT_G128_BYTES    = 298856448ULL;
static constexpr uint32_t MAX_LOOP_PASSES            = 3u;

// Contraction & Safety Thresholds (P224c RiccatiDecoupling, S96, S164, P224a IRIS)
static constexpr double   RICCATI_RHO_LIMIT          = 0.6180;       // rho < 0.6180 (G3 <= 2)
static constexpr float    S164_STEP_RATIO_LIMIT      = 1.5f;         // s_p < 1.5
static constexpr float    LOOP_EXIT_COS_MIN          = 0.75f;
static constexpr uint32_t IRIS_WEYL_MIX              = 0x9E3779B9u;
static constexpr uint32_t LORA_WOODBURY_RANK         = 4u;
static constexpr uint32_t WOODBURY_DIM               = 8u;
static constexpr size_t   WOODBURY_CONST_BYTES       = 256u;         // 8x8 FP32 = 256 B
static constexpr int32_t  BATCH_ENVELOPE_MAX         = 16;

// Exact Arena Partition Constants (PREREG_W22_GEMMA4_FORMB.json)
static constexpr uint64_t KV_RING_SLIDING_BYTES      = 6291456ULL;   // 12 * 512 * 2 * 256 * 2
static constexpr uint64_t PLE_STAGE_BYTES            = 573440ULL;    // 2 * 16 * 17,920 B
static constexpr uint64_t ACTIVATION_SCRATCH_BYTES   = 20217856ULL;  // P7 B=16 static workspace
static constexpr uint64_t CANARY_TELEMETRY_BYTES     = 65536ULL;     // P8 64 KiB Canary+Telemetry

// ----------------------------------------------------------------------------
// 2. P224a IRIS + [16, 11, 4] 64-Byte Telemetry Canary & Watchdog ABI
// ----------------------------------------------------------------------------

struct alignas(64) HoloSphereRazorCanary64 {
    uint32_t sec_count_u32;            // [0..3]   Tier-1 1-bit LSB SEC count
    uint32_t ded_count_u32;            // [4..7]   Tier-2 2-bit LSB DED / IRIS mismatch count
    uint32_t epoch_u32;                // [8..11]  Monotonic token/graph epoch
    uint32_t rollback_token_u32;       // [12..15] Last verified clean token step
    uint32_t non_lsb_canary_u32;       // [16..19] Upper-bit / structural anomaly canary
    uint32_t last_syndrome_u32;        // [20..23] Hi16 = rolling IRIS CRC; Lo16 = SECDED syndrome
    uint32_t ewma_ber_q16_u32;         // [24..27] Q16 EWMA bit-error rate
    uint32_t clean_streak_u32;         // [28..31] Consecutive clean epochs
    uint32_t hold_remaining_u32;       // [32..35] Min-dwell hold epochs
    uint32_t ded_latch_remaining_u32;  // [36..39] DED recovery latch epochs
    uint32_t iris_runtime_crc_u32;     // [40..43] Full 32-bit rolling IRIS runtime state CRC
    uint32_t iris_mismatch_count_u32;  // [44..47] Runtime trace mismatch counter
    uint32_t reserved_u32[4];          // [48..63] Reserved zero padding
};
static_assert(sizeof(HoloSphereRazorCanary64) == 64u, "HoloSphereRazorCanary64 must be 64 B");
static_assert(offsetof(HoloSphereRazorCanary64, sec_count_u32) == 0u, "sec_count offset");
static_assert(offsetof(HoloSphereRazorCanary64, ded_count_u32) == 4u, "ded_count offset");
static_assert(offsetof(HoloSphereRazorCanary64, last_syndrome_u32) == 20u, "last_syndrome offset");
static_assert(offsetof(HoloSphereRazorCanary64, iris_runtime_crc_u32) == 40u, "iris_crc offset");

enum class WatchdogAction : uint32_t {
    CONTINUE = 0u,
    RAMP_SLEW_CAPPED = 1u,
    SAFE_HALT_DED_ROLLBACK = 2u,
    SAFE_HALT_DEADLINE_OVERRUN = 3u,
    SAFE_HALT_BOS_CONTRACT = 4u,
    SAFE_HALT_RICCATI_INSTABILITY = 5u,
};

enum class LinearRole : uint32_t {
    Q_PROJ = 0u,
    K_PROJ = 1u,
    V_PROJ = 2u,
    O_PROJ = 3u,
    GATE_PROJ = 4u,
    UP_PROJ = 5u,
    DOWN_PROJ = 6u,
    PLE_GATE = 7u,
    PLE_PROJ = 8u,
    GLOBAL_PLE_MODEL_PROJ = 9u,
};

enum class ValidationError : int32_t {
    OK = 0,
    ERR_BAD_MAGIC = -101,
    ERR_BAD_VERSION = -102,
    ERR_BOS_NOT_2 = -103,
    ERR_EMPTY_TOKENS = -104,
    ERR_BAD_TOPOLOGY = -105,
    ERR_BAD_GROUP_SIZE = -106,
    ERR_BAD_LINEAR_COUNT = -107,
    ERR_DEAD_KV_PRESENT = -108,
    ERR_MISSING_STORE_KV = -109,
    ERR_BAD_WIRE_BYTES = -110,
    ERR_BAD_KV_SOURCE = -111,
    ERR_BAD_DIMS = -112,
    ERR_DED_SAFE_HALT = -113,
    ERR_IRIS_MISMATCH = -114,
    ERR_RICCATI_RHO_GE_LIMIT = -115,
    ERR_S164_STEP_DIVERGENCE = -116,
    ERR_LOOP_K_INVALID = -117,
    ERR_BAD_TOKEN_ID = -118,
    ERR_BAD_CRC32 = -119,
    ERR_BAD_FILE_SIZE = -120,
};

constexpr uint32_t crc32_ieee_update(uint32_t crc, uint8_t byte) noexcept {
    crc ^= static_cast<uint32_t>(byte);
    for (int k = 0; k < 8; ++k) {
        const uint32_t mask = static_cast<uint32_t>(-static_cast<int32_t>(crc & 1u));
        crc = (crc >> 1) ^ (0xEDB88320u & mask);
    }
    return crc;
}

inline uint32_t crc32_ieee(const void* data, size_t len) noexcept {
    const auto* p = reinterpret_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0u; i < len; ++i) {
        crc = crc32_ieee_update(crc, p[i]);
    }
    return crc ^ 0xFFFFFFFFu;
}

// ----------------------------------------------------------------------------
// 3. Binary Header, Zero-Anchor Descriptor Table, & Static Arena Layout
// ----------------------------------------------------------------------------

struct alignas(64) SlgbHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t bos_token_id;
    uint32_t eos_token_ids[2];
    uint32_t pad_token_id;
    uint32_t vocab_size;
    uint16_t hidden_size;
    uint16_t num_layers;
    uint16_t num_kv_shared_layers;
    uint16_t group_size;
    uint16_t num_live_linears;
    uint16_t num_dead_kv_omitted;
    float    final_logit_softcapping;
    float    rms_norm_eps;
    uint64_t total_wire_bytes;
    uint32_t flags;
    uint32_t header_crc32;
};
static_assert(sizeof(SlgbHeader) == 64u, "SlgbHeader must be 64 B");

struct alignas(16) SiliGemmaLinearDesc {
    uint64_t words_byte_offset;
    uint64_t scales_byte_offset;
    int32_t  out_dim;
    int32_t  in_dim;
    int32_t  group_size;
    uint32_t layer_idx;       // 0..34, or 0xFFFFFFFFu for global PLE projection
    uint32_t role_id;
    uint32_t flags;           // bit0: L10..14 hub steerable, bit1: L25..33 loop segment
};

struct SiliGemmaLayerDesc {
    uint32_t layer_idx;
    uint32_t is_full_attn;
    uint32_t is_kv_shared;
    uint32_t kv_source_layer;
    uint32_t head_dim;
    uint32_t intermediate_size;
    int32_t  linear_idx[9];        // Q,K,V,O,GATE,UP,DOWN,PLE_GATE,PLE_PROJ (-1 when absent)
    int32_t  norm_offsets_f32[8];  // in,post_a,pre_f,post_f,post_p,q_norm,k_norm(-1 if shared),scalar
};

constexpr uint64_t align_up_n(uint64_t n, uint64_t align) noexcept {
    return (n + (align - 1ULL)) & ~(align - 1ULL);
}

struct SiliGemmaArenaLayout {
    uint32_t t_max;
    uint32_t group_size;
    bool     with_lm_head_shadow;

    // Partition offsets & sizes inside single contiguous cudaMalloc slab
    uint64_t off_p1_wire_weights;
    uint64_t sz_p1_wire_weights;
    uint64_t off_p2_embed_bf16;
    uint64_t sz_p2_embed_bf16;
    uint64_t off_p2_lm_shadow;
    uint64_t sz_p2_lm_shadow;
    uint64_t off_p3_norms_f32;
    uint64_t sz_p3_norms_f32;
    uint64_t off_p4_loop_ctrl_bf16;
    uint64_t sz_p4_loop_ctrl_bf16;
    uint64_t off_p5_zed_kv_ring;
    uint64_t sz_p5_zed_kv_ring;
    uint64_t off_p6_ple_stage;
    uint64_t sz_p6_ple_stage;
    uint64_t off_p7_act_scratch;
    uint64_t sz_p7_act_scratch;
    uint64_t off_p8_canary_telem;
    uint64_t sz_p8_canary_telem;
    uint64_t total_arena_bytes;

    // P5 zed sub-layout: L0..14 producers only; L15..34 & K=2,3 reserve 0 B
    uint64_t kv_ring_sliding_offsets[15];
    uint64_t kv_ring_full_offsets[3];   // L4, L9, L14

    // P7 scratch sub-offsets (relative to off_p7_act_scratch)
    uint64_t sub_h_cur_f32;
    uint64_t sub_h_tmp_f32;
    uint64_t sub_h_pre24_f32;
    uint64_t sub_loop_prev_bf16;
    uint64_t sub_q_f32;
    uint64_t sub_k_f32;
    uint64_t sub_v_f32;
    uint64_t sub_ffn_gate_f32;
    uint64_t sub_ffn_up_f32;
    uint64_t sub_ple_proj_bf16;
    uint64_t sub_x_q8_i32;
    uint64_t sub_x_sc_f32;
    uint64_t sub_u4_f4;
    uint64_t sub_m_inv_tmp_f32;
    uint64_t sub_logits_f32;
    uint64_t sub_tok_in_i32;
    uint64_t sub_tok_out_i32;
    uint64_t sub_argmax_score_f32;
    uint64_t p7_used_bytes;
};

constexpr SiliGemmaArenaLayout compute_siligemma_arena_layout(
    uint32_t t_max,
    uint32_t group_size,
    bool with_lm_head_shadow
) noexcept {
    SiliGemmaArenaLayout lay{};
    lay.t_max = t_max;
    lay.group_size = group_size;
    lay.with_lm_head_shadow = with_lm_head_shadow;

    uint64_t cur = 0ULL;

    // Partition 1: 276 live A_16-BlockH4 linears
    lay.off_p1_wire_weights = cur;
    lay.sz_p1_wire_weights = (group_size == 128u) ? WIRE_BYTES_G128_4125BPW : WIRE_BYTES_G32_45BPW;
    cur = align_up_n(cur + lay.sz_p1_wire_weights, 256ULL);

    // Partition 2: Tied BF16 embed_tokens + optional 4.5 bpw lm_head shadow
    lay.off_p2_embed_bf16 = cur;
    lay.sz_p2_embed_bf16 = EMBED_TOKENS_BF16_BYTES;
    cur = align_up_n(cur + lay.sz_p2_embed_bf16, 256ULL);

    lay.off_p2_lm_shadow = cur;
    lay.sz_p2_lm_shadow = with_lm_head_shadow ? LM_HEAD_SHADOW_G32_BYTES : 0ULL;
    cur = align_up_n(cur + lay.sz_p2_lm_shadow, 256ULL);

    // Partition 3: FP32 norms & layer_scalars (285,987 params -> 1,144,064 B aligned)
    lay.off_p3_norms_f32 = cur;
    lay.sz_p3_norms_f32 = align_up_n(uint64_t(NORM_SCALAR_PARAMS_FP32) * sizeof(float), 256ULL);
    cur = align_up_n(cur + lay.sz_p3_norms_f32, 256ULL);

    // Partition 4: U-4 r=16 BF16 L25..33 loop controllers (5,391,360 B)
    lay.off_p4_loop_ctrl_bf16 = cur;
    lay.sz_p4_loop_ctrl_bf16 = LOOP_CONTROLLER_BF16_BYTES;
    cur = align_up_n(cur + lay.sz_p4_loop_ctrl_bf16, 256ULL);

    // Partition 5: zed Static KV Ring (4096 B aligned; L0..14 only, 0 B for L15..34 & K=2,3)
    lay.off_p5_zed_kv_ring = cur;
    uint64_t kv_off = 0ULL;
    int full_slot = 0;
    for (uint32_t l = 0u; l < GEMMA4_FIRST_SHARED_LAYER; ++l) {
        if ((l % 5u) == 4u) {
            lay.kv_ring_full_offsets[full_slot++] = kv_off;
            lay.kv_ring_sliding_offsets[l] = 0ULL;
            kv_off += uint64_t(t_max) * 2ULL * uint64_t(GEMMA4_GLOBAL_HEAD_DIM) * 2ULL;
        } else {
            lay.kv_ring_sliding_offsets[l] = kv_off;
            kv_off += uint64_t(GEMMA4_SLIDING_WINDOW) * 2ULL * uint64_t(GEMMA4_SLIDING_HEAD_DIM) * 2ULL;
        }
    }
    lay.sz_p5_zed_kv_ring = align_up_n(kv_off, 256ULL);
    cur = align_up_n(cur + lay.sz_p5_zed_kv_ring, 256ULL);

    // Partition 6: Double-Buffered Async PLE DMA Staging (2 x 16 x 17,920 B = 573,440 B)
    lay.off_p6_ple_stage = cur;
    lay.sz_p6_ple_stage = PLE_STAGE_BYTES;
    cur = align_up_n(cur + lay.sz_p6_ple_stage, 256ULL);

    // Partition 7: Activations, u[4]/x_q8, U-4 frames, & Logits (B=16 envelope)
    lay.off_p7_act_scratch = cur;
    lay.sz_p7_act_scratch = ACTIVATION_SCRATCH_BYTES;
    uint64_t s = 0ULL;
    lay.sub_h_cur_f32        = s; s += align_up_n(16ULL * 1536ULL * 4ULL, 256ULL);
    lay.sub_h_tmp_f32        = s; s += align_up_n(16ULL * 1536ULL * 4ULL, 256ULL);
    lay.sub_h_pre24_f32      = s; s += align_up_n(16ULL * 1536ULL * 4ULL, 256ULL);
    lay.sub_loop_prev_bf16   = s; s += align_up_n(9ULL * 16ULL * 1536ULL * 2ULL, 256ULL);
    lay.sub_q_f32            = s; s += align_up_n(16ULL * 4096ULL * 4ULL, 256ULL);
    lay.sub_k_f32            = s; s += align_up_n(16ULL * 512ULL * 4ULL, 256ULL);
    lay.sub_v_f32            = s; s += align_up_n(16ULL * 512ULL * 4ULL, 256ULL);
    lay.sub_ffn_gate_f32     = s; s += align_up_n(16ULL * 12288ULL * 4ULL, 256ULL);
    lay.sub_ffn_up_f32       = s; s += align_up_n(16ULL * 12288ULL * 4ULL, 256ULL);
    lay.sub_ple_proj_bf16    = s; s += align_up_n(16ULL * 8960ULL * 2ULL, 256ULL);
    lay.sub_x_q8_i32         = s; s += align_up_n(16ULL * (12288ULL / 4ULL) * 4ULL, 256ULL);
    lay.sub_x_sc_f32         = s; s += align_up_n(16ULL * (12288ULL / 32ULL) * 4ULL, 256ULL);
    lay.sub_u4_f4            = s; s += align_up_n(16ULL * sizeof(float) * 4ULL, 256ULL);
    lay.sub_m_inv_tmp_f32    = s; s += align_up_n(64ULL * sizeof(float), 256ULL);
    lay.sub_logits_f32       = s; s += align_up_n(16ULL * uint64_t(GEMMA4_VOCAB_SIZE) * 4ULL, 256ULL);
    lay.sub_tok_in_i32       = s; s += 256ULL;
    lay.sub_tok_out_i32      = s; s += 256ULL;
    lay.sub_argmax_score_f32 = s; s += 256ULL;
    lay.p7_used_bytes        = s;
    cur = align_up_n(cur + lay.sz_p7_act_scratch, 256ULL);

    // Partition 8: HoloSphereRazorCanary64 + Telemetry Slab
    lay.off_p8_canary_telem = cur;
    lay.sz_p8_canary_telem = CANARY_TELEMETRY_BYTES;
    cur += lay.sz_p8_canary_telem;

    lay.total_arena_bytes = cur;
    return lay;
}

// Compile-time verification against PREREG_W22_GEMMA4_FORMB.json exact byte budgets
static constexpr SiliGemmaArenaLayout kArena16K = compute_siligemma_arena_layout(16384u, 32u, false);
static constexpr SiliGemmaArenaLayout kArena64KShadow = compute_siligemma_arena_layout(65536u, 32u, true);
static constexpr SiliGemmaArenaLayout kArena128K = compute_siligemma_arena_layout(131072u, 32u, false);

static_assert(kArena16K.sz_p5_zed_kv_ring == 106954752ULL, "T=16384 KV ring bytes must equal PREREG");
static_assert(kArena64KShadow.sz_p5_zed_kv_ring == 408944640ULL, "T=65536 KV ring bytes must equal PREREG");
static_assert(kArena128K.sz_p5_zed_kv_ring == 811597824ULL, "T=131072 KV ring bytes must equal PREREG");
static_assert(kArena16K.p7_used_bytes <= kArena16K.sz_p7_act_scratch, "P7 workspace must fit in 20,217,856 B");
static_assert(kArena16K.total_arena_bytes == 1995364608ULL, "16K arena must match PREREG exact byte count");
static_assert(kArena64KShadow.total_arena_bytes == 2523846912ULL, "64K+shadow arena must match PREREG exact byte count");
static_assert(kArena128K.total_arena_bytes == 2700007680ULL, "128K arena must match PREREG exact byte count");

// ----------------------------------------------------------------------------
// 4. Zero-Anchor Descriptor Builder & Fail-Closed Validation
// ----------------------------------------------------------------------------

inline uint64_t linear_wire_words_bytes(int32_t out_dim, int32_t in_dim) noexcept {
    // 16 nibbles per uint64_t (8 B / 16 weights) regardless of group_size
    return uint64_t(out_dim) * uint64_t(in_dim / 16) * sizeof(uint64_t);
}

inline uint64_t linear_wire_scales_bytes(int32_t out_dim, int32_t in_dim, int32_t group_size) noexcept {
    return uint64_t(out_dim) * uint64_t(in_dim / group_size) * sizeof(uint16_t);
}

inline uint32_t compute_slgb_header_crc32(const SlgbHeader& hdr_in) noexcept {
    SlgbHeader hdr = hdr_in;
    hdr.header_crc32 = 0u;
    return crc32_ieee(&hdr, sizeof(hdr));
}

inline uint64_t expected_slgb_file_bytes(const SlgbHeader& hdr) noexcept {
    const uint64_t hdr_bytes = sizeof(SlgbHeader)
        + static_cast<uint64_t>(GEMMA4_NUM_LAYERS) * sizeof(SiliGemmaLayerDesc)
        + static_cast<uint64_t>(LIVE_LINEAR_COUNT) * sizeof(SiliGemmaLinearDesc);
    uint64_t off = align_up_n(hdr_bytes, 256ULL);
    off = align_up_n(off + hdr.total_wire_bytes, 256ULL);
    off += EMBED_TOKENS_BF16_BYTES;
    if (hdr.flags & 1u) {
        off = align_up_n(off, 256ULL);
        off += LM_HEAD_SHADOW_G32_BYTES;
    }
    off = align_up_n(off, 256ULL);
    off += static_cast<uint64_t>(NORM_SCALAR_PARAMS_FP32) * sizeof(float);
    if (hdr.flags & 2u) {
        off = align_up_n(off, 256ULL);
        off += LOOP_CONTROLLER_BF16_BYTES;
    }
    return off;
}

inline ValidationError build_gemma4_descriptor_table(
    uint32_t group_size,
    SlgbHeader& hdr,
    std::array<SiliGemmaLayerDesc, GEMMA4_NUM_LAYERS>& layers,
    std::vector<SiliGemmaLinearDesc>& linears,
    uint32_t flags = 0u
) {
    if (group_size != 32u && group_size != 128u) {
        return ValidationError::ERR_BAD_GROUP_SIZE;
    }
    if (flags > 3u) {
        return ValidationError::ERR_BAD_TOPOLOGY;
    }

    linears.clear();
    linears.reserve(LIVE_LINEAR_COUNT);

    uint64_t total_params = 0ULL;
    uint64_t total_words_bytes = 0ULL;
    uint64_t total_scales_bytes = 0ULL;
    int32_t norm_cursor = 0;

    auto push_linear = [&](uint32_t l_idx, LinearRole role, int32_t m, int32_t k, uint32_t l_flags) -> int32_t {
        const int32_t idx = static_cast<int32_t>(linears.size());
        SiliGemmaLinearDesc d{};
        d.words_byte_offset = total_words_bytes;
        d.scales_byte_offset = total_scales_bytes;
        d.out_dim = m;
        d.in_dim = k;
        d.group_size = static_cast<int32_t>(group_size);
        d.layer_idx = l_idx;
        d.role_id = static_cast<uint32_t>(role);
        d.flags = l_flags;
        total_words_bytes += linear_wire_words_bytes(m, k);
        total_scales_bytes += linear_wire_scales_bytes(m, k, static_cast<int32_t>(group_size));
        total_params += uint64_t(m) * uint64_t(k);
        linears.push_back(d);
        return idx;
    };

    for (uint32_t l = 0u; l < GEMMA4_NUM_LAYERS; ++l) {
        SiliGemmaLayerDesc& ly = layers[l];
        ly.layer_idx = l;
        ly.is_full_attn = ((l % 5u) == 4u) ? 1u : 0u;
        ly.is_kv_shared = (l >= GEMMA4_FIRST_SHARED_LAYER) ? 1u : 0u;
        ly.kv_source_layer = ly.is_kv_shared ? (ly.is_full_attn ? GEMMA4_KV_STORE_FULL : GEMMA4_KV_STORE_SLIDING) : l;
        ly.head_dim = ly.is_full_attn ? GEMMA4_GLOBAL_HEAD_DIM : GEMMA4_SLIDING_HEAD_DIM;
        ly.intermediate_size = ly.is_kv_shared ? GEMMA4_INTERMEDIATE_HI : GEMMA4_INTERMEDIATE_LO;
        for (int i = 0; i < 9; ++i) ly.linear_idx[i] = -1;
        for (int i = 0; i < 8; ++i) ly.norm_offsets_f32[i] = -1;

        uint32_t l_flags = 0u;
        if (l >= HUB_STEER_START_LAYER && l <= HUB_STEER_END_LAYER) l_flags |= 1u;
        if (l >= LOOP_START_LAYER && l <= LOOP_END_LAYER) l_flags |= 2u;

        const int32_t H = static_cast<int32_t>(GEMMA4_HIDDEN_SIZE);
        const int32_t Q_DIM = static_cast<int32_t>(GEMMA4_NUM_ATTN_HEADS * ly.head_dim);
        const int32_t KV_DIM = static_cast<int32_t>(GEMMA4_NUM_KV_HEADS * ly.head_dim);
        const int32_t FF = static_cast<int32_t>(ly.intermediate_size);
        const int32_t PLE = static_cast<int32_t>(GEMMA4_PLE_DIM);

        ly.linear_idx[0] = push_linear(l, LinearRole::Q_PROJ, Q_DIM, H, l_flags);
        if (!ly.is_kv_shared) {
            ly.linear_idx[1] = push_linear(l, LinearRole::K_PROJ, KV_DIM, H, l_flags);
            ly.linear_idx[2] = push_linear(l, LinearRole::V_PROJ, KV_DIM, H, l_flags);
        }
        // L15..34 k_proj & v_proj stay -1 (physically absent from wire format!)
        ly.linear_idx[3] = push_linear(l, LinearRole::O_PROJ, H, Q_DIM, l_flags);
        ly.linear_idx[4] = push_linear(l, LinearRole::GATE_PROJ, FF, H, l_flags);
        ly.linear_idx[5] = push_linear(l, LinearRole::UP_PROJ, FF, H, l_flags);
        ly.linear_idx[6] = push_linear(l, LinearRole::DOWN_PROJ, H, FF, l_flags);
        ly.linear_idx[7] = push_linear(l, LinearRole::PLE_GATE, PLE, H, l_flags);
        ly.linear_idx[8] = push_linear(l, LinearRole::PLE_PROJ, H, PLE, l_flags);

        // Norm offsets: 5*1536 + q_norm + optional k_norm + scalar
        for (int n = 0; n < 5; ++n) {
            ly.norm_offsets_f32[n] = norm_cursor;
            norm_cursor += H;
        }
        ly.norm_offsets_f32[5] = norm_cursor;
        norm_cursor += static_cast<int32_t>(ly.head_dim);
        if (!ly.is_kv_shared) {
            ly.norm_offsets_f32[6] = norm_cursor;
            norm_cursor += static_cast<int32_t>(ly.head_dim);
        }
        ly.norm_offsets_f32[7] = norm_cursor;
        norm_cursor += 1;
    }

    // 276th live linear: global per_layer_model_projection [35*256=8960, 1536]
    push_linear(0xFFFFFFFFu, LinearRole::GLOBAL_PLE_MODEL_PROJ,
                static_cast<int32_t>(GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM),
                static_cast<int32_t>(GEMMA4_HIDDEN_SIZE), 0u);
    norm_cursor += static_cast<int32_t>(GEMMA4_HIDDEN_SIZE); // final model.norm
    norm_cursor += static_cast<int32_t>(GEMMA4_PLE_DIM);     // per_layer_projection_norm

    // Convert relative scales offset to absolute byte offsets after contiguous words slab
    for (auto& d : linears) {
        d.scales_byte_offset += total_words_bytes;
    }

    if (linears.size() != LIVE_LINEAR_COUNT || total_params != LIVE_LINEAR_PARAMS) {
        return ValidationError::ERR_BAD_LINEAR_COUNT;
    }
    if (static_cast<uint32_t>(norm_cursor) != NORM_SCALAR_PARAMS_FP32) {
        return ValidationError::ERR_BAD_TOPOLOGY;
    }

    const uint64_t wire_bytes = total_words_bytes + total_scales_bytes;
    const uint64_t expected_wire = (group_size == 32u) ? WIRE_BYTES_G32_45BPW : WIRE_BYTES_G128_4125BPW;
    if (wire_bytes != expected_wire) {
        return ValidationError::ERR_BAD_WIRE_BYTES;
    }

    std::memset(&hdr, 0, sizeof(hdr));
    hdr.magic = SLGB_MAGIC;
    hdr.version = SLGB_VERSION;
    hdr.bos_token_id = GEMMA4_BOS_TOKEN_ID;
    hdr.eos_token_ids[0] = GEMMA4_EOS_TOKEN_ID_0;
    hdr.eos_token_ids[1] = GEMMA4_EOS_TOKEN_ID_1;
    hdr.pad_token_id = GEMMA4_PAD_TOKEN_ID;
    hdr.vocab_size = GEMMA4_VOCAB_SIZE;
    hdr.hidden_size = static_cast<uint16_t>(GEMMA4_HIDDEN_SIZE);
    hdr.num_layers = static_cast<uint16_t>(GEMMA4_NUM_LAYERS);
    hdr.num_kv_shared_layers = static_cast<uint16_t>(GEMMA4_NUM_KV_SHARED);
    hdr.group_size = static_cast<uint16_t>(group_size);
    hdr.num_live_linears = static_cast<uint16_t>(LIVE_LINEAR_COUNT);
    hdr.num_dead_kv_omitted = static_cast<uint16_t>(DEAD_KV_TENSORS_OMITTED);
    hdr.final_logit_softcapping = GEMMA4_FINAL_SOFTCAP;
    hdr.rms_norm_eps = GEMMA4_RMS_NORM_EPS;
    hdr.total_wire_bytes = wire_bytes;
    hdr.flags = flags;
    hdr.header_crc32 = compute_slgb_header_crc32(hdr);
    return ValidationError::OK;
}

inline ValidationError validate_runtime_contract(
    const SlgbHeader& hdr,
    const std::array<SiliGemmaLayerDesc, GEMMA4_NUM_LAYERS>& layers,
    const std::vector<SiliGemmaLinearDesc>& linears,
    const uint32_t* tokens,
    size_t n_tokens
) noexcept {
    if (hdr.magic != SLGB_MAGIC) return ValidationError::ERR_BAD_MAGIC;
    if (hdr.version != SLGB_VERSION) return ValidationError::ERR_BAD_VERSION;
    if (hdr.bos_token_id != GEMMA4_BOS_TOKEN_ID) return ValidationError::ERR_BOS_NOT_2;
    if (tokens == nullptr || n_tokens == 0u) return ValidationError::ERR_EMPTY_TOKENS;
    if (tokens[0] != GEMMA4_BOS_TOKEN_ID) return ValidationError::ERR_BOS_NOT_2;
    for (size_t i = 0u; i < n_tokens; ++i) {
        if (tokens[i] >= hdr.vocab_size) return ValidationError::ERR_BAD_TOKEN_ID;
    }
    if (hdr.hidden_size != GEMMA4_HIDDEN_SIZE || hdr.num_layers != GEMMA4_NUM_LAYERS ||
        hdr.num_kv_shared_layers != GEMMA4_NUM_KV_SHARED || hdr.vocab_size != GEMMA4_VOCAB_SIZE ||
        hdr.eos_token_ids[0] != GEMMA4_EOS_TOKEN_ID_0 || hdr.eos_token_ids[1] != GEMMA4_EOS_TOKEN_ID_1 ||
        hdr.pad_token_id != GEMMA4_PAD_TOKEN_ID || hdr.final_logit_softcapping != GEMMA4_FINAL_SOFTCAP ||
        hdr.rms_norm_eps != GEMMA4_RMS_NORM_EPS || hdr.flags > 3u) {
        return ValidationError::ERR_BAD_TOPOLOGY;
    }
    if (hdr.header_crc32 != 0u && hdr.header_crc32 != compute_slgb_header_crc32(hdr)) {
        return ValidationError::ERR_BAD_CRC32;
    }
    if (hdr.group_size != 32u && hdr.group_size != 128u) return ValidationError::ERR_BAD_GROUP_SIZE;
    if (hdr.num_live_linears != LIVE_LINEAR_COUNT || linears.size() != LIVE_LINEAR_COUNT ||
        hdr.num_dead_kv_omitted != DEAD_KV_TENSORS_OMITTED) {
        return ValidationError::ERR_BAD_LINEAR_COUNT;
    }
    const uint64_t expected_wire = (hdr.group_size == 32u) ? WIRE_BYTES_G32_45BPW : WIRE_BYTES_G128_4125BPW;
    if (hdr.total_wire_bytes != expected_wire) return ValidationError::ERR_BAD_WIRE_BYTES;

    for (uint32_t l = 0u; l < GEMMA4_NUM_LAYERS; ++l) {
        const auto& ly = layers[l];
        if (ly.layer_idx != l) return ValidationError::ERR_BAD_TOPOLOGY;
        const bool expected_shared = (l >= GEMMA4_FIRST_SHARED_LAYER);
        if ((ly.is_kv_shared != 0u) != expected_shared) return ValidationError::ERR_BAD_TOPOLOGY;
        if (expected_shared) {
            if (ly.linear_idx[1] >= 0 || ly.linear_idx[2] >= 0 || ly.norm_offsets_f32[6] >= 0) {
                return ValidationError::ERR_DEAD_KV_PRESENT;
            }
            const uint32_t exp_src = ly.is_full_attn ? GEMMA4_KV_STORE_FULL : GEMMA4_KV_STORE_SLIDING;
            if (ly.kv_source_layer != exp_src) return ValidationError::ERR_BAD_KV_SOURCE;
        } else {
            if (ly.linear_idx[1] < 0 || ly.linear_idx[2] < 0 || ly.norm_offsets_f32[6] < 0) {
                return ValidationError::ERR_MISSING_STORE_KV;
            }
            if (ly.kv_source_layer != l) return ValidationError::ERR_BAD_KV_SOURCE;
        }
    }
    for (const auto& d : linears) {
        if (d.in_dim <= 0 || d.out_dim <= 0 || (d.in_dim % 128) != 0 || (d.out_dim % 16) != 0 ||
            d.group_size != static_cast<int32_t>(hdr.group_size)) {
            return ValidationError::ERR_BAD_DIMS;
        }
    }
    return ValidationError::OK;
}

inline ValidationError validate_slgb_file_layout(
    const SlgbHeader& hdr,
    const std::array<SiliGemmaLayerDesc, GEMMA4_NUM_LAYERS>& layers,
    const std::vector<SiliGemmaLinearDesc>& linears,
    uint64_t actual_file_size,
    bool require_crc32 = true
) {
    const uint32_t dummy_tok[1] = {GEMMA4_BOS_TOKEN_ID};
    const ValidationError base_err = validate_runtime_contract(hdr, layers, linears, dummy_tok, 1u);
    if (base_err != ValidationError::OK) return base_err;
    if (require_crc32 && hdr.header_crc32 == 0u) return ValidationError::ERR_BAD_CRC32;
    if (hdr.header_crc32 != compute_slgb_header_crc32(hdr)) return ValidationError::ERR_BAD_CRC32;
    if (actual_file_size != expected_slgb_file_bytes(hdr)) return ValidationError::ERR_BAD_FILE_SIZE;

    SlgbHeader exp_hdr{};
    std::array<SiliGemmaLayerDesc, GEMMA4_NUM_LAYERS> exp_layers{};
    std::vector<SiliGemmaLinearDesc> exp_linears;
    const ValidationError build_err = build_gemma4_descriptor_table(
        static_cast<uint32_t>(hdr.group_size), exp_hdr, exp_layers, exp_linears, hdr.flags
    );
    if (build_err != ValidationError::OK) return build_err;
    if (std::memcmp(&hdr, &exp_hdr, sizeof(SlgbHeader)) != 0) return ValidationError::ERR_BAD_TOPOLOGY;
    if (std::memcmp(layers.data(), exp_layers.data(), sizeof(exp_layers)) != 0) return ValidationError::ERR_BAD_TOPOLOGY;
    if (linears.size() != exp_linears.size() ||
        std::memcmp(linears.data(), exp_linears.data(), exp_linears.size() * sizeof(SiliGemmaLinearDesc)) != 0) {
        return ValidationError::ERR_BAD_WIRE_BYTES;
    }
    return ValidationError::OK;
}

// ----------------------------------------------------------------------------
// 5. Pure-CPU C++20 Reference Primitives (0-ULP Trees, SECDED, IRIS, Riccati)
// ----------------------------------------------------------------------------

inline size_t next_power_of_two(size_t n) noexcept {
    if (n <= 1u) return 1u;
    size_t p = 1u;
    while (p < n) p <<= 1u;
    return p;
}

// P223c 0-ULP Fixed-Shape 7-Stage Binary Reduction Tree (matches tree_reduce_128_sm86 bit-for-bit)
inline float tree_reduce_128_cpu(std::array<float, 128>& tile) noexcept {
    for (size_t half = 64u; half >= 1u; half >>= 1u) {
        for (size_t t = 0u; t < half; ++t) {
            tile[t] = tile[t] + tile[t + half];
        }
    }
    return tile[0];
}

// Block-128 hierarchical canonical tree matching CUDA canonical_sum_sq_d128 & tree_reduce_128_sm86 (0-ULP).
inline float canonical_tree_sum_block128_f32(const float* data, size_t n) {
    if (n == 0u || data == nullptr) return 0.0f;
    const size_t n_blocks = (n + 127u) / 128u;
    if (n_blocks <= 128u) {
        std::array<float, 128> outer{};
        outer.fill(0.0f);
        std::array<float, 128> tile{};
        for (size_t b = 0u; b < n_blocks; ++b) {
            tile.fill(0.0f);
            const size_t base = b * 128u;
            const size_t lim = (base + 128u <= n) ? 128u : (n - base);
            for (size_t t = 0u; t < lim; ++t) tile[t] = data[base + t];
            outer[b] = tree_reduce_128_cpu(tile);
        }
        return tree_reduce_128_cpu(outer);
    }
    std::vector<float> outer_vec(n_blocks, 0.0f);
    std::array<float, 128> tile{};
    for (size_t b = 0u; b < n_blocks; ++b) {
        tile.fill(0.0f);
        const size_t base = b * 128u;
        const size_t lim = (base + 128u <= n) ? 128u : (n - base);
        for (size_t t = 0u; t < lim; ++t) tile[t] = data[base + t];
        outer_vec[b] = tree_reduce_128_cpu(tile);
    }
    return canonical_tree_sum_block128_f32(outer_vec.data(), n_blocks);
}

inline float canonical_tree_sum_f32(const float* data, size_t n) {
    return canonical_tree_sum_block128_f32(data, n);
}

inline void canonical_tree_rmsnorm_f32(
    const float* x,
    const float* gamma,
    float* out,
    size_t d_model,
    float eps = GEMMA4_RMS_NORM_EPS
) {
    if (d_model == 0u) return;
    std::vector<float> sq(d_model, 0.0f);
    for (size_t i = 0u; i < d_model; ++i) sq[i] = x[i] * x[i];
    const float sum_sq = canonical_tree_sum_block128_f32(sq.data(), d_model);
    const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(d_model) + eps);
    for (size_t i = 0u; i < d_model; ++i) {
        out[i] = (x[i] * inv_rms) * (gamma != nullptr ? gamma[i] : 1.0f);
    }
}

inline void h4_apply_4_f32(const float in[4], float out[4]) noexcept {
    const float s01 = in[0] + in[1];
    const float d01 = in[0] - in[1];
    const float s23 = in[2] + in[3];
    const float d23 = in[2] - in[3];
    out[0] = 0.5f * (s01 + s23);
    out[1] = 0.5f * (d01 + d23);
    out[2] = 0.5f * (s01 - s23);
    out[3] = 0.5f * (d01 - d23);
}

inline uint32_t a16_encode_data11_cpu(uint32_t data11) noexcept {
    static constexpr int kDataPos[11] = {3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15};
    static constexpr uint32_t kMasks[4] = {0xAAAAu, 0xCCCCu, 0xF0F0u, 0xFF00u};
    static constexpr uint32_t kPbits[4] = {1u, 2u, 4u, 8u};
    uint32_t odd16 = 0u;
    for (int i = 0; i < 11; ++i) {
        if ((data11 >> i) & 1u) odd16 |= (1u << kDataPos[i]);
    }
    for (int p = 0; p < 4; ++p) {
        if ((__builtin_popcount(odd16 & kMasks[p]) & 1) != 0) odd16 |= (1u << kPbits[p]);
    }
    if ((__builtin_popcount(odd16) & 1) != 0) odd16 |= 1u;
    return odd16;
}

inline uint32_t a16_secded_syndrome_cpu(uint64_t w64) noexcept {
    const uint32_t w0 = static_cast<uint32_t>(w64);
    const uint32_t w1 = static_cast<uint32_t>(w64 >> 32);
    // Exact match to GPU 1-LOP3 (0x28: (a ^ b) & c) + 5-POPC
    const uint32_t px = (w0 ^ w1) & 0x11111111u;
    const uint32_t p0 = static_cast<uint32_t>(__builtin_popcount(px)) & 1u;
    const uint32_t s0 = static_cast<uint32_t>(__builtin_popcount(px & 0x10101010u)) & 1u;
    const uint32_t s1 = static_cast<uint32_t>(__builtin_popcount(px & 0x11001100u)) & 1u;
    const uint32_t s2 = static_cast<uint32_t>(__builtin_popcount(px & 0x11110000u)) & 1u;
    const uint32_t s3 = static_cast<uint32_t>(__builtin_popcount(w1 & 0x11111111u)) & 1u;
    return p0 | (s0 << 1) | (s1 << 2) | (s2 << 3) | (s3 << 4);
}

inline uint32_t iris_mix_word_cpu(uint32_t state, uint32_t word) noexcept {
    uint32_t x = state ^ (word + IRIS_WEYL_MIX + (state << 6) + (state >> 2));
    x ^= (x >> 16);
    x *= 0x85EBCA6Bu;
    x ^= (x >> 13);
    x *= 0xC2B2AE35u;
    x ^= (x >> 16);
    return x;
}

inline uint32_t float_bits_u32(float v) noexcept {
    uint32_t u = 0u;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

inline float gelu_pytorch_tanh_cpu(float g) noexcept {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCoeff = 0.044715f;
    const float inner = kSqrt2OverPi * (g + kCoeff * g * g * g);
    return 0.5f * g * (1.0f + std::tanh(inner));
}

// Pure-CPU C++20 reference for fused Sandwich-RMSNorm + Block-H4 + A_16 GEMV (g=32 or g=128)
inline void sandwich_h4_a16_gemv_ref(
    const float* x_in,
    const float* rms_gamma,
    const uint64_t* w_packed_u64,
    const float* w_scales_f32,
    float* y_out,
    int out_dim,
    int in_dim,
    int group_size,
    float eps = GEMMA4_RMS_NORM_EPS
) {
    std::vector<float> x_norm(static_cast<size_t>(in_dim), 0.0f);
    canonical_tree_rmsnorm_f32(x_in, rms_gamma, x_norm.data(), static_cast<size_t>(in_dim), eps);
    std::vector<float> x_rot(static_cast<size_t>(in_dim), 0.0f);
    for (int i = 0; i < in_dim; i += 4) {
        h4_apply_4_f32(&x_norm[static_cast<size_t>(i)], &x_rot[static_cast<size_t>(i)]);
    }
    const int n_blk16 = in_dim / 16;
    const int n_grp_w = in_dim / group_size;
    const int blk_per_grp = group_size / 16;
    for (int row = 0; row < out_dim; ++row) {
        std::array<float, 128> acc_lanes{};
        acc_lanes.fill(0.0f);
        const uint64_t* r_u64 = w_packed_u64 + static_cast<size_t>(row) * static_cast<size_t>(n_blk16);
        const float* r_sc = w_scales_f32 + static_cast<size_t>(row) * static_cast<size_t>(n_grp_w);
        for (int blk = 0; blk < n_blk16; ++blk) {
            const uint64_t w64 = r_u64[blk];
            const float sc = r_sc[blk / blk_per_grp];
            float dot16 = 0.0f;
            for (int j = 0; j < 16; ++j) {
                const int32_t nib = static_cast<int32_t>((w64 >> (4 * j)) & 0xFu);
                const int32_t w_i = (nib << 28) >> 28;
                dot16 += static_cast<float>(w_i) * x_rot[static_cast<size_t>(blk * 16 + j)];
            }
            acc_lanes[static_cast<size_t>(blk & 127)] += dot16 * sc;
        }
        y_out[row] = tree_reduce_128_cpu(acc_lanes);
    }
}

inline ValidationError evaluate_riccati_loop_guard(
    const float* prev_exit,
    const float* cur_exit,
    const float* prev_delta,
    size_t dim,
    uint32_t pass_idx,
    double& out_rho_hat,
    float& out_step_ratio,
    float& out_cos
) {
    if (pass_idx < 1u || pass_idx > MAX_LOOP_PASSES) return ValidationError::ERR_LOOP_K_INVALID;
    if (prev_exit == nullptr || cur_exit == nullptr || dim == 0u) return ValidationError::ERR_BAD_DIMS;
    if (pass_idx >= 3u && prev_delta == nullptr) return ValidationError::ERR_BAD_DIMS;

    std::vector<float> sq_prev(dim, 0.0f), sq_cur(dim, 0.0f), sq_diff(dim, 0.0f), dot_pc(dim, 0.0f);
    std::vector<float> sq_pdelta(dim, 0.0f);
    for (size_t i = 0u; i < dim; ++i) {
        const float a = prev_exit[i];
        const float b = cur_exit[i];
        const float d = b - a;
        sq_prev[i] = a * a;
        sq_cur[i] = b * b;
        sq_diff[i] = d * d;
        dot_pc[i] = a * b;
        if (prev_delta != nullptr) sq_pdelta[i] = prev_delta[i] * prev_delta[i];
    }
    const float n_prev = std::sqrt(canonical_tree_sum_block128_f32(sq_prev.data(), dim));
    const float n_cur = std::sqrt(canonical_tree_sum_block128_f32(sq_cur.data(), dim));
    const float n_diff = std::sqrt(canonical_tree_sum_block128_f32(sq_diff.data(), dim));
    const float dot_val = canonical_tree_sum_block128_f32(dot_pc.data(), dim);

    out_step_ratio = n_diff / (n_prev + 1.0e-6f);
    out_cos = (n_prev > 1.0e-12f && n_cur > 1.0e-12f) ? (dot_val / (n_prev * n_cur)) : 0.0f;

    out_rho_hat = 0.0;
    if (prev_delta != nullptr) {
        const float n_pd = std::sqrt(canonical_tree_sum_block128_f32(sq_pdelta.data(), dim));
        out_rho_hat = (n_pd > 1.0e-12f) ? (static_cast<double>(n_diff) / static_cast<double>(n_pd)) : (0.0 / 0.0);
        if (!std::isfinite(out_rho_hat) || out_rho_hat >= RICCATI_RHO_LIMIT) {
            return ValidationError::ERR_RICCATI_RHO_GE_LIMIT;
        }
    }

    if (!std::isfinite(out_step_ratio) || out_step_ratio >= S164_STEP_RATIO_LIMIT || out_cos < LOOP_EXIT_COS_MIN) {
        return ValidationError::ERR_S164_STEP_DIVERGENCE;
    }
    return ValidationError::OK;
}

inline WatchdogAction evaluate_canary_watchdog(
    const HoloSphereRazorCanary64& canary,
    uint32_t expected_iris_crc,
    bool check_iris
) noexcept {
    if (canary.ded_count_u32 > 0u) return WatchdogAction::SAFE_HALT_DED_ROLLBACK;
    if (check_iris && canary.iris_runtime_crc_u32 != expected_iris_crc) {
        return WatchdogAction::SAFE_HALT_DED_ROLLBACK;
    }
    if (canary.iris_mismatch_count_u32 > 0u) return WatchdogAction::SAFE_HALT_DED_ROLLBACK;
    return WatchdogAction::CONTINUE;
}

}  // namespace siligemma

#endif  // SILIGEMMA_ARENA_HPP
