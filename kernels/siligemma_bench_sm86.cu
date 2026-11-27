// SPDX-License-Identifier: Apache-2.0
// Track S184 (EXP-Q81): SiliGemma-Native Form B Benchmark & Verification Harness
//
// Zero Python / PyTorch / transformers / cuBLAS / Form A CSR dependencies.
// Supports:
//   - Pure CPU self-test (`--cpu-selftest`, 0 CUDA runtime calls, safe under CUDA_VISIBLE_DEVICES="")
//   - Full RTX 3080 sm_86 CUDA Graph benchmark (`--run-live-gpu` for KG-GFB1..KG-GFB4)

#include "siligemma_kernels_sm86.cu"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <malloc.h>
#include <numeric>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

#define SG_CUDA_OK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err__ = (call);                                                         \
        if (err__ != cudaSuccess) {                                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,           \
                         cudaGetErrorString(err__));                                        \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while (0)

#define SG_REQUIRE(cond, msg)                                                               \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::fprintf(stderr, "Contract check failed at %s:%d: %s\n",                    \
                         __FILE__, __LINE__, (msg));                                        \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while (0)

struct SplitMix64 {
    uint64_t s;
    explicit SplitMix64(uint64_t seed) : s(seed) {}
    uint64_t next_u64() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    float next_sym_f32() {
        const uint64_t r = next_u64() >> 40;
        return (static_cast<float>(r) * (1.0f / 8388608.0f)) - 1.0f;
    }
};

static uint64_t pack_a16_codeword_rng(uint32_t data11, SplitMix64& rng) {
    const uint32_t odd16 = siligemma::a16_encode_data11_cpu(data11 & 0x7FFu);
    uint64_t w64 = 0ULL;
    for (int j = 0; j < 16; ++j) {
        const int lsb = static_cast<int>((odd16 >> j) & 1u);
        const int k = static_cast<int>(rng.next_u64() % 7ULL) - 3;
        int v = 2 * k + lsb;
        if (v < -7) v += 2;
        if (v > 7) v -= 2;
        w64 |= (uint64_t(uint32_t(v) & 0xFu) << (4 * j));
    }
    return w64;
}

static float bf16_bits_to_f32(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static uint16_t f32_to_f16_bits(float v) {
    const __half h = __float2half(v);
    const __half_raw rw(h);
    return rw.x;
}

static float f16_bits_to_f32(uint16_t b) {
    __half_raw rw{};
    rw.x = b;
    return __half2float(__half(rw));
}

static void pread_exact(int fd, void* dst, size_t nbytes, uint64_t offset) {
    uint8_t* p = reinterpret_cast<uint8_t*>(dst);
    size_t done = 0u;
    while (done < nbytes) {
        const size_t chunk = std::min<size_t>(nbytes - done, 16ULL * 1024ULL * 1024ULL);
        const ssize_t got = ::pread(fd, p + done, chunk, static_cast<off_t>(offset + done));
        SG_REQUIRE(got > 0, "pread failed reading binary file");
        done += static_cast<size_t>(got);
    }
}

static uint64_t next_multiple_256(uint64_t x) {
    return siligemma::align_up_n(x, 256ULL);
}

static int run_cpu_contract_selftest() {
    siligemma::SlgbHeader hdr{};
    std::array<siligemma::SiliGemmaLayerDesc, siligemma::GEMMA4_NUM_LAYERS> layers{};
    std::vector<siligemma::SiliGemmaLinearDesc> linears;

    if (siligemma::build_gemma4_descriptor_table(32u, hdr, layers, linears) != siligemma::ValidationError::OK) {
        std::fprintf(stderr, "Failed to build g=32 descriptor table\n");
        return 1;
    }
    const uint32_t valid_tokens[4] = {2u, 100u, 200u, 300u};
    if (siligemma::validate_runtime_contract(hdr, layers, linears, valid_tokens, 4u) != siligemma::ValidationError::OK) {
        std::fprintf(stderr, "Valid runtime contract rejected\n");
        return 1;
    }

    // R5 negative control: bos != 2 must fail closed
    const uint32_t bad_tokens[2] = {99u, 100u};
    if (siligemma::validate_runtime_contract(hdr, layers, linears, bad_tokens, 2u) != siligemma::ValidationError::ERR_BOS_NOT_2) {
        std::fprintf(stderr, "R5 bos!=2 negative control failed\n");
        return 1;
    }

    // R5 negative control: non-null dead KV in L15..34 must fail closed
    auto bad_layers = layers;
    bad_layers[15].linear_idx[1] = 0;
    if (siligemma::validate_runtime_contract(hdr, bad_layers, linears, valid_tokens, 4u) != siligemma::ValidationError::ERR_DEAD_KV_PRESENT) {
        std::fprintf(stderr, "R5 dead KV negative control failed\n");
        return 1;
    }

    // Verify 0-ULP tree parity & SECDED syndrome
    SplitMix64 rng(20261127ULL);
    std::vector<float> vec(1536);
    for (float& v : vec) v = rng.next_sym_f32();
    const float s_direct = siligemma::canonical_tree_sum_f32(vec.data(), vec.size());
    const float s_blk128 = siligemma::canonical_tree_sum_block128_f32(vec.data(), vec.size());
    if (siligemma::float_bits_u32(s_direct) != siligemma::float_bits_u32(s_blk128)) {
        std::fprintf(stderr, "0-ULP tree parity mismatch\n");
        return 1;
    }

    const uint64_t cw = pack_a16_codeword_rng(0x3A5u, rng);
    if (siligemma::a16_secded_syndrome_cpu(cw) != 0u) return 1;
    const uint64_t ded_cw = cw ^ (1ULL << 0) ^ (1ULL << 4);
    const uint32_t ded_syn = siligemma::a16_secded_syndrome_cpu(ded_cw);
    if (ded_syn == 0u || (ded_syn & 1u) != 0u) return 1;

    std::printf(
        "SILIGEMMA_CPU_SELFTEST_PASS live_linears=%zu dead_kv_omitted=%u wire_g32=%llu "
        "arena_16k=%llu arena_64k_shadow=%llu arena_128k=%llu sum_bits=0x%08x\n",
        linears.size(),
        hdr.num_dead_kv_omitted,
        static_cast<unsigned long long>(hdr.total_wire_bytes),
        static_cast<unsigned long long>(siligemma::kArena16K.total_arena_bytes),
        static_cast<unsigned long long>(siligemma::kArena64KShadow.total_arena_bytes),
        static_cast<unsigned long long>(siligemma::kArena128K.total_arena_bytes),
        siligemma::float_bits_u32(s_direct)
    );
    return 0;
}

struct BenchCliArgs {
    bool cpu_selftest = false;
    bool selftest_gpu = false;
    bool run_live_gpu = false;
    std::string slgb_path;
    std::string ple_safetensors;
    uint64_t ple_offset = 0ULL;
    bool has_ple_offset = false;
    std::string tokens_bin;
    std::string wt2_bin;
    std::string dump_logits_bf16;
    std::string dump_logits_4p5;
    std::string json_out;
};

static BenchCliArgs parse_cli_args(int argc, char** argv) {
    BenchCliArgs a{};
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--cpu-selftest") == 0) {
            a.cpu_selftest = true;
        } else if (std::strcmp(arg, "--selftest-gpu") == 0) {
            a.selftest_gpu = true;
        } else if (std::strcmp(arg, "--run-live-gpu") == 0) {
            a.run_live_gpu = true;
        } else if (std::strcmp(arg, "--slgb") == 0 && i + 1 < argc) {
            a.slgb_path = argv[++i];
        } else if (std::strcmp(arg, "--ple-safetensors") == 0 && i + 1 < argc) {
            a.ple_safetensors = argv[++i];
        } else if (std::strcmp(arg, "--ple-offset") == 0 && i + 1 < argc) {
            const char* raw_off = argv[++i];
            if (raw_off[0] < '0' || raw_off[0] > '9') {
                std::fprintf(stderr, "Invalid --ple-offset: %s\n", raw_off);
                std::exit(2);
            }
            errno = 0;
            char* endptr = nullptr;
            const unsigned long long parsed = std::strtoull(raw_off, &endptr, 10);
            if (errno == ERANGE || endptr == raw_off || *endptr != '\0') {
                std::fprintf(stderr, "Invalid or overflowing --ple-offset: %s\n", raw_off);
                std::exit(2);
            }
            a.ple_offset = static_cast<uint64_t>(parsed);
            a.has_ple_offset = true;
        } else if (std::strcmp(arg, "--tokens-bin") == 0 && i + 1 < argc) {
            a.tokens_bin = argv[++i];
        } else if (std::strcmp(arg, "--wt2-bin") == 0 && i + 1 < argc) {
            a.wt2_bin = argv[++i];
        } else if (std::strcmp(arg, "--dump-logits-bf16") == 0 && i + 1 < argc) {
            a.dump_logits_bf16 = argv[++i];
        } else if (std::strcmp(arg, "--dump-logits-4p5") == 0 && i + 1 < argc) {
            a.dump_logits_4p5 = argv[++i];
        } else if (std::strcmp(arg, "--json-out") == 0 && i + 1 < argc) {
            a.json_out = argv[++i];
        } else {
            std::fprintf(stderr, "Unknown or incomplete argument: %s\n", arg);
            std::exit(2);
        }
    }
    return a;
}

struct SlgbLoadedMeta {
    siligemma::SlgbHeader hdr{};
    std::array<siligemma::SiliGemmaLayerDesc, siligemma::GEMMA4_NUM_LAYERS> layers{};
    std::vector<siligemma::SiliGemmaLinearDesc> linears;
    uint64_t file_off_p1_wire = 0ULL;
    uint64_t file_off_p2_embed = 0ULL;
    uint64_t file_off_p2_lm_shadow = 0ULL;
    uint64_t file_off_p3_norms = 0ULL;
    uint64_t file_off_p4_loop = 0ULL;
    bool has_lm_shadow = false;
    bool has_u4_loop = false;
};

static SlgbLoadedMeta parse_slgb_meta(int fd) {
    SlgbLoadedMeta m{};
    pread_exact(fd, &m.hdr, sizeof(m.hdr), 0ULL);
    uint64_t off = sizeof(m.hdr);
    pread_exact(fd, m.layers.data(), m.layers.size() * sizeof(siligemma::SiliGemmaLayerDesc), off);
    off += m.layers.size() * sizeof(siligemma::SiliGemmaLayerDesc);
    m.linears.resize(siligemma::LIVE_LINEAR_COUNT);
    pread_exact(fd, m.linears.data(), m.linears.size() * sizeof(siligemma::SiliGemmaLinearDesc), off);
    off += m.linears.size() * sizeof(siligemma::SiliGemmaLinearDesc);

    m.has_lm_shadow = (m.hdr.flags & 1u) != 0u;
    m.has_u4_loop = (m.hdr.flags & 2u) != 0u;

    off = next_multiple_256(off);
    m.file_off_p1_wire = off;
    off += m.hdr.total_wire_bytes;

    off = next_multiple_256(off);
    m.file_off_p2_embed = off;
    off += siligemma::EMBED_TOKENS_BF16_BYTES;

    if (m.has_lm_shadow) {
        off = next_multiple_256(off);
        m.file_off_p2_lm_shadow = off;
        off += siligemma::LM_HEAD_SHADOW_G32_BYTES;
    }

    off = next_multiple_256(off);
    m.file_off_p3_norms = off;
    off += uint64_t(siligemma::NORM_SCALAR_PARAMS_FP32) * sizeof(float);

    if (m.has_u4_loop) {
        off = next_multiple_256(off);
        m.file_off_p4_loop = off;
        off += siligemma::LOOP_CONTROLLER_BF16_BYTES;
    }

    struct stat st{};
    SG_REQUIRE(::fstat(fd, &st) == 0, "fstat slgb failed");
    const uint64_t fsz = static_cast<uint64_t>(st.st_size);
    SG_REQUIRE(
        siligemma::validate_slgb_file_layout(m.hdr, m.layers, m.linears, fsz, true) == siligemma::ValidationError::OK,
        "SLGB header/descriptor/CRC32/size contract validation failed"
    );
    SG_REQUIRE(fsz == off, "slgb file size mismatch vs section offsets");
    return m;
}

static void stream_file_to_device(int fd, uint64_t file_off, void* d_dst, uint64_t nbytes, void* h_xfer, size_t chunk_cap) {
    uint8_t* dst = reinterpret_cast<uint8_t*>(d_dst);
    uint64_t done = 0ULL;
    while (done < nbytes) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(nbytes - done, chunk_cap));
        pread_exact(fd, h_xfer, chunk, file_off + done);
        SG_CUDA_OK(cudaMemcpy(dst + done, h_xfer, chunk, cudaMemcpyHostToDevice));
        done += chunk;
    }
}

static void read_u32_vec_bin(const std::string& path, std::vector<uint32_t>& out) {
    struct stat st{};
    SG_REQUIRE(::stat(path.c_str(), &st) == 0, "stat u32 bin failed");
    SG_REQUIRE(st.st_size > 0 && (st.st_size % sizeof(uint32_t)) == 0, "invalid u32 bin size");
    const size_t n = static_cast<size_t>(st.st_size) / sizeof(uint32_t);
    out.resize(n);
    int fd = ::open(path.c_str(), O_RDONLY);
    SG_REQUIRE(fd >= 0, "open u32 bin failed");
    pread_exact(fd, out.data(), static_cast<size_t>(st.st_size), 0ULL);
    ::close(fd);
}

static double median_of_vec(std::vector<double> vals) {
    if (vals.empty()) return 0.0;
    std::sort(vals.begin(), vals.end());
    const size_t n = vals.size();
    if ((n & 1u) != 0u) return vals[n >> 1];
    return 0.5 * (vals[(n >> 1) - 1] + vals[n >> 1]);
}

static double percentile_of_vec(std::vector<double> vals, double q) {
    if (vals.empty()) return 0.0;
    std::sort(vals.begin(), vals.end());
    const double idx = q * static_cast<double>(vals.size() - 1u);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = std::min(lo + 1u, vals.size() - 1u);
    const double frac = idx - static_cast<double>(lo);
    return vals[lo] * (1.0 - frac) + vals[hi] * frac;
}

static uint32_t ulp_diff_f32(float a, float b) {
    const uint32_t ua = siligemma::float_bits_u32(a);
    const uint32_t ub = siligemma::float_bits_u32(b);
    auto to_ord = [](uint32_t u) -> int64_t {
        return (u & 0x80000000u) ? (0x80000000LL - static_cast<int64_t>(u)) : static_cast<int64_t>(u);
    };
    const int64_t da = to_ord(ua);
    const int64_t db = to_ord(ub);
    return static_cast<uint32_t>(std::llabs(da - db));
}

// Compute log-softmax NLL for [B, V] CPU buffer across rows where target != 0xFFFFFFFF
static void compute_batch_nll_cpu(
    const float* logits,
    const uint32_t* targets,
    int batch_size,
    int vocab_size,
    double& nll_sum,
    uint64_t& tok_cnt
) {
    for (int b = 0; b < batch_size; ++b) {
        const uint32_t tgt = targets[b];
        if (tgt >= static_cast<uint32_t>(vocab_size)) continue;
        const float* row = logits + static_cast<size_t>(b) * static_cast<size_t>(vocab_size);
        float max_v = -1.0e30f;
        for (int v = 0; v < vocab_size; ++v) {
            if (row[v] > max_v) max_v = row[v];
        }
        double sum_exp = 0.0;
        for (int v = 0; v < vocab_size; ++v) {
            sum_exp += std::exp(static_cast<double>(row[v] - max_v));
        }
        const double lse = static_cast<double>(max_v) + std::log(sum_exp);
        nll_sum += lse - static_cast<double>(row[tgt]);
        tok_cnt += 1u;
    }
}

struct ArenaDevicePtrs {
    uint8_t* d_arena = nullptr;
    siligemma::SiliGemmaArenaLayout layout{};
    uint8_t* d_wire = nullptr;
    __nv_bfloat16* d_embed = nullptr;
    uint64_t* d_lm_shadow_u64 = nullptr;
    __half* d_lm_shadow_sc = nullptr;
    float* d_norms = nullptr;
    __nv_bfloat16* d_loop_ctrl = nullptr;
    uint8_t* d_kv_base = nullptr;
    __nv_bfloat16* d_ple_stage_0 = nullptr;
    __nv_bfloat16* d_ple_stage_1 = nullptr;

    float* d_h_cur = nullptr;
    float* d_h_tmp = nullptr;
    float* d_h_pre24 = nullptr;
    __nv_bfloat16* d_loop_prev = nullptr;
    float* d_q = nullptr;
    float* d_k = nullptr;
    float* d_v = nullptr;
    float* d_ffn_gate = nullptr;
    float* d_ffn_up = nullptr;
    float* d_ple_model_proj_f32 = nullptr;
    int32_t* d_x_q8 = nullptr;
    float* d_x_sc = nullptr;
    float4* d_u4 = nullptr;
    float* d_m_inv_tmp = nullptr;
    float* d_logits = nullptr;
    int32_t* d_tok_in = nullptr;
    int32_t* d_tok_out = nullptr;
    float* d_argmax_score = nullptr;

    // P7 unused tail [p7_used_bytes .. sz_p7_act_scratch)
    int32_t* d_pos_i32 = nullptr;
    float4* d_hub_a_f4 = nullptr; // [5 * 1536] float4
    float4* d_hub_b_f4 = nullptr; // [5 * 1536] float4

    float* d_model_norm = nullptr;
    float* d_ple_proj_norm = nullptr;
    siligemma::HoloSphereRazorCanary64* host_canary = nullptr;
    siligemma::HoloSphereRazorCanary64* dev_canary = nullptr;
};

static ArenaDevicePtrs bind_arena_ptrs(
    uint8_t* d_arena,
    const siligemma::SiliGemmaArenaLayout& lay,
    siligemma::HoloSphereRazorCanary64* host_canary,
    siligemma::HoloSphereRazorCanary64* dev_canary
) {
    ArenaDevicePtrs p{};
    p.d_arena = d_arena;
    p.layout = lay;
    p.d_wire = d_arena + lay.off_p1_wire_weights;
    p.d_embed = reinterpret_cast<__nv_bfloat16*>(d_arena + lay.off_p2_embed_bf16);
    p.d_lm_shadow_u64 = reinterpret_cast<uint64_t*>(d_arena + lay.off_p2_lm_shadow);
    // [262144, 1536] -> 262144 * 96 uint64 words = 201,326,592 B, then scales
    p.d_lm_shadow_sc = reinterpret_cast<__half*>(d_arena + lay.off_p2_lm_shadow + 201326592ULL);
    p.d_norms = reinterpret_cast<float*>(d_arena + lay.off_p3_norms_f32);
    p.d_loop_ctrl = reinterpret_cast<__nv_bfloat16*>(d_arena + lay.off_p4_loop_ctrl_bf16);
    p.d_kv_base = d_arena + lay.off_p5_zed_kv_ring;
    p.d_ple_stage_0 = reinterpret_cast<__nv_bfloat16*>(d_arena + lay.off_p6_ple_stage);
    p.d_ple_stage_1 = p.d_ple_stage_0 + 16ULL * 35ULL * 256ULL;

    uint8_t* p7 = d_arena + lay.off_p7_act_scratch;
    p.d_h_cur = reinterpret_cast<float*>(p7 + lay.sub_h_cur_f32);
    p.d_h_tmp = reinterpret_cast<float*>(p7 + lay.sub_h_tmp_f32);
    p.d_h_pre24 = reinterpret_cast<float*>(p7 + lay.sub_h_pre24_f32);
    p.d_loop_prev = reinterpret_cast<__nv_bfloat16*>(p7 + lay.sub_loop_prev_bf16);
    p.d_q = reinterpret_cast<float*>(p7 + lay.sub_q_f32);
    p.d_k = reinterpret_cast<float*>(p7 + lay.sub_k_f32);
    p.d_v = reinterpret_cast<float*>(p7 + lay.sub_v_f32);
    p.d_ffn_gate = reinterpret_cast<float*>(p7 + lay.sub_ffn_gate_f32);
    p.d_ffn_up = reinterpret_cast<float*>(p7 + lay.sub_ffn_up_f32);
    // sub_logits_f32 is 16*262144*4 = 16,777,216 B, unused during L0..34
    p.d_logits = reinterpret_cast<float*>(p7 + lay.sub_logits_f32);
    p.d_ple_model_proj_f32 = p.d_logits;
    p.d_x_q8 = reinterpret_cast<int32_t*>(p7 + lay.sub_x_q8_i32);
    p.d_x_sc = reinterpret_cast<float*>(p7 + lay.sub_x_sc_f32);
    p.d_u4 = reinterpret_cast<float4*>(p7 + lay.sub_u4_f4);
    p.d_m_inv_tmp = reinterpret_cast<float*>(p7 + lay.sub_m_inv_tmp_f32);
    p.d_tok_in = reinterpret_cast<int32_t*>(p7 + lay.sub_tok_in_i32);
    p.d_tok_out = reinterpret_cast<int32_t*>(p7 + lay.sub_tok_out_i32);
    p.d_argmax_score = reinterpret_cast<float*>(p7 + lay.sub_argmax_score_f32);

    // Tail of P7: verify exact byte fit (246,016 B <= 292,864 B)
    const uint64_t tail0 = siligemma::align_up_n(lay.p7_used_bytes, 256ULL);
    const uint64_t hub_bytes = 5ULL * 1536ULL * sizeof(float4); // 122,880 B
    SG_REQUIRE(tail0 + 256ULL + 2ULL * hub_bytes <= lay.sz_p7_act_scratch, "P7 tail overflow");
    p.d_pos_i32 = reinterpret_cast<int32_t*>(p7 + tail0);
    p.d_hub_a_f4 = reinterpret_cast<float4*>(p7 + tail0 + 256ULL);
    p.d_hub_b_f4 = reinterpret_cast<float4*>(p7 + tail0 + 256ULL + hub_bytes);

    // End of P3: model.norm (1536) + per_layer_projection_norm (256)
    p.d_model_norm = p.d_norms + (siligemma::NORM_SCALAR_PARAMS_FP32 - (1536 + 256));
    p.d_ple_proj_norm = p.d_model_norm + 1536;

    p.host_canary = host_canary;
    p.dev_canary = dev_canary;
    return p;
}

static void launch_a16_linear(
    const ArenaDevicePtrs& p,
    const siligemma::SiliGemmaLinearDesc& desc,
    const int32_t* x_q8,
    const float* x_sc,
    float* y_out,
    float softcap,
    int batch_size,
    cudaStream_t stream
) {
    const uint64_t* w64 = reinterpret_cast<const uint64_t*>(p.d_wire + desc.words_byte_offset);
    const __half* wsc = reinterpret_cast<const __half*>(p.d_wire + desc.scales_byte_offset);
    if (batch_size == 1) {
        const int grid = (desc.out_dim + 3) / 4;
        siligemma_sm86::siligemma_a16_h4_dp4a_gemv_r4_sm86<<<grid, 128, 0, stream>>>(
            x_q8, x_sc, w64, wsc, y_out, p.dev_canary, softcap, desc.out_dim, desc.in_dim, desc.group_size
        );
    } else {
        siligemma_sm86::siligemma_a16_h4_dp4a_gemm_b16_sm86<<<desc.out_dim, 128, 0, stream>>>(
            x_q8, x_sc, w64, wsc, y_out, p.dev_canary, softcap, batch_size, desc.out_dim, desc.in_dim, desc.group_size
        );
    }
}

struct KvRingView {
    __nv_bfloat16* buf = nullptr;
    int ring_slots = 0;
    int batch_stride = 0;
};

static KvRingView get_kv_view_for_layer(
    const ArenaDevicePtrs& p,
    const siligemma::SiliGemmaLayerDesc& ly,
    int batch_size
) {
    const uint32_t src_l = ly.kv_source_layer; // 0..14
    if (batch_size == 1) {
        if (ly.is_full_attn) {
            const int full_idx = (src_l == 4u) ? 0 : ((src_l == 9u) ? 1 : 2);
            return {
                reinterpret_cast<__nv_bfloat16*>(p.d_kv_base + p.layout.kv_ring_full_offsets[full_idx]),
                static_cast<int>(p.layout.t_max),
                static_cast<int>(p.layout.t_max)
            };
        }
        return {
            reinterpret_cast<__nv_bfloat16*>(p.d_kv_base + p.layout.kv_ring_sliding_offsets[src_l]),
            static_cast<int>(siligemma::GEMMA4_SLIDING_WINDOW),
            static_cast<int>(siligemma::GEMMA4_SLIDING_WINDOW)
        };
    }

    // Batch mode (B <= 16, T_max = 1024) re-indexes the exact 408,944,640 B P5 slab cleanly (201.3 MiB used).
    constexpr uint64_t kSlidingLayerBytes = 16ULL * 512ULL * 2ULL * 256ULL * 2ULL;  // 8,388,608 B
    constexpr uint64_t kFullLayerBytes    = 16ULL * 1024ULL * 2ULL * 512ULL * 2ULL; // 33,554,432 B
    if (ly.is_full_attn) {
        const int full_idx = (src_l == 4u) ? 0 : ((src_l == 9u) ? 1 : 2);
        const uint64_t off = 12ULL * kSlidingLayerBytes + static_cast<uint64_t>(full_idx) * kFullLayerBytes;
        SG_REQUIRE(off + kFullLayerBytes <= p.layout.sz_p5_zed_kv_ring, "B=16 full KV overflow");
        return {reinterpret_cast<__nv_bfloat16*>(p.d_kv_base + off), 1024, 1024};
    }
    int sliding_idx = 0;
    for (uint32_t j = 0u; j < src_l; ++j) {
        if ((j % 5u) != 4u) ++sliding_idx;
    }
    const uint64_t off = static_cast<uint64_t>(sliding_idx) * kSlidingLayerBytes;
    SG_REQUIRE(off + kSlidingLayerBytes <= p.layout.sz_p5_zed_kv_ring, "B=16 sliding KV overflow");
    return {reinterpret_cast<__nv_bfloat16*>(p.d_kv_base + off), 512, 512};
}

struct LoopGatePtrs {
    const __nv_bfloat16* w_down;
    const __nv_bfloat16* w_forget;
    const __nv_bfloat16* w_gain;
    const __nv_bfloat16* msg_query;
};

static LoopGatePtrs get_loop_gate_ptrs(const ArenaDevicePtrs& p, int loop_layer_idx, int gate_idx) {
    // 27 gate modules contiguous; per gate: 49152 (down) + 24576 (forget) + 24576 (gain) + 1536 (mq) = 99840 BF16
    constexpr uint64_t kStride = 99840ULL;
    const int flat = loop_layer_idx * 3 + gate_idx;
    const __nv_bfloat16* base = p.d_loop_ctrl + static_cast<uint64_t>(flat) * kStride;
    return {
        base,
        base + 49152ULL,
        base + 49152ULL + 24576ULL,
        base + 49152ULL + 24576ULL + 24576ULL
    };
}

static void emit_single_layer_ops(
    const ArenaDevicePtrs& p,
    const SlgbLoadedMeta& m,
    const std::vector<float>& h_norms,
    uint32_t layer_idx,
    int batch_size,
    int static_pos,
    bool use_d_pos,
    bool use_u4_gates,
    bool snapshot_loop_state,
    bool enable_hub_steer,
    float hub_alpha,
    __nv_bfloat16* d_ple_dma_slot,
    float*& cur_h,
    float*& tmp_h,
    cudaStream_t stream
) {
    const auto& ly = m.layers[layer_idx];
    const float eps = m.hdr.rms_norm_eps;
    const int H = static_cast<int>(siligemma::GEMMA4_HIDDEN_SIZE);
    const int HD = static_cast<int>(ly.head_dim);
    const int Q_DIM = 8 * HD;
    const int FF = static_cast<int>(ly.intermediate_size);
    const int PLE = static_cast<int>(siligemma::GEMMA4_PLE_DIM);
    const size_t hidden_bytes = static_cast<size_t>(batch_size) * static_cast<size_t>(H) * sizeof(float);

    const float* d_norm_in    = p.d_norms + ly.norm_offsets_f32[0];
    const float* d_norm_posta = p.d_norms + ly.norm_offsets_f32[1];
    const float* d_norm_pref  = p.d_norms + ly.norm_offsets_f32[2];
    const float* d_norm_postf = p.d_norms + ly.norm_offsets_f32[3];
    const float* d_norm_postp = p.d_norms + ly.norm_offsets_f32[4];
    const float* d_norm_q     = p.d_norms + ly.norm_offsets_f32[5];
    const float* d_norm_k     = (ly.norm_offsets_f32[6] >= 0) ? (p.d_norms + ly.norm_offsets_f32[6]) : nullptr;
    const float  layer_scalar = h_norms[static_cast<size_t>(ly.norm_offsets_f32[7])];

    const int lli = static_cast<int>(layer_idx) - static_cast<int>(siligemma::LOOP_START_LAYER);
    __nv_bfloat16* m_prev = (lli >= 0 && lli < 9)
        ? (p.d_loop_prev + static_cast<size_t>(lli) * static_cast<size_t>(batch_size) * static_cast<size_t>(H))
        : nullptr;

    // U-4 q-gate at layer entry (pass p >= 2)
    if (use_u4_gates && m_prev != nullptr) {
        const auto gp = get_loop_gate_ptrs(p, lli, 0);
        siligemma_sm86::siligemma_u4_loop_gate_step_sm86<<<batch_size, 128, 0, stream>>>(
            cur_h, m_prev, gp.msg_query, gp.w_down, gp.w_forget, gp.w_gain, tmp_h, eps, H
        );
        std::swap(cur_h, tmp_h);
    }

    // 1. Attention block
    siligemma_sm86::siligemma_sandwich_norm_u4_q8_sm86<<<batch_size, 128, 0, stream>>>(
        cur_h, d_norm_in, nullptr, p.d_x_q8, p.d_x_sc, nullptr, eps, H
    );
    launch_a16_linear(p, m.linears[ly.linear_idx[0]], p.d_x_q8, p.d_x_sc, p.d_q, 0.0f, batch_size, stream);
    const float* k_raw = nullptr;
    const float* v_raw = nullptr;
    if (!ly.is_kv_shared) {
        launch_a16_linear(p, m.linears[ly.linear_idx[1]], p.d_x_q8, p.d_x_sc, p.d_k, 0.0f, batch_size, stream);
        launch_a16_linear(p, m.linears[ly.linear_idx[2]], p.d_x_q8, p.d_x_sc, p.d_v, 0.0f, batch_size, stream);
        k_raw = p.d_k;
        v_raw = p.d_v;
    }
    const KvRingView kv = get_kv_view_for_layer(p, ly, batch_size);
    siligemma_sm86::siligemma_attention_kv_step_sm86<<<dim3(8, batch_size), 128, 0, stream>>>(
        p.d_q, k_raw, v_raw, d_norm_q, d_norm_k, kv.buf, p.d_ffn_gate,
        static_pos, use_d_pos ? p.d_pos_i32 : nullptr, kv.ring_slots, kv.batch_stride,
        HD, static_cast<int>(ly.is_full_attn), static_cast<int>(ly.is_kv_shared), eps
    );
    siligemma_sm86::siligemma_sandwich_norm_u4_q8_sm86<<<batch_size, 128, 0, stream>>>(
        p.d_ffn_gate, nullptr, nullptr, p.d_x_q8, p.d_x_sc, nullptr, -1.0f, Q_DIM
    );
    launch_a16_linear(p, m.linears[ly.linear_idx[3]], p.d_x_q8, p.d_x_sc, p.d_ffn_up, 0.0f, batch_size, stream);
    const float* attn_block_out = p.d_ffn_up;
    if (enable_hub_steer && layer_idx >= 10u && layer_idx <= 14u && batch_size == 1) {
        const int hub_slot = static_cast<int>(layer_idx - 10u);
        const float4* a_ptr = p.d_hub_a_f4 + static_cast<size_t>(hub_slot) * 1536ULL;
        const float4* b_ptr = p.d_hub_b_f4 + static_cast<size_t>(hub_slot) * 1536ULL;
        siligemma_sm86::siligemma_cayley_woodbury_hub_l10_14_sm86<<<1, 256, 0, stream>>>(
            p.d_ffn_up, a_ptr, b_ptr, tmp_h, hub_alpha, H, hub_slot
        );
        attn_block_out = tmp_h;
    }
    siligemma_sm86::siligemma_fused_geglu_ple_sandwich_sm86<<<batch_size, 128, 0, stream>>>(
        cur_h, attn_block_out, d_norm_posta, nullptr, nullptr, p.d_ffn_gate, 1.0f, eps, H
    );
    SG_CUDA_OK(cudaMemcpyAsync(cur_h, p.d_ffn_gate, hidden_bytes, cudaMemcpyDeviceToDevice, stream));

    // U-4 mlp-gate
    if (use_u4_gates && m_prev != nullptr) {
        const auto gp = get_loop_gate_ptrs(p, lli, 1);
        siligemma_sm86::siligemma_u4_loop_gate_step_sm86<<<batch_size, 128, 0, stream>>>(
            cur_h, m_prev, gp.msg_query, gp.w_down, gp.w_forget, gp.w_gain, tmp_h, eps, H
        );
        std::swap(cur_h, tmp_h);
    }

    // 2. GeGLU MLP block
    siligemma_sm86::siligemma_sandwich_norm_u4_q8_sm86<<<batch_size, 128, 0, stream>>>(
        cur_h, d_norm_pref, nullptr, p.d_x_q8, p.d_x_sc, nullptr, eps, H
    );
    launch_a16_linear(p, m.linears[ly.linear_idx[4]], p.d_x_q8, p.d_x_sc, p.d_ffn_gate, 0.0f, batch_size, stream);
    launch_a16_linear(p, m.linears[ly.linear_idx[5]], p.d_x_q8, p.d_x_sc, p.d_ffn_up, 0.0f, batch_size, stream);
    float* d_ffn_act = p.d_logits + 16ULL * 8960ULL; // free region of P7 sub_logits_f32 before lm_head
    siligemma_sm86::siligemma_geglu_act_sm86<<<(batch_size * FF + 127) / 128, 128, 0, stream>>>(
        p.d_ffn_gate, p.d_ffn_up, d_ffn_act, batch_size * FF
    );
    siligemma_sm86::siligemma_sandwich_norm_u4_q8_sm86<<<batch_size, 128, 0, stream>>>(
        d_ffn_act, nullptr, nullptr, p.d_x_q8, p.d_x_sc, nullptr, -1.0f, FF
    );
    launch_a16_linear(p, m.linears[ly.linear_idx[6]], p.d_x_q8, p.d_x_sc, p.d_ffn_up, 0.0f, batch_size, stream);
    siligemma_sm86::siligemma_fused_geglu_ple_sandwich_sm86<<<batch_size, 128, 0, stream>>>(
        cur_h, p.d_ffn_up, d_norm_postf, nullptr, nullptr, tmp_h, 1.0f, eps, H
    );
    std::swap(cur_h, tmp_h);

    // U-4 ple-gate
    if (use_u4_gates && m_prev != nullptr) {
        const auto gp = get_loop_gate_ptrs(p, lli, 2);
        siligemma_sm86::siligemma_u4_loop_gate_step_sm86<<<batch_size, 128, 0, stream>>>(
            cur_h, m_prev, gp.msg_query, gp.w_down, gp.w_forget, gp.w_gain, tmp_h, eps, H
        );
        std::swap(cur_h, tmp_h);
    }

    // 3. PLE block
    siligemma_sm86::siligemma_sandwich_norm_u4_q8_sm86<<<batch_size, 128, 0, stream>>>(
        cur_h, nullptr, nullptr, p.d_x_q8, p.d_x_sc, nullptr, -1.0f, H
    );
    launch_a16_linear(p, m.linears[ly.linear_idx[7]], p.d_x_q8, p.d_x_sc, p.d_k, 0.0f, batch_size, stream);
    siligemma_sm86::siligemma_ple_combine_gate_sm86<<<batch_size, 128, 0, stream>>>(
        p.d_k, p.d_ple_model_proj_f32, d_ple_dma_slot, p.d_ple_proj_norm, p.d_v,
        static_cast<int>(layer_idx), 35, PLE, 1.0f / std::sqrt(1536.0f), eps
    );
    siligemma_sm86::siligemma_sandwich_norm_u4_q8_sm86<<<batch_size, 128, 0, stream>>>(
        p.d_v, nullptr, nullptr, p.d_x_q8, p.d_x_sc, nullptr, -1.0f, PLE
    );
    launch_a16_linear(p, m.linears[ly.linear_idx[8]], p.d_x_q8, p.d_x_sc, p.d_ffn_up, 0.0f, batch_size, stream);
    siligemma_sm86::siligemma_fused_geglu_ple_sandwich_sm86<<<batch_size, 128, 0, stream>>>(
        cur_h, p.d_ffn_up, d_norm_postp, nullptr, nullptr, tmp_h, layer_scalar, eps, H
    );
    std::swap(cur_h, tmp_h);

    if (snapshot_loop_state && m_prev != nullptr) {
        siligemma_sm86::siligemma_embed_lookup_sm86<<<batch_size, 128, 0, stream>>>(
            nullptr, nullptr, cur_h, nullptr, m_prev, 1.0f, H, 1
        );
    }
}

struct CapturedGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    void destroy() {
        if (exec != nullptr) { cudaGraphExecDestroy(exec); exec = nullptr; }
        if (graph != nullptr) { cudaGraphDestroy(graph); graph = nullptr; }
    }
};

static CapturedGraph capture_full_step_graph(
    const ArenaDevicePtrs& p,
    const SlgbLoadedMeta& m,
    const std::vector<float>& h_norms,
    int batch_size,
    int loop_k,
    bool use_4p5_lm_head,
    int static_pos,
    bool use_d_pos,
    bool include_dma_in_graph,
    __nv_bfloat16* h_ple_pin_src,
    __nv_bfloat16* d_ple_dma_slot,
    bool enable_hub_steer,
    float hub_alpha,
    cudaStream_t stream_compute,
    cudaStream_t stream_dma,
    cudaEvent_t ev_fork,
    cudaEvent_t ev_dma_done
) {
    CapturedGraph g{};
    const int H = static_cast<int>(siligemma::GEMMA4_HIDDEN_SIZE);
    const int V = static_cast<int>(siligemma::GEMMA4_VOCAB_SIZE);
    const float eps = m.hdr.rms_norm_eps;
    const size_t n_dma_bytes = static_cast<size_t>(batch_size) * siligemma::PLE_DMA_BYTES_PER_TOKEN;

    SG_CUDA_OK(cudaStreamBeginCapture(stream_compute, cudaStreamCaptureModeThreadLocal));

    if (include_dma_in_graph) {
        SG_CUDA_OK(cudaEventRecord(ev_fork, stream_compute));
        SG_CUDA_OK(cudaStreamWaitEvent(stream_dma, ev_fork, 0));
        SG_CUDA_OK(cudaMemcpyAsync(d_ple_dma_slot, h_ple_pin_src, n_dma_bytes, cudaMemcpyHostToDevice, stream_dma));
        SG_CUDA_OK(cudaEventRecord(ev_dma_done, stream_dma));
    }

    // Token embedding (* sqrt(1536)) + Global per_layer_model_projection
    siligemma_sm86::siligemma_embed_lookup_sm86<<<batch_size, 128, 0, stream_compute>>>(
        p.d_embed, p.d_tok_in, nullptr, p.d_h_cur, nullptr, std::sqrt(1536.0f), H, 0
    );
    siligemma_sm86::siligemma_sandwich_norm_u4_q8_sm86<<<batch_size, 128, 0, stream_compute>>>(
        p.d_h_cur, nullptr, nullptr, p.d_x_q8, p.d_x_sc, nullptr, -1.0f, H
    );
    launch_a16_linear(
        p, m.linears[275], p.d_x_q8, p.d_x_sc, p.d_ple_model_proj_f32, 0.0f, batch_size, stream_compute
    );

    if (include_dma_in_graph) {
        // Join async DMA only right before L0 PLE combination!
        SG_CUDA_OK(cudaStreamWaitEvent(stream_compute, ev_dma_done, 0));
    }

    float* cur_h = p.d_h_cur;
    float* tmp_h = p.d_h_tmp;

    // L0..L24 (0..24)
    for (uint32_t l = 0u; l < siligemma::LOOP_START_LAYER; ++l) {
        emit_single_layer_ops(
            p, m, h_norms, l, batch_size, static_pos, use_d_pos,
            false, false, enable_hub_steer, hub_alpha, d_ple_dma_slot, cur_h, tmp_h, stream_compute
        );
    }

    // Save h_pre24 if K > 1
    if (loop_k > 1) {
        SG_CUDA_OK(cudaMemcpyAsync(
            p.d_h_pre24, cur_h, static_cast<size_t>(batch_size) * static_cast<size_t>(H) * sizeof(float),
            cudaMemcpyDeviceToDevice, stream_compute
        ));
    }

    // Pass 1..K over L25..33 (saving L33 exits into d_loop_prev[0], d_loop_prev[8], & d_h_pre24 for Riccati probe)
    for (int pass = 1; pass <= loop_k; ++pass) {
        if (pass >= 2) {
            SG_CUDA_OK(cudaMemcpyAsync(
                cur_h, p.d_h_pre24, static_cast<size_t>(batch_size) * static_cast<size_t>(H) * sizeof(float),
                cudaMemcpyDeviceToDevice, stream_compute
            ));
        }
        const bool snapshot = (pass < loop_k);
        const bool u4_active = (pass >= 2);
        for (uint32_t l = siligemma::LOOP_START_LAYER; l <= siligemma::LOOP_END_LAYER; ++l) {
            if (l == siligemma::LOOP_END_LAYER && loop_k == 3 && pass == 2) {
                // Preserve Pass 1 L33 exit into d_loop_prev[0] (already consumed in pass 2) before pass 2 overwrites slot 8
                const size_t st_b = static_cast<size_t>(batch_size) * static_cast<size_t>(H) * sizeof(__nv_bfloat16);
                SG_CUDA_OK(cudaMemcpyAsync(
                    p.d_loop_prev,
                    p.d_loop_prev + 8ULL * static_cast<size_t>(batch_size) * static_cast<size_t>(H),
                    st_b, cudaMemcpyDeviceToDevice, stream_compute
                ));
            }
            emit_single_layer_ops(
                p, m, h_norms, l, batch_size, static_cast<int>(static_pos), use_d_pos,
                u4_active, snapshot, false, 0.0f, d_ple_dma_slot, cur_h, tmp_h, stream_compute
            );
        }
        if (loop_k > 1 && pass == loop_k) {
            SG_CUDA_OK(cudaMemcpyAsync(
                p.d_h_pre24, cur_h, static_cast<size_t>(batch_size) * static_cast<size_t>(H) * sizeof(float),
                cudaMemcpyDeviceToDevice, stream_compute
            ));
        }
    }

    // L34 exit layer
    emit_single_layer_ops(
        p, m, h_norms, 34u, batch_size, static_pos, use_d_pos,
        false, false, false, 0.0f, d_ple_dma_slot, cur_h, tmp_h, stream_compute
    );

    // Final model.norm + LM Head (+ softcap=30.0) + argmax
    if (use_4p5_lm_head) {
        siligemma_sm86::siligemma_sandwich_norm_u4_q8_sm86<<<batch_size, 128, 0, stream_compute>>>(
            cur_h, p.d_model_norm, nullptr, p.d_x_q8, p.d_x_sc, nullptr, eps, H
        );
        if (batch_size == 1) {
            siligemma_sm86::siligemma_a16_h4_dp4a_gemv_r4_sm86<<<(V + 3) / 4, 128, 0, stream_compute>>>(
                p.d_x_q8, p.d_x_sc, p.d_lm_shadow_u64, p.d_lm_shadow_sc, p.d_logits,
                p.dev_canary, m.hdr.final_logit_softcapping, V, H, static_cast<int>(m.hdr.group_size)
            );
        } else {
            siligemma_sm86::siligemma_a16_h4_dp4a_gemm_b16_sm86<<<V, 128, 0, stream_compute>>>(
                p.d_x_q8, p.d_x_sc, p.d_lm_shadow_u64, p.d_lm_shadow_sc, p.d_logits,
                p.dev_canary, m.hdr.final_logit_softcapping, batch_size, V, H, static_cast<int>(m.hdr.group_size)
            );
        }
    } else {
        siligemma_sm86::siligemma_fused_geglu_ple_sandwich_sm86<<<batch_size, 128, 0, stream_compute>>>(
            nullptr, cur_h, p.d_model_norm, nullptr, nullptr, tmp_h, 1.0f, eps, H
        );
        if (batch_size == 1) {
            siligemma_sm86::siligemma_bf16_lm_head_r4_sm86<<<(V + 3) / 4, 128, 0, stream_compute>>>(
                tmp_h, p.d_embed, p.d_logits, m.hdr.final_logit_softcapping, V, H
            );
        } else {
            siligemma_sm86::siligemma_bf16_lm_head_b16_sm86<<<V, 128, 0, stream_compute>>>(
                tmp_h, p.d_embed, p.d_logits, m.hdr.final_logit_softcapping, batch_size, V, H
            );
        }
    }

    siligemma_sm86::siligemma_iris_canary_finalize_sm86<<<1, 128, 0, stream_compute>>>(
        cur_h, p.dev_canary, 34u, static_cast<uint32_t>(loop_k), 0u, 0, H
    );
    siligemma_sm86::siligemma_softcap_logits_argmax_sm86<<<batch_size, 128, 0, stream_compute>>>(
        p.d_logits, nullptr,
        p.d_tok_out, p.d_argmax_score, use_d_pos ? p.d_pos_i32 : nullptr, -1.0f, V
    );

    SG_CUDA_OK(cudaStreamEndCapture(stream_compute, &g.graph));
    SG_CUDA_OK(cudaGraphInstantiate(&g.exec, g.graph, nullptr, nullptr, 0));
    return g;
}

static void benchmark_abba_graphs(
    CapturedGraph& g_a,
    CapturedGraph& g_b,
    cudaStream_t stream,
    int n_pairs,
    int inner_launches,
    std::vector<double>& times_a_ms,
    std::vector<double>& times_b_ms
) {
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    SG_CUDA_OK(cudaEventCreate(&ev0));
    SG_CUDA_OK(cudaEventCreate(&ev1));
    // Warmup
    for (int w = 0; w < 3; ++w) {
        SG_CUDA_OK(cudaGraphLaunch(g_a.exec, stream));
        SG_CUDA_OK(cudaGraphLaunch(g_b.exec, stream));
    }
    SG_CUDA_OK(cudaStreamSynchronize(stream));

    auto run_one = [&](cudaGraphExec_t exec) -> double {
        SG_CUDA_OK(cudaEventRecord(ev0, stream));
        for (int i = 0; i < inner_launches; ++i) {
            SG_CUDA_OK(cudaGraphLaunch(exec, stream));
        }
        SG_CUDA_OK(cudaEventRecord(ev1, stream));
        SG_CUDA_OK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        SG_CUDA_OK(cudaEventElapsedTime(&ms, ev0, ev1));
        return static_cast<double>(ms) / static_cast<double>(inner_launches);
    };

    times_a_ms.clear();
    times_b_ms.clear();
    for (int p = 0; p < n_pairs; ++p) {
        if ((p & 1) == 0) {
            times_a_ms.push_back(run_one(g_a.exec));
            times_b_ms.push_back(run_one(g_b.exec));
        } else {
            times_b_ms.push_back(run_one(g_b.exec));
            times_a_ms.push_back(run_one(g_a.exec));
        }
    }
    SG_CUDA_OK(cudaEventDestroy(ev1));
    SG_CUDA_OK(cudaEventDestroy(ev0));
}

static int run_live_gpu_bench(const BenchCliArgs& args) {
    SG_REQUIRE(!args.slgb_path.empty() && !args.ple_safetensors.empty() && args.has_ple_offset && !args.json_out.empty(),
               "--run-live-gpu requires --slgb, --ple-safetensors, --ple-offset, --json-out");

    int dev_count = 0;
    SG_CUDA_OK(cudaGetDeviceCount(&dev_count));
    SG_REQUIRE(dev_count > 0, "No CUDA device available");
    SG_CUDA_OK(cudaSetDevice(0));

    // 1. Validate peak VRAM for 128K context envelope (kArena128K = 2,700,007,680 B <= 3072 MiB delta)
    size_t free_init = 0u, total_vram = 0u;
    SG_CUDA_OK(cudaMemGetInfo(&free_init, &total_vram));
    {
        uint8_t* d_128k = nullptr;
        SG_CUDA_OK(cudaMalloc(&d_128k, siligemma::kArena128K.total_arena_bytes));
        SG_CUDA_OK(cudaMemset(d_128k, 0, siligemma::kArena128K.total_arena_bytes));
        size_t free_128k = 0u;
        SG_CUDA_OK(cudaMemGetInfo(&free_128k, &total_vram));
        const double vram_128k_used_mib = static_cast<double>(total_vram - free_128k) / (1024.0 * 1024.0);
        SG_CUDA_OK(cudaFree(d_128k));
        SG_REQUIRE(vram_128k_used_mib <= 3072.0, "kArena128K total VRAM exceeded 3072 MiB cap");
        std::fprintf(stderr, "[S184] kArena128K (%llu B) verified: total used VRAM = %.2f MiB\n",
                     static_cast<unsigned long long>(siligemma::kArena128K.total_arena_bytes), vram_128k_used_mib);

        // Allocate production 64K + LM-Shadow static arena (2,523,846,912 B)
        const auto& AL = siligemma::kArena64KShadow;
        uint8_t* d_arena = nullptr;
        SG_CUDA_OK(cudaMalloc(&d_arena, AL.total_arena_bytes));
        SG_CUDA_OK(cudaMemset(d_arena, 0, AL.total_arena_bytes));
        size_t free_64k = 0u;
        SG_CUDA_OK(cudaMemGetInfo(&free_64k, &total_vram));
        const double vram_64k_used_mib = static_cast<double>(total_vram - free_64k) / (1024.0 * 1024.0);

        // Pinned buffers
        __nv_bfloat16* h_ple_pin = nullptr;
        SG_CUDA_OK(cudaHostAlloc(&h_ple_pin, siligemma::PLE_STAGE_BYTES, cudaHostAllocDefault));
        siligemma::HoloSphereRazorCanary64* host_canary = nullptr;
        siligemma::HoloSphereRazorCanary64* dev_canary = nullptr;
        SG_CUDA_OK(cudaHostAlloc(&host_canary, sizeof(siligemma::HoloSphereRazorCanary64), cudaHostAllocMapped));
        SG_CUDA_OK(cudaHostGetDevicePointer(&dev_canary, host_canary, 0));
        std::memset(host_canary, 0, sizeof(*host_canary));

        // Load PLE table (4.375 GiB) into pinned host RAM after verifying safetensors header entry & byte span
        bool ple_pinned_ok = true;
        __nv_bfloat16* h_ple_table = nullptr;
        if (cudaHostAlloc(&h_ple_table, siligemma::PLE_HOST_PINNED_BYTES, cudaHostAllocDefault) != cudaSuccess) {
            ple_pinned_ok = false;
            h_ple_table = reinterpret_cast<__nv_bfloat16*>(std::malloc(siligemma::PLE_HOST_PINNED_BYTES));
            SG_REQUIRE(h_ple_table != nullptr, "Failed to allocate host PLE table");
        }
        {
            int ple_fd = ::open(args.ple_safetensors.c_str(), O_RDONLY);
            SG_REQUIRE(ple_fd >= 0, "Failed to open PLE safetensors");
            struct stat ple_st{};
            SG_REQUIRE(::fstat(ple_fd, &ple_st) == 0 && ple_st.st_size > 8, "fstat failed on PLE safetensors");
            const uint64_t ple_fsz = static_cast<uint64_t>(ple_st.st_size);
            uint64_t hdr_len = 0ULL;
            pread_exact(ple_fd, &hdr_len, sizeof(hdr_len), 0ULL);
            SG_REQUIRE(hdr_len > 0ULL && hdr_len <= 64ULL * 1024ULL * 1024ULL && (8ULL + hdr_len) <= ple_fsz, "Invalid safetensors header length");
            std::string hdr_json(static_cast<size_t>(hdr_len), '\0');
            pread_exact(ple_fd, hdr_json.data(), static_cast<size_t>(hdr_len), sizeof(hdr_len));

            const std::string target_raw = "model.language_model.embed_tokens_per_layer.weight";
            size_t obj_start = std::string::npos;
            size_t obj_end = std::string::npos;
            size_t match_count = 0u;
            {
                int depth = 0;
                bool in_str = false;
                bool esc = false;
                bool expect_key = false;
                bool capture_next_obj = false;
                std::string cur_str;
                for (size_t idx = 0u; idx < hdr_json.size(); ++idx) {
                    const char ch = hdr_json[idx];
                    if (in_str) {
                        if (esc) { cur_str.push_back(ch); esc = false; }
                        else if (ch == '\\') { cur_str.push_back(ch); esc = true; }
                        else if (ch == '"') {
                            in_str = false;
                            if (depth == 1 && expect_key) {
                                expect_key = false;
                                if (cur_str == target_raw) {
                                    capture_next_obj = true;
                                    ++match_count;
                                    SG_REQUIRE(match_count == 1u, "Duplicate PLE tensor key in safetensors header");
                                }
                            }
                        } else {
                            if (depth == 1 && expect_key && cur_str.size() < 256u) cur_str.push_back(ch);
                        }
                        continue;
                    }
                    if (ch == '"') {
                        in_str = true;
                        if (depth == 1 && expect_key) cur_str.clear();
                    } else if (ch == '{') {
                        ++depth;
                        if (depth == 1) {
                            expect_key = true;
                        } else if (depth == 2 && capture_next_obj) {
                            obj_start = idx;
                        }
                    } else if (ch == '}') {
                        if (depth == 2 && capture_next_obj) {
                            obj_end = idx;
                            capture_next_obj = false;
                        }
                        --depth;
                        if (depth == 1) expect_key = false;
                    } else if (ch == ',' && depth == 1) {
                        expect_key = true;
                    }
                }
            }
            SG_REQUIRE(match_count == 1u && obj_start != std::string::npos && obj_end != std::string::npos,
                       "PLE tensor missing or malformed in safetensors header");
            const std::string entry = hdr_json.substr(obj_start, obj_end - obj_start + 1u);

            auto skip_ws = [&](size_t& p) {
                while (p < entry.size() && (entry[p] == ' ' || entry[p] == '\t' || entry[p] == '\n' || entry[p] == '\r')) ++p;
            };
            auto find_unique_entry_key = [&](const std::string& key) -> size_t {
                const size_t pos = entry.find(key);
                SG_REQUIRE(pos != std::string::npos, "PLE tensor missing required metadata key");
                SG_REQUIRE(entry.find(key, pos + key.size()) == std::string::npos, "Duplicate key in PLE tensor metadata");
                return pos;
            };
            auto parse_u64 = [&](size_t& p, uint64_t& out) -> bool {
                skip_ws(p);
                if (p >= entry.size() || entry[p] < '0' || entry[p] > '9') return false;
                uint64_t v = 0ULL;
                while (p < entry.size() && entry[p] >= '0' && entry[p] <= '9') {
                    const uint64_t d = static_cast<uint64_t>(entry[p] - '0');
                    if (v > (UINT64_MAX - d) / 10ULL) return false;
                    v = v * 10ULL + d;
                    ++p;
                }
                out = v;
                return true;
            };

            const size_t dt_k = find_unique_entry_key("\"dtype\"");
            size_t dt_c = entry.find(':', dt_k + 7u);
            SG_REQUIRE(dt_c != std::string::npos, "Malformed PLE dtype");
            ++dt_c; skip_ws(dt_c);
            SG_REQUIRE(dt_c + 6u <= entry.size() && entry.compare(dt_c, 6u, "\"BF16\"") == 0, "PLE tensor dtype must be BF16");

            const size_t sh_k = find_unique_entry_key("\"shape\"");
            size_t sh_p = entry.find('[', sh_k + 7u);
            SG_REQUIRE(sh_p != std::string::npos, "Malformed PLE shape array");
            ++sh_p;
            uint64_t sh0 = 0ULL, sh1 = 0ULL;
            SG_REQUIRE(parse_u64(sh_p, sh0) && sh0 == siligemma::GEMMA4_VOCAB_SIZE, "PLE shape[0] must equal vocab_size");
            skip_ws(sh_p);
            SG_REQUIRE(sh_p < entry.size() && entry[sh_p] == ',', "Malformed PLE shape separator");
            ++sh_p;
            SG_REQUIRE(parse_u64(sh_p, sh1) && sh1 == (siligemma::GEMMA4_NUM_LAYERS * siligemma::GEMMA4_PLE_DIM), "PLE shape[1] must equal 8960");
            skip_ws(sh_p);
            SG_REQUIRE(sh_p < entry.size() && entry[sh_p] == ']', "Malformed PLE shape closing bracket");

            const size_t do_k = find_unique_entry_key("\"data_offsets\"");
            size_t do_p = entry.find('[', do_k + 14u);
            SG_REQUIRE(do_p != std::string::npos, "Malformed PLE data_offsets");
            ++do_p;
            uint64_t rel_s = 0ULL, rel_e = 0ULL;
            SG_REQUIRE(parse_u64(do_p, rel_s), "Malformed PLE data_offsets[0]");
            skip_ws(do_p);
            SG_REQUIRE(do_p < entry.size() && entry[do_p] == ',', "Malformed PLE data_offsets separator");
            ++do_p;
            SG_REQUIRE(parse_u64(do_p, rel_e), "Malformed PLE data_offsets[1]");
            skip_ws(do_p);
            SG_REQUIRE(do_p < entry.size() && entry[do_p] == ']', "Malformed PLE data_offsets closing bracket");
            SG_REQUIRE(rel_e > rel_s && (rel_e - rel_s) == siligemma::PLE_HOST_PINNED_BYTES, "PLE byte span mismatch");
            const uint64_t verified_abs_off = 8ULL + hdr_len + rel_s;
            SG_REQUIRE(verified_abs_off + siligemma::PLE_HOST_PINNED_BYTES <= ple_fsz, "PLE span exceeds safetensors file size");
            SG_REQUIRE(verified_abs_off == args.ple_offset, "--ple-offset mismatches safetensors embed_tokens_per_layer.weight offset");
            pread_exact(ple_fd, h_ple_table, siligemma::PLE_HOST_PINNED_BYTES, verified_abs_off);
            ::close(ple_fd);
        }

        // Open & validate .slgb
        int slgb_fd = ::open(args.slgb_path.c_str(), O_RDONLY);
        SG_REQUIRE(slgb_fd >= 0, "Failed to open .slgb file");
        const SlgbLoadedMeta m = parse_slgb_meta(slgb_fd);

        std::vector<uint32_t> prompt_tokens = {2u, 108u, 528u, 2299u};
        if (!args.tokens_bin.empty()) {
            read_u32_vec_bin(args.tokens_bin, prompt_tokens);
        }
        SG_REQUIRE(
            siligemma::validate_runtime_contract(m.hdr, m.layers, m.linears, prompt_tokens.data(), prompt_tokens.size())
            == siligemma::ValidationError::OK,
            "Runtime SLGB contract validation failed"
        );

        // Bind arena & upload partitions cleanly in 16 MiB chunks
        ArenaDevicePtrs ptrs = bind_arena_ptrs(d_arena, AL, host_canary, dev_canary);
        std::vector<uint8_t> h_xfer(16ULL * 1024ULL * 1024ULL);
        stream_file_to_device(slgb_fd, m.file_off_p1_wire, ptrs.d_wire, m.hdr.total_wire_bytes, h_xfer.data(), h_xfer.size());
        stream_file_to_device(slgb_fd, m.file_off_p2_embed, ptrs.d_embed, siligemma::EMBED_TOKENS_BF16_BYTES, h_xfer.data(), h_xfer.size());
        SG_REQUIRE(m.has_lm_shadow, "SLGB must carry 4.5bpw lm_head shadow for S184 matrix");
        stream_file_to_device(slgb_fd, m.file_off_p2_lm_shadow, ptrs.d_lm_shadow_u64, siligemma::LM_HEAD_SHADOW_G32_BYTES, h_xfer.data(), h_xfer.size());

        std::vector<float> h_norms(siligemma::NORM_SCALAR_PARAMS_FP32);
        pread_exact(slgb_fd, h_norms.data(), h_norms.size() * sizeof(float), m.file_off_p3_norms);

        if (m.has_u4_loop) {
            stream_file_to_device(slgb_fd, m.file_off_p4_loop, ptrs.d_loop_ctrl, siligemma::LOOP_CONTROLLER_BF16_BYTES, h_xfer.data(), h_xfer.size());
        }
        ::close(slgb_fd);
        h_xfer.clear();
        h_xfer.shrink_to_fit();

        // Seed 5 L10..L14 hub Woodbury pairs cleanly (bounded rank-4 skew perturbations, norm-preserving!)
        SplitMix64 rng(184002ULL);
        std::vector<float> h_hub_a(5ULL * 1536ULL * 4ULL), h_hub_b(5ULL * 1536ULL * 4ULL);
        for (size_t i = 0u; i < h_hub_a.size(); ++i) {
            h_hub_a[i] = 0.04f * rng.next_sym_f32();
            h_hub_b[i] = 0.04f * rng.next_sym_f32();
        }

        cudaStream_t stream_compute = nullptr, stream_dma = nullptr;
        cudaEvent_t ev_fork = nullptr, ev_dma_done = nullptr;
        SG_CUDA_OK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
        SG_CUDA_OK(cudaStreamCreateWithFlags(&stream_dma, cudaStreamNonBlocking));
        SG_CUDA_OK(cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming));
        SG_CUDA_OK(cudaEventCreateWithFlags(&ev_dma_done, cudaEventDisableTiming));
        SG_CUDA_OK(cudaMemcpyAsync(ptrs.d_norms, h_norms.data(), h_norms.size() * sizeof(float), cudaMemcpyHostToDevice, stream_compute));
        SG_CUDA_OK(cudaMemcpyAsync(ptrs.d_hub_a_f4, h_hub_a.data(), h_hub_a.size() * sizeof(float), cudaMemcpyHostToDevice, stream_compute));
        SG_CUDA_OK(cudaMemcpyAsync(ptrs.d_hub_b_f4, h_hub_b.data(), h_hub_b.size() * sizeof(float), cudaMemcpyHostToDevice, stream_compute));

        // Precompute L10..14 hub 256 B Woodbury inverses in __constant__ c_wb_inv_hub_l10_14[5]
        for (int s = 0; s < 5; ++s) {
            siligemma_sm86::siligemma_precompute_woodbury_inv8_sm86<<<1, 256, 0, stream_compute>>>(
                ptrs.d_hub_a_f4 + static_cast<size_t>(s) * 1536ULL,
                ptrs.d_hub_b_f4 + static_cast<size_t>(s) * 1536ULL,
                ptrs.d_m_inv_tmp, 0.20f, 1536
            );
            SG_CUDA_OK(cudaMemcpyToSymbolAsync(
                siligemma_sm86::c_wb_inv_hub_l10_14, ptrs.d_m_inv_tmp,
                siligemma::WOODBURY_CONST_BYTES, static_cast<size_t>(s) * siligemma::WOODBURY_CONST_BYTES,
                cudaMemcpyDeviceToDevice, stream_compute
            ));
        }
        SG_CUDA_OK(cudaStreamSynchronize(stream_compute));

        // ------------------------------------------------------------------
        // KG-GFB2 Tests:
        //   (2a) 0-ULP tree_reduce_128_sm86 & fused Sandwich+H4+A16 GEMV parity
        //   (2b) Live SECDED (SEC repair + 3/3 DED trap) & IRIS Trojan catch
        //   (2c) Live Woodbury norm preservation & c_wb_inv_256b overhead
        // ------------------------------------------------------------------
        uint32_t max_tree_ulp = 0u;
        {
            const int dims[5] = {256, 512, 1536, 6144, 12288};
            const int batches[2] = {1, 16};
            std::vector<float> hv(16ULL * 12288ULL), hout(16ULL);
            for (int d : dims) {
                for (int b : batches) {
                    for (int bi = 0; bi < b; ++bi) {
                        for (int j = 0; j < d; ++j) hv[bi * d + j] = rng.next_sym_f32() * 1.3f;
                    }
                    SG_CUDA_OK(cudaMemcpyAsync(ptrs.d_ffn_gate, hv.data(), static_cast<size_t>(b) * d * sizeof(float), cudaMemcpyHostToDevice, stream_compute));
                    siligemma_sm86::siligemma_tree_reduce_probe_sm86<<<b, 128, 0, stream_compute>>>(
                        ptrs.d_ffn_gate, ptrs.d_ffn_up, d
                    );
                    SG_CUDA_OK(cudaMemcpyAsync(hout.data(), ptrs.d_ffn_up, static_cast<size_t>(b) * sizeof(float), cudaMemcpyDeviceToHost, stream_compute));
                    SG_CUDA_OK(cudaStreamSynchronize(stream_compute));
                    for (int bi = 0; bi < b; ++bi) {
                        const float cpu_s = siligemma::canonical_tree_sum_block128_f32(&hv[bi * d], static_cast<size_t>(d));
                        const uint32_t ulp = ulp_diff_f32(cpu_s, hout[bi]);
                        if (ulp > max_tree_ulp) {
                            max_tree_ulp = ulp;
                            std::fprintf(stderr, "[S184-DBG] tree mismatch d=%d b=%d bi=%d cpu=%.9g (0x%08x) gpu=%.9g (0x%08x) ulp=%u\n",
                                         d, b, bi, cpu_s, siligemma::float_bits_u32(cpu_s), hout[bi], siligemma::float_bits_u32(hout[bi]), ulp);
                        }
                    }
                }
            }
        }

        // Fused Sandwich+H4+dp4a GEMV vs CPU FP32 reference check (D=1536, M=64)
        double max_dp4a_gemv_drift = 0.0;
        uint32_t sec_repaired = 0u, ded_trapped = 0u, iris_trojan_trapped = 0u;
        float sec_repair_err = 0.0f;
        siligemma::WatchdogAction ded_action = siligemma::WatchdogAction::CONTINUE;
        siligemma::WatchdogAction trojan_action = siligemma::WatchdogAction::CONTINUE;
        {
            const int M = 64, K = 1536, G = 32;
            const int n_blk = K / 16, n_grp = K / G;
            std::vector<float> hx(K), hgam(K), hsc_f32(M * n_grp), hy_gpu(M), hy_ref(M), hy_sec(M);
            std::vector<uint16_t> hsc_f16(M * n_grp);
            std::vector<uint64_t> hw64(M * n_blk);
            for (int i = 0; i < K; ++i) {
                hx[i] = rng.next_sym_f32();
                hgam[i] = 0.8f + 0.4f * std::fabs(rng.next_sym_f32());
            }
            for (int r = 0; r < M; ++r) {
                for (int g = 0; g < n_grp; ++g) {
                    const float sc = 0.02f + 0.05f * std::fabs(rng.next_sym_f32());
                    const uint16_t b16 = f32_to_f16_bits(sc);
                    hsc_f16[r * n_grp + g] = b16;
                    hsc_f32[r * n_grp + g] = f16_bits_to_f32(b16);
                }
                for (int b = 0; b < n_blk; ++b) {
                    hw64[r * n_blk + b] = pack_a16_codeword_rng(static_cast<uint32_t>(rng.next_u64() & 0x7FFu), rng);
                }
            }
            siligemma::sandwich_h4_a16_gemv_ref(hx.data(), hgam.data(), hw64.data(), hsc_f32.data(), hy_ref.data(), M, K, G, 1e-6f);

            // Use P7 ffn_gate/ffn_up as temporary device buffers for clean SECDED test (no permanent P1 overwrite)
            float* dx = ptrs.d_h_tmp;
            float* dgam = ptrs.d_q;
            uint64_t* dw = reinterpret_cast<uint64_t*>(ptrs.d_ffn_gate);
            __half* dsc = reinterpret_cast<__half*>(ptrs.d_ffn_up);
            float* dy = ptrs.d_k;
            SG_CUDA_OK(cudaMemcpyAsync(dx, hx.data(), K * sizeof(float), cudaMemcpyHostToDevice, stream_compute));
            SG_CUDA_OK(cudaMemcpyAsync(dgam, hgam.data(), K * sizeof(float), cudaMemcpyHostToDevice, stream_compute));
            SG_CUDA_OK(cudaMemcpyAsync(dw, hw64.data(), hw64.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream_compute));
            SG_CUDA_OK(cudaMemcpyAsync(dsc, hsc_f16.data(), hsc_f16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream_compute));

            std::memset(host_canary, 0, sizeof(*host_canary));
            siligemma_sm86::siligemma_fused_sandwich_h4_dp4a_gemv_sm86<<<dim3(M, 1), 128, 0, stream_compute>>>(
                dx, dgam, dw, dsc, dy, dev_canary, 1e-6f, M, K, G
            );
            SG_CUDA_OK(cudaMemcpyAsync(hy_gpu.data(), dy, M * sizeof(float), cudaMemcpyDeviceToHost, stream_compute));
            SG_CUDA_OK(cudaStreamSynchronize(stream_compute));
            SG_REQUIRE(host_canary->sec_count_u32 == 0u && host_canary->ded_count_u32 == 0u, "Clean codeword triggered SECDED");
            for (int r = 0; r < M; ++r) {
                const double diff = std::fabs(static_cast<double>(hy_gpu[r] - hy_ref[r]));
                if (diff > max_dp4a_gemv_drift) max_dp4a_gemv_drift = diff;
            }

            // Inject 3 single-bit LSB faults (SEC repair check)
            auto hw_sec = hw64;
            hw_sec[0 * n_blk + 3] ^= (1ULL << (5 * 4));
            hw_sec[17 * n_blk + 11] ^= (1ULL << (9 * 4));
            hw_sec[44 * n_blk + 25] ^= (1ULL << (12 * 4));
            SG_CUDA_OK(cudaMemcpyAsync(dw, hw_sec.data(), hw_sec.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream_compute));
            std::memset(host_canary, 0, sizeof(*host_canary));
            siligemma_sm86::siligemma_fused_sandwich_h4_dp4a_gemv_sm86<<<dim3(M, 1), 128, 0, stream_compute>>>(
                dx, dgam, dw, dsc, dy, dev_canary, 1e-6f, M, K, G
            );
            SG_CUDA_OK(cudaMemcpyAsync(hy_sec.data(), dy, M * sizeof(float), cudaMemcpyDeviceToHost, stream_compute));
            SG_CUDA_OK(cudaStreamSynchronize(stream_compute));
            sec_repaired = host_canary->sec_count_u32;
            for (int r = 0; r < M; ++r) sec_repair_err = std::fmax(sec_repair_err, std::fabs(hy_sec[r] - hy_gpu[r]));

            // Inject 3 double-bit LSB faults (DED SafeHalt trap)
            auto hw_ded = hw64;
            hw_ded[1 * n_blk + 0] ^= (1ULL << 0) ^ (1ULL << 4);
            hw_ded[22 * n_blk + 7] ^= (1ULL << 8) ^ (1ULL << 20);
            hw_ded[59 * n_blk + 19] ^= (1ULL << 12) ^ (1ULL << 60);
            SG_CUDA_OK(cudaMemcpyAsync(dw, hw_ded.data(), hw_ded.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream_compute));
            std::memset(host_canary, 0, sizeof(*host_canary));
            siligemma_sm86::siligemma_fused_sandwich_h4_dp4a_gemv_sm86<<<dim3(M, 1), 128, 0, stream_compute>>>(
                dx, dgam, dw, dsc, dy, dev_canary, 1e-6f, M, K, G
            );
            SG_CUDA_OK(cudaStreamSynchronize(stream_compute));
            ded_trapped = host_canary->ded_count_u32;
            ded_action = siligemma::evaluate_canary_watchdog(*host_canary, 0u, false);

            // Inject IRIS state Trojan mutant
            std::memset(host_canary, 0, sizeof(*host_canary));
            siligemma_sm86::siligemma_iris_canary_finalize_sm86<<<1, 128, 0, stream_compute>>>(
                dx, dev_canary, 34u, 1u, 0u, 0, K
            );
            SG_CUDA_OK(cudaStreamSynchronize(stream_compute));
            const uint32_t golden_iris = host_canary->iris_runtime_crc_u32;

            hx[77] += 1.0e-3f; // Trojan mutant perturbation
            SG_CUDA_OK(cudaMemcpyAsync(dx, hx.data(), K * sizeof(float), cudaMemcpyHostToDevice, stream_compute));
            std::memset(host_canary, 0, sizeof(*host_canary));
            siligemma_sm86::siligemma_iris_canary_finalize_sm86<<<1, 128, 0, stream_compute>>>(
                dx, dev_canary, 34u, 1u, golden_iris, 1, K
            );
            SG_CUDA_OK(cudaStreamSynchronize(stream_compute));
            iris_trojan_trapped = host_canary->iris_mismatch_count_u32;
            trojan_action = siligemma::evaluate_canary_watchdog(*host_canary, golden_iris, true);
        }

        // Woodbury 10,000-vector + 4-corner norm preservation + c_wb_inv_256b AB/BA hoist overhead
        double max_wb_norm_drift = 0.0;
        double wb_overhead_pct_d1536 = 0.0;
        double wb_med_us = 0.0, euc_med_us = 0.0;
        {
            const int D = 1536;
            const int df4 = D * 4;
            std::vector<float> hA(df4), hB(df4), hX(D), hY(D);
            float4* d_wb_a_tmp = reinterpret_cast<float4*>(ptrs.d_q);
            float4* d_wb_b_tmp = d_wb_a_tmp + D;
            auto test_wb_case = [&](float alpha, int n_vecs, uint64_t seed) {
                SplitMix64 rcase(seed);
                for (int i = 0; i < df4; ++i) {
                    if (seed == 1ULL) { hA[i] = 0.0f; hB[i] = 0.0f; } // corner 1: alpha=0 / zeros
                    else if (seed == 2ULL) { float v = 0.05f * rcase.next_sym_f32(); hA[i] = v; hB[i] = v; } // corner 2: A=B
                    else if (seed == 3ULL) { hA[i] = (i < df4 / 2) ? 0.05f * rcase.next_sym_f32() : 0.0f; hB[i] = (i >= df4 / 2) ? 0.05f * rcase.next_sym_f32() : 0.0f; } // orthogonal
                    else { hA[i] = 0.06f * rcase.next_sym_f32(); hB[i] = 0.06f * rcase.next_sym_f32(); }
                }
                SG_CUDA_OK(cudaMemcpyAsync(d_wb_a_tmp, hA.data(), df4 * sizeof(float), cudaMemcpyHostToDevice, stream_compute));
                SG_CUDA_OK(cudaMemcpyAsync(d_wb_b_tmp, hB.data(), df4 * sizeof(float), cudaMemcpyHostToDevice, stream_compute));
                siligemma_sm86::siligemma_precompute_woodbury_inv8_sm86<<<1, 256, 0, stream_compute>>>(
                    d_wb_a_tmp, d_wb_b_tmp, ptrs.d_m_inv_tmp, alpha, D
                );
                SG_CUDA_OK(cudaMemcpyToSymbolAsync(
                    siligemma_sm86::c_wb_inv_256b, ptrs.d_m_inv_tmp,
                    siligemma::WOODBURY_CONST_BYTES, 0, cudaMemcpyDeviceToDevice, stream_compute
                ));
                for (int v = 0; v < n_vecs; ++v) {
                    double n_in_sq = 0.0;
                    for (int j = 0; j < D; ++j) {
                        hX[j] = 1.2f * rcase.next_sym_f32();
                        n_in_sq += static_cast<double>(hX[j]) * static_cast<double>(hX[j]);
                    }
                    SG_CUDA_OK(cudaMemcpyAsync(ptrs.d_h_cur, hX.data(), D * sizeof(float), cudaMemcpyHostToDevice, stream_compute));
                    siligemma_sm86::siligemma_cayley_woodbury_r4_hoisted_sm86<<<1, 256, 0, stream_compute>>>(
                        ptrs.d_h_cur, d_wb_a_tmp, d_wb_b_tmp, ptrs.d_h_tmp, alpha, D
                    );
                    SG_CUDA_OK(cudaMemcpyAsync(hY.data(), ptrs.d_h_tmp, D * sizeof(float), cudaMemcpyDeviceToHost, stream_compute));
                    SG_CUDA_OK(cudaStreamSynchronize(stream_compute));
                    double n_out_sq = 0.0;
                    for (int j = 0; j < D; ++j) n_out_sq += static_cast<double>(hY[j]) * static_cast<double>(hY[j]);
                    const double rel = std::fabs(std::sqrt(n_out_sq) - std::sqrt(n_in_sq)) / std::max(std::sqrt(n_in_sq), 1e-12);
                    if (rel > max_wb_norm_drift) max_wb_norm_drift = rel;
                }
            };
            test_wb_case(0.0f, 4, 1ULL);
            test_wb_case(0.25f, 4, 2ULL);
            test_wb_case(0.25f, 4, 3ULL);
            test_wb_case(10.0f, 4, 4ULL);
            // Re-seed clean 0.25f case for 10,000-vector GPU sweep & AB/BA overhead graph
            test_wb_case(0.25f, 4, 5ULL);
            for (int blk = 0; blk < 100; ++blk) {
                std::vector<float> hXbatch(100ULL * D), hYbatch(100ULL * D);
                for (size_t idx = 0u; idx < hXbatch.size(); ++idx) hXbatch[idx] = 1.2f * rng.next_sym_f32();
                SG_CUDA_OK(cudaMemcpyAsync(ptrs.d_ffn_gate, hXbatch.data(), hXbatch.size() * sizeof(float), cudaMemcpyHostToDevice, stream_compute));
                for (int vi = 0; vi < 100; ++vi) {
                    siligemma_sm86::siligemma_cayley_woodbury_r4_hoisted_sm86<<<1, 256, 0, stream_compute>>>(
                        ptrs.d_ffn_gate + vi * D, d_wb_a_tmp, d_wb_b_tmp,
                        ptrs.d_ffn_up + vi * D, 0.25f, D
                    );
                }
                SG_CUDA_OK(cudaMemcpyAsync(hYbatch.data(), ptrs.d_ffn_up, hYbatch.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_compute));
                SG_CUDA_OK(cudaStreamSynchronize(stream_compute));
                for (int vi = 0; vi < 100; ++vi) {
                    double s0 = 0.0, s1 = 0.0;
                    for (int j = 0; j < D; ++j) {
                        const double a = hXbatch[vi * D + j], b = hYbatch[vi * D + j];
                        s0 += a * a; s1 += b * b;
                    }
                    const double rel = std::fabs(std::sqrt(s1) - std::sqrt(s0)) / std::max(std::sqrt(s0), 1e-12);
                    if (rel > max_wb_norm_drift) max_wb_norm_drift = rel;
                }
            }

            // AB/BA CUDA Graph overhead check: 16x-batched graph per launch (matching S180c B=16 pattern)
            CapturedGraph g_euc{}, g_wb{};
            SG_CUDA_OK(cudaStreamBeginCapture(stream_compute, cudaStreamCaptureModeThreadLocal));
            for (int c = 0; c < 16; ++c) {
                siligemma_sm86::siligemma_euclidean_lora_r4_sm86<<<1, 128, 0, stream_compute>>>(
                    ptrs.d_ffn_gate + c * D, d_wb_a_tmp, d_wb_b_tmp, ptrs.d_ffn_up + c * D, 0.25f, D
                );
            }
            SG_CUDA_OK(cudaStreamEndCapture(stream_compute, &g_euc.graph));
            SG_CUDA_OK(cudaGraphInstantiate(&g_euc.exec, g_euc.graph, nullptr, nullptr, 0));

            SG_CUDA_OK(cudaStreamBeginCapture(stream_compute, cudaStreamCaptureModeThreadLocal));
            for (int c = 0; c < 16; ++c) {
                siligemma_sm86::siligemma_cayley_woodbury_r4_hoisted_sm86<<<1, 256, 0, stream_compute>>>(
                    ptrs.d_ffn_gate + c * D, d_wb_a_tmp, d_wb_b_tmp, ptrs.d_ffn_up + c * D, 0.25f, D
                );
            }
            SG_CUDA_OK(cudaStreamEndCapture(stream_compute, &g_wb.graph));
            SG_CUDA_OK(cudaGraphInstantiate(&g_wb.exec, g_wb.graph, nullptr, nullptr, 0));

            std::vector<double> ta, tb;
            benchmark_abba_graphs(g_euc, g_wb, stream_compute, 60, 10, ta, tb);
            euc_med_us = median_of_vec(ta) * 1000.0;
            wb_med_us = median_of_vec(tb) * 1000.0;
            wb_overhead_pct_d1536 = (wb_med_us - euc_med_us) / std::max(euc_med_us, 1e-9) * 100.0;
            g_euc.destroy();
            g_wb.destroy();
        }
        std::fprintf(stderr, "[S184] KG-GFB2 done: tree_ulp=%u gemv_drift=%.3e wb_norm_drift=%.3e wb_oh=%.2f%%\n",
                     max_tree_ulp, max_dp4a_gemv_drift, max_wb_norm_drift, wb_overhead_pct_d1536);

        // ------------------------------------------------------------------
        // Clear KV ring & Canary before Prompt / WT2 / Roofline Suite
        // ------------------------------------------------------------------
        SG_CUDA_OK(cudaMemset(ptrs.d_kv_base, 0, AL.sz_p5_zed_kv_ring));
        std::memset(host_canary, 0, sizeof(*host_canary));

        // Prompt logits capture (real Gemma-4 tokens, B=1, K=1): BF16 lm_head + 4.5bpw lm_head
        auto run_prompt_and_dump = [&](bool use_4p5_lm, const std::string& out_path) {
            if (out_path.empty()) return;
            SG_CUDA_OK(cudaMemset(ptrs.d_kv_base, 0, AL.sz_p5_zed_kv_ring));
            std::memset(host_canary, 0, sizeof(*host_canary));
            CapturedGraph g_prompt = capture_full_step_graph(
                ptrs, m, h_norms, 1, 1, use_4p5_lm, 0, true, true,
                h_ple_pin, ptrs.d_ple_stage_0, false, 0.0f,
                stream_compute, stream_dma, ev_fork, ev_dma_done
            );
            SG_REQUIRE(prompt_tokens.size() <= static_cast<size_t>(AL.t_max), "prompt_tokens exceeds arena t_max capacity");
            for (size_t t = 0u; t < prompt_tokens.size(); ++t) {
                const int32_t tok = static_cast<int32_t>(prompt_tokens[t]);
                const int32_t pos_val = static_cast<int32_t>(t);
                std::memcpy(
                    h_ple_pin,
                    h_ple_table + static_cast<size_t>(tok) * (siligemma::PLE_DMA_BYTES_PER_TOKEN / 2u),
                    siligemma::PLE_DMA_BYTES_PER_TOKEN
                );
                SG_CUDA_OK(cudaMemcpyAsync(ptrs.d_tok_in, &tok, sizeof(int32_t), cudaMemcpyHostToDevice, stream_compute));
                SG_CUDA_OK(cudaMemcpyAsync(ptrs.d_pos_i32, &pos_val, sizeof(int32_t), cudaMemcpyHostToDevice, stream_compute));
                SG_CUDA_OK(cudaGraphLaunch(g_prompt.exec, stream_compute));
                SG_CUDA_OK(cudaStreamSynchronize(stream_compute));
                SG_REQUIRE(
                    siligemma::evaluate_canary_watchdog(*host_canary, 0u, false) == siligemma::WatchdogAction::CONTINUE,
                    "Watchdog SAFE_HALT_DED_ROLLBACK triggered during prompt decode before output commit"
                );
            }
            std::vector<float> logits_out(siligemma::GEMMA4_VOCAB_SIZE);
            SG_CUDA_OK(cudaMemcpy(logits_out.data(), ptrs.d_logits, logits_out.size() * sizeof(float), cudaMemcpyDeviceToHost));
            std::ofstream ofs(out_path, std::ios::binary | std::ios::trunc);
            SG_REQUIRE(ofs.good(), "Failed opening logits dump path");
            ofs.write(reinterpret_cast<const char*>(logits_out.data()), static_cast<std::streamsize>(logits_out.size() * sizeof(float)));
            ofs.close();
            g_prompt.destroy();
        };
        run_prompt_and_dump(false, args.dump_logits_bf16);
        run_prompt_and_dump(true, args.dump_logits_4p5);

        // Real WT2_test B=16 NLL in bare-metal CUDA (K=1,2,3 BF16 lm_head + K=1 4.5bpw lm_head)
        const bool wt2_cuda_measured = !args.wt2_bin.empty();
        double wt2_cuda_k1_bf16 = 0.0, wt2_cuda_k1_4p5 = 0.0, wt2_cuda_k2_bf16 = 0.0, wt2_cuda_k3_bf16 = 0.0;
        double wt2_cuda_k2_cos = 0.0, wt2_cuda_k2_step_ratio = 0.0;
        double wt2_cuda_k3_rho_hat = 0.0, wt2_cuda_k3_step_ratio = 0.0, wt2_cuda_k3_cos = 0.0;
        if (wt2_cuda_measured) {
            std::vector<uint32_t> wt2_flat;
            read_u32_vec_bin(args.wt2_bin, wt2_flat);
            SG_REQUIRE(wt2_flat.size() == 16ULL * 256ULL, "wt2-bin must be 16x256 uint32");
            for (int r = 0; r < 16; ++r) {
                SG_REQUIRE(wt2_flat[r * 256] == 2u, "WT2 sequence missing bos=2");
            }
            for (const uint32_t tk : wt2_flat) {
                SG_REQUIRE(tk < siligemma::GEMMA4_VOCAB_SIZE, "WT2 token id >= vocab_size");
            }
            auto eval_wt2_cuda = [&](int k_pass, bool use_4p5) -> double {
                SG_CUDA_OK(cudaMemset(ptrs.d_kv_base, 0, AL.sz_p5_zed_kv_ring));
                std::memset(host_canary, 0, sizeof(*host_canary));
                CapturedGraph g_w = capture_full_step_graph(
                    ptrs, m, h_norms, 16, k_pass, use_4p5, 0, true, true,
                    h_ple_pin, ptrs.d_ple_stage_0, false, 0.0f,
                    stream_compute, stream_dma, ev_fork, ev_dma_done
                );
                std::vector<float> h_logits(16ULL * siligemma::GEMMA4_VOCAB_SIZE);
                std::vector<int32_t> cur_toks(16);
                std::vector<uint32_t> next_tgts(16);
                double nll_sum = 0.0;
                uint64_t tok_cnt = 0ULL;
                SG_REQUIRE(255u < AL.t_max, "WT2 sequence exceeds arena t_max capacity");
                for (int t = 0; t < 255; ++t) {
                    for (int r = 0; r < 16; ++r) {
                        const uint32_t tk = wt2_flat[r * 256 + t];
                        cur_toks[r] = static_cast<int32_t>(tk);
                        next_tgts[r] = wt2_flat[r * 256 + t + 1];
                        std::memcpy(
                            h_ple_pin + static_cast<size_t>(r) * (siligemma::PLE_DMA_BYTES_PER_TOKEN / 2u),
                            h_ple_table + static_cast<size_t>(tk) * (siligemma::PLE_DMA_BYTES_PER_TOKEN / 2u),
                            siligemma::PLE_DMA_BYTES_PER_TOKEN
                        );
                    }
                    const int32_t pos_val = t;
                    SG_CUDA_OK(cudaMemcpyAsync(ptrs.d_tok_in, cur_toks.data(), 16 * sizeof(int32_t), cudaMemcpyHostToDevice, stream_compute));
                    SG_CUDA_OK(cudaMemcpyAsync(ptrs.d_pos_i32, &pos_val, sizeof(int32_t), cudaMemcpyHostToDevice, stream_compute));
                    SG_CUDA_OK(cudaGraphLaunch(g_w.exec, stream_compute));
                    SG_CUDA_OK(cudaMemcpyAsync(h_logits.data(), ptrs.d_logits, h_logits.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_compute));
                    SG_CUDA_OK(cudaStreamSynchronize(stream_compute));
                    SG_REQUIRE(
                        siligemma::evaluate_canary_watchdog(*host_canary, 0u, false) == siligemma::WatchdogAction::CONTINUE,
                        "Watchdog SAFE_HALT_DED_ROLLBACK triggered during WT2 eval"
                    );
                    compute_batch_nll_cpu(h_logits.data(), next_tgts.data(), 16, static_cast<int>(siligemma::GEMMA4_VOCAB_SIZE), nll_sum, tok_cnt);
                }
                g_w.destroy();
                return nll_sum / static_cast<double>(tok_cnt);
            };
            wt2_cuda_k1_bf16 = eval_wt2_cuda(1, false);
            wt2_cuda_k1_4p5 = eval_wt2_cuda(1, true);
            wt2_cuda_k2_bf16 = eval_wt2_cuda(2, false);
            {
                std::vector<float> h_l33_p2(1536ULL);
                std::vector<uint16_t> h_l33_p1(1536ULL);
                SG_CUDA_OK(cudaMemcpy(h_l33_p2.data(), ptrs.d_h_pre24, h_l33_p2.size() * sizeof(float), cudaMemcpyDeviceToHost));
                SG_CUDA_OK(cudaMemcpy(h_l33_p1.data(), ptrs.d_loop_prev + 8ULL * 16ULL * 1536ULL, 1536ULL * 2ULL, cudaMemcpyDeviceToHost));
                std::vector<float> h_l33_p1_f(1536ULL);
                for (size_t j = 0; j < 1536ULL; ++j) h_l33_p1_f[j] = bf16_bits_to_f32(h_l33_p1[j]);
                double dummy_rho = 0.0;
                float sr = 0.0f, cs = 0.0f;
                const siligemma::ValidationError k2_guard_err = siligemma::evaluate_riccati_loop_guard(
                    h_l33_p1_f.data(), h_l33_p2.data(), nullptr, 1536u, 2u, dummy_rho, sr, cs
                );
                SG_REQUIRE(k2_guard_err == siligemma::ValidationError::OK, "K=2 loop step divergence guard failed");
                wt2_cuda_k2_step_ratio = static_cast<double>(sr);
                wt2_cuda_k2_cos = static_cast<double>(cs);
            }
            wt2_cuda_k3_bf16 = eval_wt2_cuda(3, false);
            {
                std::vector<float> h_p3(1536ULL), h_p1(1536ULL), h_p2(1536ULL), h_prev_delta(1536ULL);
                std::vector<uint16_t> b_p1(1536ULL), b_p2(1536ULL);
                SG_CUDA_OK(cudaMemcpy(b_p1.data(), ptrs.d_loop_prev, 1536ULL * 2ULL, cudaMemcpyDeviceToHost));
                SG_CUDA_OK(cudaMemcpy(b_p2.data(), ptrs.d_loop_prev + 8ULL * 16ULL * 1536ULL, 1536ULL * 2ULL, cudaMemcpyDeviceToHost));
                SG_CUDA_OK(cudaMemcpy(h_p3.data(), ptrs.d_h_pre24, 1536ULL * sizeof(float), cudaMemcpyDeviceToHost));
                for (size_t j = 0; j < 1536ULL; ++j) {
                    h_p1[j] = bf16_bits_to_f32(b_p1[j]);
                    h_p2[j] = bf16_bits_to_f32(b_p2[j]);
                    h_prev_delta[j] = h_p2[j] - h_p1[j];
                }
                double rho3 = 0.0;
                float sr3 = 0.0f, cs3 = 0.0f;
                const siligemma::ValidationError k3_guard_err = siligemma::evaluate_riccati_loop_guard(
                    h_p2.data(), h_p3.data(), h_prev_delta.data(), 1536u, 3u, rho3, sr3, cs3
                );
                SG_REQUIRE(k3_guard_err == siligemma::ValidationError::OK, "K=3 Riccati contraction guard failed");
                wt2_cuda_k3_rho_hat = rho3;
                wt2_cuda_k3_step_ratio = static_cast<double>(sr3);
                wt2_cuda_k3_cos = static_cast<double>(cs3);
            }
            std::fprintf(stderr, "[S184] Bare-metal CUDA WT2_test (B=16): K1_BF16=%.6f K1_4.5=%.6f K2_BF16=%.6f K3_BF16=%.6f rho_k3=%.6f\n",
                         wt2_cuda_k1_bf16, wt2_cuda_k1_4p5, wt2_cuda_k2_bf16, wt2_cuda_k3_bf16, wt2_cuda_k3_rho_hat);
        }

        // ------------------------------------------------------------------
        // KG-GFB1: 0 B Alloc Check across 1,000 Decode Steps + AB/BA Async PLE DMA Stall %
        // ------------------------------------------------------------------
        size_t alloc_vram_delta_1k = 0u;
        size_t alloc_heap_delta_1k = 0u;
        double dma_stall_b1_pct = 0.0, dma_stall_b1_no_ms = 0.0, dma_stall_b1_with_ms = 0.0;
        {
            const int32_t tok_bos = 2;
            SG_CUDA_OK(cudaMemcpyAsync(ptrs.d_tok_in, &tok_bos, sizeof(int32_t), cudaMemcpyHostToDevice, stream_compute));
            std::memcpy(h_ple_pin, h_ple_table + static_cast<size_t>(tok_bos) * (siligemma::PLE_DMA_BYTES_PER_TOKEN / 2u), siligemma::PLE_DMA_BYTES_PER_TOKEN);
            SG_CUDA_OK(cudaMemcpyAsync(ptrs.d_ple_stage_0, h_ple_pin, siligemma::PLE_DMA_BYTES_PER_TOKEN, cudaMemcpyHostToDevice, stream_compute));
            SG_CUDA_OK(cudaStreamSynchronize(stream_compute));

            // AB/BA async PLE DMA vs pre-resident at K=1 B=1 T=128
            CapturedGraph g_no_dma = capture_full_step_graph(
                ptrs, m, h_norms, 1, 1, true, 127, false, false,
                h_ple_pin, ptrs.d_ple_stage_0, false, 0.0f,
                stream_compute, stream_dma, ev_fork, ev_dma_done
            );
            CapturedGraph g_with_dma = capture_full_step_graph(
                ptrs, m, h_norms, 1, 1, true, 127, false, true,
                h_ple_pin, ptrs.d_ple_stage_0, false, 0.0f,
                stream_compute, stream_dma, ev_fork, ev_dma_done
            );
            std::vector<double> t_no, t_with;
            benchmark_abba_graphs(g_no_dma, g_with_dma, stream_compute, 20, 10, t_no, t_with);
            dma_stall_b1_no_ms = median_of_vec(t_no);
            dma_stall_b1_with_ms = median_of_vec(t_with);
            dma_stall_b1_pct = (dma_stall_b1_with_ms - dma_stall_b1_no_ms) / std::max(dma_stall_b1_no_ms, 1e-9) * 100.0;

            // 1,000-step 0 B dynamic alloc check (K=1 + K=2 + K=3 sanity steps)
            CapturedGraph g_k2 = capture_full_step_graph(
                ptrs, m, h_norms, 1, 2, true, 127, false, true,
                h_ple_pin, ptrs.d_ple_stage_0, false, 0.0f,
                stream_compute, stream_dma, ev_fork, ev_dma_done
            );
            CapturedGraph g_k3 = capture_full_step_graph(
                ptrs, m, h_norms, 1, 3, true, 127, false, true,
                h_ple_pin, ptrs.d_ple_stage_0, false, 0.0f,
                stream_compute, stream_dma, ev_fork, ev_dma_done
            );
            // Warmup graphs once so driver state is settled
            SG_CUDA_OK(cudaGraphLaunch(g_with_dma.exec, stream_compute));
            SG_CUDA_OK(cudaGraphLaunch(g_k2.exec, stream_compute));
            SG_CUDA_OK(cudaGraphLaunch(g_k3.exec, stream_compute));
            SG_CUDA_OK(cudaStreamSynchronize(stream_compute));

            size_t free_b = 0u, free_a = 0u;
            SG_CUDA_OK(cudaMemGetInfo(&free_b, &total_vram));
            const auto mi_b = ::mallinfo2();
            for (int step = 0; step < 1000; ++step) {
                SG_CUDA_OK(cudaGraphLaunch(g_with_dma.exec, stream_compute));
            }
            SG_CUDA_OK(cudaGraphLaunch(g_k2.exec, stream_compute));
            SG_CUDA_OK(cudaGraphLaunch(g_k3.exec, stream_compute));
            SG_CUDA_OK(cudaStreamSynchronize(stream_compute));
            const auto mi_a = ::mallinfo2();
            SG_CUDA_OK(cudaMemGetInfo(&free_a, &total_vram));
            alloc_vram_delta_1k = (free_b > free_a) ? (free_b - free_a) : 0u;
            alloc_heap_delta_1k = (mi_a.uordblks > mi_b.uordblks) ? (mi_a.uordblks - mi_b.uordblks) : 0u;

            g_k3.destroy();
            g_k2.destroy();
            g_with_dma.destroy();
            g_no_dma.destroy();
        }
        std::fprintf(stderr, "[S184] KG-GFB1 done: vram_delta=%zu heap_delta=%zu dma_stall=%.2f%%\n",
                     alloc_vram_delta_1k, alloc_heap_delta_1k, dma_stall_b1_pct);

        // ------------------------------------------------------------------
        // KG-GFB3: 12-Matrix Live RTX 3080 CUDA Graph Throughput (B in {1, 16} x K in {1,2,3} x {4.5bpw, BF16})
        //          + Live L10..L14 20x Hub Steering Overhead Check
        // ------------------------------------------------------------------
        struct ThroughputEntry {
            int batch;
            int k_pass;
            bool use_4p5;
            bool hub_steer;
            double median_ms;
            double p95_ms;
            double tok_per_s;
        };
        std::vector<ThroughputEntry> tp_results;
        for (int B : {1, 16}) {
            std::vector<int32_t> toks_b(static_cast<size_t>(B), 2);
            SG_CUDA_OK(cudaMemcpyAsync(ptrs.d_tok_in, toks_b.data(), toks_b.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream_compute));
            SG_CUDA_OK(cudaStreamSynchronize(stream_compute));
            for (int r = 0; r < B; ++r) {
                std::memcpy(
                    h_ple_pin + static_cast<size_t>(r) * (siligemma::PLE_DMA_BYTES_PER_TOKEN / 2u),
                    h_ple_table + 2ULL * (siligemma::PLE_DMA_BYTES_PER_TOKEN / 2u),
                    siligemma::PLE_DMA_BYTES_PER_TOKEN
                );
            }
            const int eval_pos = (B == 1) ? 127 : 1023;
            const int iters = (B == 1) ? 80 : 12;
            const int warmup = (B == 1) ? 10 : 3;
            for (int k_pass : {1, 2, 3}) {
                for (bool use_4p5 : {true, false}) {
                    CapturedGraph g = capture_full_step_graph(
                        ptrs, m, h_norms, B, k_pass, use_4p5, eval_pos, false, true,
                        h_ple_pin, ptrs.d_ple_stage_0, false, 0.0f,
                        stream_compute, stream_dma, ev_fork, ev_dma_done
                    );
                    cudaEvent_t e0 = nullptr, e1 = nullptr;
                    SG_CUDA_OK(cudaEventCreate(&e0));
                    SG_CUDA_OK(cudaEventCreate(&e1));
                    for (int w = 0; w < warmup; ++w) SG_CUDA_OK(cudaGraphLaunch(g.exec, stream_compute));
                    SG_CUDA_OK(cudaStreamSynchronize(stream_compute));
                    std::vector<double> ms_samples;
                    ms_samples.reserve(static_cast<size_t>(iters));
                    for (int it = 0; it < iters; ++it) {
                        SG_CUDA_OK(cudaEventRecord(e0, stream_compute));
                        SG_CUDA_OK(cudaGraphLaunch(g.exec, stream_compute));
                        SG_CUDA_OK(cudaEventRecord(e1, stream_compute));
                        SG_CUDA_OK(cudaEventSynchronize(e1));
                        float dt_ms = 0.0f;
                        SG_CUDA_OK(cudaEventElapsedTime(&dt_ms, e0, e1));
                        ms_samples.push_back(static_cast<double>(dt_ms));
                    }
                    SG_CUDA_OK(cudaEventDestroy(e1));
                    SG_CUDA_OK(cudaEventDestroy(e0));
                    const double med = median_of_vec(ms_samples);
                    const double p95 = percentile_of_vec(ms_samples, 0.95);
                    const double tps = (static_cast<double>(B) * 1000.0) / std::max(med, 1e-9);
                    tp_results.push_back({B, k_pass, use_4p5, false, med, p95, tps});
                    std::fprintf(stderr, "[S184] KG-GFB3 B=%d T=%d K=%d lm=%s: med=%.3fms p95=%.3fms tok/s=%.2f\n",
                                 B, eval_pos + 1, k_pass, use_4p5 ? "4.5bpw" : "bf16", med, p95, tps);
                    g.destroy();
                }
            }
        }

        // Hub L10..L14 steering AB/BA end-to-end overhead at B=1 K=1 4.5bpw lm_head + 5-slot norm check
        double hub_base_ms = 0.0, hub_steer_ms = 0.0, hub_overhead_pct = 0.0, hub_max_norm_drift = 0.0;
        {
            std::vector<float> hx(1536ULL), hy(1536ULL);
            for (size_t j = 0; j < 1536ULL; ++j) hx[j] = rng.next_sym_f32() * 1.1f;
            double n_in = 0.0;
            for (float v : hx) n_in += static_cast<double>(v) * static_cast<double>(v);
            n_in = std::sqrt(n_in);
            SG_CUDA_OK(cudaMemcpyAsync(ptrs.d_h_cur, hx.data(), 1536ULL * sizeof(float), cudaMemcpyHostToDevice, stream_compute));
            for (int hs = 0; hs < 5; ++hs) {
                siligemma_sm86::siligemma_cayley_woodbury_hub_l10_14_sm86<<<1, 256, 0, stream_compute>>>(
                    ptrs.d_h_cur,
                    ptrs.d_hub_a_f4 + static_cast<size_t>(hs) * 1536ULL,
                    ptrs.d_hub_b_f4 + static_cast<size_t>(hs) * 1536ULL,
                    ptrs.d_h_tmp, 0.20f, 1536, hs
                );
                SG_CUDA_OK(cudaMemcpyAsync(hy.data(), ptrs.d_h_tmp, 1536ULL * sizeof(float), cudaMemcpyDeviceToHost, stream_compute));
                SG_CUDA_OK(cudaStreamSynchronize(stream_compute));
                double n_out = 0.0;
                for (float v : hy) n_out += static_cast<double>(v) * static_cast<double>(v);
                const double rel = std::fabs(std::sqrt(n_out) - n_in) / std::max(n_in, 1e-12);
                if (rel > hub_max_norm_drift) hub_max_norm_drift = rel;
            }

            CapturedGraph g_hb = capture_full_step_graph(
                ptrs, m, h_norms, 1, 1, true, 127, false, true,
                h_ple_pin, ptrs.d_ple_stage_0, false, 0.0f,
                stream_compute, stream_dma, ev_fork, ev_dma_done
            );
            CapturedGraph g_hs = capture_full_step_graph(
                ptrs, m, h_norms, 1, 1, true, 127, false, true,
                h_ple_pin, ptrs.d_ple_stage_0, true, 0.20f,
                stream_compute, stream_dma, ev_fork, ev_dma_done
            );
            std::vector<double> tb, ts;
            benchmark_abba_graphs(g_hb, g_hs, stream_compute, 20, 5, tb, ts);
            hub_base_ms = median_of_vec(tb);
            hub_steer_ms = median_of_vec(ts);
            hub_overhead_pct = (hub_steer_ms - hub_base_ms) / std::max(hub_base_ms, 1e-9) * 100.0;
            g_hs.destroy();
            g_hb.destroy();
        }

        // Write structured JSON out
        {
            std::ofstream jout(args.json_out, std::ios::trunc);
            SG_REQUIRE(jout.good(), "Failed opening --json-out");
            jout << "{\n";
            jout << "  \"arena_128k_bytes\": " << siligemma::kArena128K.total_arena_bytes << ",\n";
            jout << "  \"arena_128k_vram_mib\": " << vram_128k_used_mib << ",\n";
            jout << "  \"arena_64k_shadow_bytes\": " << AL.total_arena_bytes << ",\n";
            jout << "  \"arena_64k_shadow_vram_mib\": " << vram_64k_used_mib << ",\n";
            jout << "  \"ple_table_bytes\": " << siligemma::PLE_HOST_PINNED_BYTES << ",\n";
            jout << "  \"ple_table_pinned\": " << (ple_pinned_ok ? "true" : "false") << ",\n";
            jout << "  \"live_linears\": " << m.hdr.num_live_linears << ",\n";
            jout << "  \"dead_kv_omitted\": " << m.hdr.num_dead_kv_omitted << ",\n";
            jout << "  \"wire_bytes\": " << m.hdr.total_wire_bytes << ",\n";
            jout << "  \"bos_token_id\": " << m.hdr.bos_token_id << ",\n";
            jout << "  \"alloc_vram_delta_bytes_1k\": " << alloc_vram_delta_1k << ",\n";
            jout << "  \"alloc_heap_delta_bytes_1k\": " << alloc_heap_delta_1k << ",\n";
            jout << "  \"dma_stall_b1_no_dma_ms\": " << dma_stall_b1_no_ms << ",\n";
            jout << "  \"dma_stall_b1_with_dma_ms\": " << dma_stall_b1_with_ms << ",\n";
            jout << "  \"dma_stall_b1_pct\": " << dma_stall_b1_pct << ",\n";
            jout << "  \"tree_reduce_max_ulp\": " << max_tree_ulp << ",\n";
            jout << "  \"dp4a_gemv_max_abs_diff_vs_fp32_ref\": " << max_dp4a_gemv_drift << ",\n";
            jout << "  \"secded_sec_repaired_count\": " << sec_repaired << ",\n";
            jout << "  \"secded_sec_repair_max_err\": " << sec_repair_err << ",\n";
            jout << "  \"secded_ded_trapped_count\": " << ded_trapped << ",\n";
            jout << "  \"secded_ded_watchdog_safehalt\": " << (ded_action == siligemma::WatchdogAction::SAFE_HALT_DED_ROLLBACK ? "true" : "false") << ",\n";
            jout << "  \"iris_trojan_trapped_count\": " << iris_trojan_trapped << ",\n";
            jout << "  \"iris_trojan_watchdog_safehalt\": " << (trojan_action == siligemma::WatchdogAction::SAFE_HALT_DED_ROLLBACK ? "true" : "false") << ",\n";
            jout << "  \"woodbury_max_rel_norm_drift\": " << max_wb_norm_drift << ",\n";
            jout << "  \"woodbury_hoisted_d1536_us\": " << wb_med_us << ",\n";
            jout << "  \"euclidean_lora_d1536_us\": " << euc_med_us << ",\n";
            jout << "  \"woodbury_hoisted_overhead_pct\": " << wb_overhead_pct_d1536 << ",\n";
            jout << "  \"hub_l10_14_max_rel_norm_drift\": " << hub_max_norm_drift << ",\n";
            jout << "  \"hub_l10_14_base_ms\": " << hub_base_ms << ",\n";
            jout << "  \"hub_l10_14_steer_ms\": " << hub_steer_ms << ",\n";
            jout << "  \"hub_l10_14_overhead_pct\": " << hub_overhead_pct << ",\n";
            auto emit_opt_f64 = [&](const char* key, double val) {
                jout << "  \"" << key << "\": ";
                if (wt2_cuda_measured) jout << val;
                else jout << "null";
                jout << ",\n";
            };
            jout << "  \"wt2_cuda_measured\": " << (wt2_cuda_measured ? "true" : "false") << ",\n";
            jout << "  \"riccati_guards_enforced\": " << (wt2_cuda_measured ? "true" : "false") << ",\n";
            emit_opt_f64("wt2_cuda_k1_bf16_nll", wt2_cuda_k1_bf16);
            emit_opt_f64("wt2_cuda_k1_4p5_nll", wt2_cuda_k1_4p5);
            emit_opt_f64("wt2_cuda_k2_bf16_nll", wt2_cuda_k2_bf16);
            emit_opt_f64("wt2_cuda_k3_bf16_nll", wt2_cuda_k3_bf16);
            emit_opt_f64("wt2_cuda_k2_step_ratio", wt2_cuda_k2_step_ratio);
            emit_opt_f64("wt2_cuda_k2_cos", wt2_cuda_k2_cos);
            emit_opt_f64("wt2_cuda_k3_rho_hat", wt2_cuda_k3_rho_hat);
            emit_opt_f64("wt2_cuda_k3_step_ratio", wt2_cuda_k3_step_ratio);
            emit_opt_f64("wt2_cuda_k3_cos", wt2_cuda_k3_cos);
            jout << "  \"canary_final_ded_count\": " << host_canary->ded_count_u32 << ",\n";
            jout << "  \"canary_final_sec_count\": " << host_canary->sec_count_u32 << ",\n";
            jout << "  \"throughput_matrix\": [\n";
            for (size_t i = 0u; i < tp_results.size(); ++i) {
                const auto& r = tp_results[i];
                jout << "    {\"batch\": " << r.batch
                     << ", \"k_pass\": " << r.k_pass
                     << ", \"lm_head\": \"" << (r.use_4p5 ? "4.5bpw_shadow" : "bf16_tied")
                     << "\", \"median_ms\": " << r.median_ms
                     << ", \"p95_ms\": " << r.p95_ms
                     << ", \"tok_per_s\": " << r.tok_per_s << "}";
                if (i + 1u < tp_results.size()) jout << ",";
                jout << "\n";
            }
            jout << "  ]\n";
            jout << "}\n";
            jout.close();
        }

        SG_CUDA_OK(cudaEventDestroy(ev_dma_done));
        SG_CUDA_OK(cudaEventDestroy(ev_fork));
        SG_CUDA_OK(cudaStreamDestroy(stream_dma));
        SG_CUDA_OK(cudaStreamDestroy(stream_compute));
        if (ple_pinned_ok) SG_CUDA_OK(cudaFreeHost(h_ple_table));
        else std::free(h_ple_table);
        SG_CUDA_OK(cudaFreeHost(host_canary));
        SG_CUDA_OK(cudaFreeHost(h_ple_pin));
        SG_CUDA_OK(cudaFree(d_arena));
    }
    return 0;
}

static int run_gpu_selftest() {
    int dev_count = 0;
    SG_CUDA_OK(cudaGetDeviceCount(&dev_count));
    SG_REQUIRE(dev_count > 0, "No CUDA device available");
    SG_CUDA_OK(cudaSetDevice(0));

    // Minimal 64 MiB P7-sized scratch + odd-offset norms check under compute-sanitizer
    const size_t scratch_b = 32ULL * 1024ULL * 1024ULL;
    uint8_t* d_mem = nullptr;
    SG_CUDA_OK(cudaMalloc(&d_mem, scratch_b));
    SG_CUDA_OK(cudaMemset(d_mem, 0, scratch_b));
    siligemma::HoloSphereRazorCanary64* host_canary = nullptr;
    siligemma::HoloSphereRazorCanary64* dev_canary = nullptr;
    SG_CUDA_OK(cudaHostAlloc(&host_canary, sizeof(siligemma::HoloSphereRazorCanary64), cudaHostAllocMapped));
    SG_CUDA_OK(cudaHostGetDevicePointer(&dev_canary, host_canary, 0));

    cudaStream_t stream_compute = nullptr;
    SG_CUDA_OK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));

    float* d_in = reinterpret_cast<float*>(d_mem);                               // [16, 12288]
    float* d_out = d_in + 16ULL * 12288ULL;                                      // [16, 12288]
    float* d_norms = d_out + 16ULL * 12288ULL + 1ULL;                            // INTENTIONAL +4B unaligned!
    int32_t* d_q8 = reinterpret_cast<int32_t*>(d_mem + 8ULL * 1024ULL * 1024ULL);
    float* d_sc = reinterpret_cast<float*>(d_mem + 10ULL * 1024ULL * 1024ULL);
    float4* d_a = reinterpret_cast<float4*>(d_mem + 12ULL * 1024ULL * 1024ULL);
    float4* d_b = reinterpret_cast<float4*>(d_mem + 14ULL * 1024ULL * 1024ULL);
    float* d_minv = reinterpret_cast<float*>(d_mem + 16ULL * 1024ULL * 1024ULL);
    uint64_t* d_w = reinterpret_cast<uint64_t*>(d_mem + 18ULL * 1024ULL * 1024ULL);
    __half* d_wsc = reinterpret_cast<__half*>(d_mem + 20ULL * 1024ULL * 1024ULL);

    SplitMix64 rng(184999ULL);

    // 1. Check 0-ULP tree_reduce_128_sm86 across {256, 512, 1536, 6144, 12288} x {1, 16}
    uint32_t max_ulp = 0u;
    {
        const int dims[5] = {256, 512, 1536, 6144, 12288};
        const int batches[2] = {1, 16};
        std::vector<float> hv(16ULL * 12288ULL), hout(16ULL);
        for (int d : dims) {
            for (int b : batches) {
                for (int bi = 0; bi < b; ++bi) {
                    for (int j = 0; j < d; ++j) hv[bi * d + j] = rng.next_sym_f32() * 1.3f;
                }
                SG_CUDA_OK(cudaMemcpyAsync(d_in, hv.data(), static_cast<size_t>(b) * d * sizeof(float), cudaMemcpyHostToDevice, stream_compute));
                siligemma_sm86::siligemma_tree_reduce_probe_sm86<<<b, 128, 0, stream_compute>>>(d_in, d_out, d);
                SG_CUDA_OK(cudaMemcpyAsync(hout.data(), d_out, static_cast<size_t>(b) * sizeof(float), cudaMemcpyDeviceToHost, stream_compute));
                SG_CUDA_OK(cudaStreamSynchronize(stream_compute));
                for (int bi = 0; bi < b; ++bi) {
                    const float cpu_s = siligemma::canonical_tree_sum_block128_f32(&hv[bi * d], static_cast<size_t>(d));
                    const uint32_t ulp = ulp_diff_f32(cpu_s, hout[bi]);
                    if (ulp > max_ulp) {
                        max_ulp = ulp;
                        std::fprintf(stderr, "[SELFTEST] tree mismatch d=%d b=%d bi=%d cpu=%.9g (0x%08x) gpu=%.9g (0x%08x) ulp=%u\n",
                                     d, b, bi, cpu_s, siligemma::float_bits_u32(cpu_s), hout[bi], siligemma::float_bits_u32(hout[bi]), ulp);
                    }
                }
            }
        }
    }

    // 2. Check unaligned +4B rms_gamma on sandwich_norm_u4_q8 & fused_sandwich_h4_dp4a_gemv
    {
        std::vector<float> hx(1536, 0.5f), hgam(1536, 1.0f);
        std::vector<uint64_t> hw(64 * 96);
        std::vector<uint16_t> hsc(64 * 48, f32_to_f16_bits(0.05f));
        for (auto& v : hw) v = pack_a16_codeword_rng(0x123u, rng);
        SG_CUDA_OK(cudaMemcpyAsync(d_in, hx.data(), 1536 * sizeof(float), cudaMemcpyHostToDevice, stream_compute));
        SG_CUDA_OK(cudaMemcpyAsync(d_norms, hgam.data(), 1536 * sizeof(float), cudaMemcpyHostToDevice, stream_compute));
        SG_CUDA_OK(cudaMemcpyAsync(d_w, hw.data(), hw.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream_compute));
        SG_CUDA_OK(cudaMemcpyAsync(d_wsc, hsc.data(), hsc.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream_compute));
        siligemma_sm86::siligemma_sandwich_norm_u4_q8_sm86<<<1, 128, 0, stream_compute>>>(
            d_in, d_norms, nullptr, d_q8, d_sc, nullptr, 1e-6f, 1536
        );
        std::memset(host_canary, 0, sizeof(*host_canary));
        siligemma_sm86::siligemma_fused_sandwich_h4_dp4a_gemv_sm86<<<dim3(64, 1), 128, 0, stream_compute>>>(
            d_in, d_norms, d_w, d_wsc, d_out, dev_canary, 1e-6f, 64, 1536, 32
        );
        SG_CUDA_OK(cudaStreamSynchronize(stream_compute));
    }

    // 3. Check Woodbury 10,000-vec norm drift & 16x AB/BA overhead
    double max_wb_drift = 0.0, wb_oh_pct = 0.0, euc_med_ms = 0.0, wb_med_ms = 0.0;
    {
        const int D = 1536, df4 = D * 4;
        std::vector<float> hA(df4), hB(df4), hXbatch(100ULL * D), hYbatch(100ULL * D);
        for (int i = 0; i < df4; ++i) { hA[i] = 0.06f * rng.next_sym_f32(); hB[i] = 0.06f * rng.next_sym_f32(); }
        SG_CUDA_OK(cudaMemcpyAsync(d_a, hA.data(), df4 * sizeof(float), cudaMemcpyHostToDevice, stream_compute));
        SG_CUDA_OK(cudaMemcpyAsync(d_b, hB.data(), df4 * sizeof(float), cudaMemcpyHostToDevice, stream_compute));
        siligemma_sm86::siligemma_precompute_woodbury_inv8_sm86<<<1, 256, 0, stream_compute>>>(d_a, d_b, d_minv, 0.25f, D);
        SG_CUDA_OK(cudaMemcpyToSymbolAsync(siligemma_sm86::c_wb_inv_256b, d_minv, 256, 0, cudaMemcpyDeviceToDevice, stream_compute));
        for (size_t idx = 0u; idx < hXbatch.size(); ++idx) hXbatch[idx] = 1.2f * rng.next_sym_f32();
        SG_CUDA_OK(cudaMemcpyAsync(d_in, hXbatch.data(), hXbatch.size() * sizeof(float), cudaMemcpyHostToDevice, stream_compute));
        for (int vi = 0; vi < 100; ++vi) {
            siligemma_sm86::siligemma_cayley_woodbury_r4_hoisted_sm86<<<1, 256, 0, stream_compute>>>(
                d_in + vi * D, d_a, d_b, d_out + vi * D, 0.25f, D
            );
        }
        SG_CUDA_OK(cudaMemcpyAsync(hYbatch.data(), d_out, hYbatch.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_compute));
        SG_CUDA_OK(cudaStreamSynchronize(stream_compute));
        for (int vi = 0; vi < 100; ++vi) {
            double s0 = 0.0, s1 = 0.0;
            for (int j = 0; j < D; ++j) {
                const double a = hXbatch[vi * D + j], b = hYbatch[vi * D + j];
                s0 += a * a; s1 += b * b;
            }
            const double rel = std::fabs(std::sqrt(s1) - std::sqrt(s0)) / std::max(std::sqrt(s0), 1e-12);
            if (rel > max_wb_drift) max_wb_drift = rel;
        }
        CapturedGraph ge{}, gw{};
        SG_CUDA_OK(cudaStreamBeginCapture(stream_compute, cudaStreamCaptureModeThreadLocal));
        for (int c = 0; c < 16; ++c) {
            siligemma_sm86::siligemma_euclidean_lora_r4_sm86<<<1, 128, 0, stream_compute>>>(
                d_in + c * D, d_a, d_b, d_out + c * D, 0.25f, D
            );
        }
        SG_CUDA_OK(cudaStreamEndCapture(stream_compute, &ge.graph));
        SG_CUDA_OK(cudaGraphInstantiate(&ge.exec, ge.graph, nullptr, nullptr, 0));

        SG_CUDA_OK(cudaStreamBeginCapture(stream_compute, cudaStreamCaptureModeThreadLocal));
        for (int c = 0; c < 16; ++c) {
            siligemma_sm86::siligemma_cayley_woodbury_r4_hoisted_sm86<<<1, 256, 0, stream_compute>>>(
                d_in + c * D, d_a, d_b, d_out + c * D, 0.25f, D
            );
        }
        SG_CUDA_OK(cudaStreamEndCapture(stream_compute, &gw.graph));
        SG_CUDA_OK(cudaGraphInstantiate(&gw.exec, gw.graph, nullptr, nullptr, 0));

        std::vector<double> ta, tb;
        benchmark_abba_graphs(ge, gw, stream_compute, 60, 10, ta, tb);
        euc_med_ms = median_of_vec(ta);
        wb_med_ms = median_of_vec(tb);
        wb_oh_pct = (wb_med_ms - euc_med_ms) / std::max(euc_med_ms, 1e-9) * 100.0;
        ge.destroy();
        gw.destroy();
    }

    std::printf("SELFTEST_GPU_OK max_ulp=%u wb_norm_drift=%.3e euc_16x_ms=%.5f wb_16x_ms=%.5f wb_oh_pct=%.2f%%\n",
                max_ulp, max_wb_drift, euc_med_ms, wb_med_ms, wb_oh_pct);
    SG_CUDA_OK(cudaStreamDestroy(stream_compute));
    SG_CUDA_OK(cudaFreeHost(host_canary));
    SG_CUDA_OK(cudaFree(d_mem));
    return (max_ulp == 0u && max_wb_drift <= 1e-6 && wb_oh_pct <= 25.0) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    const BenchCliArgs args = parse_cli_args(argc, argv);
    if (args.cpu_selftest) {
        return run_cpu_contract_selftest();
    }
    if (args.selftest_gpu) {
        return run_gpu_selftest();
    }
    if (args.run_live_gpu) {
        return run_live_gpu_bench(args);
    }
    std::fprintf(stderr, "Usage: %s [--cpu-selftest | --selftest-gpu | --run-live-gpu ...]\n", argv[0]);
    return 2;
}
