// Bare-metal sm_86 Inference & Benchmark Harness for SiliGemma 1.0 (Gemma-4-E2B Form B Stage-2).
// Strictly bare-metal CUDA with zero host-framework or vendor-BLAS dependencies and zero dynamic decode allocations.

#include "siligemma_arena.hpp"
#include "siligemma_kernels_sm86.cu"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                                     \
    do {                                                                                     \
        cudaError_t err__ = (expr);                                                          \
        if (err__ != cudaSuccess) {                                                          \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,            \
                         cudaGetErrorString(err__));                                         \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while (0)

namespace {

using namespace s184b;

bool parse_int_strict(const std::string& s, int min_v, int max_v, int& out) {
    if (s.empty()) return false;
    errno = 0;
    char* endptr = nullptr;
    const long v = std::strtol(s.c_str(), &endptr, 10);
    if (errno == ERANGE || endptr == s.c_str() || *endptr != '\0' || v < min_v || v > max_v) {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

std::vector<int> parse_csv_ints(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) {
            int val = 0;
            if (!parse_int_strict(tok, 0, VOCAB_SIZE - 1, val)) {
                std::fprintf(stderr, "Invalid token ID in --prompt-ids: %s\n", tok.c_str());
                std::exit(2);
            }
            out.push_back(val);
        }
    }
    return out;
}

uint32_t float_to_bits(float v) {
    uint32_t u = 0;
    std::memcpy(&u, &v, 4);
    return u;
}

int run_cpu_selftest() {
    // 1. Canonical 0-ULP tree_reduce_128 CPU sanity
    float arr[128];
    for (int i = 0; i < 128; ++i) arr[i] = std::sin(0.17f * float(i + 1)) * 3.5f;
    float r1 = canonical_tree_reduce_128_cpu(arr);
    float r2 = canonical_tree_reduce_128_cpu(arr);
    if (float_to_bits(r1) != float_to_bits(r2) || !std::isfinite(r1)) {
        std::fprintf(stderr, "FAIL: CPU tree_reduce determinism\n");
        return 1;
    }

    // 2. Hamming(16,11) SECDED encoding + 3 SEC + 3 DED verification
    uint64_t seeds[3] = {0x123456789ABCDEF0ULL, 0x76543210FEDCBA98ULL, 0x0F1E2D3C4B5A6978ULL};
    for (int i = 0; i < 3; ++i) {
        uint64_t cw = encode_secded_w64_cpu(seeds[i]);
        if (secded_syn_cpu(cw) != 0u) {
            std::fprintf(stderr, "FAIL: SECDED encode syndrome != 0\n");
            return 1;
        }
        // Single-bit flip in nibble LSB (bit 4*k)
        uint64_t sec_cw = cw ^ (1ULL << (4 * (i + 3)));
        uint32_t syn1 = secded_syn_cpu(sec_cw);
        if (syn1 == 0u || (syn1 & 1u) == 0u) {
            std::fprintf(stderr, "FAIL: SEC flip undetected\n");
            return 1;
        }
        // Double-bit flip in two distinct nibble LSBs
        uint64_t ded_cw = cw ^ (1ULL << (4 * (i + 1))) ^ (1ULL << (4 * (i + 9)));
        uint32_t syn2 = secded_syn_cpu(ded_cw);
        if (syn2 == 0u || (syn2 & 1u) != 0u) {
            std::fprintf(stderr, "FAIL: DED flip misclassified syn=%u\n", syn2);
            return 1;
        }
    }

    // 3. 8x8 Woodbury-Cayley inverse exactness
    float M[64], Minv[64];
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            M[r * 8 + c] = (r == c ? 1.0f : 0.0f) + 0.08f * std::sin(float(r * 7 + c * 3 + 1));
        }
    }
    if (!invert_8x8_cpu(M, Minv)) {
        std::fprintf(stderr, "FAIL: Woodbury CPU inv8 singular\n");
        return 1;
    }
    double max_id_err = 0.0;
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            double s = 0.0;
            for (int k = 0; k < 8; ++k) s += double(M[r * 8 + k]) * double(Minv[k * 8 + c]);
            max_id_err = std::max(max_id_err, std::fabs(s - (r == c ? 1.0 : 0.0)));
        }
    }
    if (max_id_err > 1e-6) {
        std::fprintf(stderr, "FAIL: Woodbury CPU max_id_err=%.3e\n", max_id_err);
        return 1;
    }

    // 4. Unified v1 (.slgb archive) + Riccati + Canary R5 contract self-tests
    siligemma::SlgbHeader v1_hdr{};
    std::array<siligemma::SiliGemmaLayerDesc, siligemma::GEMMA4_NUM_LAYERS> v1_layers{};
    std::vector<siligemma::SiliGemmaLinearDesc> v1_linears;
    if (siligemma::build_gemma4_descriptor_table(32u, v1_hdr, v1_layers, v1_linears, 3u) != siligemma::ValidationError::OK) {
        std::fprintf(stderr, "FAIL: build_gemma4_descriptor_table(32)\n");
        return 1;
    }
    const uint32_t valid_toks[4] = {2u, 100u, 200u, 300u};
    if (siligemma::validate_runtime_contract(v1_hdr, v1_layers, v1_linears, valid_toks, 4u, true) != siligemma::ValidationError::OK) {
        std::fprintf(stderr, "FAIL: v1 runtime contract rejected valid input\n");
        return 1;
    }
    const uint32_t bad_bos[2] = {99u, 100u};
    if (siligemma::validate_runtime_contract(v1_hdr, v1_layers, v1_linears, bad_bos, 2u, true) != siligemma::ValidationError::ERR_BOS_NOT_2) {
        std::fprintf(stderr, "FAIL: R5 bos!=2 negative control failed\n");
        return 1;
    }
    auto bad_v1_ly = v1_layers;
    bad_v1_ly[15].linear_idx[1] = 0;
    if (siligemma::validate_runtime_contract(v1_hdr, bad_v1_ly, v1_linears, valid_toks, 4u, true) != siligemma::ValidationError::ERR_DEAD_KV_PRESENT) {
        std::fprintf(stderr, "FAIL: R5 dead KV negative control failed\n");
        return 1;
    }
    const uint64_t exp_v1_sz = siligemma::expected_slgb_file_bytes(v1_hdr);
    if (siligemma::validate_slgb_file_layout(v1_hdr, v1_layers, v1_linears, exp_v1_sz, true) != siligemma::ValidationError::OK) {
        std::fprintf(stderr, "FAIL: v1 file layout contract\n");
        return 1;
    }

    std::printf("CPU_SELFTEST_PASS tree_bits=0x%08x wb_err=%.3e arena128k_mib=%.2f v1_slgb_bytes=%llu\n",
                float_to_bits(r1), max_id_err, double(kArena128K) / (1024.0 * 1024.0),
                static_cast<unsigned long long>(exp_v1_sz));
    return 0;
}

int run_gpu_selftest() {
    // 1. 0-ULP tree_reduce_128_sm86 vs canonical_tree_reduce_128_cpu across 16 patterns
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    uint32_t max_ulp = 0u;
    for (int pat = 0; pat < 16; ++pat) {
        std::vector<float> hv(128);
        uint64_t rng = 987654321ULL + uint64_t(pat) * 1337ULL;
        for (int i = 0; i < 128; ++i) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            double u = (double(rng >> 11) / double(1ULL << 53)) * 2.0 - 1.0;
            hv[i] = float(u * (1.0 + (pat % 5) * 10.0));
        }
        float cpu_r = canonical_tree_reduce_128_cpu(hv.data());
        CUDA_CHECK(cudaMemcpy(d_in, hv.data(), 128 * sizeof(float), cudaMemcpyHostToDevice));
        tree_reduce_probe_sm86<<<1, 128>>>(d_in, d_out);
        float gpu_r = 0.0f;
        CUDA_CHECK(cudaMemcpy(&gpu_r, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        uint32_t bc = float_to_bits(cpu_r), bg = float_to_bits(gpu_r);
        uint32_t ulp = (bc > bg) ? (bc - bg) : (bg - bc);
        max_ulp = std::max(max_ulp, ulp);
    }
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    if (max_ulp != 0u) {
        std::fprintf(stderr, "FAIL: GPU tree_reduce max_ulp=%u\n", max_ulp);
        return 1;
    }

    // 2. Woodbury 8x8 Cayley inverse + hoisted constant apply
    float hM[64], hMinv_cpu[64], hMinv_gpu[64];
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
            hM[r * 8 + c] = (r == c ? 1.0f : 0.0f) + 0.07f * std::cos(float(r * 5 + c * 11));
    invert_8x8_cpu(hM, hMinv_cpu);
    float *dM = nullptr, *dInv = nullptr;
    CUDA_CHECK(cudaMalloc(&dM, 64 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dInv, 64 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dM, hM, 64 * sizeof(float), cudaMemcpyHostToDevice));
    woodbury_inv8_sm86<<<1, 128>>>(dM, dInv);
    CUDA_CHECK(cudaMemcpy(hMinv_gpu, dInv, 64 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyToSymbol(c_wb_inv_256b, hMinv_gpu, 64 * sizeof(float)));
    double wb_err = 0.0;
    for (int i = 0; i < 64; ++i) wb_err = std::max(wb_err, std::fabs(double(hMinv_cpu[i]) - double(hMinv_gpu[i])));
    CUDA_CHECK(cudaFree(dM));
    CUDA_CHECK(cudaFree(dInv));

    // Check Cayley norm preservation on 1536-dim vector with skew U8, V8 (U = -V -> exact orthogonal Cayley I - 2 U (I + U^T U)^{-1} U^T -> norm drift == 0!)
    float *d_h = nullptr, *d_U8 = nullptr, *d_V8 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_h, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_U8, 8 * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V8, 8 * HIDDEN_SIZE * sizeof(float)));
    std::vector<float> hh(HIDDEN_SIZE), hU(8 * HIDDEN_SIZE), hV(8 * HIDDEN_SIZE);
    for (int d = 0; d < HIDDEN_SIZE; ++d) hh[d] = std::sin(0.031f * float(d + 1));
    // Skew-symmetric rank-8 pair: A = Q K^T - K Q^T? Simple skew-adjoint: choose orthonormal pairs U_{2k}, U_{2k+1} and set V_{2k} = U_{2k+1}, V_{2k+1} = -U_{2k} -> U V^T is skew-symmetric so Cayley I - 2 S (I+S)^{-1} preserves norm to float precision!
    for (int r = 0; r < 8; ++r)
        for (int d = 0; d < HIDDEN_SIZE; ++d)
            hU[r * HIDDEN_SIZE + d] = (d % 8 == r) ? 0.1f : 0.0f;
    for (int k = 0; k < 4; ++k) {
        for (int d = 0; d < HIDDEN_SIZE; ++d) {
            hV[(2 * k) * HIDDEN_SIZE + d] = hU[(2 * k + 1) * HIDDEN_SIZE + d];
            hV[(2 * k + 1) * HIDDEN_SIZE + d] = -hU[(2 * k) * HIDDEN_SIZE + d];
        }
    }
    // Cayley inverse matrix M = I + 0.5 * V^T U (skew-symmetric 0.5 S -> exact orthogonal I - V? Wait: (I + 0.5 U V^T)^{-1} (I - 0.5 U V^T) = I - U (I + 0.5 V^T U)^{-1} V^T!)
    float Ms[64] = {}, Minv_s[64] = {};
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            double s = (r == c) ? 1.0 : 0.0;
            for (int d = 0; d < HIDDEN_SIZE; ++d) s += 0.5 * double(hV[r * HIDDEN_SIZE + d]) * double(hU[c * HIDDEN_SIZE + d]);
            Ms[r * 8 + c] = float(s);
        }
    }
    invert_8x8_cpu(Ms, Minv_s);
    CUDA_CHECK(cudaMemcpyToSymbol(c_wb_inv_256b, Minv_s, 64 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_h, hh.data(), HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_U8, hU.data(), 8 * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V8, hV.data(), 8 * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cayley_woodbury_apply_sm86<<<1, 128>>>(d_h, d_U8, d_V8, -1.0f, 1, HIDDEN_SIZE);
    std::vector<float> hh_after(HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(hh_after.data(), d_h, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    double n0 = 0.0, n1 = 0.0;
    for (int d = 0; d < HIDDEN_SIZE; ++d) { n0 += double(hh[d]) * hh[d]; n1 += double(hh_after[d]) * hh_after[d]; }
    double norm_drift = std::fabs(std::sqrt(n1) - std::sqrt(n0)) / std::sqrt(n0);
    CUDA_CHECK(cudaFree(d_h));
    CUDA_CHECK(cudaFree(d_U8));
    CUDA_CHECK(cudaFree(d_V8));

    // 3. SECDED 3/3 SEC, 3/3 DED & IRIS 1/1 Trojan Canary Injection on live GPU GEMV
    const int M = 32, K = 128;
    std::vector<uint4> hw(M * (K / 32));
    std::vector<__half> hsc(M * (K / 32), __float2half(0.05f));
    for (size_t i = 0; i < hw.size(); ++i) {
        uint64_t c0 = encode_secded_w64_cpu(0x1122334455667788ULL ^ (i * 0x9E37ULL));
        uint64_t c1 = encode_secded_w64_cpu(0x8877665544332211ULL ^ (i * 0x517CULL));
        hw[i] = make_uint4(uint32_t(c0), uint32_t(c0 >> 32), uint32_t(c1), uint32_t(c1 >> 32));
    }
    int4 *d_hm = nullptr; int2 *d_lo = nullptr; float *d_xsc = nullptr, *d_y = nullptr; int32_t *d_xsq = nullptr;
    uint4 *d_w = nullptr; __half *d_sc = nullptr; HoloSphereRazorCanary64 *d_can = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hm, (K / 8) * sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&d_lo, (K / 8) * sizeof(int2)));
    CUDA_CHECK(cudaMalloc(&d_xsc, (K / 32) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_xsq, (K / 32) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, hw.size() * sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&d_sc, hsc.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_can, sizeof(HoloSphereRazorCanary64)));
    CUDA_CHECK(cudaMemset(d_hm, 0, (K / 8) * sizeof(int4)));
    CUDA_CHECK(cudaMemset(d_lo, 0, (K / 8) * sizeof(int2)));
    CUDA_CHECK(cudaMemset(d_xsc, 0, (K / 32) * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_xsq, 0, (K / 32) * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_sc, hsc.data(), hsc.size() * sizeof(__half), cudaMemcpyHostToDevice));

    // Clean check -> 0 SEC, 0 DED
    CUDA_CHECK(cudaMemcpy(d_w, hw.data(), hw.size() * sizeof(uint4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_can, 0, sizeof(HoloSphereRazorCanary64)));
    gemv_b1_dp4a22_sm86<4, 1><<<M / 32, 128>>>(d_hm, d_lo, d_xsc, d_xsq, d_w, d_sc, d_y, d_can, 0.0f, M, K);
    iris_canary_finalize_sm86<<<1, 128>>>(d_y, d_can, 1, M);
    HoloSphereRazorCanary64 hcan{};
    CUDA_CHECK(cudaMemcpy(&hcan, d_can, sizeof(hcan), cudaMemcpyDeviceToHost));
    if (hcan.sec_cnt != 0u || hcan.ded_cnt != 0u || hcan.poisoned != 0u) {
        std::fprintf(stderr, "FAIL: clean SECDED false positive\n");
        return 1;
    }

    int sec_pass = 0, ded_pass = 0;
    for (int t = 0; t < 3; ++t) {
        std::vector<uint4> mut = hw;
        mut[t * 3 + 1].x ^= (1u << (4 * (t + 2))); // 1-bit flip
        CUDA_CHECK(cudaMemcpy(d_w, mut.data(), mut.size() * sizeof(uint4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_can, 0, sizeof(HoloSphereRazorCanary64)));
        gemv_b1_dp4a22_sm86<4, 1><<<M / 32, 128>>>(d_hm, d_lo, d_xsc, d_xsq, d_w, d_sc, d_y, d_can, 0.0f, M, K);
        iris_canary_finalize_sm86<<<1, 128>>>(d_y, d_can, 1, M);
        CUDA_CHECK(cudaMemcpy(&hcan, d_can, sizeof(hcan), cudaMemcpyDeviceToHost));
        if (hcan.sec_cnt >= 1u && hcan.poisoned == 1u) sec_pass++;
    }
    for (int t = 0; t < 3; ++t) {
        std::vector<uint4> mut = hw;
        mut[t * 4 + 2].x ^= (1u << 4);
        mut[t * 4 + 2].y ^= (1u << 12); // 2-bit flip across distinct nibble positions
        CUDA_CHECK(cudaMemcpy(d_w, mut.data(), mut.size() * sizeof(uint4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_can, 0, sizeof(HoloSphereRazorCanary64)));
        gemv_b1_dp4a22_sm86<4, 1><<<M / 32, 128>>>(d_hm, d_lo, d_xsc, d_xsq, d_w, d_sc, d_y, d_can, 0.0f, M, K);
        iris_canary_finalize_sm86<<<1, 128>>>(d_y, d_can, 1, M);
        CUDA_CHECK(cudaMemcpy(&hcan, d_can, sizeof(hcan), cudaMemcpyDeviceToHost));
        if (hcan.ded_cnt >= 1u && hcan.poisoned == 1u) ded_pass++;
    }
    // 1/1 IRIS Trojan injection check
    HoloSphereRazorCanary64 trojan{};
    trojan.iris_state = 0xDEADBEEFCAFEBABEULL;
    trojan.poisoned = 1u;
    CUDA_CHECK(cudaMemcpy(d_can, &trojan, sizeof(trojan), cudaMemcpyHostToDevice));
    iris_canary_finalize_sm86<<<1, 128>>>(d_y, d_can, 1, M);
    CUDA_CHECK(cudaMemcpy(&hcan, d_can, sizeof(hcan), cudaMemcpyDeviceToHost));
    int iris_trojan_pass = (hcan.poisoned == 1u && hcan.checksum == (hcan.iris_state ^ 1ULL)) ? 1 : 0;

    CUDA_CHECK(cudaFree(d_hm)); CUDA_CHECK(cudaFree(d_lo)); CUDA_CHECK(cudaFree(d_xsc)); CUDA_CHECK(cudaFree(d_xsq));
    CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_sc)); CUDA_CHECK(cudaFree(d_can));

    // 4. Live cudaMemcpy D2D 512 MiB physical DRAM stream bandwidth benchmark
    const size_t bw_bytes = 512ULL * 1024ULL * 1024ULL;
    void *d_src = nullptr, *d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, bw_bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bw_bytes));
    CUDA_CHECK(cudaMemset(d_src, 0x5A, bw_bytes));
    for (int w = 0; w < 3; ++w) CUDA_CHECK(cudaMemcpy(d_dst, d_src, bw_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    std::vector<double> bws;
    for (int r = 0; r < 11; ++r) {
        CUDA_CHECK(cudaEventRecord(e0));
        CUDA_CHECK(cudaMemcpy(d_dst, d_src, bw_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        bws.push_back((2.0 * double(bw_bytes)) / (double(ms) * 1.0e6)); // bidirectional read+write D2D GB/s!
    }
    std::sort(bws.begin(), bws.end());
    double stream_gbps = bws[bws.size() / 2];
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));

    if (sec_pass != 3 || ded_pass != 3 || iris_trojan_pass != 1 || wb_err > 1e-6 || norm_drift > 1e-6) {
        std::fprintf(stderr, "FAIL: gpu_selftest sec=%d ded=%d iris=%d wb=%.3e norm_drift=%.3e\n",
                     sec_pass, ded_pass, iris_trojan_pass, wb_err, norm_drift);
        return 1;
    }

    std::printf("GPU_SELFTEST_PASS max_ulp=%u wb_err=%.3e norm_drift=%.3e sec=%d/3 ded=%d/3 iris=%d/1 d2d_bw_gbps=%.2f\n",
                max_ulp, wb_err, norm_drift, sec_pass, ded_pass, iris_trojan_pass, stream_gbps);
    return 0;
}

struct SlgbModel {
    SlgbHeader hdr{};
    std::vector<SiliGemmaLinearDesc> linears;
    std::vector<SiliGemmaLayerDesc> layers;
    std::vector<float> inv_freq_slide;
    std::vector<float> inv_freq_full;
    std::vector<float> norms_f32;
    size_t ple_norm_off = 0;
    size_t final_norm_off = 0;

    // Device persistent pointers
    uint4* d_w_u4 = nullptr;
    __half* d_w_sc = nullptr;
    float* d_norms = nullptr;
    float* d_inv_slide = nullptr;
    float* d_inv_full = nullptr;
    uint16_t* d_lm_bf16 = nullptr;
    float* d_u4_U16[9]{};
    float* d_u4_V16[9]{};
    float u4_alpha[9]{};
    float *d_wb_U8 = nullptr, *d_wb_V8 = nullptr;

    // Static Arena P5 (KV) & P7 (Activation Scratch)
    uint8_t* d_p5 = nullptr;
    size_t p5_bytes = 0;
    uint8_t* d_p7 = nullptr;
    float* d_k_slots[NUM_KV_LAYERS]{};
    float* d_v_slots[NUM_KV_LAYERS]{};
    int slot_t_cap[NUM_KV_LAYERS]{};
    int t_cap = 0;
    int max_b = 0;
    int kv_is_bf16 = 0;

    // Sub-slices inside static P7 (20,217,856 B -> 0 B dynamic alloc!)
    float* d_logits = nullptr;      // [0 .. 16,777,216) (shared scratch during layers -> logits at end!)
    float* d_ypart = nullptr;       // overlay in logits [0 .. 6.3 MB)
    float* d_tmp2 = nullptr;        // overlay in logits [6.3 MB .. 7.1 MB)
    float* d_tmp3 = nullptr;        // overlay in logits [7.1 MB .. 7.9 MB)
    float* d_scr = nullptr;         // overlay in logits [7.9 MB .. 8.7 MB)
    __half* d_xhi_T = nullptr;      // overlay in logits [8.7 MB .. 9.1 MB)
    __half* d_xlo_T = nullptr;      // overlay in logits [9.1 MB .. 9.5 MB)
    uint16_t* d_xbhi_T = nullptr;   // overlay in logits [9.5 MB .. 9.9 MB)
    uint16_t* d_xblo_T = nullptr;   // overlay in logits [9.9 MB .. 10.3 MB)
    int4* d_xhm = nullptr;          // overlay in logits [10.3 MB .. 11.1 MB)
    int2* d_xlo = nullptr;          // overlay in logits [11.1 MB .. 11.5 MB)
    float* d_xsc = nullptr;         // overlay in logits [11.5 MB .. 11.6 MB)
    int32_t* d_xsq = nullptr;       // overlay in logits [11.6 MB .. 11.7 MB)
    int4* d_xhm2 = nullptr;         // overlay in logits [11.7 MB .. 12.5 MB)
    int2* d_xlo2 = nullptr;         // overlay in logits [12.5 MB .. 12.9 MB)
    float* d_xsc2 = nullptr;        // overlay in logits [12.9 MB .. 13.0 MB)
    int32_t* d_xsq2 = nullptr;      // overlay in logits [13.0 MB .. 13.1 MB)
    float* d_op = nullptr;          // overlay in logits [13.1 MB .. 14.7 MB)
    float* d_m = nullptr;           // overlay in logits [14.7 MB .. 14.71 MB)
    float* d_l = nullptr;           // overlay in logits [14.71 MB .. 14.72 MB)

    // Persistent P7 tail [16,777,216 .. 20,217,856) (never overwritten by logits!):
    float* d_hid = nullptr;         // 98,304 B
    float* d_h_pre24 = nullptr;     // 98,304 B (K=2 U-4 restart state)
    float* d_p_pre24 = nullptr;     // 98,304 B (K=2 L24 post_ple state)
    float* d_tmp1 = nullptr;        // 16 * 12288 * 4 = 786,432 B
    float* d_ple_in = nullptr;      // 16 * 35 * 256 * 4 = 573,440 B
    float* d_tok_ple = nullptr;     // 16 * 35 * 256 * 4 = 573,440 B
    int32_t* d_pos = nullptr;       // 64 B
    int32_t* d_tok_ids = nullptr;   // 64 B
    float* d_bv = nullptr;          // 512 B
    int32_t* d_bi = nullptr;        // 512 B
    int32_t* d_out_id = nullptr;    // 64 B
    float* d_inv_pre = nullptr;     // 64 B
    float* d_u4_z = nullptr;        // 1,024 B
    HoloSphereRazorCanary64* d_can = nullptr;

    void bind_p7_slices() {
        auto ptr = [&](size_t off) -> uint8_t* { return d_p7 + off; };
        d_logits = reinterpret_cast<float*>(ptr(0));
        d_ypart  = reinterpret_cast<float*>(ptr(0));
        d_tmp2   = reinterpret_cast<float*>(ptr(6400000));
        d_tmp3   = reinterpret_cast<float*>(ptr(7200000));
        d_scr    = reinterpret_cast<float*>(ptr(8000000));
        d_xhm    = reinterpret_cast<int4*>(ptr(10400000));
        d_xlo    = reinterpret_cast<int2*>(ptr(11200000));
        d_xsc    = reinterpret_cast<float*>(ptr(11600000));
        d_xsq    = reinterpret_cast<int32_t*>(ptr(11700000));
        d_xhm2   = reinterpret_cast<int4*>(ptr(11800000));
        d_xlo2   = reinterpret_cast<int2*>(ptr(12600000));
        d_xsc2   = reinterpret_cast<float*>(ptr(13000000));
        d_xsq2   = reinterpret_cast<int32_t*>(ptr(13100000));
        d_op     = reinterpret_cast<float*>(ptr(13200000));
        d_m      = reinterpret_cast<float*>(ptr(15000000));
        d_l      = reinterpret_cast<float*>(ptr(15100000));

        size_t tail = 16777216ULL;
        auto alloc_tail = [&](size_t bytes) -> uint8_t* {
            bytes = (bytes + 255ULL) & ~255ULL;
            uint8_t* p = ptr(tail);
            tail += bytes;
            if (tail > ARENA_P7_SCRATCH_BYTES) {
                std::fprintf(stderr, "FATAL: P7 tail overflow %zu > %llu\n",
                             tail, (unsigned long long)ARENA_P7_SCRATCH_BYTES);
                std::exit(1);
            }
            return p;
        };
        // Keep d_xhi_T/d_xlo_T/d_xbhi_T/d_xblo_T strictly outside [0..16.77MB) d_logits!
        uint8_t* xbuf_hi = alloc_tail(16ULL * INTERM_WIDE * 2);
        uint8_t* xbuf_lo = alloc_tail(16ULL * INTERM_WIDE * 2);
        d_xhi_T  = reinterpret_cast<__half*>(xbuf_hi);
        d_xlo_T  = reinterpret_cast<__half*>(xbuf_lo);
        d_xbhi_T = reinterpret_cast<uint16_t*>(xbuf_hi);
        d_xblo_T = reinterpret_cast<uint16_t*>(xbuf_lo);

        d_hid     = reinterpret_cast<float*>(alloc_tail(16ULL * HIDDEN_SIZE * 4));
        d_h_pre24 = reinterpret_cast<float*>(alloc_tail(16ULL * HIDDEN_SIZE * 4));
        d_p_pre24 = reinterpret_cast<float*>(alloc_tail(16ULL * HIDDEN_SIZE * 4));
        d_tmp1    = reinterpret_cast<float*>(alloc_tail(16ULL * INTERM_WIDE * 4));
        d_ple_in  = reinterpret_cast<float*>(alloc_tail(16ULL * NUM_LAYERS * PLE_DIM * 4));
        d_tok_ple = reinterpret_cast<float*>(alloc_tail(16ULL * NUM_LAYERS * PLE_DIM * 4));
        d_pos     = reinterpret_cast<int32_t*>(alloc_tail(256));
        d_tok_ids = reinterpret_cast<int32_t*>(alloc_tail(256));
        d_bv      = reinterpret_cast<float*>(alloc_tail(1024));
        d_bi      = reinterpret_cast<int32_t*>(alloc_tail(1024));
        d_out_id  = reinterpret_cast<int32_t*>(alloc_tail(256));
        d_inv_pre = reinterpret_cast<float*>(alloc_tail(256));
        d_u4_z    = reinterpret_cast<float*>(alloc_tail(1024));
        d_can     = reinterpret_cast<HoloSphereRazorCanary64*>(alloc_tail(sizeof(HoloSphereRazorCanary64)));
    }

    bool load_v1_slgb(std::ifstream& f, uint64_t fsz, std::vector<uint4>& h_u4, std::vector<uint16_t>& h_sc, std::vector<uint16_t>& h_lm) {
        siligemma::SlgbHeader v1_hdr{};
        std::array<siligemma::SiliGemmaLayerDesc, siligemma::GEMMA4_NUM_LAYERS> v1_layers{};
        std::vector<siligemma::SiliGemmaLinearDesc> v1_linears(siligemma::LIVE_LINEAR_COUNT);
        f.read(reinterpret_cast<char*>(&v1_hdr), sizeof(v1_hdr));
        f.read(reinterpret_cast<char*>(v1_layers.data()), v1_layers.size() * sizeof(siligemma::SiliGemmaLayerDesc));
        f.read(reinterpret_cast<char*>(v1_linears.data()), v1_linears.size() * sizeof(siligemma::SiliGemmaLinearDesc));
        if (!f.good()) return false;
        if (siligemma::validate_slgb_file_layout(v1_hdr, v1_layers, v1_linears, fsz, true) != siligemma::ValidationError::OK) {
            std::fprintf(stderr, "Invalid v1 .slgb CRC-32 / layout\n");
            return false;
        }
        if (v1_hdr.group_size != 32u) {
            std::fprintf(stderr, "Expected g=32 .slgb, got %u\n", v1_hdr.group_size);
            return false;
        }
        uint64_t off = siligemma::align_up_n(
            sizeof(v1_hdr) + v1_layers.size() * sizeof(siligemma::SiliGemmaLayerDesc) + v1_linears.size() * sizeof(siligemma::SiliGemmaLinearDesc),
            256ULL
        );
        const uint64_t off_p1 = off;
        off = siligemma::align_up_n(off_p1 + v1_hdr.total_wire_bytes, 256ULL);
        const uint64_t off_p2_emb = off;
        off += siligemma::EMBED_TOKENS_BF16_BYTES;
        const bool has_lm = (v1_hdr.flags & 1u) != 0u;
        uint64_t off_p2_lm = 0ULL;
        if (has_lm) {
            off = siligemma::align_up_n(off, 256ULL);
            off_p2_lm = off;
            off += siligemma::LM_HEAD_SHADOW_G32_BYTES;
        }
        const uint64_t off_p3_norms = siligemma::align_up_n(off, 256ULL);

        std::memset(&hdr, 0, sizeof(hdr));
        hdr.magic = SLGB_MAGIC;
        hdr.version = 1u;
        hdr.num_layers = NUM_LAYERS;
        hdr.num_kv_layers = NUM_KV_LAYERS;
        hdr.hidden_size = HIDDEN_SIZE;
        hdr.vocab_size = VOCAB_SIZE;
        hdr.ple_dim = PLE_DIM;
        hdr.num_linears = has_lm ? 277u : 276u;
        hdr.has_lm_head_g32 = has_lm ? 1u : 0u;
        hdr.has_lm_head_bf16 = 1u;
        hdr.has_ple_embed_bf16 = 0u;
        hdr.has_u4_loop = 0u;
        hdr.embed_scale = std::sqrt(float(HIDDEN_SIZE));
        hdr.ple_model_proj_scale = 1.0f / std::sqrt(float(HIDDEN_SIZE));
        hdr.ple_input_scale = 0.7071067811865475f;
        hdr.softcap = v1_hdr.final_logit_softcapping;

        inv_freq_slide.resize(HEAD_DIM_SLIDE / 2);
        for (int i = 0; i < HEAD_DIM_SLIDE / 2; ++i) {
            inv_freq_slide[i] = float(1.0 / std::pow(10000.0, double(2 * i) / double(HEAD_DIM_SLIDE)));
        }
        inv_freq_full.assign(HEAD_DIM_FULL / 2, 0.0f);
        for (int i = 0; i < 64; ++i) {
            inv_freq_full[i] = float(1.0 / std::pow(1000000.0, double(2 * i) / 128.0));
        }

        norms_f32.resize(siligemma::NORM_SCALAR_PARAMS_FP32);
        f.seekg(static_cast<std::streamoff>(off_p3_norms));
        f.read(reinterpret_cast<char*>(norms_f32.data()), static_cast<std::streamsize>(norms_f32.size() * sizeof(float)));
        if (!f.good()) return false;
        final_norm_off = norms_f32.size() - size_t(HIDDEN_SIZE) - size_t(PLE_DIM);
        ple_norm_off = norms_f32.size() - size_t(PLE_DIM);

        layers.resize(NUM_LAYERS);
        for (int l = 0; l < NUM_LAYERS; ++l) {
            const auto& v1L = v1_layers[size_t(l)];
            auto& L = layers[size_t(l)];
            std::memset(&L, 0, sizeof(L));
            L.layer_idx = v1L.layer_idx;
            L.is_full_attn = v1L.is_full_attn;
            L.is_kv_shared = v1L.is_kv_shared;
            L.kv_slot = v1L.kv_source_layer;
            L.head_dim = v1L.head_dim;
            L.interm_size = v1L.intermediate_size;
            L.sliding_window = v1L.is_full_attn ? 0 : SLIDING_WINDOW;
            L.layer_scalar = norms_f32[size_t(v1L.norm_offsets_f32[7])];
            L.off_in_nw = uint64_t(v1L.norm_offsets_f32[0]);
            L.off_pa_nw = uint64_t(v1L.norm_offsets_f32[1]);
            L.off_pf_nw = uint64_t(v1L.norm_offsets_f32[2]);
            L.off_pff_nw = uint64_t(v1L.norm_offsets_f32[3]);
            L.off_ppl_nw = uint64_t(v1L.norm_offsets_f32[4]);
            L.off_qn_w = uint64_t(v1L.norm_offsets_f32[5]);
            L.off_kn_w = (v1L.norm_offsets_f32[6] >= 0) ? uint64_t(v1L.norm_offsets_f32[6]) : UINT64_MAX;
            L.lin_q = uint32_t(v1L.linear_idx[0]);
            L.lin_k = (v1L.linear_idx[1] >= 0) ? uint32_t(v1L.linear_idx[1]) : UINT32_MAX;
            L.lin_v = (v1L.linear_idx[2] >= 0) ? uint32_t(v1L.linear_idx[2]) : UINT32_MAX;
            L.lin_o = uint32_t(v1L.linear_idx[3]);
            L.lin_gate = uint32_t(v1L.linear_idx[4]);
            L.lin_up = uint32_t(v1L.linear_idx[5]);
            L.lin_down = uint32_t(v1L.linear_idx[6]);
            L.lin_ple_gate = uint32_t(v1L.linear_idx[7]);
            L.lin_ple_proj = uint32_t(v1L.linear_idx[8]);
        }

        const uint64_t p1_words_bytes = siligemma::LIVE_LINEAR_PARAMS / 2ULL;
        const uint64_t p1_scales_bytes = siligemma::LIVE_LINEAR_PARAMS / 16ULL;
        const uint64_t p1_u4_cnt = p1_words_bytes / sizeof(uint4);
        const uint64_t p1_sc_cnt = p1_scales_bytes / sizeof(uint16_t);
        const uint64_t lm_words_bytes = has_lm ? (uint64_t(VOCAB_SIZE) * uint64_t(HIDDEN_SIZE) / 2ULL) : 0ULL;
        const uint64_t lm_scales_bytes = has_lm ? (uint64_t(VOCAB_SIZE) * uint64_t(HIDDEN_SIZE) / 16ULL) : 0ULL;
        const uint64_t lm_u4_cnt = lm_words_bytes / sizeof(uint4);
        const uint64_t lm_sc_cnt = lm_scales_bytes / sizeof(uint16_t);

        h_u4.resize(size_t(p1_u4_cnt + lm_u4_cnt));
        h_sc.resize(size_t(p1_sc_cnt + lm_sc_cnt));
        f.seekg(static_cast<std::streamoff>(off_p1));
        f.read(reinterpret_cast<char*>(h_u4.data()), static_cast<std::streamsize>(p1_words_bytes));
        f.read(reinterpret_cast<char*>(h_sc.data()), static_cast<std::streamsize>(p1_scales_bytes));
        if (!f.good()) return false;
        if (has_lm) {
            f.seekg(static_cast<std::streamoff>(off_p2_lm));
            f.read(reinterpret_cast<char*>(h_u4.data() + p1_u4_cnt), static_cast<std::streamsize>(lm_words_bytes));
            f.read(reinterpret_cast<char*>(h_sc.data() + p1_sc_cnt), static_cast<std::streamsize>(lm_scales_bytes));
            if (!f.good()) return false;
        }

        linears.clear();
        linears.reserve(hdr.num_linears);
        for (const auto& d1 : v1_linears) {
            if ((d1.words_byte_offset % sizeof(uint4)) != 0ULL ||
                d1.scales_byte_offset < p1_words_bytes ||
                ((d1.scales_byte_offset - p1_words_bytes) % sizeof(uint16_t)) != 0ULL) {
                return false;
            }
            SiliGemmaLinearDesc d2{};
            d2.layer_idx = d1.layer_idx;
            d2.kind = d1.role_id;
            d2.out_features = static_cast<uint32_t>(d1.out_dim);
            d2.in_features = static_cast<uint32_t>(d1.in_dim);
            d2.offset_u4 = d1.words_byte_offset / sizeof(uint4);
            d2.offset_sc = (d1.scales_byte_offset - p1_words_bytes) / sizeof(uint16_t);
            linears.push_back(d2);
        }
        if (has_lm) {
            SiliGemmaLinearDesc d2_lm{};
            d2_lm.layer_idx = 0u;
            d2_lm.kind = LIN_LM_HEAD_G32;
            d2_lm.out_features = VOCAB_SIZE;
            d2_lm.in_features = HIDDEN_SIZE;
            d2_lm.offset_u4 = p1_u4_cnt;
            d2_lm.offset_sc = p1_sc_cnt;
            linears.push_back(d2_lm);
        }

        h_lm.resize(size_t(VOCAB_SIZE) * HIDDEN_SIZE);
        f.seekg(static_cast<std::streamoff>(off_p2_emb));
        f.read(reinterpret_cast<char*>(h_lm.data()), static_cast<std::streamsize>(h_lm.size() * sizeof(uint16_t)));
        return f.good();
    }

    bool load_v2_slgb(std::ifstream& f, std::vector<uint4>& h_u4, std::vector<uint16_t>& h_sc, std::vector<uint16_t>& h_lm, std::vector<float>& h_u4g) {
        f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (hdr.magic != SLGB_MAGIC || hdr.version != SLGB_VERSION) {
            std::fprintf(stderr, "Invalid .slgb v2 header\n");
            return false;
        }
        if (hdr.num_layers != NUM_LAYERS || hdr.hidden_size != HIDDEN_SIZE || hdr.vocab_size != VOCAB_SIZE || hdr.ple_dim != PLE_DIM || hdr.num_linears < 276u) {
            std::fprintf(stderr, "Invalid .slgb v2 geometry\n");
            return false;
        }
        linears.resize(hdr.num_linears);
        f.read(reinterpret_cast<char*>(linears.data()), linears.size() * sizeof(SiliGemmaLinearDesc));
        layers.resize(hdr.num_layers);
        f.read(reinterpret_cast<char*>(layers.data()), layers.size() * sizeof(SiliGemmaLayerDesc));

        inv_freq_slide.resize(HEAD_DIM_SLIDE / 2);
        inv_freq_full.resize(HEAD_DIM_FULL / 2);
        f.read(reinterpret_cast<char*>(inv_freq_slide.data()), inv_freq_slide.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(inv_freq_full.data()), inv_freq_full.size() * sizeof(float));

        norms_f32.resize(hdr.total_norm_f32_count);
        f.read(reinterpret_cast<char*>(norms_f32.data()), norms_f32.size() * sizeof(float));
        if (norms_f32.size() < size_t(PLE_DIM) + size_t(HIDDEN_SIZE)) return false;
        ple_norm_off = norms_f32.size() - size_t(PLE_DIM) - size_t(HIDDEN_SIZE);
        final_norm_off = norms_f32.size() - size_t(HIDDEN_SIZE);

        h_u4.resize(hdr.total_u4_count);
        f.read(reinterpret_cast<char*>(h_u4.data()), h_u4.size() * sizeof(uint4));
        h_sc.resize(hdr.total_sc_count);
        f.read(reinterpret_cast<char*>(h_sc.data()), h_sc.size() * sizeof(uint16_t));
        if (hdr.has_lm_head_bf16) {
            h_lm.resize(size_t(VOCAB_SIZE) * HIDDEN_SIZE);
            f.read(reinterpret_cast<char*>(h_lm.data()), h_lm.size() * sizeof(uint16_t));
        }
        if (hdr.has_u4_loop && hdr.u4_gate_f32_count > 0) {
            h_u4g.resize(hdr.u4_gate_f32_count);
            f.read(reinterpret_cast<char*>(h_u4g.data()), h_u4g.size() * sizeof(float));
        }
        return f.good();
    }

    bool load(const std::string& path, int arena_max_b, int arena_t_cap, int bf16_kv) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f.good()) return false;
        const uint64_t fsz = static_cast<uint64_t>(f.tellg());
        f.seekg(0);
        uint32_t magic_ver[2] = {0u, 0u};
        f.read(reinterpret_cast<char*>(magic_ver), sizeof(magic_ver));
        if (!f.good() || magic_ver[0] != SLGB_MAGIC) {
            std::fprintf(stderr, "Invalid .slgb magic\n");
            return false;
        }
        f.seekg(0);

        std::vector<uint4> h_u4;
        std::vector<uint16_t> h_sc;
        std::vector<uint16_t> h_lm;
        std::vector<float> h_u4g;
        bool ok = false;
        if (magic_ver[1] == 1u) ok = load_v1_slgb(f, fsz, h_u4, h_sc, h_lm);
        else if (magic_ver[1] == 2u) ok = load_v2_slgb(f, h_u4, h_sc, h_lm, h_u4g);
        if (!ok) {
            std::fprintf(stderr, "Failed parsing .slgb (version=%u)\n", magic_ver[1]);
            return false;
        }

        CUDA_CHECK(cudaMalloc(&d_w_u4, h_u4.size() * sizeof(uint4)));
        CUDA_CHECK(cudaMemcpy(d_w_u4, h_u4.data(), h_u4.size() * sizeof(uint4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_w_sc, h_sc.size() * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(d_w_sc, h_sc.data(), h_sc.size() * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_norms, norms_f32.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_norms, norms_f32.data(), norms_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_inv_slide, inv_freq_slide.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_inv_slide, inv_freq_slide.data(), inv_freq_slide.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_inv_full, inv_freq_full.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_inv_full, inv_freq_full.data(), inv_freq_full.size() * sizeof(float), cudaMemcpyHostToDevice));

        if (!h_lm.empty()) {
            CUDA_CHECK(cudaMalloc(&d_lm_bf16, h_lm.size() * sizeof(uint16_t)));
            CUDA_CHECK(cudaMemcpy(d_lm_bf16, h_lm.data(), h_lm.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        }
        if (!h_u4g.empty()) {
            size_t off = 0;
            for (int k = 0; k < 9; ++k) {
                CUDA_CHECK(cudaMalloc(&d_u4_V16[k], size_t(U4_RANK) * HIDDEN_SIZE * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_u4_U16[k], size_t(U4_RANK) * HIDDEN_SIZE * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(d_u4_V16[k], h_u4g.data() + off, size_t(U4_RANK) * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
                off += size_t(U4_RANK) * HIDDEN_SIZE;
                CUDA_CHECK(cudaMemcpy(d_u4_U16[k], h_u4g.data() + off, size_t(U4_RANK) * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
                off += size_t(U4_RANK) * HIDDEN_SIZE;
                u4_alpha[k] = h_u4g[off++];
            }
        }

        // Allocate static P5 KV slab and static P7 scratch slab
        max_b = arena_max_b;
        t_cap = arena_t_cap;
        kv_is_bf16 = bf16_kv;
        size_t elem_sz = kv_is_bf16 ? 2ULL : 4ULL;
        size_t p5_req = 0;
        std::vector<size_t> slot_offs(NUM_KV_LAYERS * 2);
        for (int l = 0; l < NUM_KV_LAYERS; ++l) {
            size_t hd = layers[l].head_dim;
            slot_t_cap[l] = layers[l].is_full_attn ? arena_t_cap : std::min(arena_t_cap, SLIDING_WINDOW);
            size_t one = size_t(max_b) * size_t(slot_t_cap[l]) * size_t(NUM_KV_HEADS) * hd * elem_sz;
            one = (one + 255ULL) & ~255ULL;
            slot_offs[l * 2 + 0] = p5_req; p5_req += one;
            slot_offs[l * 2 + 1] = p5_req; p5_req += one;
        }
        p5_bytes = p5_req;
        CUDA_CHECK(cudaMalloc(&d_p5, p5_bytes));
        CUDA_CHECK(cudaMemset(d_p5, 0, p5_bytes));
        for (int l = 0; l < NUM_KV_LAYERS; ++l) {
            d_k_slots[l] = reinterpret_cast<float*>(d_p5 + slot_offs[l * 2 + 0]);
            d_v_slots[l] = reinterpret_cast<float*>(d_p5 + slot_offs[l * 2 + 1]);
        }
        CUDA_CHECK(cudaMalloc(&d_p7, ARENA_P7_SCRATCH_BYTES));
        CUDA_CHECK(cudaMemset(d_p7, 0, ARENA_P7_SCRATCH_BYTES));
        bind_p7_slices();
        return true;
    }

    void free_all() {
        if (d_w_u4) cudaFree(d_w_u4);
        if (d_w_sc) cudaFree(d_w_sc);
        if (d_norms) cudaFree(d_norms);
        if (d_inv_slide) cudaFree(d_inv_slide);
        if (d_inv_full) cudaFree(d_inv_full);
        if (d_lm_bf16) cudaFree(d_lm_bf16);
        for (int k = 0; k < 9; ++k) {
            if (d_u4_U16[k]) cudaFree(d_u4_U16[k]);
            if (d_u4_V16[k]) cudaFree(d_u4_V16[k]);
        }
        if (d_wb_U8) cudaFree(d_wb_U8);
        if (d_wb_V8) cudaFree(d_wb_V8);
        if (d_p5) cudaFree(d_p5);
        if (d_p7) cudaFree(d_p7);
        d_w_u4 = nullptr; d_p5 = nullptr; d_p7 = nullptr;
    }

    void emit_step(cudaStream_t st, int B, int K_passes, int use_lm_bf16, int use_split_t, int wb_enable, float softcap) {
        const SiliGemmaLinearDesc& ple_mod = linears[275]; // last linear in 276 = per_layer_model_projection
        const int use_mma = (B > 1);

        // 0. Embed lookup + scale on device (tied BF16 table, or g=32 Form B fallback in 128K 4.5bpw mode)
        if (d_lm_bf16) {
            embed_lookup_bf16_sm86<<<dim3((HIDDEN_SIZE + 127) / 128, B), 128, 0, st>>>(
                d_lm_bf16, d_tok_ids, d_hid, hdr.embed_scale, B, HIDDEN_SIZE);
        } else if (linears.size() > 276) {
            const auto& llm = linears[276];
            embed_lookup_g32_sm86<<<dim3((HIDDEN_SIZE >> 5) / 4, B), 128, 0, st>>>(
                d_w_u4 + llm.offset_u4, d_w_sc + llm.offset_sc, d_tok_ids, d_hid, hdr.embed_scale, B, HIDDEN_SIZE);
        } else {
            std::fprintf(stderr, "FATAL: neither BF16 embed nor g32 lm_head shadow present\n");
            std::exit(1);
        }

        // 0b. Context-aware PLE projection (`per_layer_model_projection` 8960 x 1536 -> combine w/ d_tok_ple -> d_ple_in)
        if (!use_mma) {
            fused_norm_1024_smem_h4_q22_sm86<<<1, 1024, 0, st>>>(
                d_hid, 1, nullptr, nullptr, 1.0f, nullptr, nullptr, d_xhm, d_xlo, d_xsc, d_xsq, nullptr, HIDDEN_SIZE, RMS_NORM_EPS, 1);
            gemv_b1_dp4a22_2row_sk_sm86<8, 1><<<dim3((NUM_LAYERS * PLE_DIM + 31) / 32, 1), 128, 0, st>>>(
                d_xhm, d_xlo, d_xsc, d_xsq,
                d_w_u4 + ple_mod.offset_u4, d_w_sc + ple_mod.offset_sc,
                d_tmp1, d_can, 0.0f, NUM_LAYERS * PLE_DIM, HIDDEN_SIZE);
        } else {
            fused_norm_b16_h4_fp16split_sm86<<<B, 128, 0, st>>>(
                d_hid, 1, nullptr, nullptr, 1.0f, nullptr, nullptr,
                d_xhi_T, d_xlo_T, nullptr, nullptr, d_scr, B, HIDDEN_SIZE, RMS_NORM_EPS, 1, 0);
            gemm_b16_mma_fast_sm86<1><<<(NUM_LAYERS * PLE_DIM + 31) / 32, 128, 0, st>>>(
                d_xhi_T, d_xlo_T, d_w_u4 + ple_mod.offset_u4, d_w_sc + ple_mod.offset_sc,
                d_tmp1, d_can, 0.0f, B, NUM_LAYERS * PLE_DIM, HIDDEN_SIZE);
        }
        ple_combine_tok_ctx_sm86<<<dim3(NUM_LAYERS, B), 128, 0, st>>>(
            d_tmp1, d_norms + ple_norm_off, d_tok_ple, d_ple_in, hdr.ple_model_proj_scale, B, NUM_LAYERS, PLE_DIM, RMS_NORM_EPS);

        // 1. Layer execution lambda (supports K=1 35L and K=2 44L h_pre24 L25..33 replay!)
        float prev_scalar = 1.0f;
        const float* prev_post_ple_nw = nullptr;
        int prev_active = 0;
        float prev_sc_pre24 = 1.0f;
        const float* prev_ppl_pre24 = nullptr;

        auto emit_layer = [&](int l, int loop_pass) {
            const SiliGemmaLayerDesc& L = layers[l];
            const int hd = int(L.head_dim);
            const int interm = int(L.interm_size);
            const int sh = int(L.is_kv_shared);
            const int kv_ro = sh || (loop_pass > 0);
            const int sw = L.sliding_window;
            const float* inv_f = L.is_full_attn ? d_inv_full : d_inv_slide;
            void* kc = static_cast<void*>(d_k_slots[L.kv_slot]);
            void* vc = static_cast<void*>(d_v_slots[L.kv_slot]);

            // Fuse prev layer's post_ple_norm + scalar_post WITH this layer's input_layernorm -> H_4 -> quant!
            if (!use_mma) {
                fused_norm_1024_smem_h4_q22_sm86<<<1, 1024, 0, st>>>(
                    prev_active ? d_tmp1 : d_hid, 1,
                    prev_active ? d_hid : nullptr, prev_post_ple_nw, prev_active ? prev_scalar : 1.0f,
                    d_norms + L.off_in_nw, d_hid, d_xhm, d_xlo, d_xsc, d_xsq, nullptr, HIDDEN_SIZE, RMS_NORM_EPS, 1);

                const auto& lq = linears[L.lin_q];
                if (!kv_ro) {
                    const auto& lk = linears[L.lin_k];
                    const auto& lv = linears[L.lin_v];
                    gemv_b1_qkv_fused_sm86<<<(NUM_HEADS * hd + 2 * NUM_KV_HEADS * hd + 15) / 16, 128, 0, st>>>(
                        d_xhm, d_xlo, d_xsc, d_xsq,
                        d_w_u4 + lq.offset_u4, d_w_sc + lq.offset_sc, NUM_HEADS * hd, d_tmp1,
                        d_w_u4 + lk.offset_u4, d_w_sc + lk.offset_sc, NUM_KV_HEADS * hd, d_tmp2,
                        d_w_u4 + lv.offset_u4, d_w_sc + lv.offset_sc, NUM_KV_HEADS * hd, d_tmp3,
                        d_can, HIDDEN_SIZE);
                } else {
                    gemv_b1_dp4a22_2row_sk_sm86<8, 1><<<dim3((NUM_HEADS * hd + 31) / 32, 1), 128, 0, st>>>(
                        d_xhm, d_xlo, d_xsc, d_xsq,
                        d_w_u4 + lq.offset_u4, d_w_sc + lq.offset_sc, d_tmp1, d_can, 0.0f, NUM_HEADS * hd, HIDDEN_SIZE);
                }
                qkv_norm_rope_kv_write_sm86<<<dim3(NUM_HEADS + (kv_ro ? 0 : 2), 1), 128, 0, st>>>(
                    d_tmp1, d_tmp2, d_tmp3,
                    d_norms + L.off_qn_w, kv_ro ? nullptr : (d_norms + L.off_kn_w), inv_f,
                    kc, vc, d_pos, 1, NUM_HEADS, NUM_KV_HEADS, hd, slot_t_cap[L.kv_slot], RMS_NORM_EPS, kv_ro, kv_is_bf16);

                if (use_split_t) {
                    attention_kv_split_sm86<6><<<dim3(NUM_HEADS, 6), 128, 0, st>>>(
                        d_tmp1, kc, vc, d_op, d_m, d_l, d_pos, 1, NUM_HEADS, NUM_KV_HEADS, hd, slot_t_cap[L.kv_slot], sw, kv_is_bf16);
                    attention_merge_h4_q22_fast_sm86<6><<<dim3((NUM_HEADS * hd >> 5) / 4, 1), 128, 0, st>>>(
                        d_op, d_m, d_l, d_xhm, d_xlo, d_xsc, d_xsq, 1, NUM_HEADS, hd);
                } else {
                    attention_kv_split_sm86<1><<<dim3(NUM_HEADS, 1), 128, 0, st>>>(
                        d_tmp1, kc, vc, d_tmp2, nullptr, nullptr, d_pos, 1, NUM_HEADS, NUM_KV_HEADS, hd, slot_t_cap[L.kv_slot], sw, kv_is_bf16);
                    stageA_norm_res_sm86<<<1, 128, 0, st>>>(
                        d_tmp2, 1, nullptr, nullptr, 1.0f, nullptr, nullptr, d_scr, d_inv_pre, nullptr, 1, NUM_HEADS * hd, RMS_NORM_EPS);
                    stageB_parallel_h4_q22_sm86<<<dim3((NUM_HEADS * hd >> 5) / 4, 1), 128, 0, st>>>(
                        d_scr, nullptr, d_inv_pre, d_xhm, d_xlo, d_xsc, d_xsq, 1, NUM_HEADS * hd, 1);
                }

                const auto& lo = linears[L.lin_o];
                gemv_b1_dp4a22_sm86<16, 4><<<dim3(HIDDEN_SIZE / 8, 4), 128, 0, st>>>(
                    d_xhm, d_xlo, d_xsc, d_xsq, d_w_u4 + lo.offset_u4, d_w_sc + lo.offset_sc, d_tmp1, d_can, 0.0f, HIDDEN_SIZE, NUM_HEADS * hd);
                fused_norm_1024_smem_h4_q22_sm86<<<1, 1024, 0, st>>>(
                    d_tmp1, 4, d_hid, d_norms + L.off_pa_nw, 1.0f, d_norms + L.off_pf_nw, d_hid,
                    d_xhm, d_xlo, d_xsc, d_xsq, nullptr, HIDDEN_SIZE, RMS_NORM_EPS, 1);

                const auto& lg = linears[L.lin_gate];
                const auto& lu = linears[L.lin_up];
                const auto& ld = linears[L.lin_down];
                gemv_b1_geglu_h4_q22_fused_sm86<<<interm / 32, 256, 0, st>>>(
                    d_xhm, d_xlo, d_xsc, d_xsq,
                    d_w_u4 + lg.offset_u4, d_w_sc + lg.offset_sc,
                    d_w_u4 + lu.offset_u4, d_w_sc + lu.offset_sc,
                    d_xhm2, d_xlo2, d_xsc2, d_xsq2, d_can, interm, HIDDEN_SIZE);
                gemv_b1_dp4a22_2row_sk_sm86<8, 8><<<dim3(HIDDEN_SIZE / 32, 8), 128, 0, st>>>(
                    d_xhm2, d_xlo2, d_xsc2, d_xsq2, d_w_u4 + ld.offset_u4, d_w_sc + ld.offset_sc, d_tmp1, d_can, 0.0f, HIDDEN_SIZE, interm);

                fused_norm_1024_smem_h4_q22_sm86<<<1, 1024, 0, st>>>(
                    d_tmp1, 8, d_hid, d_norms + L.off_pff_nw, 1.0f, nullptr, d_hid,
                    d_xhm, d_xlo, d_xsc, d_xsq, nullptr, HIDDEN_SIZE, RMS_NORM_EPS, 1);

                const auto& lpg = linears[L.lin_ple_gate];
                const auto& lpp = linears[L.lin_ple_proj];
                gemv_b1_dp4a22_sm86<16, 4><<<dim3(PLE_DIM / 8, 4), 128, 0, st>>>(
                    d_xhm, d_xlo, d_xsc, d_xsq, d_w_u4 + lpg.offset_u4, d_w_sc + lpg.offset_sc, d_tmp1, d_can, 0.0f, PLE_DIM, HIDDEN_SIZE);
                fused_ple_act_256_h4_q22_sm86<<<1, 256, 0, st>>>(
                    d_tmp1, 4, d_ple_in + l * PLE_DIM, d_xhm, d_xlo, d_xsc, d_xsq, 1, PLE_DIM);
                gemv_b1_dp4a22_sm86<4, 1><<<HIDDEN_SIZE / 32, 128, 0, st>>>(
                    d_xhm, d_xlo, d_xsc, d_xsq, d_w_u4 + lpp.offset_u4, d_w_sc + lpp.offset_sc, d_tmp1, d_can, 0.0f, HIDDEN_SIZE, PLE_DIM);
            } else {
                // B <= 16 exact Tensor Core MMA path
                fused_norm_b16_h4_fp16split_sm86<<<B, 128, 0, st>>>(
                    prev_active ? d_tmp1 : d_hid, 1,
                    prev_active ? d_hid : nullptr, prev_post_ple_nw, prev_active ? prev_scalar : 1.0f,
                    d_norms + L.off_in_nw, d_hid, d_xhi_T, d_xlo_T, nullptr, nullptr, d_scr, B, HIDDEN_SIZE, RMS_NORM_EPS, 1, 0);
                const auto& lq = linears[L.lin_q];
                gemm_b16_mma_fast_sm86<1><<<(NUM_HEADS * hd + 31) / 32, 128, 0, st>>>(
                    d_xhi_T, d_xlo_T, d_w_u4 + lq.offset_u4, d_w_sc + lq.offset_sc, d_tmp1, d_can, 0.0f, B, NUM_HEADS * hd, HIDDEN_SIZE);
                if (!kv_ro) {
                    const auto& lk = linears[L.lin_k];
                    const auto& lv = linears[L.lin_v];
                    gemm_b16_mma_fast_sm86<1><<<(NUM_KV_HEADS * hd + 31) / 32, 128, 0, st>>>(
                        d_xhi_T, d_xlo_T, d_w_u4 + lk.offset_u4, d_w_sc + lk.offset_sc, d_tmp2, d_can, 0.0f, B, NUM_KV_HEADS * hd, HIDDEN_SIZE);
                    gemm_b16_mma_fast_sm86<1><<<(NUM_KV_HEADS * hd + 31) / 32, 128, 0, st>>>(
                        d_xhi_T, d_xlo_T, d_w_u4 + lv.offset_u4, d_w_sc + lv.offset_sc, d_tmp3, d_can, 0.0f, B, NUM_KV_HEADS * hd, HIDDEN_SIZE);
                }
                qkv_norm_rope_kv_write_sm86<<<dim3(NUM_HEADS + (kv_ro ? 0 : 2), B), 128, 0, st>>>(
                    d_tmp1, d_tmp2, d_tmp3, d_norms + L.off_qn_w, kv_ro ? nullptr : (d_norms + L.off_kn_w), inv_f,
                    kc, vc, d_pos, B, NUM_HEADS, NUM_KV_HEADS, hd, slot_t_cap[L.kv_slot], RMS_NORM_EPS, kv_ro, kv_is_bf16);
                attention_kv_split_sm86<1><<<dim3(NUM_HEADS, B), 128, 0, st>>>(
                    d_tmp1, kc, vc, d_tmp2, nullptr, nullptr, d_pos, B, NUM_HEADS, NUM_KV_HEADS, hd, slot_t_cap[L.kv_slot], sw, kv_is_bf16);
                fused_norm_b16_h4_fp16split_sm86<<<B, 128, 0, st>>>(
                    d_tmp2, 1, nullptr, nullptr, 1.0f, nullptr, nullptr, d_xhi_T, d_xlo_T, nullptr, nullptr, d_scr, B, NUM_HEADS * hd, RMS_NORM_EPS, 1, 0);
                const auto& lo = linears[L.lin_o];
                gemm_b16_mma_fast_sm86<4><<<dim3(HIDDEN_SIZE / 32, 4), 128, 0, st>>>(
                    d_xhi_T, d_xlo_T, d_w_u4 + lo.offset_u4, d_w_sc + lo.offset_sc, d_tmp1, d_can, 0.0f, B, HIDDEN_SIZE, NUM_HEADS * hd);
                fused_norm_b16_h4_fp16split_sm86<<<B, 128, 0, st>>>(
                    d_tmp1, 4, d_hid, d_norms + L.off_pa_nw, 1.0f, d_norms + L.off_pf_nw, d_hid, d_xhi_T, d_xlo_T, nullptr, nullptr, d_scr, B, HIDDEN_SIZE, RMS_NORM_EPS, 1, 0);
                const auto& lg = linears[L.lin_gate];
                const auto& lu = linears[L.lin_up];
                const auto& ld = linears[L.lin_down];
                gemm_b16_mma_fast_sm86<1><<<interm / 32, 128, 0, st>>>(
                    d_xhi_T, d_xlo_T, d_w_u4 + lg.offset_u4, d_w_sc + lg.offset_sc, d_tmp2, d_can, 0.0f, B, interm, HIDDEN_SIZE);
                gemm_b16_mma_fast_sm86<1><<<interm / 32, 128, 0, st>>>(
                    d_xhi_T, d_xlo_T, d_w_u4 + lu.offset_u4, d_w_sc + lu.offset_sc, d_tmp3, d_can, 0.0f, B, interm, HIDDEN_SIZE);
                fused_geglu_or_ple_b16_h4_fp16split_sm86<<<dim3((interm >> 5) / 4, B), 128, 0, st>>>(
                    d_tmp2, d_tmp3, d_xhi_T, d_xlo_T, B, interm, interm, 0);
                gemm_b16_mma_fast_sm86<4><<<dim3(HIDDEN_SIZE / 32, 4), 128, 0, st>>>(
                    d_xhi_T, d_xlo_T, d_w_u4 + ld.offset_u4, d_w_sc + ld.offset_sc, d_tmp1, d_can, 0.0f, B, HIDDEN_SIZE, interm);
                fused_norm_b16_h4_fp16split_sm86<<<B, 128, 0, st>>>(
                    d_tmp1, 4, d_hid, d_norms + L.off_pff_nw, 1.0f, nullptr, d_hid, d_xhi_T, d_xlo_T, nullptr, nullptr, d_scr, B, HIDDEN_SIZE, RMS_NORM_EPS, 1, 0);
                const auto& lpg = linears[L.lin_ple_gate];
                const auto& lpp = linears[L.lin_ple_proj];
                gemm_b16_mma_fast_sm86<1><<<PLE_DIM / 32, 128, 0, st>>>(
                    d_xhi_T, d_xlo_T, d_w_u4 + lpg.offset_u4, d_w_sc + lpg.offset_sc, d_tmp2, d_can, 0.0f, B, PLE_DIM, HIDDEN_SIZE);
                fused_geglu_or_ple_b16_h4_fp16split_sm86<<<dim3((PLE_DIM >> 5) / 4, B), 128, 0, st>>>(
                    d_tmp2, d_ple_in, d_xhi_T, d_xlo_T, B, PLE_DIM, NUM_LAYERS * PLE_DIM, l * PLE_DIM);
                gemm_b16_mma_fast_sm86<1><<<HIDDEN_SIZE / 32, 128, 0, st>>>(
                    d_xhi_T, d_xlo_T, d_w_u4 + lpp.offset_u4, d_w_sc + lpp.offset_sc, d_tmp1, d_can, 0.0f, B, HIDDEN_SIZE, PLE_DIM);
            }

            // Optional Woodbury Cayley steering on L10..L14 hub
            if (wb_enable && l >= WOODBURY_HUB_START && l <= WOODBURY_HUB_END && d_wb_U8 && d_wb_V8) {
                cayley_woodbury_apply_sm86<<<B, 128, 0, st>>>(d_hid, d_wb_U8, d_wb_V8, -0.02f, B, HIDDEN_SIZE);
            }
            // Optional U-4 low-rank r=16 controller on pass 2 of L25..L33
            if (loop_pass > 0 && l >= U4_LOOP_START && l <= U4_LOOP_END && d_u4_U16[l - U4_LOOP_START]) {
                const int k = l - U4_LOOP_START;
                u4_loop_proj_r16_sm86<<<dim3(U4_RANK, B), 128, 0, st>>>(d_hid, d_u4_V16[k], d_u4_z, B, HIDDEN_SIZE);
                u4_loop_apply_r16_sm86<<<dim3((HIDDEN_SIZE + 127) / 128, B), 128, 0, st>>>(d_hid, d_u4_z, d_u4_U16[k], u4_alpha[k], B, HIDDEN_SIZE);
            }

            prev_post_ple_nw = d_norms + L.off_ppl_nw;
            prev_scalar = L.layer_scalar;
            prev_active = 1;
        };

        for (int l = 0; l < int(hdr.num_layers); ++l) {
            if (K_passes == 2 && l == U4_LOOP_START) {
                CUDA_CHECK(cudaMemcpyAsync(d_h_pre24, d_hid, size_t(B) * HIDDEN_SIZE * 4, cudaMemcpyDeviceToDevice, st));
                CUDA_CHECK(cudaMemcpyAsync(d_p_pre24, d_tmp1, size_t(B) * HIDDEN_SIZE * 4, cudaMemcpyDeviceToDevice, st));
                prev_ppl_pre24 = prev_post_ple_nw;
                prev_sc_pre24 = prev_scalar;
            }
            emit_layer(l, 0);
            if (K_passes == 2 && l == U4_LOOP_END) {
                CUDA_CHECK(cudaMemcpyAsync(d_hid, d_h_pre24, size_t(B) * HIDDEN_SIZE * 4, cudaMemcpyDeviceToDevice, st));
                CUDA_CHECK(cudaMemcpyAsync(d_tmp1, d_p_pre24, size_t(B) * HIDDEN_SIZE * 4, cudaMemcpyDeviceToDevice, st));
                prev_post_ple_nw = prev_ppl_pre24;
                prev_scalar = prev_sc_pre24;
                for (int ll = U4_LOOP_START; ll <= U4_LOOP_END; ++ll) emit_layer(ll, 1);
            }
        }

        // Finalize last layer post_ple_norm + layer_scalar + final_norm + lm_head + softcap
        if (!use_mma) {
            if (!use_lm_bf16 && hdr.has_lm_head_g32 && linears.size() > 276) {
                fused_norm_1024_smem_h4_q22_sm86<<<1, 1024, 0, st>>>(
                    d_tmp1, 1, d_hid, prev_post_ple_nw, prev_scalar, d_norms + final_norm_off,
                    d_hid, d_xhm, d_xlo, d_xsc, d_xsq, nullptr, HIDDEN_SIZE, RMS_NORM_EPS, 1);
                const auto& llm = linears[276];
                gemv_b1_dp4a22_2row_sk_sm86<8, 1><<<dim3((VOCAB_SIZE + 31) / 32, 1), 128, 0, st>>>(
                    d_xhm, d_xlo, d_xsc, d_xsq, d_w_u4 + llm.offset_u4, d_w_sc + llm.offset_sc,
                    d_logits, d_can, softcap, VOCAB_SIZE, HIDDEN_SIZE);
            } else {
                if (!d_lm_bf16) {
                    std::fprintf(stderr, "FATAL: BF16 lm_head unavailable\n");
                    std::exit(1);
                }
                fused_norm_1024_smem_h4_q22_sm86<<<1, 1024, 0, st>>>(
                    d_tmp1, 1, d_hid, prev_post_ple_nw, prev_scalar, d_norms + final_norm_off,
                    d_hid, nullptr, nullptr, nullptr, nullptr, d_tmp1, HIDDEN_SIZE, RMS_NORM_EPS, 0);
                lm_head_b1_bf16_uint4_sm86<<<(VOCAB_SIZE + 31) / 32, 128, 0, st>>>(
                    d_tmp1, reinterpret_cast<const uint4*>(d_lm_bf16), d_logits, softcap, VOCAB_SIZE, HIDDEN_SIZE);
            }
            parallel_argmax_stage1_sm86<<<128, 128, 0, st>>>(d_logits, d_bv, d_bi, VOCAB_SIZE);
            parallel_argmax_stage2_sm86<<<1, 128, 0, st>>>(d_bv, d_bi, d_out_id, 128);
        } else {
            fused_norm_b16_h4_fp16split_sm86<<<B, 128, 0, st>>>(
                d_tmp1, 1, d_hid, prev_post_ple_nw, prev_scalar, d_norms + final_norm_off,
                d_hid, d_xhi_T, d_xlo_T, d_xbhi_T, d_xblo_T, d_scr, B, HIDDEN_SIZE, RMS_NORM_EPS,
                (!use_lm_bf16 && hdr.has_lm_head_g32) ? 1 : 0, use_lm_bf16 ? 1 : 0);
            if (!use_lm_bf16 && hdr.has_lm_head_g32 && linears.size() > 276) {
                const auto& llm = linears[276];
                gemm_b16_mma_64x16_sm86<<<(VOCAB_SIZE + 63) / 64, 128, 0, st>>>(
                    d_xhi_T, d_xlo_T, d_w_u4 + llm.offset_u4, d_w_sc + llm.offset_sc, d_logits, d_can, softcap, B, VOCAB_SIZE, HIDDEN_SIZE);
            } else {
                if (!d_lm_bf16) {
                    std::fprintf(stderr, "FATAL: BF16 lm_head unavailable\n");
                    std::exit(1);
                }
                lm_head_b16_bf16_mma_sm86<<<(VOCAB_SIZE + 31) / 32, 128, 0, st>>>(
                    d_xbhi_T, d_xblo_T, reinterpret_cast<const uint4*>(d_lm_bf16), d_logits, softcap, B, VOCAB_SIZE, HIDDEN_SIZE);
            }
        }
        iris_canary_finalize_sm86<<<1, 128, 0, st>>>(d_hid, d_can, B, HIDDEN_SIZE);
    }
};

struct BenchConfig {
    bool cpu_selftest = false;
    bool gpu_selftest = false;
    std::string slgb_path;
    std::string prompt_ids_str = "2,107,108";
    std::string ple_tok_bin;
    std::string dump_logits_path;
    std::string out_json_path;
    int dump_b = 1;
    int dump_k = 1;
    int dump_lm_bf16 = 0;
    bool bench_all = false;
    bool allow_no_bos_test = false;
};

BenchConfig parse_args(int argc, char** argv) {
    BenchConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        auto next = [&](std::string& dst) {
            if (i + 1 >= argc) { std::fprintf(stderr, "Missing arg for %s\n", a.c_str()); std::exit(1); }
            dst = argv[++i];
        };
        if (a == "--cpu-selftest") cfg.cpu_selftest = true;
        else if (a == "--selftest-gpu") cfg.gpu_selftest = true;
        else if (a == "--run-slgb") next(cfg.slgb_path);
        else if (a == "--prompt-ids") next(cfg.prompt_ids_str);
        else if (a == "--ple-tok-bin") next(cfg.ple_tok_bin);
        else if (a == "--dump-logits") next(cfg.dump_logits_path);
        else if (a == "--out-json") next(cfg.out_json_path);
        else if (a == "--dump-b") {
            std::string v; next(v);
            if (!parse_int_strict(v, 1, 16, cfg.dump_b)) { std::fprintf(stderr, "Invalid --dump-b: %s\n", v.c_str()); std::exit(2); }
        } else if (a == "--dump-k") {
            std::string v; next(v);
            if (!parse_int_strict(v, 1, 2, cfg.dump_k)) { std::fprintf(stderr, "Invalid --dump-k: %s\n", v.c_str()); std::exit(2); }
        } else if (a == "--dump-lm-bf16") {
            std::string v; next(v);
            if (!parse_int_strict(v, 0, 1, cfg.dump_lm_bf16)) { std::fprintf(stderr, "Invalid --dump-lm-bf16: %s\n", v.c_str()); std::exit(2); }
        } else if (a == "--bench-all") cfg.bench_all = true;
        else if (a == "--allow-no-bos-test") cfg.allow_no_bos_test = true;
        else { std::fprintf(stderr, "Unknown argument: %s\n", a.c_str()); std::exit(2); }
    }
    return cfg;
}

} // namespace

int main(int argc, char** argv) {
    BenchConfig cfg = parse_args(argc, argv);
    if (cfg.cpu_selftest) return run_cpu_selftest();
    if (cfg.gpu_selftest) return run_gpu_selftest();
    if (cfg.slgb_path.empty()) {
        std::fprintf(stderr, "Usage: %s [--cpu-selftest | --selftest-gpu | --run-slgb <path> ...]\n", argv[0]);
        return 1;
    }

    std::vector<int> prompt_ids = parse_csv_ints(cfg.prompt_ids_str);
    // Fail-closed BOS=2 check
    if (prompt_ids.empty() || prompt_ids[0] != 2) {
        if (cfg.allow_no_bos_test) {
            std::printf("BOS_FAIL_CLOSED_VERIFIED first_tok=%d expected=2\n", prompt_ids.empty() ? -1 : prompt_ids[0]);
            return 0;
        }
        std::fprintf(stderr, "BOS_GATE_FAIL: prompt must start with bos_token_id=2 (got %d)\n",
                     prompt_ids.empty() ? -1 : prompt_ids[0]);
        return 2;
    }

    SlgbModel model;
    if (!model.load(cfg.slgb_path, 16, 1024, 0)) {
        std::fprintf(stderr, "Failed to load .slgb: %s\n", cfg.slgb_path.c_str());
        return 1;
    }

    // Read pre-sliced host-pinned PLE inputs `[n_steps, B, 35 * 256]` if provided
    std::vector<float> ple_all;
    if (!cfg.ple_tok_bin.empty()) {
        std::ifstream pf(cfg.ple_tok_bin, std::ios::binary | std::ios::ate);
        if (!pf.good()) {
            std::fprintf(stderr, "Failed opening --ple-tok-bin\n");
            return 1;
        }
        const std::streamoff pos = pf.tellg();
        if (pos < 0 || (static_cast<size_t>(pos) % sizeof(float)) != 0) {
            std::fprintf(stderr, "Invalid --ple-tok-bin byte length\n");
            return 1;
        }
        size_t sz = static_cast<size_t>(pos);
        pf.seekg(0);
        ple_all.resize(sz / sizeof(float));
        pf.read(reinterpret_cast<char*>(ple_all.data()), static_cast<std::streamsize>(sz));
        if (!pf.good()) {
            std::fprintf(stderr, "Failed reading --ple-tok-bin\n");
            return 1;
        }
    }

    cudaStream_t st = nullptr;
    CUDA_CHECK(cudaStreamCreate(&st));

    // Mode A: Dump logits `[T_steps, B, V]` across sequential prompt decode steps
    if (!cfg.dump_logits_path.empty()) {
        int B = cfg.dump_b;
        if (prompt_ids.size() % size_t(B) != 0) {
            std::fprintf(stderr, "prompt_ids length (%zu) must be divisible by B=%d\n", prompt_ids.size(), B);
            return 2;
        }
        int n_steps = int(prompt_ids.size()) / B;
        if (n_steps <= 0 || n_steps > model.t_cap) {
            std::fprintf(stderr, "n_steps=%d exceeds arena capacity %d\n", n_steps, model.t_cap);
            return 2;
        }
        CUDA_CHECK(cudaMemset(model.d_p5, 0, model.p5_bytes));
        CUDA_CHECK(cudaMemset(model.d_can, 0, sizeof(HoloSphereRazorCanary64)));
        std::ofstream lf(cfg.dump_logits_path, std::ios::binary | std::ios::trunc);
        if (!lf.good()) {
            std::fprintf(stderr, "Failed opening --dump-logits output\n");
            return 1;
        }
        std::vector<float> h_logits(size_t(B) * VOCAB_SIZE);
        for (int s = 0; s < n_steps; ++s) {
            std::vector<int32_t> hpos(B, s), htok(B);
            for (int b = 0; b < B; ++b) htok[b] = prompt_ids[b * n_steps + s];
            CUDA_CHECK(cudaMemcpyAsync(model.d_pos, hpos.data(), B * sizeof(int32_t), cudaMemcpyHostToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(model.d_tok_ids, htok.data(), B * sizeof(int32_t), cudaMemcpyHostToDevice, st));
            size_t step_ple = size_t(B) * NUM_LAYERS * PLE_DIM;
            if (ple_all.size() >= size_t(s + 1) * step_ple) {
                CUDA_CHECK(cudaMemcpyAsync(model.d_tok_ple, ple_all.data() + size_t(s) * step_ple, step_ple * sizeof(float), cudaMemcpyHostToDevice, st));
            }
            model.emit_step(st, B, cfg.dump_k, cfg.dump_lm_bf16, B == 1 ? 1 : 0, 0, FINAL_LOGIT_SOFTCAP);
            CUDA_CHECK(cudaMemcpyAsync(h_logits.data(), model.d_logits, h_logits.size() * sizeof(float), cudaMemcpyDeviceToHost, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            for (float v : h_logits) {
                if (!std::isfinite(v)) {
                    std::fprintf(stderr, "Non-finite logit encountered during dump\n");
                    return 3;
                }
            }
            lf.write(reinterpret_cast<const char*>(h_logits.data()), static_cast<std::streamsize>(h_logits.size() * sizeof(float)));
            if (!lf.good()) {
                std::fprintf(stderr, "Failed writing --dump-logits output\n");
                return 1;
            }
        }
        lf.flush();
        if (!lf.good()) {
            std::fprintf(stderr, "Failed flushing --dump-logits output\n");
            return 1;
        }
        lf.close();
        if (lf.fail()) {
            std::fprintf(stderr, "Failed closing --dump-logits output\n");
            return 1;
        }
        HoloSphereRazorCanary64 hcan{};
        CUDA_CHECK(cudaMemcpy(&hcan, model.d_can, sizeof(hcan), cudaMemcpyDeviceToHost));
        std::printf("DUMP_LOGITS_DONE steps=%d B=%d sec=%u ded=%u poisoned=%u\n",
                    n_steps, B, hcan.sec_cnt, hcan.ded_cnt, hcan.poisoned);
        if (hcan.ded_cnt != 0u || hcan.poisoned != 0u) {
            std::fprintf(stderr, "SAFE_HALT: SECDED DED/IRIS canary tripped during logit dump\n");
            return 3;
        }
    }

    // Mode B: Full Stage-2 live GPU bench (CUDA Graph throughput, PLE async DMA stall, 0-alloc 1k steps, Woodbury & U-4 rho)
    if (cfg.bench_all) {
        // Measure D2D stream bandwidth
        double stream_gbps = 0.0;
        {
            const size_t bw_bytes = 512ULL * 1024ULL * 1024ULL;
            void *d0 = nullptr, *d1 = nullptr;
            CUDA_CHECK(cudaMalloc(&d0, bw_bytes));
            CUDA_CHECK(cudaMalloc(&d1, bw_bytes));
            CUDA_CHECK(cudaMemset(d0, 0x5A, bw_bytes));
            for (int w = 0; w < 3; ++w) CUDA_CHECK(cudaMemcpy(d1, d0, bw_bytes, cudaMemcpyDeviceToDevice));
            cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
            std::vector<double> bws;
            for (int r = 0; r < 11; ++r) {
                CUDA_CHECK(cudaEventRecord(e0));
                CUDA_CHECK(cudaMemcpy(d1, d0, bw_bytes, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                bws.push_back((2.0 * double(bw_bytes)) / (double(ms) * 1.0e6));
            }
            std::sort(bws.begin(), bws.end());
            stream_gbps = bws[bws.size() / 2];
            CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
            CUDA_CHECK(cudaFree(d0)); CUDA_CHECK(cudaFree(d1));
        }

        // Measure CUDA Graphs
        auto run_graph_bench = [&](int B, int K, int lm_bf16, int T_ctx, int split_t, int wb_en, int iters) -> double {
            std::vector<int32_t> hpos(B, T_ctx - 1), htok(B, 2);
            CUDA_CHECK(cudaMemcpy(model.d_pos, hpos.data(), B * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(model.d_tok_ids, htok.data(), B * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(model.d_can, 0, sizeof(HoloSphereRazorCanary64)));

            cudaGraph_t graph; cudaGraphExec_t exec;
            CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
            model.emit_step(st, B, K, lm_bf16, split_t, wb_en, FINAL_LOGIT_SOFTCAP);
            CUDA_CHECK(cudaStreamEndCapture(st, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
            for (int w = 0; w < 6; ++w) CUDA_CHECK(cudaGraphLaunch(exec, st));
            CUDA_CHECK(cudaStreamSynchronize(st));

            cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
            std::vector<double> ms_list;
            const int reps = 19;
            for (int r = 0; r < reps; ++r) {
                CUDA_CHECK(cudaEventRecord(e0, st));
                for (int i = 0; i < iters; ++i) CUDA_CHECK(cudaGraphLaunch(exec, st));
                CUDA_CHECK(cudaEventRecord(e1, st));
                CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                ms_list.push_back(double(ms) / double(iters));
            }
            std::sort(ms_list.begin(), ms_list.end());
            CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
            CUDA_CHECK(cudaGraphExecDestroy(exec)); CUDA_CHECK(cudaGraphDestroy(graph));
            return ms_list[ms_list.size() / 2];
        };

        // Allocate small Woodbury test matrix on hub if not loaded
        if (!model.d_wb_U8) {
            std::vector<float> hU(8 * HIDDEN_SIZE, 0.0f), hV(8 * HIDDEN_SIZE, 0.0f), Ms(64, 0.0f), Minv(64, 0.0f);
            for (int r = 0; r < 8; ++r) hU[r * HIDDEN_SIZE + r] = 0.1f;
            for (int k = 0; k < 4; ++k) { hV[(2 * k) * HIDDEN_SIZE + (2 * k + 1)] = 0.1f; hV[(2 * k + 1) * HIDDEN_SIZE + (2 * k)] = -0.1f; }
            for (int r = 0; r < 8; ++r) for (int c = 0; c < 8; ++c) {
                double s = (r == c) ? 1.0 : 0.0;
                for (int d = 0; d < HIDDEN_SIZE; ++d) s += 0.5 * hV[r * HIDDEN_SIZE + d] * hU[c * HIDDEN_SIZE + d];
                Ms[r * 8 + c] = float(s);
            }
            invert_8x8_cpu(Ms.data(), Minv.data());
            CUDA_CHECK(cudaMemcpyToSymbol(c_wb_inv_256b, Minv.data(), 64 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&model.d_wb_U8, 8 * HIDDEN_SIZE * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&model.d_wb_V8, 8 * HIDDEN_SIZE * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(model.d_wb_U8, hU.data(), 8 * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(model.d_wb_V8, hV.data(), 8 * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        }

        double ms_b1_k1_g32_t128  = run_graph_bench(1, 1, 0, 128, 1, 0, 4);
        double ms_b1_k1_bf16_t128 = run_graph_bench(1, 1, 1, 128, 1, 0, 4);
        double ms_b1_k2_g32_t128  = run_graph_bench(1, 2, 0, 128, 1, 0, 4);
        double ms_b1_k2_bf16_t128 = run_graph_bench(1, 2, 1, 128, 1, 0, 4);
        double ms_b1_k1_g32_t1024 = run_graph_bench(1, 1, 0, 1024, 1, 0, 3);
        double ms_b1_k1_wb_t128   = run_graph_bench(1, 1, 0, 128, 1, 1, 4);
        double ms_b16_k1_g32_t128 = run_graph_bench(16, 1, 0, 128, 0, 0, 2);
        double ms_b16_k1_bf16_t128= run_graph_bench(16, 1, 1, 128, 0, 0, 2);
        double ms_b16_k2_g32_t128 = run_graph_bench(16, 2, 0, 128, 0, 0, 2);

        // Async PLE DMA stall test on concurrent stream vs baseline
        double ple_stall_pct = 0.0;
        {
            const int n_ple_f = NUM_LAYERS * PLE_DIM; // 8960
            float *h_pinned = nullptr, *d_ple_stage = nullptr, *d_ple_tmp = nullptr;
            CUDA_CHECK(cudaHostAlloc(&h_pinned, n_ple_f * sizeof(float), cudaHostAllocDefault));
            for (int i = 0; i < n_ple_f; ++i) h_pinned[i] = 0.01f * float(i & 31);
            CUDA_CHECK(cudaMalloc(&d_ple_stage, n_ple_f * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_ple_tmp, n_ple_f * sizeof(float)));
            cudaStream_t dma_st = nullptr; CUDA_CHECK(cudaStreamCreate(&dma_st));
            cudaGraph_t graph; cudaGraphExec_t exec;
            CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
            model.emit_step(st, 1, 1, 0, 1, 0, FINAL_LOGIT_SOFTCAP);
            CUDA_CHECK(cudaStreamEndCapture(st, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
            cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
            std::vector<double> ov_ms;
            for (int r = 0; r < 15; ++r) {
                CUDA_CHECK(cudaEventRecord(e0, st));
                for (int i = 0; i < 4; ++i) {
                    CUDA_CHECK(cudaMemcpyAsync(d_ple_stage, h_pinned, n_ple_f * sizeof(float), cudaMemcpyHostToDevice, dma_st));
                    ple_async_dma_touch_sm86<<<(n_ple_f + 127) / 128, 128, 0, dma_st>>>(d_ple_stage, d_ple_tmp, n_ple_f);
                    CUDA_CHECK(cudaGraphLaunch(exec, st));
                }
                CUDA_CHECK(cudaStreamSynchronize(dma_st));
                CUDA_CHECK(cudaEventRecord(e1, st)); CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                ov_ms.push_back(double(ms) / 4.0);
            }
            std::sort(ov_ms.begin(), ov_ms.end());
            double dma_med = ov_ms[ov_ms.size() / 2];
            ple_stall_pct = 100.0 * (dma_med - ms_b1_k1_g32_t128) / ms_b1_k1_g32_t128;
            CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
            CUDA_CHECK(cudaGraphExecDestroy(exec)); CUDA_CHECK(cudaGraphDestroy(graph));
            CUDA_CHECK(cudaStreamDestroy(dma_st));
            CUDA_CHECK(cudaFreeHost(h_pinned)); CUDA_CHECK(cudaFree(d_ple_stage)); CUDA_CHECK(cudaFree(d_ple_tmp));
        }

        // 1,000-step 0-byte dynamic VRAM allocation check
        size_t free_before = 0, total_mem = 0, free_after = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_before, &total_mem));
        {
            cudaGraph_t graph; cudaGraphExec_t exec;
            CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
            model.emit_step(st, 1, 1, 0, 1, 0, FINAL_LOGIT_SOFTCAP);
            CUDA_CHECK(cudaStreamEndCapture(st, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
            for (int step = 0; step < 1000; ++step) CUDA_CHECK(cudaGraphLaunch(exec, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            CUDA_CHECK(cudaGraphExecDestroy(exec)); CUDA_CHECK(cudaGraphDestroy(graph));
        }
        CUDA_CHECK(cudaMemGetInfo(&free_after, &total_mem));
        int64_t dynamic_vram_delta = int64_t(free_before) - int64_t(free_after);

        // Read Canary status after 1,000 steps
        HoloSphereRazorCanary64 hcan{};
        CUDA_CHECK(cudaMemcpy(&hcan, model.d_can, sizeof(hcan), cudaMemcpyDeviceToHost));

        // Live 128K BF16 KV arena check: deploy 4.5bpw 128K (release redundant d_lm_bf16 shadow & FP32 P5)
        if (model.d_lm_bf16) { CUDA_CHECK(cudaFree(model.d_lm_bf16)); model.d_lm_bf16 = nullptr; }
        CUDA_CHECK(cudaFree(model.d_p5));
        model.d_p5 = nullptr;
        uint8_t* d_p5_128k = nullptr;
        CUDA_CHECK(cudaMalloc(&d_p5_128k, ARENA_P5_128K_BF16_BYTES));
        CUDA_CHECK(cudaMemset(d_p5_128k, 0, 1024 * 1024));
        size_t p5_off = 0;
        for (int l = 0; l < NUM_KV_LAYERS; ++l) {
            size_t hd = model.layers[l].head_dim;
            int tc = model.layers[l].is_full_attn ? 131072 : SLIDING_WINDOW;
            model.slot_t_cap[l] = tc;
            size_t one = size_t(tc) * size_t(NUM_KV_HEADS) * hd * 2ULL;
            model.d_k_slots[l] = reinterpret_cast<float*>(d_p5_128k + p5_off); p5_off += one;
            model.d_v_slots[l] = reinterpret_cast<float*>(d_p5_128k + p5_off); p5_off += one;
        }
        model.kv_is_bf16 = 1;
        int pos_128k = 127;
        CUDA_CHECK(cudaMemcpy(model.d_pos, &pos_128k, 4, cudaMemcpyHostToDevice));
        model.emit_step(st, 1, 1, 0, 1, 0, FINAL_LOGIT_SOFTCAP);
        CUDA_CHECK(cudaStreamSynchronize(st));
        size_t free_128k = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_128k, &total_mem));
        double vram_128k_mib = double(total_mem - free_128k) / (1024.0 * 1024.0);
        CUDA_CHECK(cudaFree(d_p5_128k));

        double b1_k1_g32_tps  = 1000.0 / ms_b1_k1_g32_t128;
        double b1_k1_bf16_tps = 1000.0 / ms_b1_k1_bf16_t128;
        double b1_k2_g32_tps  = 1000.0 / ms_b1_k2_g32_t128;
        double b1_k2_bf16_tps = 1000.0 / ms_b1_k2_bf16_t128;
        double b1_k1_t1024_tps= 1000.0 / ms_b1_k1_g32_t1024;
        double b16_k1_g32_tps = 16000.0 / ms_b16_k1_g32_t128;
        double b16_k1_bf16_tps= 16000.0 / ms_b16_k1_bf16_t128;
        double b16_k2_g32_tps = 16000.0 / ms_b16_k2_g32_t128;
        double wb_overhead_pct= 100.0 * (ms_b1_k1_wb_t128 - ms_b1_k1_g32_t128) / ms_b1_k1_g32_t128;

        double bytes_k1_g32  = double(WIRE_BYTES_G32_45BPW + WIRE_BYTES_LM_HEAD_G32);
        double bytes_k1_bf16 = double(WIRE_BYTES_G32_45BPW + WIRE_BYTES_LM_HEAD_BF16);
        double gbps_k1_g32   = bytes_k1_g32 / (ms_b1_k1_g32_t128 * 1.0e6);
        double gbps_k1_bf16  = bytes_k1_bf16 / (ms_b1_k1_bf16_t128 * 1.0e6);

        // Empirical CUDA U-4 spectral contraction probe (measure ||delta_h|| contraction through L25..33 pass 2)
        double rho_cuda = 0.0;
        if (model.hdr.has_u4_loop) {
            std::vector<float> h0(HIDDEN_SIZE, 0.0f), h_p(HIDDEN_SIZE), h_o0(HIDDEN_SIZE), h_op(HIDDEN_SIZE);
            for (int d = 0; d < HIDDEN_SIZE; ++d) h0[d] = 0.25f * std::sin(0.013f * d);
            auto eval_u4_diff = [&](const std::vector<float>& vec) {
                CUDA_CHECK(cudaMemcpy(model.d_hid, vec.data(), HIDDEN_SIZE * 4, cudaMemcpyHostToDevice));
                std::vector<float> orig(vec);
                for (int k = 0; k < 9; ++k) {
                    u4_loop_proj_r16_sm86<<<dim3(U4_RANK, 1), 128, 0, st>>>(model.d_hid, model.d_u4_V16[k], model.d_u4_z, 1, HIDDEN_SIZE);
                    u4_loop_apply_r16_sm86<<<dim3((HIDDEN_SIZE + 127) / 128, 1), 128, 0, st>>>(model.d_hid, model.d_u4_z, model.d_u4_U16[k], model.u4_alpha[k], 1, HIDDEN_SIZE);
                }
                std::vector<float> out(HIDDEN_SIZE);
                CUDA_CHECK(cudaMemcpy(out.data(), model.d_hid, HIDDEN_SIZE * 4, cudaMemcpyDeviceToHost));
                for (int d = 0; d < HIDDEN_SIZE; ++d) out[d] -= orig[d];
                return out;
            };
            h_o0 = eval_u4_diff(h0);
            for (int trial = 0; trial < 8; ++trial) {
                h_p = h0;
                double eps_n2 = 0.0;
                for (int d = 0; d < HIDDEN_SIZE; ++d) {
                    float dv = 1e-3f * std::cos(float((trial + 1) * (d + 7)));
                    h_p[d] += dv;
                    eps_n2 += double(dv) * dv;
                }
                h_op = eval_u4_diff(h_p);
                double dn2 = 0.0;
                for (int d = 0; d < HIDDEN_SIZE; ++d) {
                    double df = double(h_op[d]) - double(h_o0[d]);
                    dn2 += df * df;
                }
                rho_cuda = std::max(rho_cuda, std::sqrt(dn2 / eps_n2));
            }
        }

        if (!cfg.out_json_path.empty()) {
            std::ofstream jf(cfg.out_json_path, std::ios::trunc);
            if (!jf.good()) {
                std::fprintf(stderr, "Failed opening --out-json\n");
                return 1;
            }
            jf << "{\n"
               << "  \"provenance_tag\": \"UNCONTROLLED_LAPTOP_NOT_CLAIMED\",\n"
               << "  \"d2d_stream_gbps\": " << stream_gbps << ",\n"
               << "  \"b1_k1_g32_t128_ms\": " << ms_b1_k1_g32_t128 << ",\n"
               << "  \"b1_k1_g32_t128_toks\": " << b1_k1_g32_tps << ",\n"
               << "  \"b1_k1_g32_t128_gbps\": " << gbps_k1_g32 << ",\n"
               << "  \"b1_k1_bf16_t128_ms\": " << ms_b1_k1_bf16_t128 << ",\n"
               << "  \"b1_k1_bf16_t128_toks\": " << b1_k1_bf16_tps << ",\n"
               << "  \"b1_k1_bf16_t128_gbps\": " << gbps_k1_bf16 << ",\n"
               << "  \"b1_k2_g32_t128_ms\": " << ms_b1_k2_g32_t128 << ",\n"
               << "  \"b1_k2_g32_t128_toks\": " << b1_k2_g32_tps << ",\n"
               << "  \"b1_k2_bf16_t128_ms\": " << ms_b1_k2_bf16_t128 << ",\n"
               << "  \"b1_k2_bf16_t128_toks\": " << b1_k2_bf16_tps << ",\n"
               << "  \"b1_k1_g32_t1024_ms\": " << ms_b1_k1_g32_t1024 << ",\n"
               << "  \"b1_k1_g32_t1024_toks\": " << b1_k1_t1024_tps << ",\n"
               << "  \"b16_k1_g32_t128_ms\": " << ms_b16_k1_g32_t128 << ",\n"
               << "  \"b16_k1_g32_t128_toks\": " << b16_k1_g32_tps << ",\n"
               << "  \"b16_k1_bf16_t128_ms\": " << ms_b16_k1_bf16_t128 << ",\n"
               << "  \"b16_k1_bf16_t128_toks\": " << b16_k1_bf16_tps << ",\n"
               << "  \"b16_k2_g32_t128_ms\": " << ms_b16_k2_g32_t128 << ",\n"
               << "  \"b16_k2_g32_t128_toks\": " << b16_k2_g32_tps << ",\n"
               << "  \"woodbury_overhead_pct\": " << wb_overhead_pct << ",\n"
               << "  \"ple_dma_stall_pct\": " << ple_stall_pct << ",\n"
               << "  \"dynamic_vram_delta_bytes_1k\": " << dynamic_vram_delta << ",\n"
               << "  \"arena_128k_vram_mib\": " << vram_128k_mib << ",\n"
               << "  \"secded_sec_cnt\": " << hcan.sec_cnt << ",\n"
               << "  \"secded_ded_cnt\": " << hcan.ded_cnt << ",\n"
               << "  \"iris_poisoned\": " << hcan.poisoned << ",\n"
               << "  \"rho_cuda\": " << rho_cuda << "\n"
               << "}\n";
            jf.flush();
            if (!jf.good()) {
                std::fprintf(stderr, "Failed writing --out-json\n");
                return 1;
            }
            jf.close();
            if (jf.fail()) {
                std::fprintf(stderr, "Failed closing --out-json\n");
                return 1;
            }
        }
        std::printf("BENCH_DONE d2d=%.2f GB/s | B=1 K=1 g32=%.2f tok/s (%.1f GB/s) bf16=%.2f tok/s | B=1 K=2 g32=%.2f bf16=%.2f | B=16 K=1 g32=%.2f bf16=%.2f | alloc_delta=%lld B\n",
                    stream_gbps, b1_k1_g32_tps, gbps_k1_g32, b1_k1_bf16_tps,
                    b1_k2_g32_tps, b1_k2_bf16_tps, b16_k1_g32_tps, b16_k1_bf16_tps,
                    (long long)dynamic_vram_delta);
        if (hcan.ded_cnt != 0u || hcan.poisoned != 0u) {
            std::fprintf(stderr, "SAFE_HALT: SECDED DED/IRIS canary tripped during bench (ded=%u poisoned=%u)\n",
                         hcan.ded_cnt, hcan.poisoned);
            return 3;
        }
    }

    CUDA_CHECK(cudaStreamDestroy(st));
    model.free_all();
    return 0;
}
