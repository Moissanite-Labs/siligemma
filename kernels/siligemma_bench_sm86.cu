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

bool parse_float_strict(const std::string& s, float& out) {
    if (s.empty()) return false;
    errno = 0;
    char* endptr = nullptr;
    const float v = std::strtof(s.c_str(), &endptr);
    if (errno == ERANGE || endptr == s.c_str() || *endptr != '\0' || !std::isfinite(v)) {
        return false;
    }
    out = v;
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

    // 2. Full Hamming(72,64) SECDED encoding/healing + inline A16 backward parity
    uint64_t seeds[3] = {0x123456789ABCDEF0ULL, 0x76543210FEDCBA98ULL, 0x0F1E2D3C4B5A6978ULL};
    const int sec_bits[3] = {7, 35, 61}; // non-LSB single-bit flips across all 64 bits
    for (int i = 0; i < 3; ++i) {
        const uint8_t ecc = siligemma::secded72_encode_cpu(seeds[i]);
        uint64_t clean_w = seeds[i];
        if (siligemma::secded72_decode_heal_cpu(clean_w, ecc) != 0 || clean_w != seeds[i]) {
            std::fprintf(stderr, "FAIL: SECDED72 clean check failed\n");
            return 1;
        }
        uint64_t sec_w = seeds[i] ^ (1ULL << sec_bits[i]);
        if (siligemma::secded72_decode_heal_cpu(sec_w, ecc) != 1 || sec_w != seeds[i]) {
            std::fprintf(stderr, "FAIL: SECDED72 SEC heal failed at bit %d\n", sec_bits[i]);
            return 1;
        }
        uint64_t ecc_flip_w = seeds[i];
        if (siligemma::secded72_decode_heal_cpu(ecc_flip_w, static_cast<uint8_t>(ecc ^ 0x08u)) != 1 || ecc_flip_w != seeds[i]) {
            std::fprintf(stderr, "FAIL: SECDED72 ECC-bit SEC heal failed\n");
            return 1;
        }
        uint64_t ded_w = seeds[i] ^ (1ULL << 5) ^ (1ULL << 52);
        if (siligemma::secded72_decode_heal_cpu(ded_w, ecc) != 2) {
            std::fprintf(stderr, "FAIL: SECDED72 DED detection failed\n");
            return 1;
        }
        uint64_t cw = encode_secded_w64_cpu(seeds[i]);
        if (secded_syn_cpu(cw) != 0u) {
            std::fprintf(stderr, "FAIL: A16 SECDED encode syndrome != 0\n");
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

    // 5. V2 LoopCD-Hidden, Goodfire P233c guard & Double Gram-Schmidt leak <= 1e-6 CPU check
    std::vector<float> h1(HIDDEN_SIZE, 0.0f), h2(HIDDEN_SIZE, 0.0f), qrows(4 * HIDDEN_SIZE, 0.0f), zin(HIDDEN_SIZE, 0.0f), zout(HIDDEN_SIZE, 0.0f);
    for (int d = 0; d < HIDDEN_SIZE; ++d) {
        h1[d] = std::sin(0.03f * float(d + 1));
        h2[d] = h1[d] + 0.1f * std::cos(0.05f * float(d + 1));
        zin[d] = std::sin(0.11f * float(d + 3));
    }
    h1[438] = 136.0f; h2[438] = 10.0f; // outlier spike must be ignored by P233c & LoopCD
    float sr_clean = 0.0f, cos_clean = 0.0f;
    p233c_clean_halt_metrics_cpu(h1.data(), h2.data(), HIDDEN_SIZE, sr_clean, cos_clean);
    if (!std::isfinite(sr_clean) || sr_clean > 0.3f || cos_clean < 0.9f) {
        std::fprintf(stderr, "FAIL: V2 Goodfire clean halt check sr=%.4f cos=%.4f\n", sr_clean, cos_clean);
        return 1;
    }
    std::vector<float> h_r = h2;
    loopcd_hidden_extrap_cpu(h_r.data(), h1.data(), 0.25f, HIDDEN_SIZE);
    if (h_r[438] != h2[438]) {
        std::fprintf(stderr, "FAIL: V2 LoopCD touched Goodfire outlier dim 438\n");
        return 1;
    }
    for (int r = 0; r < 4; ++r) qrows[size_t(r) * HIDDEN_SIZE + r * 7] = 1.0f;
    double leak1 = 0.0, leak2 = 0.0;
    double_gs_project_cpu(zin.data(), qrows.data(), 4, HIDDEN_SIZE, zout.data(), &leak1, &leak2);
    if (leak2 > S97_ORTH_LEAK_LIMIT) {
        std::fprintf(stderr, "FAIL: V2 double-GS leak %.3e > 1e-6\n", leak2);
        return 1;
    }

    const double arena128k_mib = double(kArena128KFormBG32) / (1024.0 * 1024.0);
    if (arena128k_mib > MAX_128K_VRAM_MIB) {
        std::fprintf(stderr, "FAIL: 128K arena %.2f MiB exceeds %.1f MiB cap\n", arena128k_mib, MAX_128K_VRAM_MIB);
        return 1;
    }

    std::printf("CPU_SELFTEST_PASS tree_bits=0x%08x wb_err=%.3e arena128k_mib=%.2f v1_slgb_bytes=%llu dgs_leak=%.3e\n",
                float_to_bits(r1), max_id_err, arena128k_mib,
                static_cast<unsigned long long>(exp_v1_sz), leak2);
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

    // 3. Full Hamming(72,64) SECDED (3/3 SEC heal + 1/1 ECC SEC + 3/3 DED) & IRIS 1/1 activation tamper check
    const int M = 32, K = 128;
    std::vector<uint4> hw(M * (K / 32));
    std::vector<uint32_t> hsc_ecc(M * (K / 32), 0u);
    const __half sc_h = __float2half(0.05f);
    uint16_t sc_u16 = 0u;
    std::memcpy(&sc_u16, &sc_h, sizeof(sc_u16));
    for (size_t i = 0; i < hw.size(); ++i) {
        const uint64_t c0 = 0x1122334455667788ULL ^ (i * 0x9E3779B9ULL);
        const uint64_t c1 = 0x8877665544332211ULL ^ (i * 0x517CC1B7ULL);
        hw[i] = make_uint4(uint32_t(c0), uint32_t(c0 >> 32), uint32_t(c1), uint32_t(c1 >> 32));
        const uint8_t e0 = siligemma::secded72_encode_cpu(c0);
        const uint8_t e1 = siligemma::secded72_encode_cpu(c1);
        hsc_ecc[i] = uint32_t(sc_u16) | (uint32_t(e0) << 16) | (uint32_t(e1) << 24);
    }
    std::vector<int4> h_hm(K / 8, make_int4(0x11111111, 0x22222222, 0x11111111, 0x22222222));
    std::vector<int2> h_lo(K / 8, make_int2(0x11111111, 0x11111111));
    std::vector<float> h_xsc(K / 32, 0.25f);
    std::vector<int32_t> h_xsq(K / 32, 16);

    int4 *d_hm = nullptr; int2 *d_lo = nullptr; float *d_xsc = nullptr, *d_y = nullptr; int32_t *d_xsq = nullptr;
    uint4 *d_w = nullptr; uint32_t *d_sc = nullptr; HoloSphereRazorCanary64 *d_can = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hm, (K / 8) * sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&d_lo, (K / 8) * sizeof(int2)));
    CUDA_CHECK(cudaMalloc(&d_xsc, (K / 32) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_xsq, (K / 32) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, hw.size() * sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&d_sc, hsc_ecc.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_can, sizeof(HoloSphereRazorCanary64)));
    CUDA_CHECK(cudaMemcpy(d_hm, h_hm.data(), (K / 8) * sizeof(int4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lo, h_lo.data(), (K / 8) * sizeof(int2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xsc, h_xsc.data(), (K / 32) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xsq, h_xsq.data(), (K / 32) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sc, hsc_ecc.data(), hsc_ecc.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Clean check -> non-zero output, 0 SEC, 0 DED, WATCHDOG CONTINUE
    CUDA_CHECK(cudaMemcpy(d_w, hw.data(), hw.size() * sizeof(uint4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_can, 0, sizeof(HoloSphereRazorCanary64)));
    gemv_b1_dp4a22_sm86<4, 1><<<M / 32, 128>>>(d_hm, d_lo, d_xsc, d_xsq, d_w, d_sc, d_y, d_can, 0.0f, M, K);
    iris_canary_finalize_sm86<<<1, 128>>>(d_y, d_can, 1, M);
    std::vector<float> h_y_clean(M), h_y_mut(M);
    CUDA_CHECK(cudaMemcpy(h_y_clean.data(), d_y, M * sizeof(float), cudaMemcpyDeviceToHost));
    HoloSphereRazorCanary64 hcan{};
    CUDA_CHECK(cudaMemcpy(&hcan, d_can, sizeof(hcan), cudaMemcpyDeviceToHost));
    double max_clean_y = 0.0;
    for (float v : h_y_clean) max_clean_y = std::max(max_clean_y, std::fabs(double(v)));
    const uint32_t clean_iris_crc = hcan.iris_runtime_crc_u32;
    if (max_clean_y < 1e-3 || hcan.sec_cnt != 0u || hcan.ded_cnt != 0u || hcan.poisoned != 0u ||
        evaluate_canary_watchdog(hcan, clean_iris_crc, true) != WatchdogAction::CONTINUE) {
        std::fprintf(stderr, "FAIL: clean SECDED72 check max_y=%.3e sec=%u ded=%u pois=%u\n",
                     max_clean_y, hcan.sec_cnt, hcan.ded_cnt, hcan.poisoned);
        return 1;
    }

    int sec_pass = 0, ecc_sec_pass = 0, ded_pass = 0;
    float max_sec_heal_diff = 0.0f;
    for (int t = 0; t < 3; ++t) {
        std::vector<uint4> mut = hw;
        if (t == 0) mut[1].x ^= (1u << 7);
        else if (t == 1) mut[5].y ^= (1u << 11);
        else mut[9].z ^= (1u << 23);
        CUDA_CHECK(cudaMemcpy(d_w, mut.data(), mut.size() * sizeof(uint4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_can, 0, sizeof(HoloSphereRazorCanary64)));
        gemv_b1_dp4a22_sm86<4, 1><<<M / 32, 128>>>(d_hm, d_lo, d_xsc, d_xsq, d_w, d_sc, d_y, d_can, 0.0f, M, K);
        iris_canary_finalize_sm86<<<1, 128>>>(d_y, d_can, 1, M);
        CUDA_CHECK(cudaMemcpy(&hcan, d_can, sizeof(hcan), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y_mut.data(), d_y, M * sizeof(float), cudaMemcpyDeviceToHost));
        float trial_diff = 0.0f;
        for (int i = 0; i < M; ++i) trial_diff = std::max(trial_diff, std::fabs(h_y_mut[i] - h_y_clean[i]));
        max_sec_heal_diff = std::max(max_sec_heal_diff, trial_diff);
        if (hcan.sec_cnt >= 1u && hcan.ded_cnt == 0u && hcan.poisoned == 0u &&
            evaluate_canary_watchdog(hcan, clean_iris_crc, false) == WatchdogAction::CONTINUE &&
            trial_diff == 0.0f) {
            sec_pass++;
        }
    }
    // 1/1 SEC flip in ECC parity byte
    {
        std::vector<uint32_t> mut_sc = hsc_ecc;
        mut_sc[2] ^= (1u << 19);
        CUDA_CHECK(cudaMemcpy(d_w, hw.data(), hw.size() * sizeof(uint4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sc, mut_sc.data(), mut_sc.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_can, 0, sizeof(HoloSphereRazorCanary64)));
        gemv_b1_dp4a22_sm86<4, 1><<<M / 32, 128>>>(d_hm, d_lo, d_xsc, d_xsq, d_w, d_sc, d_y, d_can, 0.0f, M, K);
        iris_canary_finalize_sm86<<<1, 128>>>(d_y, d_can, 1, M);
        CUDA_CHECK(cudaMemcpy(&hcan, d_can, sizeof(hcan), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y_mut.data(), d_y, M * sizeof(float), cudaMemcpyDeviceToHost));
        float trial_diff = 0.0f;
        for (int i = 0; i < M; ++i) trial_diff = std::max(trial_diff, std::fabs(h_y_mut[i] - h_y_clean[i]));
        max_sec_heal_diff = std::max(max_sec_heal_diff, trial_diff);
        if (hcan.sec_cnt >= 1u && hcan.ded_cnt == 0u && hcan.poisoned == 0u && trial_diff == 0.0f) {
            ecc_sec_pass = 1;
        }
        CUDA_CHECK(cudaMemcpy(d_sc, hsc_ecc.data(), hsc_ecc.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    for (int t = 0; t < 3; ++t) {
        std::vector<uint4> mut = hw;
        std::vector<uint32_t> mut_sc = hsc_ecc;
        if (t == 0) {
            mut[0].x ^= (1u << 4) | (1u << 19);
        } else if (t == 1) {
            mut[4].x ^= (1u << 12);
            mut[4].y ^= (1u << 5);
        } else {
            mut[6].x ^= (1u << 9);
            mut_sc[6] ^= (1u << 17);
        }
        CUDA_CHECK(cudaMemcpy(d_w, mut.data(), mut.size() * sizeof(uint4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sc, mut_sc.data(), mut_sc.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_can, 0, sizeof(HoloSphereRazorCanary64)));
        gemv_b1_dp4a22_sm86<4, 1><<<M / 32, 128>>>(d_hm, d_lo, d_xsc, d_xsq, d_w, d_sc, d_y, d_can, 0.0f, M, K);
        iris_canary_finalize_sm86<<<1, 128>>>(d_y, d_can, 1, M);
        CUDA_CHECK(cudaMemcpy(&hcan, d_can, sizeof(hcan), cudaMemcpyDeviceToHost));
        if (hcan.ded_cnt >= 1u && hcan.poisoned == 1u &&
            evaluate_canary_watchdog(hcan, clean_iris_crc, false) == WatchdogAction::SAFE_HALT_DED_ROLLBACK) {
            ded_pass++;
        }
    }
    // 1/1 Real IRIS runtime activation-tampering check
    int iris_trojan_pass = 0;
    {
        CUDA_CHECK(cudaMemcpy(d_w, hw.data(), hw.size() * sizeof(uint4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sc, hsc_ecc.data(), hsc_ecc.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_can, 0, sizeof(HoloSphereRazorCanary64)));
        gemv_b1_dp4a22_sm86<4, 1><<<M / 32, 128>>>(d_hm, d_lo, d_xsc, d_xsq, d_w, d_sc, d_y, d_can, 0.0f, M, K);
        float y0_tampered = h_y_clean[0] + 1.5f;
        CUDA_CHECK(cudaMemcpy(d_y, &y0_tampered, sizeof(float), cudaMemcpyHostToDevice));
        iris_canary_finalize_sm86<<<1, 128>>>(d_y, d_can, 1, M);
        CUDA_CHECK(cudaMemcpy(&hcan, d_can, sizeof(hcan), cudaMemcpyDeviceToHost));
        if (hcan.iris_runtime_crc_u32 != clean_iris_crc &&
            evaluate_canary_watchdog(hcan, clean_iris_crc, true) == WatchdogAction::SAFE_HALT_DED_ROLLBACK) {
            HoloSphereRazorCanary64 mismatch_chk{};
            mismatch_chk.iris_mismatch_count_u32 = 1u;
            CUDA_CHECK(cudaMemcpy(d_can, &mismatch_chk, sizeof(mismatch_chk), cudaMemcpyHostToDevice));
            iris_canary_finalize_sm86<<<1, 128>>>(d_y, d_can, 1, M);
            CUDA_CHECK(cudaMemcpy(&hcan, d_can, sizeof(hcan), cudaMemcpyDeviceToHost));
            if (hcan.poisoned == 1u && hcan.checksum == (hcan.iris_state ^ 1ULL)) {
                iris_trojan_pass = 1;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_hm)); CUDA_CHECK(cudaFree(d_lo)); CUDA_CHECK(cudaFree(d_xsc)); CUDA_CHECK(cudaFree(d_xsq));
    CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_sc)); CUDA_CHECK(cudaFree(d_can));

    // 4. Live cudaMemcpy D2D 512 MiB physical DRAM stream bandwidth benchmark (bidir 2x and 1x copy)
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
    std::vector<double> bws_bidir, bws_copy;
    for (int r = 0; r < 11; ++r) {
        CUDA_CHECK(cudaEventRecord(e0));
        CUDA_CHECK(cudaMemcpy(d_dst, d_src, bw_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        bws_bidir.push_back((2.0 * double(bw_bytes)) / (double(ms) * 1.0e6));
        bws_copy.push_back(double(bw_bytes) / (double(ms) * 1.0e6));
    }
    std::sort(bws_bidir.begin(), bws_bidir.end());
    std::sort(bws_copy.begin(), bws_copy.end());
    double stream_bidir_gbps = bws_bidir[bws_bidir.size() / 2];
    double stream_copy_gbps = bws_copy[bws_copy.size() / 2];
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));

    // 4b. Quick synthetic 262144 x 1536 LM-head GEMV bandwidth sanity with secded72
    double lm_syn_ms = 0.0;
    {
        const int LM_M = VOCAB_SIZE, LM_K = HIDDEN_SIZE;
        int4 *dl_hm = nullptr; int2 *dl_lo = nullptr; float *dl_xsc = nullptr, *dl_y = nullptr; int32_t *dl_xsq = nullptr;
        uint4 *dl_w = nullptr; uint32_t *dl_sc = nullptr; HoloSphereRazorCanary64 *dl_can = nullptr;
        CUDA_CHECK(cudaMalloc(&dl_hm, size_t(LM_K / 8) * sizeof(int4)));
        CUDA_CHECK(cudaMalloc(&dl_lo, size_t(LM_K / 8) * sizeof(int2)));
        CUDA_CHECK(cudaMalloc(&dl_xsc, size_t(LM_K / 32) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dl_xsq, size_t(LM_K / 32) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&dl_y, size_t(LM_M) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dl_w, size_t(LM_M) * size_t(LM_K / 32) * sizeof(uint4)));
        CUDA_CHECK(cudaMalloc(&dl_sc, size_t(LM_M) * size_t(LM_K / 32) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dl_can, sizeof(HoloSphereRazorCanary64)));
        CUDA_CHECK(cudaMemset(dl_hm, 0, size_t(LM_K / 8) * sizeof(int4)));
        CUDA_CHECK(cudaMemset(dl_lo, 0, size_t(LM_K / 8) * sizeof(int2)));
        CUDA_CHECK(cudaMemset(dl_xsc, 0, size_t(LM_K / 32) * sizeof(float)));
        CUDA_CHECK(cudaMemset(dl_xsq, 0, size_t(LM_K / 32) * sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(dl_w, 0, size_t(LM_M) * size_t(LM_K / 32) * sizeof(uint4)));
        CUDA_CHECK(cudaMemset(dl_sc, 0, size_t(LM_M) * size_t(LM_K / 32) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(dl_can, 0, sizeof(HoloSphereRazorCanary64)));
        for (int w = 0; w < 3; ++w) {
            v2::gemv_b1_dp4a22_2row_smem_sk_sm86<8, 1><<<dim3((LM_M + 31) / 32, 1), 128>>>(
                dl_hm, dl_lo, dl_xsc, dl_xsq, dl_w, dl_sc, dl_y, dl_can, 0.0f, LM_M, LM_K);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> lm_ms;
        for (int r = 0; r < 9; ++r) {
            CUDA_CHECK(cudaEventRecord(e0));
            v2::gemv_b1_dp4a22_2row_smem_sk_sm86<8, 1><<<dim3((LM_M + 31) / 32, 1), 128>>>(
                dl_hm, dl_lo, dl_xsc, dl_xsq, dl_w, dl_sc, dl_y, dl_can, 0.0f, LM_M, LM_K);
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            lm_ms.push_back(double(ms));
        }
        std::sort(lm_ms.begin(), lm_ms.end());
        lm_syn_ms = lm_ms[lm_ms.size() / 2];
        CUDA_CHECK(cudaFree(dl_hm)); CUDA_CHECK(cudaFree(dl_lo)); CUDA_CHECK(cudaFree(dl_xsc));
        CUDA_CHECK(cudaFree(dl_xsq)); CUDA_CHECK(cudaFree(dl_y)); CUDA_CHECK(cudaFree(dl_w));
        CUDA_CHECK(cudaFree(dl_sc)); CUDA_CHECK(cudaFree(dl_can));
    }
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    if (sec_pass != 3 || ecc_sec_pass != 1 || ded_pass != 3 || iris_trojan_pass != 1 || wb_err > 1e-6 || norm_drift > 1e-6) {
        std::fprintf(stderr, "FAIL: gpu_selftest sec=%d/3 ecc_sec=%d/1 ded=%d/3 iris=%d/1 wb=%.3e norm_drift=%.3e\n",
                     sec_pass, ecc_sec_pass, ded_pass, iris_trojan_pass, wb_err, norm_drift);
        return 1;
    }

    // 5. V2 GPU Double-GS QR (s97_leak <= 1e-6), LoopCD-Hidden & P233c Goodfire halt selftest
    double gpu_dgs_leak = 0.0;
    {
        float *dz = nullptr, *dq = nullptr, *dh1 = nullptr, *dh2 = nullptr, *dsr = nullptr, *dcs = nullptr;
        int32_t *dhalt = nullptr;
        CUDA_CHECK(cudaMalloc(&dz, HIDDEN_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dq, 16 * HIDDEN_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dh1, HIDDEN_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dh2, HIDDEN_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dsr, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dcs, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dhalt, sizeof(int32_t)));
        std::vector<float> hz(HIDDEN_SIZE), hq(16 * HIDDEN_SIZE, 0.0f), v1(HIDDEN_SIZE), v2(HIDDEN_SIZE);
        for (int d = 0; d < HIDDEN_SIZE; ++d) {
            hz[d] = std::sin(0.071f * float(d + 1)) * 10.0f;
            v1[d] = std::cos(0.031f * float(d + 1));
            v2[d] = v1[d] + 0.05f * std::sin(0.019f * float(d + 1));
        }
        v1[438] = 136.0f; v2[438] = -20.0f;
        for (int r = 0; r < 16; ++r) hq[size_t(r) * HIDDEN_SIZE + r * 11] = 1.0f;
        CUDA_CHECK(cudaMemcpy(dz, hz.data(), HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dq, hq.data(), 16 * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dh1, v1.data(), HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dh2, v2.data(), HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        v2::riccati_cayley_extrap_f32_sm86<<<1, 128>>>(dz, dq, 16, 1, HIDDEN_SIZE);
        v2::p233c_outlier_safe_halt_sm86<<<1, 128>>>(dh1, dh2, 0.25f, 0.90f, dsr, dcs, dhalt, 1, HIDDEN_SIZE);
        v2::loopcd_hidden_extrap_f32_sm86<<<dim3((HIDDEN_SIZE + 127) / 128, 1), 128>>>(dh2, dh1, 0.15f, 1, HIDDEN_SIZE);
        std::vector<float> hz_out(HIDDEN_SIZE), hv2_out(HIDDEN_SIZE);
        float sr_g = 0.0f, cs_g = 0.0f; int32_t halt_g = 0;
        CUDA_CHECK(cudaMemcpy(hz_out.data(), dz, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hv2_out.data(), dh2, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&sr_g, dsr, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&cs_g, dcs, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&halt_g, dhalt, sizeof(int32_t), cudaMemcpyDeviceToHost));
        double zn = 0.0, pn = 0.0;
        for (int d = 0; d < HIDDEN_SIZE; ++d) zn += double(hz_out[d]) * hz_out[d];
        for (int r = 0; r < 16; ++r) {
            double s = double(hz_out[r * 11]);
            pn += s * s;
        }
        gpu_dgs_leak = std::sqrt(pn / std::max(zn, 1.0e-24));
        CUDA_CHECK(cudaFree(dz)); CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dh1));
        CUDA_CHECK(cudaFree(dh2)); CUDA_CHECK(cudaFree(dsr)); CUDA_CHECK(cudaFree(dcs));
        CUDA_CHECK(cudaFree(dhalt));
        if (gpu_dgs_leak > S97_ORTH_LEAK_LIMIT || halt_g != 1 || std::fabs(hv2_out[438] - v2[438]) > 1e-6f) {
            std::fprintf(stderr, "FAIL: V2 gpu_dgs_leak=%.3e halt=%d outlier_guard=%.3e\n",
                         gpu_dgs_leak, halt_g, std::fabs(hv2_out[438] - v2[438]));
            return 1;
        }
    }

    std::printf("GPU_SELFTEST_PASS max_ulp=%u wb_err=%.3e norm_drift=%.3e sec=%d/3 ecc_sec=%d/1 max_heal_diff=%.3e ded=%d/3 iris=%d/1 d2d_bidir_gbps=%.2f d2d_copy_gbps=%.2f lm_syn_ms=%.3f dgs_leak=%.3e\n",
                max_ulp, wb_err, norm_drift, sec_pass, ecc_sec_pass, max_sec_heal_diff, ded_pass, iris_trojan_pass, stream_bidir_gbps, stream_copy_gbps, lm_syn_ms, gpu_dgs_leak);
    return 0;
}

struct SlgbModel {
    SlgbRuntimeHeader hdr{};
    std::vector<SlgbRuntimeLinearDesc> linears;
    std::vector<SlgbRuntimeLayerDesc> layers;
    std::vector<float> inv_freq_slide;
    std::vector<float> inv_freq_full;
    std::vector<float> norms_f32;
    size_t ple_norm_off = 0;
    size_t final_norm_off = 0;
    size_t total_u4_count = 0;
    size_t total_sc_count = 0;

    // Device persistent pointers
    uint4* d_w_u4 = nullptr;
    uint32_t* d_w_sc_ecc = nullptr;
    float* d_norms = nullptr;
    float* d_inv_slide = nullptr;
    float* d_inv_full = nullptr;
    uint16_t* d_lm_bf16 = nullptr;
    float* d_u4_U16[9]{};
    float* d_u4_V16[9]{};
    float u4_alpha[9]{};
    float loopcd_w_sidecar = 0.0f;
    float tau_sr_sidecar = 0.25f;
    float tau_cos_sidecar = 0.90f;
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
    __half* d_xhi_T = nullptr;      // overlay in P7 tail
    __half* d_xlo_T = nullptr;      // overlay in P7 tail
    uint16_t* d_xbhi_T = nullptr;   // overlay in P7 tail
    uint16_t* d_xblo_T = nullptr;   // overlay in P7 tail
    int4* d_xhm = nullptr;          // overlay in logits [10.4 MB .. 11.2 MB)
    int2* d_xlo = nullptr;          // overlay in logits [11.2 MB .. 11.6 MB)
    float* d_xsc = nullptr;         // overlay in logits [11.6 MB .. 11.7 MB)
    int32_t* d_xsq = nullptr;       // overlay in logits [11.7 MB .. 11.8 MB)
    int4* d_xhm2 = nullptr;         // overlay in logits [11.8 MB .. 12.6 MB)
    int2* d_xlo2 = nullptr;         // overlay in logits [12.6 MB .. 13.0 MB)
    float* d_xsc2 = nullptr;        // overlay in logits [13.0 MB .. 13.1 MB)
    int32_t* d_xsq2 = nullptr;      // overlay in logits [13.1 MB .. 13.2 MB)
    float* d_op = nullptr;          // overlay in logits [13.2 MB .. 15.0 MB)
    float* d_m = nullptr;           // overlay in logits [15.0 MB .. 15.1 MB)
    float* d_l = nullptr;           // overlay in logits [15.1 MB .. 15.2 MB)

    // Persistent P7 tail [16,777,216 .. 20,217,856) (never overwritten by logits!):
    float* d_hid = nullptr;         // 98,304 B
    float* d_h_pre24 = nullptr;     // 98,304 B (finalized L24 output for K>=2 / K=3 basis d0)
    float* d_h_b33 = nullptr;       // 98,304 B (finalized L33 pass-1 baseline state h1)
    float* d_h_p2 = nullptr;        // 98,304 B (finalized L33 pass-2 state h2 for K=3 basis d1)
    float* d_q_basis = nullptr;     // 196,608 B ([16 x 2 x 1536] DGS orthonormal basis [q0, q1])
    float* d_d2_raw = nullptr;      // 98,304 B (K=3 raw jump d2 = h3_raw - h2)
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
    float* d_sr_clean = nullptr;
    float* d_cos_clean = nullptr;
    int32_t* d_halt_flag = nullptr;
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
        d_h_b33   = reinterpret_cast<float*>(alloc_tail(16ULL * HIDDEN_SIZE * 4));
        d_h_p2    = reinterpret_cast<float*>(alloc_tail(16ULL * HIDDEN_SIZE * 4));
        d_q_basis = reinterpret_cast<float*>(alloc_tail(16ULL * 2ULL * HIDDEN_SIZE * 4));
        d_d2_raw  = reinterpret_cast<float*>(alloc_tail(16ULL * HIDDEN_SIZE * 4));
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
        d_sr_clean = reinterpret_cast<float*>(alloc_tail(256));
        d_cos_clean = reinterpret_cast<float*>(alloc_tail(256));
        d_halt_flag = reinterpret_cast<int32_t*>(alloc_tail(256));
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
            SlgbRuntimeLinearDesc d2{};
            d2.layer_idx = d1.layer_idx;
            d2.kind = d1.role_id;
            d2.out_features = static_cast<uint32_t>(d1.out_dim);
            d2.in_features = static_cast<uint32_t>(d1.in_dim);
            d2.offset_u4 = d1.words_byte_offset / sizeof(uint4);
            d2.offset_sc = (d1.scales_byte_offset - p1_words_bytes) / sizeof(uint16_t);
            linears.push_back(d2);
        }
        if (has_lm) {
            SlgbRuntimeLinearDesc d2_lm{};
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
        f.read(reinterpret_cast<char*>(linears.data()), linears.size() * sizeof(SlgbRuntimeLinearDesc));
        layers.resize(hdr.num_layers);
        f.read(reinterpret_cast<char*>(layers.data()), layers.size() * sizeof(SlgbRuntimeLayerDesc));

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

    bool load_u4_sidecar(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f.good()) {
            std::fprintf(stderr, "Failed opening U-4 sidecar: %s\n", path.c_str());
            return false;
        }
        const size_t mat_floats = size_t(9) * size_t(U4_RANK) * size_t(HIDDEN_SIZE);
        const size_t payload_floats = 3ULL + 9ULL + 2ULL * mat_floats;
        const size_t expected_bytes = 16ULL + payload_floats * sizeof(float);
        const std::streamoff fpos = f.tellg();
        if (fpos < 0 || static_cast<size_t>(fpos) != expected_bytes) {
            std::fprintf(stderr, "Invalid U-4 sidecar byte size: got=%lld expected=%zu\n",
                         static_cast<long long>(fpos), expected_bytes);
            return false;
        }
        f.seekg(0);
        std::vector<uint8_t> raw(expected_bytes);
        f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(expected_bytes));
        if (!f.good()) return false;
        uint32_t hdr4[4]{};
        std::memcpy(hdr4, raw.data(), 16);
        constexpr uint32_t SLU4_MAGIC = 0x34554C53u; // "SLU4"
        if (hdr4[0] != SLU4_MAGIC || hdr4[1] != 1u) {
            std::fprintf(stderr, "Invalid SLU4 sidecar header magic/version\n");
            return false;
        }
        const uint32_t calc_crc = siligemma::crc32_ieee(raw.data() + 16, expected_bytes - 16ULL);
        if (calc_crc != hdr4[3]) {
            std::fprintf(stderr, "SLU4 sidecar CRC32 mismatch: got=0x%08x exp=0x%08x\n", calc_crc, hdr4[3]);
            return false;
        }
        std::vector<float> payload(payload_floats);
        std::memcpy(payload.data(), raw.data() + 16, payload_floats * sizeof(float));
        for (float v : payload) {
            if (!std::isfinite(v)) {
                std::fprintf(stderr, "Non-finite value inside SLU4 sidecar\n");
                return false;
            }
        }
        loopcd_w_sidecar = payload[0];
        tau_sr_sidecar = payload[1];
        tau_cos_sidecar = payload[2];
        for (int k = 0; k < 9; ++k) u4_alpha[k] = payload[3 + k];
        const float* v_ptr = payload.data() + 12;
        const float* u_ptr = v_ptr + mat_floats;
        const size_t one_mat_bytes = size_t(U4_RANK) * size_t(HIDDEN_SIZE) * sizeof(float);
        for (int k = 0; k < 9; ++k) {
            if (d_u4_V16[k]) { CUDA_CHECK(cudaFree(d_u4_V16[k])); d_u4_V16[k] = nullptr; }
            if (d_u4_U16[k]) { CUDA_CHECK(cudaFree(d_u4_U16[k])); d_u4_U16[k] = nullptr; }
            CUDA_CHECK(cudaMalloc(&d_u4_V16[k], one_mat_bytes));
            CUDA_CHECK(cudaMalloc(&d_u4_U16[k], one_mat_bytes));
            CUDA_CHECK(cudaMemcpy(d_u4_V16[k], v_ptr + size_t(k) * U4_RANK * HIDDEN_SIZE, one_mat_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_u4_U16[k], u_ptr + size_t(k) * U4_RANK * HIDDEN_SIZE, one_mat_bytes, cudaMemcpyHostToDevice));
        }
        hdr.has_u4_loop = 1u;
        hdr.u4_gate_f32_count = 9ULL * (2ULL * U4_RANK * HIDDEN_SIZE + 1ULL);
        return true;
    }

    bool load(const std::string& path, int arena_max_b, int arena_t_cap, int bf16_kv, const std::string& u4_sidecar_path = "") {
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
        if (!ok || h_u4.size() != h_sc.size()) {
            std::fprintf(stderr, "Failed parsing .slgb (version=%u)\n", magic_ver[1]);
            return false;
        }
        total_u4_count = h_u4.size();
        total_sc_count = h_sc.size();

        // Compute out-of-band full Hamming(72, 64) ECC bytes for every 64-bit half of every 32-weight uint4 group
        std::vector<uint32_t> h_sc_ecc(h_sc.size());
        for (size_t i = 0; i < h_u4.size(); ++i) {
            const uint64_t lo = static_cast<uint64_t>(h_u4[i].x) | (static_cast<uint64_t>(h_u4[i].y) << 32);
            const uint64_t hi = static_cast<uint64_t>(h_u4[i].z) | (static_cast<uint64_t>(h_u4[i].w) << 32);
            const uint8_t ecc0 = siligemma::secded72_encode_cpu(lo);
            const uint8_t ecc1 = siligemma::secded72_encode_cpu(hi);
            h_sc_ecc[i] = static_cast<uint32_t>(h_sc[i]) |
                          (static_cast<uint32_t>(ecc0) << 16) |
                          (static_cast<uint32_t>(ecc1) << 24);
        }

        CUDA_CHECK(cudaMalloc(&d_w_u4, h_u4.size() * sizeof(uint4)));
        CUDA_CHECK(cudaMemcpy(d_w_u4, h_u4.data(), h_u4.size() * sizeof(uint4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_w_sc_ecc, h_sc_ecc.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_w_sc_ecc, h_sc_ecc.data(), h_sc_ecc.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
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
        if (!u4_sidecar_path.empty()) {
            if (!load_u4_sidecar(u4_sidecar_path)) return false;
        } else if (!h_u4g.empty()) {
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
        if (d_w_u4) { cudaFree(d_w_u4); d_w_u4 = nullptr; }
        if (d_w_sc_ecc) { cudaFree(d_w_sc_ecc); d_w_sc_ecc = nullptr; }
        if (d_norms) { cudaFree(d_norms); d_norms = nullptr; }
        if (d_inv_slide) { cudaFree(d_inv_slide); d_inv_slide = nullptr; }
        if (d_inv_full) { cudaFree(d_inv_full); d_inv_full = nullptr; }
        if (d_lm_bf16) { cudaFree(d_lm_bf16); d_lm_bf16 = nullptr; }
        for (int k = 0; k < 9; ++k) {
            if (d_u4_U16[k]) { cudaFree(d_u4_U16[k]); d_u4_U16[k] = nullptr; }
            if (d_u4_V16[k]) { cudaFree(d_u4_V16[k]); d_u4_V16[k] = nullptr; }
        }
        if (d_wb_U8) { cudaFree(d_wb_U8); d_wb_U8 = nullptr; }
        if (d_wb_V8) { cudaFree(d_wb_V8); d_wb_V8 = nullptr; }
        if (d_p5) { cudaFree(d_p5); d_p5 = nullptr; }
        if (d_p7) { cudaFree(d_p7); d_p7 = nullptr; }
    }

    void finalize_to_buf(
        cudaStream_t st, int B, float* dst_buf, int prev_active, int prev_n_sp,
        const float* prev_post_ple_nw, float prev_scalar
    ) {
        if (!prev_active) {
            if (dst_buf != d_hid) {
                CUDA_CHECK(cudaMemcpyAsync(dst_buf, d_hid, size_t(B) * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice, st));
            }
            return;
        }
        if (B <= 1) {
            v2::stageA_norm_res_v2_sm86<<<1, 128, 0, st>>>(
                d_tmp1, prev_n_sp, d_hid, prev_post_ple_nw, prev_scalar, nullptr, dst_buf, d_scr, HIDDEN_SIZE, RMS_NORM_EPS);
        } else {
            v2::norm_b16_smem_stageA_sm86<<<B, 128, 0, st>>>(
                d_tmp1, prev_n_sp, d_hid, prev_post_ple_nw, prev_scalar, nullptr, dst_buf, d_scr, B, HIDDEN_SIZE, RMS_NORM_EPS);
        }
    }

    void emit_embed_and_ple(cudaStream_t st, int B) {
        const SlgbRuntimeLinearDesc& ple_mod = linears[275];
        const int use_mma = (B > 1);
        if (d_lm_bf16) {
            embed_lookup_bf16_sm86<<<dim3((HIDDEN_SIZE + 127) / 128, B), 128, 0, st>>>(
                d_lm_bf16, d_tok_ids, d_hid, hdr.embed_scale, B, HIDDEN_SIZE);
        } else if (linears.size() > 276) {
            const auto& llm = linears[276];
            embed_lookup_g32_sm86<<<dim3((HIDDEN_SIZE >> 5) / 4, B), 128, 0, st>>>(
                d_w_u4 + llm.offset_u4, d_w_sc_ecc + llm.offset_sc, d_tok_ids, d_hid, d_can, hdr.embed_scale, B, HIDDEN_SIZE);
        } else {
            std::fprintf(stderr, "FATAL: neither BF16 embed nor g32 lm_head shadow present\n");
            std::exit(1);
        }

        if (!use_mma) {
            v2::stageA_norm_res_v2_sm86<<<1, 128, 0, st>>>(
                d_hid, 1, nullptr, nullptr, 1.0f, nullptr, nullptr, d_scr, HIDDEN_SIZE, RMS_NORM_EPS);
            stageB_parallel_h4_q22_sm86<<<(HIDDEN_SIZE >> 5) / 4, 128, 0, st>>>(
                d_scr, nullptr, nullptr, d_xhm, d_xlo, d_xsc, d_xsq, 1, HIDDEN_SIZE, 1);
            v2::gemv_b1_dp4a22_2row_smem_sk_sm86<8, 1><<<dim3((NUM_LAYERS * PLE_DIM + 31) / 32, 1), 128, 0, st>>>(
                d_xhm, d_xlo, d_xsc, d_xsq,
                d_w_u4 + ple_mod.offset_u4, d_w_sc_ecc + ple_mod.offset_sc,
                d_tmp1, d_can, 0.0f, NUM_LAYERS * PLE_DIM, HIDDEN_SIZE);
        } else {
            v2::norm_b16_smem_stageA_sm86<<<B, 128, 0, st>>>(
                d_hid, 1, nullptr, nullptr, 1.0f, nullptr, nullptr, d_scr, B, HIDDEN_SIZE, RMS_NORM_EPS);
            v2::h4_fp16split_b16_coalesced_sm86<<<HIDDEN_SIZE >> 5, 128, 0, st>>>(d_scr, d_xhi_T, d_xlo_T, B, HIDDEN_SIZE);
            v2::gemm_b16_hmma_64x16_k64_sk_sm86<1><<<(NUM_LAYERS * PLE_DIM + 63) / 64, 128, 0, st>>>(
                d_xhi_T, d_xlo_T, d_w_u4 + ple_mod.offset_u4, d_w_sc_ecc + ple_mod.offset_sc,
                d_tmp1, d_can, 0.0f, B, NUM_LAYERS * PLE_DIM, HIDDEN_SIZE);
        }
        ple_combine_tok_ctx_sm86<<<dim3(NUM_LAYERS, B), 128, 0, st>>>(
            d_tmp1, d_norms + ple_norm_off, d_tok_ple, d_ple_in, hdr.ple_model_proj_scale, B, NUM_LAYERS, PLE_DIM, RMS_NORM_EPS);
    }

    void emit_single_layer(
        cudaStream_t st, int l, int loop_pass, int B, int use_split_t, int wb_enable,
        float& prev_scalar, const float*& prev_post_ple_nw, int& prev_active, int& prev_n_sp
    ) {
        const int use_mma = (B > 1);
        auto norm_b1_v2 = [&](const float* vp, int nsp, const float* rin, const float* nwp, float scp, const float* nwpre, float* rout) {
            v2::stageA_norm_res_v2_sm86<<<1, 128, 0, st>>>(vp, nsp, rin, nwp, scp, nwpre, rout, d_scr, HIDDEN_SIZE, RMS_NORM_EPS);
            stageB_parallel_h4_q22_sm86<<<(HIDDEN_SIZE >> 5) / 4, 128, 0, st>>>(
                d_scr, nullptr, nullptr, d_xhm, d_xlo, d_xsc, d_xsq, 1, HIDDEN_SIZE, 1);
        };
        auto norm_b16_v2 = [&](const float* vp, int nsp, const float* rin, const float* nwp, float scp, const float* nwpre, float* rout, int D) {
            v2::norm_b16_smem_stageA_sm86<<<B, 128, 0, st>>>(vp, nsp, rin, nwp, scp, nwpre, rout, d_scr, B, D, RMS_NORM_EPS);
            v2::h4_fp16split_b16_coalesced_sm86<<<D >> 5, 128, 0, st>>>(d_scr, d_xhi_T, d_xlo_T, B, D);
        };

        const SlgbRuntimeLayerDesc& L = layers[l];
        const int hd = int(L.head_dim);
        const int interm = int(L.interm_size);
        const int sh = int(L.is_kv_shared);
        const int kv_ro = sh || (loop_pass > 0);
        const int sw = L.sliding_window;
        const float* inv_f = L.is_full_attn ? d_inv_full : d_inv_slide;
        void* kc = static_cast<void*>(d_k_slots[L.kv_slot]);
        void* vc = static_cast<void*>(d_v_slots[L.kv_slot]);

        if (!use_mma) {
            norm_b1_v2(prev_active ? d_tmp1 : d_hid, prev_active ? prev_n_sp : 1,
                       prev_active ? d_hid : nullptr, prev_active ? prev_post_ple_nw : nullptr, prev_active ? prev_scalar : 1.0f,
                       d_norms + L.off_in_nw, d_hid);

            const auto& lq = linears[L.lin_q];
            if (!kv_ro) {
                const auto& lk = linears[L.lin_k];
                const auto& lv = linears[L.lin_v];
                gemv_b1_qkv_fused_sm86<<<(NUM_HEADS * hd + 2 * NUM_KV_HEADS * hd + 15) / 16, 128, 0, st>>>(
                    d_xhm, d_xlo, d_xsc, d_xsq,
                    d_w_u4 + lq.offset_u4, d_w_sc_ecc + lq.offset_sc, NUM_HEADS * hd, d_tmp1,
                    d_w_u4 + lk.offset_u4, d_w_sc_ecc + lk.offset_sc, NUM_KV_HEADS * hd, d_tmp2,
                    d_w_u4 + lv.offset_u4, d_w_sc_ecc + lv.offset_sc, NUM_KV_HEADS * hd, d_tmp3,
                    d_can, HIDDEN_SIZE);
            } else {
                gemv_b1_dp4a22_sm86<16, 1><<<(NUM_HEADS * hd) / 8, 128, 0, st>>>(
                    d_xhm, d_xlo, d_xsc, d_xsq,
                    d_w_u4 + lq.offset_u4, d_w_sc_ecc + lq.offset_sc, d_tmp1, d_can, 0.0f, NUM_HEADS * hd, HIDDEN_SIZE);
            }
            qkv_norm_rope_kv_write_sm86<<<dim3(NUM_HEADS + (kv_ro ? 0 : 2), 1), 128, 0, st>>>(
                d_tmp1, d_tmp2, d_tmp3,
                d_norms + L.off_qn_w, kv_ro ? nullptr : (d_norms + L.off_kn_w), inv_f,
                kc, vc, d_pos, 1, NUM_HEADS, NUM_KV_HEADS, hd, slot_t_cap[L.kv_slot], RMS_NORM_EPS, kv_ro, kv_is_bf16);

            if (use_split_t) {
                attention_kv_split_sm86<16><<<dim3(NUM_HEADS, 16), 128, 0, st>>>(
                    d_tmp1, kc, vc, d_op, d_m, d_l, d_pos, 1, NUM_HEADS, NUM_KV_HEADS, hd, slot_t_cap[L.kv_slot], sw, kv_is_bf16);
                attention_merge_h4_q22_fast_sm86<16><<<dim3((NUM_HEADS * hd >> 5) / 4, 1), 128, 0, st>>>(
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
                d_xhm, d_xlo, d_xsc, d_xsq, d_w_u4 + lo.offset_u4, d_w_sc_ecc + lo.offset_sc, d_tmp1, d_can, 0.0f, HIDDEN_SIZE, NUM_HEADS * hd);
            norm_b1_v2(d_tmp1, 4, d_hid, d_norms + L.off_pa_nw, 1.0f, d_norms + L.off_pf_nw, d_hid);

            const auto& lg = linears[L.lin_gate];
            const auto& lu = linears[L.lin_up];
            const auto& ld = linears[L.lin_down];
            v2::gemv_b1_geglu_h4_q22_smem_sm86<<<interm / 32, 256, 0, st>>>(
                d_xhm, d_xlo, d_xsc, d_xsq,
                d_w_u4 + lg.offset_u4, d_w_sc_ecc + lg.offset_sc,
                d_w_u4 + lu.offset_u4, d_w_sc_ecc + lu.offset_sc,
                d_xhm2, d_xlo2, d_xsc2, d_xsq2, d_can, interm, HIDDEN_SIZE);
            v2::gemv_b1_dp4a22_2row_smem_sk_sm86<8, 8><<<dim3(HIDDEN_SIZE / 32, 8), 128, 0, st>>>(
                d_xhm2, d_xlo2, d_xsc2, d_xsq2, d_w_u4 + ld.offset_u4, d_w_sc_ecc + ld.offset_sc, d_tmp1, d_can, 0.0f, HIDDEN_SIZE, interm);

            norm_b1_v2(d_tmp1, 8, d_hid, d_norms + L.off_pff_nw, 1.0f, nullptr, d_hid);

            const auto& lpg = linears[L.lin_ple_gate];
            const auto& lpp = linears[L.lin_ple_proj];
            gemv_b1_dp4a22_sm86<16, 6><<<dim3(PLE_DIM / 8, 6), 128, 0, st>>>(
                d_xhm, d_xlo, d_xsc, d_xsq, d_w_u4 + lpg.offset_u4, d_w_sc_ecc + lpg.offset_sc, d_tmp1, d_can, 0.0f, PLE_DIM, HIDDEN_SIZE);
            fused_ple_act_256_h4_q22_sm86<<<1, 256, 0, st>>>(
                d_tmp1, 6, d_ple_in + l * PLE_DIM, d_xhm, d_xlo, d_xsc, d_xsq, 1, PLE_DIM);
            gemv_b1_dp4a22_sm86<8, 1><<<HIDDEN_SIZE / 16, 128, 0, st>>>(
                d_xhm, d_xlo, d_xsc, d_xsq, d_w_u4 + lpp.offset_u4, d_w_sc_ecc + lpp.offset_sc, d_tmp1, d_can, 0.0f, HIDDEN_SIZE, PLE_DIM);
            prev_n_sp = 1;
        } else {
            norm_b16_v2(prev_active ? d_tmp1 : d_hid, prev_active ? prev_n_sp : 1,
                        prev_active ? d_hid : nullptr, prev_active ? prev_post_ple_nw : nullptr, prev_active ? prev_scalar : 1.0f,
                        d_norms + L.off_in_nw, d_hid, HIDDEN_SIZE);
            const auto& lq = linears[L.lin_q];
            if (!kv_ro) {
                const auto& lk = linears[L.lin_k];
                const auto& lv = linears[L.lin_v];
                int nblks = (NUM_HEADS * hd + 63) / 64 + 2 * ((NUM_KV_HEADS * hd + 63) / 64);
                v2::gemm_b16_hmma_64x16_k64_qkv_sm86<<<nblks, 128, 0, st>>>(
                    d_xhi_T, d_xlo_T,
                    d_w_u4 + lq.offset_u4, d_w_sc_ecc + lq.offset_sc, NUM_HEADS * hd, d_tmp1,
                    d_w_u4 + lk.offset_u4, d_w_sc_ecc + lk.offset_sc, NUM_KV_HEADS * hd, d_tmp2,
                    d_w_u4 + lv.offset_u4, d_w_sc_ecc + lv.offset_sc, NUM_KV_HEADS * hd, d_tmp3,
                    d_can, B, HIDDEN_SIZE);
            } else {
                v2::gemm_b16_hmma_64x16_k64_sk_sm86<1><<<(NUM_HEADS * hd + 63) / 64, 128, 0, st>>>(
                    d_xhi_T, d_xlo_T, d_w_u4 + lq.offset_u4, d_w_sc_ecc + lq.offset_sc, d_tmp1, d_can, 0.0f, B, NUM_HEADS * hd, HIDDEN_SIZE);
            }
            qkv_norm_rope_kv_write_sm86<<<dim3(NUM_HEADS + (kv_ro ? 0 : 2), B), 128, 0, st>>>(
                d_tmp1, d_tmp2, d_tmp3, d_norms + L.off_qn_w, kv_ro ? nullptr : (d_norms + L.off_kn_w), inv_f,
                kc, vc, d_pos, B, NUM_HEADS, NUM_KV_HEADS, hd, slot_t_cap[L.kv_slot], RMS_NORM_EPS, kv_ro, kv_is_bf16);
            attention_kv_split_sm86<1><<<dim3(NUM_HEADS, B), 128, 0, st>>>(
                d_tmp1, kc, vc, d_tmp2, nullptr, nullptr, d_pos, B, NUM_HEADS, NUM_KV_HEADS, hd, slot_t_cap[L.kv_slot], sw, kv_is_bf16);
            v2::h4_fp16split_b16_coalesced_sm86<<<(NUM_HEADS * hd) >> 5, 128, 0, st>>>(
                d_tmp2, d_xhi_T, d_xlo_T, B, NUM_HEADS * hd);
            const auto& lo = linears[L.lin_o];
            v2::gemm_b16_hmma_64x16_k64_sk_sm86<4><<<dim3(HIDDEN_SIZE / 64, 4), 128, 0, st>>>(
                d_xhi_T, d_xlo_T, d_w_u4 + lo.offset_u4, d_w_sc_ecc + lo.offset_sc, d_tmp1, d_can, 0.0f, B, HIDDEN_SIZE, NUM_HEADS * hd);
            norm_b16_v2(d_tmp1, 4, d_hid, d_norms + L.off_pa_nw, 1.0f, d_norms + L.off_pf_nw, d_hid, HIDDEN_SIZE);
            const auto& lg = linears[L.lin_gate];
            const auto& lu = linears[L.lin_up];
            const auto& ld = linears[L.lin_down];
            v2::gemm_b16_hmma_64x16_k64_pair_sm86<<<(interm / 64) * 2, 128, 0, st>>>(
                d_xhi_T, d_xlo_T, d_w_u4 + lg.offset_u4, d_w_sc_ecc + lg.offset_sc, d_tmp2,
                d_w_u4 + lu.offset_u4, d_w_sc_ecc + lu.offset_sc, d_tmp3, d_can, B, interm, HIDDEN_SIZE);
            v2::fused_geglu_or_ple_b16_h4_fp16split_v2_sm86<<<interm >> 5, 128, 0, st>>>(
                d_tmp2, 1, d_tmp3, d_xhi_T, d_xlo_T, B, interm, interm, 0);
            v2::gemm_b16_hmma_64x16_k64_sk_sm86<4><<<dim3(HIDDEN_SIZE / 64, 4), 128, 0, st>>>(
                d_xhi_T, d_xlo_T, d_w_u4 + ld.offset_u4, d_w_sc_ecc + ld.offset_sc, d_tmp1, d_can, 0.0f, B, HIDDEN_SIZE, interm);
            norm_b16_v2(d_tmp1, 4, d_hid, d_norms + L.off_pff_nw, 1.0f, nullptr, d_hid, HIDDEN_SIZE);
            const auto& lpg = linears[L.lin_ple_gate];
            const auto& lpp = linears[L.lin_ple_proj];
            v2::gemm_b16_hmma_64x16_k64_sk_sm86<12><<<dim3(PLE_DIM / 64, 12), 128, 0, st>>>(
                d_xhi_T, d_xlo_T, d_w_u4 + lpg.offset_u4, d_w_sc_ecc + lpg.offset_sc, d_tmp2, d_can, 0.0f, B, PLE_DIM, HIDDEN_SIZE);
            v2::fused_geglu_or_ple_b16_h4_fp16split_v2_sm86<<<PLE_DIM >> 5, 128, 0, st>>>(
                d_tmp2, 12, d_ple_in, d_xhi_T, d_xlo_T, B, PLE_DIM, NUM_LAYERS * PLE_DIM, l * PLE_DIM);
            v2::gemm_b16_hmma_64x16_k64_sk_sm86<2><<<dim3(HIDDEN_SIZE / 64, 2), 128, 0, st>>>(
                d_xhi_T, d_xlo_T, d_w_u4 + lpp.offset_u4, d_w_sc_ecc + lpp.offset_sc, d_tmp1, d_can, 0.0f, B, HIDDEN_SIZE, PLE_DIM);
            prev_n_sp = 2;
        }

        // Optional Woodbury Cayley steering on L10..L14 hub
        if (wb_enable && l >= WOODBURY_HUB_START && l <= WOODBURY_HUB_END && d_wb_U8 && d_wb_V8) {
            finalize_to_buf(st, B, d_hid, 1, prev_n_sp, d_norms + L.off_ppl_nw, L.layer_scalar);
            cayley_woodbury_apply_sm86<<<B, 128, 0, st>>>(d_hid, d_wb_U8, d_wb_V8, -0.02f, B, HIDDEN_SIZE);
            prev_active = 0;
            prev_n_sp = 1;
            return;
        }
        prev_post_ple_nw = d_norms + L.off_ppl_nw;
        prev_scalar = L.layer_scalar;
        prev_active = 1;

        // U-4 low-rank r=16 controller applied AFTER layer l post_ple_norm + layer_scalar
        if (loop_pass > 0 && l >= U4_LOOP_START && l <= U4_LOOP_END && d_u4_U16[l - U4_LOOP_START]) {
            finalize_to_buf(st, B, d_hid, prev_active, prev_n_sp, prev_post_ple_nw, prev_scalar);
            const int k = l - U4_LOOP_START;
            u4_loop_proj_r16_sm86<<<dim3(U4_RANK, B), 128, 0, st>>>(d_hid, d_u4_V16[k], d_u4_z, B, HIDDEN_SIZE);
            u4_loop_apply_r16_sm86<<<dim3((HIDDEN_SIZE + 127) / 128, B), 128, 0, st>>>(d_hid, d_u4_z, d_u4_U16[k], u4_alpha[k], B, HIDDEN_SIZE);
            prev_active = 0;
            prev_n_sp = 1;
        }
    }

    void emit_head_and_canary(
        cudaStream_t st, int B, int use_lm_bf16, float softcap,
        int prev_active, int prev_n_sp, const float* prev_post_ple_nw, float prev_scalar
    ) {
        const int use_mma = (B > 1);
        if (!use_mma) {
            if (!use_lm_bf16 && hdr.has_lm_head_g32 && linears.size() > 276) {
                v2::stageA_norm_res_v2_sm86<<<1, 128, 0, st>>>(
                    prev_active ? d_tmp1 : d_hid, prev_active ? prev_n_sp : 1,
                    prev_active ? d_hid : nullptr, prev_active ? prev_post_ple_nw : nullptr, prev_active ? prev_scalar : 1.0f,
                    d_norms + final_norm_off, d_hid, d_scr, HIDDEN_SIZE, RMS_NORM_EPS);
                stageB_parallel_h4_q22_sm86<<<(HIDDEN_SIZE >> 5) / 4, 128, 0, st>>>(
                    d_scr, nullptr, nullptr, d_xhm, d_xlo, d_xsc, d_xsq, 1, HIDDEN_SIZE, 1);
                const auto& llm = linears[276];
                v2::gemv_b1_dp4a22_2row_smem_sk_sm86<8, 1><<<dim3((VOCAB_SIZE + 31) / 32, 1), 128, 0, st>>>(
                    d_xhm, d_xlo, d_xsc, d_xsq, d_w_u4 + llm.offset_u4, d_w_sc_ecc + llm.offset_sc,
                    d_logits, d_can, softcap, VOCAB_SIZE, HIDDEN_SIZE);
            } else {
                if (!d_lm_bf16) {
                    std::fprintf(stderr, "FATAL: BF16 lm_head unavailable\n");
                    std::exit(1);
                }
                v2::stageA_norm_res_v2_sm86<<<1, 128, 0, st>>>(
                    prev_active ? d_tmp1 : d_hid, prev_active ? prev_n_sp : 1,
                    prev_active ? d_hid : nullptr, prev_active ? prev_post_ple_nw : nullptr, prev_active ? prev_scalar : 1.0f,
                    d_norms + final_norm_off, d_hid, d_tmp1, HIDDEN_SIZE, RMS_NORM_EPS);
                lm_head_b1_bf16_uint4_sm86<<<(VOCAB_SIZE + 31) / 32, 128, 0, st>>>(
                    d_tmp1, reinterpret_cast<const uint4*>(d_lm_bf16), d_logits, softcap, VOCAB_SIZE, HIDDEN_SIZE);
            }
            parallel_argmax_stage1_sm86<<<128, 128, 0, st>>>(d_logits, d_bv, d_bi, VOCAB_SIZE);
            parallel_argmax_stage2_sm86<<<1, 128, 0, st>>>(d_bv, d_bi, d_out_id, 128);
        } else {
            if (!use_lm_bf16 && hdr.has_lm_head_g32 && linears.size() > 276) {
                v2::norm_b16_smem_stageA_sm86<<<B, 128, 0, st>>>(
                    prev_active ? d_tmp1 : d_hid, prev_active ? prev_n_sp : 1,
                    prev_active ? d_hid : nullptr, prev_active ? prev_post_ple_nw : nullptr, prev_active ? prev_scalar : 1.0f,
                    d_norms + final_norm_off, d_hid, d_scr, B, HIDDEN_SIZE, RMS_NORM_EPS);
                v2::h4_fp16split_b16_coalesced_sm86<<<HIDDEN_SIZE >> 5, 128, 0, st>>>(d_scr, d_xhi_T, d_xlo_T, B, HIDDEN_SIZE);
                const auto& llm = linears[276];
                v2::gemm_b16_hmma_64x16_k64_sk_sm86<1><<<(VOCAB_SIZE + 63) / 64, 128, 0, st>>>(
                    d_xhi_T, d_xlo_T, d_w_u4 + llm.offset_u4, d_w_sc_ecc + llm.offset_sc, d_logits, d_can, softcap, B, VOCAB_SIZE, HIDDEN_SIZE);
            } else {
                if (!d_lm_bf16) {
                    std::fprintf(stderr, "FATAL: BF16 lm_head unavailable\n");
                    std::exit(1);
                }
                fused_norm_b16_h4_fp16split_sm86<<<B, 128, 0, st>>>(
                    prev_active ? d_tmp1 : d_hid, prev_active ? prev_n_sp : 1,
                    prev_active ? d_hid : nullptr, prev_active ? prev_post_ple_nw : nullptr, prev_active ? prev_scalar : 1.0f,
                    d_norms + final_norm_off, d_hid, d_xhi_T, d_xlo_T, d_xbhi_T, d_xblo_T, d_scr, B, HIDDEN_SIZE, RMS_NORM_EPS, 0, 1);
                lm_head_b16_bf16_mma_sm86<<<(VOCAB_SIZE + 31) / 32, 128, 0, st>>>(
                    d_xbhi_T, d_xblo_T, reinterpret_cast<const uint4*>(d_lm_bf16), d_logits, softcap, B, VOCAB_SIZE, HIDDEN_SIZE);
            }
        }
        iris_canary_finalize_sm86<<<1, 128, 0, st>>>(d_hid, d_can, B, HIDDEN_SIZE);
    }

    void emit_step(cudaStream_t st, int B, int K_passes, int use_lm_bf16, int use_split_t, int wb_enable, float softcap, float loopcd_w = 0.0f) {
        emit_embed_and_ple(st, B);

        float prev_scalar = 1.0f;
        const float* prev_post_ple_nw = nullptr;
        int prev_active = 0;
        int prev_n_sp = 1;

        for (int l = 0; l < int(hdr.num_layers); ++l) {
            if (K_passes >= 2 && l == U4_LOOP_START) {
                finalize_to_buf(st, B, d_h_pre24, prev_active, prev_n_sp, prev_post_ple_nw, prev_scalar);
            }
            emit_single_layer(st, l, 0, B, use_split_t, wb_enable, prev_scalar, prev_post_ple_nw, prev_active, prev_n_sp);
            if (K_passes >= 2 && l == U4_LOOP_END) {
                finalize_to_buf(st, B, d_h_b33, prev_active, prev_n_sp, prev_post_ple_nw, prev_scalar);
                CUDA_CHECK(cudaMemcpyAsync(d_hid, d_h_b33, size_t(B) * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice, st));
                prev_active = 0;
                prev_n_sp = 1;
                for (int ll = U4_LOOP_START; ll <= U4_LOOP_END; ++ll) {
                    emit_single_layer(st, ll, 1, B, use_split_t, wb_enable, prev_scalar, prev_post_ple_nw, prev_active, prev_n_sp);
                }
                finalize_to_buf(st, B, d_hid, prev_active, prev_n_sp, prev_post_ple_nw, prev_scalar);
                prev_active = 0;
                prev_n_sp = 1;
                if (K_passes == 2) {
                    if (loopcd_w != 0.0f) {
                        v2::loopcd_hidden_extrap_f32_sm86<<<dim3((HIDDEN_SIZE + 127) / 128, B), 128, 0, st>>>(
                            d_hid, d_h_b33, loopcd_w, B, HIDDEN_SIZE);
                    }
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(d_h_p2, d_hid, size_t(B) * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice, st));
                    for (int ll = U4_LOOP_START; ll <= U4_LOOP_END; ++ll) {
                        emit_single_layer(st, ll, 2, B, use_split_t, wb_enable, prev_scalar, prev_post_ple_nw, prev_active, prev_n_sp);
                    }
                    finalize_to_buf(st, B, d_hid, prev_active, prev_n_sp, prev_post_ple_nw, prev_scalar);
                    prev_active = 0;
                    prev_n_sp = 1;
                    v2::build_dgs_basis2_sm86<<<B, 128, 0, st>>>(d_h_pre24, d_h_b33, d_h_p2, d_q_basis, d_d2_raw, d_hid, B, HIDDEN_SIZE);
                    for (int b = 0; b < B; ++b) {
                        v2::riccati_cayley_extrap_f32_sm86<<<1, 128, 0, st>>>(
                            d_d2_raw + size_t(b) * HIDDEN_SIZE, d_q_basis + size_t(b) * 2ULL * HIDDEN_SIZE, 2, 1, HIDDEN_SIZE);
                    }
                    v2::riccati_damped_add_sm86<<<dim3((HIDDEN_SIZE + 127) / 128, B), 128, 0, st>>>(
                        d_hid, d_h_p2, d_d2_raw, 0.45f, B, HIDDEN_SIZE);
                }
            }
        }

        emit_head_and_canary(st, B, use_lm_bf16, softcap, prev_active, prev_n_sp, prev_post_ple_nw, prev_scalar);
    }

    void emit_b1_adaptive_prefix(cudaStream_t st, int use_split_t, int wb_enable, float tau_sr, float tau_cos) {
        emit_embed_and_ple(st, 1);
        float prev_scalar = 1.0f;
        const float* prev_post_ple_nw = nullptr;
        int prev_active = 0;
        int prev_n_sp = 1;
        for (int l = 0; l <= U4_LOOP_END; ++l) {
            if (l == U4_LOOP_START) {
                finalize_to_buf(st, 1, d_h_pre24, prev_active, prev_n_sp, prev_post_ple_nw, prev_scalar);
            }
            emit_single_layer(st, l, 0, 1, use_split_t, wb_enable, prev_scalar, prev_post_ple_nw, prev_active, prev_n_sp);
        }
        finalize_to_buf(st, 1, d_h_b33, prev_active, prev_n_sp, prev_post_ple_nw, prev_scalar);
        v2::p233c_outlier_safe_halt_sm86<<<1, 128, 0, st>>>(
            d_h_pre24, d_h_b33, tau_sr, tau_cos, d_sr_clean, d_cos_clean, d_halt_flag, 1, HIDDEN_SIZE);
    }

    void emit_b1_adaptive_k1_tail(cudaStream_t st, int use_lm_bf16, int use_split_t, int wb_enable, float softcap) {
        CUDA_CHECK(cudaMemcpyAsync(d_hid, d_h_b33, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice, st));
        float prev_scalar = 1.0f;
        const float* prev_post_ple_nw = nullptr;
        int prev_active = 0;
        int prev_n_sp = 1;
        emit_single_layer(st, NUM_LAYERS - 1, 0, 1, use_split_t, wb_enable, prev_scalar, prev_post_ple_nw, prev_active, prev_n_sp);
        emit_head_and_canary(st, 1, use_lm_bf16, softcap, prev_active, prev_n_sp, prev_post_ple_nw, prev_scalar);
    }

    void emit_b1_adaptive_k2_tail(cudaStream_t st, int use_lm_bf16, int use_split_t, int wb_enable, float softcap, float loopcd_w) {
        CUDA_CHECK(cudaMemcpyAsync(d_hid, d_h_b33, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice, st));
        float prev_scalar = 1.0f;
        const float* prev_post_ple_nw = nullptr;
        int prev_active = 0;
        int prev_n_sp = 1;
        for (int ll = U4_LOOP_START; ll <= U4_LOOP_END; ++ll) {
            emit_single_layer(st, ll, 1, 1, use_split_t, wb_enable, prev_scalar, prev_post_ple_nw, prev_active, prev_n_sp);
        }
        finalize_to_buf(st, 1, d_hid, prev_active, prev_n_sp, prev_post_ple_nw, prev_scalar);
        prev_active = 0;
        prev_n_sp = 1;
        if (loopcd_w != 0.0f) {
            v2::loopcd_hidden_extrap_f32_sm86<<<dim3((HIDDEN_SIZE + 127) / 128, 1), 128, 0, st>>>(
                d_hid, d_h_b33, loopcd_w, 1, HIDDEN_SIZE);
        }
        emit_single_layer(st, NUM_LAYERS - 1, 0, 1, use_split_t, wb_enable, prev_scalar, prev_post_ple_nw, prev_active, prev_n_sp);
        emit_head_and_canary(st, 1, use_lm_bf16, softcap, prev_active, prev_n_sp, prev_post_ple_nw, prev_scalar);
    }
};

struct BenchConfig {
    bool cpu_selftest = false;
    bool gpu_selftest = false;
    std::string slgb_path;
    std::string u4_sidecar_path;
    std::string prompt_ids_str = "2,107,108";
    std::string ple_tok_bin;
    std::string dump_logits_path;
    std::string out_json_path;
    int dump_b = 1;
    int dump_k = 1;
    int dump_lm_bf16 = 0;
    bool bench_all = false;
    bool bench_b1_fast = false;
    bool allow_no_bos_test = false;
    float loopcd_w = 0.0f;
    bool loopcd_w_set = false;
    float tau_sr = 0.25f;
    bool tau_sr_set = false;
    float tau_cos = 0.90f;
    bool tau_cos_set = false;
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
        else if (a == "--u4-sidecar") next(cfg.u4_sidecar_path);
        else if (a == "--prompt-ids") next(cfg.prompt_ids_str);
        else if (a == "--ple-tok-bin") next(cfg.ple_tok_bin);
        else if (a == "--dump-logits") next(cfg.dump_logits_path);
        else if (a == "--out-json") next(cfg.out_json_path);
        else if (a == "--dump-b") {
            std::string v; next(v);
            if (!parse_int_strict(v, 1, 16, cfg.dump_b)) { std::fprintf(stderr, "Invalid --dump-b: %s\n", v.c_str()); std::exit(2); }
        } else if (a == "--dump-k") {
            std::string v; next(v);
            if (!parse_int_strict(v, 1, 3, cfg.dump_k)) { std::fprintf(stderr, "Invalid --dump-k: %s\n", v.c_str()); std::exit(2); }
        } else if (a == "--dump-lm-bf16") {
            std::string v; next(v);
            if (!parse_int_strict(v, 0, 1, cfg.dump_lm_bf16)) { std::fprintf(stderr, "Invalid --dump-lm-bf16: %s\n", v.c_str()); std::exit(2); }
        } else if (a == "--loopcd-w") {
            std::string v; next(v);
            if (!parse_float_strict(v, cfg.loopcd_w)) { std::fprintf(stderr, "Invalid --loopcd-w: %s\n", v.c_str()); std::exit(2); }
            cfg.loopcd_w_set = true;
        } else if (a == "--tau-sr") {
            std::string v; next(v);
            if (!parse_float_strict(v, cfg.tau_sr) || cfg.tau_sr <= 0.0f) { std::fprintf(stderr, "Invalid --tau-sr: %s\n", v.c_str()); std::exit(2); }
            cfg.tau_sr_set = true;
        } else if (a == "--tau-cos") {
            std::string v; next(v);
            if (!parse_float_strict(v, cfg.tau_cos)) { std::fprintf(stderr, "Invalid --tau-cos: %s\n", v.c_str()); std::exit(2); }
            cfg.tau_cos_set = true;
        } else if (a == "--bench-all") cfg.bench_all = true;
        else if (a == "--bench-b1-fast") cfg.bench_b1_fast = true;
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
    if (!model.load(cfg.slgb_path, 16, 1024, 0, cfg.u4_sidecar_path)) {
        std::fprintf(stderr, "Failed to load .slgb / sidecar: %s\n", cfg.slgb_path.c_str());
        return 1;
    }
    const float eff_loopcd_w = cfg.loopcd_w_set ? cfg.loopcd_w : (model.hdr.has_u4_loop ? model.loopcd_w_sidecar : 0.0f);
    const float eff_tau_sr = cfg.tau_sr_set ? cfg.tau_sr : model.tau_sr_sidecar;
    const float eff_tau_cos = cfg.tau_cos_set ? cfg.tau_cos : model.tau_cos_sidecar;

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
        for (int b = 0; b < B; ++b) {
            if (prompt_ids[size_t(b) * size_t(n_steps)] != 2) {
                std::fprintf(stderr, "BOS_GATE_FAIL: batch row %d must start with bos_token_id=2 (got %d)\n",
                             b, prompt_ids[size_t(b) * size_t(n_steps)]);
                return 2;
            }
        }
        const size_t expected_ple_floats = size_t(n_steps) * size_t(B) * NUM_LAYERS * PLE_DIM;
        if (!cfg.ple_tok_bin.empty() && ple_all.size() != expected_ple_floats) {
            std::fprintf(stderr, "PLE_BIN_SIZE_MISMATCH: got=%zu floats expected=%zu\n",
                         ple_all.size(), expected_ple_floats);
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
            for (int b = 0; b < B; ++b) htok[b] = prompt_ids[size_t(b) * size_t(n_steps) + size_t(s)];
            CUDA_CHECK(cudaMemcpyAsync(model.d_pos, hpos.data(), B * sizeof(int32_t), cudaMemcpyHostToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(model.d_tok_ids, htok.data(), B * sizeof(int32_t), cudaMemcpyHostToDevice, st));
            size_t step_ple = size_t(B) * NUM_LAYERS * PLE_DIM;
            if (!ple_all.empty()) {
                CUDA_CHECK(cudaMemcpyAsync(model.d_tok_ple, ple_all.data() + size_t(s) * step_ple, step_ple * sizeof(float), cudaMemcpyHostToDevice, st));
            }
            model.emit_step(st, B, cfg.dump_k, cfg.dump_lm_bf16, B == 1 ? 1 : 0, 0, FINAL_LOGIT_SOFTCAP, eff_loopcd_w);
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
        std::printf("DUMP_LOGITS_DONE steps=%d B=%d K=%d sec=%u ded=%u poisoned=%u\n",
                    n_steps, B, cfg.dump_k, hcan.sec_cnt, hcan.ded_cnt, hcan.poisoned);
        if (evaluate_canary_watchdog(hcan, 0u, false) != WatchdogAction::CONTINUE) {
            std::fprintf(stderr, "SAFE_HALT: SECDED DED/IRIS canary tripped during logit dump\n");
            return 3;
        }
    }

    // Mode B: Full Stage-2 live GPU bench (CUDA Graph throughput, live Adaptive B=1, NVML 100Hz, PLE async DMA, 0-alloc 1k, 128K full fill, U-4 rho)
    if (cfg.bench_all) {
        double stream_bidir_gbps = 0.0;
        double stream_copy_gbps = 0.0;
        {
            const size_t bw_bytes = 512ULL * 1024ULL * 1024ULL;
            void *d0 = nullptr, *d1 = nullptr;
            CUDA_CHECK(cudaMalloc(&d0, bw_bytes));
            CUDA_CHECK(cudaMalloc(&d1, bw_bytes));
            CUDA_CHECK(cudaMemset(d0, 0x5A, bw_bytes));
            for (int w = 0; w < 3; ++w) CUDA_CHECK(cudaMemcpy(d1, d0, bw_bytes, cudaMemcpyDeviceToDevice));
            cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
            std::vector<double> bws_bidir, bws_copy;
            for (int r = 0; r < 11; ++r) {
                CUDA_CHECK(cudaEventRecord(e0));
                CUDA_CHECK(cudaMemcpy(d1, d0, bw_bytes, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                bws_bidir.push_back((2.0 * double(bw_bytes)) / (double(ms) * 1.0e6));
                bws_copy.push_back(double(bw_bytes) / (double(ms) * 1.0e6));
            }
            std::sort(bws_bidir.begin(), bws_bidir.end());
            std::sort(bws_copy.begin(), bws_copy.end());
            stream_bidir_gbps = bws_bidir[bws_bidir.size() / 2];
            stream_copy_gbps = bws_copy[bws_copy.size() / 2];
            CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
            CUDA_CHECK(cudaFree(d0)); CUDA_CHECK(cudaFree(d1));
        }

        NvmlSampler100Hz nvml;
        if (!nvml.init(0)) {
            std::fprintf(stderr, "FATAL: NVML init failed (fail-closed telemetry)\n");
            return 3;
        }
        struct GraphPerf {
            double med_ms = 0.0;
            double toks = 0.0;
            double avg_power_w = 0.0;
            double energy_j_per_tok = 0.0;
            double energy_tok_per_j = 0.0;
            double energy_wh_run = 0.0;
            double peak_temp_c = 0.0;
            double nvml_hz = 0.0;
            double mean_k = 1.0;
            uint32_t nvml_samples = 0u;
        };

        uint64_t sec_cnt_total = 0ULL;
        auto check_arm_watchdog = [&](const char* arm_name) {
            HoloSphereRazorCanary64 hcan_arm{};
            CUDA_CHECK(cudaMemcpy(&hcan_arm, model.d_can, sizeof(hcan_arm), cudaMemcpyDeviceToHost));
            sec_cnt_total += hcan_arm.sec_cnt;
            if (evaluate_canary_watchdog(hcan_arm, 0u, false) != WatchdogAction::CONTINUE) {
                std::fprintf(stderr, "SAFE_HALT: canary watchdog tripped in arm=%s (ded=%u pois=%u)\n",
                             arm_name, hcan_arm.ded_cnt, hcan_arm.poisoned);
                std::exit(3);
            }
        };

        auto run_graph_bench = [&](const char* arm_name, int B, int K, int lm_bf16, int T_ctx, int split_t, int wb_en, int iters, float lcd_w, double nvml_target_s, bool advance_pos) -> GraphPerf {
            std::vector<int32_t> hpos(B, T_ctx - 1), htok(B, 2);
            CUDA_CHECK(cudaMemcpy(model.d_pos, hpos.data(), B * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(model.d_tok_ids, htok.data(), B * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(model.d_can, 0, sizeof(HoloSphereRazorCanary64)));

            cudaGraph_t graph; cudaGraphExec_t exec;
            CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
            model.emit_step(st, B, K, lm_bf16, split_t, wb_en, FINAL_LOGIT_SOFTCAP, lcd_w);
            CUDA_CHECK(cudaStreamEndCapture(st, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
            const int n_warm = (B == 1) ? 64 : 16;
            for (int w = 0; w < n_warm; ++w) CUDA_CHECK(cudaGraphLaunch(exec, st));
            CUDA_CHECK(cudaStreamSynchronize(st));

            cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
            std::vector<double> ms_list;
            const int reps = 19;
            int step_ctr = 0;
            for (int r = 0; r < reps; ++r) {
                CUDA_CHECK(cudaEventRecord(e0, st));
                for (int i = 0; i < iters; ++i) {
                    if (advance_pos) {
                        std::fill(hpos.begin(), hpos.end(), (step_ctr++) % T_ctx);
                        CUDA_CHECK(cudaMemcpyAsync(model.d_pos, hpos.data(), B * 4, cudaMemcpyHostToDevice, st));
                    }
                    CUDA_CHECK(cudaGraphLaunch(exec, st));
                }
                CUDA_CHECK(cudaEventRecord(e1, st));
                CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                ms_list.push_back(double(ms) / double(iters));
            }
            std::sort(ms_list.begin(), ms_list.end());
            GraphPerf p{};
            p.med_ms = ms_list[ms_list.size() / 2];
            p.toks = (1000.0 * double(B)) / p.med_ms;
            p.mean_k = double(K);

            if (nvml_target_s > 0.0) {
                const int steps_nvml = std::max(16, int(std::ceil(nvml_target_s / (p.med_ms * 1.0e-3))));
                CUDA_CHECK(cudaStreamSynchronize(st));
                if (!nvml.start()) {
                    std::fprintf(stderr, "FATAL: NVML start failed on %s\n", arm_name);
                    std::exit(3);
                }
                for (int s = 0; s < steps_nvml; ++s) CUDA_CHECK(cudaGraphLaunch(exec, st));
                CUDA_CHECK(cudaStreamSynchronize(st));
                NvmlSampleWindow nw = nvml.stop();
                if (!nw.valid || nw.sample_count < 10u || nw.effective_hz < 95.0) {
                    std::fprintf(stderr, "FATAL: NVML 100Hz sample window invalid on %s (samples=%u hz=%.1f)\n",
                                 arm_name, nw.sample_count, nw.effective_hz);
                    std::exit(3);
                }
                const double total_toks = double(steps_nvml) * double(B);
                p.avg_power_w = nw.avg_power_w;
                p.energy_j_per_tok = nw.energy_j / total_toks;
                p.energy_tok_per_j = total_toks / nw.energy_j;
                p.energy_wh_run = nw.energy_wh;
                p.peak_temp_c = nw.peak_temp_c;
                p.nvml_hz = nw.effective_hz;
                p.nvml_samples = nw.sample_count;
            }
            check_arm_watchdog(arm_name);
            CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
            CUDA_CHECK(cudaGraphExecDestroy(exec)); CUDA_CHECK(cudaGraphDestroy(graph));
            return p;
        };

        auto run_adaptive_b1_bench = [&](float lcd_w, float tau_sr, float tau_cos, double nvml_target_s) -> GraphPerf {
            CUDA_CHECK(cudaMemset(model.d_can, 0, sizeof(HoloSphereRazorCanary64)));
            int32_t* h_halt_pinned = nullptr;
            int32_t* h_pos_pinned = nullptr;
            int32_t* h_tok_pinned = nullptr;
            CUDA_CHECK(cudaHostAlloc(&h_halt_pinned, sizeof(int32_t), cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc(&h_pos_pinned, sizeof(int32_t), cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc(&h_tok_pinned, sizeof(int32_t), cudaHostAllocDefault));
            *h_halt_pinned = 1;
            *h_pos_pinned = 0;
            *h_tok_pinned = 2;

            cudaGraph_t g_pfx, g_k1, g_k2;
            cudaGraphExec_t e_pfx, e_k1, e_k2;
            CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
            model.emit_b1_adaptive_prefix(st, 1, 0, tau_sr, tau_cos);
            CUDA_CHECK(cudaStreamEndCapture(st, &g_pfx));
            CUDA_CHECK(cudaGraphInstantiate(&e_pfx, g_pfx, 0));

            CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
            model.emit_b1_adaptive_k1_tail(st, 0, 1, 0, FINAL_LOGIT_SOFTCAP);
            CUDA_CHECK(cudaStreamEndCapture(st, &g_k1));
            CUDA_CHECK(cudaGraphInstantiate(&e_k1, g_k1, 0));

            CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
            model.emit_b1_adaptive_k2_tail(st, 0, 1, 0, FINAL_LOGIT_SOFTCAP, lcd_w);
            CUDA_CHECK(cudaStreamEndCapture(st, &g_k2));
            CUDA_CHECK(cudaGraphInstantiate(&e_k2, g_k2, 0));

            std::vector<int32_t> tok_cycle = {2, 107, 500, 1200, 3100, 750, 2, 108};
            auto step_once = [&](int step_i) -> int {
                *h_pos_pinned = step_i % 128;
                *h_tok_pinned = tok_cycle[size_t(step_i) % tok_cycle.size()];
                CUDA_CHECK(cudaMemcpyAsync(model.d_pos, h_pos_pinned, sizeof(int32_t), cudaMemcpyHostToDevice, st));
                CUDA_CHECK(cudaMemcpyAsync(model.d_tok_ids, h_tok_pinned, sizeof(int32_t), cudaMemcpyHostToDevice, st));
                CUDA_CHECK(cudaGraphLaunch(e_pfx, st));
                CUDA_CHECK(cudaMemcpyAsync(h_halt_pinned, model.d_halt_flag, sizeof(int32_t), cudaMemcpyDeviceToHost, st));
                CUDA_CHECK(cudaStreamSynchronize(st));
                const int k_used = (*h_halt_pinned != 0) ? 1 : 2;
                CUDA_CHECK(cudaGraphLaunch((k_used == 1) ? e_k1 : e_k2, st));
                return k_used;
            };

            for (int w = 0; w < 8; ++w) step_once(w);
            CUDA_CHECK(cudaStreamSynchronize(st));

            std::vector<double> ms_list;
            const int reps = 19, iters = 8;
            int step_i = 8;
            double k_sum_t = 0.0;
            for (int r = 0; r < reps; ++r) {
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < iters; ++i) k_sum_t += double(step_once(step_i++));
                CUDA_CHECK(cudaStreamSynchronize(st));
                auto t1 = std::chrono::steady_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                ms_list.push_back(ms / double(iters));
            }
            std::sort(ms_list.begin(), ms_list.end());
            GraphPerf p{};
            p.med_ms = ms_list[ms_list.size() / 2];
            p.toks = 1000.0 / p.med_ms;

            const int steps_nvml = std::max(24, int(std::ceil(nvml_target_s / (p.med_ms * 1.0e-3))));
            double k_sum_nv = 0.0;
            if (!nvml.start()) {
                std::fprintf(stderr, "FATAL: NVML start failed on b1_adaptive\n");
                std::exit(3);
            }
            for (int s = 0; s < steps_nvml; ++s) k_sum_nv += double(step_once(step_i++));
            CUDA_CHECK(cudaStreamSynchronize(st));
            NvmlSampleWindow nw = nvml.stop();
            if (!nw.valid || nw.sample_count < 10u || nw.effective_hz < 95.0) {
                std::fprintf(stderr, "FATAL: NVML 100Hz sample window invalid on b1_adaptive (samples=%u hz=%.1f)\n",
                             nw.sample_count, nw.effective_hz);
                std::exit(3);
            }
            const double total_toks = double(steps_nvml);
            p.avg_power_w = nw.avg_power_w;
            p.energy_j_per_tok = nw.energy_j / total_toks;
            p.energy_tok_per_j = total_toks / nw.energy_j;
            p.energy_wh_run = nw.energy_wh;
            p.peak_temp_c = nw.peak_temp_c;
            p.nvml_hz = nw.effective_hz;
            p.nvml_samples = nw.sample_count;
            p.mean_k = k_sum_nv / total_toks;
            (void)k_sum_t;

            check_arm_watchdog("b1_adaptive_live");
            CUDA_CHECK(cudaGraphExecDestroy(e_pfx)); CUDA_CHECK(cudaGraphDestroy(g_pfx));
            CUDA_CHECK(cudaGraphExecDestroy(e_k1)); CUDA_CHECK(cudaGraphDestroy(g_k1));
            CUDA_CHECK(cudaGraphExecDestroy(e_k2)); CUDA_CHECK(cudaGraphDestroy(g_k2));
            CUDA_CHECK(cudaFreeHost(h_halt_pinned));
            CUDA_CHECK(cudaFreeHost(h_pos_pinned));
            CUDA_CHECK(cudaFreeHost(h_tok_pinned));
            return p;
        };

        // Allocate small Woodbury test matrix on hub if not loaded
        size_t wb_bytes = 0;
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
            wb_bytes = 2ULL * 8ULL * HIDDEN_SIZE * sizeof(float);
        }

        GraphPerf p_b1_k1_g32     = run_graph_bench("b1_k1_g32_t128",   1, 1, 0, 128,  1, 0, 8, 0.0f,        1.5, false);
        GraphPerf p_b1_k1_g32_adv = run_graph_bench("b1_k1_g32_adv",    1, 1, 0, 128,  1, 0, 8, 0.0f,        0.0, true);
        GraphPerf p_b1_k1_bf16    = run_graph_bench("b1_k1_bf16_t128",  1, 1, 1, 128,  1, 0, 4, 0.0f,        1.2, false);
        GraphPerf p_b1_k2_g32     = run_graph_bench("b1_k2_g32_t128",   1, 2, 0, 128,  1, 0, 4, eff_loopcd_w,1.5, false);
        GraphPerf p_b1_k2_bf16    = run_graph_bench("b1_k2_bf16_t128",  1, 2, 1, 128,  1, 0, 4, eff_loopcd_w,0.0, false);
        GraphPerf p_b1_k3_g32     = run_graph_bench("b1_k3_g32_t128",   1, 3, 0, 128,  1, 0, 2, eff_loopcd_w,0.0, false);
        GraphPerf p_b1_adaptive   = run_adaptive_b1_bench(eff_loopcd_w, eff_tau_sr, eff_tau_cos, 1.5);
        GraphPerf p_b1_k1_t1024   = run_graph_bench("b1_k1_g32_t1024",  1, 1, 0, 1024, 1, 0, 3, 0.0f,        0.0, false);
        GraphPerf p_b1_k1_wb      = run_graph_bench("b1_k1_wb_t128",    1, 1, 0, 128,  1, 1, 4, 0.0f,        0.0, false);
        GraphPerf p_b16_k1_g32    = run_graph_bench("b16_k1_g32_t128", 16, 1, 0, 128,  0, 0, 2, 0.0f,        1.5, false);
        GraphPerf p_b16_k1_bf16   = run_graph_bench("b16_k1_bf16_t128",16, 1, 1, 128,  0, 0, 2, 0.0f,        1.2, false);
        GraphPerf p_b16_k2_g32    = run_graph_bench("b16_k2_g32_t128", 16, 2, 0, 128,  0, 0, 2, eff_loopcd_w,1.5, false);
        nvml.shutdown();

        double ms_b1_k1_g32_t128  = p_b1_k1_g32.med_ms;
        double ms_b1_k1_bf16_t128 = p_b1_k1_bf16.med_ms;
        double ms_b1_k2_g32_t128  = p_b1_k2_g32.med_ms;
        double ms_b1_k2_bf16_t128 = p_b1_k2_bf16.med_ms;
        double ms_b1_k1_g32_t1024 = p_b1_k1_t1024.med_ms;
        double ms_b1_k1_wb_t128   = p_b1_k1_wb.med_ms;
        double ms_b16_k1_g32_t128 = p_b16_k1_g32.med_ms;
        double ms_b16_k1_bf16_t128= p_b16_k1_bf16.med_ms;
        double ms_b16_k2_g32_t128 = p_b16_k2_g32.med_ms;

        // Async PLE DMA stall test on concurrent stream vs baseline
        double ple_stall_pct = 0.0;
        {
            const int n_ple_f = NUM_LAYERS * PLE_DIM;
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

        // 1,000-step 0-byte dynamic VRAM allocation check INSIDE instantiated graph steady state (advancing pos)
        size_t free_before = 0, total_mem = 0, free_after = 0;
        int64_t dynamic_vram_delta = 0;
        {
            cudaGraph_t graph; cudaGraphExec_t exec;
            CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
            model.emit_step(st, 1, 1, 0, 1, 0, FINAL_LOGIT_SOFTCAP);
            CUDA_CHECK(cudaStreamEndCapture(st, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
            CUDA_CHECK(cudaGraphLaunch(exec, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            CUDA_CHECK(cudaMemGetInfo(&free_before, &total_mem));
            for (int step = 0; step < 1000; ++step) {
                int32_t adv_pos = step % 128;
                CUDA_CHECK(cudaMemcpyAsync(model.d_pos, &adv_pos, sizeof(int32_t), cudaMemcpyHostToDevice, st));
                CUDA_CHECK(cudaGraphLaunch(exec, st));
            }
            CUDA_CHECK(cudaStreamSynchronize(st));
            CUDA_CHECK(cudaMemGetInfo(&free_after, &total_mem));
            dynamic_vram_delta = int64_t(free_before) - int64_t(free_after);
            CUDA_CHECK(cudaGraphExecDestroy(exec)); CUDA_CHECK(cudaGraphDestroy(graph));
        }
        check_arm_watchdog("1k_zero_alloc");

        // Live 128K BF16 KV arena check: deploy 4.5bpw 128K (release redundant d_lm_bf16 shadow & T=1024 P5),
        // fill FULL 811,597,824 B KV slab, and decode at pos=131071!
        if (model.d_lm_bf16) { CUDA_CHECK(cudaFree(model.d_lm_bf16)); model.d_lm_bf16 = nullptr; }
        CUDA_CHECK(cudaFree(model.d_p5));
        model.d_p5 = nullptr;
        uint8_t* d_p5_128k = nullptr;
        CUDA_CHECK(cudaMalloc(&d_p5_128k, ARENA_P5_128K_BF16_BYTES));
        v2::fill_kv_pattern_bf16_sm86<<<256, 128, 0, st>>>(
            reinterpret_cast<uint16_t*>(d_p5_128k), ARENA_P5_128K_BF16_BYTES / 2ULL);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(st));
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
        int pos_128k = 131071;
        int tok_128k = 2;
        CUDA_CHECK(cudaMemcpy(model.d_pos, &pos_128k, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(model.d_tok_ids, &tok_128k, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(model.d_can, 0, sizeof(HoloSphereRazorCanary64)));
        model.emit_step(st, 1, 1, 0, 1, 0, FINAL_LOGIT_SOFTCAP);
        CUDA_CHECK(cudaStreamSynchronize(st));
        std::vector<float> h_logits_128k(VOCAB_SIZE);
        CUDA_CHECK(cudaMemcpy(h_logits_128k.data(), model.d_logits, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        for (float v : h_logits_128k) {
            if (!std::isfinite(v)) {
                std::fprintf(stderr, "FATAL: non-finite logit at 128K pos=131071\n");
                return 3;
            }
        }
        HoloSphereRazorCanary64 hcan{};
        CUDA_CHECK(cudaMemcpy(&hcan, model.d_can, sizeof(hcan), cudaMemcpyDeviceToHost));
        sec_cnt_total += hcan.sec_cnt;
        if (evaluate_canary_watchdog(hcan, 0u, false) != WatchdogAction::CONTINUE) {
            std::fprintf(stderr, "SAFE_HALT: 128K pos=131071 decode tripped watchdog\n");
            return 3;
        }
        size_t free_128k = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_128k, &total_mem));
        double device_mem_info_128k_mib = double(total_mem - free_128k) / (1024.0 * 1024.0);
        const size_t u4_loop_bytes = model.hdr.has_u4_loop
            ? (size_t(9) * 2ULL * size_t(U4_RANK) * size_t(HIDDEN_SIZE) * sizeof(float))
            : 0ULL;
        const size_t exact_128k_alloc_bytes =
            model.total_u4_count * sizeof(uint4) +
            model.total_sc_count * sizeof(uint32_t) +
            model.norms_f32.size() * sizeof(float) +
            (model.inv_freq_slide.size() + model.inv_freq_full.size()) * sizeof(float) +
            u4_loop_bytes + wb_bytes +
            size_t(ARENA_P5_128K_BF16_BYTES) + size_t(ARENA_P7_SCRATCH_BYTES);
        const double exact_128k_alloc_mib = double(exact_128k_alloc_bytes) / (1024.0 * 1024.0);

        double b1_k1_g32_tps   = 1000.0 / ms_b1_k1_g32_t128;
        double b1_k1_bf16_tps  = 1000.0 / ms_b1_k1_bf16_t128;
        double b1_k2_g32_tps   = 1000.0 / ms_b1_k2_g32_t128;
        double b1_k2_bf16_tps  = 1000.0 / ms_b1_k2_bf16_t128;
        double b1_k1_t1024_tps = 1000.0 / ms_b1_k1_g32_t1024;
        double b16_k1_g32_tps  = 16000.0 / ms_b16_k1_g32_t128;
        double b16_k1_bf16_tps = 16000.0 / ms_b16_k1_bf16_t128;
        double b16_k2_g32_tps  = 16000.0 / ms_b16_k2_g32_t128;
        double wb_overhead_pct = 100.0 * (ms_b1_k1_wb_t128 - ms_b1_k1_g32_t128) / ms_b1_k1_g32_t128;

        double bytes_k1_g32  = double(WIRE_BYTES_G32_45BPW + siligemma::WEIGHT_ECC72_BYTES + WIRE_BYTES_LM_HEAD_G32);
        double bytes_k1_bf16 = double(WIRE_BYTES_G32_45BPW + (model.total_sc_count - (VOCAB_SIZE * HIDDEN_SIZE / 32)) * sizeof(uint32_t) + WIRE_BYTES_LM_HEAD_BF16);
        double gbps_k1_g32   = bytes_k1_g32 / (ms_b1_k1_g32_t128 * 1.0e6);
        double gbps_k1_bf16  = bytes_k1_bf16 / (ms_b1_k1_bf16_t128 * 1.0e6);

        // Empirical CUDA U-4 Jacobian sensitivity + chained L25..L33 (+U4) spectral contraction probe
        double rho_cuda = 0.0;
        double rho_cuda_layer_chain = 0.0;
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

            // Consecutive chained L25..33 (+U4) map contraction on live pos=131071 filled KV
            auto eval_chain = [&](const std::vector<float>& vec) {
                CUDA_CHECK(cudaMemcpy(model.d_hid, vec.data(), HIDDEN_SIZE * 4, cudaMemcpyHostToDevice));
                float prev_sc = 1.0f;
                const float* prev_ppl = nullptr;
                int prev_act = 0, prev_nsp = 1;
                for (int ll = U4_LOOP_START; ll <= U4_LOOP_END; ++ll) {
                    model.emit_single_layer(st, ll, 1, 1, 1, 0, prev_sc, prev_ppl, prev_act, prev_nsp);
                }
                model.finalize_to_buf(st, 1, model.d_hid, prev_act, prev_nsp, prev_ppl, prev_sc);
                std::vector<float> out(HIDDEN_SIZE);
                CUDA_CHECK(cudaMemcpy(out.data(), model.d_hid, HIDDEN_SIZE * 4, cudaMemcpyDeviceToHost));
                return out;
            };
            std::vector<float> hc1 = eval_chain(h0);
            std::vector<float> hc2 = eval_chain(hc1);
            double n1_2 = 0.0, n2_2 = 0.0;
            for (int d = 0; d < HIDDEN_SIZE; ++d) {
                double d1 = double(hc1[d]) - double(h0[d]);
                double d2 = double(hc2[d]) - double(hc1[d]);
                n1_2 += d1 * d1;
                n2_2 += d2 * d2;
            }
            rho_cuda_layer_chain = std::sqrt(n2_2 / std::max(n1_2, 1.0e-16));
        }

        CUDA_CHECK(cudaFree(d_p5_128k));

        if (!cfg.out_json_path.empty()) {
            std::ofstream jf(cfg.out_json_path, std::ios::trunc);
            if (!jf.good()) {
                std::fprintf(stderr, "Failed opening --out-json\n");
                return 1;
            }
            jf << "{\n"
               << "  \"provenance_tag\": \"UNCONTROLLED_LAPTOP_NOT_CLAIMED\",\n"
               << "  \"u4_loaded\": " << (model.hdr.has_u4_loop ? "true" : "false") << ",\n"
               << "  \"d2d_stream_gbps\": " << stream_bidir_gbps << ",\n"
               << "  \"d2d_stream_bidir_gbps\": " << stream_bidir_gbps << ",\n"
               << "  \"d2d_stream_copy_gbps\": " << stream_copy_gbps << ",\n"
               << "  \"b1_k1_g32_t128_ms\": " << ms_b1_k1_g32_t128 << ",\n"
               << "  \"b1_k1_g32_t128_toks\": " << b1_k1_g32_tps << ",\n"
               << "  \"b1_k1_g32_t128_gbps\": " << gbps_k1_g32 << ",\n"
               << "  \"b1_k1_g32_adv_ms\": " << p_b1_k1_g32_adv.med_ms << ",\n"
               << "  \"b1_k1_g32_adv_toks\": " << p_b1_k1_g32_adv.toks << ",\n"
               << "  \"b1_k1_bf16_t128_ms\": " << ms_b1_k1_bf16_t128 << ",\n"
               << "  \"b1_k1_bf16_t128_toks\": " << b1_k1_bf16_tps << ",\n"
               << "  \"b1_k1_bf16_t128_gbps\": " << gbps_k1_bf16 << ",\n"
               << "  \"b1_k2_g32_t128_ms\": " << ms_b1_k2_g32_t128 << ",\n"
               << "  \"b1_k2_g32_t128_toks\": " << b1_k2_g32_tps << ",\n"
               << "  \"b1_k2_bf16_t128_ms\": " << ms_b1_k2_bf16_t128 << ",\n"
               << "  \"b1_k2_bf16_t128_toks\": " << b1_k2_bf16_tps << ",\n"
               << "  \"b1_k3_g32_t128_ms\": " << p_b1_k3_g32.med_ms << ",\n"
               << "  \"b1_k3_g32_t128_toks\": " << p_b1_k3_g32.toks << ",\n"
               << "  \"b1_adaptive_ms\": " << p_b1_adaptive.med_ms << ",\n"
               << "  \"b1_adaptive_toks\": " << p_b1_adaptive.toks << ",\n"
               << "  \"b1_adaptive_mean_k\": " << p_b1_adaptive.mean_k << ",\n"
               << "  \"b1_adaptive_tau_sr\": " << eff_tau_sr << ",\n"
               << "  \"b1_adaptive_tau_cos\": " << eff_tau_cos << ",\n"
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
               << "  \"exact_128k_alloc_bytes\": " << exact_128k_alloc_bytes << ",\n"
               << "  \"exact_128k_alloc_mib\": " << exact_128k_alloc_mib << ",\n"
               << "  \"device_mem_info_128k_mib\": " << device_mem_info_128k_mib << ",\n"
               << "  \"arena_128k_vram_mib\": " << exact_128k_alloc_mib << ",\n"
               << "  \"secded_sec_cnt\": " << sec_cnt_total << ",\n"
               << "  \"secded_ded_cnt\": " << hcan.ded_cnt << ",\n"
               << "  \"iris_poisoned\": " << hcan.poisoned << ",\n"
               << "  \"rho_cuda\": " << rho_cuda << ",\n"
               << "  \"rho_cuda_layer_chain\": " << rho_cuda_layer_chain << ",\n"
               << "  \"nvml_arms\": {\n";
            auto emit_nvml = [&](const char* nm, const GraphPerf& gp, bool comma) {
                jf << "    \"" << nm << "\": {\"toks\": " << gp.toks
                   << ", \"ms\": " << gp.med_ms
                   << ", \"nvml_avg_power_w\": " << gp.avg_power_w
                   << ", \"energy_j_per_tok\": " << gp.energy_j_per_tok
                   << ", \"energy_tok_per_j\": " << gp.energy_tok_per_j
                   << ", \"energy_wh_run\": " << gp.energy_wh_run
                   << ", \"peak_temp_c\": " << gp.peak_temp_c
                   << ", \"nvml_hz\": " << gp.nvml_hz
                   << ", \"mean_k\": " << gp.mean_k
                   << ", \"nvml_samples\": " << gp.nvml_samples << "}" << (comma ? "," : "") << "\n";
            };
            emit_nvml("b1_k1_g32_t128", p_b1_k1_g32, true);
            emit_nvml("b1_k1_bf16_t128", p_b1_k1_bf16, true);
            emit_nvml("b1_k2_g32_t128", p_b1_k2_g32, true);
            emit_nvml("b1_adaptive_live", p_b1_adaptive, true);
            emit_nvml("b16_k1_g32_t128", p_b16_k1_g32, true);
            emit_nvml("b16_k1_bf16_t128", p_b16_k1_bf16, true);
            emit_nvml("b16_k2_g32_t128", p_b16_k2_g32, false);
            jf << "  }\n}\n";
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
        std::printf("BENCH_DONE d2d_bidir=%.2f d2d_copy=%.2f GB/s | B=1 K=1 g32=%.2f tok/s (adv=%.2f) bf16=%.2f | B=1 K=2=%.2f Adp=%.2f (K=%.2f) K=3=%.2f | B=16 K=1=%.2f bf16=%.2f | alloc_delta=%lld B | 128K=%.2f MiB (dev=%.2f MiB)\n",
                    stream_bidir_gbps, stream_copy_gbps, b1_k1_g32_tps, p_b1_k1_g32_adv.toks, b1_k1_bf16_tps,
                    b1_k2_g32_tps, p_b1_adaptive.toks, p_b1_adaptive.mean_k, p_b1_k3_g32.toks,
                    b16_k1_g32_tps, b16_k1_bf16_tps,
                    (long long)dynamic_vram_delta, exact_128k_alloc_mib, device_mem_info_128k_mib);
        if (evaluate_canary_watchdog(hcan, 0u, false) != WatchdogAction::CONTINUE) {
            std::fprintf(stderr, "SAFE_HALT: SECDED DED/IRIS canary tripped during bench (ded=%u poisoned=%u)\n",
                         hcan.ded_cnt, hcan.poisoned);
            return 3;
        }
    }

    if (cfg.bench_b1_fast) {
        auto run_fast_graph = [&](const char* name, int iters, bool adv) {
            std::vector<int32_t> hpos(1, 127), htok(1, 2);
            CUDA_CHECK(cudaMemcpy(model.d_pos, hpos.data(), 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(model.d_tok_ids, htok.data(), 4, cudaMemcpyHostToDevice));
            cudaGraph_t graph; cudaGraphExec_t exec;
            CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
            model.emit_step(st, 1, 1, 0, 1, 0, FINAL_LOGIT_SOFTCAP, eff_loopcd_w);
            CUDA_CHECK(cudaStreamEndCapture(st, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
            for (int w = 0; w < 64; ++w) CUDA_CHECK(cudaGraphLaunch(exec, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
            std::vector<double> ms_list;
            int step_ctr = 0;
            for (int r = 0; r < 19; ++r) {
                CUDA_CHECK(cudaEventRecord(e0, st));
                for (int i = 0; i < iters; ++i) {
                    if (adv) {
                        int32_t ap = (step_ctr++) % 128;
                        CUDA_CHECK(cudaMemcpyAsync(model.d_pos, &ap, 4, cudaMemcpyHostToDevice, st));
                    }
                    CUDA_CHECK(cudaGraphLaunch(exec, st));
                }
                CUDA_CHECK(cudaEventRecord(e1, st)); CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                ms_list.push_back(double(ms) / double(iters));
            }
            std::sort(ms_list.begin(), ms_list.end());
            double med = ms_list[ms_list.size() / 2];
            std::printf("FAST_%s: ms=%.5f toks=%.3f\n", name, med, 1000.0 / med);
            CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
            CUDA_CHECK(cudaGraphExecDestroy(exec)); CUDA_CHECK(cudaGraphDestroy(graph));
        };
        run_fast_graph("b1_k1_g32_t128", 8, false);
        run_fast_graph("b1_k1_g32_adv", 8, true);

        int32_t hpos0 = 127, htok0 = 2;
        CUDA_CHECK(cudaMemset(model.d_can, 0, sizeof(HoloSphereRazorCanary64)));
        CUDA_CHECK(cudaMemcpy(model.d_pos, &hpos0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(model.d_tok_ids, &htok0, 4, cudaMemcpyHostToDevice));
        model.emit_step(st, 1, 1, 0, 1, 0, FINAL_LOGIT_SOFTCAP, eff_loopcd_w);
        CUDA_CHECK(cudaStreamSynchronize(st));
        HoloSphereRazorCanary64 hcan_fast{};
        CUDA_CHECK(cudaMemcpy(&hcan_fast, model.d_can, sizeof(hcan_fast), cudaMemcpyDeviceToHost));
        if (evaluate_canary_watchdog(hcan_fast, 0u, false) != WatchdogAction::CONTINUE) {
            std::fprintf(stderr, "SAFE_HALT: canary watchdog tripped in --bench-b1-fast (ded=%u pois=%u)\n", hcan_fast.ded_cnt, hcan_fast.poisoned);
            return 3;
        }
        double sum_emb = 0, sum_norm = 0, sum_qkv = 0, sum_attn = 0, sum_o = 0, sum_geglu = 0, sum_down = 0, sum_n3 = 0, sum_pg = 0, sum_pa = 0, sum_pp = 0, sum_lm = 0;
        cudaEvent_t ev[2]; CUDA_CHECK(cudaEventCreate(&ev[0])); CUDA_CHECK(cudaEventCreate(&ev[1]));
        auto timed = [&](auto&& fn, double& acc) {
            CUDA_CHECK(cudaEventRecord(ev[0], st));
            fn();
            CUDA_CHECK(cudaEventRecord(ev[1], st));
            CUDA_CHECK(cudaEventSynchronize(ev[1]));
            float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, ev[0], ev[1]));
            acc += double(ms) * 1000.0;
        };
        for (int rep = 0; rep < 4; ++rep) {
            timed([&]() { model.emit_embed_and_ple(st, 1); }, sum_emb);
            float prev_sc = 1.0f; const float* prev_ppl = nullptr; int prev_act = 0, prev_nsp = 1;
            for (int l = 0; l < NUM_LAYERS; ++l) {
                const auto& L = model.layers[l];
                const int hd = int(L.head_dim), interm = int(L.interm_size), kv_ro = int(L.is_kv_shared);
                const float* inv_f = L.is_full_attn ? model.d_inv_full : model.d_inv_slide;
                void* kc = static_cast<void*>(model.d_k_slots[L.kv_slot]);
                void* vc = static_cast<void*>(model.d_v_slots[L.kv_slot]);
                timed([&]() {
                    v2::stageA_norm_res_v2_sm86<<<1, 128, 0, st>>>(prev_act ? model.d_tmp1 : model.d_hid, prev_act ? prev_nsp : 1, prev_act ? model.d_hid : nullptr, prev_act ? prev_ppl : nullptr, prev_act ? prev_sc : 1.0f, model.d_norms + L.off_in_nw, model.d_hid, model.d_scr, HIDDEN_SIZE, RMS_NORM_EPS);
                    stageB_parallel_h4_q22_sm86<<<(HIDDEN_SIZE >> 5) / 4, 128, 0, st>>>(model.d_scr, nullptr, nullptr, model.d_xhm, model.d_xlo, model.d_xsc, model.d_xsq, 1, HIDDEN_SIZE, 1);
                }, sum_norm);
                timed([&]() {
                    const auto& lq = model.linears[L.lin_q];
                    if (!kv_ro) {
                        const auto& lk = model.linears[L.lin_k]; const auto& lv = model.linears[L.lin_v];
                        gemv_b1_qkv_fused_sm86<<<(NUM_HEADS * hd + 2 * NUM_KV_HEADS * hd + 15) / 16, 128, 0, st>>>(model.d_xhm, model.d_xlo, model.d_xsc, model.d_xsq, model.d_w_u4 + lq.offset_u4, model.d_w_sc_ecc + lq.offset_sc, NUM_HEADS * hd, model.d_tmp1, model.d_w_u4 + lk.offset_u4, model.d_w_sc_ecc + lk.offset_sc, NUM_KV_HEADS * hd, model.d_tmp2, model.d_w_u4 + lv.offset_u4, model.d_w_sc_ecc + lv.offset_sc, NUM_KV_HEADS * hd, model.d_tmp3, model.d_can, HIDDEN_SIZE);
                    } else {
                        gemv_b1_dp4a22_sm86<16, 1><<<(NUM_HEADS * hd) / 8, 128, 0, st>>>(model.d_xhm, model.d_xlo, model.d_xsc, model.d_xsq, model.d_w_u4 + lq.offset_u4, model.d_w_sc_ecc + lq.offset_sc, model.d_tmp1, model.d_can, 0.0f, NUM_HEADS * hd, HIDDEN_SIZE);
                    }
                }, sum_qkv);
                timed([&]() {
                    qkv_norm_rope_kv_write_sm86<<<dim3(NUM_HEADS + (kv_ro ? 0 : 2), 1), 128, 0, st>>>(model.d_tmp1, model.d_tmp2, model.d_tmp3, model.d_norms + L.off_qn_w, kv_ro ? nullptr : (model.d_norms + L.off_kn_w), inv_f, kc, vc, model.d_pos, 1, NUM_HEADS, NUM_KV_HEADS, hd, model.slot_t_cap[L.kv_slot], RMS_NORM_EPS, kv_ro, model.kv_is_bf16);
                    attention_kv_split_sm86<16><<<dim3(NUM_HEADS, 16), 128, 0, st>>>(model.d_tmp1, kc, vc, model.d_op, model.d_m, model.d_l, model.d_pos, 1, NUM_HEADS, NUM_KV_HEADS, hd, model.slot_t_cap[L.kv_slot], L.sliding_window, model.kv_is_bf16);
                    attention_merge_h4_q22_fast_sm86<16><<<dim3((NUM_HEADS * hd >> 5) / 4, 1), 128, 0, st>>>(model.d_op, model.d_m, model.d_l, model.d_xhm, model.d_xlo, model.d_xsc, model.d_xsq, 1, NUM_HEADS, hd);
                }, sum_attn);
                timed([&]() {
                    const auto& lo = model.linears[L.lin_o];
                    gemv_b1_dp4a22_sm86<16, 4><<<dim3(HIDDEN_SIZE / 8, 4), 128, 0, st>>>(model.d_xhm, model.d_xlo, model.d_xsc, model.d_xsq, model.d_w_u4 + lo.offset_u4, model.d_w_sc_ecc + lo.offset_sc, model.d_tmp1, model.d_can, 0.0f, HIDDEN_SIZE, NUM_HEADS * hd);
                }, sum_o);
                timed([&]() {
                    v2::stageA_norm_res_v2_sm86<<<1, 128, 0, st>>>(model.d_tmp1, 4, model.d_hid, model.d_norms + L.off_pa_nw, 1.0f, model.d_norms + L.off_pf_nw, model.d_hid, model.d_scr, HIDDEN_SIZE, RMS_NORM_EPS);
                    stageB_parallel_h4_q22_sm86<<<(HIDDEN_SIZE >> 5) / 4, 128, 0, st>>>(model.d_scr, nullptr, nullptr, model.d_xhm, model.d_xlo, model.d_xsc, model.d_xsq, 1, HIDDEN_SIZE, 1);
                }, sum_norm);
                timed([&]() {
                    const auto& lg = model.linears[L.lin_gate]; const auto& lu = model.linears[L.lin_up];
                    v2::gemv_b1_geglu_h4_q22_smem_sm86<<<interm / 32, 256, 0, st>>>(model.d_xhm, model.d_xlo, model.d_xsc, model.d_xsq, model.d_w_u4 + lg.offset_u4, model.d_w_sc_ecc + lg.offset_sc, model.d_w_u4 + lu.offset_u4, model.d_w_sc_ecc + lu.offset_sc, model.d_xhm2, model.d_xlo2, model.d_xsc2, model.d_xsq2, model.d_can, interm, HIDDEN_SIZE);
                }, sum_geglu);
                timed([&]() {
                    const auto& ld = model.linears[L.lin_down];
                    v2::gemv_b1_dp4a22_2row_smem_sk_sm86<8, 8><<<dim3(HIDDEN_SIZE / 32, 8), 128, 0, st>>>(model.d_xhm2, model.d_xlo2, model.d_xsc2, model.d_xsq2, model.d_w_u4 + ld.offset_u4, model.d_w_sc_ecc + ld.offset_sc, model.d_tmp1, model.d_can, 0.0f, HIDDEN_SIZE, interm);
                }, sum_down);
                timed([&]() {
                    v2::stageA_norm_res_v2_sm86<<<1, 128, 0, st>>>(model.d_tmp1, 8, model.d_hid, model.d_norms + L.off_pff_nw, 1.0f, nullptr, model.d_hid, model.d_scr, HIDDEN_SIZE, RMS_NORM_EPS);
                    stageB_parallel_h4_q22_sm86<<<(HIDDEN_SIZE >> 5) / 4, 128, 0, st>>>(model.d_scr, nullptr, nullptr, model.d_xhm, model.d_xlo, model.d_xsc, model.d_xsq, 1, HIDDEN_SIZE, 1);
                }, sum_n3);
                const auto& lpg = model.linears[L.lin_ple_gate]; const auto& lpp = model.linears[L.lin_ple_proj];
                timed([&]() {
                    gemv_b1_dp4a22_sm86<16, 6><<<dim3(PLE_DIM / 8, 6), 128, 0, st>>>(model.d_xhm, model.d_xlo, model.d_xsc, model.d_xsq, model.d_w_u4 + lpg.offset_u4, model.d_w_sc_ecc + lpg.offset_sc, model.d_tmp1, model.d_can, 0.0f, PLE_DIM, HIDDEN_SIZE);
                }, sum_pg);
                timed([&]() {
                    fused_ple_act_256_h4_q22_sm86<<<1, 256, 0, st>>>(model.d_tmp1, 6, model.d_ple_in + l * PLE_DIM, model.d_xhm, model.d_xlo, model.d_xsc, model.d_xsq, 1, PLE_DIM);
                }, sum_pa);
                timed([&]() {
                    gemv_b1_dp4a22_sm86<8, 1><<<HIDDEN_SIZE / 16, 128, 0, st>>>(model.d_xhm, model.d_xlo, model.d_xsc, model.d_xsq, model.d_w_u4 + lpp.offset_u4, model.d_w_sc_ecc + lpp.offset_sc, model.d_tmp1, model.d_can, 0.0f, HIDDEN_SIZE, PLE_DIM);
                }, sum_pp);
                prev_ppl = model.d_norms + L.off_ppl_nw; prev_sc = L.layer_scalar; prev_act = 1; prev_nsp = 1;
            }
            timed([&]() { model.emit_head_and_canary(st, 1, 0, FINAL_LOGIT_SOFTCAP, prev_act, prev_nsp, prev_ppl, prev_sc); }, sum_lm);
        }
        CUDA_CHECK(cudaEventDestroy(ev[0])); CUDA_CHECK(cudaEventDestroy(ev[1]));
        std::printf("PROFILE_US: emb=%.0f norm12=%.0f qkv=%.0f attn=%.0f o=%.0f geglu=%.0f down=%.0f n3=%.0f pg=%.0f pa=%.0f pp=%.0f lm=%.0f sum=%.0f\n",
                    sum_emb/4, sum_norm/4, sum_qkv/4, sum_attn/4, sum_o/4, sum_geglu/4, sum_down/4, sum_n3/4, sum_pg/4, sum_pa/4, sum_pp/4, sum_lm/4,
                    (sum_emb+sum_norm+sum_qkv+sum_attn+sum_o+sum_geglu+sum_down+sum_n3+sum_pg+sum_pa+sum_pp+sum_lm)/4);
    }

    CUDA_CHECK(cudaStreamDestroy(st));
    model.free_all();
    return 0;
}
