// SPDX-License-Identifier: Apache-2.0
// Pure-CPU C++20 streaming A_16-BlockH4 (g=32, 4.5 bpw) Form B .slgb packer & verifier.
// Mirrors verify_s184_formb.py::quantize_formb_s4_g32_torch + pack_formb_u64_words_torch
// bit-for-bit using < 200 MiB RAM and zero PyTorch / GPU dependencies.

#include "../kernels/siligemma_arena.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

struct CosetTable {
    float bits[8][16][8];
    uint8_t u8_bits[8][16][8];
};

static CosetTable build_coset_table() {
    CosetTable ct{};
    int cnt[8] = {};
    for (int x = 0; x < 256; ++x) {
        int b[8];
        int s = 0;
        for (int i = 0; i < 8; ++i) {
            b[i] = (x >> i) & 1;
            s += b[i];
        }
        if ((s & 1) == 0) {
            int k = (b[1] ^ b[3] ^ b[5] ^ b[7]) |
                    ((b[2] ^ b[3] ^ b[6] ^ b[7]) << 1) |
                    ((b[4] ^ b[5] ^ b[6] ^ b[7]) << 2);
            int idx = cnt[k]++;
            for (int i = 0; i < 8; ++i) {
                ct.bits[k][idx][i] = static_cast<float>(b[i]);
                ct.u8_bits[k][idx][i] = static_cast<uint8_t>(b[i]);
            }
        }
    }
    return ct;
}

static inline float bf16_to_f32(uint16_t b) noexcept {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static inline uint16_t f32_to_bf16_rne(float f) noexcept {
    uint32_t u = 0u;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) {
        return static_cast<uint16_t>((u >> 16) | 0x0040u);
    }
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t rounding_bias = 0x7FFFu + lsb;
    u += rounding_bias;
    return static_cast<uint16_t>(u >> 16);
}

// IEEE-754 binary32 -> binary16 round-to-nearest-even
static inline uint16_t f32_to_f16_ieee(float value) noexcept {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t mant = bits & 0x007FFFFFu;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu);

    if (exp == 255) {
        if (mant != 0u) return static_cast<uint16_t>(sign | 0x7E00u | (mant >> 13));
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    int32_t new_exp = exp - 127 + 15;
    if (new_exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    if (new_exp <= 0) {
        if (new_exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x00800000u;
        uint32_t shift = static_cast<uint32_t>(14 - new_exp);
        uint32_t half_m = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1u);
        uint32_t mid = 1u << (shift - 1u);
        if (rem > mid || (rem == mid && (half_m & 1u))) {
            ++half_m;
        }
        return static_cast<uint16_t>(sign | half_m);
    }
    uint32_t half_m = mant >> 13;
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half_m & 1u))) {
        ++half_m;
        if (half_m == 0x0400u) {
            half_m = 0u;
            ++new_exp;
            if (new_exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(new_exp) << 10) | half_m);
}

static inline float f16_to_f32_ieee(uint16_t h) noexcept {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
    uint32_t mant = static_cast<uint32_t>(h) & 0x03FFu;
    uint32_t out = 0u;
    if (exp == 0u) {
        if (mant == 0u) {
            out = sign;
        } else {
            int32_t e = -14;
            while ((mant & 0x0400u) == 0u) {
                mant <<= 1u;
                --e;
            }
            mant &= 0x03FFu;
            out = sign | (static_cast<uint32_t>(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

static inline void ml_snap_a16_signed4_block16(
    const float z[16],
    const CosetTable& ct,
    int32_t y_out[16]
) noexcept {
    float ye[16], yo[16], diff[16];
    for (int i = 0; i < 16; ++i) {
        float e = 2.0f * std::nearbyintf(z[i] * 0.5f);
        float o = 2.0f * std::nearbyintf((z[i] - 1.0f) * 0.5f) + 1.0f;
        if (e < -8.0f) e = -8.0f;
        if (e > 6.0f) e = 6.0f;
        if (o < -7.0f) o = -7.0f;
        if (o > 7.0f) o = 7.0f;
        ye[i] = e;
        yo[i] = o;
        float d0 = (z[i] - e) * (z[i] - e);
        float d1 = (z[i] - o) * (z[i] - o);
        diff[i] = d1 - d0;
    }

    int best_k = 0;
    int best_il = 0;
    int best_ir = 0;
    float best_tot = 1.0e30f;

    for (int k = 0; k < 8; ++k) {
        float vl = 1.0e30f;
        int il = 0;
        float vr = 1.0e30f;
        int ir = 0;
        for (int j = 0; j < 16; ++j) {
            float sl = 0.0f;
            float sr = 0.0f;
            for (int i = 0; i < 8; ++i) {
                sl += diff[i] * ct.bits[k][j][i];
                sr += diff[8 + i] * ct.bits[k][j][i];
            }
            if (sl < vl) { vl = sl; il = j; }
            if (sr < vr) { vr = sr; ir = j; }
        }
        if (vl + vr < best_tot) {
            best_tot = vl + vr;
            best_k = k;
            best_il = il;
            best_ir = ir;
        }
    }

    for (int i = 0; i < 8; ++i) {
        float fl = ct.u8_bits[best_k][best_il][i] ? yo[i] : ye[i];
        float fr = ct.u8_bits[best_k][best_ir][i] ? yo[8 + i] : ye[8 + i];
        y_out[i] = static_cast<int32_t>(std::nearbyintf(fl));
        y_out[8 + i] = static_cast<int32_t>(std::nearbyintf(fr));
    }
}

static void quantize_rows_g32(
    const uint16_t* bf16_rows,
    int m,
    int k_dim,
    const CosetTable& ct,
    uint64_t* words_out,
    uint16_t* scales_out
) {
    const int g_per_row = k_dim / 32;
    const int blk_per_row = k_dim / 16;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
    for (int r = 0; r < m; ++r) {
        const uint16_t* row_bf16 = bf16_rows + static_cast<size_t>(r) * static_cast<size_t>(k_dim);
        uint64_t* row_w = words_out + static_cast<size_t>(r) * static_cast<size_t>(blk_per_row);
        uint16_t* row_s = scales_out + static_cast<size_t>(r) * static_cast<size_t>(g_per_row);
        std::vector<float> row_rot(static_cast<size_t>(k_dim));

        for (int c = 0; c < k_dim; c += 4) {
            float x0 = bf16_to_f32(row_bf16[c + 0]);
            float x1 = bf16_to_f32(row_bf16[c + 1]);
            float x2 = bf16_to_f32(row_bf16[c + 2]);
            float x3 = bf16_to_f32(row_bf16[c + 3]);
            float s01 = x0 + x1;
            float d01 = x0 - x1;
            float s23 = x2 + x3;
            float d23 = x2 - x3;
            row_rot[c + 0] = 0.5f * (s01 + s23);
            row_rot[c + 1] = 0.5f * (d01 + d23);
            row_rot[c + 2] = 0.5f * (s01 - s23);
            row_rot[c + 3] = 0.5f * (d01 - d23);
        }

        for (int g = 0; g < g_per_row; ++g) {
            const float* xg = &row_rot[static_cast<size_t>(g) * 32u];
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                float a = std::fabs(xg[i]);
                if (a > amax) amax = a;
            }
            float sc = std::max(amax / 7.25f, 1.0e-7f);
            float z[16];
            int32_t y_tmp[32];

            for (int it = 0; it < 2; ++it) {
                for (int b = 0; b < 2; ++b) {
                    for (int i = 0; i < 16; ++i) z[i] = xg[b * 16 + i] / sc;
                    ml_snap_a16_signed4_block16(z, ct, &y_tmp[b * 16]);
                }
                float den = 0.0f;
                float num = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    float yf = static_cast<float>(y_tmp[i]);
                    den += yf * yf;
                    num += xg[i] * yf;
                }
                den = std::max(den, 1.0f);
                sc = std::max(std::fabs(num / den), 1.0e-7f);
            }

            const uint16_t sc_f16 = f32_to_f16_ieee(sc);
            const float sc_eff = std::max(f16_to_f32_ieee(sc_f16), 1.0e-7f);
            row_s[g] = sc_f16;

            for (int b = 0; b < 2; ++b) {
                for (int i = 0; i < 16; ++i) z[i] = xg[b * 16 + i] / sc_eff;
                int32_t y_fin[16];
                ml_snap_a16_signed4_block16(z, ct, y_fin);
                uint64_t w64 = 0ULL;
                for (int i = 0; i < 16; ++i) {
                    if (y_fin[i] < -8 || y_fin[i] > 7) {
                        std::fprintf(stderr, "Signed 4-bit OOB at row=%d g=%d i=%d val=%d\n", r, g, i, y_fin[i]);
                        std::exit(2);
                    }
                    w64 |= (static_cast<uint64_t>(static_cast<uint32_t>(y_fin[i]) & 0xFu) << (4 * i));
                }
                if (siligemma::a16_secded_syndrome_cpu(w64) != 0u) {
                    std::fprintf(stderr, "SECDED failure at row=%d g=%d b=%d\n", r, g, b);
                    std::exit(3);
                }
                row_w[g * 2 + b] = w64;
            }
        }
    }
}

static void pread_exact(int fd, void* dst, size_t nbytes, uint64_t offset) {
    uint8_t* p = reinterpret_cast<uint8_t*>(dst);
    size_t done = 0u;
    while (done < nbytes) {
        size_t chunk = std::min<size_t>(nbytes - done, 16ULL * 1024ULL * 1024ULL);
        ssize_t got = ::pread(fd, p + done, chunk, static_cast<off_t>(offset + done));
        if (got <= 0) {
            std::perror("pread");
            std::exit(4);
        }
        done += static_cast<size_t>(got);
    }
}

static void pwrite_exact(int fd, const void* src, size_t nbytes, uint64_t offset) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(src);
    size_t done = 0u;
    while (done < nbytes) {
        size_t chunk = std::min<size_t>(nbytes - done, 16ULL * 1024ULL * 1024ULL);
        ssize_t put = ::pwrite(fd, p + done, chunk, static_cast<off_t>(offset + done));
        if (put <= 0) {
            std::perror("pwrite");
            std::exit(5);
        }
        done += static_cast<size_t>(put);
    }
}

static int cmd_pack(const char* st_path, const char* slgb_path, const char* jobs_path) {
    std::fesetround(FE_TONEAREST);
    const CosetTable ct = build_coset_table();

    int fd_in = ::open(st_path, O_RDONLY);
    if (fd_in < 0) { std::perror("open input safetensors"); return 1; }
    struct stat st_in{};
    if (::fstat(fd_in, &st_in) != 0 || st_in.st_size <= 0) {
        std::fprintf(stderr, "Invalid input safetensors size\n");
        ::close(fd_in);
        return 1;
    }
    const uint64_t in_file_bytes = static_cast<uint64_t>(st_in.st_size);
    int fd_out = ::open(slgb_path, O_RDWR | O_CREAT, 0644);
    if (fd_out < 0) { std::perror("open output slgb"); ::close(fd_in); return 1; }
    struct stat st_out{};
    if (::fstat(fd_out, &st_out) != 0) {
        std::perror("fstat output slgb");
        ::close(fd_in);
        ::close(fd_out);
        return 1;
    }
    const uint64_t out_file_bytes = static_cast<uint64_t>(st_out.st_size);

    std::ifstream jf(jobs_path);
    if (!jf) {
        std::fprintf(stderr, "Cannot open jobs file: %s\n", jobs_path);
        ::close(fd_in);
        ::close(fd_out);
        return 1;
    }

    std::string line;
    std::vector<uint16_t> bf16_buf;
    std::vector<uint64_t> words_buf;
    std::vector<uint16_t> scales_buf;
    std::vector<float> f32_buf;
    uint64_t total_linears_done = 0u;
    uint64_t total_norms_done = 0u;
    auto span_fits = [](uint64_t off, uint64_t len, uint64_t lim) -> bool {
        return off <= lim && len <= (lim - off);
    };

    while (std::getline(jf, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string kind;
        if (!(iss >> kind)) continue;
        if (kind == "LINEAR") {
            int m = 0, k = 0;
            uint64_t in_off = 0, out_w_off = 0, out_s_off = 0;
            if (!(iss >> m >> k >> in_off >> out_w_off >> out_s_off) || m <= 0 || k <= 0 || (k % 32) != 0) {
                std::fprintf(stderr, "Malformed LINEAR job line\n");
                return 1;
            }
            const uint64_t in_bytes = static_cast<uint64_t>(m) * static_cast<uint64_t>(k) * 2ULL;
            const uint64_t w_bytes = static_cast<uint64_t>(m) * static_cast<uint64_t>(k / 16) * 8ULL;
            const uint64_t s_bytes = static_cast<uint64_t>(m) * static_cast<uint64_t>(k / 32) * 2ULL;
            if (!span_fits(in_off, in_bytes, in_file_bytes) ||
                !span_fits(out_w_off, w_bytes, out_file_bytes) ||
                !span_fits(out_s_off, s_bytes, out_file_bytes)) {
                std::fprintf(stderr, "LINEAR job offset/span out of bounds\n");
                return 1;
            }
            const size_t n_elems = static_cast<size_t>(m) * static_cast<size_t>(k);
            bf16_buf.resize(n_elems);
            pread_exact(fd_in, bf16_buf.data(), n_elems * sizeof(uint16_t), in_off);
            words_buf.resize(static_cast<size_t>(m) * static_cast<size_t>(k / 16));
            scales_buf.resize(static_cast<size_t>(m) * static_cast<size_t>(k / 32));
            quantize_rows_g32(bf16_buf.data(), m, k, ct, words_buf.data(), scales_buf.data());
            pwrite_exact(fd_out, words_buf.data(), words_buf.size() * sizeof(uint64_t), out_w_off);
            pwrite_exact(fd_out, scales_buf.data(), scales_buf.size() * sizeof(uint16_t), out_s_off);
            ++total_linears_done;
        } else if (kind == "EMBED") {
            int total_rows = 0, k = 0, chunk_rows = 0;
            uint64_t in_off = 0, out_emb_off = 0, out_w_off = 0, out_s_off = 0;
            if (!(iss >> total_rows >> k >> chunk_rows >> in_off >> out_emb_off >> out_w_off >> out_s_off) ||
                total_rows <= 0 || k <= 0 || chunk_rows <= 0 || (k % 32) != 0) {
                std::fprintf(stderr, "Malformed EMBED job line\n");
                return 1;
            }
            const uint64_t emb_bytes = static_cast<uint64_t>(total_rows) * static_cast<uint64_t>(k) * 2ULL;
            const uint64_t w_bytes = static_cast<uint64_t>(total_rows) * static_cast<uint64_t>(k / 16) * 8ULL;
            const uint64_t s_bytes = static_cast<uint64_t>(total_rows) * static_cast<uint64_t>(k / 32) * 2ULL;
            if (!span_fits(in_off, emb_bytes, in_file_bytes) ||
                !span_fits(out_emb_off, emb_bytes, out_file_bytes) ||
                !span_fits(out_w_off, w_bytes, out_file_bytes) ||
                !span_fits(out_s_off, s_bytes, out_file_bytes)) {
                std::fprintf(stderr, "EMBED job offset/span out of bounds\n");
                return 1;
            }
            for (int r0 = 0; r0 < total_rows; r0 += chunk_rows) {
                int r_cnt = std::min(chunk_rows, total_rows - r0);
                size_t n_elems = static_cast<size_t>(r_cnt) * static_cast<size_t>(k);
                bf16_buf.resize(n_elems);
                uint64_t src_byte = in_off + static_cast<uint64_t>(r0) * static_cast<uint64_t>(k) * 2ULL;
                pread_exact(fd_in, bf16_buf.data(), n_elems * sizeof(uint16_t), src_byte);
                uint64_t dst_emb = out_emb_off + static_cast<uint64_t>(r0) * static_cast<uint64_t>(k) * 2ULL;
                pwrite_exact(fd_out, bf16_buf.data(), n_elems * sizeof(uint16_t), dst_emb);

                words_buf.resize(static_cast<size_t>(r_cnt) * static_cast<size_t>(k / 16));
                scales_buf.resize(static_cast<size_t>(r_cnt) * static_cast<size_t>(k / 32));
                quantize_rows_g32(bf16_buf.data(), r_cnt, k, ct, words_buf.data(), scales_buf.data());
                uint64_t dst_w = out_w_off + static_cast<uint64_t>(r0) * static_cast<uint64_t>(k / 16) * 8ULL;
                uint64_t dst_s = out_s_off + static_cast<uint64_t>(r0) * static_cast<uint64_t>(k / 32) * 2ULL;
                pwrite_exact(fd_out, words_buf.data(), words_buf.size() * sizeof(uint64_t), dst_w);
                pwrite_exact(fd_out, scales_buf.data(), scales_buf.size() * sizeof(uint16_t), dst_s);
            }
        } else if (kind == "NORM") {
            size_t cnt = 0;
            uint64_t in_off = 0, out_off = 0;
            if (!(iss >> cnt >> in_off >> out_off) || cnt == 0u) {
                std::fprintf(stderr, "Malformed NORM job line\n");
                return 1;
            }
            if (!span_fits(in_off, static_cast<uint64_t>(cnt) * 2ULL, in_file_bytes) ||
                !span_fits(out_off, static_cast<uint64_t>(cnt) * 4ULL, out_file_bytes)) {
                std::fprintf(stderr, "NORM job offset/span out of bounds\n");
                return 1;
            }
            bf16_buf.resize(cnt);
            f32_buf.resize(cnt);
            pread_exact(fd_in, bf16_buf.data(), cnt * sizeof(uint16_t), in_off);
            for (size_t i = 0; i < cnt; ++i) f32_buf[i] = bf16_to_f32(bf16_buf[i]);
            pwrite_exact(fd_out, f32_buf.data(), cnt * sizeof(float), out_off);
            ++total_norms_done;
        } else {
            std::fprintf(stderr, "Unknown job kind: %s\n", kind.c_str());
            return 1;
        }
    }
    ::close(fd_in);
    ::close(fd_out);
    std::printf("PACK_COMPLETE linears=%llu norms=%llu\n",
                static_cast<unsigned long long>(total_linears_done),
                static_cast<unsigned long long>(total_norms_done));
    return 0;
}

static int cmd_verify(const char* slgb_path) {
    int fd = ::open(slgb_path, O_RDONLY);
    if (fd < 0) { std::perror("open slgb"); return 1; }
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(siligemma::SlgbHeader))) {
        std::fprintf(stderr, "slgb file missing or truncated\n");
        ::close(fd);
        return 1;
    }
    const uint64_t file_bytes = static_cast<uint64_t>(st.st_size);
    siligemma::SlgbHeader hdr{};
    pread_exact(fd, &hdr, sizeof(hdr), 0ULL);
    if (hdr.magic != siligemma::SLGB_MAGIC || hdr.version != siligemma::SLGB_VERSION) {
        std::fprintf(stderr, "Bad SLGB magic/version\n");
        ::close(fd);
        return 1;
    }
    std::array<siligemma::SiliGemmaLayerDesc, siligemma::GEMMA4_NUM_LAYERS> layers{};
    const uint64_t min_meta = sizeof(hdr) + layers.size() * sizeof(siligemma::SiliGemmaLayerDesc)
                            + siligemma::LIVE_LINEAR_COUNT * sizeof(siligemma::SiliGemmaLinearDesc);
    if (file_bytes < min_meta) {
        std::fprintf(stderr, "slgb descriptor table truncated\n");
        ::close(fd);
        return 1;
    }
    uint64_t off = sizeof(hdr);
    pread_exact(fd, layers.data(), layers.size() * sizeof(siligemma::SiliGemmaLayerDesc), off);
    off += layers.size() * sizeof(siligemma::SiliGemmaLayerDesc);
    std::vector<siligemma::SiliGemmaLinearDesc> linears(siligemma::LIVE_LINEAR_COUNT);
    pread_exact(fd, linears.data(), linears.size() * sizeof(siligemma::SiliGemmaLinearDesc), off);
    off += linears.size() * sizeof(siligemma::SiliGemmaLinearDesc);

    const siligemma::ValidationError verr = siligemma::validate_slgb_file_layout(hdr, layers, linears, file_bytes, true);
    if (verr != siligemma::ValidationError::OK) {
        std::fprintf(stderr, "validate_slgb_file_layout failed err=%d\n", static_cast<int>(verr));
        ::close(fd);
        return 1;
    }

    uint64_t off_p1 = siligemma::align_up_n(off, 256ULL);
    uint64_t off_p2_emb = siligemma::align_up_n(off_p1 + hdr.total_wire_bytes, 256ULL);
    uint64_t off_p2_lm = siligemma::align_up_n(off_p2_emb + siligemma::EMBED_TOKENS_BF16_BYTES, 256ULL);

    uint64_t total_words = 0ULL;
    uint64_t syn_errors = 0ULL;
    std::vector<uint64_t> chunk_u64(4ULL * 1024ULL * 1024ULL);

    auto scan_words = [&](uint64_t base_off, uint64_t n_words) {
        uint64_t done = 0ULL;
        while (done < n_words) {
            size_t cnt = static_cast<size_t>(std::min<uint64_t>(n_words - done, chunk_u64.size()));
            pread_exact(fd, chunk_u64.data(), cnt * sizeof(uint64_t), base_off + done * sizeof(uint64_t));
            for (size_t i = 0; i < cnt; ++i) {
                if (siligemma::a16_secded_syndrome_cpu(chunk_u64[i]) != 0u) ++syn_errors;
            }
            total_words += cnt;
            done += cnt;
        }
    };

    const uint64_t p1_words = siligemma::LIVE_LINEAR_PARAMS / 16ULL;
    scan_words(off_p1, p1_words);
    if (hdr.flags & 1u) {
        const uint64_t lm_words = siligemma::EMBED_TOKENS_PARAMS / 16ULL;
        scan_words(off_p2_lm, lm_words);
    }

    ::close(fd);
    if (syn_errors != 0ULL) {
        std::fprintf(stderr, "SECDED errors detected: %llu\n", static_cast<unsigned long long>(syn_errors));
        return 1;
    }
    std::printf("SLGB_VERIFY_PASS total_words=%llu secded_errors=0\n",
                static_cast<unsigned long long>(total_words));
    return 0;
}

static int cmd_selftest_quant(const char* in_bf16_path, int m, int k, const char* out_w_path, const char* out_s_path) {
    if (m <= 0 || k <= 0 || (k % 32) != 0) {
        std::fprintf(stderr, "Invalid selftest-quant dims: m=%d k=%d\n", m, k);
        return 1;
    }
    std::fesetround(FE_TONEAREST);
    const CosetTable ct = build_coset_table();
    const size_t n = static_cast<size_t>(m) * static_cast<size_t>(k);
    std::vector<uint16_t> bf16(n);
    int fd_i = ::open(in_bf16_path, O_RDONLY);
    if (fd_i < 0) return 1;
    pread_exact(fd_i, bf16.data(), n * sizeof(uint16_t), 0ULL);
    ::close(fd_i);

    std::vector<uint64_t> w(static_cast<size_t>(m) * static_cast<size_t>(k / 16));
    std::vector<uint16_t> s(static_cast<size_t>(m) * static_cast<size_t>(k / 32));
    quantize_rows_g32(bf16.data(), m, k, ct, w.data(), s.data());

    int fd_w = ::open(out_w_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    int fd_s = ::open(out_s_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd_w < 0 || fd_s < 0) return 1;
    pwrite_exact(fd_w, w.data(), w.size() * sizeof(uint64_t), 0ULL);
    pwrite_exact(fd_s, s.data(), s.size() * sizeof(uint16_t), 0ULL);
    ::close(fd_w);
    ::close(fd_s);

    // Also verify f32_to_bf16_rne helper doesn't get stripped unused
    volatile uint16_t z = f32_to_bf16_rne(0.0f);
    (void)z;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--verify" && argc == 3) {
        return cmd_verify(argv[2]);
    }
    if (argc >= 2 && std::string(argv[1]) == "--pack" && argc == 5) {
        return cmd_pack(argv[2], argv[3], argv[4]);
    }
    if (argc >= 2 && std::string(argv[1]) == "--selftest-quant" && argc == 7) {
        return cmd_selftest_quant(argv[2], std::atoi(argv[3]), std::atoi(argv[4]), argv[5], argv[6]);
    }
    std::fprintf(stderr, "Usage:\n  %s --pack <model.safetensors> <out.slgb> <jobs.txt>\n  %s --verify <in.slgb>\n", argv[0], argv[0]);
    return 2;
}
