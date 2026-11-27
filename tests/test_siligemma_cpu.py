#!/usr/bin/env python3
"""Moissanite-Labs/siligemma Pure-CPU C++20 Parity, 0-ULP Tree, SLGB Packer, & R5 Controls.

Strictly CPU-only (CUDA_VISIBLE_DEVICES="", < 200 MiB RAM, 0 PyTorch/HF weight loads).
Zero bare asserts, zero unseeded RNG, LF-only, bit-identical stdout across:
  python3 -I -B tests/test_siligemma_cpu.py
  python3 -I -O -B tests/test_siligemma_cpu.py
  python3 -I -B -W error tests/test_siligemma_cpu.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Sequence
import zlib

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNELS_DIR = REPO_ROOT / "kernels"
DOCS_DIR = REPO_ROOT / "docs"
TOOLS_DIR = REPO_ROOT / "tools"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import pack_siligemma_formb as packer  # noqa: E402

EXPECTED_PINNED_SHA256 = {
    "Makefile": "7cb15397c024c1f0cc47efb6a9b67fad17b6a276023e5631e1366cb56e7271df",
    "docs/ARCHITECTURE_FORMB_SILIGEMMA.md": "b7410d0c491748632585f884115d8241efe0b837290ce164e78fe41a0b074fd6",
    "docs/PREREG_W22_GEMMA4_FORMB.json": "03724afa55d9008675225ae48cf78bc2fdcd180418ec6a5ad60a4e15facf37e0",
    "docs/STAGE1_REPORT.md": "86f95ebae659922693853c030723d120996dd8d56de82cb875c922076023af99",
    "docs/STAGE1_VERDICT.json": "d67403ae29265313d163e93ea0184ef5eb9c17e4a5786c89bff186540df4c97b",
    "kernels/siligemma_arena.hpp": "dd44365c2f43577c10df625cb958fb386567507d787f01d26f79533b9849fb23",
    "kernels/siligemma_bench_sm86.cu": "bbc026e88570066ab8695362ec1bae02a9ea6e7a959ec8c38eb5b1ab0852113e",
    "kernels/siligemma_kernels_sm86.cu": "5ee2e547bcacc0b14659160645c0044b32accf10e0c4326db97f55d440413818",
    "tools/pack_siligemma_formb.py": "689ed377f7bd32061c0bf01561bae2a665c2b2320b282e77a87e540f75a97faf",
    "tools/slgb_cpu_worker.cpp": "da51f06391612947efeebf66fa3b2224ff6f02701829fd14ca3ed7bed42e6516",
}

MAX_KERNEL_REGS = 48
MAX_STACK_BYTES = 0
MAX_SPILL_STORE_BYTES = 0
MAX_SPILL_LOAD_BYTES = 0

SLASH = chr(47)
HOST_USER_RE = re.compile(
    "(?:" + SLASH + "ho" + "me" + SLASH + "|" + SLASH + "Us" + "ers" + SLASH + ")[A-Za-z0-9._-]+"
)
FORBIDDEN_ANCHORS = (
    "Python.h",
    "torch/torch.h",
    "pybind11",
    "cublas_v2.h",
    "c10/",
    "import torch",
    "import transformers",
    "import peft",
)
BINARY_SUFFIXES = frozenset((".o", ".so", ".a", ".out", ".exe", ".bin", ".ptx", ".cubin", ".fatbin"))
WEIGHT_SUFFIXES = frozenset((".safetensors", ".slgb", ".pt", ".pth", ".ckpt", ".gguf"))


class SiliGemmaGateError(ValueError):
    """Fail-closed typed error for SiliGemma CPU & SASS verification."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


def require_ok(cond: bool, code: str, message: str) -> None:
    if cond is not True:
        raise SiliGemmaGateError(code, message)


def read_regular_bytes_nofollow(path: Path, max_bytes: int = 32 * 1024 * 1024) -> bytes:
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise SiliGemmaGateError("FILE_MISSING", f"{path.name}: {exc}") from exc
    require_ok(not stat.S_ISLNK(st.st_mode), "SYMLINK_FORBIDDEN", f"symlink forbidden: {path.name}")
    require_ok(stat.S_ISREG(st.st_mode), "NON_REGULAR_FILE", f"not regular file: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(path), flags)
    try:
        fst = os.fstat(fd)
        require_ok(stat.S_ISREG(fst.st_mode), "NON_REGULAR_FD", f"non-regular fd: {path.name}")
        require_ok(fst.st_size <= max_bytes, "FILE_TOO_LARGE", f"{path.name} exceeds byte limit")
        raw = os.read(fd, max_bytes + 1)
    finally:
        os.close(fd)
    require_ok(len(raw) <= max_bytes, "FILE_TOO_LARGE", f"{path.name} exceeded read cap")
    return raw


def parse_strict_json_bytes(raw: bytes, label: str = "json") -> dict[str, Any]:
    require_ok(len(raw.strip()) > 0, "EMPTY_JSON", f"{label}: empty JSON")
    require_ok(not raw.startswith(b"\xef\xbb\xbf"), "BOM_FORBIDDEN", f"{label}: BOM forbidden")
    require_ok(b"\r" not in raw, "CRLF_FORBIDDEN", f"{label}: CRLF forbidden")
    text = raw.decode("utf-8", "strict")

    def no_dup_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in pairs:
            require_ok(k not in out, "JSON_DUP_KEY", f"{label}: duplicate key {k!r}")
            out[k] = v
        return out

    def reject_const(tok: str) -> Any:
        raise SiliGemmaGateError("JSON_NONFINITE", f"{label}: non-finite token {tok!r}")

    obj = json.loads(text, object_pairs_hook=no_dup_pairs, parse_constant=reject_const)
    require_ok(isinstance(obj, dict), "JSON_ROOT_NOT_DICT", f"{label}: root must be object")
    return obj


def f32(val: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(val)))[0]


def f32_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(val)))[0]


def tree_reduce_128_py(tile: Sequence[float]) -> float:
    require_ok(len(tile) == 128, "TREE_LEN", f"expected 128 lanes, got {len(tile)}")
    buf = [f32(v) for v in tile]
    half = 64
    while half >= 1:
        for t in range(half):
            buf[t] = f32(buf[t] + buf[t + half])
        half >>= 1
    return buf[0]


def canonical_tree_sum_block128_py(vals: Sequence[float]) -> float:
    n = len(vals)
    if n == 0:
        return f32(0.0)
    n_blocks = (n + 127) // 128
    outer = [f32(0.0)] * (128 if n_blocks <= 128 else n_blocks)
    for b in range(n_blocks):
        tile = [f32(0.0)] * 128
        base = b * 128
        lim = min(128, n - base)
        for t in range(lim):
            tile[t] = f32(vals[base + t])
        outer[b] = tree_reduce_128_py(tile)
    if n_blocks <= 128:
        return tree_reduce_128_py(outer)
    return canonical_tree_sum_block128_py(outer)


def canonical_tree_rmsnorm_py(x: Sequence[float], gamma: Sequence[float] | None = None, eps: float = 1.0e-6) -> list[float]:
    d = len(x)
    require_ok(d > 0 and (gamma is None or len(gamma) == d), "RMSNORM_DIM", "invalid RMSNorm dimensions")
    sq = [f32(f32(v) * f32(v)) for v in x]
    sum_sq = canonical_tree_sum_block128_py(sq)
    mean_sq = f32(sum_sq / f32(float(d)))
    inv_rms = f32(1.0 / math.sqrt(float(f32(mean_sq + f32(eps)))))
    out: list[float] = []
    for i, xv in enumerate(x):
        g = f32(gamma[i]) if gamma is not None else f32(1.0)
        out.append(f32(f32(f32(xv) * inv_rms) * g))
    return out


def a16_secded_syndrome_py(w64: int) -> int:
    w0 = w64 & 0xFFFFFFFF
    w1 = (w64 >> 32) & 0xFFFFFFFF
    px = (w0 ^ w1) & 0x11111111
    p0 = px.bit_count() & 1
    s0 = (px & 0x10101010).bit_count() & 1
    s1 = (px & 0x11001100).bit_count() & 1
    s2 = (px & 0x11110000).bit_count() & 1
    s3 = (w1 & 0x11111111).bit_count() & 1
    return p0 | (s0 << 1) | (s1 << 2) | (s2 << 3) | (s3 << 4)


def parse_ptxas_log(text: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    cur_name: str | None = None
    for ln in text.splitlines():
        m_fn = re.search(r"Function properties for (\S+)", ln)
        if m_fn:
            cur_name = m_fn.group(1)
            out[cur_name] = {"stack_bytes": 0, "spill_store_bytes": 0, "spill_load_bytes": 0, "registers": 0}
            continue
        if cur_name is None:
            continue
        m_sp = re.search(r"(\d+)\s+bytes stack frame,\s*(\d+)\s+bytes spill stores,\s*(\d+)\s+bytes spill loads", ln)
        if m_sp:
            out[cur_name]["stack_bytes"] = int(m_sp.group(1))
            out[cur_name]["spill_store_bytes"] = int(m_sp.group(2))
            out[cur_name]["spill_load_bytes"] = int(m_sp.group(3))
        m_rg = re.search(r"Used\s+(\d+)\s+registers", ln)
        if m_rg:
            out[cur_name]["registers"] = int(m_rg.group(1))
    return out


def validate_ptxas_and_sass(
    kernels: dict[str, dict[str, int]],
    sass_counts: dict[str, int],
    *,
    min_kernels: int = 10,
) -> dict[str, int]:
    require_ok(len(kernels) >= min_kernels, "PTXAS_KERNEL_COUNT", f"expected >= {min_kernels} kernels, got {len(kernels)}")
    for kname, stats in kernels.items():
        require_ok(stats["stack_bytes"] == MAX_STACK_BYTES, "PTXAS_NONZERO_STACK", f"{kname}: stack={stats['stack_bytes']}")
        require_ok(stats["spill_store_bytes"] == MAX_SPILL_STORE_BYTES, "PTXAS_SPILL_STORE", f"{kname}: spill_st={stats['spill_store_bytes']}")
        require_ok(stats["spill_load_bytes"] == MAX_SPILL_LOAD_BYTES, "PTXAS_SPILL_LOAD", f"{kname}: spill_ld={stats['spill_load_bytes']}")
        require_ok(0 < stats["registers"] <= MAX_KERNEL_REGS, "PTXAS_REGS_OVER_48", f"{kname}: regs={stats['registers']} > {MAX_KERNEL_REGS}")
    require_ok(sass_counts.get("ldl_insns", -1) == 0, "SASS_LDL_PRESENT", f"ldl={sass_counts.get('ldl_insns')}")
    require_ok(sass_counts.get("stl_insns", -1) == 0, "SASS_STL_PRESENT", f"stl={sass_counts.get('stl_insns')}")
    return {
        "total_kernels": len(kernels),
        "max_registers": max(s["registers"] for s in kernels.values()),
        "max_stack_bytes": max(s["stack_bytes"] for s in kernels.values()),
        "max_spill_store_bytes": max(s["spill_store_bytes"] for s in kernels.values()),
        "max_spill_load_bytes": max(s["spill_load_bytes"] for s in kernels.values()),
        "ldl_insns": sass_counts["ldl_insns"],
        "stl_insns": sass_counts["stl_insns"],
    }


def _recompute_s184_kill_gates(raw: dict[str, Any]) -> dict[str, str]:
    ptx = raw["ptxas_sass_summary"]
    cb = raw["cuda_bench"]
    tp = raw["torch_phase"]
    lg = raw["logit_comparisons"]
    tp_map = {(int(r["batch"]), int(r["k_pass"]), str(r["lm_head"])): r for r in cb["throughput_matrix"]}

    g1_ok = (
        bool(raw["ldd_zero_anchors"] and raw["ptxas_zero_anchors"])
        and cb["bos_token_id"] == 2
        and bool(tp["bos_sanity_pass"])
        and cb["live_linears"] == 276
        and cb["dead_kv_omitted"] == 60
        and cb["wire_bytes"] == 1055711232
        and cb["alloc_vram_delta_bytes_1k"] == 0
        and cb["alloc_heap_delta_bytes_1k"] == 0
        and float(cb["dma_stall_b1_pct"]) <= 2.0
        and float(cb["arena_128k_vram_mib"]) <= 3072.0
        and float(lg["cuda_formb_bf16_vs_torch_forma_g0"]["max_abs_diff"]) <= 1.0e-3
    )
    g2_ok = (
        int(ptx["total_kernels"]) >= 13
        and int(ptx["max_stack_bytes"]) == 0
        and int(ptx["max_spill_store_bytes"]) == 0
        and int(ptx["max_spill_load_bytes"]) == 0
        and int(ptx["ldl_insns"]) == 0
        and int(ptx["stl_insns"]) == 0
        and int(ptx["max_registers"]) <= 48
        and int(cb["tree_reduce_max_ulp"]) == 0
        and int(cb["secded_sec_repaired_count"]) == 3
        and int(cb["secded_ded_trapped_count"]) == 3
        and bool(cb["secded_ded_watchdog_safehalt"])
        and int(cb["iris_trojan_trapped_count"]) == 1
        and bool(cb["iris_trojan_watchdog_safehalt"])
        and int(cb["canary_final_ded_count"]) == 0
        and float(cb["woodbury_max_rel_norm_drift"]) <= 1.0e-6
        and float(cb["woodbury_hoisted_overhead_pct"]) <= 25.0
    )
    hub = tp["hub_steering_l10_14"]
    g3_ok = (
        (float(tp_map[(1, 1, "4.5bpw_shadow")]["tok_per_s"]) >= 190.0 or float(tp_map[(1, 1, "bf16_tied")]["tok_per_s"]) >= 135.0)
        and (float(tp_map[(1, 2, "4.5bpw_shadow")]["tok_per_s"]) >= 155.0 or float(tp_map[(1, 2, "bf16_tied")]["tok_per_s"]) >= 115.0)
        and float(cb["hub_l10_14_max_rel_norm_drift"]) <= 1.0e-6
        and float(hub["max_row_norm_drift"]) <= 1.0e-6
        and float(hub["wt2_test_drift_nats"]) <= 0.03
        and int(hub["kv_writes_l15_34"]) == 0
    )
    u4 = tp["u4_loop_probe"]
    wt2_d = float(u4["k2_wt2_test_nll"]) - float(u4["k1_wt2_test_nll"])
    cd_g = float(u4["k1_cd_test_nll"]) - float(u4["k2_cd_test_nll"])
    gsm_g = float(u4["k1_gsm_ood_nll"]) - float(u4["k2_gsm_ood_nll"])
    g4_ok = (
        int(u4["extra_kv_bytes"]) == 0
        and float(u4["k1_zero_init_max_abs_diff_vs_35l"]) <= 1.0e-5
        and float(u4["k3_rho_hat"]) < 0.6180
        and float(u4["k2_step_ratio"]) < 1.5
        and float(u4["k3_step_ratio"]) < 1.5
        and float(u4["k2_exit_cos"]) >= 0.75
        and float(u4["k3_exit_cos"]) >= 0.75
        and float(cb["wt2_cuda_k3_rho_hat"]) < 0.6180
        and float(cb["wt2_cuda_k2_step_ratio"]) < 1.5
        and float(cb["wt2_cuda_k3_step_ratio"]) < 1.5
        and wt2_d <= 0.05
        and (cd_g >= 0.015 or gsm_g >= 0.015)
    )
    return {
        "KG-GFB1": "PASS" if g1_ok else "KILL_FALSIFIED",
        "KG-GFB2": "PASS" if g2_ok else "KILL_FALSIFIED",
        "KG-GFB3": "PASS" if g3_ok else "KILL_FALSIFIED",
        "KG-GFB4": "PASS" if g4_ok else "KILL_FALSIFIED",
        "overall_disposition": "PASS" if (g1_ok and g2_ok and g3_ok and g4_ok) else "KILL_FALSIFIED",
    }


def _build_cpp_parity_source(vec_1536: Sequence[float], gamma_1536: Sequence[float]) -> str:
    vec_lit = ", ".join(f"{v:.9g}f" for v in vec_1536)
    gam_lit = ", ".join(f"{g:.9g}f" for g in gamma_1536)
    return f"""
#include "siligemma_arena.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

static uint64_t pack_valid_a16(uint32_t data11, uint32_t& rng) {{
    uint32_t odd16 = siligemma::a16_encode_data11_cpu(data11 & 0x7FFu);
    uint64_t w = 0ULL;
    for (int j = 0; j < 16; ++j) {{
        int lsb = static_cast<int>((odd16 >> j) & 1u);
        rng = rng * 1664525u + 1013904223u;
        int k = static_cast<int>((rng >> 24) % 7u) - 3;
        int v = 2 * k + lsb;
        if (v < -7) v += 2;
        if (v > 7) v -= 2;
        w |= (static_cast<uint64_t>(static_cast<uint32_t>(v) & 0xFu) << (4 * j));
    }}
    return w;
}}

int main() {{
    siligemma::SlgbHeader hdr32{{}}, hdr128{{}};
    std::array<siligemma::SiliGemmaLayerDesc, siligemma::GEMMA4_NUM_LAYERS> layers32{{}}, layers128{{}};
    std::vector<siligemma::SiliGemmaLinearDesc> lin32, lin128;
    if (siligemma::build_gemma4_descriptor_table(32u, hdr32, layers32, lin32) != siligemma::ValidationError::OK) return 2;
    if (siligemma::build_gemma4_descriptor_table(128u, hdr128, layers128, lin128) != siligemma::ValidationError::OK) return 3;

    const uint32_t ok_toks[3] = {{2u, 42u, 106u}};
    if (siligemma::validate_runtime_contract(hdr32, layers32, lin32, ok_toks, 3u) != siligemma::ValidationError::OK) return 4;
    const uint32_t bad_bos[2] = {{1u, 42u}};
    if (siligemma::validate_runtime_contract(hdr32, layers32, lin32, bad_bos, 2u) != siligemma::ValidationError::ERR_BOS_NOT_2) return 5;
    if (siligemma::validate_runtime_contract(hdr32, layers32, lin32, nullptr, 0u) != siligemma::ValidationError::ERR_EMPTY_TOKENS) return 6;
    const uint32_t oob_toks[2] = {{2u, 262144u}};
    if (siligemma::validate_runtime_contract(hdr32, layers32, lin32, oob_toks, 2u) != siligemma::ValidationError::ERR_BAD_TOKEN_ID) return 15;

    siligemma::SlgbHeader hdr32_f3{{}};
    std::array<siligemma::SiliGemmaLayerDesc, siligemma::GEMMA4_NUM_LAYERS> layers32_f3{{}};
    std::vector<siligemma::SiliGemmaLinearDesc> lin32_f3;
    if (siligemma::build_gemma4_descriptor_table(32u, hdr32_f3, layers32_f3, lin32_f3, 3u) != siligemma::ValidationError::OK) return 16;
    const uint64_t exp_sz = siligemma::expected_slgb_file_bytes(hdr32_f3);
    if (siligemma::validate_slgb_file_layout(hdr32_f3, layers32_f3, lin32_f3, exp_sz, true) != siligemma::ValidationError::OK) return 17;
    auto bad_crc_hdr = hdr32_f3;
    bad_crc_hdr.header_crc32 ^= 0x1u;
    if (siligemma::validate_slgb_file_layout(bad_crc_hdr, layers32_f3, lin32_f3, exp_sz, true) != siligemma::ValidationError::ERR_BAD_CRC32) return 18;
    auto zero_crc_hdr = hdr32_f3;
    zero_crc_hdr.header_crc32 = 0u;
    if (siligemma::validate_slgb_file_layout(zero_crc_hdr, layers32_f3, lin32_f3, exp_sz, false) != siligemma::ValidationError::OK) return 22;
    if (siligemma::validate_slgb_file_layout(hdr32_f3, layers32_f3, lin32_f3, exp_sz - 16ULL, true) != siligemma::ValidationError::ERR_BAD_FILE_SIZE) return 19;
    auto bad_lin = lin32_f3;
    bad_lin[7].words_byte_offset += 64ULL;
    if (siligemma::validate_slgb_file_layout(hdr32_f3, layers32_f3, bad_lin, exp_sz, true) != siligemma::ValidationError::ERR_BAD_WIRE_BYTES) return 20;

    for (uint32_t l = 15u; l < 35u; ++l) {{
        auto bad_ly = layers32;
        bad_ly[l].linear_idx[1] = 0;
        if (siligemma::validate_runtime_contract(hdr32, bad_ly, lin32, ok_toks, 3u) != siligemma::ValidationError::ERR_DEAD_KV_PRESENT) return 7;
    }}
    {{
        auto bad_idx = layers32;
        bad_idx[0].linear_idx[0] = 9999;
        if (siligemma::validate_runtime_contract(hdr32, bad_idx, lin32, ok_toks, 3u) != siligemma::ValidationError::ERR_BAD_LINEAR_COUNT) return 24;
        auto dup_idx = layers32;
        dup_idx[0].linear_idx[0] = dup_idx[0].linear_idx[3];
        if (siligemma::validate_runtime_contract(hdr32, dup_idx, lin32, ok_toks, 3u) != siligemma::ValidationError::ERR_BAD_TOPOLOGY) return 25;
        auto bad_noff = layers32;
        bad_noff[0].norm_offsets_f32[0] = 285000;
        if (siligemma::validate_runtime_contract(hdr32, bad_noff, lin32, ok_toks, 3u) != siligemma::ValidationError::ERR_BAD_DIMS) return 26;
        auto ov_noff = layers32;
        ov_noff[1].norm_offsets_f32[0] = ov_noff[0].norm_offsets_f32[0];
        if (siligemma::validate_runtime_contract(hdr32, ov_noff, lin32, ok_toks, 3u) != siligemma::ValidationError::ERR_BAD_DIMS) return 29;
        auto ov_wire = lin32;
        ov_wire[1].words_byte_offset = ov_wire[0].words_byte_offset;
        if (siligemma::validate_runtime_contract(hdr32, layers32, ov_wire, ok_toks, 3u) != siligemma::ValidationError::ERR_BAD_WIRE_BYTES) return 30;
        auto zero_rt_hdr = hdr32;
        zero_rt_hdr.header_crc32 = 0u;
        if (siligemma::validate_runtime_contract(zero_rt_hdr, layers32, lin32, ok_toks, 3u, true) != siligemma::ValidationError::ERR_BAD_CRC32) return 27;
        if (siligemma::build_gemma4_descriptor_table(128u, hdr128, layers128, lin128, 1u) != siligemma::ValidationError::ERR_BAD_TOPOLOGY) return 32;
        auto bad_g128 = hdr128;
        bad_g128.flags = 1u;
        bad_g128.header_crc32 = siligemma::compute_slgb_header_crc32(bad_g128);
        if (siligemma::validate_runtime_contract(bad_g128, layers128, lin128, ok_toks, 3u) != siligemma::ValidationError::ERR_BAD_TOPOLOGY) return 33;
    }}

    uint32_t rng = 184099u;
    uint64_t cw = pack_valid_a16(0x5A3u, rng);
    if (siligemma::a16_secded_syndrome_cpu(cw) != 0u) return 8;
    uint64_t ded_cw = cw ^ (1ULL << 0u) ^ (1ULL << 12u);
    uint32_t ded_syn = siligemma::a16_secded_syndrome_cpu(ded_cw);
    if (ded_syn == 0u || (ded_syn & 1u) != 0u) return 9;

    siligemma::HoloSphereRazorCanary64 canary{{}};
    canary.ded_count_u32 = 1u;
    if (siligemma::evaluate_canary_watchdog(canary, 0u, false) != siligemma::WatchdogAction::SAFE_HALT_DED_ROLLBACK) return 10;
    canary.ded_count_u32 = 0u;
    canary.iris_runtime_crc_u32 = siligemma::iris_mix_word_cpu(0u, 0x12345678u);
    if (siligemma::evaluate_canary_watchdog(canary, canary.iris_runtime_crc_u32 ^ 1u, true) != siligemma::WatchdogAction::SAFE_HALT_DED_ROLLBACK) return 11;

    const float vec[1536] = {{ {vec_lit} }};
    const float gam[1536] = {{ {gam_lit} }};
    const float tree_sum = siligemma::canonical_tree_sum_block128_f32(vec, 1536u);
    std::vector<float> big20k(20000u, 0.0f);
    for (size_t i = 0u; i < 20000u; ++i) big20k[i] = vec[i % 1536u] * 0.25f;
    const float sum_20k = siligemma::canonical_tree_sum_block128_f32(big20k.data(), big20k.size());
    std::vector<float> rms_out(1536u, 0.0f);
    siligemma::canonical_tree_rmsnorm_f32(vec, gam, rms_out.data(), 1536u);

    const size_t D = 1536u;
    std::vector<float> prev_h(D, 1.0f), cur_h(D), prev_d(D, 0.2f), bad_h(D);
    for (size_t i = 0; i < D; ++i) {{
        cur_h[i] = prev_h[i] + 0.09f;
        bad_h[i] = prev_h[i] + 0.2f * 0.6180f;
    }}
    double rho_ok = 0.0, rho_bad = 0.0;
    float sr_ok = 0.0f, cos_ok = 0.0f, sr_bad = 0.0f, cos_bad = 0.0f;
    if (siligemma::evaluate_riccati_loop_guard(prev_h.data(), cur_h.data(), prev_d.data(), D, 2u, rho_ok, sr_ok, cos_ok) != siligemma::ValidationError::OK) return 12;
    if (siligemma::evaluate_riccati_loop_guard(prev_h.data(), bad_h.data(), prev_d.data(), D, 2u, rho_bad, sr_bad, cos_bad) != siligemma::ValidationError::ERR_RICCATI_RHO_GE_LIMIT) return 13;
    if (siligemma::evaluate_riccati_loop_guard(prev_h.data(), cur_h.data(), prev_d.data(), D, 4u, rho_ok, sr_ok, cos_ok) != siligemma::ValidationError::ERR_LOOP_K_INVALID) return 14;
    if (siligemma::evaluate_riccati_loop_guard(prev_h.data(), cur_h.data(), nullptr, D, 3u, rho_ok, sr_ok, cos_ok) != siligemma::ValidationError::ERR_BAD_DIMS) return 21;
    std::vector<float> zero_d(D, 0.0f);
    double rho_exact = -1.0;
    if (siligemma::evaluate_riccati_loop_guard(prev_h.data(), prev_h.data(), zero_d.data(), D, 3u, rho_exact, sr_ok, cos_ok) != siligemma::ValidationError::OK || rho_exact != 0.0) return 23;
    double rho_z = -1.0;
    float sr_z = 1.0f, cos_z = 0.0f;
    if (siligemma::evaluate_riccati_loop_guard(zero_d.data(), zero_d.data(), zero_d.data(), D, 3u, rho_z, sr_z, cos_z) != siligemma::ValidationError::OK ||
        rho_z != 0.0 || cos_z != 1.0f) return 28;
    std::vector<float> tiny_pos(D, 0.0f), tiny_neg(D, 0.0f);
    tiny_pos[0] = 1.0e-13f;
    tiny_neg[0] = -1.0e-13f;
    if (siligemma::evaluate_riccati_loop_guard(tiny_pos.data(), tiny_neg.data(), zero_d.data(), D, 3u, rho_z, sr_z, cos_z) != siligemma::ValidationError::ERR_S164_STEP_DIVERGENCE ||
        cos_z >= -0.99f) return 31;

    std::printf(
        "{{\\"tree_sum_bits\\": \\"0x%08x\\", \\"sum_20k_bits\\": \\"0x%08x\\", \\"rms0_bits\\": \\"0x%08x\\", \\"rms1535_bits\\": \\"0x%08x\\", "
        "\\"wire_g32\\": %llu, \\"wire_g128\\": %llu, \\"arena_16k\\": %llu, \\"arena_64k_shadow\\": %llu, "
        "\\"arena_128k\\": %llu, \\"slgb_exp_sz\\": %llu, \\"rho_ok\\": %.6f, \\"rho_bad\\": %.6f, \\"all_cpp_r5_passed\\": true}}\\n",
        siligemma::float_bits_u32(tree_sum),
        siligemma::float_bits_u32(sum_20k),
        siligemma::float_bits_u32(rms_out[0]),
        siligemma::float_bits_u32(rms_out[1535]),
        static_cast<unsigned long long>(hdr32.total_wire_bytes),
        static_cast<unsigned long long>(hdr128.total_wire_bytes),
        static_cast<unsigned long long>(siligemma::kArena16K.total_arena_bytes),
        static_cast<unsigned long long>(siligemma::kArena64KShadow.total_arena_bytes),
        static_cast<unsigned long long>(siligemma::kArena128K.total_arena_bytes),
        static_cast<unsigned long long>(exp_sz),
        rho_ok,
        rho_bad
    );
    return 0;
}}
"""


class TestSiliGemmaCPU(unittest.TestCase):
    """Fail-closed R1-R5 CPU parity, pinned receipt, and Form B packer tests."""

    def test_01_repo_hygiene_ast_purity_and_pinned_hashes(self) -> None:
        for rel, exp in EXPECTED_PINNED_SHA256.items():
            got = hashlib.sha256(read_regular_bytes_nofollow(REPO_ROOT / rel)).hexdigest()
            self.assertEqual(got, exp, f"Pinned SHA-256 mismatch on {rel}")

        for p in sorted(REPO_ROOT.rglob("*")):
            if ".git" in p.parts:
                continue
            self.assertFalse(p.is_symlink(), f"Symlink forbidden: {p.name}")
            self.assertNotEqual(p.name, "__pycache__", f"__pycache__ forbidden: {p}")
            if p.is_file():
                suf = p.suffix.lower()
                self.assertNotIn(suf, BINARY_SUFFIXES, f"Binary artifact forbidden: {p.name}")
                self.assertNotIn(suf, WEIGHT_SUFFIXES, f"Weight artifact forbidden: {p.name}")
                raw = read_regular_bytes_nofollow(p)
                self.assertNotEqual(raw[:4], b"\x7fELF", f"ELF header forbidden: {p.name}")
                if suf in (".py", ".cpp", ".hpp", ".cu", ".cuh", ".md", ".json", ".yml", ".yaml", ".txt") or p.name == "Makefile":
                    self.assertNotIn(b"\r", raw, f"CRLF forbidden: {p.name}")
                    txt = raw.decode("utf-8", "strict")
                    self.assertIsNone(HOST_USER_RE.search(txt), f"Personal path leak in {p.name}")
                    if suf == ".py":
                        tree = ast.parse(txt, filename=p.name)
                        for node in ast.walk(tree):
                            self.assertNotIsInstance(node, ast.Assert, f"Bare assert in {p.name}:{getattr(node, 'lineno', 0)}")
                            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                                if isinstance(node.func.value, ast.Name) and node.func.value.id == "random":
                                    if node.func.attr in {"random", "randint", "uniform", "choice", "shuffle", "seed"}:
                                        self.fail(f"Unseeded random.{node.func.attr} in {p.name}")
                    if suf in (".hpp", ".cu", ".cpp"):
                        for bad in FORBIDDEN_ANCHORS:
                            self.assertNotIn(bad, txt, f"Forbidden anchor {bad!r} in {p.name}")

    def test_02_cpp20_reference_parity_and_r5_controls(self) -> None:
        cxx = shutil.which("g++")
        self.assertIsNotNone(cxx, "g++ compiler required")
        rng = random.Random(1842026)
        vec = [f32(rng.gauss(0.0, 1.25)) for _ in range(1536)]
        gam = [f32(rng.uniform(0.75, 1.25)) for _ in range(1536)]
        py_sum = canonical_tree_sum_block128_py(vec)
        py_sum_20k = canonical_tree_sum_block128_py([f32(vec[i % 1536] * 0.25) for i in range(20000)])
        py_rms = canonical_tree_rmsnorm_py(vec, gam, eps=1.0e-6)
        cpp_src = _build_cpp_parity_source(vec, gam)

        with tempfile.TemporaryDirectory(prefix="siligemma_cpp_ref_") as td:
            td_p = Path(td)
            src_p = td_p / "ref_test.cpp"
            bin_p = td_p / "ref_test"
            src_p.write_text(cpp_src, encoding="utf-8")
            c_proc = subprocess.run(
                [str(cxx), "-O3", "-std=c++20", "-Wall", "-Wextra", "-Werror", f"-I{KERNELS_DIR}", str(src_p), "-lm", "-o", str(bin_p)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(c_proc.returncode, 0, f"g++ failed: {c_proc.stderr}")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
            r_proc = subprocess.run([str(bin_p)], capture_output=True, text=True, env=env, check=False)
            self.assertEqual(r_proc.returncode, 0, f"C++20 parity binary failed rc={r_proc.returncode}: {r_proc.stderr}")

        res = parse_strict_json_bytes(r_proc.stdout.encode("utf-8"), label="cpp_ref")
        self.assertTrue(res["all_cpp_r5_passed"])
        self.assertEqual(res["wire_g32"], 1055711232)
        self.assertEqual(res["wire_g128"], 967735296)
        self.assertEqual(res["arena_16k"], 1995364608)
        self.assertEqual(res["arena_64k_shadow"], 2523846912)
        self.assertEqual(res["arena_128k"], 2700007680)
        self.assertLess(res["rho_ok"], 0.618)
        self.assertGreaterEqual(res["rho_bad"], 0.618)
        self.assertEqual(res["tree_sum_bits"], f"0x{f32_bits(py_sum):08x}")
        self.assertEqual(res["sum_20k_bits"], f"0x{f32_bits(py_sum_20k):08x}")
        self.assertEqual(res["slgb_exp_sz"], 2094062080)
        self.assertEqual(res["rms0_bits"], f"0x{f32_bits(py_rms[0]):08x}")
        self.assertEqual(res["rms1535_bits"], f"0x{f32_bits(py_rms[1535]):08x}")

    def test_03_geometry_verdict_recompute_and_packer_contracts(self) -> None:
        geom = packer.build_gemma4_descriptor_geometry(32)
        self.assertEqual(len(geom["linears"]), 276)
        self.assertEqual(geom["total_wire_bytes"], 1055711232)
        sec = packer.compute_slgb_section_offsets(packer.WIRE_BYTES_G32, has_lm_shadow=True, has_u4=True)
        self.assertEqual(sec["total_file_bytes"], 2094062080)
        hdr = packer.pack_slgb_header_and_descs(geom, flags=3)
        self.assertEqual(len(hdr), 64 + 35 * 92 + 276 * 48)
        exp_crc = zlib.crc32(hdr[:60] + b"\x00\x00\x00\x00") & 0xFFFFFFFF
        got_crc = struct.unpack("<I", hdr[60:64])[0]
        self.assertEqual(got_crc, exp_crc)
        self.assertNotEqual(got_crc, 0)

        # Negative control: norm shape mismatch & overlapping tensor spans must fail validation
        fake_hdr: dict[str, Any] = {}
        cur_off = 0
        for d in geom["linears"]:
            nb = d["out_dim"] * d["in_dim"] * 2
            fake_hdr[f"model.language_model.{d['name']}.weight"] = {
                "dtype": "BF16", "shape": [d["out_dim"], d["in_dim"]], "data_offsets": [cur_off, cur_off + nb]
            }
            cur_off += nb
        fake_hdr["model.language_model.embed_tokens.weight"] = {
            "dtype": "BF16", "shape": [packer.GEMMA4_VOCAB, packer.GEMMA4_HIDDEN], "data_offsets": [cur_off, cur_off + packer.EMBED_BF16_BYTES]
        }
        cur_off += packer.EMBED_BF16_BYTES
        for ns in geom["norm_specs"]:
            nb = ns["count"] * 2
            fake_hdr[f"model.language_model.{ns['key']}"] = {
                "dtype": "BF16", "shape": [ns["count"]], "data_offsets": [cur_off, cur_off + nb]
            }
            cur_off += nb
        packer.validate_safetensors_tensor_spans(fake_hdr, geom, data_bytes_avail=cur_off)
        fake_hdr["model.language_model.norm.weight"]["shape"] = [999]
        with self.assertRaises(ValueError):
            packer.validate_safetensors_tensor_spans(fake_hdr, geom, data_bytes_avail=cur_off)
        fake_hdr["model.language_model.norm.weight"]["shape"] = [packer.GEMMA4_HIDDEN]
        fake_hdr["model.language_model.norm.weight"]["data_offsets"] = [0, packer.GEMMA4_HIDDEN * 2]
        with self.assertRaises(ValueError):
            packer.validate_safetensors_tensor_spans(fake_hdr, geom, data_bytes_avail=cur_off)
        fake_hdr["model.language_model.norm.weight"]["data_offsets"] = [0.9, packer.GEMMA4_HIDDEN * 2 + 0.9]
        with self.assertRaises(ValueError):
            packer.validate_safetensors_tensor_spans(fake_hdr, geom, data_bytes_avail=cur_off)

        verdict = parse_strict_json_bytes(read_regular_bytes_nofollow(DOCS_DIR / "STAGE1_VERDICT.json"), label="verdict")
        recomp = _recompute_s184_kill_gates(verdict["raw_measurements"])
        for g_name in ("KG-GFB1", "KG-GFB2", "KG-GFB3", "KG-GFB4"):
            self.assertEqual(recomp[g_name], verdict["kill_gates"][g_name]["disposition"])
        self.assertEqual(recomp["overall_disposition"], verdict["kill_gates"]["overall_disposition"])

        self.assertEqual(a16_secded_syndrome_py(0), 0)
        syn1 = a16_secded_syndrome_py(1 << 20)
        self.assertTrue(syn1 != 0 and (syn1 & 1) == 1 and ((syn1 >> 1) & 15) == 5)
        syn2 = a16_secded_syndrome_py((1 << 0) | (1 << 36))
        self.assertTrue(syn2 != 0 and (syn2 & 1) == 0)

        packer._verify_worker_outputs(
            "PACK_COMPLETE linears=276 embeds=1 norms=262\n",
            "SLGB_VERIFY_PASS flags=3 total_words=142467072 total_scales=71233536 secded_errors=0\n",
        )
        with self.assertRaises(RuntimeError):
            packer._verify_worker_outputs(
                "PACK_COMPLETE linears=275 embeds=1 norms=262\n",
                "SLGB_VERIFY_PASS flags=3 total_words=142467072 total_scales=71233536 secded_errors=0\n",
            )
        with self.assertRaises(RuntimeError):
            packer._verify_worker_outputs(
                "PACK_COMPLETE linears=276 embeds=1 norms=262\n",
                "SLGB_VERIFY_PASS flags=1 total_words=142467072 total_scales=71233536 secded_errors=0\n",
            )
        with self.assertRaises(RuntimeError):
            packer._verify_worker_outputs(
                "PACK_COMPLETE linears=276 embeds=1 norms=262\n",
                "SLGB_VERIFY_PASS flags=3 total_words=142467072 total_scales=123 secded_errors=0\n",
            )

    def test_04_slgb_cpu_worker_quantize_and_tamper_controls(self) -> None:
        rng = random.Random(184007)
        m, k = 16, 128
        bf16_u16 = []
        for _ in range(m * k):
            u = struct.unpack("<I", struct.pack("<f", rng.gauss(0.0, 0.04)))[0]
            bf16_u16.append((u >> 16) & 0xFFFF)

        with tempfile.TemporaryDirectory(prefix="siligemma_pack_unit_") as td:
            td_p = Path(td)
            worker = packer._compile_worker(td_p)
            in_p = td_p / "w.bf16"
            w_p = td_p / "w.u64"
            s_p = td_p / "s.f16"
            in_p.write_bytes(struct.pack(f"<{m * k}H", *bf16_u16))
            proc = subprocess.run([str(worker), "--selftest-quant", str(in_p), str(m), str(k), str(w_p), str(s_p)], capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, f"selftest-quant failed: {proc.stderr}")

            words = struct.unpack(f"<{m * (k // 16)}Q", w_p.read_bytes())
            for w64 in words:
                self.assertEqual(a16_secded_syndrome_py(w64), 0)
                for j in range(16):
                    nib = (w64 >> (4 * j)) & 0xF
                    s_val = nib - 16 if nib >= 8 else nib
                    self.assertGreaterEqual(s_val, -8)
                    self.assertLessEqual(s_val, 7)

            # Negative control: bad magic & zero-filled valid-header sparse .slgb & invalid --selftest-quant args must fail
            bad_slgb = td_p / "bad.slgb"
            bad_slgb.write_bytes(b"\x00" * 64)
            v_bad = subprocess.run([str(worker), "--verify", str(bad_slgb)], capture_output=True, text=True, check=False)
            self.assertNotEqual(v_bad.returncode, 0)

            geom = packer.build_gemma4_descriptor_geometry(32)
            zero_slgb = td_p / "zero_payload.slgb"
            with zero_slgb.open("wb") as fz:
                fz.truncate(packer.compute_slgb_section_offsets(packer.WIRE_BYTES_G32, has_lm_shadow=True, has_u4=True)["total_file_bytes"])
                fz.seek(0)
                fz.write(packer.pack_slgb_header_and_descs(geom, flags=3))
            v_zero = subprocess.run([str(worker), "--verify", str(zero_slgb)], capture_output=True, text=True, check=False)
            self.assertNotEqual(v_zero.returncode, 0)

            q_bad_arg = subprocess.run([str(worker), "--selftest-quant", str(in_p), "not_an_int", str(k), str(w_p), str(s_p)], capture_output=True, text=True, check=False)
            self.assertNotEqual(q_bad_arg.returncode, 0)
            q_bad_sz = subprocess.run([str(worker), "--selftest-quant", str(in_p), "32", str(k), str(w_p), str(s_p)], capture_output=True, text=True, check=False)
            self.assertNotEqual(q_bad_sz.returncode, 0)

            nan_p = td_p / "nan.bf16"
            nan_vals = list(bf16_u16)
            nan_vals[17] = 0x7FC0
            nan_p.write_bytes(struct.pack(f"<{m * k}H", *nan_vals))
            q_nan = subprocess.run([str(worker), "--selftest-quant", str(nan_p), str(m), str(k), str(w_p), str(s_p)], capture_output=True, text=True, check=False)
            self.assertNotEqual(q_nan.returncode, 0)

            ovf_p = td_p / "ovf.bf16"
            ovf_p.write_bytes(struct.pack(f"<{m * k}H", *([0x7F7F] * (m * k))))
            q_ovf = subprocess.run([str(worker), "--selftest-quant", str(ovf_p), str(m), str(k), str(w_p), str(s_p)], capture_output=True, text=True, check=False)
            self.assertNotEqual(q_ovf.returncode, 0)

    def test_05_r5_ptxas_and_sass_must_fail_controls(self) -> None:
        sample_ptxas = "\n".join(
            f"ptxas info    : Function properties for k{i}\n"
            "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
            f"ptxas info    : Used {48 if i == 0 else 32} registers, used 1 barriers, 1664 bytes smem"
            for i in range(20)
        )
        parsed = parse_ptxas_log(sample_ptxas)
        ok = validate_ptxas_and_sass(parsed, {"ldl_insns": 0, "stl_insns": 0}, min_kernels=13)
        self.assertEqual(ok["total_kernels"], 20)
        self.assertEqual(ok["max_registers"], 48)
        bad_reg = parse_ptxas_log(sample_ptxas.replace("Used 48 registers", "Used 49 registers"))
        with self.assertRaises(SiliGemmaGateError):
            validate_ptxas_and_sass(bad_reg, {"ldl_insns": 0, "stl_insns": 0}, min_kernels=13)
        with self.assertRaises(SiliGemmaGateError):
            validate_ptxas_and_sass(parsed, {"ldl_insns": 1, "stl_insns": 0}, min_kernels=13)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSiliGemmaCPU)
    runner = unittest.TextTestRunner(stream=sys.stderr, verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        return 1
    sys.stdout.write(f"SILIGEMMA_CPU_SUITE_PASS tests_ran={result.testsRun} skipped={len(result.skipped)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
