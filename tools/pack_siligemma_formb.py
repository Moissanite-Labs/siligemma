#!/usr/bin/env python3
"""Pure-CPU streaming Form B (.slgb) builder and verifier for google/gemma-4-E2B-it.

Compiles `slgb_cpu_worker.cpp` into a temporary directory (deleted on exit), streams tensors
one at a time from `model.safetensors` (< 250 MiB peak RAM, zero PyTorch / GPU locks), packs
`gemma4_formb_g32.slgb` (276 live A_16-BlockH4 g=32 linears, 60 dead KV tensors omitted,
BF16 embed_tokens + 4.5 bpw lm_head shadow, 262 FP32 norms/scalars, zero-init K=1 identity U-4 slot),
and verifies full [16, 11, 4] Extended Hamming syndromes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any
import zlib

sys.dont_write_bytecode = True

SLGB_MAGIC = 0x534C4734
SLGB_VERSION = 1
GEMMA4_BOS_ID = 2
GEMMA4_EOS_IDS = (1, 106)
GEMMA4_PAD_ID = 0
GEMMA4_VOCAB = 262144
GEMMA4_HIDDEN = 1536
GEMMA4_NUM_LAYERS = 35
GEMMA4_FIRST_SHARED = 15
GEMMA4_PLE_DIM = 256
GEMMA4_SOFTCAP = 30.0
GEMMA4_RMS_EPS = 1.0e-6
LIVE_LINEAR_COUNT = 276
DEAD_KV_OMITTED = 60
WIRE_BYTES_G32 = 1055711232
EMBED_BF16_BYTES = 805306368
LM_SHADOW_G32_BYTES = 226492416
NORM_PARAMS_FP32 = 285987
LOOP_CTRL_BF16_BYTES = 5391360


def _align_up_256(val: int) -> int:
    return (val + 255) & ~255


def build_gemma4_descriptor_geometry(group_size: int = 32) -> dict[str, Any]:
    if group_size not in (32, 128):
        raise ValueError(f"Unsupported group_size={group_size}")
    linears: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []
    words_off = 0
    scales_off = 0
    norm_cursor = 0
    norm_specs: list[dict[str, Any]] = []

    def _push_linear(name: str, layer_idx: int, role_id: int, out_d: int, in_d: int, flags: int) -> int:
        nonlocal words_off, scales_off
        idx = len(linears)
        w_bytes = (out_d * (in_d // 16)) * 8
        s_bytes = (out_d * (in_d // group_size)) * 2
        linears.append({
            "idx": idx,
            "name": name,
            "layer_idx": layer_idx,
            "role_id": role_id,
            "out_dim": out_d,
            "in_dim": in_d,
            "group_size": group_size,
            "flags": flags,
            "words_byte_offset": words_off,
            "scales_byte_offset_rel": scales_off,
            "words_bytes": w_bytes,
            "scales_bytes": s_bytes,
        })
        words_off += w_bytes
        scales_off += s_bytes
        return idx

    def _push_norm(key: str, count: int) -> int:
        nonlocal norm_cursor
        off = norm_cursor
        norm_specs.append({"key": key, "offset": off, "count": count})
        norm_cursor += count
        return off

    for l in range(GEMMA4_NUM_LAYERS):
        is_full = 1 if (l % 5) == 4 else 0
        is_shared = 1 if l >= GEMMA4_FIRST_SHARED else 0
        kv_src = (14 if is_full else 13) if is_shared else l
        hd = 512 if is_full else 256
        ff = 12288 if is_shared else 6144
        q_dim = 8 * hd
        kv_dim = hd
        flags = 0
        if 10 <= l <= 14:
            flags |= 1
        if 25 <= l <= 33:
            flags |= 2

        lin_idx = [-1] * 9
        lin_idx[0] = _push_linear(f"layers.{l}.self_attn.q_proj", l, 0, q_dim, GEMMA4_HIDDEN, flags)
        if not is_shared:
            lin_idx[1] = _push_linear(f"layers.{l}.self_attn.k_proj", l, 1, kv_dim, GEMMA4_HIDDEN, flags)
            lin_idx[2] = _push_linear(f"layers.{l}.self_attn.v_proj", l, 2, kv_dim, GEMMA4_HIDDEN, flags)
        lin_idx[3] = _push_linear(f"layers.{l}.self_attn.o_proj", l, 3, GEMMA4_HIDDEN, q_dim, flags)
        lin_idx[4] = _push_linear(f"layers.{l}.mlp.gate_proj", l, 4, ff, GEMMA4_HIDDEN, flags)
        lin_idx[5] = _push_linear(f"layers.{l}.mlp.up_proj", l, 5, ff, GEMMA4_HIDDEN, flags)
        lin_idx[6] = _push_linear(f"layers.{l}.mlp.down_proj", l, 6, GEMMA4_HIDDEN, ff, flags)
        lin_idx[7] = _push_linear(f"layers.{l}.per_layer_input_gate", l, 7, GEMMA4_PLE_DIM, GEMMA4_HIDDEN, flags)
        lin_idx[8] = _push_linear(f"layers.{l}.per_layer_projection", l, 8, GEMMA4_HIDDEN, GEMMA4_PLE_DIM, flags)

        n_offs = [-1] * 8
        n_offs[0] = _push_norm(f"layers.{l}.input_layernorm.weight", GEMMA4_HIDDEN)
        n_offs[1] = _push_norm(f"layers.{l}.post_attention_layernorm.weight", GEMMA4_HIDDEN)
        n_offs[2] = _push_norm(f"layers.{l}.pre_feedforward_layernorm.weight", GEMMA4_HIDDEN)
        n_offs[3] = _push_norm(f"layers.{l}.post_feedforward_layernorm.weight", GEMMA4_HIDDEN)
        n_offs[4] = _push_norm(f"layers.{l}.post_per_layer_input_norm.weight", GEMMA4_HIDDEN)
        n_offs[5] = _push_norm(f"layers.{l}.self_attn.q_norm.weight", hd)
        if not is_shared:
            n_offs[6] = _push_norm(f"layers.{l}.self_attn.k_norm.weight", hd)
        n_offs[7] = _push_norm(f"layers.{l}.layer_scalar", 1)

        layers.append({
            "layer_idx": l,
            "is_full_attn": is_full,
            "is_kv_shared": is_shared,
            "kv_source_layer": kv_src,
            "head_dim": hd,
            "intermediate_size": ff,
            "linear_idx": lin_idx,
            "norm_offsets_f32": n_offs,
        })

    _push_linear("per_layer_model_projection", 0xFFFFFFFF, 9, GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM, GEMMA4_HIDDEN, 0)
    _push_norm("norm.weight", GEMMA4_HIDDEN)
    _push_norm("per_layer_projection_norm.weight", GEMMA4_PLE_DIM)

    for d in linears:
        d["scales_byte_offset"] = words_off + d["scales_byte_offset_rel"]

    total_wire = words_off + scales_off
    if len(linears) != LIVE_LINEAR_COUNT:
        raise ValueError(f"Expected {LIVE_LINEAR_COUNT} linears, got {len(linears)}")
    if norm_cursor != NORM_PARAMS_FP32:
        raise ValueError(f"Expected {NORM_PARAMS_FP32} norm params, got {norm_cursor}")
    if group_size == 32 and total_wire != WIRE_BYTES_G32:
        raise ValueError(f"Expected {WIRE_BYTES_G32} wire bytes, got {total_wire}")

    return {
        "layers": layers,
        "linears": linears,
        "norm_specs": norm_specs,
        "total_words_bytes": words_off,
        "total_scales_bytes": scales_off,
        "total_wire_bytes": total_wire,
    }


def compute_slgb_section_offsets(wire_bytes: int, *, has_lm_shadow: bool = True, has_u4: bool = True) -> dict[str, int]:
    hdr_b = 64
    layers_b = GEMMA4_NUM_LAYERS * 92
    linears_b = LIVE_LINEAR_COUNT * 48
    off = _align_up_256(hdr_b + layers_b + linears_b)
    off_p1 = off
    off = _align_up_256(off + wire_bytes)
    off_p2_emb = off
    off = off + EMBED_BF16_BYTES
    off_p2_lm = 0
    if has_lm_shadow:
        off_p2_lm = _align_up_256(off)
        off = off_p2_lm + LM_SHADOW_G32_BYTES
    off_p3 = _align_up_256(off)
    off = off_p3 + NORM_PARAMS_FP32 * 4
    off_p4 = 0
    if has_u4:
        off_p4 = _align_up_256(off)
        off = off_p4 + LOOP_CTRL_BF16_BYTES
    return {
        "off_p1_wire": off_p1,
        "off_p2_embed": off_p2_emb,
        "off_p2_lm_shadow": off_p2_lm,
        "off_p3_norms": off_p3,
        "off_p4_loop": off_p4,
        "total_file_bytes": off,
    }


def pack_slgb_header_and_descs(geom: dict[str, Any], *, flags: int = 3) -> bytes:
    group_size = int(geom["linears"][0]["group_size"])
    hdr_prefix = struct.pack(
        "<7I6H2fQI",
        SLGB_MAGIC,
        SLGB_VERSION,
        GEMMA4_BOS_ID,
        GEMMA4_EOS_IDS[0],
        GEMMA4_EOS_IDS[1],
        GEMMA4_PAD_ID,
        GEMMA4_VOCAB,
        GEMMA4_HIDDEN,
        GEMMA4_NUM_LAYERS,
        20,
        group_size,
        LIVE_LINEAR_COUNT,
        DEAD_KV_OMITTED,
        GEMMA4_SOFTCAP,
        GEMMA4_RMS_EPS,
        int(geom["total_wire_bytes"]),
        int(flags),
    )
    crc = zlib.crc32(hdr_prefix + b"\x00\x00\x00\x00") & 0xFFFFFFFF
    hdr = hdr_prefix + struct.pack("<I", crc)
    chunks = [hdr]
    for ly in geom["layers"]:
        chunks.append(struct.pack(
            "<6I17i",
            ly["layer_idx"],
            ly["is_full_attn"],
            ly["is_kv_shared"],
            ly["kv_source_layer"],
            ly["head_dim"],
            ly["intermediate_size"],
            *ly["linear_idx"],
            *ly["norm_offsets_f32"],
        ))
    for d in geom["linears"]:
        chunks.append(struct.pack(
            "<2Q3i3I8x",
            d["words_byte_offset"],
            d["scales_byte_offset"],
            d["out_dim"],
            d["in_dim"],
            d["group_size"],
            d["layer_idx"],
            d["role_id"],
            d["flags"],
        ))
    return b"".join(chunks)


def _f32_to_bf16_u16(val: float) -> int:
    u = struct.unpack("<I", struct.pack("<f", float(val)))[0]
    if (u & 0x7FFFFFFF) > 0x7F800000:
        return ((u >> 16) | 0x0040) & 0xFFFF
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) & 0xFFFFFFFF
    return (u >> 16) & 0xFFFF


def _build_zero_init_u4_blob(seed: int = 1842533) -> bytes:
    """Build exact 5,391,360 B Partition 4 with K=1 identity zero gates (wf=0, wg=0, mq=0)."""
    rng = random.Random(int(seed))
    std = 1.0 / math.sqrt(3072.0)
    # 9 layers * 3 gates * (wd[16*3072] + wf[1536*16] + wg[1536*16] + mq[1536]) BF16
    zero_tail = bytes(2 * (1536 * 16 + 1536 * 16 + 1536))
    parts: list[bytes] = []
    for _li in range(9):
        for _gi in range(3):
            wd_arr = [_f32_to_bf16_u16(rng.gauss(0.0, std)) for _ in range(16 * 3072)]
            parts.append(struct.pack(f"<{len(wd_arr)}H", *wd_arr))
            parts.append(zero_tail)
    blob = b"".join(parts)
    if len(blob) != LOOP_CTRL_BF16_BYTES:
        raise ValueError(f"Unexpected U-4 blob length: {len(blob)} != {LOOP_CTRL_BF16_BYTES}")
    return blob


def _compile_worker(td: Path) -> Path:
    cxx = shutil.which("g++")
    if not cxx:
        raise RuntimeError("g++ compiler required for CPU Form B packer/verifier")
    tools_dir = Path(__file__).resolve().parent
    src = tools_dir / "slgb_cpu_worker.cpp"
    inc = tools_dir.parent / "kernels"
    bin_p = td / "slgb_cpu_worker"
    cmd = [cxx, "-O3", "-std=c++20", "-fopenmp", "-Wall", "-Wextra", "-Werror", f"-I{inc}", str(src), "-lm", "-o", str(bin_p)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        # Retry without -fopenmp if environment lacks libgomp
        cmd_no_omp = [c for c in cmd if c != "-fopenmp"]
        proc = subprocess.run(cmd_no_omp, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed compiling slgb_cpu_worker.cpp:\n{proc.stderr}")
    return bin_p


def validate_safetensors_tensor_spans(hdr: dict[str, Any], geom: dict[str, Any]) -> None:
    """Validate exact dtype, shape, and byte-span length for every linear, embedding, and norm tensor."""
    for d in geom["linears"]:
        key = f"model.language_model.{d['name']}.weight"
        entry = hdr[key]
        if entry["dtype"] != "BF16":
            raise ValueError(f"Expected BF16 for {key}, got {entry['dtype']}")
        if list(entry["shape"]) != [d["out_dim"], d["in_dim"]]:
            raise ValueError(f"Shape mismatch for {key}: {entry['shape']}")
        exp_b = int(d["out_dim"]) * int(d["in_dim"]) * 2
        got_b = int(entry["data_offsets"][1]) - int(entry["data_offsets"][0])
        if got_b != exp_b:
            raise ValueError(f"Byte-span mismatch for {key}: {got_b} != {exp_b}")

    emb_entry = hdr["model.language_model.embed_tokens.weight"]
    if emb_entry["dtype"] != "BF16" or list(emb_entry["shape"]) != [GEMMA4_VOCAB, GEMMA4_HIDDEN]:
        raise ValueError("Unexpected embed_tokens shape/dtype")
    if int(emb_entry["data_offsets"][1]) - int(emb_entry["data_offsets"][0]) != EMBED_BF16_BYTES:
        raise ValueError("Unexpected embed_tokens byte span")

    for ns in geom["norm_specs"]:
        key = f"model.language_model.{ns['key']}"
        entry = hdr[key]
        if entry["dtype"] != "BF16":
            raise ValueError(f"Expected BF16 norm {key}, got {entry['dtype']}")
        expected_shape = [] if int(ns["count"]) == 1 and list(entry["shape"]) == [] else [int(ns["count"])]
        if list(entry["shape"]) != expected_shape:
            raise ValueError(f"Norm shape mismatch for {key}: {entry['shape']} != {expected_shape}")
        got_b = int(entry["data_offsets"][1]) - int(entry["data_offsets"][0])
        if got_b != int(ns["count"]) * 2:
            raise ValueError(f"Norm byte-span mismatch for {key}: {got_b} != {int(ns['count']) * 2}")


def pack_safetensors_to_slgb(st_path: Path, slgb_out: Path) -> dict[str, Any]:
    """Stream google/gemma-4-E2B-it model.safetensors -> g=32 Form B .slgb in < 250 MiB RAM."""
    st_resolved = st_path.resolve()
    with st_resolved.open("rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen).decode("utf-8"))
    hdr.pop("__metadata__", None)
    data_base = 8 + int(hlen)

    geom = build_gemma4_descriptor_geometry(32)
    validate_safetensors_tensor_spans(hdr, geom)
    offsets = compute_slgb_section_offsets(WIRE_BYTES_G32, has_lm_shadow=True, has_u4=True)
    hdr_bytes = pack_slgb_header_and_descs(geom, flags=3)

    job_lines: list[str] = []
    for d in geom["linears"]:
        key = f"model.language_model.{d['name']}.weight"
        entry = hdr[key]
        in_off = data_base + int(entry["data_offsets"][0])
        out_w = offsets["off_p1_wire"] + int(d["words_byte_offset"])
        out_s = offsets["off_p1_wire"] + int(d["scales_byte_offset"])
        job_lines.append(f"LINEAR {d['out_dim']} {d['in_dim']} {in_off} {out_w} {out_s}")

    emb_entry = hdr["model.language_model.embed_tokens.weight"]
    emb_in_off = data_base + int(emb_entry["data_offsets"][0])
    lm_w_off = offsets["off_p2_lm_shadow"]
    lm_s_off = offsets["off_p2_lm_shadow"] + 201326592
    job_lines.append(f"EMBED {GEMMA4_VOCAB} {GEMMA4_HIDDEN} 16384 {emb_in_off} {offsets['off_p2_embed']} {lm_w_off} {lm_s_off}")

    for ns in geom["norm_specs"]:
        key = f"model.language_model.{ns['key']}"
        entry = hdr[key]
        in_off = data_base + int(entry["data_offsets"][0])
        out_off = offsets["off_p3_norms"] + int(ns["offset"]) * 4
        job_lines.append(f"NORM {ns['count']} {in_off} {out_off}")

    slgb_out.parent.mkdir(parents=True, exist_ok=True)
    tmp_slgb = slgb_out.with_name(f".{slgb_out.name}.tmp.{os.getpid()}")
    with tmp_slgb.open("wb") as fout:
        fout.truncate(offsets["total_file_bytes"])
        fout.seek(0)
        fout.write(hdr_bytes)
        fout.seek(offsets["off_p4_loop"])
        fout.write(_build_zero_init_u4_blob())

    try:
        with tempfile.TemporaryDirectory(prefix="siligemma_pack_") as td:
            td_p = Path(td)
            worker = _compile_worker(td_p)
            jobs_p = td_p / "jobs.txt"
            jobs_p.write_text("\n".join(job_lines) + "\n", encoding="utf-8")

            p1 = subprocess.run([str(worker), "--pack", str(st_resolved), str(tmp_slgb), str(jobs_p)], capture_output=True, text=True, check=False)
            if p1.returncode != 0:
                raise RuntimeError(f"Worker --pack failed:\n{p1.stderr}\n{p1.stdout}")
            p2 = subprocess.run([str(worker), "--verify", str(tmp_slgb)], capture_output=True, text=True, check=False)
            if p2.returncode != 0:
                raise RuntimeError(f"Worker --verify failed:\n{p2.stderr}\n{p2.stdout}")
        os.replace(tmp_slgb, slgb_out)
    finally:
        if tmp_slgb.exists():
            tmp_slgb.unlink()

    h = hashlib.sha256()
    with slgb_out.open("rb") as f:
        for ch in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(ch)

    return {
        "status": "PACK_AND_VERIFY_PASS",
        "slgb_path": slgb_out.name,
        "slgb_bytes": slgb_out.stat().st_size,
        "slgb_sha256": h.hexdigest(),
        "pack_stdout": p1.stdout.strip(),
        "verify_stdout": p2.stdout.strip(),
    }


def verify_slgb_file(slgb_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="siligemma_verify_") as td:
        worker = _compile_worker(Path(td))
        proc = subprocess.run([str(worker), "--verify", str(slgb_path)], capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Verify failed:\n{proc.stderr}\n{proc.stdout}")
    return {"status": "VERIFY_PASS", "slgb_bytes": slgb_path.stat().st_size, "output": proc.stdout.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SiliGemma Form B (.slgb) CPU packer & SECDED verifier")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_pk = sub.add_parser("pack", help="Pack model.safetensors into Form B g=32 .slgb (< 250 MiB RAM)")
    p_pk.add_argument("--safetensors", type=Path, required=True)
    p_pk.add_argument("--out", type=Path, required=True)
    p_vf = sub.add_parser("verify", help="Verify .slgb header, descriptors, and full A_16 SECDED")
    p_vf.add_argument("slgb", type=Path)

    args = parser.parse_args(argv)
    if args.cmd == "pack":
        out = pack_safetensors_to_slgb(args.safetensors, args.out)
    else:
        out = verify_slgb_file(args.slgb)
    sys.stdout.write(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
