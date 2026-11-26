#!/usr/bin/env python3
"""Multi-strip, mid-run promotion, --fast, and hidden-flag regression suite
for the clean-room BED INT8 GRM engine (comparing against --grm-legacy-engine).

Usage:
    python3 tests/test_grm_multistrip.py [GCTAG_BIN] [WORK_DIR]
"""

import array
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile

GCTAG_BIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join("build", "GCTAg")
USER_WORK = sys.argv[2] if len(sys.argv) > 2 else ""

CODE = {2: 0b00, 1: 0b10, 0: 0b11, None: 0b01}
HDR_FMT = "<8sIIIIQQdQddQQQQd16s"
HDR_SIZE = struct.calcsize(HDR_FMT)


def run_cmd(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + "\n" + proc.stderr


def write_plink(prefix, n, snps, fam_lines):
    nb = (n + 3) // 4
    with open(prefix + ".bed", "wb") as bf:
        bf.write(bytes([0x6C, 0x1B, 0x01]))
        rem = n & 3
        for _, _, _, _, _, g in snps:
            buf = bytearray(nb)
            for i, v in enumerate(g):
                buf[i >> 2] |= CODE[v] << (2 * (i & 3))
            if rem:
                buf[-1] |= (0b01010101 << (2 * rem)) & 0xFF
            bf.write(bytes(buf))
    with open(prefix + ".bim", "w") as bm:
        for idx, (chrom, rsid, bp, a1, a2, _) in enumerate(snps):
            bm.write(f"{chrom}\t{rsid}\t0\t{bp}\t{a1}\t{a2}\n")
    with open(prefix + ".fam", "w") as fm:
        fm.write("\n".join(fam_lines) + "\n")


def generate_datasets(data_dir):
    rng = random.Random(20261123)
    n = 603  # Deliberately non-multiple of 4, 64, and 256
    fam_lines = [f"F{i} I{i} 0 0 {(i % 2) + 1} -9" for i in range(n)]

    def make_snps(m_zero, m_miss):
        snps = []
        for j in range(m_zero + m_miss):
            p = 0.08 + 0.84 * rng.random()
            mr = 0.0 if j < m_zero else (0.05 if j % 7 != 0 else 0.12)
            g = []
            for i in range(n):
                if mr > 0.0 and (rng.random() < mr or (j >= m_zero and i in (5, 17, 130, 412) and j % 11 == 0)):
                    g.append(None)
                else:
                    g.append((1 if rng.random() < p else 0) + (1 if rng.random() < p else 0))
            snps.append(("1", f"snp_{j}", 1000 + j * 100, "A", "G", g))
        return snps

    write_plink(os.path.join(data_dir, "zero_miss"), n, make_snps(300, 0), fam_lines)
    write_plink(os.path.join(data_dir, "with_miss"), n, make_snps(0, 300), fam_lines)
    write_plink(os.path.join(data_dir, "late_miss"), n, make_snps(128, 172), fam_lines)

    # All monomorphic / all missing -> 0 valid SNPs edge case
    mono_snps = []
    for j in range(80):
        val = 0 if (j % 3 == 0) else (2 if (j % 3 == 1) else None)
        mono_snps.append(("1", f"mono_{j}", 1000 + j * 50, "A", "G", [val] * n))
    write_plink(os.path.join(data_dir, "mono_all"), n, mono_snps, fam_lines)

    # Edge case: samples 0 and 65 have 100% missing genotypes
    all_miss_snps = make_snps(0, 180)
    for _, _, _, _, _, g in all_miss_snps:
        g[0] = None
        g[65] = None
    write_plink(os.path.join(data_dir, "all_miss_sample"), n, all_miss_snps, fam_lines)

    keep_idx = sorted(rng.sample(range(n), 417))
    with open(os.path.join(data_dir, "keep_sub.txt"), "w") as kf:
        for i in keep_idx:
            kf.write(f"F{i} I{i}\n")


def read_f16(path):
    with open(path, "rb") as f:
        hdr_raw = f.read(HDR_SIZE)
        hdr = struct.unpack(HDR_FMT, hdr_raw)
        magic, ver, flags, dtype, hbytes, n, valid_m, m_mean, id_hash, tr_k, tr_k2, diag_off, body_off, rs, re_row, mtd_w, _ = hdr
        assert magic == b"GCTAG16\x00", f"Bad f16 magic: {magic!r}"
        assert ver == 1 and dtype == 1 and hbytes == 128, f"Bad header version/dtype/size: {hdr[:5]}"
        f.seek(diag_off)
        diag_bytes = f.read(n * 4)
        f.seek(body_off)
        body_bytes = f.read()
        expected_body = ((re_row * (re_row + 1) - rs * (rs + 1)) // 2) * 2
        assert len(body_bytes) == expected_body, f"Body size {len(body_bytes)} != {expected_body}"
    return {
        "flags": flags,
        "n": n,
        "valid_m": valid_m,
        "m_mean": m_mean,
        "id_hash": id_hash,
        "tr_k": tr_k,
        "tr_k2": tr_k2,
        "mtd_w": mtd_w,
        "diag": diag_bytes,
        "body": body_bytes,
    }


def compare_f16(new_pfx, ref_pfx, name):
    nf = read_f16(new_pfx + ".grm.f16")
    rf = read_f16(ref_pfx + ".grm.f16")
    for key in ("flags", "n", "valid_m", "id_hash"):
        if nf[key] != rf[key]:
            raise AssertionError(f"{name}: f16 {key} mismatch {nf[key]} vs {rf[key]}")
    if abs(nf["m_mean"] - rf["m_mean"]) > 1e-12:
        raise AssertionError(f"{name}: f16 m_mean mismatch {nf['m_mean']} vs {rf['m_mean']}")
    if abs(nf["mtd_w"] - rf["mtd_w"]) > 1e-12:
        raise AssertionError(f"{name}: f16 mtd_w mismatch {nf['mtd_w']} vs {rf['mtd_w']}")
    rel_trk = abs(nf["tr_k"] - rf["tr_k"]) / max(1.0, abs(rf["tr_k"]))
    rel_trk2 = abs(nf["tr_k2"] - rf["tr_k2"]) / max(1.0, abs(rf["tr_k2"]))
    if rel_trk > 1e-12 or rel_trk2 > 1e-11:
        raise AssertionError(f"{name}: f16 trace mismatch rel_trk={rel_trk} rel_trk2={rel_trk2}")
    if nf["diag"] != rf["diag"]:
        raise AssertionError(f"{name}: f16 float32 diagonal bytes differ")
    if nf["body"] != rf["body"]:
        raise AssertionError(f"{name}: f16 packed fp16 body bytes differ")
    for ext in (".grm.id", ".grm.N.bin"):
        np = new_pfx + ext
        rp = ref_pfx + ext
        if os.path.exists(np) != os.path.exists(rp):
            raise AssertionError(f"{name}: {ext} existence mismatch")
        if os.path.exists(np) and open(np, "rb").read() != open(rp, "rb").read():
            raise AssertionError(f"{name}: {ext} byte mismatch")


def collect_grm_exts(prefix):
    parent = os.path.dirname(prefix)
    base = os.path.basename(prefix)
    return sorted(
        f[len(base):]
        for f in os.listdir(parent)
        if f.startswith(base + ".") and ".grm." in f[len(base):]
    )


def compare_byte_identical(new_pfx, ref_pfx, name):
    new_exts = collect_grm_exts(new_pfx)
    ref_exts = collect_grm_exts(ref_pfx)
    if new_exts != ref_exts or not new_exts:
        raise AssertionError(f"{name}: output extensions {new_exts} vs {ref_exts}")
    for ext in new_exts:
        nb = open(new_pfx + ext, "rb").read()
        rb = open(ref_pfx + ext, "rb").read()
        if nb != rb:
            if ext.endswith(".bin") and len(nb) == len(rb):
                n = len(nb) // 4
                fa = array.array("f"); fa.frombytes(nb)
                fb = array.array("f"); fb.frombytes(rb)
                nd = sum(1 for a, b in zip(fa, fb) if a != b)
                md = max(abs(a - b) for a, b in zip(fa, fb))
                raise AssertionError(f"{name} {ext}: DIFF {nd}/{n} floats max|d|={md:.3e}")
            raise AssertionError(f"{name} {ext}: byte mismatch (len {len(nb)} vs {len(rb)})")


def run_pair(name, base_args, dev_args, work_dir, is_fast=False, threads=4, check_profile=False):
    new_pfx = os.path.join(work_dir, f"{name}.new")
    ref_pfx = os.path.join(work_dir, f"{name}.ref")

    rc_new, log_new = run_cmd(
        [GCTAG_BIN, "grm", "make"] + base_args + dev_args + ["--threads", str(threads), "--out", new_pfx]
    )
    if rc_new != 0:
        raise AssertionError(f"{name}: new binary failed rc={rc_new}\n{log_new}")

    rc_ref, log_ref = run_cmd(
        [GCTAG_BIN, "grm", "make"] + base_args + ["--grm-legacy-engine", "--threads", str(threads), "--out", ref_pfx]
    )
    if rc_ref != 0:
        raise AssertionError(f"{name}: legacy engine failed rc={rc_ref}\n{log_ref}")

    if "GRM profile (s):" in log_ref:
        raise AssertionError(f"{name}: legacy unexpectedly emitted profile output")
    if check_profile:
        m = re.search(
            r"GRM profile \(s\): init=\d+\.\d+ io_wait=\d+\.\d+ count\+decode=\d+\.\d+ miss=\d+\.\d+ gemm=\d+\.\d+ flush=\d+\.\d+",
            log_new,
        )
        if not m:
            raise AssertionError(f"{name}: --grm-profile output missing or malformed:\n{log_new}")

    if is_fast:
        compare_f16(new_pfx, ref_pfx, name)
    else:
        compare_byte_identical(new_pfx, ref_pfx, name)
    print(f"  PASS {name}")
    return log_new


def decode_fp16_array(raw_bytes):
    arr = array.array("H")
    arr.frombytes(raw_bytes)
    out = []
    for h in arr:
        s = (h >> 15) & 1
        e = (h >> 10) & 0x1F
        m = h & 0x3FF
        if e == 0:
            v = (m / 1024.0) * (2.0 ** -14)
        elif e == 31:
            v = float("nan") if m else float("inf")
        else:
            v = (1.0 + m / 1024.0) * (2.0 ** (e - 15))
        out.append(-v if s else v)
    return out


def compute_exact_impute_mean_grm(bfile_pfx, keep_file=None):
    fam = [line.split() for line in open(bfile_pfx + ".fam") if line.strip()]
    n_raw = len(fam)
    if keep_file:
        keep_set = {tuple(line.split()[:2]) for line in open(keep_file) if line.strip()}
        keep_idx = [i for i, r in enumerate(fam) if (r[0], r[1]) in keep_set]
    else:
        keep_idx = list(range(n_raw))
    n = len(keep_idx)
    nb = (n_raw + 3) // 4
    m_raw = sum(1 for _ in open(bfile_pfx + ".bim"))
    code_map = {0b00: 2.0, 0b10: 1.0, 0b11: 0.0, 0b01: None}

    q = 1 if m_raw == 0 else max(1, min(63, int(math.floor(math.sqrt(2000000000.0 / m_raw) * 0.5))))
    centered_cols = []
    quant_cols = []
    diag_err_sum = [0.0] * n
    sd_sum = 0.0
    valid_m = 0
    with open(bfile_pfx + ".bed", "rb") as bf:
        magic = bf.read(3)
        assert magic == b"\x6c\x1b\x01"
        for _ in range(m_raw):
            rec = bf.read(nb)
            vals = []
            obs_sum = 0.0
            n_obs = 0
            for raw_i in keep_idx:
                code = (rec[raw_i >> 2] >> (2 * (raw_i & 3))) & 3
                v = code_map[code]
                vals.append(v)
                if v is not None:
                    obs_sum += v
                    n_obs += 1
            if n_obs == 0:
                continue
            af = obs_sum / (2.0 * n_obs)
            mu = 2.0 * af
            sd = mu * (1.0 - af)
            if sd < 1e-50:
                continue
            sd_sum += sd
            valid_m += 1
            mu_q = q * mu
            mq = max(0, min(2 * q, int(math.floor(mu_q + 0.5))))
            err_q2 = (mq - mu_q) ** 2
            centered_cols.append([(0.0 if v is None else (v - mu)) for v in vals])
            qcol = []
            for s_idx, v in enumerate(vals):
                if v is None:
                    qcol.append((mq - mu_q) / q)
                    diag_err_sum[s_idx] += err_q2 / (q * q)
                else:
                    qcol.append(v - mu)
            quant_cols.append(qcol)

    tri = []
    q_tri = []
    if valid_m == 0:
        tri = [0.0] * (n * (n + 1) // 2)
        return tri, tri, 0, 1.0
    for i in range(n):
        for j in range(i + 1):
            dot = 0.0
            qdot = 0.0
            for col, qcol in zip(centered_cols, quant_cols):
                dot += col[i] * col[j]
                qdot += qcol[i] * qcol[j]
            if i == j:
                qdot = max(0.0, qdot - diag_err_sum[i])
            tri.append(dot / sd_sum)
            q_tri.append(qdot / sd_sum)
    return tri, q_tri, valid_m, valid_m / sd_sum


def verify_impute_mean_case(name, base_args, dev_args, bfile_pfx, runs_dir, keep_file=None, is_fast=False, check_profile=False):
    out_pfx = os.path.join(runs_dir, f"{name}.imp")
    rc, log = run_cmd([GCTAG_BIN, "grm", "make"] + base_args + dev_args + ["--threads", "4", "--out", out_pfx])
    if rc != 0:
        raise AssertionError(f"{name}: impute-mean failed rc={rc}\n{log}")
    if check_profile:
        m = re.search(
            r"GRM profile \(s\): init=\d+\.\d+ io_wait=\d+\.\d+ count\+decode=\d+\.\d+ miss=([0-9.]+) gemm=\d+\.\d+ flush=\d+\.\d+",
            log,
        )
        if not m:
            raise AssertionError(f"{name}: --grm-profile missing:\n{log}")
        if float(m.group(1)) != 0.0:
            raise AssertionError(f"{name}: expected miss=0.000 under --impute-mean, got miss={m.group(1)}")

    ref_tri, qref_tri, valid_m, _ = compute_exact_impute_mean_grm(bfile_pfx, keep_file=keep_file)
    if is_fast:
        assert not os.path.exists(out_pfx + ".grm.N.bin"), f"{name}: .grm.N.bin should be omitted for --fast --impute-mean"
        f16 = read_f16(out_pfx + ".grm.f16")
        assert (f16["flags"] & (1 << 3)) != 0, f"{name}: kGrmF16FlagZeroMissing should be set"
        assert f16["valid_m"] == valid_m, f"{name}: valid_m {f16['valid_m']} != {valid_m}"
        assert abs(f16["m_mean"] - valid_m) <= 1e-12, f"{name}: m_mean {f16['m_mean']} != {valid_m}"
        diag = array.array("f"); diag.frombytes(f16["diag"])
        body = decode_fp16_array(f16["body"])
        n = f16["n"]
        idx = 0
        max_diag_diff = 0.0
        max_off_diff = 0.0
        max_q_diag_diff = 0.0
        for i in range(n):
            for j in range(i + 1):
                if i == j:
                    d_err = abs(diag[i] - ref_tri[idx])
                    qd_err = abs(diag[i] - qref_tri[idx])
                    if d_err > max_diag_diff:
                        max_diag_diff = d_err
                    if qd_err > max_q_diag_diff:
                        max_q_diag_diff = qd_err
                else:
                    o_err = abs(body[idx] - ref_tri[idx])
                    if o_err > max_off_diff:
                        max_off_diff = o_err
                idx += 1
        assert max_q_diag_diff <= 2e-6, f"{name}: diag vs quant-ref max diff {max_q_diag_diff:.3e} > 2e-6"
        assert max_diag_diff <= 2e-6, f"{name}: diag max diff {max_diag_diff:.3e} > 2e-6"
        assert max_off_diff <= 2.5e-3, f"{name}: fp16 off-diag max diff {max_off_diff:.3e} > 2.5e-3"
    else:
        nb = open(out_pfx + ".grm.bin", "rb").read()
        nn = open(out_pfx + ".grm.N.bin", "rb").read()
        got_g = array.array("f"); got_g.frombytes(nb)
        got_n = array.array("f"); got_n.frombytes(nn)
        assert len(got_g) == len(ref_tri), f"{name}: len {len(got_g)} != {len(ref_tri)}"
        for nv in got_n:
            assert nv == float(valid_m), f"{name}: expected constant N={valid_m}, got {nv}"
        n = int((math.isqrt(1 + 8 * len(ref_tri)) - 1) // 2)
        idx = 0
        max_diag_diff = 0.0
        max_off_diff = 0.0
        max_q_diff = 0.0
        for i in range(n):
            for j in range(i + 1):
                err = abs(got_g[idx] - ref_tri[idx])
                qerr = abs(got_g[idx] - qref_tri[idx])
                if qerr > max_q_diff:
                    max_q_diff = qerr
                if i == j:
                    if err > max_diag_diff:
                        max_diag_diff = err
                else:
                    if err > max_off_diff:
                        max_off_diff = err
                idx += 1
        assert max_q_diff <= 2e-6, f"{name}: f32 vs quant-ref max diff {max_q_diff:.3e} > 2e-6"
        assert max_diag_diff <= 2e-6, f"{name}: diag max diff {max_diag_diff:.3e} > 2e-6"
        assert max_off_diff <= 1.5e-3, f"{name}: f32 off-diag max diff {max_off_diff:.3e} > 1.5e-3"
    print(f"  PASS {name} (diag_err={max_diag_diff:.2e}, off_err={max_off_diff:.2e})")


def main():
    if not os.path.exists(GCTAG_BIN):
        sys.exit(f"Binary not found: {GCTAG_BIN}")

    owned_tmp = not USER_WORK
    work_dir = USER_WORK if USER_WORK else tempfile.mkdtemp(prefix="gcta_multistrip_")
    os.makedirs(work_dir, exist_ok=True)
    data_dir = os.path.join(work_dir, "data")
    runs_dir = os.path.join(work_dir, "runs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    try:
        generate_datasets(data_dir)
        D = lambda f: os.path.join(data_dir, f)

        # 1. Hidden dev options omitted from --help & invalid strip-rows rejected
        rc_help, help_out = run_cmd([GCTAG_BIN, "grm", "make", "--help"])
        assert rc_help == 0, f"--help exited with {rc_help}"
        for hidden_flag in (
            "--grm-strip-rows",
            "--grm-layout",
            "--grm-gemm-col-chunk",
            "--grm-diag-mode",
            "--grm-diag-block",
            "--grm-io-slot-mb",
            "--grm-legacy-engine",
            "--grm-profile",
        ):
            assert hidden_flag not in help_out, f"Hidden flag leaked into --help: {hidden_flag}"
        print("  PASS hidden_flags_omitted_from_help")

        rc_bad, _ = run_cmd(
            [GCTAG_BIN, "grm", "make", "--bfile", D("zero_miss"), "--alg", "1",
             "--grm-strip-rows", "100", "--out", os.path.join(runs_dir, "bad_strip")]
        )
        assert rc_bad != 0, "Expected non-zero exit for invalid --grm-strip-rows 100"
        print("  PASS invalid_strip_rows_rejected")

        # 2. Multi-strip layout x diag-mode x col-chunk matrix
        for ds in ("zero_miss", "with_miss"):
            for layout in ("rowmajor", "colmajor"):
                for diag in ("fused", "syrk"):
                    for chunk in ("0", "128", "256"):
                        if ds == "with_miss" and chunk == "256":
                            continue
                        run_pair(
                            f"{ds}_{layout}_{diag}_c{chunk}",
                            ["--bfile", D(ds), "--alg", "1", "--n-markers", "64"],
                            ["--grm-strip-rows", "64", "--grm-layout", layout,
                             "--grm-diag-mode", diag, "--grm-diag-block", "64",
                             "--grm-gemm-col-chunk", chunk, "--grm-io-slot-mb", "1"],
                            runs_dir,
                        )
            for strip_h in ("128", "256"):
                run_pair(
                    f"{ds}_strip{strip_h}",
                    ["--bfile", D(ds), "--alg", "1", "--n-markers", "64"],
                    ["--grm-strip-rows", strip_h],
                    runs_dir,
                )

        # 3. Mid-run i32 -> f64 promotion (2 clean blocks of 64 markers, then missingness in block 3+)
        for layout in ("rowmajor", "colmajor"):
            for diag in ("fused", "syrk"):
                run_pair(
                    f"late_miss_{layout}_{diag}",
                    ["--bfile", D("late_miss"), "--alg", "1", "--n-markers", "64"],
                    ["--grm-strip-rows", "128", "--grm-layout", layout,
                     "--grm-diag-mode", diag, "--grm-diag-block", "64"],
                    runs_dir,
                )

        # 4. Tiled multi-tile execution (Tile-1 valid-marker cache + budget adaptation)
        for ds in ("zero_miss", "with_miss", "late_miss"):
            for layout in ("rowmajor", "colmajor"):
                run_pair(
                    f"{ds}_tiled_{layout}",
                    ["--bfile", D(ds), "--alg", "1", "--n-markers", "64", "--tile-budget", "0.0005"],
                    ["--grm-strip-rows", "64", "--grm-layout", layout],
                    runs_dir,
                )

        # 5. Native --fast (.grm.f16) dense & tiled across zero_miss, with_miss, late_miss
        for ds in ("zero_miss", "with_miss", "late_miss"):
            run_pair(
                f"{ds}_fast_dense",
                ["--bfile", D(ds), "--fast", "--n-markers", "64"],
                ["--grm-strip-rows", "64", "--grm-io-slot-mb", "1", "--grm-profile"],
                runs_dir,
                is_fast=True,
                check_profile=True,
            )
            run_pair(
                f"{ds}_fast_tiled",
                ["--bfile", D(ds), "--fast", "--n-markers", "64", "--tile-budget", "0.0005"],
                ["--grm-strip-rows", "64"] + (["--grm-profile"] if ds == "zero_miss" else []),
                runs_dir,
                is_fast=True,
                check_profile=(ds == "zero_miss"),
            )

        # 6. Subsetting (--keep), partitioning (--part), sparse (--sparse-cutoff), MAF/geno filters
        run_pair(
            "with_miss_keep",
            ["--bfile", D("with_miss"), "--alg", "1", "--keep", D("keep_sub.txt"), "--n-markers", "64"],
            ["--grm-strip-rows", "64"],
            runs_dir,
        )
        run_pair(
            "late_miss_part_3_2",
            ["--bfile", D("late_miss"), "--alg", "1", "--part", "3", "2", "--n-markers", "64"],
            ["--grm-strip-rows", "64"],
            runs_dir,
        )
        run_pair(
            "with_miss_part_tiled_3_3",
            ["--bfile", D("with_miss"), "--alg", "1", "--part", "3", "3", "--n-markers", "64", "--tile-budget", "0.0004"],
            ["--grm-strip-rows", "64"],
            runs_dir,
        )
        run_pair(
            "zero_miss_part_tiled_3_2",
            ["--bfile", D("zero_miss"), "--alg", "1", "--part", "3", "2", "--n-markers", "64", "--tile-budget", "0.0004"],
            ["--grm-strip-rows", "64"],
            runs_dir,
        )
        run_pair(
            "zero_miss_part_fast_tiled_3_2",
            ["--bfile", D("zero_miss"), "--fast", "--part", "3", "2", "--n-markers", "64", "--tile-budget", "0.0004"],
            ["--grm-strip-rows", "64"],
            runs_dir,
        )
        run_pair(
            "late_miss_sparse_rm",
            ["--bfile", D("late_miss"), "--alg", "1", "--sparse-cutoff", "0.02", "--n-markers", "64"],
            ["--grm-strip-rows", "64", "--grm-layout", "rowmajor"],
            runs_dir,
        )
        run_pair(
            "with_miss_sparse_cm",
            ["--bfile", D("with_miss"), "--alg", "1", "--sparse-cutoff", "0.02", "--n-markers", "64"],
            ["--grm-strip-rows", "64", "--grm-layout", "colmajor"],
            runs_dir,
        )
        run_pair(
            "with_miss_maf_geno",
            ["--bfile", D("with_miss"), "--alg", "1", "--maf", "0.15", "--geno", "0.08", "--n-markers", "64"],
            ["--grm-strip-rows", "64"],
            runs_dir,
        )

        # 7. Zero valid markers edge case
        for layout in ("rowmajor", "colmajor"):
            run_pair(
                f"mono_all_dense_{layout}",
                ["--bfile", D("mono_all"), "--alg", "1"],
                ["--grm-strip-rows", "64", "--grm-layout", layout],
                runs_dir,
            )
        run_pair(
            "mono_all_fast",
            ["--bfile", D("mono_all"), "--fast"],
            ["--grm-strip-rows", "64"],
            runs_dir,
            is_fast=True,
        )
        run_pair(
            "mono_all_sparse_0",
            ["--bfile", D("mono_all"), "--alg", "1", "--sparse-cutoff", "0.0"],
            ["--grm-strip-rows", "64"],
            runs_dir,
        )
        run_pair(
            "mono_all_sparse_005",
            ["--bfile", D("mono_all"), "--alg", "1", "--sparse-cutoff", "0.05"],
            ["--grm-strip-rows", "64"],
            runs_dir,
        )

        # 8. Quantized INT8 mean-imputation (--impute-mean)
        for bad_args in (
            ["--bfile", D("with_miss"), "--alg", "0", "--impute-mean"],
            ["--bfile", D("with_miss"), "--dominance", "--impute-mean"],
            ["--bfile", D("with_miss"), "--alg", "1", "--grm-legacy-engine", "--impute-mean"],
        ):
            rc_bad, _ = run_cmd([GCTAG_BIN, "grm", "make"] + bad_args + ["--out", os.path.join(runs_dir, "bad_imp")])
            assert rc_bad != 0, f"Expected non-zero exit for invalid args: {bad_args}"
        print("  PASS impute_mean_cli_validation")

        for ds in ("with_miss", "late_miss"):
            for layout in ("rowmajor", "colmajor"):
                for diag in ("fused", "syrk"):
                    verify_impute_mean_case(
                        f"imp_{ds}_{layout}_{diag}",
                        ["--bfile", D(ds), "--alg", "1", "--impute-mean", "--n-markers", "64"],
                        ["--grm-strip-rows", "64", "--grm-layout", layout, "--grm-diag-mode", diag, "--grm-profile"],
                        D(ds),
                        runs_dir,
                        check_profile=True,
                    )
            verify_impute_mean_case(
                f"imp_{ds}_fast_dense",
                ["--bfile", D(ds), "--fast", "--impute-mean", "--n-markers", "64"],
                ["--grm-strip-rows", "64", "--grm-profile"],
                D(ds),
                runs_dir,
                is_fast=True,
                check_profile=True,
            )
            for layout in ("rowmajor", "colmajor"):
                verify_impute_mean_case(
                    f"imp_{ds}_fast_tiled_{layout}",
                    ["--bfile", D(ds), "--fast", "--impute-mean", "--n-markers", "64", "--tile-budget", "0.0005"],
                    ["--grm-strip-rows", "64", "--grm-layout", layout],
                    D(ds),
                    runs_dir,
                    is_fast=True,
                )

        verify_impute_mean_case(
            "imp_zero_miss_fast",
            ["--bfile", D("zero_miss"), "--fast", "--impute-mean", "--n-markers", "64"],
            ["--grm-strip-rows", "64"],
            D("zero_miss"),
            runs_dir,
            is_fast=True,
        )
        verify_impute_mean_case(
            "imp_with_miss_keep",
            ["--bfile", D("with_miss"), "--alg", "1", "--impute-mean", "--keep", D("keep_sub.txt"), "--n-markers", "64"],
            ["--grm-strip-rows", "64"],
            D("with_miss"),
            runs_dir,
            keep_file=D("keep_sub.txt"),
        )
        verify_impute_mean_case(
            "imp_all_miss_sample",
            ["--bfile", D("all_miss_sample"), "--alg", "1", "--impute-mean", "--n-markers", "64"],
            ["--grm-strip-rows", "64"],
            D("all_miss_sample"),
            runs_dir,
        )
        # Verify 100% missing sample has exact 0.0 diagonal
        all_miss_g = array.array("f")
        all_miss_g.frombytes(open(os.path.join(runs_dir, "imp_all_miss_sample.imp.grm.bin"), "rb").read())
        assert all_miss_g[0] == 0.0, f"Expected 0.0 diagonal for 100%-missing sample 0, got {all_miss_g[0]}"
        idx65 = 65 * 66 // 2 + 65
        assert all_miss_g[idx65] == 0.0, f"Expected 0.0 diagonal for 100%-missing sample 65, got {all_miss_g[idx65]}"
        print("  PASS imp_all_miss_sample_zero_diag")

        verify_impute_mean_case(
            "imp_mono_all_fast",
            ["--bfile", D("mono_all"), "--fast", "--impute-mean"],
            ["--grm-strip-rows", "64"],
            D("mono_all"),
            runs_dir,
            is_fast=True,
        )

        # Partitioned & sparse with --impute-mean smoke checks
        rc_p, log_p = run_cmd([
            GCTAG_BIN, "grm", "make", "--bfile", D("with_miss"), "--alg", "1", "--impute-mean",
            "--part", "3", "2", "--n-markers", "64", "--grm-strip-rows", "64",
            "--threads", "4", "--out", os.path.join(runs_dir, "imp_part"),
        ])
        assert rc_p == 0, f"imp_part failed:\n{log_p}"
        rc_sp, log_sp = run_cmd([
            GCTAG_BIN, "grm", "make", "--bfile", D("with_miss"), "--alg", "1", "--impute-mean",
            "--sparse-cutoff", "0.02", "--n-markers", "64", "--grm-strip-rows", "64",
            "--threads", "4", "--out", os.path.join(runs_dir, "imp_sparse"),
        ])
        assert rc_sp == 0, f"imp_sparse failed:\n{log_sp}"
        print("  PASS imp_part_and_sparse")

        print("ALL MULTI-STRIP & HIDDEN-FLAG TESTS PASSED")
    finally:
        if owned_tmp:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
