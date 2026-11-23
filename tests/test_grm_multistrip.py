#!/usr/bin/env python3
"""Multi-strip, mid-run promotion, --fast, and hidden-flag regression suite
for the clean-room BED INT8 GRM engine (comparing against --grm-legacy-engine).

Usage:
    python3 tests/test_grm_multistrip.py [GCTAG_BIN] [WORK_DIR]
"""

import array
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
                ["--grm-strip-rows", "64"],
                runs_dir,
                is_fast=True,
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

        print("ALL MULTI-STRIP & HIDDEN-FLAG TESTS PASSED")
    finally:
        if owned_tmp:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
