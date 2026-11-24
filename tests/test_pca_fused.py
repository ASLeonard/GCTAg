#!/usr/bin/env python3
"""Regression suite for Track A (shared GRM I/O + fused FP16 PCA engine).

Usage:
    python3 tests/test_pca_fused.py [NEW_BIN] [REF_BIN] [WORK_DIR]
"""

import math
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile

NEW_BIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join("build", "GCTAg")
REF_BIN = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "/Users/alleonard/.gemini/antigravity/brain/6adc595f-11cf-4f9c-998c-6140ad1f3429/scratch/GCTAg_77cbefc"
)
USER_WORK = sys.argv[3] if len(sys.argv) > 3 else ""

CODE = {2: 0b00, 1: 0b10, 0: 0b11, None: 0b01}


def run_cmd(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + "\n" + proc.stderr


def write_synthetic_plink(prefix, n, m, seed=1234):
    rng = random.Random(seed)
    nb = (n + 3) // 4
    rem = n & 3
    with open(prefix + ".bed", "wb") as bf, open(prefix + ".bim", "w") as bm:
        bf.write(bytes([0x6C, 0x1B, 0x01]))
        # Inject latent population axes so top PCs are well-separated and non-degenerate.
        pop_scores = [[rng.gauss(0.0, 1.0) for _ in range(3)] for _ in range(n)]
        for s in range(m):
            maf = rng.uniform(0.10, 0.48)
            load = [rng.gauss(0.0, 0.12) for _ in range(3)]
            buf = bytearray(nb)
            for i in range(n):
                if rng.random() < 0.02:
                    g = None
                else:
                    p_i = maf + sum(load[c] * pop_scores[i][c] for c in range(3))
                    p_i = min(0.95, max(0.05, p_i))
                    g = (1 if rng.random() < p_i else 0) + (1 if rng.random() < p_i else 0)
                buf[i >> 2] |= CODE[g] << (2 * (i & 3))
            if rem:
                buf[-1] |= (0b01010101 << (2 * rem)) & 0xFF
            bf.write(bytes(buf))
            chrom = (s % 22) + 1
            bm.write(f"{chrom}\trs{s}\t0\t{1000 + s}\tA\tG\n")
    with open(prefix + ".fam", "w") as fm:
        for i in range(n):
            fm.write(f"F{i}\tI{i}\t0\t0\t{(i % 2) + 1}\t{rng.gauss(0.0, 1.0):.4f}\n")


def read_eigenval(path):
    with open(path) as f:
        return [float(line.strip()) for line in f if line.strip()]


def read_eigenvec_text(path):
    ids = []
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            ids.append((parts[0], parts[1]))
            rows.append([float(x) for x in parts[2:]])
    # Transpose to k x n column-major list of vectors for comparison
    n = len(rows)
    k = len(rows[0]) if n else 0
    cols = [[rows[i][j] for i in range(n)] for j in range(k)]
    return ids, cols


def read_eigenvec_bin(path):
    data = open(path, "rb").read()
    magic, ver, dtype, n, k, reserved, id_hash, id_bytes = struct.unpack_from("<4sIIiiIQQ", data, 0)
    assert magic == b"EVEC", f"Bad magic: {magic!r}"
    assert ver == 1, f"Bad version: {ver}"
    assert dtype in (0, 1), f"Bad dtype: {dtype}"
    pos = 40
    id_blob = data[pos : pos + id_bytes].decode("utf-8")
    pos += id_bytes
    ids = [line for line in id_blob.splitlines() if line]
    assert len(ids) == n
    if dtype == 1:
        scales = struct.unpack_from(f"<{k}f", data, pos)
        pos += 4 * k
        cols = []
        for j in range(k):
            raw = struct.unpack_from(f"<{n}h", data, pos)
            pos += 2 * n
            s = scales[j]
            cols.append([v * s for v in raw])
    else:
        scales = None
        cols = []
        for j in range(k):
            raw = struct.unpack_from(f"<{n}f", data, pos)
            pos += 4 * n
            cols.append(list(raw))
    assert pos == len(data), f"Trailing bytes {pos} != {len(data)}"
    return {
        "dtype": dtype,
        "n": n,
        "k": k,
        "id_hash": id_hash,
        "ids": ids,
        "scales": scales,
        "cols": cols,
    }


def check_eigenvalues_close(name, ev_a, ev_b, rtol=1e-4, atol=1e-5):
    assert len(ev_a) == len(ev_b), f"{name}: len mismatch {len(ev_a)} vs {len(ev_b)}"
    for i, (a, b) in enumerate(zip(ev_a, ev_b)):
        tol = atol + rtol * max(abs(a), abs(b))
        diff = abs(a - b)
        assert diff <= tol, f"{name}: eval[{i}] {a} vs {b} (diff={diff} > tol={tol})"


def check_subspace_alignment(name, evals, cols_a, cols_b, max_check_pcs=None, min_dot=0.9999):
    assert len(cols_a) == len(cols_b), f"{name}: k mismatch {len(cols_a)} vs {len(cols_b)}"
    k = len(cols_a)
    n = len(cols_a[0]) if k else 0
    limit = min(k, max_check_pcs) if max_check_pcs is not None else k
    for j in range(limit):
        # Skip nearly-degenerate eigenpairs where adjacent eigenvalues are within 1e-3 relative
        gap_prev = abs(evals[j - 1] - evals[j]) / max(1.0, abs(evals[j])) if j > 0 else 1.0
        gap_next = abs(evals[j] - evals[j + 1]) / max(1.0, abs(evals[j])) if j + 1 < k else 1.0
        if min(gap_prev, gap_next) < 2e-3:
            continue
        va = cols_a[j]
        vb = cols_b[j]
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(x * x for x in vb))
        assert na > 0.0 and nb > 0.0
        dot = abs(sum(x * y for x, y in zip(va, vb))) / (na * nb)
        assert dot >= min_dot, f"{name}: PC {j} alignment |dot|={dot:.6f} < {min_dot}"


def run_pca_case(args, out_prefix, bin_path=NEW_BIN):
    rc, log = run_cmd([bin_path, "pca"] + args + ["--out", out_prefix])
    assert rc == 0, f"{bin_path} pca failed (rc={rc}):\n{log}"
    return log


def main():
    tmp_ctx = None
    if USER_WORK:
        work = USER_WORK
        os.makedirs(work, exist_ok=True)
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="gcta_test_pca_fused_")
        work = tmp_ctx.name

    try:
        # 1. Hidden flags must be omitted from `pca --help`
        rc, help_out = run_cmd([NEW_BIN, "pca", "--help"])
        assert rc == 0, f"pca --help failed:\n{help_out}"
        hidden_flags = [
            "--pca-profile",
            "--pca-matvec-mode",
            "--pca-fused-max-k",
            "--pca-j-block",
            "--pca-load-mode",
            "--pca-io-slot-mb",
            "--pca-tile-size",
        ]
        for hf in hidden_flags:
            assert hf not in help_out, f"Hidden flag {hf} leaked into pca --help"
        print("  PASS hidden_pca_flags_omitted_from_help")

        # 2. Generate synthetic BED datasets & GRMs (n=257 non-multiple-of-16, n=600 multi-slab)
        b257 = os.path.join(work, "syn257")
        write_synthetic_plink(b257, n=257, m=900, seed=101)
        g257_f16 = os.path.join(work, "g257_f16")
        rc, log = run_cmd([NEW_BIN, "grm", "make", "--bfile", b257, "--fast", "--threads", "4", "--out", g257_f16])
        assert rc == 0, f"grm make syn257 f16 failed:\n{log}"
        g257_f32 = os.path.join(work, "g257_f32")
        rc, log = run_cmd([NEW_BIN, "grm", "make", "--bfile", b257, "--alg", "1", "--threads", "4", "--out", g257_f32])
        assert rc == 0, f"grm make syn257 f32 failed:\n{log}"

        b600 = os.path.join(work, "syn600")
        write_synthetic_plink(b600, n=600, m=1200, seed=202)
        g600_f16 = os.path.join(work, "g600_f16")
        rc, log = run_cmd([NEW_BIN, "grm", "make", "--bfile", b600, "--fast", "--threads", "4", "--out", g600_f16])
        assert rc == 0, f"grm make syn600 f16 failed:\n{log}"

        has_ref = os.path.isfile(REF_BIN)

        # Keep / remove files for syn600
        with open(g600_f16 + ".grm.id") as f:
            ids600 = [line.strip() for line in f if line.strip()]
        keep_file = os.path.join(work, "keep_600.txt")
        with open(keep_file, "w") as f:
            for idx in range(0, 600, 2):
                f.write(ids600[idx] + "\n")
        remove_file = os.path.join(work, "remove_600.txt")
        with open(remove_file, "w") as f:
            for idx in range(0, 120):
                f.write(ids600[idx] + "\n")

        # 3. Test fused vs tiled (+ ref binary) across k=6 (P=16), k=20 (P=32), k=40 (multi-panel P=32)
        for k_pcs in (6, 20, 40):
            base_args = [
                "--grm", g257_f16, "--pcs", str(k_pcs), "--approx", "rSVD",
                "--grm-fp16", "--eigenvec-format", "both", "--seed", "17", "--threads", "4",
            ]
            out_fused = os.path.join(work, f"p257_k{k_pcs}_fused")
            out_tiled = os.path.join(work, f"p257_k{k_pcs}_tiled")
            log_f = run_pca_case(base_args + ["--pca-matvec-mode", "fused", "--pca-profile"], out_fused)
            log_t = run_pca_case(base_args + ["--pca-matvec-mode", "tiled", "--pca-profile"], out_tiled)
            assert "[pca-profile]" in log_f and "matvec_mode=fused" in log_f, f"Missing fused profile:\n{log_f}"
            assert "[pca-profile]" in log_t and "matvec_mode=tiled" in log_t, f"Missing tiled profile:\n{log_t}"

            ev_f = read_eigenval(out_fused + ".eigenval")
            ev_t = read_eigenval(out_tiled + ".eigenval")
            check_eigenvalues_close(f"n257_k{k_pcs}_fused_vs_tiled", ev_f, ev_t)

            bin_f = read_eigenvec_bin(out_fused + ".eigenvec.bin")
            bin_t = read_eigenvec_bin(out_tiled + ".eigenvec.bin")
            assert bin_f["id_hash"] == bin_t["id_hash"]
            assert bin_f["dtype"] == 0 and bin_f["n"] == 257 and bin_f["k"] == k_pcs
            check_subspace_alignment(f"n257_k{k_pcs}_bin32", ev_t, bin_f["cols"], bin_t["cols"], max_check_pcs=min(k_pcs, 6))

            txt_ids_f, txt_cols_f = read_eigenvec_text(out_fused + ".eigenvec")
            txt_ids_t, txt_cols_t = read_eigenvec_text(out_tiled + ".eigenvec")
            assert txt_ids_f == txt_ids_t and len(txt_ids_f) == 257
            check_subspace_alignment(f"n257_k{k_pcs}_text", ev_t, txt_cols_f, txt_cols_t, max_check_pcs=min(k_pcs, 6))

            if has_ref:
                out_ref = os.path.join(work, f"p257_k{k_pcs}_ref")
                run_pca_case(base_args, out_ref, bin_path=REF_BIN)
                ev_r = read_eigenval(out_ref + ".eigenval")
                bin_r = read_eigenvec_bin(out_ref + ".eigenvec.bin")
                check_eigenvalues_close(f"n257_k{k_pcs}_fused_vs_ref", ev_f, ev_r)
                assert bin_f["id_hash"] == bin_r["id_hash"]
                check_subspace_alignment(f"n257_k{k_pcs}_vs_ref", ev_r, bin_f["cols"], bin_r["cols"], max_check_pcs=min(k_pcs, 6))

            print(f"  PASS fused_vs_tiled_n257_k{k_pcs}")

        # 4. Classic .grm.bin with --grm-fp16 (conversion load path) + --fast (bin16 output)
        out_f32_fused = os.path.join(work, "p257_f32src_fused")
        out_f32_tiled = os.path.join(work, "p257_f32src_tiled")
        f32_args = ["--grm", g257_f32, "--pcs", "6", "--fast", "--seed", "33", "--threads", "4"]
        run_pca_case(f32_args + ["--pca-matvec-mode", "fused"], out_f32_fused)
        run_pca_case(f32_args + ["--pca-matvec-mode", "tiled"], out_f32_tiled)
        ev_f32_f = read_eigenval(out_f32_fused + ".eigenval")
        ev_f32_t = read_eigenval(out_f32_tiled + ".eigenval")
        check_eigenvalues_close("n257_f32src_bin16", ev_f32_f, ev_f32_t)
        bin16_f = read_eigenvec_bin(out_f32_fused + ".eigenvec.bin")
        bin16_t = read_eigenvec_bin(out_f32_tiled + ".eigenvec.bin")
        assert bin16_f["dtype"] == 1
        check_subspace_alignment("n257_f32src_bin16", ev_f32_t, bin16_f["cols"], bin16_t["cols"], max_check_pcs=4, min_dot=0.999)
        print("  PASS f32_source_grm_fp16_bin16")

        # 5. Multi-slab j_block=64 on n=600 + direct vs async load parity + --pca-fused-max-k override
        jb_args = [
            "--grm", g600_f16, "--pcs", "20", "--fast", "--eigenvec-format", "both",
            "--pca-j-block", "64", "--seed", "7", "--threads", "4", "--pca-profile",
        ]
        out_direct = os.path.join(work, "p600_direct")
        out_async = os.path.join(work, "p600_async")
        out_tiled600 = os.path.join(work, "p600_tiled")
        log_d = run_pca_case(jb_args + ["--pca-load-mode", "direct"], out_direct)
        log_a = run_pca_case(jb_args + ["--pca-load-mode", "async"], out_async)
        log_t600 = run_pca_case(jb_args + ["--pca-fused-max-k", "10", "--pca-tile-size", "256"], out_tiled600)
        assert "direct pread" in log_d and "load_mode=direct" in log_d, f"Expected direct load:\n{log_d}"
        assert "async stream" in log_a and "load_mode=async" in log_a, f"Expected async load:\n{log_a}"
        assert "matvec_mode=tiled" in log_t600, f"Expected max-k cutoff to fall back to tiled:\n{log_t600}"
        ev_d600 = read_eigenval(out_direct + ".eigenval")
        ev_a600 = read_eigenval(out_async + ".eigenval")
        ev_t600 = read_eigenval(out_tiled600 + ".eigenval")
        check_eigenvalues_close("n600_jb64_direct_vs_async_t4", ev_d600, ev_a600, rtol=1e-5, atol=1e-5)
        check_eigenvalues_close("n600_jb64_fused_vs_tiled", ev_d600, ev_t600)

        # Deterministic exact byte parity check (direct vs async load under tiled T4 and fused T1)
        out_dt = os.path.join(work, "p600_dt")
        out_at = os.path.join(work, "p600_at")
        tiled_args = [
            "--grm", g600_f16, "--pcs", "20", "--fast", "--pca-matvec-mode", "tiled",
            "--seed", "7", "--threads", "4",
        ]
        run_pca_case(tiled_args + ["--pca-load-mode", "direct"], out_dt)
        run_pca_case(tiled_args + ["--pca-load-mode", "async"], out_at)
        assert open(out_dt + ".eigenval", "rb").read() == open(out_at + ".eigenval", "rb").read()
        assert open(out_dt + ".eigenvec.bin", "rb").read() == open(out_at + ".eigenvec.bin", "rb").read()

        out_d1 = os.path.join(work, "p600_d1")
        out_a1 = os.path.join(work, "p600_a1")
        fused_t1_args = [
            "--grm", g600_f16, "--pcs", "20", "--fast", "--pca-j-block", "64",
            "--seed", "7", "--threads", "1",
        ]
        run_pca_case(fused_t1_args + ["--pca-load-mode", "direct"], out_d1)
        run_pca_case(fused_t1_args + ["--pca-load-mode", "async"], out_a1)
        assert open(out_d1 + ".eigenval", "rb").read() == open(out_a1 + ".eigenval", "rb").read()
        assert open(out_d1 + ".eigenvec.bin", "rb").read() == open(out_a1 + ".eigenvec.bin", "rb").read()
        print("  PASS multi_slab_jb64_and_direct_vs_async_exact_parity")

        # 6. Subsetting (--keep / --remove) exercises monotonic kp fast path + non-direct loader
        for tag, extra in (("keep", ["--keep", keep_file]), ("remove", ["--remove", remove_file])):
            sub_args = [
                "--grm", g600_f16, "--pcs", "6", "--fast", "--eigenvec-format", "bin",
                "--seed", "99", "--threads", "4",
            ] + extra
            out_sub_f = os.path.join(work, f"p600_{tag}_fused")
            out_sub_t = os.path.join(work, f"p600_{tag}_tiled")
            run_pca_case(sub_args + ["--pca-matvec-mode", "fused"], out_sub_f)
            run_pca_case(sub_args + ["--pca-matvec-mode", "tiled"], out_sub_t)
            ev_sf = read_eigenval(out_sub_f + ".eigenval")
            ev_st = read_eigenval(out_sub_t + ".eigenval")
            check_eigenvalues_close(f"n600_{tag}", ev_sf, ev_st)
            if has_ref:
                out_sub_r = os.path.join(work, f"p600_{tag}_ref")
                run_pca_case(sub_args, out_sub_r, bin_path=REF_BIN)
                ev_sr = read_eigenval(out_sub_r + ".eigenval")
                check_eigenvalues_close(f"n600_{tag}_vs_ref", ev_sf, ev_sr)
            print(f"  PASS subset_{tag}")

        # 7. Exact full dsyevd/dsyevr mode still works cleanly on small f32 GRM
        out_exact = os.path.join(work, "p257_exact")
        run_pca_case(["--grm", g257_f32, "--pcs", "6", "--eigenvec-format", "both", "--threads", "4", "--pca-profile"], out_exact)
        ev_ex = read_eigenval(out_exact + ".eigenval")
        assert len(ev_ex) == 6
        print("  PASS exact_dsyevr_mode")

        print("ALL PCA FUSED & TRACK A TESTS PASSED")
        return 0
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    sys.exit(main())
