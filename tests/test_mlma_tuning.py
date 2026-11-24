#!/usr/bin/env python3
"""Regression suite for Track B (streaming TUN3/TREF, fast_i8 / std_i8 kernels, compaction, and dev flags).

Usage:
    python3 tests/test_mlma_tuning.py [NEW_BIN] [REF_BIN] [BED_DIR] [WORK_DIR]
"""

import math
import os
import shutil
import subprocess
import sys
import tempfile

NEW_BIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join("build", "GCTAg")
REF_BIN = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "/Users/alleonard/.gemini/antigravity/brain/6adc595f-11cf-4f9c-998c-6140ad1f3429/scratch/GCTAg_77cbefc"
)
BED_DIR = (
    sys.argv[3]
    if len(sys.argv) > 3
    else "/Users/alleonard/.gemini/antigravity/brain/6adc595f-11cf-4f9c-998c-6140ad1f3429/scratch/bed_data"
)
USER_WORK = sys.argv[4] if len(sys.argv) > 4 else ""

HIDDEN_MLMA_FLAGS = [
    "--mlma-profile",
    "--mlma-p-budget-mb",
    "--mlma-q-rows",
    "--mlma-snp-tile",
    "--mlma-reorder-q",
    "--mlma-block-size",
    "--mlma-sample-tile",
    "--mlma-ukx-budget-mb",
    "--mlma-tun3-slot-mb",
]


def run_cmd(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + "\n" + proc.stderr


def must_run(cmd, desc):
    rc, out = run_cmd(cmd)
    if rc != 0:
        raise AssertionError(f"{desc} failed (rc={rc}):\n{' '.join(cmd)}\n{out}")
    return out


def parse_mlma(path):
    rows = []
    with open(path) as f:
        header = f.readline().strip().split()
        for line in f:
            parts = line.strip().split()
            if parts:
                rows.append(parts)
    return header, rows


def compare_mlma(path_a, path_b, exact=False, rel_tol=1e-4, abs_tol=1e-6):
    if exact:
        txt_a = open(path_a, "rb").read()
        txt_b = open(path_b, "rb").read()
        assert txt_a == txt_b, f".mlma mismatch (expected exact): {path_a} vs {path_b}"
        return
    h_a, r_a = parse_mlma(path_a)
    h_b, r_b = parse_mlma(path_b)
    assert h_a == h_b, f"Header mismatch: {h_a} vs {h_b}"
    assert len(r_a) == len(r_b), f"Row count mismatch: {len(r_a)} vs {len(r_b)}"
    for i, (ra, rb) in enumerate(zip(r_a, r_b)):
        assert ra[:5] == rb[:5], f"Row {i} ID/meta mismatch: {ra[:5]} vs {rb[:5]}"
        for col in range(5, len(ra)):
            va, vb = ra[col], rb[col]
            if va.lower() in ("nan", "na") or vb.lower() in ("nan", "na"):
                assert va.lower() == vb.lower(), f"Row {i} col {col} NaN/NA mismatch: {va} vs {vb}"
                continue
            fa, fb = float(va), float(vb)
            diff = abs(fa - fb)
            lim = max(abs_tol, rel_tol * max(abs(fa), abs(fb)))
            assert diff <= lim, (
                f"Row {i} ({ra[1]}) col {h_a[col]} mismatch: {fa} vs {fb} "
                f"(diff={diff:.3e} > lim={lim:.3e})"
            )


def extract_profile_line(out):
    for line in out.splitlines():
        if "[mlma-profile]" in line:
            return line.strip()
    raise AssertionError(f"Missing [mlma-profile] line in output:\n{out}")


def parse_profile_kv(line):
    kv = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k] = v
    return kv


def check_magic(path, expected_magic):
    with open(path, "rb") as f:
        got = f.read(4)
    assert got == expected_magic, f"Expected magic {expected_magic!r} in {path}, got {got!r}"


def test_hidden_flags_omitted_from_help():
    rc, out = run_cmd([NEW_BIN, "mlma", "--help"])
    assert rc == 0, out
    for flag in HIDDEN_MLMA_FLAGS:
        assert flag not in out, f"Hidden flag {flag} leaked into `mlma --help`:\n{out}"
    print("  PASS hidden_mlma_flags_omitted_from_help")


def main():
    if not os.path.exists(NEW_BIN):
        raise FileNotFoundError(f"Missing NEW_BIN: {NEW_BIN}")
    if not os.path.exists(REF_BIN):
        raise FileNotFoundError(f"Missing REF_BIN: {REF_BIN}")

    test_hidden_flags_omitted_from_help()

    tmpdir = USER_WORK if USER_WORK else tempfile.mkdtemp(prefix="mlma_tuning_")
    os.makedirs(tmpdir, exist_ok=True)
    own_tmp = not bool(USER_WORK)
    try:
        def D(name):
            return os.path.join(BED_DIR, name)

        def W(name):
            return os.path.join(tmpdir, name)

        # Build baseline GRM, TUN3 states, and TREF states
        must_run([NEW_BIN, "grm", "make", "--bfile", D("syn"), "--out", W("g_syn")], "grm syn")
        must_run(
            [
                REF_BIN,
                "reml",
                "--fam",
                D("syn.fam"),
                "--grm",
                W("g_syn"),
                "--pheno",
                D("syn.phen"),
                "--woodbury-basis",
                "160",
                "--seed",
                "7",
                "--out",
                W("reml_syn"),
            ],
            "reml TUN3 syn",
        )
        check_magic(W("reml_syn.reml"), b"TUN3")

        must_run(
            [
                REF_BIN,
                "reml",
                "--fam",
                D("syn.fam"),
                "--grm",
                W("g_syn"),
                "--pheno",
                D("syn_w.phen"),
                "--weighted-phenotypes",
                "--woodbury-basis",
                "160",
                "--seed",
                "7",
                "--out",
                W("reml_syn_w"),
            ],
            "reml TUN3 syn_w",
        )
        check_magic(W("reml_syn_w.reml"), b"TUN3")

        must_run(
            [
                REF_BIN,
                "reml",
                "--fam",
                D("syn.fam"),
                "--woodbury-reuse",
                W("reml_syn.reml"),
                "--pheno",
                D("syn.phen"),
                "--out",
                W("reml_tref_syn"),
            ],
            "reml TREF syn",
        )
        check_magic(W("reml_tref_syn.reml"), b"TREF")

        # 1. 2slice with missingness + --maf 0.05 compaction + dev tuning flags + --mlma-profile
        base_2slice = [
            "mlma",
            "--bfile",
            D("syn"),
            "--load-reml",
            W("reml_syn.reml"),
            "--pheno",
            D("syn.phen"),
            "--fast-mlma",
            "2slice",
            "--maf",
            "0.05",
        ]
        must_run([REF_BIN] + base_2slice + ["--out", W("ref_2slice_maf")], "ref 2slice maf")
        must_run([NEW_BIN] + base_2slice + ["--out", W("new_2slice_maf_def")], "new 2slice maf def")
        compare_mlma(W("new_2slice_maf_def.mlma"), W("ref_2slice_maf.mlma"), exact=True)

        tuned_out = must_run(
            [NEW_BIN]
            + base_2slice
            + [
                "--mlma-q-rows",
                "256",
                "--mlma-p-budget-mb",
                "16",
                "--mlma-snp-tile",
                "64",
                "--mlma-block-size",
                "128",
                "--mlma-tun3-slot-mb",
                "1",
                "--mlma-reorder-q",
                "on",
                "--mlma-profile",
                "--out",
                W("new_2slice_maf_tuned"),
            ],
            "new 2slice maf tuned",
        )
        compare_mlma(W("new_2slice_maf_tuned.mlma"), W("ref_2slice_maf.mlma"), exact=False)
        prof = parse_profile_kv(extract_profile_line(tuned_out))
        for key in (
            "reml_load_s",
            "q_build_s",
            "q_reorder_s",
            "decode_s",
            "xvx_gemm_s",
            "xvx_reduce_s",
            "format_write_s",
            "total_s",
            "blocks",
            "valid_snps",
            "total_snps",
            "q_tiles",
            "block_size",
        ):
            assert key in prof, f"Missing {key} in [mlma-profile]: {prof}"
        assert int(prof["block_size"]) == 128, prof
        assert int(prof["blocks"]) > 0 and int(prof["q_tiles"]) > 1, prof
        assert 0 < int(prof["valid_snps"]) < int(prof["total_snps"]), (
            f"Expected --maf compaction valid_snps < total_snps: {prof}"
        )
        print("  PASS 2slice_missing_maf_compaction_and_profile")

        # Test explicit --mlma-reorder-q off
        must_run(
            [NEW_BIN]
            + base_2slice
            + ["--mlma-reorder-q", "off", "--mlma-q-rows", "256", "--out", W("new_2slice_rq_off")],
            "new 2slice reorder-q off",
        )
        compare_mlma(W("new_2slice_rq_off.mlma"), W("ref_2slice_maf.mlma"), exact=False)
        print("  PASS reorder_q_off")

        # 2. syn0 (0% missingness fast path)
        syn0_cmd = [
            "mlma",
            "--bfile",
            D("syn0"),
            "--load-reml",
            W("reml_syn.reml"),
            "--pheno",
            D("syn.phen"),
            "--fast-mlma",
            "2slice",
        ]
        must_run([REF_BIN] + syn0_cmd + ["--out", W("ref_syn0")], "ref syn0")
        must_run(
            [NEW_BIN]
            + syn0_cmd
            + ["--mlma-q-rows", "256", "--mlma-block-size", "128", "--mlma-profile", "--out", W("new_syn0")],
            "new syn0",
        )
        compare_mlma(W("new_syn0.mlma"), W("ref_syn0.mlma"), exact=False)
        print("  PASS syn0_zero_missing")

        # 3. hybrid and 1slice modes
        for mode in ("hybrid", "1slice"):
            cmd = [
                "mlma",
                "--bfile",
                D("syn"),
                "--load-reml",
                W("reml_syn.reml"),
                "--pheno",
                D("syn.phen"),
                "--fast-mlma",
                mode,
            ]
            must_run([REF_BIN] + cmd + ["--out", W(f"ref_{mode}")], f"ref {mode}")
            out = must_run(
                [NEW_BIN] + cmd + ["--mlma-q-rows", "256", "--mlma-profile", "--out", W(f"new_{mode}")],
                f"new {mode}",
            )
            extract_profile_line(out)
            compare_mlma(W(f"new_{mode}.mlma"), W(f"ref_{mode}.mlma"), exact=False)
        print("  PASS hybrid_and_1slice")

        # 4. std_i8 path with --mlma-sample-tile and --mlma-ukx-budget-mb
        std_cmd = [
            "mlma",
            "--bfile",
            D("syn"),
            "--load-reml",
            W("reml_syn.reml"),
            "--pheno",
            D("syn.phen"),
        ]
        must_run([REF_BIN] + std_cmd + ["--out", W("ref_std")], "ref std_i8")
        std_out = must_run(
            [NEW_BIN]
            + std_cmd
            + [
                "--mlma-sample-tile",
                "256",
                "--mlma-ukx-budget-mb",
                "32",
                "--mlma-profile",
                "--out",
                W("new_std"),
            ],
            "new std_i8",
        )
        std_prof = parse_profile_kv(extract_profile_line(std_out))
        assert int(std_prof["blocks"]) > 0 and int(std_prof["valid_snps"]) > 0, std_prof
        compare_mlma(W("new_std.mlma"), W("ref_std.mlma"), exact=True)
        print("  PASS std_i8_sample_tile_and_ukx_budget")

        # 5. TREF matching phenotype + TREF mismatched phenotype (reconstruct_vi_y_from_q_uk)
        tref_match = [
            "mlma",
            "--bfile",
            D("syn"),
            "--load-reml",
            W("reml_tref_syn.reml"),
            "--pheno",
            D("syn.phen"),
            "--fast-mlma",
            "2slice",
        ]
        must_run([REF_BIN] + tref_match + ["--out", W("ref_tref_match")], "ref tref match")
        tref_out = must_run(
            [NEW_BIN]
            + tref_match
            + [
                "--mlma-q-rows",
                "256",
                "--mlma-tun3-slot-mb",
                "1",
                "--mlma-profile",
                "--out",
                W("new_tref_match"),
            ],
            "new tref match",
        )
        tref_prof = parse_profile_kv(extract_profile_line(tref_out))
        assert int(tref_prof["q_tiles"]) > 1, tref_prof
        compare_mlma(W("new_tref_match.mlma"), W("ref_tref_match.mlma"), exact=False)

        tref_mismatch = [
            "mlma",
            "--bfile",
            D("syn"),
            "--load-reml",
            W("reml_tref_syn.reml"),
            "--pheno",
            D("syn_w.phen"),
            "--weighted-phenotypes",
            "--fast-mlma",
            "2slice",
        ]
        must_run([REF_BIN] + tref_mismatch + ["--out", W("ref_tref_mismatch")], "ref tref mismatch")
        must_run(
            [NEW_BIN] + tref_mismatch + ["--mlma-q-rows", "256", "--out", W("new_tref_mismatch")],
            "new tref mismatch",
        )
        compare_mlma(W("new_tref_mismatch.mlma"), W("ref_tref_mismatch.mlma"), exact=False)
        print("  PASS tref_matching_and_mismatched_pheno")

        # 6. Weighted phenotypes via TUN3
        w_cmd = [
            "mlma",
            "--bfile",
            D("syn"),
            "--load-reml",
            W("reml_syn_w.reml"),
            "--pheno",
            D("syn_w.phen"),
            "--weighted-phenotypes",
            "--fast-mlma",
            "2slice",
        ]
        must_run([REF_BIN] + w_cmd + ["--out", W("ref_w_tun3")], "ref weighted tun3")
        must_run(
            [NEW_BIN] + w_cmd + ["--mlma-q-rows", "256", "--out", W("new_w_tun3")],
            "new weighted tun3",
        )
        compare_mlma(W("new_w_tun3.mlma"), W("ref_w_tun3.mlma"), exact=False)
        print("  PASS weighted_pheno_tun3")

        # 7. Inline MLMA with --fast --woodbury-basis 40
        inline_cmd = [
            "mlma",
            "--bfile",
            D("syn"),
            "--grm",
            W("g_syn"),
            "--pheno",
            D("syn.phen"),
            "--fast",
            "--woodbury-basis",
            "160",
            "--seed",
            "7",
        ]
        must_run([REF_BIN] + inline_cmd + ["--out", W("ref_inline_fast")], "ref inline fast")
        inline_out = must_run(
            [NEW_BIN]
            + inline_cmd
            + ["--mlma-q-rows", "256", "--mlma-profile", "--out", W("new_inline_fast")],
            "new inline fast",
        )
        in_prof = parse_profile_kv(extract_profile_line(inline_out))
        assert int(in_prof["q_tiles"]) > 1 and int(in_prof["blocks"]) > 0, in_prof
        compare_mlma(W("new_inline_fast.mlma"), W("ref_inline_fast.mlma"), exact=False)
        print("  PASS inline_mlma_fast_woodbury_basis_40")

        print("ALL MLMA TUNING & TRACK B TESTS PASSED")
    finally:
        if own_tmp and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
