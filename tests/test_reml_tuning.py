#!/usr/bin/env python3
"""Regression suite for Track C (FP16 symm_large_k, blocked Gram/rSVD, zero-Uk streaming TREF reuse, and --reml-* dev flags).

Usage:
    python3 tests/test_reml_tuning.py [NEW_BIN] [REF_BIN] [BED_DIR] [WORK_DIR]
"""

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

HIDDEN_REML_FLAGS = [
    "--reml-profile",
    "--reml-matvec-band-rows",
    "--reml-matvec-col-chunk",
    "--reml-qr-power-passes",
    "--reml-eigh-fp32",
    "--reml-eigh-threads",
    "--reml-io-slot-mb",
    "--reml-reuse-stream",
]

PROFILE_KEYS = [
    "grm_load_s",
    "sketch_fill_s",
    "sketch_matvec_s",
    "power_qr_s",
    "power_matvec_s",
    "final_qr_s",
    "rr_matvec_s",
    "gram_s",
    "eigh_s",
    "rotate_s",
    "ai_reml_s",
    "save_s",
    "total_s",
    "matvec_unpack_s",
    "matvec_gemm_s",
    "matvec_tiles",
    "matvec_gemm_calls",
    "qr_calls",
    "qr_passes",
    "qr_shifted",
    "qr_householder",
    "reuse_stream",
]


def run_cmd(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + "\n" + proc.stderr


def must_run(cmd, desc):
    rc, out = run_cmd(cmd)
    if rc != 0:
        raise AssertionError(f"{desc} failed (rc={rc}):\n{' '.join(cmd)}\n{out}")
    return out


def parse_hsq(path):
    vals = {}
    with open(path) as f:
        header = f.readline()
        assert header.startswith("Source"), f"Unexpected .hsq header in {path}: {header!r}"
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                vals[parts[0]] = float(parts[1]) if parts[0] not in ("reml_iterations", "n") else int(parts[1])
    return vals


def compare_hsq(name, path_a, path_b, rel_tol=1e-4, abs_tol=1e-5):
    ha = parse_hsq(path_a)
    hb = parse_hsq(path_b)
    assert ha["n"] == hb["n"], f"{name}: n mismatch {ha['n']} vs {hb['n']}"
    for key in ("V(G)", "V(e)", "Vp", "V(G)/Vp", "logL"):
        assert key in ha and key in hb, f"{name}: missing {key} in .hsq ({path_a} vs {path_b})"
        va, vb = ha[key], hb[key]
        diff = abs(va - vb)
        lim = max(abs_tol, rel_tol * max(abs(va), abs(vb)))
        assert diff <= lim, (
            f"{name}: {key} mismatch {va} vs {vb} (diff={diff:.3e} > lim={lim:.3e})"
        )


def parse_mlma(path):
    rows = []
    with open(path) as f:
        header = f.readline().strip().split()
        for line in f:
            parts = line.strip().split()
            if parts:
                rows.append(parts)
    return header, rows


def compare_mlma(name, path_a, path_b, rel_tol=1e-4, abs_tol=1e-6):
    h_a, r_a = parse_mlma(path_a)
    h_b, r_b = parse_mlma(path_b)
    assert h_a == h_b, f"{name}: header mismatch {h_a} vs {h_b}"
    assert len(r_a) == len(r_b), f"{name}: row count mismatch {len(r_a)} vs {len(r_b)}"
    for i, (ra, rb) in enumerate(zip(r_a, r_b)):
        assert ra[:5] == rb[:5], f"{name}: row {i} ID/meta mismatch: {ra[:5]} vs {rb[:5]}"
        for col in range(5, len(ra)):
            va, vb = ra[col], rb[col]
            if va.lower() in ("nan", "na") or vb.lower() in ("nan", "na"):
                assert va.lower() == vb.lower(), f"{name}: row {i} col {col} NaN/NA mismatch: {va} vs {vb}"
                continue
            fa, fb = float(va), float(vb)
            diff = abs(fa - fb)
            lim = max(abs_tol, rel_tol * max(abs(fa), abs(fb)))
            assert diff <= lim, (
                f"{name}: row {i} ({ra[1]}) col {h_a[col]} mismatch: {fa} vs {fb} "
                f"(diff={diff:.3e} > lim={lim:.3e})"
            )


def read_reml_magic(path):
    with open(path, "rb") as f:
        return f.read(4)


def extract_reml_profile(out):
    for line in out.splitlines():
        if "[reml-profile]" in line:
            kv = {}
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    kv[k] = v
            missing = [k for k in PROFILE_KEYS if k not in kv]
            assert not missing, f"Missing keys in [reml-profile]: {missing}\nLine: {line}"
            return kv
    raise AssertionError(f"Missing [reml-profile] line in output:\n{out}")


def main():
    if not os.path.isfile(NEW_BIN):
        raise SystemExit(f"NEW_BIN not found: {NEW_BIN}")
    has_ref = os.path.isfile(REF_BIN)
    for req in ("syn.bed", "syn.bim", "syn.fam", "syn.phen", "syn_w.phen", "test.bed", "test.bim", "test.fam", "test.phen"):
        p = os.path.join(BED_DIR, req)
        if not os.path.isfile(p):
            raise SystemExit(f"Missing BED input: {p}")

    tmp_ctx = None
    if USER_WORK:
        work = USER_WORK
        os.makedirs(work, exist_ok=True)
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="gcta_test_reml_tuning_")
        work = tmp_ctx.name

    try:
        # 1. Hidden --reml-* flags must stay omitted from --help
        for subcmd in ("reml", "mlma"):
            help_out = must_run([NEW_BIN, subcmd, "--help"], f"{subcmd} --help")
            for hf in HIDDEN_REML_FLAGS:
                assert hf not in help_out, f"Hidden flag {hf} leaked into `{subcmd} --help`"
        print("  PASS hidden_reml_flags_omitted_from_help")

        syn_bfile = os.path.join(BED_DIR, "syn")
        test_bfile = os.path.join(BED_DIR, "test")

        syn_grm = os.path.join(work, "syn_grm")
        must_run(
            [NEW_BIN, "grm", "make", "--bfile", syn_bfile, "--fast", "--threads", "4", "--out", syn_grm],
            "grm make syn",
        )
        test_grm = os.path.join(work, "test_grm")
        must_run(
            [NEW_BIN, "grm", "make", "--bfile", test_bfile, "--fast", "--threads", "4", "--out", test_grm],
            "grm make test",
        )

        # 2. Fixed-k FP16 REML parity + [reml-profile] + downstream MLMA TUN3 load
        c4_syn_k60 = os.path.join(work, "c4_syn_k60")
        log_k60 = must_run(
            [
                NEW_BIN, "reml",
                "--fam", syn_bfile + ".fam",
                "--pheno", syn_bfile + ".phen",
                "--grm", syn_grm,
                "--fast",
                "--woodbury-basis", "60",
                "--seed", "7",
                "--threads", "4",
                "--reml-profile",
                "--out", c4_syn_k60,
            ],
            "reml syn k60 new",
        )
        kv_k60 = extract_reml_profile(log_k60)
        assert int(kv_k60["matvec_tiles"]) > 0, kv_k60
        assert int(kv_k60["matvec_gemm_calls"]) > 0, kv_k60
        assert int(kv_k60["qr_calls"]) > 0, kv_k60
        assert int(kv_k60["reuse_stream"]) == 0, kv_k60
        assert float(kv_k60["total_s"]) > 0.0, kv_k60
        assert read_reml_magic(c4_syn_k60 + ".reml") == b"TUN3"

        if has_ref:
            ref_syn_k60 = os.path.join(work, "ref_syn_k60")
            must_run(
                [
                    REF_BIN, "reml",
                    "--fam", syn_bfile + ".fam",
                    "--pheno", syn_bfile + ".phen",
                    "--grm", syn_grm,
                    "--fast",
                    "--woodbury-basis", "60",
                    "--seed", "7",
                    "--threads", "4",
                    "--out", ref_syn_k60,
                ],
                "reml syn k60 ref",
            )
            compare_hsq("syn_k60_vs_ref", c4_syn_k60 + ".hsq", ref_syn_k60 + ".hsq", rel_tol=1e-4, abs_tol=1e-5)

            c4_mlma_k60 = os.path.join(work, "c4_mlma_k60")
            ref_mlma_k60 = os.path.join(work, "ref_mlma_k60")
            must_run(
                [
                    NEW_BIN, "mlma",
                    "--bfile", syn_bfile,
                    "--pheno", syn_bfile + ".phen",
                    "--load-reml", c4_syn_k60 + ".reml",
                    "--fast-mlma", "2slice",
                    "--threads", "4",
                    "--out", c4_mlma_k60,
                ],
                "mlma load c4 syn k60",
            )
            must_run(
                [
                    REF_BIN, "mlma",
                    "--bfile", syn_bfile,
                    "--pheno", syn_bfile + ".phen",
                    "--load-reml", ref_syn_k60 + ".reml",
                    "--fast-mlma", "2slice",
                    "--threads", "4",
                    "--out", ref_mlma_k60,
                ],
                "mlma load ref syn k60",
            )
            compare_mlma("syn_k60_mlma_vs_ref", c4_mlma_k60 + ".mlma", ref_mlma_k60 + ".mlma", rel_tol=1e-4, abs_tol=1e-6)
        print("  PASS reml_fp16_fixed_k_parity_and_profile")

        # 3. Strip tuning, QR power pass tuning, blocked Gram (k=600 >= 512), and fp32 eigendecomp
        ref_test_k600 = os.path.join(work, "ref_test_k600")
        c4_test_def = os.path.join(work, "c4_test_k600_def")
        c4_test_p1 = os.path.join(work, "c4_test_k600_p1")
        c4_test_p2 = os.path.join(work, "c4_test_k600_p2")
        c4_test_fp32 = os.path.join(work, "c4_test_k600_fp32")

        log_test_def = must_run(
            [
                NEW_BIN, "reml",
                "--fam", test_bfile + ".fam",
                "--pheno", test_bfile + ".phen",
                "--grm", test_grm,
                "--fast",
                "--woodbury-basis", "600",
                "--seed", "7",
                "--threads", "4",
                "--reml-profile",
                "--out", c4_test_def,
            ],
            "reml test k600 default",
        )
        kv_test_def = extract_reml_profile(log_test_def)
        assert int(kv_test_def["matvec_tiles"]) > 0 and int(kv_test_def["matvec_gemm_calls"]) > 0, kv_test_def

        log_p1 = must_run(
            [
                NEW_BIN, "reml",
                "--fam", test_bfile + ".fam",
                "--pheno", test_bfile + ".phen",
                "--grm", test_grm,
                "--fast",
                "--woodbury-basis", "600",
                "--seed", "7",
                "--threads", "4",
                "--reml-matvec-band-rows", "128",
                "--reml-matvec-col-chunk", "128",
                "--reml-qr-power-passes", "1",
                "--reml-eigh-threads", "2",
                "--reml-io-slot-mb", "1",
                "--reml-profile",
                "--out", c4_test_p1,
            ],
            "reml test k600 strip+p1",
        )
        kv_p1 = extract_reml_profile(log_p1)

        log_p2 = must_run(
            [
                NEW_BIN, "reml",
                "--fam", test_bfile + ".fam",
                "--pheno", test_bfile + ".phen",
                "--grm", test_grm,
                "--fast",
                "--woodbury-basis", "600",
                "--seed", "7",
                "--threads", "4",
                "--reml-matvec-band-rows", "128",
                "--reml-matvec-col-chunk", "128",
                "--reml-qr-power-passes", "2",
                "--reml-eigh-threads", "2",
                "--reml-io-slot-mb", "1",
                "--reml-profile",
                "--out", c4_test_p2,
            ],
            "reml test k600 strip+p2",
        )
        kv_p2 = extract_reml_profile(log_p2)
        assert int(kv_p1["qr_passes"]) < int(kv_p2["qr_passes"]), (
            f"Expected qr_passes(1) < qr_passes(2), got {kv_p1['qr_passes']} vs {kv_p2['qr_passes']}"
        )

        log_fp32 = must_run(
            [
                NEW_BIN, "reml",
                "--fam", test_bfile + ".fam",
                "--pheno", test_bfile + ".phen",
                "--grm", test_grm,
                "--fast",
                "--woodbury-basis", "600",
                "--seed", "7",
                "--threads", "4",
                "--reml-eigh-fp32",
                "--reml-eigh-threads", "2",
                "--reml-profile",
                "--out", c4_test_fp32,
            ],
            "reml test k600 eigh fp32",
        )
        extract_reml_profile(log_fp32)

        compare_hsq("test_k600_def_vs_p1", c4_test_def + ".hsq", c4_test_p1 + ".hsq", rel_tol=1e-4, abs_tol=1e-5)
        compare_hsq("test_k600_def_vs_p2", c4_test_def + ".hsq", c4_test_p2 + ".hsq", rel_tol=1e-4, abs_tol=1e-5)
        compare_hsq("test_k600_def_vs_fp32", c4_test_def + ".hsq", c4_test_fp32 + ".hsq", rel_tol=2e-4, abs_tol=2e-5)

        if has_ref:
            must_run(
                [
                    REF_BIN, "reml",
                    "--fam", test_bfile + ".fam",
                    "--pheno", test_bfile + ".phen",
                    "--grm", test_grm,
                    "--fast",
                    "--woodbury-basis", "600",
                    "--seed", "7",
                    "--threads", "4",
                    "--out", ref_test_k600,
                ],
                "reml test k600 ref",
            )
            compare_hsq("test_k600_def_vs_ref", c4_test_def + ".hsq", ref_test_k600 + ".hsq", rel_tol=1e-4, abs_tol=1e-5)
            compare_hsq("test_k600_p1_vs_ref", c4_test_p1 + ".hsq", ref_test_k600 + ".hsq", rel_tol=1e-4, abs_tol=1e-5)
            compare_hsq("test_k600_fp32_vs_ref", c4_test_fp32 + ".hsq", ref_test_k600 + ".hsq", rel_tol=2e-4, abs_tol=2e-5)
        print("  PASS reml_strip_tuning_qr_passes_and_eigh_fp32")

        # 4. Adaptive EIG expansion parity (low mass target so syn triggers expansion and converges within k_max)
        c4_eig = os.path.join(work, "c4_eig")
        log_eig = must_run(
            [
                NEW_BIN, "reml",
                "--fam", syn_bfile + ".fam",
                "--pheno", syn_bfile + ".phen",
                "--grm", syn_grm,
                "--fast",
                "--woodbury-basis", "EIG",
                "--woodbury-eig-mass", "0.15",
                "--woodbury-range", "30", "200",
                "--seed", "7",
                "--threads", "4",
                "--reml-profile",
                "--out", c4_eig,
            ],
            "reml syn EIG new",
        )
        assert "locked expansion" in log_eig, f"Expected locked expansion in log:\n{log_eig}"
        kv_eig = extract_reml_profile(log_eig)
        assert int(kv_eig["matvec_tiles"]) > 0, kv_eig
        if has_ref:
            ref_eig = os.path.join(work, "ref_eig")
            must_run(
                [
                    REF_BIN, "reml",
                    "--fam", syn_bfile + ".fam",
                    "--pheno", syn_bfile + ".phen",
                    "--grm", syn_grm,
                    "--fast",
                    "--woodbury-basis", "EIG",
                    "--woodbury-eig-mass", "0.15",
                    "--woodbury-range", "30", "200",
                    "--seed", "7",
                    "--threads", "4",
                    "--out", ref_eig,
                ],
                "reml syn EIG ref",
            )
            compare_hsq("syn_eig_vs_ref", c4_eig + ".hsq", ref_eig + ".hsq", rel_tol=1e-4, abs_tol=1e-5)
        print("  PASS reml_adaptive_eig_expansion")

        # 5. Zero-Uk streaming TREF reuse + covariates + weighted phenotypes + downstream MLMA
        pca_pfx = os.path.join(work, "syn_pca")
        must_run(
            [
                NEW_BIN, "pca",
                "--grm", syn_grm,
                "--pcs", "6",
                "--eigenvec-format", "both",
                "--seed", "7",
                "--threads", "4",
                "--out", pca_pfx,
            ],
            "pca syn 6pcs",
        )
        qcovar_path = pca_pfx + ".eigenvec"
        assert os.path.isfile(qcovar_path), f"Missing {qcovar_path}"

        c4_base_qcov = os.path.join(work, "c4_base_qcov")
        must_run(
            [
                NEW_BIN, "reml",
                "--fam", syn_bfile + ".fam",
                "--pheno", syn_bfile + ".phen",
                "--qcovar", qcovar_path,
                "--grm", syn_grm,
                "--fast",
                "--woodbury-basis", "80",
                "--seed", "7",
                "--threads", "4",
                "--out", c4_base_qcov,
            ],
            "reml base qcov new",
        )
        assert read_reml_magic(c4_base_qcov + ".reml") == b"TUN3"

        reuse_on = os.path.join(work, "c4_reuse_on")
        log_ron = must_run(
            [
                NEW_BIN, "reml",
                "--fam", syn_bfile + ".fam",
                "--pheno", os.path.join(BED_DIR, "syn_w.phen"),
                "--weighted-phenotypes",
                "--qcovar", qcovar_path,
                "--woodbury-reuse", c4_base_qcov + ".reml",
                "--reml-reuse-stream", "on",
                "--reml-io-slot-mb", "1",
                "--reml-profile",
                "--threads", "4",
                "--out", reuse_on,
            ],
            "reml reuse stream on",
        )
        kv_ron = extract_reml_profile(log_ron)
        assert int(kv_ron["reuse_stream"]) == 1, kv_ron
        assert read_reml_magic(reuse_on + ".reml") == b"TREF"

        reuse_off = os.path.join(work, "c4_reuse_off")
        log_roff = must_run(
            [
                NEW_BIN, "reml",
                "--fam", syn_bfile + ".fam",
                "--pheno", os.path.join(BED_DIR, "syn_w.phen"),
                "--weighted-phenotypes",
                "--qcovar", qcovar_path,
                "--woodbury-reuse", c4_base_qcov + ".reml",
                "--reml-reuse-stream", "off",
                "--reml-profile",
                "--threads", "4",
                "--out", reuse_off,
            ],
            "reml reuse stream off",
        )
        kv_roff = extract_reml_profile(log_roff)
        assert int(kv_roff["reuse_stream"]) == 0, kv_roff
        assert read_reml_magic(reuse_off + ".reml") == b"TREF"
        compare_hsq("reuse_stream_on_vs_off", reuse_on + ".hsq", reuse_off + ".hsq", rel_tol=1e-5, abs_tol=1e-6)

        reuse_standalone = os.path.join(work, "c4_reuse_standalone")
        log_rsa = must_run(
            [
                NEW_BIN, "reml",
                "--fam", syn_bfile + ".fam",
                "--pheno", os.path.join(BED_DIR, "syn_w.phen"),
                "--weighted-phenotypes",
                "--qcovar", qcovar_path,
                "--woodbury-reuse", c4_base_qcov + ".reml",
                "--reml-standalone",
                "--reml-reuse-stream", "on",
                "--reml-profile",
                "--threads", "4",
                "--out", reuse_standalone,
            ],
            "reml reuse standalone",
        )
        kv_rsa = extract_reml_profile(log_rsa)
        assert int(kv_rsa["reuse_stream"]) == 0, kv_rsa
        assert read_reml_magic(reuse_standalone + ".reml") == b"TUN3"
        compare_hsq("reuse_standalone_vs_off", reuse_standalone + ".hsq", reuse_off + ".hsq", rel_tol=1e-5, abs_tol=1e-6)

        mlma_ron = os.path.join(work, "c4_mlma_reuse_on")
        mlma_roff = os.path.join(work, "c4_mlma_reuse_off")
        must_run(
            [
                NEW_BIN, "mlma",
                "--bfile", syn_bfile,
                "--pheno", os.path.join(BED_DIR, "syn_w.phen"),
                "--weighted-phenotypes",
                "--qcovar", qcovar_path,
                "--load-reml", reuse_on + ".reml",
                "--fast-mlma", "2slice",
                "--threads", "4",
                "--out", mlma_ron,
            ],
            "mlma from streaming TREF",
        )
        must_run(
            [
                NEW_BIN, "mlma",
                "--bfile", syn_bfile,
                "--pheno", os.path.join(BED_DIR, "syn_w.phen"),
                "--weighted-phenotypes",
                "--qcovar", qcovar_path,
                "--load-reml", reuse_off + ".reml",
                "--fast-mlma", "2slice",
                "--threads", "4",
                "--out", mlma_roff,
            ],
            "mlma from standard TREF",
        )
        compare_mlma("reuse_on_vs_off_mlma", mlma_ron + ".mlma", mlma_roff + ".mlma", rel_tol=1e-5, abs_tol=2e-6)

        if has_ref:
            ref_base_qcov = os.path.join(work, "ref_base_qcov")
            must_run(
                [
                    REF_BIN, "reml",
                    "--fam", syn_bfile + ".fam",
                    "--pheno", syn_bfile + ".phen",
                    "--qcovar", qcovar_path,
                    "--grm", syn_grm,
                    "--fast",
                    "--woodbury-basis", "80",
                    "--seed", "7",
                    "--threads", "4",
                    "--out", ref_base_qcov,
                ],
                "reml base qcov ref",
            )
            ref_reuse = os.path.join(work, "ref_reuse")
            must_run(
                [
                    REF_BIN, "reml",
                    "--fam", syn_bfile + ".fam",
                    "--pheno", os.path.join(BED_DIR, "syn_w.phen"),
                    "--weighted-phenotypes",
                    "--qcovar", qcovar_path,
                    "--woodbury-reuse", ref_base_qcov + ".reml",
                    "--threads", "4",
                    "--out", ref_reuse,
                ],
                "reml reuse ref",
            )
            compare_hsq("reuse_on_vs_ref", reuse_on + ".hsq", ref_reuse + ".hsq", rel_tol=1e-5, abs_tol=1e-6)
            compare_hsq("reuse_standalone_vs_ref", reuse_standalone + ".hsq", ref_reuse + ".hsq", rel_tol=1e-5, abs_tol=1e-6)

            ref_mlma_reuse = os.path.join(work, "ref_mlma_reuse")
            must_run(
                [
                    REF_BIN, "mlma",
                    "--bfile", syn_bfile,
                    "--pheno", os.path.join(BED_DIR, "syn_w.phen"),
                    "--weighted-phenotypes",
                    "--qcovar", qcovar_path,
                    "--load-reml", ref_reuse + ".reml",
                    "--fast-mlma", "2slice",
                    "--threads", "4",
                    "--out", ref_mlma_reuse,
                ],
                "mlma from ref TREF",
            )
            compare_mlma("reuse_on_mlma_vs_ref", mlma_ron + ".mlma", ref_mlma_reuse + ".mlma", rel_tol=1e-4, abs_tol=1e-6)
        print("  PASS reml_reuse_zero_uk_streaming_tref")

        print("ALL REML TUNING & TRACK C TESTS PASSED")
    finally:
        if tmp_ctx is not None:
            shutil.rmtree(tmp_ctx.name, ignore_errors=True)


if __name__ == "__main__":
    main()
