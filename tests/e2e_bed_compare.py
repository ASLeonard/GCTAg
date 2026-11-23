#!/usr/bin/env python3
"""End-to-end baseline-vs-new comparison for the native BED pipeline.

usage: e2e_bed_compare.py BASE_BIN NEW_BIN DATA_DIR WORK_DIR [case-filter]

DATA_DIR must contain the output of tests/gen_bed_testdata.py plus (optionally)
tests/data/test.{bed,bim,fam,phen,grm.*}.  Every case is run with both binaries;
.grm.bin/.grm.N.bin/.grm.id/.grm.sp must be byte-identical, .mlma files are
compared column-wise (exact for Chr..Freq, max |diff| reported for b/se/p).
"""
import os, subprocess, sys, struct, math, shutil

base, new, data, work = sys.argv[1:5]
filt = sys.argv[5] if len(sys.argv) > 5 else ""
T = ["--threads", "4"]
D = lambda f: os.path.join(data, f)

grm_cases = {
    "g_alg0":          ["--bfile", D("syn")],
    "g_alg1":          ["--bfile", D("syn"), "--alg", "1"],
    "g_alg0_keep":     ["--bfile", D("syn"), "--keep", D("keep_odd.txt")],
    "g_alg1_keep":     ["--bfile", D("syn"), "--alg", "1", "--keep", D("keep_odd.txt")],
    "g_alg1_small":    ["--bfile", D("syn"), "--alg", "1", "--keep", D("keep_small.txt")],
    "g_alg1_remove":   ["--bfile", D("syn"), "--alg", "1", "--remove", D("remove.txt")],
    "g_dom":           ["--bfile", D("syn"), "--dominance"],
    "g_dom_keep":      ["--bfile", D("syn"), "--dominance", "--keep", D("keep_small.txt")],
    "g_x":             ["--bfile", D("syn"), "--homogametic", "--homogametic-chr", "23"],
    "g_x_dc0":         ["--bfile", D("syn"), "--homogametic", "--homogametic-chr", "23", "--dc", "0"],
    "g_x_dc1":         ["--bfile", D("syn"), "--homogametic", "--homogametic-chr", "23", "--dc", "1"],
    "g_x_keep":        ["--bfile", D("syn"), "--homogametic", "--homogametic-chr", "23", "--keep", D("keep_odd.txt")],
    "g_x_keep_dc0":    ["--bfile", D("syn"), "--homogametic", "--homogametic-chr", "23", "--dc", "0", "--keep", D("keep_small.txt")],
    "g_part_alg1":     ["--bfile", D("syn"), "--alg", "1", "--part", "3", "2"],
    "g_part_alg1_k":   ["--bfile", D("syn"), "--alg", "1", "--part", "3", "3", "--keep", D("keep_odd.txt")],
    "g_part_alg0":     ["--bfile", D("syn"), "--part", "3", "2"],
    "g_tile_alg1":     ["--bfile", D("syn"), "--alg", "1", "--tile-budget", "0.004"],
    "g_tile_alg0":     ["--bfile", D("syn"), "--tile-budget", "0.004"],
    "g_tile_alg1_k":   ["--bfile", D("syn"), "--alg", "1", "--tile-budget", "0.003", "--keep", D("keep_odd.txt")],
    "g_nm100_alg1":    ["--bfile", D("syn"), "--alg", "1", "--n-markers", "100"],
    "g_nm64_alg1_k":   ["--bfile", D("syn"), "--alg", "1", "--n-markers", "64", "--keep", D("keep_odd.txt")],
    "g_nm100_alg0":    ["--bfile", D("syn"), "--n-markers", "100"],
    "g_mb_alg1":       ["--mbfile", D("syn.mbfile"), "--alg", "1"],
    "g_mb_alg0":       ["--mbfile", D("syn.mbfile")],
    "g_mb_x":          ["--mbfile", D("syn.mbfile"), "--homogametic", "--homogametic-chr", "23"],
    "g_zero_alg1":     ["--bfile", D("syn0"), "--alg", "1"],
    "g_zero_alg0":     ["--bfile", D("syn0")],
    "g_zero_alg1_k":   ["--bfile", D("syn0"), "--alg", "1", "--keep", D("keep_odd.txt")],
    "g_zero_dom":      ["--bfile", D("syn0"), "--dominance"],
    "g_maf_alg1":      ["--bfile", D("syn"), "--alg", "1", "--maf", "0.05", "--geno", "0.1"],
    "g_maf_alg0":      ["--bfile", D("syn"), "--maf", "0.05", "--geno", "0.1"],
    "g_sparse_alg1":   ["--bfile", D("syn"), "--alg", "1", "--tile-budget", "0.004", "--sparse-cutoff", "0.02"],
    "g_excl_alg1":     ["--bfile", D("syn"), "--alg", "1", "--exclude", D("exclude30.txt")],
    "g_excl_alg0_k":   ["--bfile", D("syn"), "--exclude", D("exclude30.txt"), "--keep", D("keep_odd.txt")],
    "g_extr_alg1":     ["--bfile", D("syn"), "--alg", "1", "--extract", D("extract_sparse.txt")],
    "g_mb_extr_alg1":  ["--mbfile", D("syn.mbfile"), "--alg", "1", "--extract", D("extract_sparse.txt")],
    "g_mb_excl_x":     ["--mbfile", D("syn.mbfile"), "--homogametic", "--homogametic-chr", "23", "--exclude", D("exclude30.txt")],
    "g_test_alg0":     ["--bfile", D("test")],
    "g_test_alg1":     ["--bfile", D("test"), "--alg", "1"],
}

mlma_cases = {
    "m_dense":        ["--bfile", D("syn"), "--grm", "@g_alg0", "--pheno", D("syn.phen")],
    "m_dense_keep":   ["--bfile", D("syn"), "--grm", "@g_alg0", "--pheno", D("syn.phen"), "--keep", D("keep_odd.txt")],
    "m_wb":           ["--bfile", D("syn"), "--grm", "@g_alg0", "--pheno", D("syn.phen"), "--woodbury-basis", "40", "--seed", "7"],
    "m_wb_keep":      ["--bfile", D("syn"), "--grm", "@g_alg0", "--pheno", D("syn.phen"), "--woodbury-basis", "40", "--seed", "7", "--keep", D("keep_odd.txt")],
    "m_fast":         ["--bfile", D("syn"), "--grm", "@g_alg0", "--pheno", D("syn.phen"), "--woodbury-basis", "40", "--seed", "7", "--fast-mlma", "2slice"],
    "m_fast_keep":    ["--bfile", D("syn"), "--grm", "@g_alg0", "--pheno", D("syn.phen"), "--woodbury-basis", "40", "--seed", "7", "--fast-mlma", "hybrid", "--keep", D("keep_small.txt")],
    "m_dom":          ["--bfile", D("syn"), "--grm", "@g_alg0", "--pheno", D("syn.phen"), "--model", "dominance"],
    "m_nonadd":       ["--bfile", D("syn"), "--grm", "@g_alg0", "--pheno", D("syn.phen"), "--model", "nonadditive"],
    "m_wb_zero":      ["--bfile", D("syn0"), "--grm", "@g_alg0", "--pheno", D("syn.phen"), "--woodbury-basis", "40", "--seed", "7"],
    "m_x":            ["--bfile", D("syn"), "--grm", "@g_alg0", "--pheno", D("syn.phen"), "--autosome-sex", "--homogametic-chr", "23"],
    "m_x_dc0":        ["--bfile", D("syn"), "--grm", "@g_alg0", "--pheno", D("syn.phen"), "--autosome-sex", "--homogametic-chr", "23", "--dc", "0"],
    "m_x_wb":         ["--bfile", D("syn"), "--grm", "@g_alg0", "--pheno", D("syn.phen"), "--autosome-sex", "--homogametic-chr", "23", "--woodbury-basis", "40", "--seed", "7"],
    "m_mb_wb":        ["--mbfile", D("syn.mbfile"), "--grm", "@g_alg0", "--pheno", D("syn.phen"), "--woodbury-basis", "40", "--seed", "7"],
    "m_dense_w":      ["--bfile", D("syn"), "--grm", "@g_alg0", "--pheno", D("syn_w.phen"), "--weighted-phenotypes"],
    "m_wb_w":         ["--bfile", D("syn"), "--grm", "@g_alg0", "--pheno", D("syn_w.phen"), "--weighted-phenotypes", "--woodbury-basis", "40", "--seed", "7"],
    "m_wb_w_keep":    ["--bfile", D("syn"), "--grm", "@g_alg0", "--pheno", D("syn_w.phen"), "--weighted-phenotypes", "--woodbury-basis", "40", "--seed", "7", "--keep", D("keep_odd.txt")],
    "m_fast_w":       ["--bfile", D("syn"), "--grm", "@g_alg0", "--pheno", D("syn_w.phen"), "--weighted-phenotypes", "--woodbury-basis", "40", "--seed", "7", "--fast-mlma", "1slice"],
    "m_wb_excl":      ["--bfile", D("syn"), "--grm", "@g_alg0", "--pheno", D("syn.phen"), "--woodbury-basis", "40", "--seed", "7", "--exclude", D("exclude30.txt")],
    "m_mb_extr":      ["--mbfile", D("syn.mbfile"), "--grm", "@g_alg0", "--pheno", D("syn.phen"), "--extract", D("extract_sparse.txt")],
    "m_test":         ["--bfile", D("test"), "--grm", D("test"), "--pheno", D("test.phen")],
    "m_test_wb":      ["--bfile", D("test"), "--grm", D("test"), "--pheno", D("test.phen"), "--woodbury-basis", "50", "--seed", "3"],
}

def run(binary, args, out):
    cmd = [binary] + args + T + ["--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-1500:]

def cmp_bytes(a, b):
    if not os.path.exists(a) and not os.path.exists(b):
        return None
    if os.path.exists(a) != os.path.exists(b):
        return "exists mismatch"
    x, y = open(a, "rb").read(), open(b, "rb").read()
    if x == y:
        return "identical"
    if len(x) != len(y):
        return f"size {len(x)} vs {len(y)}"
    if a.endswith(".bin"):
        n = len(x) // 4
        fx = struct.unpack(f"{n}f", x); fy = struct.unpack(f"{n}f", y)
        nd = sum(1 for i in range(n) if fx[i] != fy[i])
        md = max(abs(fx[i] - fy[i]) for i in range(n))
        return f"DIFF {nd}/{n} floats, max|d|={md:.3g}"
    return "DIFF"

def cmp_mlma(a, b):
    if not (os.path.exists(a) and os.path.exists(b)):
        return "missing file"
    la, lb = open(a).read().splitlines(), open(b).read().splitlines()
    if la == lb:
        return "identical"
    if len(la) != len(lb):
        return f"rows {len(la)} vs {len(lb)}"
    exact_bad = 0; mx = [0.0, 0.0, 0.0]; rel = [0.0, 0.0, 0.0]; nd = 0
    for x, y in zip(la[1:], lb[1:]):
        cx, cy = x.split("\t"), y.split("\t")
        if cx[:6] != cy[:6]:
            exact_bad += 1
        if cx != cy:
            nd += 1
        for k in range(3):
            try:
                u, v = float(cx[6 + k]), float(cy[6 + k])
            except ValueError:
                if cx[6 + k] != cy[6 + k]:
                    exact_bad += 1
                continue
            d = abs(u - v)
            mx[k] = max(mx[k], d)
            if max(abs(u), abs(v)) > 0:
                rel[k] = max(rel[k], d / max(abs(u), abs(v)))
    return (f"{'FAIL' if exact_bad else 'close'} rows_diff={nd}/{len(la)-1} exact_col_mismatch={exact_bad} "
            f"max|d| b={mx[0]:.2g} se={mx[1]:.2g} p={mx[2]:.2g} maxrel b={rel[0]:.2g} se={rel[1]:.2g} p={rel[2]:.2g}")

os.makedirs(work, exist_ok=True)
ok = True
for name, args in grm_cases.items():
    if filt and filt not in name:
        continue
    res = {}
    for tag, b in (("base", base), ("new", new)):
        out = os.path.join(work, f"{name}.{tag}")
        rc, log = run(b, ["grm", "make"] + args, out)
        res[tag] = (rc, log, out)
    if res["base"][0] != 0 or res["new"][0] != 0:
        print(f"{name:18s} rc base={res['base'][0]} new={res['new'][0]}")
        if res["base"][0] != res["new"][0]:
            ok = False
            print(res["new"][1])
        continue
    msgs = []
    pb = os.path.basename(res["base"][2])
    exts = sorted(f[len(pb):] for f in os.listdir(work)
                  if f.startswith(pb) and ".grm." in f[len(pb):])
    if not exts:
        ok = False
        msgs.append("NO GRM OUTPUT")
    for ext in exts:
        m = cmp_bytes(res["base"][2] + ext, res["new"][2] + ext)
        if m is None:
            continue
        msgs.append(f"{ext}:{m}")
        if m != "identical":
            ok = False
    print(f"{name:18s} " + " ".join(msgs))

for name, args in mlma_cases.items():
    if filt and filt not in name:
        continue
    args = [os.path.join(work, a[1:] + ".base") if a.startswith("@") else a for a in args]
    res = {}
    for tag, b in (("base", base), ("new", new)):
        out = os.path.join(work, f"{name}.{tag}")
        rc, log = run(b, ["mlma"] + args, out)
        res[tag] = (rc, log, out)
    if res["base"][0] != 0 or res["new"][0] != 0:
        print(f"{name:18s} rc base={res['base'][0]} new={res['new'][0]}")
        if res["base"][0] != res["new"][0]:
            ok = False
            print(res["new"][1])
        continue
    m = cmp_mlma(res["base"][2] + ".mlma", res["new"][2] + ".mlma")
    if m.startswith("FAIL") or m.startswith("rows") or m.startswith("missing"):
        ok = False
    print(f"{name:18s} {m}")

print("ALL OK" if ok else "SOME FAILURES")
sys.exit(0 if ok else 1)
