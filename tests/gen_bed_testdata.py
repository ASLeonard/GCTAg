#!/usr/bin/env python3
"""Synthetic PLINK1 test data for the native BED pipeline (pure python).

Creates in <outdir>:
  syn.{bed,bim,fam}      N=2003 samples (sex 1/2 + a few 0), chr 1, 2 (autosomes)
                         and 23 (X).  Per-SNP missing rate drawn from
                         {0 (45%), 0.001, 0.01, 0.05, 0.2}; a few monomorphic /
                         all-missing-ish SNPs; male X hets on some SNPs.
  syn0.{bed,bim,fam}     same genotypes with ZERO missing (post-imputation style)
  synA/synB.{bed,bim,fam} syn split by chromosome (chr1 | chr2+X) for --mbfile
  syn.mbfile             list for --mbfile
  syn.phen               phenotype
  keep_odd.txt           1371 random samples (non-aligned subset)
  keep_small.txt         257 samples
  remove.txt             131 samples
"""
import os, random, struct, sys

out = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out, exist_ok=True)
rng = random.Random(20261120)
N = 2003
CHRS = [("1", 1300), ("2", 1100), ("23", 420)]

sex = []
for i in range(N):
    r = rng.random()
    sex.append(0 if r < 0.01 else (1 if r < 0.5 else 2))

fam_lines = [f"F{i} I{i} 0 0 {sex[i]} -9" for i in range(N)]

snps = []   # (chr, name, bp, a1, a2, genos list with None=missing)
for chrom, m in CHRS:
    for j in range(m):
        p = rng.choice([0.0, 0.002, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.95, 1.0]) if j % 37 == 0 else rng.uniform(0.01, 0.99)
        mr = rng.choices([0.0, 0.001, 0.01, 0.05, 0.2, 0.9], weights=[45, 15, 15, 12, 10, 3])[0]
        g = []
        for i in range(N):
            if mr and rng.random() < mr:
                g.append(None)
                continue
            if chrom == "23" and sex[i] == 1:
                # males hemizygous: 0/2, occasional het error on some SNPs
                if j % 50 == 7 and rng.random() < 0.02:
                    g.append(1)
                else:
                    g.append(2 if rng.random() < p else 0)
            else:
                g.append((rng.random() < p) + (rng.random() < p))
        snps.append((chrom, f"rs{chrom}_{j}", 1000 + 137 * j, "A", "G", g))

CODE = {2: 0b00, 1: 0b10, 0: 0b11, None: 0b01}

def write_bed(prefix, sel, zero_missing=False):
    nb = (N + 3) // 4
    with open(prefix + ".bed", "wb") as f:
        f.write(bytes([0x6C, 0x1B, 0x01]))
        for s in sel:
            g = s[5]
            buf = bytearray(nb)
            for i, x in enumerate(g):
                if zero_missing and x is None:
                    x = 0 if (i * 7 + len(s[1])) % 3 else 1
                buf[i >> 2] |= CODE[x] << (2 * (i & 3))
            # garbage in trailing padding bits of the last byte (must be ignored)
            rem = N & 3
            if rem:
                buf[-1] |= (0b01010101 << (2 * rem)) & 0xFF
            f.write(bytes(buf))
    with open(prefix + ".bim", "w") as f:
        for s in sel:
            f.write(f"{s[0]}\t{s[1]}\t0\t{s[2]}\t{s[3]}\t{s[4]}\n")
    with open(prefix + ".fam", "w") as f:
        f.write("\n".join(fam_lines) + "\n")

write_bed(os.path.join(out, "syn"), snps)
write_bed(os.path.join(out, "syn0"), snps, zero_missing=True)
write_bed(os.path.join(out, "synA"), [s for s in snps if s[0] == "1"])
write_bed(os.path.join(out, "synB"), [s for s in snps if s[0] != "1"])
with open(os.path.join(out, "syn.mbfile"), "w") as f:
    f.write(os.path.join(out, "synA") + "\n" + os.path.join(out, "synB") + "\n")

ys = [rng.gauss(0, 1) for _ in range(N)]
with open(os.path.join(out, "syn.phen"), "w") as f:
    for i in range(N):
        f.write(f"F{i} I{i} {ys[i]:.6f}\n")
# phenotype + per-sample residual weight column (--weighted-phenotypes)
with open(os.path.join(out, "syn_w.phen"), "w") as f:
    for i in range(N):
        f.write(f"F{i} I{i} {ys[i]:.6f} {rng.uniform(0.5, 2.0):.4f}\n")

def write_ids(name, idx):
    with open(os.path.join(out, name), "w") as f:
        for i in sorted(idx):
            f.write(f"F{i} I{i}\n")

# variant subsets exercising non-contiguous reads (gap merging / new batches)
names = [x[1] for x in snps]
with open(os.path.join(out, "exclude30.txt"), "w") as f:
    f.write("\n".join(sorted(rng.sample(names, int(0.3 * len(names))))) + "\n")
with open(os.path.join(out, "extract_sparse.txt"), "w") as f:
    sel = [n for k, n in enumerate(names) if k % 23 == 0 or (400 <= k % 1000 < 440)]
    f.write("\n".join(sel) + "\n")

write_ids("keep_odd.txt", rng.sample(range(N), 1371))
write_ids("keep_small.txt", rng.sample(range(N), 257))
write_ids("remove.txt", rng.sample(range(N), 131))
print("wrote", out)
