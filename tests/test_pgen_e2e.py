#!/usr/bin/env python3
"""End-to-end PLINK1 (.bed) vs PLINK2 (.pgen) parity & fallback regression suite.

Exercises:
  * Matched .bed/.bim/.fam and .pgen/.pvar/.psam/.pgi datasets (N=317, non-multiple
    of 4/8/64/256, nonzero trailing .bed padding bits, autosomes 1,2 + sex chr 23)
  * Mode 0x01 (.bed via --bpfile), Mode 0x02 (fixed-width hardcall), Mode 0x10
    (inline variable-width across ALL vrtypes 0..7 + phased hardcalls | 0x10:
    vrtype 0 2-bit, vrtype 1 1-bit + difflist across all 6 common2_code pairs and
    multi-group L>64 VINT31 deltas, vrtype 2 LD-compressed chains, vrtype 3
    inverted-LD with missing transitions, vrtype 4/6/7 pure difflists, vrtype 5
    0-byte all-hom-ref), and Mode 0x20 + external .pgen.pgi index
  * --bfile, --mbfile, --pfile, --bpfile, --mpfile, --mbpfile
  * Missing-call and zero-missing cohorts, --keep, --remove, --extract explicitly
    dropping ldbase anchors, --maf / --geno multi-pass filtering
  * Bit-for-bit parity on:
      - grm make (.grm.bin + .grm.N.bin + .grm.id, --alg 0 & --alg 1, --grm-fp16,
        --fast, --impute-mean, --grm-profile compressed I/O & unpack stats,
        multi-tile --GRM-tile-budget, --dominance, --homogametic, --sparse-cutoff,
        --part, --grm-legacy-engine, --update-freq)
      - mlma (INT8 Woodbury wb_std_i8, quantized --fast-mlma adaptive/2slice/
        hybrid/1slice, two-step reml TUN3/TREF + --load-reml, dense FP32, --model
        dominance/nonadditive, --stream-budget compact PGEN tile sizing,
        --weighted-phenotypes, --qcovar, --autosome-sex)
      - pca (--pcs 5 on PGEN GRM vs BED GRM)
      - legacy --freq / --make-bed / --ld cross-checks & V2 CLI diagnostics
  * Dosage .pgen fallback (mode 0x03 fixed dosage & mode 0x10 mixed vrtype 0x00 /
    sparse 0x60 / dense 0x40 with fractional & missing 0xFFFF dosages, --keep,
    and --impute-mean rejection)
  * GCTA_DISABLE_NATIVE_PGEN=1 fallback parity

Usage:
    python3 tests/test_pgen_e2e.py [GCTAG_BIN] [WORK_DIR]
"""

from __future__ import annotations

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
from typing import Dict, List, Optional, Sequence, Tuple

GCTAG_BIN = (
    os.path.abspath(sys.argv[1])
    if len(sys.argv) > 1
    else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "GCTAg"))
)
USER_WORK = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else ""

# PLINK2 2-bit code (0=homREF/A2, 1=het, 2=homALT/A1, 3=missing) -> PLINK1 .bed 2-bit code
P2_TO_BED = {0: 0b11, 1: 0b10, 2: 0b00, 3: 0b01}
P2_INV = {0: 2, 1: 1, 2: 0, 3: 3}

PASS_COUNT = 0


def ok(label: str) -> None:
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  PASS [{PASS_COUNT:03d}] {label}")


def run_cmd(
    args: Sequence[str],
    env_extra: Optional[Dict[str, str]] = None,
    expect_rc: Optional[int] = 0,
    desc: str = "",
) -> Tuple[int, str]:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(list(args), capture_output=True, text=True, env=env)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if expect_rc is not None and proc.returncode != expect_rc:
        raise AssertionError(
            f"{desc or 'command'} exited with rc={proc.returncode} (expected {expect_rc}):\n"
            f"  cmd: {' '.join(args)}\n{out[-3000:]}"
        )
    return proc.returncode, out


# -----------------------------------------------------------------------------
# Pure-Python PLINK1 & PLINK2 binary encoders
# -----------------------------------------------------------------------------
def sample_id_byte_ct(raw_n: int) -> int:
    assert raw_n >= 1
    return 1 + (raw_n.bit_length() - 1) // 8


def encode_vint31(val: int) -> bytes:
    assert 0 <= val < (1 << 31)
    out = bytearray()
    while val > 0x7F:
        out.append((val & 0x7F) | 0x80)
        val >>= 7
    out.append(val)
    return bytes(out)


def encode_sub_u32_le(val: int, byte_ct: int) -> bytes:
    return int(val).to_bytes(byte_ct, "little", signed=False)


def pack_p2_2bit(codes: Sequence[int]) -> bytes:
    n = len(codes)
    out = bytearray((n + 3) // 4)
    for i, c in enumerate(codes):
        out[i >> 2] |= (c & 3) << ((i & 3) * 2)
    return bytes(out)


def pack_bed_2bit(codes: Sequence[int], dirty_tail: bool = True) -> bytes:
    n = len(codes)
    out = bytearray((n + 3) // 4)
    for i, c in enumerate(codes):
        out[i >> 2] |= P2_TO_BED[c & 3] << ((i & 3) * 2)
    rem = n & 3
    if dirty_tail and rem:
        out[-1] |= (0b01010101 << (2 * rem)) & 0xFF
    return bytes(out)


def encode_difflist(raw_n: int, diffs: Sequence[Tuple[int, int]]) -> bytes:
    """Encode strictly increasing (sample_idx, p2_code) pairs as a PLINK2 difflist."""
    L = len(diffs)
    out = bytearray(encode_vint31(L))
    if L == 0:
        return bytes(out)
    assert L <= raw_n // 8, f"difflist L={L} exceeds raw_n//8={raw_n // 8}"
    group_ct = (L + 63) // 64
    sid_bytes = sample_id_byte_ct(raw_n)

    group_deltas: List[bytearray] = []
    for g in range(group_ct):
        base = g * 64
        cnt = min(64, L - base)
        out.extend(encode_sub_u32_le(diffs[base][0], sid_bytes))
        gd = bytearray()
        for e in range(1, cnt):
            delta = diffs[base + e][0] - diffs[base + e - 1][0]
            assert delta > 0
            gd.extend(encode_vint31(delta))
        group_deltas.append(gd)

    for g in range(group_ct - 1):
        seg_minus_63 = len(group_deltas[g]) - 63
        assert 0 <= seg_minus_63 <= 255
        out.append(seg_minus_63)

    rg = bytearray((L + 3) // 4)
    for idx, (_, code) in enumerate(diffs):
        rg[idx >> 2] |= (code & 3) << ((idx & 3) * 2)
    out.extend(rg)

    for gd in group_deltas:
        out.extend(gd)
    return bytes(out)


def make_hphase_tail(codes: Sequence[int], seed: int) -> bytes:
    """Append valid PLINK2 hardcall phase bitarrays for (vrtype | 0x10) records."""
    het_ct = sum(1 for c in codes if c == 1)
    # First bit of hphase section is all_phased flag (bit 0), followed by phaseinfo
    # bits for het calls (if all_phased=1) or phasepresent + phaseinfo bits.
    # Setting bit 0 = 1 (all_phased) needs ceil((1 + het_ct) / 8) bytes!
    total_bits = 1 + het_ct
    buf = bytearray((total_bits + 7) // 8)
    buf[0] |= 1  # explicit all-hets-phased bit
    rng = random.Random(seed)
    for h in range(het_ct):
        if rng.getrandbits(1):
            bpos = 1 + h
            buf[bpos >> 3] |= 1 << (bpos & 7)
    return bytes(buf)


def encode_onebit_record(
    codes: Sequence[int], c0: int, c1: int, phased: bool = False, seed: int = 0
) -> Tuple[int, bytes]:
    assert 0 <= c0 < c1 <= 3
    n = len(codes)
    common2_code = (c0 << 2) | (c1 - c0)
    onebit_bytes = bytearray(((n + 15) // 8) - 1)
    diffs: List[Tuple[int, int]] = []
    for s, g in enumerate(codes):
        if g == c0:
            pass
        elif g == c1:
            onebit_bytes[s >> 3] |= 1 << (s & 7)
        else:
            # Leave 0 in bitarray (c0), patch via difflist
            diffs.append((s, g))
    assert len(diffs) <= n // 8
    payload = bytearray([common2_code])
    payload.extend(onebit_bytes)
    payload.extend(encode_difflist(n, diffs))
    vt = 1 | (0x10 if phased else 0)
    if phased:
        payload.extend(make_hphase_tail(codes, seed))
    return vt, bytes(payload)


def encode_var_pgen_0x10_or_0x20(
    raw_n: int,
    vrtypes: Sequence[int],
    payloads: Sequence[bytes],
    external_pgi: bool = False,
    len_bytes: int = 2,
    with_nonref_flags: bool = False,
) -> Tuple[bytes, Optional[bytes]]:
    raw_m = len(vrtypes)
    assert raw_m == len(payloads) and raw_m > 0
    vblock_ct = (raw_m + 65535) // 65536
    header_ctrl = ((len_bytes - 1) & 3) | 4  # 8-bit vrtypes
    if with_nonref_flags:
        header_ctrl |= 3 << 6

    meta = bytearray()
    for vb in range(vblock_ct):
        v0 = vb * 65536
        cnt = min(65536, raw_m - v0)
        assert (vrtypes[v0] & 6) != 2, "first variant in vblock cannot be LD"
        for i in range(cnt):
            meta.append(vrtypes[v0 + i] & 0xFF)
        for i in range(cnt):
            meta.extend(encode_sub_u32_le(len(payloads[v0 + i]), len_bytes))
        if with_nonref_flags:
            meta.extend(b"\x5A" * ((cnt + 7) // 8))

    if external_pgi:
        first_fpos = 3
        vb_fpos = []
        pos = first_fpos
        for v in range(raw_m):
            if v % 65536 == 0:
                vb_fpos.append(pos)
            pos += len(payloads[v])
        pgen = bytearray([0x6C, 0x1B, 0x20])
        for pl in payloads:
            pgen.extend(pl)
        pgi = bytearray([0x6C, 0x1B, 0x30])
        pgi.extend(struct.pack("<IIB", raw_m, raw_n, header_ctrl))
        for fp in vb_fpos:
            pgi.extend(struct.pack("<Q", fp))
        pgi.extend(meta)
        return bytes(pgen), bytes(pgi)

    first_fpos = 12 + 8 * vblock_ct + len(meta)
    vb_fpos = []
    pos = first_fpos
    for v in range(raw_m):
        if v % 65536 == 0:
            vb_fpos.append(pos)
        pos += len(payloads[v])

    pgen = bytearray([0x6C, 0x1B, 0x10])
    pgen.extend(struct.pack("<IIB", raw_m, raw_n, header_ctrl))
    for fp in vb_fpos:
        pgen.extend(struct.pack("<Q", fp))
    pgen.extend(meta)
    for pl in payloads:
        pgen.extend(pl)
    return bytes(pgen), None


def encode_fixed_pgen_0x02(raw_n: int, var_codes: Sequence[Sequence[int]]) -> bytes:
    raw_m = len(var_codes)
    pgen = bytearray([0x6C, 0x1B, 0x02])
    pgen.extend(struct.pack("<IIB", raw_m, raw_n, 0))
    for codes in var_codes:
        pgen.extend(pack_p2_2bit(codes))
    return bytes(pgen)


def write_fam(path: str, samples: Sequence[Tuple[str, str, int]]) -> None:
    with open(path, "w") as f:
        for fid, iid, sex in samples:
            f.write(f"{fid}\t{iid}\t0\t0\t{sex}\t-9\n")


def write_psam(path: str, samples: Sequence[Tuple[str, str, int]]) -> None:
    with open(path, "w") as f:
        f.write("#FID\tIID\tPAT\tMAT\tSEX\n")
        for fid, iid, sex in samples:
            sex_str = str(sex) if sex in (1, 2) else "NA"
            f.write(f"{fid}\t{iid}\t0\t0\t{sex_str}\n")


def write_bim(path: str, snps: Sequence[Tuple[str, str, int, str, str, List[int]]]) -> None:
    with open(path, "w") as f:
        for chrom, rsid, bp, a1, a2, _ in snps:
            f.write(f"{chrom}\t{rsid}\t0\t{bp}\t{a1}\t{a2}\n")


def write_pvar(path: str, snps: Sequence[Tuple[str, str, int, str, str, List[int]]]) -> None:
    # In PLINK2 PVAR, REF = A2 and ALT = A1 so p2 code {0,1,2} == copies of A1 (ALT).
    with open(path, "w") as f:
        f.write("##fileformat=PVARv1.0\n#CHROM\tPOS\tID\tREF\tALT\n")
        for chrom, rsid, bp, a1, a2, _ in snps:
            f.write(f"{chrom}\t{bp}\t{rsid}\t{a2}\t{a1}\n")


def write_bed_fileset(
    prefix: str,
    samples: Sequence[Tuple[str, str, int]],
    snps: Sequence[Tuple[str, str, int, str, str, List[int]]],
) -> None:
    with open(prefix + ".bed", "wb") as bf:
        bf.write(bytes([0x6C, 0x1B, 0x01]))
        for *_, codes in snps:
            bf.write(pack_bed_2bit(codes, dirty_tail=True))
    write_bim(prefix + ".bim", snps)
    write_fam(prefix + ".fam", samples)


def compress_snps_all_vrtypes(
    raw_n: int,
    snps: Sequence[Tuple[str, str, int, str, str, List[int]]],
) -> Tuple[List[int], List[bytes], List[int]]:
    """Encode SNPs into rich Mode 0x10 records exercising vrtypes 0..7 and | 0x10."""
    vrtypes: List[int] = []
    payloads: List[bytes] = []
    ld_children_indices: List[int] = []

    last_non_ld_codes: Optional[List[int]] = None

    for v, (chrom, _rsid, _bp, _a1, _a2, codes) in enumerate(snps):
        phased = (v % 3 == 1) and chrom != "23"
        ph_bit = 0x10 if phased else 0

        # Check if this variant was constructed as an LD or inverted-LD child
        if v % 65536 != 0 and last_non_ld_codes is not None:
            direct_diffs = [(s, g) for s, (b, g) in enumerate(zip(last_non_ld_codes, codes)) if b != g]
            if len(direct_diffs) <= raw_n // 8 and (v % 9 in (2, 3)):
                vt = 2 | ph_bit
                pl = bytearray(encode_difflist(raw_n, direct_diffs))
                if phased:
                    pl.extend(make_hphase_tail(codes, 1000 + v))
                vrtypes.append(vt)
                payloads.append(bytes(pl))
                ld_children_indices.append(v)
                continue

            inv_diffs = [
                (s, P2_INV[g])
                for s, (b, g) in enumerate(zip(last_non_ld_codes, codes))
                if P2_INV[b] != g
            ]
            if len(inv_diffs) <= raw_n // 8 and (v % 9 == 4):
                vt = 3 | ph_bit
                pl = bytearray(encode_difflist(raw_n, inv_diffs))
                if phased:
                    pl.extend(make_hphase_tail(codes, 2000 + v))
                vrtypes.append(vt)
                payloads.append(bytes(pl))
                ld_children_indices.append(v)
                continue

        # All hom-ref -> vrtype 5
        if all(g == 0 for g in codes):
            vt = 5 | ph_bit
            pl = make_hphase_tail(codes, 3000 + v) if phased else b""
            vrtypes.append(vt)
            payloads.append(pl)
            last_non_ld_codes = list(codes)
            continue

        # Pure difflist vrtypes 4 (modal 0), 6 (modal 2), 7 (modal 3/missing)
        counts = [codes.count(c) for c in range(4)]
        for modal_c, base_vt in ((0, 4), (2, 6), (3, 7)):
            if raw_n - counts[modal_c] <= raw_n // 8 and (modal_c != 0 or v % 5 != 0):
                diffs = [(s, g) for s, g in enumerate(codes) if g != modal_c]
                vt = base_vt | ph_bit
                pl = bytearray(encode_difflist(raw_n, diffs))
                if phased:
                    pl.extend(make_hphase_tail(codes, 4000 + v))
                vrtypes.append(vt)
                payloads.append(bytes(pl))
                last_non_ld_codes = list(codes)
                break
        else:
            # Check 1-bit + difflist (vrtype 1) across any valid pair (c0 < c1)
            best_pair = None
            for c0 in range(3):
                for c1 in range(c0 + 1, 4):
                    rem = raw_n - counts[c0] - counts[c1]
                    if rem <= raw_n // 8:
                        best_pair = (c0, c1)
                        break
                if best_pair is not None:
                    break
            if best_pair is not None and v % 4 != 0:
                vt, pl = encode_onebit_record(codes, best_pair[0], best_pair[1], phased=phased, seed=5000 + v)
                vrtypes.append(vt)
                payloads.append(pl)
                last_non_ld_codes = list(codes)
            else:
                # vrtype 0 (2-bit PLINK2 bitarray)
                vt = 0 | ph_bit
                pl = bytearray(pack_p2_2bit(codes))
                if phased:
                    pl.extend(make_hphase_tail(codes, 6000 + v))
                vrtypes.append(vt)
                payloads.append(bytes(pl))
                last_non_ld_codes = list(codes)

    return vrtypes, payloads, ld_children_indices


def generate_cohort(
    data_dir: str,
) -> Dict[str, object]:
    """Generate matched PLINK1 & PLINK2 cohorts with every vrtype and edge case."""
    rng = random.Random(20261126)
    N = 317  # Non-multiple of 4 (rem=1), 8, 64, 256; sample_id_byte_ct=2; raw_n//8=39
    samples: List[Tuple[str, str, int]] = []
    for i in range(N):
        r = rng.random()
        sex = 0 if r < 0.02 else (1 if r < 0.51 else 2)
        samples.append((f"FAM{i}", f"IND{i}", sex))

    def build_snps(allow_missing: bool) -> List[Tuple[str, str, int, str, str, List[int]]]:
        snps: List[Tuple[str, str, int, str, str, List[int]]] = []
        alleles = [("A", "G"), ("C", "T"), ("G", "A"), ("T", "C")]

        # Explicit curated block on chr 1 guaranteeing every vrtype & common2_code
        curated: List[List[int]] = []
        # 0: dense vrtype 0 anchor (MAF ~0.42)
        anchor = [
            (3 if (allow_missing and s % 47 == 3) else ((rng.random() < 0.42) + (rng.random() < 0.42)))
            for s in range(N)
        ]
        curated.append(anchor)
        # 1: all hom-ref -> vrtype 5
        curated.append([0] * N)
        # 2, 3: LD children (vrtype 2 chain of length 2 from anchor? Wait: v=1 was vrtype 5!
        # Put a fresh anchor at v=2, followed by two vrtype 2 children and one vrtype 3 child!)
        anchor2 = [
            (3 if (allow_missing and s in (4, 88, 211)) else ((rng.random() < 0.35) + (rng.random() < 0.35)))
            for s in range(N)
        ]
        curated.append(anchor2)  # v=2 (non-LD anchor)
        ld1 = list(anchor2)
        for s in rng.sample(range(N), 18):
            ld1[s] = (ld1[s] + 1) % (4 if allow_missing else 3)
        curated.append(ld1)  # v=3 -> vrtype 2
        ld2 = list(anchor2)
        for s in rng.sample(range(N), 25):
            ld2[s] = (ld2[s] + 2) % (4 if allow_missing else 3)
        curated.append(ld2)  # v=4 -> vrtype 3? Wait: v=4 has v%9==4! Make inv_ld at v=4:
        inv_ld = [P2_INV[g] for g in anchor2]
        for s in rng.sample(range(N), 14):
            inv_ld[s] = 3 if (allow_missing and s % 3 == 0) else ((inv_ld[s] + 1) % 3)
        curated[-1] = inv_ld  # v=4 -> vrtype 3!

        # Pure difflists: vrtype 4 (modal 0), vrtype 6 (modal 2), vrtype 7 (modal 3 if allow_missing)
        vt4_codes = [0] * N
        for s in rng.sample(range(N), 31):
            vt4_codes[s] = 3 if (allow_missing and s % 7 == 0) else (2 if s % 5 == 0 else 1)
        curated.append(vt4_codes)

        vt6_codes = [2] * N
        for s in rng.sample(range(N), 29):
            vt6_codes[s] = 3 if (allow_missing and s % 6 == 0) else (0 if s % 4 == 0 else 1)
        curated.append(vt6_codes)

        if allow_missing:
            vt7_codes = [3] * N  # ~90% missing (tested with --geno 1.0 default & filtered by --geno 0.1)
            for s in rng.sample(range(N), 34):
                vt7_codes[s] = s % 3
            curated.append(vt7_codes)

        # All 6 common2_code pairs for vrtype 1: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        pairs = [(0, 1), (0, 2), (1, 2)]
        if allow_missing:
            pairs += [(0, 3), (1, 3), (2, 3)]
        for c0, c1 in pairs:
            arr = [c0 if rng.random() < 0.55 else c1 for _ in range(N)]
            other = [x for x in (0, 1, 2, (3 if allow_missing else 0)) if x not in (c0, c1)]
            for s in rng.sample(range(N), 12):
                arr[s] = rng.choice(other)
            curated.append(arr)

        for idx, codes in enumerate(curated):
            a1, a2 = alleles[idx % len(alleles)]
            snps.append(("1", f"rs1_cur_{idx}", 1000 + idx * 250, a1, a2, codes))

        # Remaining structured LD + MAF spectrum across chr 1, 2, and 23 (X)
        chr_plan = [("1", 110 - len(curated)), ("2", 110), ("23", 50)]
        for chrom, count in chr_plan:
            cur_anchor: Optional[List[int]] = None
            for j in range(count):
                global_v = len(snps)
                a1, a2 = alleles[global_v % len(alleles)]
                mod = global_v % 9
                if chrom != "23" and cur_anchor is not None and mod in (2, 3):
                    codes = list(cur_anchor)
                    for s in rng.sample(range(N), rng.randint(1, 28)):
                        codes[s] = (3 if (allow_missing and s % 9 == 0) else (codes[s] + 1) % 3)
                elif chrom != "23" and cur_anchor is not None and mod == 4:
                    codes = [P2_INV[g] for g in cur_anchor]
                    for s in rng.sample(range(N), rng.randint(1, 24)):
                        codes[s] = (3 if (allow_missing and s % 8 == 0) else (codes[s] + 1) % 3)
                else:
                    p = rng.choice([0.0, 0.015, 0.04, 0.12, 0.28, 0.48, 0.75, 0.96])
                    mr = (
                        rng.choice([0.0, 0.0, 0.01, 0.04, 0.12])
                        if allow_missing
                        else 0.0
                    )
                    codes = []
                    for i in range(N):
                        if mr > 0.0 and rng.random() < mr:
                            codes.append(3)
                        elif chrom == "23" and samples[i][2] == 1:
                            codes.append(2 if rng.random() < p else 0)
                        else:
                            codes.append((rng.random() < p) + (rng.random() < p))
                    cur_anchor = list(codes)
                snps.append((chrom, f"rs{chrom}_{j}", 50000 + j * 300, a1, a2, codes))
        return snps

    # Also build a larger-N mini dataset (N2=600) so multi-group difflists (L > 64, raw_n//8=75) are exercised!
    N_big = 600
    samples_big = [(f"BF{i}", f"BI{i}", (i % 2) + 1) for i in range(N_big)]
    snps_big: List[Tuple[str, str, int, str, str, List[int]]] = []
    anchor_big = [(rng.random() < 0.35) + (rng.random() < 0.35) for _ in range(N_big)]
    snps_big.append(("1", "big_anchor", 1000, "A", "G", anchor_big))
    # LD child with L = 70 diffs (> 64 -> 2 difflist groups!)
    ld_big = list(anchor_big)
    for s in sorted(rng.sample(range(N_big), 70)):
        ld_big[s] = (ld_big[s] + 1) % 3
    snps_big.append(("1", "big_ld_2grp", 2000, "A", "G", ld_big))
    # Pure difflist vrtype 4 with L = 72 diffs (> 64 -> 2 difflist groups!)
    dl_big = [0] * N_big
    for s in sorted(rng.sample(range(N_big), 72)):
        dl_big[s] = 1 if (s & 1) else 2
    snps_big.append(("1", "big_vt4_2grp", 3000, "C", "T", dl_big))
    # 1-bit + 2-group difflist (L = 68 diffs!)
    ob_big = [0 if rng.random() < 0.6 else 1 for _ in range(N_big)]
    for s in sorted(rng.sample(range(N_big), 68)):
        ob_big[s] = 2 if (s % 3) else 3
    snps_big.append(("1", "big_vt1_2grp", 4000, "G", "A", ob_big))
    for k in range(28):
        p = 0.1 + 0.35 * rng.random()
        snps_big.append((
            "1",
            f"big_dense_{k}",
            5000 + k * 500,
            "A",
            "G",
            [(rng.random() < p) + (rng.random() < p) for _ in range(N_big)],
        ))

    write_bed_fileset(os.path.join(data_dir, "big600_bed"), samples_big, snps_big)
    vt_big, pl_big, _ = compress_snps_all_vrtypes(N_big, snps_big)
    pgen_big, _ = encode_var_pgen_0x10_or_0x20(N_big, vt_big, pl_big, external_pgi=False)
    with open(os.path.join(data_dir, "big600_p10.pgen"), "wb") as f:
        f.write(pgen_big)
    write_pvar(os.path.join(data_dir, "big600_p10.pvar"), snps_big)
    write_psam(os.path.join(data_dir, "big600_p10.psam"), samples_big)

    snps_miss = build_snps(allow_missing=True)
    snps_nomiss = build_snps(allow_missing=False)

    info: Dict[str, object] = {"N": N, "samples": samples}

    for tag, snps in (("miss", snps_miss), ("nomiss", snps_nomiss)):
        bed_pfx = os.path.join(data_dir, f"{tag}_bed")
        write_bed_fileset(bed_pfx, samples, snps)

        # Mode 0x02 fixed-width .pgen
        p02_pfx = os.path.join(data_dir, f"{tag}_p02")
        with open(p02_pfx + ".pgen", "wb") as f:
            f.write(encode_fixed_pgen_0x02(N, [s[5] for s in snps]))
        write_pvar(p02_pfx + ".pvar", snps)
        write_psam(p02_pfx + ".psam", samples)

        # Mode 0x10 variable-width .pgen (all vrtypes + phased 0x10)
        vts, pls, ld_children = compress_snps_all_vrtypes(N, snps)
        if tag == "miss":
            base_vts = {vt & 7 for vt in vts}
            phased_vts = {vt & 7 for vt in vts if (vt & 0x10) != 0}
            assert base_vts == {0, 1, 2, 3, 4, 5, 6, 7}, f"Missing base vrtypes: {base_vts}"
            assert len(phased_vts) >= 5, f"Too few phased vrtypes: {phased_vts}"
            info["ld_children_miss"] = ld_children

        p10_pfx = os.path.join(data_dir, f"{tag}_p10")
        p10_bytes, _ = encode_var_pgen_0x10_or_0x20(
            N, vts, pls, external_pgi=False, len_bytes=2, with_nonref_flags=True
        )
        with open(p10_pfx + ".pgen", "wb") as f:
            f.write(p10_bytes)
        write_pvar(p10_pfx + ".pvar", snps)
        write_psam(p10_pfx + ".psam", samples)

        # Mode 0x20 + external .pgen.pgi
        p20_pfx = os.path.join(data_dir, f"{tag}_p20")
        p20_bytes, pgi_bytes = encode_var_pgen_0x10_or_0x20(N, vts, pls, external_pgi=True)
        assert pgi_bytes is not None
        with open(p20_pfx + ".pgen", "wb") as f:
            f.write(p20_bytes)
        with open(p20_pfx + ".pgen.pgi", "wb") as f:
            f.write(pgi_bytes)
        write_pvar(p20_pfx + ".pvar", snps)
        write_psam(p20_pfx + ".psam", samples)

        # --bpfile (Mode 0x10 .pgen + .bim + .fam) AND Mode 0x01 (.bed bytes in .pgen + .bim + .fam)
        bp10_pfx = os.path.join(data_dir, f"{tag}_bp10")
        shutil.copyfile(p10_pfx + ".pgen", bp10_pfx + ".pgen")
        shutil.copyfile(bed_pfx + ".bim", bp10_pfx + ".bim")
        shutil.copyfile(bed_pfx + ".fam", bp10_pfx + ".fam")

        bp01_pfx = os.path.join(data_dir, f"{tag}_bp01")
        shutil.copyfile(bed_pfx + ".bed", bp01_pfx + ".pgen")
        shutil.copyfile(bed_pfx + ".bim", bp01_pfx + ".bim")
        shutil.copyfile(bed_pfx + ".fam", bp01_pfx + ".fam")

        # Multi-file split across 3 files (chr1, chr2, chr23) for --mbfile / --mpfile / --mbpfile
        mb_list, mp_list, mbp_list = [], [], []
        for c_label in ("1", "2", "23"):
            sub_snps = [s for s in snps if s[0] == c_label]
            mb_sub = os.path.join(data_dir, f"{tag}_chr{c_label}_bed")
            write_bed_fileset(mb_sub, samples, sub_snps)
            mb_list.append(mb_sub)

            sub_vt, sub_pl, _ = compress_snps_all_vrtypes(N, sub_snps)
            mp_sub = os.path.join(data_dir, f"{tag}_chr{c_label}_p10")
            sub_bytes, _ = encode_var_pgen_0x10_or_0x20(N, sub_vt, sub_pl, external_pgi=False)
            with open(mp_sub + ".pgen", "wb") as f:
                f.write(sub_bytes)
            write_pvar(mp_sub + ".pvar", sub_snps)
            write_psam(mp_sub + ".psam", samples)
            mp_list.append(mp_sub)

            mbp_sub = os.path.join(data_dir, f"{tag}_chr{c_label}_bp10")
            shutil.copyfile(mp_sub + ".pgen", mbp_sub + ".pgen")
            shutil.copyfile(mb_sub + ".bim", mbp_sub + ".bim")
            shutil.copyfile(mb_sub + ".fam", mbp_sub + ".fam")
            mbp_list.append(mbp_sub)

        with open(os.path.join(data_dir, f"{tag}.mbfile"), "w") as f:
            f.write("\n".join(mb_list) + "\n")
        with open(os.path.join(data_dir, f"{tag}.mpfile"), "w") as f:
            f.write("\n".join(mp_list) + "\n")
        with open(os.path.join(data_dir, f"{tag}.mbpfile"), "w") as f:
            f.write("\n".join(mbp_list) + "\n")

    # Phenotypes, weights, covariates, --keep, --remove, --extract (dropping LD anchors!), --update-freq
    phen_path = os.path.join(data_dir, "trait.phen")
    wphen_path = os.path.join(data_dir, "trait_w.phen")
    qcov_path = os.path.join(data_dir, "trait.qcovar")
    with open(phen_path, "w") as fp, open(wphen_path, "w") as fwp, open(qcov_path, "w") as fq:
        for i, (fid, iid, _) in enumerate(samples):
            # Add genetic signal from first few SNPs so REML h2 is well away from 0/1 boundary
            g_sig = sum(
                (snps_nomiss[k][5][i] - 1.0) * (0.25 if k % 2 == 0 else -0.2)
                for k in range(0, 40, 2)
            )
            cov1 = rng.gauss(0.0, 1.0)
            y = 0.4 * cov1 + g_sig + rng.gauss(0.0, 1.2)
            w = rng.uniform(0.6, 1.8)
            fp.write(f"{fid}\t{iid}\t{y:.6f}\n")
            fwp.write(f"{fid}\t{iid}\t{y:.6f}\t{w:.4f}\n")
            fq.write(f"{fid}\t{iid}\t{cov1:.6f}\n")

    keep_idx = sorted(rng.sample(range(N), 199))
    with open(os.path.join(data_dir, "keep199.txt"), "w") as f:
        for i in keep_idx:
            f.write(f"{samples[i][0]}\t{samples[i][1]}\n")

    rem_idx = sorted(rng.sample(range(N), 43))
    with open(os.path.join(data_dir, "remove43.txt"), "w") as f:
        for i in rem_idx:
            f.write(f"{samples[i][0]}\t{samples[i][1]}\n")

    # --extract list that keeps ONLY LD-compressed children (dropping ALL their ldbase anchors!)
    # plus scattered non-contiguous variants across chr 1, 2, 23
    ld_child_set = set(info["ld_children_miss"])  # type: ignore[arg-type]
    assert len(ld_child_set) >= 25
    extract_names: List[str] = []
    for idx, s in enumerate(snps_miss):
        if idx in ld_child_set or (idx % 17 == 11 and (idx + 1) not in ld_child_set):
            # Explicitly exclude any SNP immediately preceding an LD child!
            if (idx + 1) not in ld_child_set and (idx + 2) not in ld_child_set:
                extract_names.append(s[1])
            elif idx in ld_child_set:
                extract_names.append(s[1])
    with open(os.path.join(data_dir, "extract_drop_anchors.txt"), "w") as f:
        f.write("\n".join(extract_names) + "\n")

    # --update-freq file (SNP A1 AF)
    with open(os.path.join(data_dir, "update_freq.txt"), "w") as f:
        for chrom, rsid, _bp, a1, _a2, codes in snps_miss:
            if chrom == "23":
                continue
            obs = [g for g in codes if g != 3]
            af = (sum(obs) / (2.0 * len(obs))) if obs else 0.25
            af = min(0.95, max(0.05, af))
            f.write(f"{rsid}\t{a1}\t{af:.6f}\n")

    info["snps_miss"] = snps_miss
    info["snps_nomiss"] = snps_nomiss
    return info


# -----------------------------------------------------------------------------
# Dosage & Multiallelic .pgen fallback builders
# -----------------------------------------------------------------------------
def build_dosage_datasets(
    data_dir: str,
    samples: Sequence[Tuple[str, str, int]],
    snps_miss: Sequence[Tuple[str, str, int, str, str, List[int]]],
) -> None:
    """Create:
      1) exact_dos_0x03 / exact_dos_0x10: dosage .pgen files storing the exact
         same hardcall genotypes {0, 16384, 32768, 65535} via mode 0x03 (fixed
         dosage) and mode 0x10 (mixed vrtype 0x00 hardcall, 0x60 sparse dosage,
         and 0x40 unconditional dosage) -> must match --bfile numerically!
      2) frac_dos_0x10: fractional dosages (0..32768 + 65535 missing) across
         sparse (0 < dosage_ct < N), dense (dosage_ct == N), and hardcall
         (dosage_ct == 0) records -> verifies sparse dosage expansion + AF accuracy!
    """
    N = len(samples)
    g2dos = {0: 0, 1: 16384, 2: 32768, 3: 65535}
    auto_snps = [s for s in snps_miss if s[0] != "23"][:80]

    # Reference BED for those 80 autosomal SNPs
    write_bed_fileset(os.path.join(data_dir, "dos_ref_bed"), samples, auto_snps)

    # 1a. Mode 0x03 fixed-width unconditional dosage
    M = len(auto_snps)
    p03 = bytearray([0x6C, 0x1B, 0x03])
    p03.extend(struct.pack("<IIB", M, N, 0))
    for *_, codes in auto_snps:
        p03.extend(pack_p2_2bit(codes))
        for c in codes:
            p03.extend(struct.pack("<H", g2dos[c]))
    p03_pfx = os.path.join(data_dir, "exact_dos_p03")
    with open(p03_pfx + ".pgen", "wb") as f:
        f.write(p03)
    write_pvar(p03_pfx + ".pvar", auto_snps)
    write_psam(p03_pfx + ".psam", samples)

    # 1b. Mode 0x10 mixed hardcall (0x00) + sparse bitarray dosage (0x60) + unconditional dosage (0x40)
    vts_ex: List[int] = []
    pls_ex: List[bytes] = []
    present_bytes_len = (N + 7) // 8
    for v, (*_, codes) in enumerate(auto_snps):
        if v % 3 == 0:
            # Pure hardcall record inside a dosage .pgen (dosage_ct == 0)
            vts_ex.append(0x00)
            pls_ex.append(pack_p2_2bit(codes))
        elif v % 3 == 1:
            # Sparse bitarray dosage (vrtype 0x60): put ~20% of non-missing calls into dosage_main
            vts_ex.append(0x60)
            pl = bytearray(pack_p2_2bit(codes))
            mask = bytearray(present_bytes_len)
            dvals = bytearray()
            for s, c in enumerate(codes):
                if c != 3 and (s % 5 == 2):
                    mask[s >> 3] |= 1 << (s & 7)
                    dvals.extend(struct.pack("<H", g2dos[c]))
            pl.extend(mask)
            pl.extend(dvals)
            pls_ex.append(bytes(pl))
        else:
            # Unconditional dosage (vrtype 0x40, dosage_ct == N, including 0xFFFF missing!)
            vts_ex.append(0x40)
            pl = bytearray(pack_p2_2bit(codes))
            for c in codes:
                pl.extend(struct.pack("<H", g2dos[c]))
            pls_ex.append(bytes(pl))

    p10_ex, _ = encode_var_pgen_0x10_or_0x20(N, vts_ex, pls_ex, external_pgi=False, len_bytes=4)
    p10_ex_pfx = os.path.join(data_dir, "exact_dos_p10")
    with open(p10_ex_pfx + ".pgen", "wb") as f:
        f.write(p10_ex)
    write_pvar(p10_ex_pfx + ".pvar", auto_snps)
    write_psam(p10_ex_pfx + ".psam", samples)

    # 1c. Mode 0x10 pure sparse bitarray dosage (vrtype 0x60, 0 < dosage_ct < N on every variant)
    # -> must match Mode 0x03 dense dosage GRM & MLMA byte-for-byte!
    vts_sp: List[int] = []
    pls_sp: List[bytes] = []
    for *_, codes in auto_snps:
        vts_sp.append(0x60)
        pl = bytearray(pack_p2_2bit(codes))
        mask = bytearray(present_bytes_len)
        dvals = bytearray()
        placed = 0
        for s, c in enumerate(codes):
            if c != 3 and (s % 4 == 1 or placed == 0):
                mask[s >> 3] |= 1 << (s & 7)
                dvals.extend(struct.pack("<H", g2dos[c]))
                placed += 1
        pl.extend(mask)
        pl.extend(dvals)
        pls_sp.append(bytes(pl))
    p10_sp, _ = encode_var_pgen_0x10_or_0x20(N, vts_sp, pls_sp, external_pgi=False, len_bytes=4)
    p10_sp_pfx = os.path.join(data_dir, "exact_dos_sparse_p10")
    with open(p10_sp_pfx + ".pgen", "wb") as f:
        f.write(p10_sp)
    write_pvar(p10_sp_pfx + ".pvar", auto_snps)
    write_psam(p10_sp_pfx + ".psam", samples)

    # 2. Fractional dosage dataset (mode 0x10) with known analytical allele frequencies
    rng = random.Random(998877)
    frac_snps_meta: List[Tuple[str, str, int, str, str, List[int]]] = []
    vts_fr: List[int] = []
    pls_fr: List[bytes] = []
    expected_afs: List[float] = []

    for v in range(36):
        hc: List[int] = []
        dos_u16: List[int] = []
        mask = bytearray(present_bytes_len)
        sparse_mode = (v % 2 == 0)
        p = 0.15 + 0.65 * (v / 36.0)
        for s in range(N):
            if s % 53 == (v % 11):
                hc.append(3)
                dos_u16.append(65535)
                continue
            if sparse_mode and (s % 4 != 1):
                g = (rng.random() < p) + (rng.random() < p)
                hc.append(g)
                dos_u16.append(g * 16384)
            else:
                d_float = min(2.0, max(0.0, rng.gauss(2.0 * p, 0.35)))
                u16 = int(round(d_float * 16384.0))
                u16 = min(32768, max(0, u16))
                # Hardcall rounded (or 3 if far from integer)
                hc_val = int(round(u16 / 16384.0))
                hc.append(hc_val)
                dos_u16.append(u16)
                if sparse_mode:
                    mask[s >> 3] |= 1 << (s & 7)

        obs_u16 = [d for d in dos_u16 if d != 65535]
        expected_afs.append((sum(obs_u16) / 16384.0) / (2.0 * len(obs_u16)))
        frac_snps_meta.append(("1", f"frac_snp_{v}", 1000 + v * 400, "A", "G", hc))

        if sparse_mode:
            vts_fr.append(0x60)
            pl = bytearray(pack_p2_2bit(hc))
            pl.extend(mask)
            for s in range(N):
                if (mask[s >> 3] >> (s & 7)) & 1:
                    pl.extend(struct.pack("<H", dos_u16[s]))
            pls_fr.append(bytes(pl))
        else:
            vts_fr.append(0x40)
            pl = bytearray(pack_p2_2bit(hc))
            for u16 in dos_u16:
                pl.extend(struct.pack("<H", u16))
            pls_fr.append(bytes(pl))

    p_frac, _ = encode_var_pgen_0x10_or_0x20(N, vts_fr, pls_fr, external_pgi=False, len_bytes=4)
    frac_pfx = os.path.join(data_dir, "frac_dos_p10")
    with open(frac_pfx + ".pgen", "wb") as f:
        f.write(p_frac)
    write_pvar(frac_pfx + ".pvar", frac_snps_meta)
    write_psam(frac_pfx + ".psam", samples)
    with open(frac_pfx + ".expected_af", "w") as f:
        for v, af in enumerate(expected_afs):
            f.write(f"frac_snp_{v}\t{af:.10f}\n")


# -----------------------------------------------------------------------------
# Comparison helpers
# -----------------------------------------------------------------------------
def collect_grm_exts(prefix: str) -> List[str]:
    parent = os.path.dirname(prefix)
    base = os.path.basename(prefix)
    return sorted(
        fn[len(base):]
        for fn in os.listdir(parent)
        if fn.startswith(base + ".") and ".grm." in fn[len(base):]
    )


F16_HDR_FMT = "<8sIIIIQQdQddQQQQd16s"
F16_HDR_SIZE = struct.calcsize(F16_HDR_FMT)


def assert_f16_identical(rb: bytes, cb: bytes, label: str) -> None:
    assert len(rb) == len(cb) and len(rb) >= F16_HDR_SIZE, (
        f"{label} (.grm.f16): size mismatch {len(rb)} vs {len(cb)}"
    )
    rh = struct.unpack(F16_HDR_FMT, rb[:F16_HDR_SIZE])
    ch = struct.unpack(F16_HDR_FMT, cb[:F16_HDR_SIZE])
    assert rh[0] == b"GCTAG16\x00" and ch[0] == b"GCTAG16\x00"
    # Exact integer/offset/flag fields: magic(0)..valid_m(6), id_hash(8), diag_off(11)..re_row(14), pad(16)
    for idx in (0, 1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 16):
        assert rh[idx] == ch[idx], f"{label} (.grm.f16) hdr[{idx}]: {rh[idx]!r} vs {ch[idx]!r}"
    # Double scalars (m_mean=7, tr_k=9, tr_k2=10, mtd_w=15; tr_k/tr_k2 use dynamic OMP reduction)
    for idx in (7, 9, 10, 15):
        u, v = rh[idx], ch[idx]
        assert abs(u - v) <= 1e-12 * max(1.0, abs(u)), (
            f"{label} (.grm.f16) hdr[{idx}]: {u!r} vs {v!r}"
        )
    doff, boff = rh[11], rh[12]
    assert rb[doff:boff] == cb[doff:boff], f"{label} (.grm.f16): float32 diagonal bytes differ"
    assert rb[boff:] == cb[boff:], f"{label} (.grm.f16): packed fp16 body bytes differ"


def assert_grm_byte_identical(ref_pfx: str, cand_pfx: str, label: str) -> None:
    ref_exts = collect_grm_exts(ref_pfx)
    cand_exts = collect_grm_exts(cand_pfx)
    assert ref_exts and ref_exts == cand_exts, (
        f"{label}: GRM extensions differ: ref={ref_exts} vs cand={cand_exts}"
    )
    for ext in ref_exts:
        rb = open(ref_pfx + ext, "rb").read()
        cb = open(cand_pfx + ext, "rb").read()
        if ext == ".grm.f16":
            assert_f16_identical(rb, cb, label)
            continue
        if rb != cb:
            if ext.endswith(".bin") and len(rb) == len(cb):
                fa = array.array("f")
                fb = array.array("f")
                fa.frombytes(rb)
                fb.frombytes(cb)
                nd = sum(1 for x, y in zip(fa, fb) if x != y)
                md = max(abs(x - y) for x, y in zip(fa, fb))
                raise AssertionError(
                    f"{label} ({ext}): {nd}/{len(fa)} floats differ, max|d|={md:.3e}"
                )
            raise AssertionError(
                f"{label} ({ext}): byte mismatch (len {len(rb)} vs {len(cb)})"
            )


def assert_grm_close(ref_pfx: str, cand_pfx: str, label: str, atol: float = 2e-6) -> None:
    for ext in (".grm.id", ".grm.N.bin"):
        assert open(ref_pfx + ext, "rb").read() == open(cand_pfx + ext, "rb").read(), (
            f"{label}: {ext} mismatch"
        )
    fa = array.array("f")
    fb = array.array("f")
    fa.frombytes(open(ref_pfx + ".grm.bin", "rb").read())
    fb.frombytes(open(cand_pfx + ".grm.bin", "rb").read())
    assert len(fa) == len(fb)
    md = max(abs(x - y) for x, y in zip(fa, fb))
    assert md <= atol, f"{label}: .grm.bin max|d|={md:.3e} > {atol:.3e}"


def assert_file_byte_identical(ref_path: str, cand_path: str, label: str) -> None:
    rb = open(ref_path, "rb").read()
    cb = open(cand_path, "rb").read()
    if rb != cb:
        ra = rb.decode("utf-8", errors="replace").splitlines()
        ca = cb.decode("utf-8", errors="replace").splitlines()
        first_diff = next(
            (i for i, (x, y) in enumerate(zip(ra, ca)) if x != y),
            min(len(ra), len(ca)),
        )
        raise AssertionError(
            f"{label}: byte mismatch at line {first_diff + 1} "
            f"(lines {len(ra)} vs {len(ca)}):\n"
            f"  ref : {ra[first_diff] if first_diff < len(ra) else '<EOF>'}\n"
            f"  cand: {ca[first_diff] if first_diff < len(ca) else '<EOF>'}"
        )


def assert_mlma_close(ref_path: str, cand_path: str, label: str, atol: float = 2e-5) -> None:
    rl = open(ref_path).read().splitlines()
    cl = open(cand_path).read().splitlines()
    assert len(rl) == len(cl) and rl[0] == cl[0], f"{label}: header/row count mismatch"
    for idx, (r, c) in enumerate(zip(rl[1:], cl[1:]), start=2):
        rc = r.split("\t")
        cc = c.split("\t")
        assert rc[:5] == cc[:5], f"{label} line {idx}: meta mismatch {rc[:5]} vs {cc[:5]}"
        for k in range(5, 9):
            if rc[k] == "NA" or cc[k] == "NA":
                assert rc[k] == cc[k], f"{label} line {idx}: NA mismatch {rc} vs {cc}"
                continue
            u, v = float(rc[k]), float(cc[k])
            lim = max(atol, 1e-4 * max(abs(u), abs(v)))
            assert abs(u - v) <= lim, f"{label} line {idx} col {k}: {u} vs {v}"


# -----------------------------------------------------------------------------
# Test suites
# -----------------------------------------------------------------------------
def run_grm_parity_suite(data_dir: str, work_dir: str) -> str:
    D = lambda fn: os.path.join(data_dir, fn)
    W = lambda fn: os.path.join(work_dir, fn)
    T = ["--threads", "4"]

    # 1. Multi-group difflist (N=600, L > 64) bit-for-bit check
    run_cmd([GCTAG_BIN, "grm", "make", "--bfile", D("big600_bed"), "--alg", "1", *T, "--out", W("big_bed")])
    run_cmd([GCTAG_BIN, "grm", "make", "--pfile", D("big600_p10"), "--alg", "1", *T, "--out", W("big_p10")])
    assert_grm_byte_identical(W("big_bed"), W("big_p10"), "grm_big600_multigroup_difflist")
    ok("grm make N=600 multi-group difflist (L>64) bit-for-bit")

    # 2. Core missing-data GRM across ALL input flags & PGEN modes (0x02, 0x10, 0x20, --bpfile 0x10/0x01, --mpfile, --mbpfile)
    for alg in ("0", "1"):
        ref_single = W(f"g_miss_alg{alg}_bed")
        ref_multi = W(f"g_miss_alg{alg}_mbfile")
        run_cmd([GCTAG_BIN, "grm", "make", "--bfile", D("miss_bed"), "--alg", alg, *T, "--out", ref_single])
        run_cmd([GCTAG_BIN, "grm", "make", "--mbfile", D("miss.mbfile"), "--alg", alg, *T, "--out", ref_multi])
        if alg == "1":
            # INT8 int32 accumulation is associative across file boundaries!
            assert_grm_byte_identical(ref_single, ref_multi, "grm_miss_alg1_bfile_vs_mbfile")
        else:
            assert_grm_close(ref_single, ref_multi, "grm_miss_alg0_bfile_vs_mbfile", atol=2e-6)

        single_variants = [
            ("p02", ["--pfile", D("miss_p02")]),
            ("p10", ["--pfile", D("miss_p10")]),
            ("p20_pgi", ["--pfile", D("miss_p20")]),
            ("bp10", ["--bpfile", D("miss_bp10")]),
            ("bp01", ["--bpfile", D("miss_bp01")]),
        ]
        for vname, g_args in single_variants:
            cand = W(f"g_miss_alg{alg}_{vname}")
            run_cmd([GCTAG_BIN, "grm", "make", *g_args, "--alg", alg, *T, "--out", cand])
            assert_grm_byte_identical(ref_single, cand, f"grm_miss_alg{alg}_{vname}")
            ok(f"grm make --alg {alg} (with missing) bit-for-bit vs --bfile: {vname}")

        multi_variants = [
            ("mpfile", ["--mpfile", D("miss.mpfile")]),
            ("mbpfile", ["--mbpfile", D("miss.mbpfile")]),
        ]
        for vname, g_args in multi_variants:
            cand = W(f"g_miss_alg{alg}_{vname}")
            run_cmd([GCTAG_BIN, "grm", "make", *g_args, "--alg", alg, *T, "--out", cand])
            assert_grm_byte_identical(ref_multi, cand, f"grm_miss_alg{alg}_{vname}_vs_mbfile")
            if alg == "1":
                assert_grm_byte_identical(ref_single, cand, f"grm_miss_alg1_{vname}_vs_bfile")
            ok(f"grm make --alg {alg} (with missing) bit-for-bit vs --mbfile: {vname}")

        # Fallback GCTA_DISABLE_NATIVE_PGEN=1:
        # On --alg 0 both use FP32 sgemm -> byte-identical to ref_single.
        # On --alg 1 GCTA_DISABLE_NATIVE_PGEN=1 disables INT8 GRM and uses FP32 sgemm ->
        # byte-identical to --bfile with GCTA_DISABLE_INT8_GRM=1, and <=2e-6 vs INT8!
        cand_fb = W(f"g_miss_alg{alg}_disable_native")
        run_cmd(
            [GCTAG_BIN, "grm", "make", "--pfile", D("miss_p10"), "--alg", alg, *T, "--out", cand_fb],
            env_extra={"GCTA_DISABLE_NATIVE_PGEN": "1"},
        )
        if alg == "0":
            assert_grm_byte_identical(ref_single, cand_fb, f"grm_miss_alg{alg}_disable_native")
        else:
            ref_bed_f32 = W("g_miss_alg1_bed_f32")
            run_cmd(
                [GCTAG_BIN, "grm", "make", "--bfile", D("miss_bed"), "--alg", "1", *T, "--out", ref_bed_f32],
                env_extra={"GCTA_DISABLE_INT8_GRM": "1"},
            )
            assert_grm_byte_identical(ref_bed_f32, cand_fb, "grm_miss_alg1_disable_native_vs_bed_f32")
            assert_grm_close(ref_single, cand_fb, "grm_miss_alg1_disable_native_vs_int8", atol=2e-6)
        ok(f"grm make --alg {alg} GCTA_DISABLE_NATIVE_PGEN=1 fallback parity")

    # 3. Zero-missing cohort + --grm-fp16 / --fast (.grm.N.bin omitted on zero-missing!)
    for tag in ("nomiss", "miss"):
        ref_f16 = W(f"g_{tag}_fast_bed")
        run_cmd([GCTAG_BIN, "grm", "make", "--bfile", D(f"{tag}_bed"), "--fast", *T, "--out", ref_f16])
        if tag == "nomiss":
            assert not os.path.exists(ref_f16 + ".grm.N.bin")
        else:
            assert os.path.exists(ref_f16 + ".grm.N.bin")

        for vname, g_args in (
            ("p02", ["--pfile", D(f"{tag}_p02")]),
            ("p10", ["--pfile", D(f"{tag}_p10")]),
            ("bp10", ["--bpfile", D(f"{tag}_bp10")]),
            ("mpfile", ["--mpfile", D(f"{tag}.mpfile")]),
        ):
            cand_f16 = W(f"g_{tag}_fast_{vname}")
            run_cmd([GCTAG_BIN, "grm", "make", *g_args, "--fast", *T, "--out", cand_f16])
            assert_grm_byte_identical(ref_f16, cand_f16, f"grm_{tag}_fast_{vname}")
        ok(f"grm make --fast (--grm-fp16) [{tag}] bit-for-bit across p02/p10/bp10/mpfile")

    # 4. --impute-mean (dense .grm.bin, --fast .grm.f16, --keep, and --extract dropping LD anchors)
    for extra_tag, extra_args in (
        ("full", ["--alg", "1", "--impute-mean"]),
        ("fast", ["--fast", "--impute-mean"]),
        ("keep_extract", ["--alg", "1", "--impute-mean", "--keep", D("keep199.txt"), "--extract", D("extract_drop_anchors.txt")]),
    ):
        ref_imp = W(f"g_imp_{extra_tag}_bed")
        run_cmd([GCTAG_BIN, "grm", "make", "--bfile", D("miss_bed"), *extra_args, *T, "--out", ref_imp])
        for vname, g_args in (
            ("p10", ["--pfile", D("miss_p10")]),
            ("bp10", ["--bpfile", D("miss_bp10")]),
            ("mpfile", ["--mpfile", D("miss.mpfile")]),
        ):
            cand_imp = W(f"g_imp_{extra_tag}_{vname}")
            run_cmd([GCTAG_BIN, "grm", "make", *g_args, *extra_args, *T, "--out", cand_imp])
            assert_grm_byte_identical(ref_imp, cand_imp, f"grm_imp_{extra_tag}_{vname}")
        ok(f"grm make --impute-mean [{extra_tag}] bit-for-bit")

    # 5. --grm-profile compressed I/O & unpack timing verification
    _, prof_log = run_cmd([
        GCTAG_BIN, "grm", "make", "--pfile", D("miss_p10"), "--alg", "1",
        "--grm-profile", *T, "--out", W("g_prof_p10"),
    ])
    assert re.search(
        r"GRM profile \(s\): init=\d+\.\d+ io_wait=\d+\.\d+ count\+decode=\d+\.\d+ "
        r"miss=\d+\.\d+ gemm=\d+\.\d+ flush=\d+\.\d+ unpack=\d+\.\d+",
        prof_log,
    ), f"Missing unpack= in --grm-profile log:\n{prof_log}"
    assert re.search(
        r"GRM PGEN I/O: compressed=\d+\.\d+ MiB vs raw_bed_equiv=\d+\.\d+ MiB \(\d+\.\d+x\)",
        prof_log,
    ), f"Missing GRM PGEN I/O summary in --grm-profile log:\n{prof_log}"
    assert_grm_byte_identical(W("g_miss_alg1_bed"), W("g_prof_p10"), "grm_profile_p10")
    ok("grm make --grm-profile logs unpack + compressed I/O ratio & matches BED bit-for-bit")

    # 6. Multi-tile --GRM-tile-budget (0.00005 GB = ~53 KB -> multiple tiles on N=317)
    # Tests zero-missing int32 tiles, missing-promoted double+N tiles, --impute-mean tiles,
    # and Tile 2+ external ldbase anchor fetching under --extract!
    tile_cases = [
        ("nomiss_tiled", "nomiss", ["--alg", "1", "--GRM-tile-budget", "0.00005", "--n-markers", "64"]),
        ("miss_tiled", "miss", ["--alg", "1", "--GRM-tile-budget", "0.00005", "--n-markers", "64"]),
        ("miss_tiled_f16", "miss", ["--fast", "--GRM-tile-budget", "0.00005", "--n-markers", "64"]),
        ("miss_tiled_imp", "miss", ["--alg", "1", "--impute-mean", "--GRM-tile-budget", "0.00005", "--n-markers", "64"]),
        (
            "miss_tiled_drop_ldbase",
            "miss",
            [
                "--alg", "1", "--GRM-tile-budget", "0.00005", "--n-markers", "32",
                "--keep", D("keep199.txt"), "--extract", D("extract_drop_anchors.txt"),
                "--maf", "0.03",
            ],
        ),
    ]
    for cname, tag, cargs in tile_cases:
        rb = W(f"gt_{cname}_bed")
        cp = W(f"gt_{cname}_p10")
        cm = W(f"gt_{cname}_mp")
        _, log_b = run_cmd([GCTAG_BIN, "grm", "make", "--bfile", D(f"{tag}_bed"), *cargs, *T, "--out", rb])
        _, log_p = run_cmd([GCTAG_BIN, "grm", "make", "--pfile", D(f"{tag}_p10"), *cargs, *T, "--out", cp])
        run_cmd([GCTAG_BIN, "grm", "make", "--mpfile", D(f"{tag}.mpfile"), *cargs, *T, "--out", cm])
        assert "Block-tiled GRM" in log_b and "Block-tiled GRM" in log_p, (
            f"Expected multi-tile execution for {cname}:\n{log_p}"
        )
        assert_grm_byte_identical(rb, cp, f"tile_{cname}_p10")
        assert_grm_byte_identical(rb, cm, f"tile_{cname}_mp")
        ok(f"grm make multi-tile ({cname}) bit-for-bit across bfile/pfile/mpfile")

    # 7. Filters & special GRM modes: --dominance, --homogametic, --sparse-cutoff, --part,
    # --grm-legacy-engine, --update-freq
    special_cases = [
        ("dom", ["--dominance", "--keep", D("keep199.txt")]),
        ("hom_x", ["--homogametic", "--homogametic-chr", "23", "--dc", "0"]),
        ("sparse", ["--alg", "1", "--sparse-cutoff", "0.05"]),
        ("part2_2", ["--alg", "1", "--part", "2", "2"]),
        ("legacy_eng", ["--alg", "1", "--grm-legacy-engine", "--extract", D("extract_drop_anchors.txt")]),
        ("legacy_tiled", ["--alg", "1", "--grm-legacy-engine", "--GRM-tile-budget", "0.00005", "--n-markers", "64"]),
        ("update_freq", ["--alg", "1", "--update-freq", D("update_freq.txt")]),
        ("maf_geno_rm", ["--alg", "1", "--maf", "0.05", "--geno", "0.10", "--remove", D("remove43.txt")]),
    ]
    for sname, sargs in special_cases:
        sb_out = W(f"gs_{sname}_bed")
        sp_out = W(f"gs_{sname}_p10")
        run_cmd([GCTAG_BIN, "grm", "make", "--bfile", D("miss_bed"), *sargs, *T, "--out", sb_out])
        run_cmd([GCTAG_BIN, "grm", "make", "--pfile", D("miss_p10"), *sargs, *T, "--out", sp_out])
        suf = ".part_2_2" if sname == "part2_2" else ""
        assert_grm_byte_identical(sb_out + suf, sp_out + suf, f"special_grm_{sname}")
        ok(f"grm make ({sname}) bit-for-bit")

    return W("g_miss_alg0_bed")


def run_mlma_reml_pca_legacy_suite(data_dir: str, work_dir: str, base_grm: str) -> None:
    D = lambda fn: os.path.join(data_dir, fn)
    W = lambda fn: os.path.join(work_dir, fn)
    T = ["--threads", "4"]

    # 1. Pre-fit REML states (TUN3 Woodbury + TREF reuse) using --pfile vs --bfile manifests
    run_cmd([
        GCTAG_BIN, "reml", "--bfile", D("miss_bed"), "--grm", base_grm,
        "--pheno", D("trait.phen"), "--woodbury-basis", "40", "--seed", "7",
        *T, "--out", W("reml_tun3_bed"),
    ])
    run_cmd([
        GCTAG_BIN, "reml", "--pfile", D("miss_p10"), "--grm", base_grm,
        "--pheno", D("trait.phen"), "--woodbury-basis", "40", "--seed", "7",
        *T, "--out", W("reml_tun3_p10"),
    ])
    assert open(W("reml_tun3_p10.reml"), "rb").read(4) == b"TUN3"
    run_cmd([
        GCTAG_BIN, "reml", "--pfile", D("miss_p10"),
        "--woodbury-reuse", W("reml_tun3_p10.reml"),
        "--pheno", D("trait.phen"), *T, "--out", W("reml_tref_p10"),
    ])
    assert open(W("reml_tref_p10.reml"), "rb").read(4) == b"TREF"
    ok("reml TUN3 & TREF state generation with --pfile manifest")

    # 2. INT8 Woodbury MLMA (wb_std_i8) across ALL PGEN formats + fallback env
    ref_wb = W("m_wb_bed")
    ref_wb_mb = W("m_wb_mbfile")
    wb_common = ["--load-reml", W("reml_tun3_bed.reml"), "--pheno", D("trait.phen")]
    run_cmd([GCTAG_BIN, "mlma", "--bfile", D("miss_bed"), *wb_common, *T, "--out", ref_wb])
    run_cmd([GCTAG_BIN, "mlma", "--mbfile", D("miss.mbfile"), *wb_common, *T, "--out", ref_wb_mb])
    assert_mlma_close(ref_wb + ".mlma", ref_wb_mb + ".mlma", "mlma_wb_bfile_vs_mbfile", atol=1e-6)

    for vname, g_args in (
        ("p02", ["--pfile", D("miss_p02")]),
        ("p10", ["--pfile", D("miss_p10")]),
        ("p20_pgi", ["--pfile", D("miss_p20")]),
        ("bp10", ["--bpfile", D("miss_bp10")]),
        ("bp01", ["--bpfile", D("miss_bp01")]),
    ):
        cand = W(f"m_wb_{vname}")
        run_cmd([GCTAG_BIN, "mlma", *g_args, *wb_common, *T, "--out", cand])
        assert_file_byte_identical(ref_wb + ".mlma", cand + ".mlma", f"mlma_wb_{vname}")
        ok(f"mlma INT8 Woodbury (wb_std_i8) bit-for-bit vs --bfile: {vname}")

    for vname, g_args in (
        ("mpfile", ["--mpfile", D("miss.mpfile")]),
        ("mbpfile", ["--mbpfile", D("miss.mbpfile")]),
    ):
        cand = W(f"m_wb_{vname}")
        run_cmd([GCTAG_BIN, "mlma", *g_args, *wb_common, *T, "--out", cand])
        assert_file_byte_identical(ref_wb_mb + ".mlma", cand + ".mlma", f"mlma_wb_{vname}")
        ok(f"mlma INT8 Woodbury (wb_std_i8) bit-for-bit vs --mbfile: {vname}")

    # GCTA_DISABLE_NATIVE_PGEN=1 fallback on MLMA (exercises fallback FP32 Woodbury; compare close + exact cols 0..5)
    cand_dis = W("m_wb_disable_native")
    run_cmd(
        [GCTAG_BIN, "mlma", "--pfile", D("miss_p10"), *wb_common, *T, "--out", cand_dis],
        env_extra={"GCTA_DISABLE_NATIVE_PGEN": "1"},
    )
    assert_mlma_close(ref_wb + ".mlma", cand_dis + ".mlma", "mlma_wb_disable_native", atol=5e-6)
    ok("mlma Woodbury GCTA_DISABLE_NATIVE_PGEN=1 fallback parity")

    # 3. Quantized --fast-mlma modes (adaptive, 2slice, hybrid, 1slice), TREF load-reml,
    # dense FP32 path, dominance/nonadditive, --keep, --extract dropping LD anchors,
    # --maf multi-pass, --weighted-phenotypes, --qcovar, --autosome-sex
    mlma_matrix = [
        ("fast_adaptive", [*wb_common, "--fast-mlma", "adaptive"]),
        ("fast_2slice", [*wb_common, "--fast-mlma", "2slice", "--mlma-block-size", "64"]),
        ("fast_hybrid", [*wb_common, "--fast-mlma", "hybrid", "--mlma-block-size", "64"]),
        ("fast_1slice", [*wb_common, "--fast-mlma", "1slice", "--mlma-block-size", "64"]),
        ("tref_load", ["--load-reml", W("reml_tref_p10.reml"), "--pheno", D("trait.phen"), "--fast-mlma", "2slice"]),
        ("dense_f32", ["--grm", base_grm, "--pheno", D("trait.phen"), "--qcovar", D("trait.qcovar")]),
        (
            "dense_keep_extract_drop_ldbase",
            [
                "--grm", base_grm, "--pheno", D("trait.phen"),
                "--keep", D("keep199.txt"), "--extract", D("extract_drop_anchors.txt"),
            ],
        ),
        (
            "wb_keep_extract_maf_geno",
            [
                "--grm", base_grm, "--pheno", D("trait.phen"),
                "--woodbury-basis", "35", "--seed", "11",
                "--keep", D("keep199.txt"), "--extract", D("extract_drop_anchors.txt"),
                "--maf", "0.05", "--geno", "0.15", "--mlma-block-size", "32",
            ],
        ),
        ("weighted_pheno_wb", ["--grm", base_grm, "--pheno", D("trait_w.phen"), "--weighted-phenotypes", "--woodbury-basis", "40", "--seed", "7"]),
        ("dom_model", [*wb_common, "--model", "dominance"]),
        ("nonadd_model", ["--grm", base_grm, "--pheno", D("trait.phen"), "--model", "nonadditive"]),
        ("sex_chr_x", [*wb_common, "--autosome-sex", "--homogametic-chr", "23"]),
    ]
    for mname, margs in mlma_matrix:
        mb_out = W(f"mm_{mname}_bed")
        mp_out = W(f"mm_{mname}_p10")
        run_cmd([GCTAG_BIN, "mlma", "--bfile", D("miss_bed"), *margs, *T, "--out", mb_out])
        run_cmd([GCTAG_BIN, "mlma", "--pfile", D("miss_p10"), *margs, *T, "--out", mp_out])
        assert_file_byte_identical(mb_out + ".mlma", mp_out + ".mlma", f"mlma_{mname}")
        ok(f"mlma ({mname}) bit-for-bit")

    # Zero-missing MLMA + --stream-budget compact PGEN tile-size check
    _, log_native_budget = run_cmd([
        GCTAG_BIN, "mlma", "--pfile", D("nomiss_p10"),
        "--load-reml", W("reml_tun3_p10.reml"), "--pheno", D("trait.phen"),
        "--stream-budget", "0.0008", *T, "--out", W("mm_budget_p10"),
    ])
    _, log_fb_budget = run_cmd(
        [
            GCTAG_BIN, "mlma", "--pfile", D("nomiss_p10"),
            "--load-reml", W("reml_tun3_p10.reml"), "--pheno", D("trait.phen"),
            "--stream-budget", "0.0008", *T, "--out", W("mm_budget_p10_fb"),
        ],
        env_extra={"GCTA_DISABLE_NATIVE_PGEN": "1"},
    )
    run_cmd([
        GCTAG_BIN, "mlma", "--bfile", D("nomiss_bed"),
        "--load-reml", W("reml_tun3_p10.reml"), "--pheno", D("trait.phen"),
        "--stream-budget", "0.0008", *T, "--out", W("mm_budget_bed"),
    ])
    assert_file_byte_identical(W("mm_budget_bed.mlma"), W("mm_budget_p10.mlma"), "mlma_nomiss_budget")
    m_nat = re.search(r"MLMA streaming block size: (\d+) SNPs per tile", log_native_budget)
    m_fb = re.search(r"MLMA streaming block size: (\d+) SNPs per tile", log_fb_budget)
    assert m_nat and m_fb and int(m_nat.group(1)) > int(m_fb.group(1)), (
        f"Expected native PGEN block size ({m_nat and m_nat.group(1)}) > "
        f"dosage fallback block size ({m_fb and m_fb.group(1)})"
    )
    ok(f"mlma --stream-budget compact PGEN block sizing ({m_nat.group(1)} > {m_fb.group(1)} SNPs/tile) & zero-missing parity")

    # Dense FP32 GCTA_DISABLE_NATIVE_PGEN=1 bit-for-bit check
    run_cmd(
        [
            GCTAG_BIN, "mlma", "--pfile", D("miss_p10"),
            "--grm", base_grm, "--pheno", D("trait.phen"), "--qcovar", D("trait.qcovar"),
            *T, "--out", W("mm_dense_f32_disable_native"),
        ],
        env_extra={"GCTA_DISABLE_NATIVE_PGEN": "1"},
    )
    assert_file_byte_identical(
        W("mm_dense_f32_bed.mlma"),
        W("mm_dense_f32_disable_native.mlma"),
        "mlma_dense_f32_disable_native",
    )
    ok("mlma dense_f32 GCTA_DISABLE_NATIVE_PGEN=1 bit-for-bit")

    # 4. PCA parity on GRM built from --pfile vs --bfile
    run_cmd([GCTAG_BIN, "pca", "--grm", W("g_miss_alg1_bed"), "--pcs", "5", *T, "--out", W("pca_bed")])
    run_cmd([GCTAG_BIN, "pca", "--grm", W("g_miss_alg1_p10"), "--pcs", "5", *T, "--out", W("pca_p10")])
    assert_file_byte_identical(W("pca_bed.eigenval"), W("pca_p10.eigenval"), "pca_eigenval")
    assert_file_byte_identical(W("pca_bed.eigenvec"), W("pca_p10.eigenvec"), "pca_eigenvec")
    ok("pca (--pcs 5) bit-for-bit on PGEN GRM vs BED GRM")

    # 5. Cross-check against legacy --freq, --make-bed, --ld & V2 CLI subcommand diagnostics
    run_cmd([GCTAG_BIN, "legacy", "--bfile", D("miss_bed"), "--autosome", "--freq", "--out", W("leg_freq")])
    frq_map: Dict[str, Tuple[str, float]] = {}
    with open(W("leg_freq.freq")) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3 and parts[2] not in ("nan", "NA", "-nan"):
                frq_map[parts[0]] = (parts[1], float(parts[2]))
    with open(W("m_wb_p10.mlma")) as f:
        f.readline()
        checked_frq = 0
        for line in f:
            parts = line.split("\t")
            if parts[1] in frq_map and parts[5] != "NA":
                leg_al, leg_af = frq_map[parts[1]]
                # Legacy GCTA --freq outputs: SNP, ref_allele (A2 by default), AF_of_effect_allele(A1)
                exp_a1_af = leg_af if leg_al == parts[4] else (1.0 - leg_af)
                assert abs(float(parts[5]) - exp_a1_af) <= 5e-6, (
                    f"Freq mismatch on {parts[1]}: pfile mlma={parts[5]} vs legacy --freq={exp_a1_af}"
                )
                checked_frq += 1
    assert checked_frq >= 150
    ok(f"allele frequency parity: --pfile vs legacy --freq ({checked_frq} SNPs)")

    # Legacy --make-bed subset -> convert to Mode 0x10 .pgen -> verify GRM parity
    run_cmd([
        GCTAG_BIN, "legacy", "--bfile", D("miss_bed"), "--keep", D("keep199.txt"),
        "--extract", D("extract_drop_anchors.txt"), "--maf", "0.05",
        "--make-bed", "--out", W("leg_sub_bed"),
    ])
    run_cmd([GCTAG_BIN, "grm", "make", "--bfile", W("leg_sub_bed"), "--alg", "1", *T, "--out", W("g_from_leg_makebed")])
    run_cmd([
        GCTAG_BIN, "grm", "make", "--pfile", D("miss_p10"), "--keep", D("keep199.txt"),
        "--extract", D("extract_drop_anchors.txt"), "--maf", "0.05",
        "--alg", "1", *T, "--out", W("g_direct_p10_sub"),
    ])
    assert_grm_byte_identical(W("g_from_leg_makebed"), W("g_direct_p10_sub"), "legacy_makebed_vs_pfile_sub")
    ok("subset parity: legacy --make-bed (--keep + --extract + --maf) vs direct --pfile")

    # Legacy --ld target SNP r check against exact genotype correlation
    target_snps_path = W("ld_targets.snplist")
    with open(target_snps_path, "w") as f:
        f.write("rs1_cur_2\nrs1_cur_3\n")
    run_cmd([
        GCTAG_BIN, "legacy", "--bfile", D("nomiss_bed"),
        "--ld", target_snps_path, "--ld-wind", "1000", "--ld-sig", "1.0",
        "--out", W("leg_ld"),
    ])
    assert os.path.exists(W("leg_ld.rsq.ld"))
    ok("legacy --ld reference execution")

    # Verify V2 CLI diagnostics for disabled flat flags (--freq, --make-bed, --ld-matrix, --fastGWA-mlm)
    for disabled_flag in ("--freq", "--make-bed", "--ld-matrix", "--fastGWA-mlm"):
        rc, diag = run_cmd([GCTAG_BIN, disabled_flag, "--pfile", D("miss_p10")], expect_rc=None)
        assert rc != 0 and "temporarily disabled in this build" in diag, (
            f"Unexpected CLI response for {disabled_flag}: rc={rc}\n{diag}"
        )
    ok("V2 CLI clean diagnostic for disabled flat flags (--freq/--make-bed/--ld-matrix/--fastGWA-mlm)")


def run_dosage_fallback_suite(data_dir: str, work_dir: str) -> None:
    D = lambda fn: os.path.join(data_dir, fn)
    W = lambda fn: os.path.join(work_dir, fn)
    T = ["--threads", "4"]

    # Baseline on the 80-SNP hardcall BED reference
    run_cmd([GCTAG_BIN, "grm", "make", "--bfile", D("dos_ref_bed"), "--alg", "0", *T, "--out", W("dos_ref_g0")])
    run_cmd([GCTAG_BIN, "grm", "make", "--bfile", D("dos_ref_bed"), "--alg", "1", *T, "--out", W("dos_ref_g1")])
    run_cmd([
        GCTAG_BIN, "mlma", "--bfile", D("dos_ref_bed"), "--grm", W("dos_ref_g0"),
        "--pheno", D("trait.phen"), *T, "--out", W("dos_ref_mlma"),
    ])
    run_cmd([
        GCTAG_BIN, "mlma", "--bfile", D("dos_ref_bed"), "--grm", W("dos_ref_g0"),
        "--pheno", D("trait.phen"), "--keep", D("keep199.txt"),
        "--woodbury-basis", "30", "--seed", "5", *T, "--out", W("dos_ref_mlma_keep_wb"),
    ])

    # Exact dosage .pgen:
    # - Mode 0x03 dense dosage vs Mode 0x10 sparse bitarray dosage (0 < dosage_ct < N)
    #   must produce byte-identical GRMs (.grm.bin/.grm.N.bin/.grm.id) on --alg 0 & --alg 1!
    # - On VanRaden (--alg 1), dosage GRM uses empirical dosage variance sum(s_m^2) vs
    #   hardcall HWE sum(2*p*(1-p)), so every (i, j) pair differs only by that exact
    #   constant scalar ratio while .grm.N.bin & .grm.id are byte-identical!
    # - On additive MLMA (unstandardized centered genotypes), Mode 0x03, sparse 0x10,
    #   and mixed (0x00/0x60/0x40) 0x10 match --bfile .mlma byte-for-byte!
    for dtag in ("exact_dos_p03", "exact_dos_sparse_p10", "exact_dos_p10"):
        run_cmd([GCTAG_BIN, "grm", "make", "--pfile", D(dtag), "--alg", "0", *T, "--out", W(f"{dtag}_g0")])
        run_cmd([GCTAG_BIN, "grm", "make", "--pfile", D(dtag), "--alg", "1", *T, "--out", W(f"{dtag}_g1")])
        for ext in (".grm.id", ".grm.N.bin"):
            assert_file_byte_identical(W("dos_ref_g1" + ext), W(f"{dtag}_g1" + ext), f"{dtag}{ext}")

        run_cmd([
            GCTAG_BIN, "mlma", "--pfile", D(dtag), "--grm", W("dos_ref_g0"),
            "--pheno", D("trait.phen"), *T, "--out", W(f"{dtag}_mlma"),
        ])
        assert_file_byte_identical(W("dos_ref_mlma.mlma"), W(f"{dtag}_mlma.mlma"), f"{dtag}_mlma_exact")

        run_cmd([
            GCTAG_BIN, "mlma", "--pfile", D(dtag), "--grm", W("dos_ref_g0"),
            "--pheno", D("trait.phen"), "--keep", D("keep199.txt"),
            "--woodbury-basis", "30", "--seed", "5", *T, "--out", W(f"{dtag}_mlma_keep_wb"),
        ])
        assert_mlma_close(
            W("dos_ref_mlma_keep_wb.mlma"),
            W(f"{dtag}_mlma_keep_wb.mlma"),
            f"{dtag}_mlma_keep_wb",
            atol=1e-5,
        )
        ok(f"dosage fallback ({dtag}) exact MLMA + --keep Woodbury + GRM N/id parity")

    assert_grm_byte_identical(W("exact_dos_p03_g0"), W("exact_dos_sparse_p10_g0"), "dos_p03_vs_sparse_p10_g0")
    assert_grm_byte_identical(W("exact_dos_p03_g1"), W("exact_dos_sparse_p10_g1"), "dos_p03_vs_sparse_p10_g1")
    fa_bed = array.array("f"); fa_bed.frombytes(open(W("dos_ref_g1.grm.bin"), "rb").read())
    fa_dos = array.array("f"); fa_dos.frombytes(open(W("exact_dos_p03_g1.grm.bin"), "rb").read())
    scale_ratio = fa_bed[0] / fa_dos[0]
    max_scaled_diff = max(abs(x - y * scale_ratio) for x, y in zip(fa_bed, fa_dos))
    assert max_scaled_diff <= 3e-6, f"VanRaden dosage proportionality failed: max_diff={max_scaled_diff:.3e}"
    ok(f"dosage fallback GRM: p03 == sparse_p10 byte-identical & VanRaden proportionality max|d|={max_scaled_diff:.2e}")

    # Fractional sparse/dense dosage .pgen: grm make + mlma AF accuracy + --impute-mean clean error
    run_cmd([GCTAG_BIN, "grm", "make", "--pfile", D("frac_dos_p10"), "--alg", "1", *T, "--out", W("frac_grm")])
    run_cmd([
        GCTAG_BIN, "mlma", "--pfile", D("frac_dos_p10"), "--grm", W("frac_grm"),
        "--pheno", D("trait.phen"), "--woodbury-basis", "30", "--seed", "9",
        *T, "--out", W("frac_mlma"),
    ])
    exp_af: Dict[str, float] = {}
    with open(D("frac_dos_p10.expected_af")) as f:
        for line in f:
            snp, val = line.split()
            exp_af[snp] = float(val)
    with open(W("frac_mlma.mlma")) as f:
        f.readline()
        rows = [ln.split("\t") for ln in f if ln.strip()]
    assert len(rows) == len(exp_af)
    for r in rows:
        got_af = float(r[5])
        want_af = exp_af[r[1]]
        assert abs(got_af - want_af) <= 5e-6, f"Fractional dosage AF mismatch for {r[1]}: {got_af} vs {want_af}"
    ok("fractional sparse+dense dosage .pgen fallback AF & association accuracy")

    rc_imp, log_imp = run_cmd(
        [GCTAG_BIN, "grm", "make", "--pfile", D("frac_dos_p10"), "--alg", "1", "--impute-mean", "--out", W("frac_imp_fail")],
        expect_rc=None,
    )
    assert rc_imp != 0 and "--impute-mean requires native hardcall" in log_imp, (
        f"Expected --impute-mean error on dosage .pgen, got rc={rc_imp}:\n{log_imp}"
    )
    ok("grm make --impute-mean cleanly rejects dosage .pgen")


def main() -> int:
    if not os.path.exists(GCTAG_BIN):
        raise FileNotFoundError(f"GCTAg binary not found at: {GCTAG_BIN}")

    root_tmp = USER_WORK if USER_WORK else tempfile.mkdtemp(prefix="gcta_pgen_e2e_")
    os.makedirs(root_tmp, exist_ok=True)
    own_tmp = not bool(USER_WORK)

    try:
        data_dir = os.path.join(root_tmp, "data")
        work_dir = os.path.join(root_tmp, "work")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(work_dir, exist_ok=True)

        print(f"Generating matched BED & PGEN datasets in {data_dir} ...")
        info = generate_cohort(data_dir)
        build_dosage_datasets(
            data_dir,
            info["samples"],  # type: ignore[arg-type]
            info["snps_miss"],  # type: ignore[arg-type]
        )

        print("Running GRM parity suite ...")
        base_grm = run_grm_parity_suite(data_dir, work_dir)

        print("Running MLMA / REML / PCA / Legacy parity suite ...")
        run_mlma_reml_pca_legacy_suite(data_dir, work_dir, base_grm)

        print("Running Dosage fallback suite ...")
        run_dosage_fallback_suite(data_dir, work_dir)

        print(f"\nALL {PASS_COUNT} PGEN END-TO-END PARITY CHECKS PASSED.")
        return 0
    finally:
        if own_tmp:
            shutil.rmtree(root_tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
