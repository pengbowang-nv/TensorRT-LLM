#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse skipSoftmaxAttention perf logs and generate a formatted data table.

Detects GPU bandwidth automatically from log output. Falls back to 4.8 TB/s (H200).

Usage:
    python parse_skip_softmax_log.py [--log_dir skip_softmax_logs]

Output format (per batch_size x seqlen):
    threshold | sparsity% | time/ms | Bandwidth(TB/s) | Speedup
    for BF16 and FP8 side-by-side
"""

import argparse
import re
import sys
from pathlib import Path

# Fallback bandwidth (TB/s). Overridden if detected from log.
DEFAULT_BANDWIDTH_TBS = 4.8


def detect_bandwidth(log_dir):
    """Try to detect theoretical bandwidth from any log file."""
    for f in Path(log_dir).glob("*.log"):
        content = f.read_text()
        m = re.search(r"bandwidth = ([\d.eE+\-]+)", content)
        if m:
            bw_bytes = float(m.group(1))
            return bw_bytes / 1e12  # bytes/s -> TB/s
    return DEFAULT_BANDWIDTH_TBS


def parse_noskip_log(filepath):
    """Parse noskip log: baseline performance without skip softmax."""
    results = {}
    if not Path(filepath).exists():
        return results
    content = Path(filepath).read_text()

    pattern = (
        r"batchSize: (\d+), num_k_heads=(\d+), seqLen: (\d+), original attention kernel\n"
        r"dramSolRatio: ([\d.]+)% \(([\d.]+) ms, TOPS = ([\d.]+)\)"
    )
    for m in re.finditer(pattern, content):
        bs, nkh, sl, dram_sol, time_ms, tops = m.groups()
        key = (int(bs), int(sl))
        results[key] = {
            "dram_sol": float(dram_sol),
            "time_ms": float(time_ms),
            "tops": float(tops),
        }
    return results


def parse_skip_log(filepath):
    """Parse skip log: performance at various thresholds."""
    results = {}
    if not Path(filepath).exists():
        return results
    content = Path(filepath).read_text()

    pattern = (
        r"batchSize: (\d+), num_k_heads=(\d+), seqLen: (\d+), skipSoftmaxThreshold: ([\d.]+)\n"
        r"kernel skippedBlockCount: \d+/\d+ \(([\d.]+)%\)\n"
        r"dramSolRatio: ([\d.]+)% \(([\d.]+) ms, TOPS = ([\d.]+)\)"
    )
    for m in re.finditer(pattern, content):
        bs, nkh, sl, threshold, sparsity, dram_sol, time_ms, tops = m.groups()
        key = (int(bs), int(sl))
        if key not in results:
            results[key] = []
        results[key].append(
            {
                "threshold": float(threshold),
                "sparsity": float(sparsity),
                "dram_sol": float(dram_sol),
                "time_ms": float(time_ms),
                "tops": float(tops),
            }
        )
    return results


def format_side_by_side(bf16_noskip, bf16_skip, fp8_noskip, fp8_skip, config_key, bandwidth_tbs):
    """Generate side-by-side BF16 vs FP8 table for a (batch_size, seqlen) pair."""
    batch_size, seq_len = config_key
    seq_len_k = seq_len // 1024

    phase = "Gen phase" if batch_size > 1 else "Prefill phase"
    lines = []
    lines.append(f"\n{'=' * 120}")
    lines.append(f"{phase}    BS={batch_size}    dH=128        num q heads 64    num kv heads 4")
    lines.append(f"seqlen = {seq_len_k}k")
    lines.append(f"{'=' * 120}")

    # Header
    bf16_hdr = f"{'BF16':^55}"
    fp8_hdr = f"{'FP8':^55}"
    lines.append(f"{'':>15}{bf16_hdr}    {fp8_hdr}")

    col_hdr = f"{'threshold':>10} {'sparsity %':>11} {'time/ms':>9} {'BW(TB/s)':>10} {'Speedup':>8}"
    lines.append(f"{'':>3}{col_hdr}    {col_hdr}")
    lines.append("-" * 120)

    def get_baseline_time(noskip_data, key):
        if key in noskip_data:
            return noskip_data[key]["time_ms"], noskip_data[key]["dram_sol"]
        return None, None

    bf16_base_ms, bf16_base_sol = get_baseline_time(bf16_noskip, config_key)
    fp8_base_ms, fp8_base_sol = get_baseline_time(fp8_noskip, config_key)

    def fmt_row(threshold_str, sparsity, time_ms, base_ms, dram_sol):
        bw = dram_sol / 100.0 * bandwidth_tbs
        speedup = base_ms / time_ms if base_ms and time_ms > 0 else 0
        return f"{threshold_str:>10} {sparsity:>10.2f}% {time_ms:>9.3f} {bw:>10.3f} {speedup:>8.3f}"

    # Baseline rows (no-skip kernel)
    bf16_base_row = (
        fmt_row("0(NoSkip)", 0.0, bf16_base_ms, bf16_base_ms, bf16_base_sol)
        if bf16_base_ms
        else " " * 55
    )
    fp8_base_row = (
        fmt_row("0(NoSkip)", 0.0, fp8_base_ms, fp8_base_ms, fp8_base_sol)
        if fp8_base_ms
        else " " * 55
    )
    lines.append(f"   {bf16_base_row}    {fp8_base_row}")

    # Skip rows - align by threshold
    bf16_entries = bf16_skip.get(config_key, [])
    fp8_entries = fp8_skip.get(config_key, [])

    # Collect all thresholds
    all_thresholds = sorted(
        set(e["threshold"] for e in bf16_entries) | set(e["threshold"] for e in fp8_entries)
    )

    bf16_map = {e["threshold"]: e for e in bf16_entries}
    fp8_map = {e["threshold"]: e for e in fp8_entries}

    for t in all_thresholds:
        bf16_e = bf16_map.get(t)
        fp8_e = fp8_map.get(t)

        bf16_col = (
            fmt_row(
                f"{t:.3f}", bf16_e["sparsity"], bf16_e["time_ms"], bf16_base_ms, bf16_e["dram_sol"]
            )
            if bf16_e and bf16_base_ms
            else " " * 55
        )
        fp8_col = (
            fmt_row(f"{t:.3f}", fp8_e["sparsity"], fp8_e["time_ms"], fp8_base_ms, fp8_e["dram_sol"])
            if fp8_e and fp8_base_ms
            else " " * 55
        )
        lines.append(f"   {bf16_col}    {fp8_col}")

    return "\n".join(lines)


def format_csv(bf16_noskip, bf16_skip, fp8_noskip, fp8_skip, config_key, bandwidth_tbs):
    """Generate CSV for spreadsheet import."""
    batch_size, seq_len = config_key
    seq_len_k = seq_len // 1024
    lines = []
    lines.append(f"\n# BS={batch_size}  seqlen={seq_len_k}k  dH=128  q_heads=64  kv_heads=4")
    lines.append(
        "threshold,BF16 sparsity%,BF16 time/ms,BF16 BW(TB/s),BF16 Speedup,"
        "FP8 sparsity%,FP8 time/ms,FP8 BW(TB/s),FP8 Speedup"
    )

    bf16_base = bf16_noskip.get(config_key)
    fp8_base = fp8_noskip.get(config_key)

    def csv_cols(entry, base, is_baseline=False):
        if not entry or not base:
            return ",,,"
        sp = 0.0 if is_baseline else entry.get("sparsity", 0.0)
        t = entry["time_ms"] if not is_baseline else base["time_ms"]
        bw = (entry["dram_sol"] if not is_baseline else base["dram_sol"]) / 100.0 * bandwidth_tbs
        speedup = base["time_ms"] / t if t > 0 else 0
        return f"{sp:.2f}%,{t:.3f},{bw:.3f},{speedup:.3f}"

    # Baseline
    bf16_base_csv = csv_cols(bf16_base, bf16_base, is_baseline=True)
    fp8_base_csv = csv_cols(fp8_base, fp8_base, is_baseline=True)
    lines.append(f"0(No Skip Kernel),{bf16_base_csv},{fp8_base_csv}")

    # Collect all thresholds
    bf16_entries = bf16_skip.get(config_key, [])
    fp8_entries = fp8_skip.get(config_key, [])
    all_thresholds = sorted(
        set(e["threshold"] for e in bf16_entries) | set(e["threshold"] for e in fp8_entries)
    )
    bf16_map = {e["threshold"]: e for e in bf16_entries}
    fp8_map = {e["threshold"]: e for e in fp8_entries}

    for t in all_thresholds:
        bf16_csv = csv_cols(bf16_map.get(t), bf16_base)
        fp8_csv = csv_cols(fp8_map.get(t), fp8_base)
        lines.append(f"{t},{bf16_csv},{fp8_csv}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Parse skipSoftmax perf logs")
    parser.add_argument(
        "--log_dir", default="skip_softmax_logs", help="Directory containing log files"
    )
    parser.add_argument("--csv", action="store_true", help="Also output CSV format")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        sys.exit(1)

    bandwidth_tbs = detect_bandwidth(log_dir)
    print(f"Using theoretical bandwidth: {bandwidth_tbs:.2f} TB/s")

    bf16_noskip = parse_noskip_log(log_dir / "noskip_bf16.log")
    bf16_skip = parse_skip_log(log_dir / "skip_bf16.log")
    fp8_noskip = parse_noskip_log(log_dir / "noskip_fp8.log")
    fp8_skip = parse_skip_log(log_dir / "skip_fp8.log")

    configs = sorted(
        set(bf16_noskip.keys())
        | set(fp8_noskip.keys())
        | set(bf16_skip.keys())
        | set(fp8_skip.keys())
    )

    if not configs:
        print("No data found in logs. Check log files in:", log_dir)
        sys.exit(1)

    print(f"\n{'#' * 120}")
    print("# skipSoftmaxAttention Hopper Performance Data")
    print(f"# Theoretical bandwidth: {bandwidth_tbs:.2f} TB/s")
    print(f"# Bandwidth = dramSolRatio * {bandwidth_tbs:.2f} TB/s")
    print(f"{'#' * 120}")

    for config in configs:
        print(
            format_side_by_side(bf16_noskip, bf16_skip, fp8_noskip, fp8_skip, config, bandwidth_tbs)
        )

    if args.csv:
        print(f"\n\n{'=' * 100}")
        print("CSV FORMAT (for spreadsheet import)")
        print(f"{'=' * 100}")
        for config in configs:
            print(format_csv(bf16_noskip, bf16_skip, fp8_noskip, fp8_skip, config, bandwidth_tbs))


if __name__ == "__main__":
    main()
