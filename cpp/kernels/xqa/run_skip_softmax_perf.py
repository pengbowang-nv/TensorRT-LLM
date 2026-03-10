#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build and run skipSoftmaxAttention performance benchmarks on Hopper.

Usage:
    python run_skip_softmax_perf.py [--head_dim 128] [--head_grp_size 16]
                                     [--build_dir build]
                                     [--log_dir skip_softmax_logs]

Default config: q_heads=64, kv_heads=4, head_dim=128 (HEAD_GRP_SIZE=16)
Builds 4 variants: {BF16, FP8} x {noskip, skip}
Runs Perf.skip_softmax_attn for each.
"""

import argparse
import os
import sys
from pathlib import Path


def run_cmd(cmd, desc=""):
    """Run a shell command, printing output."""
    if desc:
        print(f"\n{'=' * 60}")
        print(f"  {desc}")
        print(f"{'=' * 60}")
    print(f"$ {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"FAILED (exit code {ret}): {cmd}")
        sys.exit(1)


def build(build_dir, dtype, head_dim, head_grp_size, enable_skip_softmax):
    """Build unitTests with specified configuration."""
    cmake_args = [
        f"-DHEAD_ELEMS={head_dim}",
        f"-DHEAD_GRP_SIZE={head_grp_size}",
    ]

    if dtype == "bf16":
        cmake_args += ["-DINPUT_FP16=0", "-DCACHE_ELEM_ENUM=0"]
    elif dtype == "fp8":
        cmake_args += ["-DINPUT_FP16=0", "-DCACHE_ELEM_ENUM=2"]
    else:
        raise ValueError(f"Invalid dtype: {dtype}")

    if enable_skip_softmax:
        cmake_args += ["-DSKIP_SOFTMAX_ATTN=ON", "-DSKIP_SOFTMAX_ATTN_BLOCK_STATS=ON"]
    else:
        cmake_args += ["-DSKIP_SOFTMAX_ATTN=OFF", "-DSKIP_SOFTMAX_ATTN_BLOCK_STATS=OFF"]

    cmake_str = " ".join(cmake_args)
    skip_str = "skip" if enable_skip_softmax else "noskip"
    desc = f"Building: {dtype} {skip_str} (head_dim={head_dim}, head_grp_size={head_grp_size})"

    # Clean build artifacts but preserve _deps (3rdparty downloads)
    os.makedirs(build_dir, exist_ok=True)
    run_cmd(
        f"find {build_dir} -maxdepth 1 ! -name '_deps' ! -path {build_dir} -exec rm -rf {{}} +",
        f"Cleaning {build_dir} (preserving _deps)",
    )
    run_cmd(
        f"cd {build_dir} && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_XQA_TESTS=ON "
        f"{cmake_str} && cmake --build . -j",
        desc,
    )


def run_test(build_dir, log_dir, dtype, enable_skip_softmax):
    """Run the skip_softmax_attn perf test and save log."""
    skip_str = "skip" if enable_skip_softmax else "noskip"
    log_file = os.path.join(log_dir, f"{skip_str}_{dtype}.log")
    desc = f"Running: Perf.skip_softmax_attn ({dtype}, {skip_str})"
    run_cmd(
        f"cd {build_dir} && ./unitTests --gtest_filter=Perf.skip_softmax_attn 2>&1 | tee {os.path.abspath(log_file)}",
        desc,
    )


def main():
    parser = argparse.ArgumentParser(description="Run skipSoftmaxAttention perf benchmarks")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension (default: 128)")
    parser.add_argument(
        "--head_grp_size",
        type=int,
        default=16,
        help="HEAD_GRP_SIZE = num_q_heads/num_kv_heads (default: 16 for 64/4)",
    )
    parser.add_argument("--build_dir", default="build", help="Build directory")
    parser.add_argument("--log_dir", default="skip_softmax_logs", help="Log output directory")
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["bf16", "fp8"],
        choices=["bf16", "fp8"],
        help="Data types to test",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    build_dir = script_dir / args.build_dir
    log_dir = script_dir / args.log_dir

    os.makedirs(log_dir, exist_ok=True)
    os.chdir(script_dir)

    for dtype in args.dtypes:
        # Build and run without skip softmax (baseline)
        build(str(build_dir), dtype, args.head_dim, args.head_grp_size, enable_skip_softmax=False)
        run_test(str(build_dir), str(log_dir), dtype, enable_skip_softmax=False)

        # Build and run with skip softmax
        build(str(build_dir), dtype, args.head_dim, args.head_grp_size, enable_skip_softmax=True)
        run_test(str(build_dir), str(log_dir), dtype, enable_skip_softmax=True)

    print(f"\n{'=' * 60}")
    print(f"All benchmarks complete. Logs saved to: {log_dir}/")
    print(f"Run: python parse_skip_softmax_log.py --log_dir {args.log_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
