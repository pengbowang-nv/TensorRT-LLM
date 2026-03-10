# Reproduce: skipSoftmaxAttention Performance Benchmark

## Prerequisites

- Hopper GPU (SM90a), e.g. H100
- CUDA toolkit with nvcc supporting sm_90a
- CMake 3.18+
- Python 3

## Quick Start

```bash
cd cpp/kernels/xqa

# Build all 4 variants (BF16/FP8 x noskip/skip) and run benchmarks
python run_skip_softmax_perf.py

# Parse logs into formatted tables
python parse_skip_softmax_log.py

# Also output CSV for spreadsheet import
python parse_skip_softmax_log.py --csv
```

## Configuration

Default: q_heads=64, kv_heads=4, head_dim=128, BS=64, seqlen={16k, 64k}.

Override via command-line:

```bash
python run_skip_softmax_perf.py --head_dim 128 --head_grp_size 16 --dtypes bf16 fp8
```

| Argument         | Default | Description                          |
|------------------|---------|--------------------------------------|
| `--head_dim`     | 128     | Head dimension                       |
| `--head_grp_size`| 16      | num_q_heads / num_kv_heads           |
| `--dtypes`       | bf16 fp8| Data types to benchmark              |
| `--build_dir`    | build   | Build directory (preserves `_deps`)  |
| `--log_dir`      | skip_softmax_logs | Log output directory       |

## What It Does

The script builds and runs 4 variants:

| Variant     | CMAKE flags                                              |
|-------------|----------------------------------------------------------|
| BF16 noskip | `INPUT_FP16=0 CACHE_ELEM_ENUM=0 SKIP_SOFTMAX_ATTN=OFF`  |
| BF16 skip   | `INPUT_FP16=0 CACHE_ELEM_ENUM=0 SKIP_SOFTMAX_ATTN=ON`   |
| FP8 noskip  | `INPUT_FP16=0 CACHE_ELEM_ENUM=2 SKIP_SOFTMAX_ATTN=OFF`  |
| FP8 skip    | `INPUT_FP16=0 CACHE_ELEM_ENUM=2 SKIP_SOFTMAX_ATTN=ON`   |

Each variant runs `Perf.skip_softmax_attn` which sweeps thresholds:
0, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 5.0

## Manual Build & Run

To build and run a single variant manually:

```bash
cd cpp/kernels/xqa
mkdir -p build && cd build

# Example: FP8 with skip softmax enabled
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_XQA_TESTS=ON \
  -DHEAD_ELEMS=128 -DHEAD_GRP_SIZE=16 \
  -DINPUT_FP16=0 -DCACHE_ELEM_ENUM=2 \
  -DSKIP_SOFTMAX_ATTN=ON -DSKIP_SOFTMAX_ATTN_BLOCK_STATS=ON

cmake --build . -j

./unitTests --gtest_filter=Perf.skip_softmax_attn
```

## Output

Logs are saved to `skip_softmax_logs/`:
- `noskip_bf16.log`, `skip_bf16.log`
- `noskip_fp8.log`, `skip_fp8.log`

`parse_skip_softmax_log.py` produces a side-by-side table:

```
                          BF16                              FP8
threshold  sparsity%  time/ms  BW(TB/s) Speedup   sparsity%  time/ms  BW(TB/s) Speedup
---------  ---------  -------  -------- -------   ---------  -------  -------- -------
0(NoSkip)      0.00%    1.135    4.457   1.000       0.00%    0.589    4.298   1.000
    0.500      4.90%    1.110    4.560   1.023      43.97%    0.483    5.249   1.221
    1.000     59.07%    0.824    6.142   1.378      59.13%    0.443    5.714   1.329
    5.000     98.49%    0.615    8.222   1.845      83.26%    0.384    6.599   1.535
```

## Notes

- Bandwidth is auto-detected from log output; falls back to 4.8 TB/s (H200).
- `dramSolRatio` uses full (non-skip-reduced) memory traffic.
- The test passes `scaleFactor = threshold * seqLen` because the kernel computes `threshold = scaleFactor / seqLen`.
- `SKIP_SOFTMAX_ATTN_FIX_THRESHOLD_GREATER_THAN_ONE=1` (default in defines.h) allows threshold > 1.
