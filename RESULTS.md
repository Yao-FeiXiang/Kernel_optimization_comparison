# SGEMM Benchmark Results

该文件由 `benchmark.py --update-results` 维护。不要手工编辑下面两个标记之间的内容；相同 GPU、CUDA、矩阵尺寸和测试参数下的同一方法会被替换，不会重复追加。

<!-- SGEMM_RESULTS_BEGIN -->
## NVIDIA A100-PCIE-40GB — M=4096, N=4096, K=4096

- Compute capability: `8.0`; CUDA runtime: `13.3`; alpha=0.8; beta=0.2
- Warmup: 5; timed samples: 50; updated: `2026-08-20T03:38:38Z`
- Reproduce: `python3 benchmark.py --all --m 4096 --n 4096 --k 4096 --update-results`

<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":291.851432775,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"scalar","k":4096,"latency_ms":471.035919189,"m":4096,"max_abs_error":0.0,"max_rel_error":0.0,"method":"naive","method_id":1,"min_latency_ms":471.033843994,"n":4096,"repeat":50,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T03:38:38Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":3143.52439242,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"coalesced-scalar","k":4096,"latency_ms":43.731967926,"m":4096,"max_abs_error":0.0,"max_rel_error":0.0,"method":"coalesced","method_id":2,"min_latency_ms":43.330558777,"n":4096,"repeat":50,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T03:38:38Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":4903.859177447,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"shared-memory","k":4096,"latency_ms":28.033535004,"m":4096,"max_abs_error":0.0,"max_rel_error":0.0,"method":"shared","method_id":3,"min_latency_ms":27.389951706,"n":4096,"repeat":50,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T03:38:38Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":10000.036937117,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"register-tile-1d","k":4096,"latency_ms":13.747200012,"m":4096,"max_abs_error":0.0,"max_rel_error":0.0,"method":"blocktiling-1d","method_id":4,"min_latency_ms":13.717503548,"n":4096,"repeat":50,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T03:38:38Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":14291.850072281,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"register-tile-2d-fast","k":4096,"latency_ms":9.618944168,"m":4096,"max_abs_error":0.0,"max_rel_error":0.0,"method":"blocktiling-2d","method_id":5,"min_latency_ms":9.531392097,"n":4096,"repeat":50,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T03:38:38Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":15169.547797942,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"float4","k":4096,"latency_ms":9.062399864,"m":4096,"max_abs_error":0.0,"max_rel_error":0.0,"method":"vectorized","method_id":6,"min_latency_ms":9.038847923,"n":4096,"repeat":50,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T03:38:38Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"BM128_BN128_BK8_TM8_TN8","cuda_runtime":"13.3","gflops":15179.840092191,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"autotuned-vectorized","k":4096,"latency_ms":9.056255341,"m":4096,"max_abs_error":0.0,"max_rel_error":0.0,"method":"autotuned","method_id":9,"min_latency_ms":8.951807976,"n":4096,"repeat":50,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T03:38:38Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"BM64_BN128_BK16_WM32_WN64_WNITER1_TM4_TN4_THREADS128","cuda_runtime":"13.3","gflops":16464.372508914,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"warp-tiled-a100","k":4096,"latency_ms":8.349696159,"m":4096,"max_abs_error":0.0,"max_rel_error":0.0,"method":"warptiling","method_id":10,"min_latency_ms":8.263680458,"n":4096,"repeat":50,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T03:38:38Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"cuBLAS","cuda_runtime":"13.3","gflops":17419.294129241,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"cublas-sgemm","k":4096,"latency_ms":7.891967773,"m":4096,"max_abs_error":0.0,"max_rel_error":0.0,"method":"cublas","method_id":0,"min_latency_ms":7.586815834,"n":4096,"repeat":50,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T03:38:38Z","warmup":5} -->

| Rank | ID | Method | Status | Variant / configuration | Median ms | Min ms | GFLOP/s | Performance vs cuBLAS | vs Naive | Max abs error | Max rel error |
|---:|---:|:---|:---:|:---|---:|---:|---:|:---|---:|---:|---:|
| 1 | 0 | cublas | pass | cublas-sgemm / cuBLAS | 7.892 | 7.587 | 17419.294 | ████████████████████ 100.0% | 59.685x | 0.000e+00 | 0.000e+00 |
| 2 | 10 | warptiling | pass | warp-tiled-a100 / BM64_BN128_BK16_WM32_WN64_WNITER1_TM4_TN4_THREADS128 | 8.350 | 8.264 | 16464.373 | ███████████████████ 94.5% | 56.414x | 0.000e+00 | 0.000e+00 |
| 3 | 9 | autotuned | pass | autotuned-vectorized / BM128_BN128_BK8_TM8_TN8 | 9.056 | 8.952 | 15179.840 | █████████████████ 87.1% | 52.012x | 0.000e+00 | 0.000e+00 |
| 4 | 6 | vectorized | pass | float4 / fixed | 9.062 | 9.039 | 15169.548 | █████████████████ 87.1% | 51.977x | 0.000e+00 | 0.000e+00 |
| 5 | 5 | blocktiling-2d | pass | register-tile-2d-fast / fixed | 9.619 | 9.531 | 14291.850 | ████████████████ 82.0% | 48.970x | 0.000e+00 | 0.000e+00 |
| 6 | 4 | blocktiling-1d | pass | register-tile-1d / fixed | 13.747 | 13.718 | 10000.037 | ███████████ 57.4% | 34.264x | 0.000e+00 | 0.000e+00 |
| 7 | 3 | shared | pass | shared-memory / fixed | 28.034 | 27.390 | 4903.859 | ██████ 28.2% | 16.803x | 0.000e+00 | 0.000e+00 |
| 8 | 2 | coalesced | pass | coalesced-scalar / fixed | 43.732 | 43.331 | 3143.524 | ████ 18.0% | 10.771x | 0.000e+00 | 0.000e+00 |
| 9 | 1 | naive | pass | scalar / fixed | 471.036 | 471.034 | 291.851 | 1.7% | 1.000x | 0.000e+00 | 0.000e+00 |
<!-- SGEMM_RESULTS_END -->
