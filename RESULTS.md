# SGEMM Benchmark Results

该文件由 `benchmark.py --update-results` 维护。不要手工编辑下面两个标记之间的内容；相同 GPU、CUDA、矩阵尺寸和测试参数下的同一方法会被替换，不会重复追加。

<!-- SGEMM_RESULTS_BEGIN -->
## NVIDIA A100-PCIE-40GB — M=4096, N=4096, K=4096

- Compute capability: `8.0`; CUDA runtime: `13.3`; alpha=0.8; beta=0.2
- Warmup: 5; timed samples: 50; updated: `2026-08-20T03:28:32Z`
- Reproduce: `python3 benchmark.py --all --m 4096 --n 4096 --k 4096 --update-results`

<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":291.85334255,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"scalar","k":4096,"latency_ms":471.032836914,"m":4096,"max_abs_error":0.0,"max_rel_error":0.0,"method":"naive","method_id":1,"min_latency_ms":471.030792236,"n":4096,"repeat":50,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T03:28:32Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":3144.223774797,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"coalesced-scalar","k":4096,"latency_ms":43.722240448,"m":4096,"max_abs_error":0.0,"max_rel_error":0.0,"method":"coalesced","method_id":2,"min_latency_ms":43.33158493,"n":4096,"repeat":50,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T03:28:32Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":4892.778421893,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"shared-memory","k":4096,"latency_ms":28.09702301,"m":4096,"max_abs_error":0.0,"max_rel_error":0.0,"method":"shared","method_id":3,"min_latency_ms":26.8175354,"n":4096,"repeat":50,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T03:28:32Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":10007.117944146,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"register-tile-1d","k":4096,"latency_ms":13.737472534,"m":4096,"max_abs_error":0.0,"max_rel_error":0.0,"method":"blocktiling-1d","method_id":4,"min_latency_ms":13.715456009,"n":4096,"repeat":50,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T03:28:32Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":14359.878081109,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"register-tile-2d-fast","k":4096,"latency_ms":9.573375702,"m":4096,"max_abs_error":0.0,"max_rel_error":0.0,"method":"blocktiling-2d","method_id":5,"min_latency_ms":9.531392097,"n":4096,"repeat":50,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T03:28:32Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":15171.262477102,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"float4","k":4096,"latency_ms":9.061375618,"m":4096,"max_abs_error":0.0,"max_rel_error":0.0,"method":"vectorized","method_id":6,"min_latency_ms":9.046015739,"n":4096,"repeat":50,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T03:28:32Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"BM128_BN128_BK8_TM8_TN8","cuda_runtime":"13.3","gflops":15172.975946863,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"autotuned-vectorized","k":4096,"latency_ms":9.060352325,"m":4096,"max_abs_error":0.0,"max_rel_error":0.0,"method":"autotuned","method_id":9,"min_latency_ms":9.044992447,"n":4096,"repeat":50,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T03:28:32Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"BM64_BN128_BK16_WM32_WN64_WNITER1_TM4_TN4_THREADS128","cuda_runtime":"13.3","gflops":16462.353093772,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"warp-tiled-a100","k":4096,"latency_ms":8.350720406,"m":4096,"max_abs_error":0.0,"max_rel_error":0.0,"method":"warptiling","method_id":10,"min_latency_ms":8.252415657,"n":4096,"repeat":50,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T03:28:32Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"cuBLAS","cuda_runtime":"13.3","gflops":17413.645177263,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"cublas-sgemm","k":4096,"latency_ms":7.894527912,"m":4096,"max_abs_error":0.0,"max_rel_error":0.0,"method":"cublas","method_id":0,"min_latency_ms":7.456768036,"n":4096,"repeat":50,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T03:28:32Z","warmup":5} -->

| Rank | ID | Method | Status | Variant / configuration | Median ms | Min ms | GFLOP/s | Performance vs cuBLAS | vs Naive | Max abs error | Max rel error |
|---:|---:|:---|:---:|:---|---:|---:|---:|:---|---:|---:|---:|
| 1 | 0 | cublas | pass | cublas-sgemm / cuBLAS | 7.895 | 7.457 | 17413.645 | ████████████████████ 100.0% | 59.666x | 0.000e+00 | 0.000e+00 |
| 2 | 10 | warptiling | pass | warp-tiled-a100 / BM64_BN128_BK16_WM32_WN64_WNITER1_TM4_TN4_THREADS128 | 8.351 | 8.252 | 16462.353 | ███████████████████ 94.5% | 56.406x | 0.000e+00 | 0.000e+00 |
| 3 | 9 | autotuned | pass | autotuned-vectorized / BM128_BN128_BK8_TM8_TN8 | 9.060 | 9.045 | 15172.976 | █████████████████ 87.1% | 51.988x | 0.000e+00 | 0.000e+00 |
| 4 | 6 | vectorized | pass | float4 / fixed | 9.061 | 9.046 | 15171.262 | █████████████████ 87.1% | 51.982x | 0.000e+00 | 0.000e+00 |
| 5 | 5 | blocktiling-2d | pass | register-tile-2d-fast / fixed | 9.573 | 9.531 | 14359.878 | ████████████████ 82.5% | 49.202x | 0.000e+00 | 0.000e+00 |
| 6 | 4 | blocktiling-1d | pass | register-tile-1d / fixed | 13.737 | 13.715 | 10007.118 | ███████████ 57.5% | 34.288x | 0.000e+00 | 0.000e+00 |
| 7 | 3 | shared | pass | shared-memory / fixed | 28.097 | 26.818 | 4892.778 | ██████ 28.1% | 16.765x | 0.000e+00 | 0.000e+00 |
| 8 | 2 | coalesced | pass | coalesced-scalar / fixed | 43.722 | 43.332 | 3144.224 | ████ 18.1% | 10.773x | 0.000e+00 | 0.000e+00 |
| 9 | 1 | naive | pass | scalar / fixed | 471.033 | 471.031 | 291.853 | 1.7% | 1.000x | 0.000e+00 | 0.000e+00 |
<!-- SGEMM_RESULTS_END -->
