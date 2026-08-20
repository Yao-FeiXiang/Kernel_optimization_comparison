# SGEMM Benchmark Results

该文件由 `benchmark.py --update-results` 维护。不要手工编辑下面两个标记之间的内容；相同 GPU、CUDA、矩阵尺寸和测试参数下的同一方法会被替换，不会重复追加。

<!-- SGEMM_RESULTS_BEGIN -->
## NVIDIA A100-PCIE-40GB — M=1024, N=1024, K=1024

- Compute capability: `8.0`; CUDA runtime: `13.3`; alpha=0.8; beta=0.2
- Warmup: 5; timed samples: 20; updated: `2026-08-20T02:02:48Z`
- Reproduce: `python3 benchmark.py --all --m 1024 --n 1024 --k 1024 --update-results`

<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":2133.333242101,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"scalar","k":1024,"latency_ms":1.007616043,"m":1024,"max_abs_error":0.0,"max_rel_error":0.0,"method":"naive","method_id":1,"min_latency_ms":1.006592035,"n":1024,"repeat":20,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T02:02:48Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":2132.249773647,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"coalesced-scalar","k":1024,"latency_ms":1.008128047,"m":1024,"max_abs_error":0.0,"max_rel_error":0.0,"method":"coalesced","method_id":2,"min_latency_ms":1.005568027,"n":1024,"repeat":20,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T02:02:48Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":2973.37101752,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"shared-memory","k":1024,"latency_ms":0.722944021,"m":1024,"max_abs_error":0.0,"max_rel_error":0.0,"method":"shared","method_id":3,"min_latency_ms":0.722944021,"n":1024,"repeat":20,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T02:02:48Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":4623.788611191,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"register-tile-1d","k":1024,"latency_ms":0.464895993,"m":1024,"max_abs_error":0.0,"max_rel_error":0.0,"method":"blocktiling-1d","method_id":4,"min_latency_ms":0.463871986,"n":1024,"repeat":20,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T02:02:48Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":3313.654340491,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"register-tile-2d","k":1024,"latency_ms":0.648703992,"m":1024,"max_abs_error":0.0,"max_rel_error":0.0,"method":"blocktiling-2d","method_id":5,"min_latency_ms":0.645120025,"n":1024,"repeat":20,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T02:02:48Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":4680.49042011,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"float4","k":1024,"latency_ms":0.45926401,"m":1024,"max_abs_error":0.0,"max_rel_error":0.0,"method":"vectorized","method_id":6,"min_latency_ms":0.336896002,"n":1024,"repeat":20,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T02:02:48Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"BM128_BN128_BK16_TM8_TN8","cuda_runtime":"13.3","gflops":5183.209891085,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"autotuned-tile","k":1024,"latency_ms":0.414719999,"m":1024,"max_abs_error":0.0,"max_rel_error":0.0,"method":"autotuned","method_id":9,"min_latency_ms":0.414719999,"n":1024,"repeat":20,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T02:02:48Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":5424.289303057,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"warp-tiled","k":1024,"latency_ms":0.396288007,"m":1024,"max_abs_error":0.0,"max_rel_error":0.0,"method":"warptiling","method_id":10,"min_latency_ms":0.394239992,"n":1024,"repeat":20,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T02:02:48Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"cuBLAS","cuda_runtime":"13.3","gflops":13120.000331439,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"cublas-sgemm","k":1024,"latency_ms":0.163839996,"m":1024,"max_abs_error":0.0,"max_rel_error":0.0,"method":"cublas","method_id":0,"min_latency_ms":0.163839996,"n":1024,"repeat":20,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T02:02:48Z","warmup":5} -->

| Rank | ID | Method | Status | Variant / configuration | Median ms | Min ms | GFLOP/s | Performance vs cuBLAS | vs Naive | Max abs error | Max rel error |
|---:|---:|:---|:---:|:---|---:|---:|---:|:---|---:|---:|---:|
| 1 | 0 | cublas | pass | cublas-sgemm / cuBLAS | 0.164 | 0.164 | 13120.000 | ████████████████████ 100.0% | 6.150x | 0.000e+00 | 0.000e+00 |
| 2 | 10 | warptiling | pass | warp-tiled / fixed | 0.396 | 0.394 | 5424.289 | ████████ 41.3% | 2.543x | 0.000e+00 | 0.000e+00 |
| 3 | 9 | autotuned | pass | autotuned-tile / BM128_BN128_BK16_TM8_TN8 | 0.415 | 0.415 | 5183.210 | ████████ 39.5% | 2.430x | 0.000e+00 | 0.000e+00 |
| 4 | 6 | vectorized | pass | float4 / fixed | 0.459 | 0.337 | 4680.490 | ███████ 35.7% | 2.194x | 0.000e+00 | 0.000e+00 |
| 5 | 4 | blocktiling-1d | pass | register-tile-1d / fixed | 0.465 | 0.464 | 4623.789 | ███████ 35.2% | 2.167x | 0.000e+00 | 0.000e+00 |
| 6 | 5 | blocktiling-2d | pass | register-tile-2d / fixed | 0.649 | 0.645 | 3313.654 | █████ 25.3% | 1.553x | 0.000e+00 | 0.000e+00 |
| 7 | 3 | shared | pass | shared-memory / fixed | 0.723 | 0.723 | 2973.371 | █████ 22.7% | 1.394x | 0.000e+00 | 0.000e+00 |
| 8 | 1 | naive | pass | scalar / fixed | 1.008 | 1.007 | 2133.333 | ███ 16.3% | 1.000x | 0.000e+00 | 0.000e+00 |
| 9 | 2 | coalesced | pass | coalesced-scalar / fixed | 1.008 | 1.006 | 2132.250 | ███ 16.3% | 0.999x | 0.000e+00 | 0.000e+00 |
<!-- SGEMM_RESULTS_END -->
