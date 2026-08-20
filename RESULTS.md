# SGEMM Benchmark Results

该文件由 `benchmark.py --update-results` 维护。不要手工编辑下面两个标记之间的内容；相同 GPU、CUDA、矩阵尺寸和测试参数下的同一方法会被替换，不会重复追加。

<!-- SGEMM_RESULTS_BEGIN -->
## NVIDIA A100-PCIE-40GB — M=1024, N=1024, K=1024

- Compute capability: `8.0`; CUDA runtime: `13.3`; alpha=0.8; beta=0.2
- Warmup: 5; timed samples: 20; updated: `2026-08-20T02:18:00Z`
- Reproduce: `python3 benchmark.py --all --m 1024 --n 1024 --k 1024 --update-results`

<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":2170.837576978,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"scalar","k":1024,"latency_ms":0.99020803,"m":1024,"max_abs_error":0.0,"max_rel_error":0.0,"method":"naive","method_id":1,"min_latency_ms":0.988160014,"n":1024,"repeat":20,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T02:18:00Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":2171.96062615,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"coalesced-scalar","k":1024,"latency_ms":0.989696026,"m":1024,"max_abs_error":0.0,"max_rel_error":0.0,"method":"coalesced","method_id":2,"min_latency_ms":0.988160014,"n":1024,"repeat":20,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T02:18:00Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":3024.783915885,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"shared-memory","k":1024,"latency_ms":0.710655987,"m":1024,"max_abs_error":0.0,"max_rel_error":0.0,"method":"shared","method_id":3,"min_latency_ms":0.709631979,"n":1024,"repeat":20,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T02:18:00Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":4706.726553266,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"register-tile-1d","k":1024,"latency_ms":0.456703991,"m":1024,"max_abs_error":0.0,"max_rel_error":0.0,"method":"blocktiling-1d","method_id":4,"min_latency_ms":0.455680013,"n":1024,"repeat":20,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T02:18:00Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":3372.208698819,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"register-tile-2d","k":1024,"latency_ms":0.637440026,"m":1024,"max_abs_error":0.0,"max_rel_error":0.0,"method":"blocktiling-2d","method_id":5,"min_latency_ms":0.635904014,"n":1024,"repeat":20,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T02:18:00Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":4760.090583664,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"float4","k":1024,"latency_ms":0.451584011,"m":1024,"max_abs_error":0.0,"max_rel_error":0.0,"method":"vectorized","method_id":6,"min_latency_ms":0.450560004,"n":1024,"repeat":20,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T02:18:00Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"BM128_BN128_BK16_TM8_TN8","cuda_runtime":"13.3","gflops":3858.823461737,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"autotuned-tile","k":1024,"latency_ms":0.55705601,"m":1024,"max_abs_error":0.0,"max_rel_error":0.0,"method":"autotuned","method_id":9,"min_latency_ms":0.555007994,"n":1024,"repeat":20,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T02:18:00Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"fixed","cuda_runtime":"13.3","gflops":4056.425258892,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"warp-tiled","k":1024,"latency_ms":0.529919982,"m":1024,"max_abs_error":0.0,"max_rel_error":0.0,"method":"warptiling","method_id":10,"min_latency_ms":0.52838397,"n":1024,"repeat":20,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T02:18:00Z","warmup":5} -->
<!-- record:{"alpha":0.800000012,"available":true,"beta":0.200000003,"compute_capability":"8.0","configuration":"cuBLAS","cuda_runtime":"13.3","gflops":16085.823241873,"gpu":"NVIDIA A100-PCIE-40GB","implementation_variant":"cublas-sgemm","k":1024,"latency_ms":0.133632004,"m":1024,"max_abs_error":0.0,"max_rel_error":0.0,"method":"cublas","method_id":0,"min_latency_ms":0.132095993,"n":1024,"repeat":20,"schema_version":1,"status":"pass","timestamp_utc":"2026-08-20T02:18:00Z","warmup":5} -->

| Rank | ID | Method | Status | Variant / configuration | Median ms | Min ms | GFLOP/s | Performance vs cuBLAS | vs Naive | Max abs error | Max rel error |
|---:|---:|:---|:---:|:---|---:|---:|---:|:---|---:|---:|---:|
| 1 | 0 | cublas | pass | cublas-sgemm / cuBLAS | 0.134 | 0.132 | 16085.823 | ████████████████████ 100.0% | 7.410x | 0.000e+00 | 0.000e+00 |
| 2 | 6 | vectorized | pass | float4 / fixed | 0.452 | 0.451 | 4760.091 | ██████ 29.6% | 2.193x | 0.000e+00 | 0.000e+00 |
| 3 | 4 | blocktiling-1d | pass | register-tile-1d / fixed | 0.457 | 0.456 | 4706.727 | ██████ 29.3% | 2.168x | 0.000e+00 | 0.000e+00 |
| 4 | 10 | warptiling | pass | warp-tiled / fixed | 0.530 | 0.528 | 4056.425 | █████ 25.2% | 1.869x | 0.000e+00 | 0.000e+00 |
| 5 | 9 | autotuned | pass | autotuned-tile / BM128_BN128_BK16_TM8_TN8 | 0.557 | 0.555 | 3858.823 | █████ 24.0% | 1.778x | 0.000e+00 | 0.000e+00 |
| 6 | 5 | blocktiling-2d | pass | register-tile-2d / fixed | 0.637 | 0.636 | 3372.209 | ████ 21.0% | 1.553x | 0.000e+00 | 0.000e+00 |
| 7 | 3 | shared | pass | shared-memory / fixed | 0.711 | 0.710 | 3024.784 | ████ 18.8% | 1.393x | 0.000e+00 | 0.000e+00 |
| 8 | 2 | coalesced | pass | coalesced-scalar / fixed | 0.990 | 0.988 | 2171.961 | ███ 13.5% | 1.001x | 0.000e+00 | 0.000e+00 |
| 9 | 1 | naive | pass | scalar / fixed | 0.990 | 0.988 | 2170.838 | ███ 13.5% | 1.000x | 0.000e+00 | 0.000e+00 |
<!-- SGEMM_RESULTS_END -->
