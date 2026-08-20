# CUDA SGEMM 优化教学复刻

这个项目复刻 Simon Boehm 的 [CUDA Matmul 优化教程](https://siboehm.com/articles/22/CUDA-MMM)，用统一的构建、正确性检查和计时流程，对比从朴素矩阵乘法到 warp tiling、自动调优以及 cuBLAS 的性能。

需要 NVIDIA GPU、可用的 CUDA 驱动与 Toolkit，以及 Python 3.9+。CMake/Ninja 是可选项；没有 CMake 时会直接使用 `nvcc`。

## 直接运行

日常实验只需要这一条命令：

```bash
python3 benchmark.py --all --update-results
```

它会测试全部方法，并自动更新 [RESULTS.md](RESULTS.md) 中的可视化表格。默认参数按教程性能实验设置为 4096×4096×4096、`warmup=5`、`repeat=50`；在当前服务器上默认使用可见的 GPU0。

首次运行会自动构建。以后源码、头文件或 CMake 配置有变化时也会自动重新构建，避免误用旧程序。`--build` 可以强制重建；`--no-build` 只允许使用存在且未过期的程序。

## 常用操作

测试单个方法、只运行自动调优，或列出全部方法：

```bash
python3 benchmark.py --kernel 10 --update-results
python3 benchmark.py --tune --update-results
python3 benchmark.py --list
```

### 手动覆盖参数

只有需要特殊尺寸或测量次数时才传参数；显式参数会覆盖默认值。`--results-file` 可以把结果写入另一个 Markdown 文件。

```bash
python3 benchmark.py --all --m 1024 --n 1024 --k 1024 --warmup 5 --repeat 20 --update-results
python3 benchmark.py --kernel 5 --results-file local-results.md --update-results
```

正式计时前，建议用不规则小尺寸检查尾块正确性。`--check-only` 只校验、不计时；不要对 4096 方阵启用朴素 CPU 参考校验。

```bash
python3 benchmark.py --all --check-only --m 37 --n 41 --k 29
```

如果换到其他架构，可在强制构建时设置环境变量，例如 `SGEMM_CUDA_ARCH=sm_86 python3 benchmark.py --list --build`；当前 A100 默认是 `sm_80`。

## 已实现方法

| ID | 名称 | 主要优化 |
|---:|:---|:---|
| 1 | `naive` | 每线程计算一个输出元素 |
| 2 | `coalesced` | 合并全局内存访问 |
| 3 | `shared` | 使用共享内存分块 |
| 4 | `blocktiling-1d` | 一维线程级分块 |
| 5 | `blocktiling-2d` | 二维寄存器分块 |
| 6 | `vectorized` | 转置共享内存并使用 `float4` |
| 9 | `autotuned` | 运行预编译候选并选择当前设备与尺寸的最快配置 |
| 10 | `warptiling` | 教程的 warp tiling；A100 使用 SM80 专用参数 |
| 0 | `cublas` | 同一输入和计时口径下的 cuBLAS 基线 |

方法 5 在对齐尺寸上使用 `register-tile-2d-fast`，不规则尺寸使用 `tail-safe`。自动调优与 warp tiling 在不能安全使用向量化快核时也会回退到安全路径；实际变体和配置会写入结果表。

自动调优候选包含教程的 A100 `64×64×16 / TM=TN=4`、A6000 `128×128×16 / TM=TN=8`，以及多组 8 深度 tile。候选搜索发生在正式测量前，选择结果按设备和 M/N/K 在本次进程中缓存。

## 结果表

带 `--update-results` 的每次实验都会更新对应的 GPU、CUDA 版本、矩阵尺寸、alpha/beta、warmup/repeat 和方法记录；相同配置覆盖旧行，不同配置保留为独立实验组。

每个实验组按 GFLOP/s 自动排序，并用 `Performance vs cuBLAS` 条形格展示相对 cuBLAS 性能，同时保留相对 Naive 加速、运行变体、调优配置和误差。文件通过临时文件原子替换，更新失败不会留下不完整表格。

## 统一实验口径

所有方法计算：

```text
C = alpha * A * B + beta * C
```

- A、B、C0 使用同一固定随机种子；每次 warmup 和计时样本前恢复相同的 C0。
- 使用同一 CUDA stream 上的 CUDA event 计时，报告中位延迟并保存最小值。
- GFLOP/s 按 `2*M*N*K + 2*M*N` 次浮点操作计算。
- 默认误差阈值为 `atol=1e-4*K`、`rtol=1e-4`，可用 `--atol`、`--rtol` 覆盖。
- GPU 型号、目标架构、功耗、温度、CUDA 版本和竞争负载都会影响结果；与网上教程比较时应更关注优化趋势及相对本机 cuBLAS 的比例。

## 自动化测试

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖 Python 结果处理、命令构造、原生注册表，以及有 GPU 时各方法在对齐和尾块尺寸上的正确性。没有可用 GPU 时，CUDA 运行测试会明确显示为跳过。

项目结构中，`src/kernels/` 保存各阶段 CUDA 实现，`src/benchmark.cu` 负责统一显存、校验与计时，`sgemm_tools/` 负责 Python 构建和结果表更新，`tests/` 保存自动化测试。
