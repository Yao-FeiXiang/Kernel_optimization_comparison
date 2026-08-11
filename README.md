# CUDA SGEMM 优化教学复刻

这个项目复刻 Simon Boehm 的 [CUDA Matmul 优化教程](https://siboehm.com/articles/22/CUDA-MMM)，从最直接的矩阵乘法开始，逐步加入合并访存、共享内存和线程级分块。项目重点不是替代 cuBLAS，而是让每一步优化都能单独运行、验证和对比。

所有实现遵循同一个单精度 SGEMM 语义：

```text
C = alpha * A * B + beta * C
```

当前实现覆盖教程的主要优化路径和 cuBLAS 对照：

| ID | 稳定名称 | 核心变化 |
|---:|:---|:---|
| 1 | `naive` | 每个线程计算一个 C 元素 |
| 2 | `coalesced` | 重新映射线程，使同一 warp 连续访问全局内存 |
| 3 | `shared` | 将 A/B 的 K 方向小块缓存到共享内存 |
| 4 | `blocktiling-1d` | 每个线程计算同一列中的 8 个结果 |
| 5 | `blocktiling-2d` | 每个线程在寄存器中累积一个 8×8 结果块 |
| 6 | `vectorized` | 转置共享内存中的 A，并用 `float4` 访问 GMEM |
| 9 | `autotuned` | 在正式计时前选择当前设备/尺寸上最快的预编译 tile |
| 10 | `warptiling` | 将 128×128 block tile 进一步分配到 64×64 warp tile |
| 0 | `cublas` | 使用相同输入、C0 恢复和计时规则的 NVIDIA cuBLAS 基线 |

所有自定义方法都支持非 tile 整数倍尺寸。Kernel 1–5 和 autotuned 候选直接进行边界保护；vectorized 与 warptiling 在不满足对齐/整除条件时报告 `scalar-fallback` 并调用 Kernel 5。这样批量正确性测试不会为了优化路径而牺牲尾块语义。

## 环境要求

- 支持 CUDA 的 NVIDIA GPU 和可工作的 NVIDIA 驱动。
- CUDA Toolkit（项目当前用 CUDA 13.3 编译验证）。
- Python 3.9 或更高版本，只使用标准库。
- 可选：CMake 3.24+ 和 Ninja。没有 CMake 时，Python 入口会直接调用 `nvcc`。
- 可选：cuBLAS。CMake 和 Python 直接构建都会自动检测；缺少可链接库时方法 0 仍可列出，但状态为 `unavailable`。

先确认工具和驱动：

```bash
nvcc --version
nvidia-smi
```

## 快速开始

`benchmark.py` 是推荐入口。首次运行会自动构建 `build/sgemm_bench`。

列出方法（这一步不需要打开 CUDA 设备）：

```bash
python3 benchmark.py --list
```

测试一个方法：

```bash
python3 benchmark.py --kernel 5 --m 1024 --n 1024 --k 1024
python3 benchmark.py --kernel blocktiling-2d --m 1024 --n 1024 --k 1024
```

批量测试当前所有方法：

```bash
python3 benchmark.py --all --m 1024 --n 1024 --k 1024
```

只运行自动调优方法。候选搜索发生在正式 CUDA event 计时之前：

```bash
python3 benchmark.py --tune --m 1024 --n 1024 --k 1024
```

当前预编译候选为 `64×64×8`、`128×64×8`、`64×128×8`、`128×128×8` 和 `128×128×16`，线程寄存器 tile 均为 8×8。选择结果会写入 JSON/Markdown 的 configuration 字段，并按设备与 M/N/K 缓存到本次进程结束。

运行小尺寸正确性检查，不进行性能计时：

```bash
python3 benchmark.py --all --check-only --m 37 --n 41 --k 29
```

性能测试前也可以启用 CPU 参考校验。CPU 参考实现是朴素三重循环，大矩阵校验会很慢，因此建议只对小尺寸使用 `--check`：

```bash
python3 benchmark.py --kernel shared --m 127 --n 131 --k 35 --check
```

强制重新构建或要求使用已有程序：

```bash
python3 benchmark.py --list --build
python3 benchmark.py --list --no-build --build-dir build
```

## 自动更新结果表

加入 `--update-results` 会把本次原生 JSON 记录写入 [RESULTS.md](RESULTS.md)：

```bash
python3 benchmark.py --all --m 1024 --n 1024 --k 1024 --warmup 5 --repeat 20 --update-results
```

也可以写到另一个 Markdown 文件：

```bash
python3 benchmark.py --kernel 5 --results-file local-results.md --update-results
```

更新键包含 GPU、compute capability、CUDA runtime、M/N/K、alpha/beta、warmup/repeat 和方法 ID。重复执行相同配置会更新原行；不同 GPU 或尺寸会保留为不同实验分组。表格同时展示运行变体、选中配置、相对 Naive 加速和 `% cuBLAS`；没有可用 cuBLAS 记录时对应列显示破折号。写入先生成临时文件再原子替换，失败时不会留下半张表。

## 计时与性能口径

- A、B、C0 由固定随机种子生成。
- 每个方法、每次 warmup 和每个计时样本执行前，都把同一份设备端 C0 恢复到 C。
- 恢复 C0、主机与设备间复制、CPU 校验和进程启动都不计入 kernel latency。
- 使用同一 CUDA stream 上的 CUDA event 计时，报告多次样本的中位数，同时保存最小值。
- 性能统一按 `2*M*N*K + 2*M*N` 次浮点操作计算；后一个 `2*M*N` 对应 alpha/beta 缩放和最终累加。
- 正确性使用 `abs_error <= atol + rtol*abs(reference)`；默认 `atol=1e-4*K`、`rtol=1e-4`，可以用 `--atol` 和 `--rtol` 覆盖。

不同 GPU、CUDA 版本、功耗限制、温度和时钟状态下的结果不能直接视为同一基准。正式对比前请关闭其他 GPU 负载，并在同一机器上重复运行。

## CMake 构建

安装 CMake 后可以使用正式构建入口：

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
./build/sgemm_bench --list
```

默认编译 compute capability 7.5、8.0、8.6、8.9 和 9.0。可以针对本机覆盖，缩短构建时间：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
```

如需显式关闭 cuBLAS：

```bash
cmake -S . -B build -DSGEMM_ENABLE_CUBLAS=OFF
```

## 自动化测试

运行全部 Python 测试：

```bash
python3 -m unittest discover -s tests -v
```

测试分三层：

- 纯 CPU/Python：CPU 参考公式、结果解析、Markdown 幂等更新和命令构造。
- CUDA 编译但不运行：注册表和 `--list`，没有 GPU 驱动时也能执行。
- CUDA 集成：在驱动和设备可用时，对包含尾块的小矩阵逐方法比较 CPU 参考；设备不可用时明确显示 `skipped`。

单独运行原生 CPU 测试：

```bash
c++ -std=c++17 -Iinclude tests/reference_test.cpp src/reference.cpp -o /tmp/sgemm_reference_test
/tmp/sgemm_reference_test
```

## 目录结构

```text
include/sgemm/benchmark.hpp   公共问题、选项、误差和结果类型
include/sgemm/kernels.cuh     kernel 注册与统一 launch 接口
src/kernels/                  每个优化阶段的独立 CUDA 实现
src/reference.cpp             CPU 参考计算和误差统计
src/benchmark.cu              原生 CLI、显存管理、校验和计时
sgemm_tools/                  Python 构建、运行和结果表工具
benchmark.py                  用户入口
tests/                        CPU、CLI 和条件 CUDA 测试
RESULTS.md                    自动维护的本机结果表
```

仓库根目录原有的单文件实现和历史编译产物已经清理；正式入口是上面的模块化实现。

## 添加新的 kernel

1. 在 `src/kernels/` 新建实现，并提供与 `LaunchFn` 一致的主机 launch wrapper。
2. 在 `include/sgemm/kernels.cuh` 声明 wrapper。
3. 在 `src/kernels/registry.cu` 注册唯一 ID、稳定名称和说明。
4. 把源文件加入 `CMakeLists.txt` 与 `sgemm_tools/runner.py`。
5. 先扩展 `tests/registry_test.cu`，再实现代码；有 GPU 时增加一个非 tile 整数倍尺寸的正确性用例。

benchmark runner 不包含某个 kernel 的索引或 block 配置细节，新增方法不需要复制显存、校验或计时代码。

## 常见问题

`cudaGetDeviceCount: CUDA driver version is insufficient` 或 `nvidia-smi` 无法通信：CUDA Toolkit 只提供编译器，仍需正确安装并加载与之兼容的 NVIDIA 驱动。

`cmake: command not found`：直接使用 `python3 benchmark.py ...`，它会回退到 `nvcc`；或安装 CMake 3.24+。

大矩阵使用 `--check` 很慢：CPU 参考是教学用朴素实现。先用 `--check-only --m 37 --n 41 --k 29` 验证尾块，再单独运行大尺寸性能测试。

批量模式某个方法失败：runner 会继续执行其他方法，最后返回非零状态，并在文本/JSON 结果中记录失败方法和 CUDA 错误。构建期不可用的 cuBLAS 记为 `unavailable`，不会把其余方法的成功批量运行变成失败。
