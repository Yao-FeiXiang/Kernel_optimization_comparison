# CUDA SGEMM 教学复刻与基准框架设计

## 背景与目标

本项目复刻 Simon Boehm 的 CUDA SGEMM 优化教程，并将当前集中在
`SGEMM.cu` 中的 kernel、计时和展示逻辑重构为可验证、可扩展的教学工程。

工作分为两个连续阶段：

1. 阶段 A：整理并修正 Kernel 1–4，实现 Kernel 5（2D Block Tiling），建立正确性校验、单项测试、批量测试、结构化输出和 Markdown 结果更新流程。
2. 阶段 B：在相同框架中继续实现向量化访存、候选参数自动调优、Warp Tiling 和 cuBLAS 基线。

最终用户可以用一条命令测试某个实现，也可以批量比较全部实现，并选择是否将本机结果幂等地更新到 `RESULTS.md`。

## 范围

### 阶段 A

- Kernel 1：Naive SGEMM。
- Kernel 2：Global Memory Coalescing。
- Kernel 3：Shared Memory Cache Blocking。
- Kernel 4：1D Block Tiling。
- Kernel 5：2D Block Tiling。
- 支持运行时设置 `M`、`N`、`K`、`alpha`、`beta`、warmup 次数和计时次数。
- 支持按编号或稳定名称选择单个 kernel，以及运行阶段 A 的全部 kernel。
- 每种实现使用相同的初始 A、B、C 数据；每次正式测试前恢复 C，保证 `beta*C` 语义和横向对比公平。
- 小矩阵使用 CPU SGEMM 参考实现校验；CUDA kernel 使用绝对误差与相对误差组合容差。
- benchmark 输出人类可读摘要，并提供一行 JSON 结构化记录供 Python 驱动解析。
- Python 驱动负责构建、执行、失败汇总和更新 `RESULTS.md`。
- 补齐 README、`.gitignore` 和不依赖 GPU 的自动化单元测试。

### 阶段 B

- Kernel 6：转置共享内存中的 A，并用 `float4` 向量化适合的 GMEM 访存；不满足对齐或整除条件时使用安全标量路径。
- Autotuning：预编译一组合法的 tile 参数候选，在当前设备和矩阵尺寸上运行并报告最快候选，不在运行时生成或修改源码。
- Warp Tiling：增加 warp 级结果块映射和寄存器复用实现。
- cuBLAS：加入可选基线，用相同 A、B、C、alpha、beta、warmup 和计时规则测量；构建时没有 cuBLAS 则明确报告该方法不可用，不影响自定义 kernel。
- `--all` 扩展为运行全部已构建方法，结果表记录各方法相对 Naive 与相对 cuBLAS 的性能。

阶段 B 不包含 Tensor Core、TF32/BF16、CUTLASS 集成、Nsight 自动分析或跨机器聚合服务。

## 架构与模块边界

### CUDA 层

计划采用以下职责边界，具体文件可在实施计划中按编译约束微调：

- `include/sgemm/types.cuh`：问题尺寸、运行选项、kernel 元数据和结果结构。
- `include/sgemm/kernels.cuh`：统一的 kernel 注册与 launch 接口。
- `src/kernels/*.cu` / `*.cuh`：各优化阶段的实现；每个模块只负责 kernel 和 launch 配置。
- `src/reference.cpp`：CPU 参考计算和误差统计。
- `src/benchmark.cu`：命令行解析、CUDA 资源生命周期、输入初始化、正确性检查、计时和输出。

所有 kernel 对外表现为同一个 SGEMM 语义：

```text
C = alpha * A * B + beta * C
```

kernel 注册项至少包含稳定名称、教程编号、说明、可用性检查和 launch 函数。benchmark 不直接包含某个 kernel 的索引细节。

### Python 自动化层

- `benchmark.py`：面向用户的入口，负责构建和运行选择器。
- `tools/results.py`：解析 JSON 结果、计算展示字段并生成 Markdown 表格。
- `tests/`：验证参数选择、输出解析、失败处理及结果文件的幂等更新。

Python 只编排测试，不参与 CUDA 计时。CUDA event 记录同一 stream 上的 kernel 时间，避免将主机到设备的数据传输和进程启动计入 kernel latency。

### 构建

使用 CMake 作为正式构建入口，自动检测 CUDA Toolkit，并在阶段 B 检测 cuBLAS。`benchmark.py` 可以调用 CMake 配置与构建，使新用户不必手工拼接 `nvcc` 参数。构建产物统一进入 `build/`。

## 用户接口

CUDA 可执行文件提供底层接口：

```bash
./build/sgemm_bench --list
./build/sgemm_bench --kernel 2d-blocktiling --m 1024 --n 1024 --k 1024
./build/sgemm_bench --all --m 1024 --n 1024 --k 1024
```

常用入口是 Python 驱动：

```bash
python benchmark.py --kernel 5 --m 1024 --n 1024 --k 1024 --update-results
python benchmark.py --all --m 1024 --n 1024 --k 1024 --update-results
python benchmark.py --all --check-only
```

编号和名称都可选择方法。`--check-only` 使用一组包含非 tile 整数倍尺寸的小矩阵，执行正确性检查而不做长时间性能统计。默认 benchmark 尺寸和次数选择能在普通开发 GPU 上完成的保守值，用户可显式提高到 4096 或 8192。

## 数据流与结果文件

一次运行按如下顺序执行：

1. 解析参数并选择 kernel。
2. 固定随机种子生成主机输入 A、B、C0。
3. 对需要检查的尺寸计算 CPU 参考结果。
4. 将 A、B、C0 复制到 GPU。
5. 对每个 kernel 恢复 C0，执行正确性运行并复制结果进行误差检查。
6. 对每个 kernel 再次恢复 C0，执行 warmup，然后用 CUDA events 测量多次迭代。
7. 输出状态、延迟、GFLOP/s、相对性能、误差和设备信息。
8. Python 收集成功和失败记录；仅在用户指定 `--update-results` 时改写结果文件。

`RESULTS.md` 包含：

- 自动生成声明和复现实验的命令。
- GPU 名称、compute capability、CUDA 版本、矩阵尺寸、alpha/beta、warmup/repeat 和时间戳。
- 方法、状态、延迟、GFLOP/s、相对 Naive、相对 cuBLAS、最大绝对/相对误差表格。

更新以“实验配置 + 方法”为键进行 upsert。同一配置重复运行会替换对应行，不会无限追加重复数据；不同 GPU 或尺寸保留为不同结果分组。Markdown 写入采用先生成完整内容再原子替换的方式，失败时保留原文件。

## 正确性、错误处理与计时规则

- 所有 CUDA API、kernel launch 和同步调用都检查错误；错误包含 kernel 名称和 CUDA 错误文本。
- 非法尺寸、未知 kernel、不可用方法或不合法 autotuning 参数以非零退出码结束。
- 单个方法失败时，批量模式继续运行其余方法，最后整体返回非零并在摘要中列出失败项。
- 浮点校验使用 `abs_error <= atol + rtol * abs(reference)`；默认容差根据 K 的累加误差选取，并允许命令行覆盖。
- 性能统计使用多次样本的中位数作为主延迟，同时在结构化结果中保留最小值和样本数。
- GFLOP/s 统一按 `2*M*N*K + 2*M*N` 计算，后项对应 alpha 与 beta 路径的缩放/累加；README 明确该口径。
- 若性能测试启用校验，校验运行与计时运行分开，设备到主机复制不计入 latency。

## 测试策略

### CPU 可运行测试

- Python CLI 的 kernel 编号/名称选择。
- JSON 行解析，包括缺字段、失败状态和未知字段兼容。
- Markdown 新建、upsert、保留不同配置、重复更新幂等性。
- GFLOP/s 与相对性能派生字段。

### CUDA 条件测试

- 在检测到 `nvcc` 和可用 NVIDIA GPU 时构建并执行。
- 对每个 kernel 测试常规小矩阵和至少一个非 tile 整数倍矩阵。
- 覆盖非零 beta，验证每个方法确实从相同 C0 开始。
- 阶段 B 对向量化路径和尾部标量路径分别覆盖。
- cuBLAS 可用时，以 cuBLAS 结果参与交叉校验。

无 CUDA 的环境跳过 CUDA 集成测试，但 Python 测试必须通过。README 会明确“跳过”与“通过”的区别。

## 迁移与兼容性

- 现有 `SGEMM.cu` 和 `script.py` 先作为迁移来源；新框架验证通过后，不再将其作为正式入口。
- 不修改或回退用户在这两个文件上的未提交更改。若最终删除旧入口会造成数据丢失，则保留为 `legacy/` 或在得到用户授权后再清理。
- 已跟踪的 `SGEMM.exe`、`SGEMM.exp` 和 `SGEMM.lib` 不会仅因新增 `.gitignore` 自动删除；本轮先停止产生新的构建垃圾，是否从 Git 索引移除作为单独、明确的清理动作处理。
- `.gitignore` 覆盖 `build/`、CMake/Ninja 产物、CUDA 对象和链接产物、Python 缓存、测试缓存及本地 benchmark 临时文件；`RESULTS.md` 保持跟踪。

## README 内容

README 使用中文为主，包含：

- 项目目标和与原教程的阶段对应关系。
- 每个 kernel 的核心优化点和当前实现状态。
- CUDA、CMake、Python 与可选 cuBLAS 依赖。
- 从零构建、列出方法、单项运行、批量运行、正确性检查和更新结果表的命令。
- benchmark 计时与 GFLOP/s 口径。
- 目录结构、添加新 kernel 的最短步骤及常见故障排查。
- 不同 GPU、CUDA 版本和频率状态下的结果不可直接等同比较的说明。

## 完成标准

阶段 A 完成需满足：

- Kernel 1–5 可由统一入口选择并通过小矩阵正确性测试。
- 单项和批量 benchmark 均能产生结构化记录。
- `--update-results` 可重复运行且不会产生重复表格行。
- CPU 层自动化测试通过；当前机器具备 CUDA 时，CUDA 构建和集成测试通过。
- README 和 `.gitignore` 与实际命令及产物一致。

阶段 B 完成需满足：

- 向量化、autotuning、Warp Tiling 和可选 cuBLAS 均接入同一注册、校验和 benchmark 接口。
- 向量化尾部路径和候选参数合法性有测试覆盖。
- `--all` 能批量对比所有可用方法，`RESULTS.md` 能展示相对 Naive 和相对 cuBLAS 数据。
- 所有适用的自动化验证均有新鲜的成功记录。
