# A100 教程一致性与配置调优设计

## 目标

在保留当前统一 CLI、自动结果表和任意矩阵尺寸正确性的前提下，让 Kernel 1–10 更忠实地复刻 Simon Boehm 教程的优化阶段，并在 NVIDIA A100、`M=N=K=4096` 下得到接近教程官方 A100 代码的性能曲线。

本轮不追求逐行复制上游代码，也不把 cuBLAS 当作必须完全追平的目标。验收重点是每个教学阶段确实引入其名称对应的优化，并消除当前 Kernel 1/2、4/5、6/9 之间不合理的性能关系。

## 方案选择

采用“对齐尺寸教程快速路径 + 任意尺寸安全回退”。

- 只修改常量无法修正 Kernel 1 已经合并访存、Kernel 9 缺少向量加载等结构问题。
- 直接替换为上游实现会丢失尾块处理、统一 stream、C0 恢复和 JSON/Markdown 集成。
- 双路径方案让 `4096³` 等教程尺寸使用编译期参数、无边界分支和 `float4`，尾块尺寸继续走经过测试的安全实现。

## Kernel 1 与 Kernel 2

Kernel 1 按教程恢复故意不合并的线程映射：同一 warp 中连续的 `threadIdx.x` 映射到不同输出行，使 A 的访问跨行。Kernel 2 保持一维 1024 线程 block，把连续线程映射到同一行的连续列。

两个方法仍计算相同 SGEMM 语义并支持任意正尺寸。`4096³` 验收要求 Kernel 2 的 GFLOP/s 至少为 Kernel 1 的 1.5 倍；该阈值用于发现映射退化，不作为跨机器单元测试。

## Kernel 5

保留现有 `64×64×8` 边界安全实现作为尾块回退，并新增教程大矩阵快速模板：

- `BM=128`
- `BN=128`
- `BK=8`
- `TM=8`
- `TN=8`
- 256 threads/block

当 M/N 是 128 的整数倍且 K 是 8 的整数倍时，启动无输入/输出边界判断的快速模板，并使用 `__launch_bounds__(256)`。其他尺寸继续使用安全实现。JSON 的 implementation variant 分别报告 `register-tile-2d-fast` 和 `register-tile-2d-tail-safe`。

## Kernel 9

将自动调优候选从标量 2D blocktiling 升级为 Kernel 6 风格：

- A/B 从 GMEM 使用 `float4` 加载。
- A 写入共享内存时转置。
- C 使用 `float4` 读写。
- 候选模板使用编译期 BM/BN/BK/TM/TN 和 `__launch_bounds__`。

候选至少包含：

- A100 教程配置 `64×64×16, TM=TN=4`
- `64×64×8, TM=TN=8`
- `128×64×8, TM=TN=8`
- `64×128×8, TM=TN=8`
- `128×128×8, TM=TN=8`
- A6000 教程配置 `128×128×16, TM=TN=8`

调优仍在正式 benchmark 前进行，每个候选预热一次、测量三次并取中位数。缓存键保持设备和 M/N/K。对不满足候选对齐条件的输入不进行伪调优，直接选择 Kernel 5 安全回退并报告 `tail-safe-fallback`。

## Kernel 10

把 warptiling 改为两个编译期变体，在 prepare 阶段根据 compute capability 选择：

- A100（SM80）：`BM=64, BN=128, BK=16, WM=32, WN=64, WNITER=1, TM=4, TN=4, 128 threads`
- 其他设备：`BM=128, BN=128, BK=16, WM=64, WN=64, WNITER=4, TM=8, TN=4, 128 threads`

两个快速模板都使用 `float4`、转置 A、`__launch_bounds__(128)`，并保持教程的 block/warp/thread 三层分块。无法满足 tile 或 16 字节对齐条件时调用 Kernel 5 安全回退。

prepare 返回所选配置，使 `RESULTS.md` 能区分 `A100-SM80`、通用变体和尾块回退。设备选择只决定 kernel 配置，不扩展为空闲 GPU 自动调度。

## 正确性与接口

- 方法 ID、稳定名称、CLI 参数和 JSON schema 不变。
- 所有方法继续实现 `C = alpha * A * B + beta * C`。
- `--all --check-only --m 37 --n 41 --k 29` 必须继续通过。
- 对齐快速路径增加至少一个 `128×128×64` 正确性用例。
- Kernel 10 注册表从 fixed prepare 改为专用 prepare；其他调用接口不变。
- 上游代码为 MIT 许可证；本轮按其公开参数和算法结构重新实现，不复制无关 runner。

## 性能反馈环

性能验收在当前较空闲的 A100 上串行运行，不写入单元测试硬阈值：

```bash
CUDA_VISIBLE_DEVICES=0 python3 benchmark.py --all \
  --m 4096 --n 4096 --k 4096 --warmup 5 --repeat 50
```

期望关系：

- Kernel 2 > 1.5 × Kernel 1
- Kernel 5 > 1.1 × Kernel 4
- Kernel 9 ≥ 0.9 × Kernel 6
- Kernel 10 ≥ 0.85 × cuBLAS
- Kernel 10 与同机教程官方 A100 参考值 15.99 TFLOP/s 的差距不超过 10%，前提是 GPU 没有明显竞争负载

若共享服务器负载导致中位数和最小值明显分裂，先记录环境证据并重跑，不为了通过阈值修改算法。

## 结果与文档

README 增加“教程可比基准”命令，明确网页顶部是 A6000/4096，A100 应使用设备专用配置。最终通过 `--update-results` 把 A100/4096 作为新实验组写入 `RESULTS.md`，保留已提交的 A100/1024 基线。

