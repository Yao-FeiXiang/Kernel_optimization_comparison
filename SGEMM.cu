#include <iostream>
using namespace std;
#include <vector>

#define M 8192
#define N 8192
#define K 8192
#define alpha 0.8
#define beta 0.2

#define CEIL_DIV(m, n) (((m) + (n) - 1) / (n))

#define BM 64
#define BN 64
#define BK 8
#define TM 8

// TODO C=alpha*A*B+beta*C

// 获取不同架构下每个SM的FP32 CUDA Core数量（辅助推算峰值）
int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.major;
    int min = devProp.minor;
    switch (mp)
    {
    case 6: // Pascal
        cores = (min == 1) ? 128 : 64;
        break;
    case 7: // Volta & Turing
        cores = 64;
        break;
    case 8: // Ampere & Ada
        cores = (min == 0) ? 64 : 128;
        break;
    case 9: // Hopper
        cores = 128;
        break;
    default:
        cores = 128;
        break;
    }
    return cores;
}

__global__ void NaiveSGEMM(float *A, float *B, float *C, int m, int n, int k)
{
    // 在当前 Block 内部的一维线性 ID：local_thread_id = threadIdx.y * blockDim.x + threadIdx.x
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < m && y < n)
    {
        float sum = 0;
        for (int i = 0; i < k; i++)
        {
            sum += A[x * k + i] * B[i * n + y];
        }
        C[x * n + y] = alpha * sum + beta * C[x * n + y];
    }
}

__global__ void GLOBAL_MEM_SGEMM(float *A, float *B, float *C, int m, int n, int k)
{
    const unsigned int x = blockIdx.x * 32 + (threadIdx.x / 32);
    const unsigned int y = blockIdx.y * 32 + (threadIdx.x % 32);
    if (x < m && y < n)
    {
        float sum = 0;
        for (int i = 0; i < k; i++)
        {
            sum += A[x * k + i] * B[i * n + y];
        }
        C[x * n + y] = alpha * sum + beta * C[x * n + y];
    }
}

__global__ void SHARED_MEM_SGEMM(float *A, float *B, float *C, int m, int n, int k)
{
    extern __shared__ float shared_mem[];
    float *tile_A = shared_mem;           // 32*32
    float *tile_B = shared_mem + 32 * 32; // 32*32

    // int block_row = blockIdx.x;
    // int block_col = blockIdx.y;
    // int local_row = threadIdx.x;
    // int local_col = threadIdx.y;
    // 1. 加载A和B时没有合并访问  2. 写入sahred momory时严重bank conflict

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    const unsigned int x = block_row * 32 + local_row;
    const unsigned int y = block_col * 32 + local_col;

    A += block_row * 32 * k;                  // A 的行偏移
    B += block_col * 32;                      // B 的列偏移
    C += block_row * 32 * n + block_col * 32; // C 的行列偏移

    float sum = 0;

    for (int tile_idx = 0; tile_idx < CEIL_DIV(k, 32); tile_idx++)
    {
        tile_A[local_row * 32 + local_col] = (x < m && tile_idx * 32 + local_col < k) ? A[local_row * k + local_col] : 0.0f;
        tile_B[local_row * 32 + local_col] = (y < n && tile_idx * 32 + local_row < k) ? B[local_row * n + local_col] : 0.0f;
        __syncthreads();

        A += 32;     // A 的列偏移
        B += 32 * n; // B 的行偏移

        if (x < m && y < n)
        {
            for (int i = 0; i < 32; i++)
            {
                sum += tile_A[local_row * 32 + i] * tile_B[i * 32 + local_col];
            }
        }
        __syncthreads();
    }
    if (x < m && y < n)
    {
        C[local_row * n + local_col] = alpha * sum + beta * C[local_row * n + local_col];
    }
}

__global__ void calculating_more_res_per_thread_1D_SGEMM(float *A, float *B, float *C, int m, int n, int k)
{
    extern __shared__ float shared_mem[];
    float *tile_A = shared_mem;           // BM*BK
    float *tile_B = shared_mem + BM * BK; // BK*BN

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    int c_row_start = block_row * BM + local_row * TM;
    int c_col = block_col * BN + local_col;
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

    int a_load_row = thread_id / BK;
    int a_load_col = thread_id % BK;
    int b_load_row = thread_id / BN;
    int b_load_col = thread_id % BN;

    A += block_row * BM * k; // A 的行偏移
    B += block_col * BN;     // B 的列偏移

    float threadResults[TM] = {0.0f};
    for (int tile_idx = 0; tile_idx < CEIL_DIV(k, BK); tile_idx++)
    {
        tile_A[a_load_row * BK + a_load_col] = (c_row_start + a_load_row < m && tile_idx * BK + a_load_col < k) ? A[a_load_row * k + a_load_col] : 0.0f;

        tile_B[b_load_row * BN + b_load_col] = (block_col * BN + b_load_col < n && tile_idx * BK + b_load_row < k) ? B[b_load_row * n + b_load_col] : 0.0f;

        __syncthreads();
        A += BK;     // A 的列偏移
        B += BK * n; // B 的行偏移
        if (c_row_start < m && c_col < n)
        {
            for (int i = 0; i < BK; i++)
            {
                float b_val = tile_B[i * BN + local_col];
                for (int j = 0; j < TM; j++)
                {
                    threadResults[j] += tile_A[(local_row * TM + j) * BK + i] * b_val;
                }
            }
        }
        __syncthreads();
    }
    for (int j = 0; j < TM; j++)
    {
        int c_row = c_row_start + j;
        if (c_row < m && c_col < n)
        {
            C[c_row * n + c_col] = alpha * threadResults[j] + beta * C[c_row * n + c_col];
        }
    }
}

int main()
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    int coresPerSM = getSPcores(deviceProp);
    // 峰值 GFLOPS = SM数量 * 单SM核心数 * 频率(Hz) * 2(FMA)
    double peak_gflops = (double)deviceProp.multiProcessorCount * coresPerSM * (deviceProp.clockRate * 1e3) * 2.0 / 1e9;

    cout << "===== GPU Information =====" << endl;
    cout << "Device: " << deviceProp.name << endl;
    cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << endl;

    // 查看共享内存大小 (单位转换成 KB)
    cout << "Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024.0 << " KB" << endl;
    cout << "Shared Memory per SM: " << deviceProp.sharedMemPerMultiprocessor / 1024.0 << " KB" << endl;

    cout << "Theoretical Peak FP32 GFLOPS: " << peak_gflops << " GFLOPS" << endl;
    cout << "===========================" << endl
         << endl;

    vector<float> A(M * K);
    vector<float> B(K * N);
    vector<float> C(M * N);

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            A[i * K + j] = rand() % 100;
        }
    }
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < N; j++)
        {
            B[i * N + j] = rand() % 100;
        }
    }
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i * N + j] = 0;
        }
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(M, blockDim.x), CEIL_DIV(N, blockDim.y));

    NaiveSGEMM<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // 1. Naive SGEMM
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    NaiveSGEMM<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float latency_ms = 0;
    cudaEventElapsedTime(&latency_ms, start, stop);

    double operations = 2.0 * M * N * K + M * N;                         // 乘加操作数 + C矩阵的缩放和累加操作数
    double gflops_achieved = (operations / (latency_ms / 1000.0)) / 1e9; // 转换为 GFLOPS
    double percent_peak = (gflops_achieved / peak_gflops) * 100.0;

    cout << "===== Naive SGEMM Results =====" << endl;
    cout << "Matrix Size: M=" << M << ", N=" << N << ", K=" << K << endl;
    cout << "Latency: " << latency_ms << " ms" << endl;
    cout << "Achieved Performance: " << gflops_achieved << " GFLOPS" << endl;
    cout << "Percent of Peak GPU: " << percent_peak << " %" << endl;
    cout << "=============================" << endl
         << endl;

    double naive_gflops = gflops_achieved;

    // 2. GLOBAL_MEM_SGEMM
    blockDim = dim3(32 * 32);
    gridDim = dim3(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    cudaEventRecord(start);
    GLOBAL_MEM_SGEMM<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    latency_ms = 0;
    cudaEventElapsedTime(&latency_ms, start, stop);
    gflops_achieved = (operations / (latency_ms / 1000.0)) / 1e9; // 转换为 GFLOPS
    percent_peak = (gflops_achieved / peak_gflops) * 100.0;
    cout << "===== GLOBAL_MEM_SGEMM Results =====" << endl;
    cout << "Matrix Size: M=" << M << ", N=" << N << ", K=" << K << endl;
    cout << "Latency: " << latency_ms << " ms" << endl;
    cout << "Achieved Performance: " << gflops_achieved << " GFLOPS" << endl;
    cout << "Percent of Peak GPU: " << percent_peak << " %" << endl;
    cout << "Speedup vs Naive: " << gflops_achieved / naive_gflops << "x" << endl
         << endl;

    // 3. SHARED_MEM_SGEMM
    blockDim = dim3(32, 32);
    gridDim = dim3(CEIL_DIV(M, blockDim.x), CEIL_DIV(N, blockDim.y));
    cudaEventRecord(start);
    SHARED_MEM_SGEMM<<<gridDim, blockDim, 2 * 32 * 32 * sizeof(float)>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    latency_ms = 0;
    cudaEventElapsedTime(&latency_ms, start, stop);
    gflops_achieved = (operations / (latency_ms / 1000.0)) / 1e9; // 转换为 GFLOPS
    percent_peak = (gflops_achieved / peak_gflops) * 100.0;
    cout << "===== SHARED_MEM_SGEMM Results =====" << endl;
    cout << "Matrix Size: M=" << M << ", N=" << N << ", K=" << K << endl;
    cout << "Latency: " << latency_ms << " ms" << endl;
    cout << "Achieved Performance: " << gflops_achieved << " GFLOPS" << endl;
    cout << "Percent of Peak GPU: " << percent_peak << " %" << endl;
    cout << "Speedup vs Naive: " << gflops_achieved / naive_gflops << "x" << endl
         << endl;

    // 4. 1D Blocktiling for Calculating Multiple Results per Thread v1

    blockDim = dim3(BN, BM / TM);
    gridDim = dim3(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    cudaEventRecord(start);
    calculating_more_res_per_thread_1D_SGEMM<<<gridDim, blockDim, (BM * BK + BK * BN) * sizeof(float)>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    latency_ms = 0;
    cudaEventElapsedTime(&latency_ms, start, stop);
    gflops_achieved = (operations / (latency_ms / 1000.0)) / 1e9; // 转换为 GFLOPS
    percent_peak = (gflops_achieved / peak_gflops) * 100.0;
    cout << "===== 1D Blocktiling SGEMM Results =====" << endl;
    cout << "Matrix Size: M=" << M << ", N=" << N << ", K=" << K << endl;
    cout << "Latency: " << latency_ms << " ms" << endl;
    cout << "Achieved Performance: " << gflops_achieved << " GFLOPS" << endl;
    cout << "Percent of Peak GPU: " << percent_peak << " %" << endl;
    cout << "Speedup vs Naive: " << gflops_achieved / naive_gflops << "x" << endl
         << endl;

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}