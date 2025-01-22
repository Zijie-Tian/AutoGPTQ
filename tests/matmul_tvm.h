
#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(64) main_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(64) main_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  float C_local[64];
  __shared__ float A_shared[512];
  __shared__ float B_shared[512];
  for (int i_2_init = 0; i_2_init < 8; ++i_2_init) {
    for (int j_2_init = 0; j_2_init < 8; ++j_2_init) {
      C_local[((i_2_init * 8) + j_2_init)] = 0.000000e+00f;
    }
  }
  for (int k_0 = 0; k_0 < 128; ++k_0) {
    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 2; ++ax0_ax1_fused_0) {
      *(float4*)(A_shared + ((ax0_ax1_fused_0 * 256) + (((int)threadIdx.x) * 4))) = *(float4*)(A + (((((((int)blockIdx.y) * 65536) + (ax0_ax1_fused_0 * 32768)) + ((((int)threadIdx.x) >> 1) * 1024)) + (k_0 * 8)) + ((((int)threadIdx.x) & 1) * 4)));
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 2; ++ax0_ax1_fused_0_1) {
      *(float4*)(B_shared + ((ax0_ax1_fused_0_1 * 256) + (((int)threadIdx.x) * 4))) = *(float4*)(B + (((((k_0 * 8192) + (ax0_ax1_fused_0_1 * 4096)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 15) * 4)));
    }
    __syncthreads();
    for (int k_1 = 0; k_1 < 8; ++k_1) {
      for (int i_2 = 0; i_2 < 8; ++i_2) {
        for (int j_2 = 0; j_2 < 8; ++j_2) {
          C_local[((i_2 * 8) + j_2)] = (C_local[((i_2 * 8) + j_2)] + (A_shared[((((((int)threadIdx.x) >> 3) * 64) + (i_2 * 8)) + k_1)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 7) * 8)) + j_2)]));
        }
      }
    }
  }
  for (int ax0 = 0; ax0 < 8; ++ax0) {
    for (int ax1 = 0; ax1 < 8; ++ax1) {
      C[((((((((int)blockIdx.y) * 65536) + ((((int)threadIdx.x) >> 3) * 8192)) + (ax0 * 1024)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + ax1)] = C_local[((ax0 * 8) + ax1)];
    }
  }
}

