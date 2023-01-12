// Include associated header file.
#include "../include/cuda_kernel.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <climits>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include "mma.h"

using namespace nvcuda;
using std::fstream;
using std::ofstream;
using std::cout;
using std::endl;
using std::vector;
using std::to_string;
using std::string;

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 16
#define N 16
#define K 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define TIMES 4

#define M_GLOBAL (64 * TIMES)
#define N_GLOBAL (64 * TIMES)
#define K_GLOBAL (64 * TIMES)

void write2txtf(int i, float* p, int a, int b) {
  string fw = "./data/matrix/" + to_string(i) + ".txt";
  ofstream fout(fw);
  for (int x = 0; x < a; x++) {
      for (int y = 0; y < b; y++) {
          // string tmp = to_string(p[x * b * c + y * c + z]);
          fout << (p[x * b + y]) << ' ';
      }
      fout << endl;
  }
}


void write2txt(int i, float* p, int a) {
    string fw = "./data/result/" + to_string(i) + ".txt";
    ofstream fout(fw);
    for (int x = 0; x < a; x++) {
        fout << p[x] << endl;
    }
}

void writetime(double t, int i) {
    string fw = "./data/time/" + to_string(i) + ".txt";
    ofstream fout(fw);

    fout << t << endl;
}

void read(float* array, int mode, int serial, int a, int b, int c, int d) {
    // mode: 1 ~ conv_w, 2 ~ conv_b, 3 ~ fc_w, 4 ~ fc_b
    string fname;
    if (mode == 1) {
        fname = "./data/model_data/Conv_" + to_string(serial) + "_W.txt";
    }
    else if (mode == 2) {
        fname = "./data/model_data/Conv_" + to_string(serial) + "_B.txt";
    }
    else if (mode == 3) {
        fname = "./data/model_data/Fc_W.txt";
    }
    else if (mode == 4) {
        fname = "./data/model_data/Fc_B.txt";
    }

    fstream infile(fname);

    if (b == 0 && c == 0 && d == 0) {
        for (int i = 0; i < a; i++) {
            infile >> array[i];
        }
    }
    else if (c == 0 && d == 0) {
        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++) {
                infile >> array[i * b + j];
            }
        }
    }
    else {
        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++) {
                for (int k = 0; k < c; k++) {
                    for (int l = 0; l < d; l++) {
                        infile >> array[i * b * c * d + j * c * d + k * d + l];
                    }
                }
            }
        }
    }
}

__global__ void simple_wmma_gemm (half *a, half *b, float *c, float *d, 
                                  int m_ld, int n_ld, int k_ld, 
                                  float alpha = 1.0, float beta = 1.0) {
    // Leading dimensions. Packed with no transpositions.
    int lda = k_ld;
    int ldb = k_ld;
    int ldc = n_ld;

    // Tile using a 2D grid  (0~64) * 128 + (0~128) = 8192 / 32 = 256
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    //           (0~64) * 4 + (0 ~4) = 256 ... 256 * 16 = 4096
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
    a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>
    b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over k
    for (int i = 0; i < k_ld; i += WMMA_K) {
        int aCol = i;
        int aRow = warpM * WMMA_M;
        int bCol = warpN * WMMA_N;
        int bRow = i;
        // aCol bRow a的列 和 b的行 以 WMMA_K递增，aRow a的行 以 WMMA_M递增， bCol b的列，以 WMM_N递增。
        // Bounds checking
        if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in the current value of c, scale it by beta, and add this our result
    // scaled by alpha
    int cCol = warpN * WMMA_N;
    int cRow = warpM * WMMA_M;

    if (cRow < m_ld && cCol < n_ld) {
        wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc,
        wmma::mem_row_major);

        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc,
        wmma::mem_row_major);
    }
}

__global__ void write2in (float *input, float *output, int i_1, int i_2, int i_3, 
                          int result_1, int result_2, int chl, int krl, int stride, int pad) {
    int x = blockIdx.x;  // result_1
    int y = threadIdx.x;  // result_2

    for (int i = 0; i < i_1; i++) {
        for (int j = 0; j < krl; j++) {
            for (int k = 0; k < krl; k++) {
                // (x * stride + j, y * stride + k)
                if ((x * stride + j >= pad) && (x * stride + j < pad + i_2) && 
                    (y * stride + k >= pad) && (y * stride + k < pad + i_3))
                    output[(x * result_2 + y) * (i_1 * krl * krl) + (i * (krl * krl) + j * krl + k)] = 
                    input[i * i_2 * i_3 + (x * stride + j - pad) * i_3 + (y * stride + k - pad)];
                else
                    output[(x * result_2 + y) * (i_1 * krl * krl) + (i * (krl * krl) + j * krl + k)] = 0.0f;
            }
        }
    }
}

__global__ void write2w(float *w, float *output, int chl, int i_1, int krl) {
    int x = blockIdx.x;  // (chl / 64)
    int y = threadIdx.x;  // 64

    for (int i = 0; i < i_1; i++) {
        for (int j = 0; j < krl; j++) {
            for (int k = 0; k < krl; k++) {
                output[(i * krl * krl + j * krl + k) * chl + (x * 64 + y)] = 
                w[(x * 64 + y) * i_1 * krl * krl + (i * krl * krl + j * krl + k)];
            }
        }
    }
}

__global__ void write2ain(float *tmp_min, half *tmp_ain, int conv_k, int dim_1, int size1, int size2) {
    int x = blockIdx.x;  // (chl / 64)
    int y = threadIdx.x;  // 64
    int i = x * 64 + y;

    for (int j = 0; j < conv_k; j++) {
        if ((dim_1 + i < size1) && (j < size2))
            // m_in[dim_1 + i][j]
            tmp_ain[i * conv_k + j] = __float2half(tmp_min[(dim_1 + i) * (size2) + j]);
            // a_in[i * conv_k + j] = m_in[dim_1 + i][j];
        else
            tmp_ain[i * conv_k + j] = __float2half(0);
    }
}

__global__ void write2aw(float *tmp_mw, half *tmp_aw, int conv_k, int dim_2, int size1, int chl) {
    int x = blockIdx.x;  // (chl / 64)
    int y = threadIdx.x;  // 64
    int j = x * 64 + y;

    for (int i = 0; i < conv_k; i++) {
        if ((i < size1) && (dim_2 + j < chl))
            // m_w[i][dim_2 + j]
            tmp_aw[j * conv_k + i] = __float2half(tmp_mw[i * (chl) + dim_2 + j]);
            // a_w[j * conv_k + i] = m_w[i][dim_2 + j];
        else
            tmp_aw[j * conv_k + i] = __float2half(0);
    }
}

__global__ void write2ab(float *tmp_mb, float *tmp_ab, int dim_1, int dim_2, int size1, int chl) {
    int x = blockIdx.x;  // (chl / 64)
    int y = threadIdx.x;  // 64
    int i = x * 64 + y;

    for (int j = 0; j < N_GLOBAL; j++) {
        if ((dim_1 + i < size1) && (dim_2 + j < chl))
            tmp_ab[i * N_GLOBAL + j] = tmp_mb[dim_2 + j];
        else
            tmp_ab[i * N_GLOBAL + j] = 0.0f;
    }
}

__global__ void transform2out(float *tmp_out, float *out, int dim_1, int dim_2, int size1, int chl) {
    int x = blockIdx.x;  // (chl / 64)
    int y = threadIdx.x;  // 64
    int i = x * 64 + y;

    for (int j = 0; j < N_GLOBAL; j++) {
        if ((dim_1 + i < size1) && (dim_2 + j < chl))
            tmp_out[(dim_2 + j) * size1 + dim_1 + i] = out[i * N_GLOBAL + j];
    }
}

__host__ void conv(float *input, float *w, float *b, float *output, 
          int i_1, int i_2, int i_3, int chl, int krl, int stride, int pad) {
    // Padding the input with zeros
    // When output channel is relatively small
    int result_1 = (1 + int((i_2 + 2 * pad - krl) / stride)), 
        result_2 = (1 + int((i_3 + 2 * pad - krl) / stride));

    // vector<vector<half> > m_in(result_1 * result_2, vector<half>(i_1 * krl * krl, __float2half(0)));
    float *m_in = new float [(result_1 * result_2) * (i_1 * krl * krl)];
    // vector<vector<half> > m_w(i_1 * krl * krl, vector<half>(chl, __float2half(0)));
    float *m_w = new float [(i_1 * krl * krl) * (chl)];

    dim3 blockDim2, gridDim2;
    blockDim2.x = result_2;
    blockDim2.y = 1;
    gridDim2.x = result_1;
    gridDim2.y = 1;

    float *tmp_min, *tmp_input;
    cudaMalloc((void**)(&tmp_input), sizeof(float) * i_1 * i_2 * i_3);
    cudaMalloc((void**)(&tmp_min), sizeof(float) * (result_1 * result_2) * (i_1 * krl * krl));
    cudaMemcpy(tmp_input ,input ,sizeof(float) * i_1 * i_2 * i_3, cudaMemcpyHostToDevice);
    write2in <<<gridDim2, blockDim2>>> (tmp_input, tmp_min, i_1, i_2, i_3, 
                                        result_1, result_2,  chl, krl, stride, pad);
    cudaMemcpy(m_in ,tmp_min ,sizeof(float) * (result_1 * result_2) * (i_1 * krl * krl) , cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    
    float *tmp_w, *tmp_mw;
    cudaMalloc((void**)(&tmp_w), sizeof(float) * chl * i_1 * krl * krl);
    cudaMalloc((void**)(&tmp_mw), sizeof(float) * (i_1 * krl * krl) * (chl));
    cudaMemcpy(tmp_w, w, sizeof(float) * chl * i_1 * krl * krl, cudaMemcpyHostToDevice);
    dim3 dimGrid2(chl / 64, 1, 1);
    dim3 dimBlock2(64, 1, 1);

    write2w <<<dimGrid2, dimBlock2>>> (tmp_w, tmp_mw, chl, i_1, krl);
    cudaMemcpy(m_w, tmp_mw, sizeof(float) * (i_1 * krl * krl) * (chl), cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();

    // Devide the input two matrices into 4096 x 4096 slices
    int conv_m = (1 + int((result_1 * result_2 - 1) / M_GLOBAL)) * M_GLOBAL;
    int conv_n = (1 + int((chl - 1) / N_GLOBAL)) * N_GLOBAL; 
    int conv_k = (1 + int((i_1 * krl * krl - 1) / WMMA_K)) * WMMA_K;
    // K is less important, it only need to be divided by 16

    float* tmp_mb;
    cudaMalloc((void**)(&tmp_mb), sizeof(float) * chl);
    cudaMemcpy(tmp_mb, b, sizeof(float) * chl, cudaMemcpyHostToDevice);
    cudaMemcpy(tmp_min, m_in, sizeof(float) * (result_1 * result_2) * (i_1 * krl * krl), cudaMemcpyHostToDevice);
    cudaMemcpy(tmp_mw, m_w, sizeof(float) * (i_1 * krl * krl) * (chl), cudaMemcpyHostToDevice);

    half* tmp_ain;
    cudaMalloc((void**)(&tmp_ain) , sizeof(half) * M_GLOBAL * conv_k);
    half* tmp_aw;
    cudaMalloc((void**)(&tmp_aw) , sizeof(half) * N_GLOBAL * conv_k);
    float* tmp_ab;
    cudaMalloc((void**)(&tmp_ab), sizeof(float) * N_GLOBAL * M_GLOBAL);
    float *out;
    cudaMalloc((void**)(&out), sizeof(float) * M_GLOBAL * N_GLOBAL);
    float *tmp_out;
    cudaMalloc((void**)(&tmp_out), sizeof(float) * chl * result_1 * result_2);

    for (int dim_1 = 0; dim_1 < conv_m; dim_1 += M_GLOBAL) {
        for (int dim_2 = 0; dim_2 < conv_n; dim_2 += N_GLOBAL) {
            dim3 dimGrid3(TIMES, 1, 1);
            dim3 dimBlock3(64, 1, 1);
            // Write into a_in
            write2ain <<<dimGrid3, dimBlock3>>> (tmp_min, tmp_ain, conv_k, dim_1, result_1 * result_2, i_1 * krl * krl);
            // cudaDeviceSynchronize();

            // Write into a_w
            write2aw <<<dimGrid3, dimBlock3>>> (tmp_mw, tmp_aw, conv_k, dim_2, i_1 * krl * krl, chl);
            // cudaDeviceSynchronize();

            // Write into a_b
            write2ab <<<dimGrid3, dimBlock3>>> (tmp_mb, tmp_ab, dim_1, dim_2, result_1 * result_2, chl);
            // cudaDeviceSynchronize();

            dim3 blockDim, gridDim;
            blockDim.x = 4 * 32;
            blockDim.y = 4;
            gridDim.x = TIMES;
            gridDim.y = TIMES;

            simple_wmma_gemm <<<gridDim, blockDim>>> (tmp_ain, tmp_aw, tmp_ab, out, 
                                                      M_GLOBAL, N_GLOBAL, conv_k);

            transform2out <<<gridDim, blockDim>>> (tmp_out, out, dim_1, dim_2, result_1 * result_2, chl);
            // cudaDeviceSynchronize();
        }
    }
    cudaMemcpy(output, tmp_out, sizeof(float) * chl * result_1 * result_2, cudaMemcpyDeviceToHost);
    cudaFree(tmp_input);
    cudaFree(tmp_min);
    cudaFree(tmp_w);
    cudaFree(tmp_mw);
    cudaFree(tmp_mb);
    cudaFree(tmp_ain);
    cudaFree(tmp_aw);
    cudaFree(tmp_ab);
    cudaFree(tmp_out);
    cudaFree(out);
}


__global__ void relu_kernel(float* input, int a, int b, int c) {
    int x = blockIdx.x;  // b
    int y = threadIdx.x; // c
    
    for (int i = 0; i < a; i++) {
        if (input[i * b * c + x * c + y] < 0)
            input[i * b * c + x * c + y] = 0;
    }
}

__host__ void relu(float* input, int a, int b, int c) {
    float* in;
    cudaMalloc((void **)&in, a * b * c * sizeof(float));
    cudaMemcpy(in, input, a * b * c * sizeof(float), cudaMemcpyHostToDevice);
    dim3 dimGrid(b, 1, 1);
    dim3 dimBlock(c, 1, 1);
    relu_kernel <<<dimGrid, dimBlock>>> (in, a, b, c);
    cudaMemcpy(input, in, a * b * c * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(in);
}

__global__ void maxpool_kernel(float* input, float* output, 
                             int a, int b, int c, int krl, int stride, int r1, int r2) {
    int x = blockIdx.x;  // result_1
    int y = threadIdx.x; // result_2
    
    for (int i = 0; i < a; i++) {
        float max_v = -10.0f;
        for (int j = 0; j < krl; j++) {
            for (int k = 0; k < krl; k++) {
                if (input[i * b * c + (stride * x + j) * c + (stride * y + k)] > max_v) 
                    max_v = input[i * b * c + (stride * x + j) * c + (stride * y + k)];
            }
        }
        output[i * r1 * r2 + x * r2 + y] = max_v;
    }
}


__host__ void maxpool(float* input_raw, float* output, int a, int b, int c, int krl, int stride, int pad) {
    // Pad the input
    float* input = new float[a * (b + 2 * pad) * (c + 2 * pad)];
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            for (int k = 0; k < c; k++) {
                input[i * (2 * pad + b) * (2 * pad + c) + (j + pad) * (2 * pad + c) + (k + pad)] = input_raw[i * b * c + j * c + k];
            }
        }
    }
    b = b + 2 * pad;
    c = c + 2 * pad;
    // Process the input into matrix
    int result_1 = (1 + int((b - krl) / stride)), result_2 = (1 + int((c - krl) / stride));
    float* in, * out;

    cudaMalloc((void **)&in, a * b * c * sizeof(float));
    cudaMalloc((void **)&out, a * result_1 * result_2 * sizeof(float));
    cudaMemcpy(in, input, a * b * c * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(result_1, 1, 1);
    dim3 dimBlock(result_2, 1, 1);
    maxpool_kernel <<<dimGrid, dimBlock>>> (in, out, a, b, c, krl, stride, result_1, result_2);

    cudaMemcpy(output, out, a * result_1 * result_2 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(in);
    cudaFree(out);
}

__global__ void add_kernel(float* A, float* B, float* C, int a, int b, int c) {
    int x = blockIdx.x;  // b
    int y = threadIdx.x; // c
    
    for (int i = 0; i < a; i++) {
        C[i * b * c + x * c + y] = A[i * b * c + x * c + y] + B[i * b * c + x * c + y];
    }
}

__host__ void add(const float* A, const float* B, float* C, int a, int b, int c) {
    float* aa, * bb, * cc;
    cudaMalloc((void **)&aa, a * b * c * sizeof(float));
    cudaMalloc((void **)&bb, a * b * c * sizeof(float));
    cudaMalloc((void **)&cc, a * b * c * sizeof(float));

    cudaMemcpy(aa, A, a * b * c * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bb, B, a * b * c * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(b, 1, 1);
    dim3 dimBlock(c, 1, 1);

    add_kernel <<<dimGrid, dimBlock>>> (aa, bb, cc, a, b, c);

    cudaMemcpy(C, cc, a * b * c * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(aa);
    cudaFree(bb);
    cudaFree(cc);
}

__global__ void gap_kernel(float* input, float* output, int a, int b, int c) {
    int x = blockIdx.x;  // (a / 64)
    int y = threadIdx.x; // 64

    float sum_v = 0;
    for (int i = 0; i < b; i++) {
        for (int j = 0; j < c; j++) {
            sum_v += input[(x * 64 + y) * b * c + i * c + j];
        }
    }

    output[x * 64 + y] = sum_v / (b * c);
}

__host__ void GlobalAvgPool(float* input, float* output, int a, int b, int c) {
    // One-dimensional multithreaded acceleration
    float* in, * out;
    cudaMalloc((void **)&in, a * b * c * sizeof(float));
    cudaMalloc((void **)&out, a * sizeof(float));

    cudaMemcpy(in, input, a * b * c * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(a / 64, 1, 1);
    dim3 dimBlock(64, 1, 1);

    gap_kernel <<<dimGrid, dimBlock>>> (in, out, a, b, c);

    cudaMemcpy(output, out, a * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(in);
    cudaFree(out);
}

__global__ void fc_kernel(float* in, float* w, float* b, float* out, int na, int nb) {
    int x = blockIdx.x;  // (na / 100)
    int y = threadIdx.x; // 100

    float sum_v = 0.0f;
    for (int i = 0; i < nb; i++) {
        sum_v += w[(x * 100 + y) * nb + i] * in[i];
    }

    out[x * 100 + y] = sum_v + b[x * 100 + y];
}

__host__ void fc(float* feature, float* fc_w, float* fc_b, float* result, int na, int nb) {
    float* in, * w, * b, * out;
    cudaMalloc((void **)&in, nb * sizeof(float));
    cudaMalloc((void **)&w, na * nb * sizeof(float));
    cudaMalloc((void **)&b, na * sizeof(float));
    cudaMalloc((void **)&out, na * sizeof(float));

    cudaMemcpy(in, feature, nb * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(w, fc_w, na * nb * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, fc_b, na * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 dimGrid(na / 100, 1, 1);
    dim3 dimBlock(100, 1, 1);

    fc_kernel <<<dimGrid, dimBlock>>> (in, w, b, out, na, nb);

    cudaMemcpy(result, out, na * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(in);
    cudaFree(w);
    cudaFree(b);
    cudaFree(out);
}