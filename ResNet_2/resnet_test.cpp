#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <climits>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
// #include "mma.h"
#include "tensor_core.h"

using std::fstream;
using std::ofstream;
using std::cout;
using std::endl;
using std::vector;
using std::to_string;
using std::string;

// model parameters defined as global variables

void write2txth(int i, __half* p, int a, int b) {
  string fw = "./data/matrix/" + to_string(i) + ".txt";
  ofstream fout(fw);
  for (int x = 0; x < a; x++) {
      for (int y = 0; y < b; y++) {
          // string tmp = to_string(p[x * b * c + y * c + z]);
          fout << __half2float(p[x * b + y]) << ' ';
      }
      fout << endl;
  }
}

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

__half __float2half(const float &a) {
    // Convert floating-point numbers
    // to half-precision floating-point numbers
    uint32_t f32 = (*(uint32_t *) &a);
    uint16_t f16 = 0;

    /* Decode IEEE 754 little-endian 32-bit floating-point value */
    int sign = (f32 >> 16) & 0x8000;
    /* Map exponent to the range [-127,128] */
    int exponent = ((f32 >> 23) & 0xff) - 127;
    int mantissa = f32 & 0x007fffff;

    if (exponent == 128)
    { /* Infinity or NaN */
        f16 = sign | F16_MAX_EXPONENT;
        if (mantissa) f16 |= (mantissa & F16_MANTISSA_BITS);
    }
    else if (exponent > 15)
    { /* Overflow - flush to Infinity */
        f16 = sign | F16_MAX_EXPONENT;
    }
    else if (exponent > -15)
    { /* Representable value */
        exponent += F16_EXPONENT_BIAS;
        mantissa >>= F16_MANTISSA_SHIFT;
        f16 = sign | exponent << F16_EXPONENT_SHIFT | mantissa;
    }
    else
    {
        f16 = sign;
    }
    return *(__half*)&f16;
}

float __half2float(const __half &a) {
    // Convert half-precision floating-point numbers
    // to  floating-point numbers
    unsigned short x=(*(short *) &a);
    unsigned sign = ((x >> 15) & 1);
    unsigned exponent = ((x >> 10) & 0x1f);
    unsigned mantissa = ((x & 0x3ff) << 13);
    if (exponent == 0x1f) {  /* NaN or Inf */
        mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
        exponent = 0xff;
    } else if (!exponent) {  /* Denorm or Zero */
        if (mantissa) {
            unsigned int msb;
            exponent = 0x71;
            do {
                msb = (mantissa & 0x400000);
                mantissa <<= 1;  /* normalize */
                --exponent;
            } while (!msb);
            mantissa &= 0x7fffff;  /* 1.mantissa is implicit */
        }
    } else {
        exponent += 0x70;
    }
    int temp = ((sign << 31) | (exponent << 23) | mantissa);

    return *((float*)((void*)&temp));
}

float operator*(const __half &lh, const __half &rh) {
    // Overloaded half-precision
    // floating-point number multiplication
    float a = __half2float(lh);
    float b = __half2float(rh);
    float c = a*b;
    // printf("a = %f, b = %f, c = %f\n", a, b, c);
    return c;
}

GPU::GPU() {
    // Initialize GPU resources reasonably, including regfile size and global
    // memory size, assuming sm=1 and warp_num=1
    regfile_ = new uint32_t [256 * WARP_SIZE_];
    pregfile_ = new bool [8 * WARP_SIZE_];
}

void GPU::SIM_LDG_INSTR(bool E, unsigned sz, unsigned Rd, unsigned Sa, unsigned imm) {
    // LDG implementation
    // When E is true, the address is 64 bit, which means Sa and Sa + 1 are loaded
    uint64_t pointer, p1, p2, index;
    for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
        if (E) {
            uint32_t &part_1 = regfile_[Sa * WARP_SIZE_ + threadIdx];
            uint32_t &part_2 = regfile_[(Sa + 1) * WARP_SIZE_ + threadIdx];
            p1 = part_1;
            p2 = part_2;
            pointer = (p2 << 32) | (p1);
            index = (pointer + imm - uint64_t(memory_)) / 4;
            for (int t = 0; t < int(sz / 32); t++)
                regfile_[(Rd + t) * WARP_SIZE_ + threadIdx] = memory_[index + t];
        }
    }
}
void GPU::SIM_STG_INSTR(bool E, unsigned sz, unsigned Rd, unsigned Sa, unsigned imm) {
    // STG implementation
    uint64_t pointer, p1, p2, index;
    for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
        if (E) {
            unsigned &part_1 = regfile_[Rd * WARP_SIZE_ + threadIdx];
            unsigned &part_2 = regfile_[(Rd + 1) * WARP_SIZE_ + threadIdx];
            p1 = part_1;
            p2 = part_2;
            pointer = (p2 << 32) | (p1);
            index = (pointer + imm - uint64_t(memory_)) / 4;
            for (int t = 0; t < int(sz / 32); t++) {
                memory_[index + t] = regfile_[(Sa + t) * WARP_SIZE_ + threadIdx];
                // if (Sa == 6)
                //     printf("%.5f\n", *(float*)&memory_[index + t]);
            }
        }
    }
}
void GPU::SIM_HMMA_INSTR_STEP0(unsigned Sa, unsigned Sb, unsigned Sc, unsigned Rd) {
    // HMMA.STEP0 implementation
    // Sa, Sb, Sc represent normal registers, they range from 0 to 255.
    // Read the binary data from registers and convert to other format.
    __half h_A[32][4], h_B[32][4];
    float f_C[32][2], f_D[32][2];
    short tmp;
    for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
        unsigned &A_0 = regfile_[Sa * WARP_SIZE_ + threadIdx];
        unsigned &A_1 = regfile_[(Sa + 1) * WARP_SIZE_ + threadIdx];
        unsigned &B_0 = regfile_[Sb * WARP_SIZE_ + threadIdx];
        unsigned &B_1 = regfile_[(Sb + 1) * WARP_SIZE_ + threadIdx];
        unsigned &C_0 = regfile_[Sc * WARP_SIZE_ + threadIdx];
        unsigned &C_1 = regfile_[(Sc + 1) * WARP_SIZE_ + threadIdx];
        tmp = A_0 & 0xffff;
        h_A[threadIdx][0] = *(__half*)&(tmp);
        tmp = (A_0 >> 16) & 0xffff;
        h_A[threadIdx][1] = *(__half*)&(tmp);
        tmp = A_1 & 0xffff;
        h_A[threadIdx][2] = *(__half*)&(tmp);
        tmp = (A_1 >> 16) & 0xffff;
        h_A[threadIdx][3] = *(__half*)&(tmp);
        tmp = B_0 & 0xffff;
        h_B[threadIdx][0] = *(__half*)&(tmp);
        tmp = (B_0 >> 16) & 0xffff;
        h_B[threadIdx][1] = *(__half*)&(tmp);
        tmp = B_1 & 0xffff;
        h_B[threadIdx][2] = *(__half*)&(tmp);
        tmp = (B_1 >> 16) & 0xffff;
        h_B[threadIdx][3] = *(__half*)&(tmp);
        f_C[threadIdx][0] = *(float*)&(C_0);
        f_C[threadIdx][1] = *(float*)&(C_1);
    }
    // Travelling through octet
    int threadA = 0, threadB = 0;
    for (int octet = 0; octet < 4; octet++) {
        // Compute the upper threadgroup
        int threadgroupA = octet, threadgroupB = octet;
        threadA = 4 * threadgroupA; threadB = 4 * threadgroupB;
        f_D[threadA][0] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA][0];
        threadA = 4 * threadgroupA; threadB = 1 + 4 * threadgroupB;
        f_D[threadA][1] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA][1];
        threadA = 1 + 4 * threadgroupA; threadB = 4 * threadgroupB;
        f_D[threadA][0] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA][0];
        threadA = 1 + 4 * threadgroupA; threadB = 1 + 4 * threadgroupB;
        f_D[threadA][1] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA][1];
        threadA = 4 * threadgroupA; threadB = 2 + 4 * threadgroupB;
        f_D[threadA+2][0] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA+2][0];
        threadA = 4 * threadgroupA; threadB = 3 + 4 * threadgroupB;
        f_D[threadA+2][1] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA+2][1];
        threadA = 1 + 4 * threadgroupA; threadB = 2 + 4 * threadgroupB;
        f_D[threadA+2][0] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA+2][0];
        threadA = 1 + 4 * threadgroupA; threadB = 3 + 4 * threadgroupB;
        f_D[threadA+2][1] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA+2][1];

        // Compute the lower threadgroup
        threadgroupA = octet + 4; threadgroupB = octet;
        threadA = 4 * threadgroupA; threadB = 4 * threadgroupB;
        f_D[threadA][0] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA][0];
        threadA = 4 * threadgroupA; threadB = 1 + 4 * threadgroupB;
        f_D[threadA][1] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA][1];
        threadA = 1 + 4 * threadgroupA; threadB = 4 * threadgroupB;
        f_D[threadA][0] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA][0];
        threadA = 1 + 4 * threadgroupA; threadB = 1 + 4 * threadgroupB;
        f_D[threadA][1] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA][1];
        threadA = 4 * threadgroupA; threadB = 2 + 4 * threadgroupB;
        f_D[threadA+2][0] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA+2][0];
        threadA = 4 * threadgroupA; threadB = 3 + 4 * threadgroupB;
        f_D[threadA+2][1] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA+2][1];
        threadA = 1 + 4 * threadgroupA; threadB = 2 + 4 * threadgroupB;
        f_D[threadA+2][0] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA+2][0];
        threadA = 1 + 4 * threadgroupA; threadB = 3 + 4 * threadgroupB;
        f_D[threadA+2][1] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA+2][1];
    }
    // Write back f_D to register Rd.
    for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
        uint32_t f32 = *(uint32_t *)&(f_D[threadIdx][0]);
        regfile_[Rd * WARP_SIZE_ + threadIdx] = f32;
        f32 = *(uint32_t *)&(f_D[threadIdx][1]);
        regfile_[(Rd + 1) * WARP_SIZE_ + threadIdx] = f32;
    }
}

void GPU::SIM_HMMA_INSTR_STEP1(unsigned Sa, unsigned Sb, unsigned Sc, unsigned Rd) {
    // HMMA.STEP1 implementation
    // Read the binary data from registers and convert to other format.
    __half h_A[32][4], h_B[32][4];
    float f_C[32][2], f_D[32][2];
    short tmp;
    for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
        unsigned &A_0 = regfile_[Sa * WARP_SIZE_ + threadIdx];
        unsigned &A_1 = regfile_[(Sa + 1) * WARP_SIZE_ + threadIdx];
        unsigned &B_0 = regfile_[Sb * WARP_SIZE_ + threadIdx];
        unsigned &B_1 = regfile_[(Sb + 1) * WARP_SIZE_ + threadIdx];
        unsigned &C_0 = regfile_[Sc * WARP_SIZE_ + threadIdx];
        unsigned &C_1 = regfile_[(Sc + 1) * WARP_SIZE_ + threadIdx];
        tmp = A_0 & 0xffff;
        h_A[threadIdx][0] = *(__half*)&(tmp);
        tmp = (A_0 >> 16) & 0xffff;
        h_A[threadIdx][1] = *(__half*)&(tmp);
        tmp = A_1 & 0xffff;
        h_A[threadIdx][2] = *(__half*)&(tmp);
        tmp = (A_1 >> 16) & 0xffff;
        h_A[threadIdx][3] = *(__half*)&(tmp);
        tmp = B_0 & 0xffff;
        h_B[threadIdx][0] = *(__half*)&(tmp);
        tmp = (B_0 >> 16) & 0xffff;
        h_B[threadIdx][1] = *(__half*)&(tmp);
        tmp = B_1 & 0xffff;
        h_B[threadIdx][2] = *(__half*)&(tmp);
        tmp = (B_1 >> 16) & 0xffff;
        h_B[threadIdx][3] = *(__half*)&(tmp);
        f_C[threadIdx][0] = *(float*)&(C_0);
        f_C[threadIdx][1] = *(float*)&(C_1);
    }
    // Travelling through octet
    int threadA = 0, threadB = 0;
    for (int octet = 0; octet < 4; octet++) {
        // Compute the upper threadgroup
        int threadgroupA = octet, threadgroupB = octet;
        threadA = 4 * threadgroupA; threadB = 4 * threadgroupB;
        f_D[threadA][0] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                          operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                          operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                          f_C[threadA][0];
        threadA = 4 * threadgroupA; threadB = 1 + 4 * threadgroupB;
        f_D[threadA][1] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                          operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                          operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                          f_C[threadA][1];
        threadA = 1 + 4 * threadgroupA; threadB = 4 * threadgroupB;
        f_D[threadA][0] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                          operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                          operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                          f_C[threadA][0];
        threadA = 1 + 4 * threadgroupA; threadB = 1 + 4 * threadgroupB;
        f_D[threadA][1] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                          operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                          operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                          f_C[threadA][1];
        threadA = 4 * threadgroupA; threadB = 2 + 4 * threadgroupB;
        f_D[threadA+2][0] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                            operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                            operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                            f_C[threadA+2][0];
        threadA = 4 * threadgroupA; threadB = 3 + 4 * threadgroupB;
        f_D[threadA+2][1] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                            operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                            operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                            f_C[threadA+2][1];
        threadA = 1 + 4 * threadgroupA; threadB = 2 + 4 * threadgroupB;
        f_D[threadA+2][0] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                            operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                            operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                            f_C[threadA+2][0];
        threadA = 1 + 4 * threadgroupA; threadB = 3 + 4 * threadgroupB;
        f_D[threadA+2][1] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                            operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                            operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                            f_C[threadA+2][1];

        // Compute the lower threadgroup
        threadgroupA = octet + 4; threadgroupB = octet;
        threadA = 4 * threadgroupA; threadB = 4 * threadgroupB;
        f_D[threadA][0] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                          operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                          operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                          f_C[threadA][0];
        threadA = 4 * threadgroupA; threadB = 1 + 4 * threadgroupB;
        f_D[threadA][1] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                          operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                          operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                          f_C[threadA][1];
        threadA = 1 + 4 * threadgroupA; threadB = 4 * threadgroupB;
        f_D[threadA][0] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                          operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                          operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                          f_C[threadA][0];
        threadA = 1 + 4 * threadgroupA; threadB = 1 + 4 * threadgroupB;
        f_D[threadA][1] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                          operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                          operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                          f_C[threadA][1];
        threadA = 4 * threadgroupA; threadB = 2 + 4 * threadgroupB;
        f_D[threadA+2][0] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                            operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                            operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                            f_C[threadA+2][0];
        threadA = 4 * threadgroupA; threadB = 3 + 4 * threadgroupB;
        f_D[threadA+2][1] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                            operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                            operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                            f_C[threadA+2][1];
        threadA = 1 + 4 * threadgroupA; threadB = 2 + 4 * threadgroupB;
        f_D[threadA+2][0] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                            operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                            operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                            f_C[threadA+2][0];
        threadA = 1 + 4 * threadgroupA; threadB = 3 + 4 * threadgroupB;
        f_D[threadA+2][1] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                            operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                            operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                            f_C[threadA+2][1];
    }
    // Write back f_D to register Rd.
    for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
        uint32_t f32 = *(uint32_t *)&(f_D[threadIdx][0]);
        regfile_[Rd * WARP_SIZE_ + threadIdx] = f32;
        f32 = *(uint32_t *)&(f_D[threadIdx][1]);
        regfile_[(Rd + 1) * WARP_SIZE_ + threadIdx] = f32;
    }
}
void GPU::SIM_HMMA_INSTR_STEP2(unsigned Sa, unsigned Sb, unsigned Sc, unsigned Rd) {
    // HMMA.STEP2 implementation
    // Read the binary data from registers and convert to other format.
    __half h_A[32][4], h_B[32][4];
    float f_C[32][2], f_D[32][2];
    short tmp;
    for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
        unsigned &A_0 = regfile_[Sa * WARP_SIZE_ + threadIdx];
        unsigned &A_1 = regfile_[(Sa + 1) * WARP_SIZE_ + threadIdx];
        unsigned &B_0 = regfile_[Sb * WARP_SIZE_ + threadIdx];
        unsigned &B_1 = regfile_[(Sb + 1) * WARP_SIZE_ + threadIdx];
        unsigned &C_0 = regfile_[Sc * WARP_SIZE_ + threadIdx];
        unsigned &C_1 = regfile_[(Sc + 1) * WARP_SIZE_ + threadIdx];
        tmp = A_0 & 0xffff;
        h_A[threadIdx][0] = *(__half*)&(tmp);
        tmp = (A_0 >> 16) & 0xffff;
        h_A[threadIdx][1] = *(__half*)&(tmp);
        tmp = A_1 & 0xffff;
        h_A[threadIdx][2] = *(__half*)&(tmp);
        tmp = (A_1 >> 16) & 0xffff;
        h_A[threadIdx][3] = *(__half*)&(tmp);
        tmp = B_0 & 0xffff;
        h_B[threadIdx][0] = *(__half*)&(tmp);
        tmp = (B_0 >> 16) & 0xffff;
        h_B[threadIdx][1] = *(__half*)&(tmp);
        tmp = B_1 & 0xffff;
        h_B[threadIdx][2] = *(__half*)&(tmp);
        tmp = (B_1 >> 16) & 0xffff;
        h_B[threadIdx][3] = *(__half*)&(tmp);
        f_C[threadIdx][0] = *(float*)&(C_0);
        f_C[threadIdx][1] = *(float*)&(C_1);
    }
    // Travelling through octet
    int threadA = 0, threadB = 0;
    for (int octet = 0; octet < 4; octet++) {
        // Compute the upper threadgroup
        int threadgroupA = octet, threadgroupB = octet + 4;
        threadA = 4 * threadgroupA; threadB = 4 * threadgroupB;
        f_D[threadA][0] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA][0];
        threadA = 4 * threadgroupA; threadB = 1 + 4 * threadgroupB;
        f_D[threadA][1] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA][1];
        threadA = 1 + 4 * threadgroupA; threadB = 4 * threadgroupB;
        f_D[threadA][0] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA][0];
        threadA = 1 + 4 * threadgroupA; threadB = 1 + 4 * threadgroupB;
        f_D[threadA][1] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA][1];
        threadA = 4 * threadgroupA; threadB = 2 + 4 * threadgroupB;
        f_D[threadA+2][0] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA][1], h_B[threadB][1]) +
                            operator*(h_A[threadA][2], h_B[threadB][2]) +
                            operator*(h_A[threadA][3], h_B[threadB][3]) +
                            f_C[threadA+2][0];
        threadA = 4 * threadgroupA; threadB = 3 + 4 * threadgroupB;
        f_D[threadA+2][1] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA][1], h_B[threadB][1]) +
                            operator*(h_A[threadA][2], h_B[threadB][2]) +
                            operator*(h_A[threadA][3], h_B[threadB][3]) +
                            f_C[threadA+2][1];
        threadA = 1 + 4 * threadgroupA; threadB = 2 + 4 * threadgroupB;
        f_D[threadA+2][0] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA][1], h_B[threadB][1]) +
                            operator*(h_A[threadA][2], h_B[threadB][2]) +
                            operator*(h_A[threadA][3], h_B[threadB][3]) +
                            f_C[threadA+2][0];
        threadA = 1 + 4 * threadgroupA; threadB = 3 + 4 * threadgroupB;
        f_D[threadA+2][1] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA][1], h_B[threadB][1]) +
                            operator*(h_A[threadA][2], h_B[threadB][2]) +
                            operator*(h_A[threadA][3], h_B[threadB][3]) +
                            f_C[threadA+2][1];

        // Compute the lower threadgroup
        threadgroupA = octet + 4; threadgroupB = octet + 4;
        threadA = 4 * threadgroupA; threadB = 4 * threadgroupB;
        f_D[threadA][0] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA][0];
        threadA = 4 * threadgroupA; threadB = 1 + 4 * threadgroupB;
        f_D[threadA][1] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA][1];
        threadA = 1 + 4 * threadgroupA; threadB = 4 * threadgroupB;
        f_D[threadA][0] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA][0];
        threadA = 1 + 4 * threadgroupA; threadB = 1 + 4 * threadgroupB;
        f_D[threadA][1] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA][1], h_B[threadB][1]) +
                          operator*(h_A[threadA][2], h_B[threadB][2]) +
                          operator*(h_A[threadA][3], h_B[threadB][3]) +
                          f_C[threadA][1];
        threadA = 4 * threadgroupA; threadB = 2 + 4 * threadgroupB;
        f_D[threadA+2][0] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA][1], h_B[threadB][1]) +
                            operator*(h_A[threadA][2], h_B[threadB][2]) +
                            operator*(h_A[threadA][3], h_B[threadB][3]) +
                            f_C[threadA+2][0];
        threadA = 4 * threadgroupA; threadB = 3 + 4 * threadgroupB;
        f_D[threadA+2][1] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA][1], h_B[threadB][1]) +
                            operator*(h_A[threadA][2], h_B[threadB][2]) +
                            operator*(h_A[threadA][3], h_B[threadB][3]) +
                            f_C[threadA+2][1];
        threadA = 1 + 4 * threadgroupA; threadB = 2 + 4 * threadgroupB;
        f_D[threadA+2][0] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA][1], h_B[threadB][1]) +
                            operator*(h_A[threadA][2], h_B[threadB][2]) +
                            operator*(h_A[threadA][3], h_B[threadB][3]) +
                            f_C[threadA+2][0];
        threadA = 1 + 4 * threadgroupA; threadB = 3 + 4 * threadgroupB;
        f_D[threadA+2][1] = operator*(h_A[threadA][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA][1], h_B[threadB][1]) +
                            operator*(h_A[threadA][2], h_B[threadB][2]) +
                            operator*(h_A[threadA][3], h_B[threadB][3]) +
                            f_C[threadA+2][1];
    }
    // Write back f_D to register Rd.
    for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
        uint32_t f32 = *(uint32_t *)&(f_D[threadIdx][0]);
        regfile_[Rd * WARP_SIZE_ + threadIdx] = f32;
        f32 = *(uint32_t *)&(f_D[threadIdx][1]);
        regfile_[(Rd + 1) * WARP_SIZE_ + threadIdx] = f32;
    }
}
void GPU::SIM_HMMA_INSTR_STEP3(unsigned Sa, unsigned Sb, unsigned Sc, unsigned Rd) {
    // HMMA.STEP3 implementation
    // Read the binary data from registers and convert to other format.
    __half h_A[32][4], h_B[32][4];
    float f_C[32][2], f_D[32][2];
    short tmp;
    for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
        unsigned &A_0 = regfile_[Sa * WARP_SIZE_ + threadIdx];
        unsigned &A_1 = regfile_[(Sa + 1) * WARP_SIZE_ + threadIdx];
        unsigned &B_0 = regfile_[Sb * WARP_SIZE_ + threadIdx];
        unsigned &B_1 = regfile_[(Sb + 1) * WARP_SIZE_ + threadIdx];
        unsigned &C_0 = regfile_[Sc * WARP_SIZE_ + threadIdx];
        unsigned &C_1 = regfile_[(Sc + 1) * WARP_SIZE_ + threadIdx];
        tmp = A_0 & 0xffff;
        h_A[threadIdx][0] = *(__half*)&(tmp);
        tmp = (A_0 >> 16) & 0xffff;
        h_A[threadIdx][1] = *(__half*)&(tmp);
        tmp = A_1 & 0xffff;
        h_A[threadIdx][2] = *(__half*)&(tmp);
        tmp = (A_1 >> 16) & 0xffff;
        h_A[threadIdx][3] = *(__half*)&(tmp);
        tmp = B_0 & 0xffff;
        h_B[threadIdx][0] = *(__half*)&(tmp);
        tmp = (B_0 >> 16) & 0xffff;
        h_B[threadIdx][1] = *(__half*)&(tmp);
        tmp = B_1 & 0xffff;
        h_B[threadIdx][2] = *(__half*)&(tmp);
        tmp = (B_1 >> 16) & 0xffff;
        h_B[threadIdx][3] = *(__half*)&(tmp);
        f_C[threadIdx][0] = *(float*)&(C_0);
        f_C[threadIdx][1] = *(float*)&(C_1);
    }
    // Travelling through octet
    int threadA = 0, threadB = 0;
    for (int octet = 0; octet < 4; octet++) {
        // Compute the upper threadgroup
        int threadgroupA = octet, threadgroupB = octet + 4;
        threadA = 4 * threadgroupA; threadB = 4 * threadgroupB;
        f_D[threadA][0] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                          operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                          operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                          f_C[threadA][0];
        threadA = 4 * threadgroupA; threadB = 1 + 4 * threadgroupB;
        f_D[threadA][1] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                          operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                          operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                          f_C[threadA][1];
        threadA = 1 + 4 * threadgroupA; threadB = 4 * threadgroupB;
        f_D[threadA][0] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                          operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                          operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                          f_C[threadA][0];
        threadA = 1 + 4 * threadgroupA; threadB = 1 + 4 * threadgroupB;
        f_D[threadA][1] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                          operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                          operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                          f_C[threadA][1];
        threadA = 4 * threadgroupA; threadB = 2 + 4 * threadgroupB;
        f_D[threadA+2][0] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                            operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                            operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                            f_C[threadA+2][0];
        threadA = 4 * threadgroupA; threadB = 3 + 4 * threadgroupB;
        f_D[threadA+2][1] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                            operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                            operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                            f_C[threadA+2][1];
        threadA = 1 + 4 * threadgroupA; threadB = 2 + 4 * threadgroupB;
        f_D[threadA+2][0] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                            operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                            operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                            f_C[threadA+2][0];
        threadA = 1 + 4 * threadgroupA; threadB = 3 + 4 * threadgroupB;
        f_D[threadA+2][1] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                            operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                            operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                            f_C[threadA+2][1];

        // Compute the lower threadgroup
        threadgroupA = octet + 4; threadgroupB = octet + 4;
        threadA = 4 * threadgroupA; threadB = 4 * threadgroupB;
        f_D[threadA][0] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                          operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                          operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                          f_C[threadA][0];
        threadA = 4 * threadgroupA; threadB = 1 + 4 * threadgroupB;
        f_D[threadA][1] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                          operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                          operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                          f_C[threadA][1];
        threadA = 1 + 4 * threadgroupA; threadB = 4 * threadgroupB;
        f_D[threadA][0] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                          operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                          operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                          f_C[threadA][0];
        threadA = 1 + 4 * threadgroupA; threadB = 1 + 4 * threadgroupB;
        f_D[threadA][1] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                          operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                          operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                          operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                          f_C[threadA][1];
        threadA = 4 * threadgroupA; threadB = 2 + 4 * threadgroupB;
        f_D[threadA+2][0] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                            operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                            operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                            f_C[threadA+2][0];
        threadA = 4 * threadgroupA; threadB = 3 + 4 * threadgroupB;
        f_D[threadA+2][1] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                            operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                            operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                            f_C[threadA+2][1];
        threadA = 1 + 4 * threadgroupA; threadB = 2 + 4 * threadgroupB;
        f_D[threadA+2][0] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                            operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                            operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                            f_C[threadA+2][0];
        threadA = 1 + 4 * threadgroupA; threadB = 3 + 4 * threadgroupB;
        f_D[threadA+2][1] = operator*(h_A[threadA+2][0], h_B[threadB][0]) + 
                            operator*(h_A[threadA+2][1], h_B[threadB][1]) +
                            operator*(h_A[threadA+2][2], h_B[threadB][2]) +
                            operator*(h_A[threadA+2][3], h_B[threadB][3]) +
                            f_C[threadA+2][1];
    }
    // Write back f_D to register Rd.
    for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
        uint32_t f32 = *(uint32_t *)&(f_D[threadIdx][0]);
        regfile_[Rd * WARP_SIZE_ + threadIdx] = f32;
        f32 = *(uint32_t *)&(f_D[threadIdx][1]);
        regfile_[(Rd + 1) * WARP_SIZE_ + threadIdx] = f32;
    }
}

void GPU::SIM_S2R_INSTR(unsigned Rd) {
    // S2R implementation
    for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
        regfile_[Rd * WARP_SIZE_ + threadIdx] = threadIdx;
    }
}
void GPU::SIM_IMAD_INSTR(bool wide, bool fmt, bool bnot, unsigned Rd, unsigned Ra, unsigned Sb, uint64_t Sc) {
    // IMAD implementation
    // bnot means Sc do not need to query register.
    uint64_t data, data1, data2;
    unsigned data_in;
    for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
        unsigned &ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
        unsigned &sb_data = regfile_[Sb * WARP_SIZE_ + threadIdx];
        unsigned &sc_data = regfile_[Sc * WARP_SIZE_ + threadIdx];
        if (wide) {
            if (bnot)
                data = ra_data * sb_data + Sc;
            else {
                unsigned &sc_data_2 = regfile_[(Sc + 1) * WARP_SIZE_ + threadIdx];
                data1 = sc_data;
                data2 = sc_data_2;
                data = ra_data * sb_data + ((data2 << 32)|data1);
            }
            data_in = data & 0xffffffff;
            regfile_[Rd * WARP_SIZE_ + threadIdx] = data_in;
            data_in = (data >> 32) & 0xffffffff;
            regfile_[(Rd + 1) * WARP_SIZE_ + threadIdx] = data_in;
        } else {
            if (bnot)
                regfile_[Rd * WARP_SIZE_ + threadIdx] = ra_data * sb_data + Sc;
            else
                regfile_[Rd * WARP_SIZE_ + threadIdx] = ra_data * sb_data + sc_data;
        }
    }
}

void GPU::SIM_LOP3_INSTR (unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc,
                          unsigned imm) {
  // for: warp execuation
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    // LOP3 implementation
    unsigned &ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
    unsigned &sb_data = regfile_[Sb * WARP_SIZE_ + threadIdx];
    unsigned &sc_data = regfile_[Sc * WARP_SIZE_ + threadIdx];
    unsigned data = 0;
    // imm is a 8-bit immediate number, indicating the type of function
    if (imm & 0x01) data |= (~ra_data) & (~sb_data) & (~sc_data);
    if (imm & 0x02) data |= (~ra_data) & (~sb_data) & (sc_data);
    if (imm & 0x04) data |= (~ra_data) & (sb_data) & (~sc_data);
    if (imm & 0x08) data |= (~ra_data) & (sb_data) & (sc_data);
    if (imm & 0x10) data |= (ra_data) & (~sb_data) & (~sc_data);
    if (imm & 0x20) data |= (ra_data) & (~sb_data) & (sc_data);
    if (imm & 0x40) data |= (ra_data) & (sb_data) & (~sc_data);
    if (imm & 0x80) data |= (ra_data) & (sb_data) & (sc_data);
    regfile_[Rd * WARP_SIZE_ + threadIdx] = data;
  }
}

void GPU::SIM_SHF_INSTR(bool dir, bool maxshift, bool HI, 
                        unsigned Rd, unsigned Ra, unsigned imm, unsigned Sc) {
  // dir: 0 -> left, 1 -> right; maxshift: 0 -> u32, 1 -> s32; HI: 0 -> no, 1 -> yes.
  // int -> unsigned -> int: no change
  // for: warp execuation
  if (HI) imm = imm + 32;
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    // SHF implementation
    unsigned &ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
    // unsigned &sb_data = regfile_[Sb * WARP_SIZE_ + threadIdx];
    // Sb do not need to search from register, it is usually a immediate number.
    unsigned &c_data = regfile_[Sc * WARP_SIZE_ + threadIdx];
    uint64_t sc_data = c_data;
    // Firstly, compute val = Sc << 32 | Ra
    uint64_t val = (sc_data << 32) | ra_data;
    // printf("0x%016lx\n", val);

    // Secondly, shift left / right with val
    if (dir) 
        val >>= imm;
    else 
        val <<= imm;

    val &= 0xffffffff;
    regfile_[Rd * WARP_SIZE_ + threadIdx] = val;
  }
}

void GPU::SIM_CS2R_INSTR(unsigned Rd) {
    // CS2R implementation
    for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
        regfile_[Rd * WARP_SIZE_ + threadIdx] = 0x0;
        regfile_[(Rd + 1) * WARP_SIZE_ + threadIdx] = 0x0;
    }
}

void GPU::SIM_LEA_INSTR(bool HI, bool X, unsigned Rd, unsigned Ra, unsigned Sb,
                        unsigned imm, unsigned Pd0, unsigned Ps0) {
  // for: warp execuation
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    // LEA implementation
    unsigned &ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
    unsigned &sb_data = regfile_[Sb * WARP_SIZE_ + threadIdx];
    uint64_t data = ra_data;
    if (HI)
      data = data >> (32 - imm);
    else
      data = data << imm;
    data += sb_data;
    if (X) data += pregfile_[Ps0 * WARP_SIZE_ + threadIdx];
    if (Pd0 != 7)
      pregfile_[Pd0 * WARP_SIZE_ + threadIdx] = ((data >> 32) & 0x1);
    data &= 0xffffffff;
    regfile_[Rd * WARP_SIZE_ + threadIdx] = data;
  }
}

void GPU::SIM_EXIT_INSTR() {
    // EXIT implementation
    // Set all the registers to zero;
    for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
        for (int i = 0; i < 40; i++) {
            regfile_[i * WARP_SIZE_ + threadIdx] = 0;
        }
        regfile_[255 * WARP_SIZE_ + threadIdx] = 0;
        for (int i = 0; i < 8; i++) {
            pregfile_[i * WARP_SIZE_ + threadIdx] = 0;
        }
    }
}

// void GPU::output(unsigned Rd) {
//     for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
//         printf("0x%08lx\n", regfile_[Rd * WARP_SIZE_ + threadIdx]);
//     }
// }

void GPU::SIM_MOV_INSTR(unsigned Rd, unsigned val) {
    for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
        regfile_[Rd * WARP_SIZE_ + threadIdx] = val;
    }
}

void simMalloc(size_t size, GPU &volta) {
    // sim cudaMalloc
    // Request GPU memory
    // volta.memory_ = (uint32_t*)malloc(uint64_t(size / 4));
    volta.memory_ = new uint32_t [uint64_t(size / 4)];
}

void simMemcpy (void *src, size_t count, size_t bias, enum simMemcpyKind kind,
                GPU &volta) {
    // sim cudaMemcpy
    // memcpy host memory to class GPU memory or
    // memcpy class GPU memory to host memory

    if (kind == MemcpyHostToDevice) {
        for (uint64_t i = 0; i < uint64_t(count); i++) {
            *((uint8_t *)volta.memory_ + i + uint64_t(bias)) = *((uint8_t *)src + i);
        }
    }
    else {
        for (uint64_t i = 0; i < uint64_t(count); i++) {
            *((uint8_t *)volta.memory_ + i) = *((uint8_t *)src + i + uint64_t(bias));
        }
    }
}

void wmma_kernel (__half *a, __half *b, float *c, GPU &volta){  
    simMemcpy(a, 16 * 16 * 2, 0, MemcpyHostToDevice, volta);
    simMemcpy(b, 16 * 16 * 2, 16 * 16 * 2, MemcpyHostToDevice, volta);
    simMemcpy(c, 16 * 16 * 4, 2 * 16 * 16 * 2, MemcpyHostToDevice, volta);

    volta.SIM_S2R_INSTR(2);
    volta.SIM_MOV_INSTR(12, 0x10);
    volta.SIM_MOV_INSTR(255, 0x10);
    volta.SIM_SHF_INSTR(1, 0, 1, 3, 255, 0x2, 2);
    volta.SIM_MOV_INSTR(39, 0x3); // store imm
    volta.SIM_LOP3_INSTR(29, 2, 39, 255, 0xc0);
    volta.SIM_LOP3_INSTR(3, 3, 39, 255, 0xc0);
    volta.SIM_SHF_INSTR(1, 0, 1, 2, 255, 0x4, 2);
    volta.SIM_SHF_INSTR(1, 0, 1, 0, 255, 0x1, 3);
    volta.SIM_MOV_INSTR(39, 0x8); // store imm
    volta.SIM_MOV_INSTR(255, 0x0); 
    volta.SIM_IMAD_INSTR(0, 0, 0, 6, 3, 39, 255);
    volta.SIM_MOV_INSTR(39, 0x1); // store imm
    volta.SIM_LOP3_INSTR(4, 2, 39, 255, 0xc0);
    volta.SIM_MOV_INSTR(39, 0x8); // store imm
    volta.SIM_IMAD_INSTR(0, 0, 0, 7, 0, 39, 29);
    volta.SIM_LOP3_INSTR(5, 6, 39, 29, 0xe2);
    volta.SIM_MOV_INSTR(39, 0x4); // store imm
    volta.SIM_IMAD_INSTR(0, 0, 0, 7, 4, 39, 7);
    volta.SIM_IMAD_INSTR(0, 0, 0, 5, 4, 39, 5);
    volta.SIM_MOV_INSTR(39, 0x2); // store imm
    volta.SIM_IMAD_INSTR(0, 0, 0, 7, 7, 39, 255);
    volta.SIM_SHF_INSTR(0, 0, 0, 5, 5, 0x1, 255);
    volta.SIM_IMAD_INSTR(1, 0, 1, 20, 5, 12, (uint64_t)volta.memory_);
    volta.SIM_IMAD_INSTR(1, 0, 1, 12, 7, 12, (uint64_t)volta.memory_ + 16 * 16 * 2);
    volta.SIM_LDG_INSTR(1, 128, 24, 20, 0);
    volta.SIM_LDG_INSTR(1, 128, 16, 12, 0);
    volta.SIM_LDG_INSTR(1, 128, 20, 20, 0x10);
    volta.SIM_LDG_INSTR(1, 128, 12, 12, 0x10);

    volta.SIM_MOV_INSTR(39, 0x4); // store imm
    volta.SIM_IMAD_INSTR(0, 0, 0, 2, 2, 39, 255);
    volta.SIM_LOP3_INSTR(29, 2, 39, 29, 0xe2);
    volta.SIM_MOV_INSTR(39, 0x2); // store imm
    volta.SIM_LOP3_INSTR(37, 29, 39, 255, 0xc0);
    volta.SIM_MOV_INSTR(39, 0x8); // store imm
    volta.SIM_IMAD_INSTR(0, 0, 0, 36, 3, 39, 255);
    volta.SIM_MOV_INSTR(39, 0x5); // store imm
    volta.SIM_LOP3_INSTR(29, 29, 39, 255, 0xc0);
    volta.SIM_MOV_INSTR(3, 0x0);
    volta.SIM_LEA_INSTR(0, 0, 2, 0, 37, 0x3);
    volta.SIM_MOV_INSTR(39, 0x8); // store imm
    volta.SIM_LOP3_INSTR(29, 36, 39, 29, 0xe2);
    volta.SIM_MOV_INSTR(39, 0x10); // store imm
    volta.SIM_IMAD_INSTR(1, 0, 0, 2, 29, 39, 2);
    volta.SIM_MOV_INSTR(39, ((uint64_t)&(volta.memory_[16 * 16 * 1]) & 0xffffffff)); // store imm
    volta.SIM_LEA_INSTR(0, 0, 32, 2, 39, 0x2);
    volta.SIM_MOV_INSTR(33, ((uint64_t)&(volta.memory_[16 * 16 * 1]) >> 32)); // store imm
    // volta.SIM_LEA_INSTR(1, 1, 33, 2, 39, 0, 7, 0);

    // Write c value into r8 r10 r4 r6, rather than 0, to accomplish accumulation.
    // Load the address of c into r12 and r13, then use ldg instruction to write into r8 r10 r4 r6.
    volta.SIM_MOV_INSTR(38, ((uint64_t)&(volta.memory_[16 * 16 * 1]) & 0xffffffff)); // store imm
    volta.SIM_LEA_INSTR(0, 0, 30, 2, 38, 0x2);
    volta.SIM_MOV_INSTR(31, ((uint64_t)&(volta.memory_[16 * 16 * 1]) >> 32)); // store imm

    volta.SIM_LDG_INSTR(1, 64, 8, 30, 0);
    volta.SIM_LDG_INSTR(1, 64, 10, 30, 0x80);
    volta.SIM_LDG_INSTR(1, 64, 4, 30, 0x10);
    volta.SIM_LDG_INSTR(1, 64, 6, 30, 0x90);

    volta.SIM_HMMA_INSTR_STEP0(24, 16, 8, 8);
    volta.SIM_HMMA_INSTR_STEP1(24, 16, 10, 10);
    volta.SIM_HMMA_INSTR_STEP2(24, 16, 4, 4);
    volta.SIM_HMMA_INSTR_STEP3(24, 16, 6, 6);

    volta.SIM_HMMA_INSTR_STEP0(26, 18, 8, 8);
    volta.SIM_HMMA_INSTR_STEP1(26, 18, 10, 10);
    volta.SIM_HMMA_INSTR_STEP2(26, 18, 4, 4);
    volta.SIM_HMMA_INSTR_STEP3(26, 18, 6, 6);

    volta.SIM_HMMA_INSTR_STEP0(20, 12, 8, 8);
    volta.SIM_HMMA_INSTR_STEP1(20, 12, 10, 10);
    volta.SIM_HMMA_INSTR_STEP2(20, 12, 4, 4);
    volta.SIM_HMMA_INSTR_STEP3(20, 12, 6, 6);

    volta.SIM_HMMA_INSTR_STEP0(22, 14, 8, 8);
    volta.SIM_HMMA_INSTR_STEP1(22, 14, 10, 10);
    volta.SIM_HMMA_INSTR_STEP2(22, 14, 4, 4);
    volta.SIM_HMMA_INSTR_STEP3(22, 14, 6, 6);

    volta.SIM_STG_INSTR(1, 64, 32, 8, 0);
    volta.SIM_STG_INSTR(1, 64, 32, 10, 0x80);
    volta.SIM_STG_INSTR(1, 64, 32, 4, 0x10);
    volta.SIM_STG_INSTR(1, 64, 32, 6, 0x90);

    volta.SIM_EXIT_INSTR();

    // Write back to d
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            c[i * 16 + j] = *(float*)&volta.memory_[i * 16 + j + 16 * 16 * 1];
        }
    }
}

void gemm(__half *a, __half *b, float *c, float *d, int k) {
    GPU volta;
    simMalloc(2 * 16 * 16 * 2 + 16 * 16 * 4, volta);

    __half *A = new __half [WMMA_M * WMMA_K];
    __half *B = new __half [WMMA_K * WMMA_N];
    float *C = new float [WMMA_M * WMMA_N];
    for (int dim_1 = 0; dim_1 < M_GLOBAL; dim_1 += WMMA_M) {
        for (int dim_2 = 0; dim_2 < N_GLOBAL; dim_2 += WMMA_N) {
            // Write into matrix c and create d.
            // float *D = new float [WMMA_M * WMMA_N];
            for (int i = 0; i < WMMA_M; i++) {
                for (int j = 0; j < WMMA_N; j++) {
                    C[i * WMMA_N + j] = c[(i + dim_1) * N_GLOBAL + (j + dim_2)];
                }
            }
            for (int dim_k = 0; dim_k < k; dim_k += WMMA_K) {
                // Write into matrix A and B.
                for (int i = 0; i < WMMA_M; i++) {
                    for (int j = 0; j < WMMA_K; j++) {
                        A[i * WMMA_K + j] = a[(i + dim_1) * k + (j + dim_k)];
                    }
                }
                for (int i = 0; i < WMMA_K; i++) {
                    for (int j = 0; j < WMMA_N; j++) {
                        B[j * WMMA_K + i] = b[(j + dim_2) * k + (i + dim_k)];
                    }
                }
                // printf("11111\n");
                wmma_kernel(A, B, C, volta);
                // printf("22222\n");
            }
            // Write back into d.
            for (int i = 0; i < WMMA_M; i++) {
                for (int j = 0; j < WMMA_N; j++) {
                    d[(i + dim_1) * N_GLOBAL + (j + dim_2)] = C[i * WMMA_N + j];
                }
            }
            // delete [] D;
        }
    }
}


void read(float* array, int mode, int serial, int a, int b = 0, int c = 0, int d = 0) {
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



void conv(float *input_raw, float *w, float *b, float *output, 
          int i_1, int i_2, int i_3, int chl, int krl, int stride, int pad) {
    // Padding the input with zeros
    // When output channel is relatively small
    float* input = new float[i_1 * (2 * pad + i_2) * (2 * pad + i_3)];
    memset(input, 0, i_1 * (2 * pad + i_2) * (2 * pad + i_3) * sizeof(float));
    for (int i = 0; i < i_1; i++) {
        for (int j = 0; j < i_2; j++) {
          for (int k = 0; k < i_3; k++) {
              input[i * (2 * pad + i_2) * (2 * pad + i_3) + (j + pad) * (2 * pad + i_3) + (k + pad)] = input_raw[i * i_2 * i_3 + j * i_3 + k];
          }
        }
    }
    i_2 = i_2 + 2 * pad;
    i_3 = i_3 + 2 * pad;
    // To avoid conv_m being too large, we compute conv by line.
    int result_1 = (1 + int((i_2 - krl) / stride)), result_2 = (1 + int((i_3 - krl) / stride));

    vector<vector<__half> > m_in(result_1 * result_2, vector<__half>(i_1 * krl * krl, __float2half(0)));
    vector<vector<__half> > m_w(i_1 * krl * krl, vector<__half>(chl, __float2half(0)));

    // Compute the matrixes.
    int counter = 0;
    for (int i = 0; i <= int((i_2 - krl) / stride) * stride; i += stride) {
        for (int j = 0; j <= int((i_3 - krl) / stride) * stride; j += stride) {
            for (int m = 0; m < i_1; m++) {
                for (int n = 0; n < krl; n++) {
                    for (int p = 0; p < krl; p++) {
                        m_in[counter][m * krl * krl + n * krl + p] = 
                        __float2half(input[m * i_2 * i_3 + (i + n) * i_3 + (j + p)]);
                    }
                }
            }
        counter++;
        }
    }

    for (int i = 0; i < chl; i++) {
        for (int j = 0; j < i_1; j++) {
          for (int k = 0; k < krl; k++) {
              for (int l = 0; l < krl; l++) {
                  m_w[j * krl * krl + k * krl + l][i] = 
                  __float2half(w[i * i_1 * krl * krl + j * krl * krl + k * krl + l]);
              }
          }
        }
    }

    // Devide the input two matrices into 4096 x 4096 slices
    int conv_m = (1 + int((result_1 * result_2 - 1) / M_GLOBAL)) * M_GLOBAL;
    int conv_n = (1 + int((m_w[0].size() - 1) / N_GLOBAL)) * N_GLOBAL; 
    int conv_k = (1 + int((m_in[0].size() - 1) / WMMA_K)) * WMMA_K;
    // K is less important, it only need to be divided by 16

    __half* a_in = new __half[M_GLOBAL * conv_k];
    __half* a_w = new __half[conv_k * N_GLOBAL];
    float* a_b = new float[M_GLOBAL * N_GLOBAL];

    // printf("%d\n", conv_n);

    for (int dim_1 = 0; dim_1 < conv_m; dim_1 += M_GLOBAL) {
        for (int dim_2 = 0; dim_2 < conv_n; dim_2 += N_GLOBAL) {
            // Write into a_in
            for (int i = 0; i < M_GLOBAL; i++) {
                for (int j = 0; j < conv_k; j++) {
                    if ((dim_1 + i < result_1 * result_2) && (j < m_in[0].size()))
                        a_in[i * conv_k + j] = m_in[dim_1 + i][j];
                    else
                        a_in[i * conv_k + j] = __float2half(0);
                }
            }
            // Write into a_w
            for (int i = 0; i < conv_k; i++) {
                for (int j = 0; j < N_GLOBAL; j++) {
                    if ((i < m_in[0].size()) && (dim_2 + j < m_w[0].size()))
                        a_w[j * conv_k + i] = m_w[i][dim_2 + j];
                    else
                        a_w[j * conv_k + i] = __float2half(0);
                }
            }
            // Write into a_b
            for (int i = 0; i < M_GLOBAL; i++) {
                for (int j = 0; j < N_GLOBAL; j++) {
                    if ((dim_1 + i < result_1 * result_2) && (dim_2 + j < m_w[0].size()))
                        a_b[i * N_GLOBAL + j] = b[dim_2 + j];
                    else
                        a_b[i * N_GLOBAL + j] = 0.0f;
                }
            }

            float* a_r = new float[M_GLOBAL * N_GLOBAL];
            gemm(a_in, a_w, a_b, a_r, conv_k);

            // write2txtf(int(dim_1 / M_GLOBAL), a_r, M_GLOBAL, N_GLOBAL);

            for (int i = 0; i < M_GLOBAL; i++) {
                for (int j = 0; j < N_GLOBAL; j++) {
                    if ((dim_1 + i < result_1 * result_2) && (dim_2 + j < m_w[0].size()))
                        output[(dim_2 + j) * result_1 * result_2 + dim_1 + i] = a_r[i * N_GLOBAL + j];
                }
            }

            delete [] a_r;
        }
    }

    delete [] a_in;
    delete [] a_w;
    delete [] a_b;
}



void relu(float* input, int a, int b, int c) {
    // float* in;
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            for (int k = 0; k < c; k++) {
                if (input[i * b * c + j * c + k] < 0)
                    input[i * b * c + j * c + k] = 0;
            }
        }
    }
}



void maxpool(float* input_raw, float* output, int a, int b, int c, int krl, int stride, int pad) {
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
    // float* in, * out;

    for (int i = 0; i < result_1; i++) {
        for (int j = 0; j < result_2; j++) {
            for (int k = 0; k < a; k++) {
                float max_v = -10.0f;
                for (int x = 0; x < krl; x++) {
                    for (int y = 0; y < krl; y++) {
                        if (input[k * b * c + (stride * i + x) * c + (stride * j + y)] > max_v) 
                            max_v = input[k * b * c + (stride * i + x) * c + (stride * j + y)];
                    }
                }
                output[k * result_1 * result_2 + i * result_2 + j] = max_v;
            }
        }
    }
}


void add(const float* A, const float* B, float* C, int a, int b, int c) {
    for (int i = 0; i < b; i++) {
        for (int j = 0; j < c; j++) {
            for (int k = 0; k < a; k++) {
                C[k * b * c + i * c + j] = A[k * b * c + i * c + j] + B[k * b * c + i * c + j];
            }
        }
    }
}


void GlobalAvgPool(float* input, float* output, int a, int b, int c) {
    for (int i = 0; i < a; i++) {
        float sum_v = 0.0;
        for (int j = 0; j < b; j++) {
            for (int k = 0; k < c; k++) {
                sum_v += input[i * b * c + j * c + k];
            }
        }
        output[i] = sum_v / (b * c);
    }
}


void fc(float* feature, float* fc_w, float* fc_b, float* result, int na, int nb) {
    for (int i = 0; i < na; i++) {
        float sum_v = 0.0f;
        for (int j = 0; j < nb; j++) {
            sum_v += fc_w[i * nb + j] * feature[j];
        }
        result[i] = sum_v + fc_b[i];
    }
}


int main() {
    // We need to complete padding in main function.
    int input_size = 10;

    float* conv_1_w = new float[64 * 3 * 7 * 7];
    float* conv_1_b = new float[64];
    read(conv_1_w, 1, 1, 64, 3, 7, 7);
    read(conv_1_b, 2, 1, 64);

    float* conv_2_w = new float[64 * 64 * 3 * 3];
    float* conv_2_b = new float[64];
    read(conv_2_w, 1, 2, 64, 64, 3, 3);
    read(conv_2_b, 2, 2, 64);

    float* conv_3_w = new float[64 * 64 * 3 * 3];
    float* conv_3_b = new float[64];
    read(conv_3_w, 1, 3, 64, 64, 3, 3);
    read(conv_3_b, 2, 3, 64);

    float* conv_4_w = new float[64 * 64 * 3 * 3];
    float* conv_4_b = new float[64];
    read(conv_4_w, 1, 4, 64, 64, 3, 3);
    read(conv_4_b, 2, 4, 64);

    float* conv_5_w = new float[64 * 64 * 3 * 3];
    float* conv_5_b = new float[64];
    read(conv_5_w, 1, 5, 64, 64, 3, 3);
    read(conv_5_b, 2, 5, 64);

    float* conv_6_w = new float[128 * 64 * 3 * 3];
    float* conv_6_b = new float[128];
    read(conv_6_w, 1, 6, 128, 64, 3, 3);
    read(conv_6_b, 2, 6, 128);

    float* conv_7_w = new float[128 * 128 * 3 * 3];
    float* conv_7_b = new float[128];
    read(conv_7_w, 1, 7, 128, 128, 3, 3);
    read(conv_7_b, 2, 7, 128);

    float* conv_8_w = new float[128 * 64 * 1 * 1];
    float* conv_8_b = new float[128];
    read(conv_8_w, 1, 8, 128, 64, 1, 1);
    read(conv_8_b, 2, 8, 128);

    float* conv_9_w = new float[128 * 128 * 3 * 3];
    float* conv_9_b = new float[128];
    read(conv_9_w, 1, 9, 128, 128, 3, 3);
    read(conv_9_b, 2, 9, 128);

    float* conv_10_w = new float[128 * 128 * 3 * 3];
    float* conv_10_b = new float[128];
    read(conv_10_w, 1, 10, 128, 128, 3, 3);
    read(conv_10_b, 2, 10, 128);

    float* conv_11_w = new float[256 * 128 * 3 * 3];
    float* conv_11_b = new float[256];
    read(conv_11_w, 1, 11, 256, 128, 3, 3);
    read(conv_11_b, 2, 11, 256);

    float* conv_12_w = new float[256 * 256 * 3 * 3];
    float* conv_12_b = new float[256];
    read(conv_12_w, 1, 12, 256, 256, 3, 3);
    read(conv_12_b, 2, 12, 256);

    float* conv_13_w = new float[256 * 128 * 1 * 1];
    float* conv_13_b = new float[256];
    read(conv_13_w, 1, 13, 256, 128, 1, 1);
    read(conv_13_b, 2, 13, 256);

    float* conv_14_w = new float[256 * 256 * 3 * 3];
    float* conv_14_b = new float[256];
    read(conv_14_w, 1, 14, 256, 256, 3, 3);
    read(conv_14_b, 2, 14, 256);

    float* conv_15_w = new float[256 * 256 * 3 * 3];
    float* conv_15_b = new float[256];
    read(conv_15_w, 1, 15, 256, 256, 3, 3);
    read(conv_15_b, 2, 15, 256);

    float* conv_16_w = new float[512 * 256 * 3 * 3];
    float* conv_16_b = new float[512];
    read(conv_16_w, 1, 16, 512, 256, 3, 3);
    read(conv_16_b, 2, 16, 512);

    float* conv_17_w = new float[512 * 512 * 3 * 3];
    float* conv_17_b = new float[512];
    read(conv_17_w, 1, 17, 512, 512, 3, 3);
    read(conv_17_b, 2, 17, 512);

    float* conv_18_w = new float[512 * 256 * 1 * 1];
    float* conv_18_b = new float[512];
    read(conv_18_w, 1, 18, 512, 256, 1, 1);
    read(conv_18_b, 2, 18, 512);

    float* conv_19_w = new float[512 * 512 * 3 * 3];
    float* conv_19_b = new float[512];
    read(conv_19_w, 1, 19, 512, 512, 3, 3);
    read(conv_19_b, 2, 19, 512);

    float* conv_20_w = new float[512 * 512 * 3 * 3];
    float* conv_20_b = new float[512];
    read(conv_20_w, 1, 20, 512, 512, 3, 3);
    read(conv_20_b, 2, 20, 512);

    clock_t begin, end;

    for (int i = 0; i < input_size; i++) {
        cout << "Figure input " << (i + 1) << endl;
        // load the image data into memory with the format of vector in c++
        float* conv_1_i = new float[3 * 224 * 224];
        memset(conv_1_i, 0, sizeof(conv_1_i));
        string fname = "./data/mini_input/" + to_string(i+1) + ".txt";
        fstream infile(fname);

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 224; y++) {
                for (int z = 0; z < 224; z++) {
                    // padding before conv_1
                    infile >> conv_1_i[x * 224 * 224 + y * 224 + z];
                }
            }
        }

        begin = clock();

        // Conv 1
        printf("Conv 1 begin\n");
        float* conv_1_o = new float[64 * 112 * 112];
        
        conv(conv_1_i, conv_1_w, conv_1_b, conv_1_o, 3, 224, 224, 64, 7, 2, 3);
        // write2txtf(1, conv_1_o, 112, 112);
        delete [] conv_1_i;

        // ReLU 1
        relu(conv_1_o, 64, 112, 112);

        float* maxpool_1_o = new float[64 * 56 * 56];

        maxpool(conv_1_o, maxpool_1_o, 64, 112, 112, 3, 2, 1);
        delete [] conv_1_o;

        // Conv 2
        printf("Conv 2 begin\n");
        float* conv_2_o = new float[64 * 56 * 56];
        
        conv(maxpool_1_o, conv_2_w, conv_2_b, conv_2_o, 64, 56, 56, 64, 3, 1, 1);
        // write2txtf(2, conv_2_o, 56, 56);
        relu(conv_2_o, 64, 56, 56);

        // break;
        // Conv 3
        printf("Conv 3 begin\n");
        float* conv_3_o = new float[64 * 56 * 56];
        
        conv(conv_2_o, conv_3_w, conv_3_b, conv_3_o, 64, 56, 56, 64, 3, 1, 1);

        delete [] conv_2_o;

        // Add 3
        float* add_1_o = new float[64 * 56 * 56];

        add(maxpool_1_o, conv_3_o, add_1_o, 64, 56, 56);
        delete [] conv_3_o;
        delete [] maxpool_1_o;

        relu(add_1_o, 64, 56, 56);

        // Conv 4
        printf("Conv 4 begin\n");
        float* conv_4_o = new float[64 * 56 * 56];
        
        conv(add_1_o, conv_4_w, conv_4_b, conv_4_o, 64, 56, 56, 64, 3, 1, 1);

        relu(conv_4_o, 64, 56, 56);

        // Conv 5
        printf("Conv 5 begin\n");
        float* conv_5_o = new float[64 * 56 * 56];
        
        conv(conv_4_o, conv_5_w, conv_5_b, conv_5_o, 64, 56, 56, 64, 3, 1, 1);

        delete [] conv_4_o;

        // Add 2
        float* add_2_o = new float[64 * 56 * 56];

        add(add_1_o, conv_5_o, add_2_o, 64, 56, 56);
        delete [] add_1_o;
        delete [] conv_5_o;

        relu(add_2_o, 64, 56, 56);

        // Conv 6
        printf("Conv 6 begin\n");
        float* conv_6_o = new float[128 * 28 * 28];
        
        conv(add_2_o, conv_6_w, conv_6_b, conv_6_o, 64, 56, 56, 128, 3, 2, 1);

        relu(conv_6_o, 128, 28, 28);

        // Conv 7
        printf("Conv 7 begin\n");
        float* conv_7_o = new float[128 * 28 * 28];
        
        conv(conv_6_o, conv_7_w, conv_7_b, conv_7_o, 128, 28, 28, 128, 3, 1, 1);

        delete [] conv_6_o;

        // Conv 8
        printf("Conv 8 begin\n");
        float* conv_8_o = new float[128 * 28 * 28];
        
        conv(add_2_o, conv_8_w, conv_8_b, conv_8_o, 64, 56, 56, 128, 1, 2, 0);

        delete [] add_2_o;

        // Add 3
        float* add_3_o = new float[128 * 28 * 28];

        add(conv_7_o, conv_8_o, add_3_o, 128, 28, 28);
        delete [] conv_7_o;
        delete [] conv_8_o;

        relu(add_3_o, 128, 28, 28);

        // Conv 9
        printf("Conv 9 begin\n");
        float* conv_9_o = new float[128 * 28 * 28];
        
        conv(add_3_o, conv_9_w, conv_9_b, conv_9_o, 128, 28, 28, 128, 3, 1, 1);

        relu(conv_9_o, 128, 28, 28);

        // Conv 10
        printf("Conv 10 begin\n");
        float* conv_10_o = new float[128 * 28 * 28];
        
        conv(conv_9_o, conv_10_w, conv_10_b, conv_10_o, 128, 28, 28, 128, 3, 1, 1);

        delete [] conv_9_o;

        // Add 4
        float* add_4_o = new float[128 * 28 * 28];

        add(add_3_o, conv_10_o, add_4_o, 128, 28, 28);
        delete [] add_3_o;
        delete [] conv_10_o;

        relu(add_4_o, 128, 28, 28);

        // Conv 11
        printf("Conv 11 begin\n");
        float* conv_11_o = new float[256 * 14 * 14];
        
        conv(add_4_o, conv_11_w, conv_11_b, conv_11_o, 128, 28, 28, 256, 3, 2, 1);

        relu(conv_11_o, 256, 14, 14);

        // Conv 12
        printf("Conv 12 begin\n");
        float* conv_12_o = new float[256 * 14 * 14];
        
        conv(conv_11_o, conv_12_w, conv_12_b, conv_12_o, 256, 14, 14, 256, 3, 1, 1);

        delete [] conv_11_o;

        // Conv 13
        printf("Conv 13 begin\n");
        float* conv_13_o = new float[128 * 28 * 28];
        
        conv(add_4_o, conv_13_w, conv_13_b, conv_13_o, 128, 28, 28, 256, 1, 2, 0);

        delete [] add_4_o;

        // Add 5
        float* add_5_o = new float[256 * 14 * 14];

        add(conv_12_o, conv_13_o, add_5_o, 256, 14, 14);
        delete [] conv_12_o;
        delete [] conv_13_o;

        relu(add_5_o, 256, 14, 14);

        // Conv 14
        printf("Conv 14 begin\n");
        float* conv_14_o = new float[256 * 14 * 14];
        
        conv(add_5_o, conv_14_w, conv_14_b, conv_14_o, 256, 14, 14, 256, 3, 1, 1);

        relu(conv_14_o, 256, 14, 14);

        // Conv 15
        float* conv_15_o = new float[256 * 14 * 14];
        
        conv(conv_14_o, conv_15_w, conv_15_b, conv_15_o, 256, 14, 14, 256, 3, 1, 1);

        delete [] conv_14_o;

        // Add 6
        float* add_6_o = new float[256 * 14 * 14];

        add(add_5_o, conv_15_o, add_6_o, 256, 14, 14);
        delete [] add_5_o;
        delete [] conv_15_o;

        relu(add_6_o, 256, 14, 14);

        // Conv 16
        float* conv_16_o = new float[512 * 7 * 7];
        
        conv(add_6_o, conv_16_w, conv_16_b, conv_16_o, 256, 14, 14, 512, 3, 2, 1);

        relu(conv_16_o, 512, 7, 7);

        // Conv 17
        printf("Conv 17 begin\n");
        float* conv_17_o = new float[512 * 7 * 7];
        
        conv(conv_16_o, conv_17_w, conv_17_b, conv_17_o, 512, 7, 7, 512, 3, 1, 1);

        delete [] conv_16_o;

        // Conv 18
        printf("Conv 18 begin\n");
        float* conv_18_o = new float[256 * 14 * 14];
        
        conv(add_6_o, conv_18_w, conv_18_b, conv_18_o, 256, 14, 14, 512, 1, 2, 0);

        delete [] add_6_o;

        // Add 7
        float* add_7_o = new float[512 * 7 * 7];

        add(conv_17_o, conv_18_o, add_7_o, 512, 7, 7);
        delete [] conv_17_o;
        delete [] conv_18_o;

        relu(add_7_o, 512, 7, 7);

        // Conv 19
        printf("Conv 19 begin\n");
        float* conv_19_o = new float[512 * 7 * 7];
        
        conv(add_7_o, conv_19_w, conv_19_b, conv_19_o, 512, 7, 7, 512, 3, 1, 1);

        relu(conv_19_o, 512, 7, 7);

        // Conv 20
        printf("Conv 20 begin\n");
        float* conv_20_o = new float[512 * 7 * 7];
        
        conv(conv_19_o, conv_20_w, conv_20_b, conv_20_o, 512, 7, 7, 512, 3, 1, 1);

        delete [] conv_19_o;

        // Add 8
        float* add_8_o = new float[512 * 7 * 7];

        add(add_7_o, conv_20_o, add_8_o, 512, 7, 7);
        delete [] add_7_o;
        delete [] conv_20_o;

        relu(add_8_o, 512, 7, 7);

        // Global Average Pool
        float* feature = new float[512];

        GlobalAvgPool(add_8_o, feature, 512, 7, 7);

        // Fully Connected Layer
        printf("FC begin\n");
        float* result = new float[1000];
        float* fc_w = new float[1000 * 512];
        float* fc_b = new float[1000];
        read(fc_w, 3, 20, 1000, 512);
        read(fc_b, 4, 20, 1000);
        fc(feature, fc_w, fc_b, result, 1000, 512);
        delete [] feature;

        end = clock();

        writetime(double(end - begin) / CLOCKS_PER_SEC, i + 1);

        write2txt(i + 1, result, 1000);
    }

    delete [] conv_1_w;
    delete [] conv_1_b;
    delete [] conv_2_w;
    delete [] conv_2_b;
    delete [] conv_3_w;
    delete [] conv_3_b;
    delete [] conv_4_w;
    delete [] conv_4_b;
    delete [] conv_5_w;
    delete [] conv_5_b;
    delete [] conv_6_w;
    delete [] conv_6_b;
    delete [] conv_7_w;
    delete [] conv_7_b;
    delete [] conv_8_w;
    delete [] conv_8_b;
    delete [] conv_9_w;
    delete [] conv_9_b;
    delete [] conv_10_w;
    delete [] conv_10_b;
    delete [] conv_11_w;
    delete [] conv_11_b;
    delete [] conv_12_w;
    delete [] conv_12_b;
    delete [] conv_13_w;
    delete [] conv_13_b;
    delete [] conv_14_w;
    delete [] conv_14_b;
    delete [] conv_15_w;
    delete [] conv_15_b;
    delete [] conv_16_w;
    delete [] conv_16_b;
    delete [] conv_17_w;
    delete [] conv_17_b;
    delete [] conv_18_w;
    delete [] conv_18_b;
    delete [] conv_19_w;
    delete [] conv_19_b;
    delete [] conv_20_w;
    delete [] conv_20_b;

    return 0;
}