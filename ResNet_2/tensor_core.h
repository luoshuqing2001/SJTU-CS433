#ifndef _TENSORE_CORE_H_
#define _TENSORE_CORE_H_
#include <stdint.h>
#include <string.h>
#include <malloc.h>

#include <iostream>
#include <vector>
using std::vector;

#define F16_EXPONENT_BITS 0x1F
#define F16_EXPONENT_SHIFT 10
#define F16_EXPONENT_BIAS 15
#define F16_MANTISSA_BITS 0x3ff
#define F16_MANTISSA_SHIFT (23 - F16_EXPONENT_SHIFT)
#define F16_MAX_EXPONENT (F16_EXPONENT_BITS << F16_EXPONENT_SHIFT)

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 16
#define N 16
#define K 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define TIMES 8

#define M_GLOBAL (64 * TIMES)
#define N_GLOBAL (64 * TIMES)
#define K_GLOBAL (64 * TIMES)

// Type enum: enumerate constant
enum simMemcpyKind {
  MemcpyHostToDevice = 0, /**< Host   -> Device */
  MemcpyDeviceToHost = 1  /**< Device -> Host */
};

enum s_reg_t { SRZ = 0, SR_LAINID, SR_TID_X, SR_TID_Y, SR_CTAID_X, SR_CTAID_Y };

struct dim3 {
  unsigned int x, y, z;
  constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
      : x(vx), y(vy), z(vz) {}
};

struct __half {
protected:
  unsigned short __x;

public:
  __half() = default;
};

extern __half __float2half(const float &a);

extern float __half2float(const __half &a);

extern float operator*(const __half &lh, const __half &rh);

class GPU {
public:
  GPU();
  void SIM_LDG_INSTR(bool E, unsigned sz, unsigned Rd, unsigned Sa, unsigned imm = 0);
  void SIM_STG_INSTR(bool E, unsigned sz, unsigned Rd, unsigned Sa, unsigned imm = 0);
  void SIM_HMMA_INSTR_STEP0(unsigned Sa, unsigned Sb, unsigned Sc, unsigned Rd);
  void SIM_HMMA_INSTR_STEP1(unsigned Sa, unsigned Sb, unsigned Sc, unsigned Rd);
  void SIM_HMMA_INSTR_STEP2(unsigned Sa, unsigned Sb, unsigned Sc, unsigned Rd);
  void SIM_HMMA_INSTR_STEP3(unsigned Sa, unsigned Sb, unsigned Sc, unsigned Rd);
  void SIM_S2R_INSTR(unsigned Rd);
  void SIM_IMAD_INSTR(bool wide, bool fmt, bool bnot, unsigned Rd, unsigned Ra, unsigned Sb, uint64_t Sc);
  void SIM_LOP3_INSTR(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc,
                      unsigned imm);
  void SIM_SHF_INSTR (bool dir, bool maxshift, bool Xmode, 
                      unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc);
  void SIM_CS2R_INSTR(unsigned Rd);
  void SIM_LEA_INSTR (bool HI, bool X, unsigned Rd, unsigned Ra, unsigned Sb,
                      unsigned imm, unsigned Pd0 = 7, unsigned Ps0 = 7);
  void SIM_EXIT_INSTR();
  void SIM_MOV_INSTR(unsigned Rd, unsigned val);
  void output(unsigned Rd);
  unsigned *memory_;

private:
  // unsigned warpNum_;
  unsigned *regfile_; // 256 registers per thread
  bool *pregfile_; // predicate file
  const unsigned WARP_SIZE_ = 32;
};

extern void simMalloc(size_t size, GPU &volta);

extern void simMemcpy(void *src, size_t count, size_t bias, 
                      enum simMemcpyKind kind, GPU &volta);

extern void wmma_kernel(__half *a, __half *b, float *c, GPU &volta);

extern void gemm(__half *a, __half *b, float *c, float *d, int k);

#endif
