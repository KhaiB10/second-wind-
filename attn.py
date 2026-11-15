"""attn_kernel.py: INT8 ATTENTION FOR PASCAL (research track, phase 4). Our own CUDA kernel, sm_61 dp4a.
Flash-style: one block takes 64 queries, walks the keys 32 at a time, keeps a running softmax (max + sum) per query,
and never writes the full query-by-key score table to memory.
  scores  Q.K in INT8 with __dp4a (4 byte-products per instruction). K is mean-centred first (the SageAttention trick:
          softmax ignores a constant shift per query, so subtracting the average key changes nothing mathematically,
          but it removes the channel spikes that hurt 8-bit math). Q and K: one scale per token.
  blend   P.V either in fp32 (safer) or INT8 (P quantized with a fixed scale of 127 since 0 < p <= 1, V one scale
          per channel), selectable.
Head size must be 128 (Z-Image)."""
import math, os, sys
import torch

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#define BM 64
#define BN 32
#define HD 128
#define DW 32
template <bool PV8>
__global__ void attn_k(const int* __restrict__ Q, const int* __restrict__ K, const float* __restrict__ qs, const float* __restrict__ ks,
                       const int* __restrict__ Vt, const float* __restrict__ vs, const float* __restrict__ Vf,
                       float* __restrict__ O, int L, int Lw, float sm) {
  __shared__ int Qs[BM][DW + 1];
  __shared__ int Ks[BN][DW + 1];
  __shared__ float Ss[BM][BN + 1];
  __shared__ float m_s[BM], l_s[BM], a_s[BM];
  __shared__ int Vts[PV8 ? HD : 1][BN / 4 + 1];
  __shared__ int Ps[PV8 ? BM : 1][BN / 4 + 1];
  __shared__ float Vs[PV8 ? 1 : BN][PV8 ? 1 : HD + 1];
  const int tid = threadIdx.x, ty = tid / 16, tx = tid % 16;
  const int bh = blockIdx.y, q0 = blockIdx.x * BM;
  for (int i = tid; i < BM * DW; i += 256) {
    int r = i / DW, w = i % DW, row = q0 + r;
    Qs[r][w] = row < L ? Q[((long)bh * L + row) * DW + w] : 0;
  }
  if (tid < BM) { m_s[tid] = -INFINITY; l_s[tid] = 0.f; }
  float qsc[4];
  for (int i = 0; i < 4; i++) { int row = q0 + ty * 4 + i; qsc[i] = row < L ? qs[(long)bh * L + row] * sm : 0.f; }
  float vsc[8];
  if (PV8) for (int j = 0; j < 8; j++) vsc[j] = vs[(long)bh * HD + tx * 8 + j] / 127.f;
  float o[4][8];
  for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) o[i][j] = 0.f;
  __syncthreads();
  for (int k0 = 0; k0 < L; k0 += BN) {
    for (int i = tid; i < BN * DW; i += 256) {
      int r = i / DW, w = i % DW, row = k0 + r;
      Ks[r][w] = row < L ? K[((long)bh * L + row) * DW + w] : 0;
    }
    if (PV8) {
      for (int i = tid; i < HD * (BN / 4); i += 256) {
        int d = i / (BN / 4), w = i % (BN / 4);
        Vts[d][w] = Vt[((long)bh * HD + d) * Lw + k0 / 4 + w];
      }
    } else {
      for (int i = tid; i < BN * HD; i += 256) {
        int r = i / HD, d = i % HD, row = k0 + r;
        Vs[r][d] = row < L ? Vf[((long)bh * L + row) * HD + d] : 0.f;
      }
    }
    __syncthreads();
    // scores: rows ty*4..+4, cols tx*2..+2
    for (int j = 0; j < 2; j++) {
      int c = tx * 2 + j, col = k0 + c;
      float ksc = col < L ? ks[(long)bh * L + col] : 0.f;
      for (int i = 0; i < 4; i++) {
        int r = ty * 4 + i, acc = 0;
#pragma unroll
        for (int w = 0; w < DW; w++) acc = __dp4a(Qs[r][w], Ks[c][w], acc);
        Ss[r][c] = col < L ? (float)acc * qsc[i] * ksc : -INFINITY;
      }
    }
    __syncthreads();
    // running softmax: 4 threads per query row, 8 keys each
    {
      int r = tid / 4, part = tid % 4;
      float mx = -INFINITY;
      for (int c = part * 8; c < part * 8 + 8; c++) mx = fmaxf(mx, Ss[r][c]);
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1)); mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
      float m_old = m_s[r], m_new = fmaxf(m_old, mx);
      __syncwarp();
      float sum = 0.f, p[8];
      for (int c = 0; c < 8; c++) { float s = Ss[r][part * 8 + c]; p[c] = (s == -INFINITY) ? 0.f : __expf(s - m_new); sum += p[c]; }
      sum += __shfl_xor_sync(0xffffffff, sum, 1); sum += __shfl_xor_sync(0xffffffff, sum, 2);
      if (PV8) {
        for (int h = 0; h < 2; h++) {
          int w = 0;
          for (int b = 0; b < 4; b++) w |= (__float2int_rn(p[h * 4 + b] * 127.f) & 0xFF) << (8 * b);
          Ps[r][part * 2 + h] = w;
        }
      } else {
        for (int c = 0; c < 8; c++) Ss[r][part * 8 + c] = p[c];
      }
      if (part == 0) {
        float alpha = (m_old == -INFINITY) ? 0.f : __expf(m_old - m_new);
        a_s[r] = alpha; l_s[r] = l_s[r] * alpha + sum; m_s[r] = m_new;
      }
    }
    __syncthreads();
    // blend: rows ty*4..+4, channels tx*8..+8
    for (int i = 0; i < 4; i++) {
      int r = ty * 4 + i; float alpha = a_s[r];
      for (int j = 0; j < 8; j++) {
        int d = tx * 8 + j; float add;
        if (PV8) {
          int acc = 0;
#pragma unroll
          for (int w = 0; w < BN / 4; w++) acc = __dp4a(Ps[r][w], Vts[d][w], acc);
          add = (float)acc * vsc[j];
        } else {
          add = 0.f;
#pragma unroll
          for (int c = 0; c < BN; c++) add += Ss[r][c] * Vs[c][d];
        }
        o[i][j] = o[i][j] * alpha + add;
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < 4; i++) {
    int r = ty * 4 + i, row = q0 + r;
    if (row >= L) continue;
    float inv = 1.f / l_s[r];
    for (int j = 0; j < 8; j++) O[((long)bh * L + row) * HD + tx * 8 + j] = o[i][j] * inv;
  }
}

template <bool PV8>
__global__ void attn_k2(const int* __restrict__ Q, const int* __restrict__ K, const float* __restrict__ qs, const float* __restrict__ ks,
                        const int* __restrict__ Vt, const float* __restrict__ vs, const float* __restrict__ Vf,
                        float* __restrict__ O, int L, int Lw, float sm, const float* __restrict__ kb_) {
  __shared__ int Qs[BM][DW + 1];
  __shared__ int Ks[BN][DW + 1];
  __shared__ float Ss[BM][BN + 1];
  __shared__ float m_s[BM], l_s[BM], a_s[BM];
  __shared__ int Vts[PV8 ? HD : 1][BN / 4 + 1];
  __shared__ int Ps[PV8 ? BM : 1][BN / 4 + 1];
  __shared__ float Vs[PV8 ? 1 : BN][PV8 ? 1 : HD + 1];
  const int tid = threadIdx.x, ty = tid / 16, tx = tid % 16;
  const int bh = blockIdx.y, q0 = blockIdx.x * BM;
  for (int i = tid; i < BM * (DW / 4); i += 256) {             // 16-byte loads
    int r = i / (DW / 4), w4 = i % (DW / 4), row = q0 + r;
    int4 v = row < L ? reinterpret_cast<const int4*>(Q + ((long)bh * L + row) * DW)[w4] : make_int4(0, 0, 0, 0);
    Qs[r][w4 * 4] = v.x; Qs[r][w4 * 4 + 1] = v.y; Qs[r][w4 * 4 + 2] = v.z; Qs[r][w4 * 4 + 3] = v.w;
  }
  if (tid < BM) { m_s[tid] = -INFINITY; l_s[tid] = 0.f; }
  float qsc[4];
  for (int i = 0; i < 4; i++) { int row = q0 + ty * 4 + i; qsc[i] = row < L ? qs[(long)bh * L + row] * sm : 0.f; }
  float vsc[8];
  if (PV8) for (int j = 0; j < 8; j++) vsc[j] = vs[(long)bh * HD + tx + 16 * j] / 127.f;
  float o[4][8];
  for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) o[i][j] = 0.f;
  __syncthreads();
  for (int k0 = 0; k0 < L; k0 += BN) {
    for (int i = tid; i < BN * (DW / 4); i += 256) {           // 16-byte loads
      int r = i / (DW / 4), w4 = i % (DW / 4), row = k0 + r;
      int4 v = row < L ? reinterpret_cast<const int4*>(K + ((long)bh * L + row) * DW)[w4] : make_int4(0, 0, 0, 0);
      Ks[r][w4 * 4] = v.x; Ks[r][w4 * 4 + 1] = v.y; Ks[r][w4 * 4 + 2] = v.z; Ks[r][w4 * 4 + 3] = v.w;
    }
    if (PV8) {
      for (int i = tid; i < HD * (BN / 4); i += 256) {
        int d = i / (BN / 4), w = i % (BN / 4);
        Vts[d][w] = Vt[((long)bh * HD + d) * Lw + k0 / 4 + w];
      }
    } else {
      for (int i = tid; i < BN * (HD / 4); i += 256) {
        int r = i / (HD / 4), d4 = i % (HD / 4), row = k0 + r;
        float4 v = row < L ? reinterpret_cast<const float4*>(Vf + ((long)bh * L + row) * HD)[d4] : make_float4(0.f, 0.f, 0.f, 0.f);
        Vs[r][d4 * 4] = v.x; Vs[r][d4 * 4 + 1] = v.y; Vs[r][d4 * 4 + 2] = v.z; Vs[r][d4 * 4 + 3] = v.w;
      }
    }
    __syncthreads();
    {   // scores, register-blocked: per word, 4 Q loads + 2 K loads feed 8 dp4a
      int acc[4][2];
      for (int i = 0; i < 4; i++) { acc[i][0] = 0; acc[i][1] = 0; }
#pragma unroll 8
      for (int w = 0; w < DW; w++) {
        int qa[4], kb[2];
        for (int i = 0; i < 4; i++) qa[i] = Qs[ty * 4 + i][w];
        for (int j = 0; j < 2; j++) kb[j] = Ks[tx * 2 + j][w];
        for (int i = 0; i < 4; i++) for (int j = 0; j < 2; j++) acc[i][j] = __dp4a(qa[i], kb[j], acc[i][j]);
      }
      for (int j = 0; j < 2; j++) {
        int c = tx * 2 + j, col = k0 + c;
        float ksc = col < L ? ks[(long)bh * L + col] : 0.f;
        float kb = (kb_ && col < L) ? kb_[(long)bh * L + col] : 0.f;
        for (int i = 0; i < 4; i++) Ss[ty * 4 + i][c] = col < L ? (float)acc[i][j] * qsc[i] * ksc + kb : -INFINITY;
      }
    }
    __syncthreads();
    {   // running softmax, 4 threads per row
      int r = tid / 4, part = tid % 4;
      float mx = -INFINITY;
      for (int c = part * 8; c < part * 8 + 8; c++) mx = fmaxf(mx, Ss[r][c]);
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1)); mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
      float m_old = m_s[r], m_new = fmaxf(m_old, mx);
      __syncwarp();
      float sum = 0.f, p[8];
      for (int c = 0; c < 8; c++) { float s = Ss[r][part * 8 + c]; p[c] = (s == -INFINITY) ? 0.f : __expf(s - m_new); sum += p[c]; }
      sum += __shfl_xor_sync(0xffffffff, sum, 1); sum += __shfl_xor_sync(0xffffffff, sum, 2);
      if (PV8) {
        for (int h = 0; h < 2; h++) {
          int w = 0;
          for (int b = 0; b < 4; b++) w |= (__float2int_rn(p[h * 4 + b] * 127.f) & 0xFF) << (8 * b);
          Ps[r][part * 2 + h] = w;
        }
      } else {
        for (int c = 0; c < 8; c++) Ss[r][part * 8 + c] = p[c];
      }
      if (part == 0) {
        float alpha = (m_old == -INFINITY) ? 0.f : __expf(m_old - m_new);
        a_s[r] = alpha; l_s[r] = l_s[r] * alpha + sum; m_s[r] = m_new;
      }
    }
    __syncthreads();
    // blend, register-blocked: per key, 4 P loads (broadcast) + 8 V loads (consecutive, no bank clashes) feed 32 FMAs
    for (int i = 0; i < 4; i++) { float a = a_s[ty * 4 + i]; for (int j = 0; j < 8; j++) o[i][j] *= a; }
    if (PV8) {
      for (int w = 0; w < BN / 4; w++) {
        int pa[4], vb[8];
        for (int i = 0; i < 4; i++) pa[i] = Ps[ty * 4 + i][w];
        for (int j = 0; j < 8; j++) vb[j] = Vts[tx + 16 * j][w];
        for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) o[i][j] += (float)__dp4a(pa[i], vb[j], 0) * vsc[j];
      }
    } else {
#pragma unroll 4
      for (int c = 0; c < BN; c++) {
        float pa[4], vb[8];
        for (int i = 0; i < 4; i++) pa[i] = Ss[ty * 4 + i][c];
        for (int j = 0; j < 8; j++) vb[j] = Vs[c][tx + 16 * j];
        for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) o[i][j] += pa[i] * vb[j];
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < 4; i++) {
    int r = ty * 4 + i, row = q0 + r;
    if (row >= L) continue;
    float inv = 1.f / l_s[r];
    for (int j = 0; j < 8; j++) O[((long)bh * L + row) * HD + tx + 16 * j] = o[i][j] * inv;
  }
}

// one warp per token row of 128 values: subtract the head's mean vector (optional), find the row peak, write 8 bits + scale
__global__ void prep_rows_k(const float* __restrict__ x, const float* __restrict__ mean, int8_t* __restrict__ q,
                            float* __restrict__ s, int L, int rows) {
  int row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (row >= rows) return;
  int bh = row / L;
  const float4 v = reinterpret_cast<const float4*>(x + (long)row * HD)[lane];
  float a[4] = {v.x, v.y, v.z, v.w};
  if (mean) { const float4 m = reinterpret_cast<const float4*>(mean + (long)bh * HD)[lane]; a[0] -= m.x; a[1] -= m.y; a[2] -= m.z; a[3] -= m.w; }
  float mx = fmaxf(fmaxf(fabsf(a[0]), fabsf(a[1])), fmaxf(fabsf(a[2]), fabsf(a[3])));
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  mx = fmaxf(mx, 1e-8f); float inv = 127.f / mx;
  int w = 0;
  for (int b = 0; b < 4; b++) w |= (__float2int_rn(a[b] * inv) & 0xFF) << (8 * b);
  reinterpret_cast<int*>(q + (long)row * HD)[lane] = w;
  if (lane == 0) s[row] = mx / 127.f;
}
std::vector<torch::Tensor> prep_rows(torch::Tensor x, torch::Tensor mean, int64_t L) {
  int rows = x.size(0) * x.size(1);
  auto q = torch::empty({x.size(0), x.size(1), HD}, x.options().dtype(torch::kInt8));
  auto s = torch::empty({x.size(0), x.size(1)}, x.options().dtype(torch::kFloat));
  prep_rows_k<<<(rows + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),
      mean.numel() ? mean.data_ptr<float>() : nullptr, q.data_ptr<int8_t>(), s.data_ptr<float>(), (int)L, rows);
  return {q, s};
}

// v3: blend with __dp2a (16-bit attention weights x 8-bit values, 2 multiply-adds per instruction)
__global__ void attn_k3(const int* __restrict__ Q, const int* __restrict__ K, const float* __restrict__ qs, const float* __restrict__ ks,
                        const int* __restrict__ Vt, const float* __restrict__ vs, float* __restrict__ O, int L, int Lw, float sm,
                        const float* __restrict__ kb_) {
  __shared__ int Qs[BM][DW + 1];
  __shared__ int Ks[BN][DW + 1];
  __shared__ float Ss[BM][BN + 1];
  __shared__ float m_s[BM], l_s[BM], a_s[BM];
  __shared__ int Vts[HD][BN / 4 + 1];
  __shared__ int P16[BM][BN / 2 + 1];
  const int tid = threadIdx.x, ty = tid / 16, tx = tid % 16;
  const int bh = blockIdx.y, q0 = blockIdx.x * BM;
  for (int i = tid; i < BM * (DW / 4); i += 256) {
    int r = i / (DW / 4), w4 = i % (DW / 4), row = q0 + r;
    int4 v = row < L ? reinterpret_cast<const int4*>(Q + ((long)bh * L + row) * DW)[w4] : make_int4(0, 0, 0, 0);
    Qs[r][w4 * 4] = v.x; Qs[r][w4 * 4 + 1] = v.y; Qs[r][w4 * 4 + 2] = v.z; Qs[r][w4 * 4 + 3] = v.w;
  }
  if (tid < BM) { m_s[tid] = -INFINITY; l_s[tid] = 0.f; }
  float qsc[4];
  for (int i = 0; i < 4; i++) { int row = q0 + ty * 4 + i; qsc[i] = row < L ? qs[(long)bh * L + row] * sm : 0.f; }
  float vsc[8];
  for (int j = 0; j < 8; j++) vsc[j] = vs[(long)bh * HD + tx + 16 * j] / 32767.f;
  float o[4][8];
  for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) o[i][j] = 0.f;
  __syncthreads();
  for (int k0 = 0; k0 < L; k0 += BN) {
    for (int i = tid; i < BN * (DW / 4); i += 256) {
      int r = i / (DW / 4), w4 = i % (DW / 4), row = k0 + r;
      int4 v = row < L ? reinterpret_cast<const int4*>(K + ((long)bh * L + row) * DW)[w4] : make_int4(0, 0, 0, 0);
      Ks[r][w4 * 4] = v.x; Ks[r][w4 * 4 + 1] = v.y; Ks[r][w4 * 4 + 2] = v.z; Ks[r][w4 * 4 + 3] = v.w;
    }
    for (int i = tid; i < HD * (BN / 4); i += 256) {
      int d = i / (BN / 4), w = i % (BN / 4);
      Vts[d][w] = Vt[((long)bh * HD + d) * Lw + k0 / 4 + w];
    }
    __syncthreads();
    {
      int acc[4][2];
      for (int i = 0; i < 4; i++) { acc[i][0] = 0; acc[i][1] = 0; }
#pragma unroll 8
      for (int w = 0; w < DW; w++) {
        int qa[4], kb[2];
        for (int i = 0; i < 4; i++) qa[i] = Qs[ty * 4 + i][w];
        for (int j = 0; j < 2; j++) kb[j] = Ks[tx * 2 + j][w];
        for (int i = 0; i < 4; i++) for (int j = 0; j < 2; j++) acc[i][j] = __dp4a(qa[i], kb[j], acc[i][j]);
      }
      for (int j = 0; j < 2; j++) {
        int c = tx * 2 + j, col = k0 + c;
        float ksc = col < L ? ks[(long)bh * L + col] : 0.f;
        float kbv = (kb_ && col < L) ? kb_[(long)bh * L + col] : 0.f;
        for (int i = 0; i < 4; i++) Ss[ty * 4 + i][c] = col < L ? (float)acc[i][j] * qsc[i] * ksc + kbv : -INFINITY;
      }
    }
    __syncthreads();
    {
      int r = tid / 4, part = tid % 4;
      float mx = -INFINITY;
      for (int c = part * 8; c < part * 8 + 8; c++) mx = fmaxf(mx, Ss[r][c]);
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1)); mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
      float m_old = m_s[r], m_new = fmaxf(m_old, mx);
      __syncwarp();
      float sum = 0.f, p[8];
      for (int c = 0; c < 8; c++) { float s = Ss[r][part * 8 + c]; p[c] = (s == -INFINITY) ? 0.f : __expf(s - m_new); sum += p[c]; }
      sum += __shfl_xor_sync(0xffffffff, sum, 1); sum += __shfl_xor_sync(0xffffffff, sum, 2);
      for (int h = 0; h < 4; h++) {            // 8 weights -> 4 words of two 16-bit values
        int lo = __float2int_rn(p[2 * h] * 32767.f), hi = __float2int_rn(p[2 * h + 1] * 32767.f);
        P16[r][part * 4 + h] = (lo & 0xFFFF) | (hi << 16);
      }
      if (part == 0) {
        float alpha = (m_old == -INFINITY) ? 0.f : __expf(m_old - m_new);
        a_s[r] = alpha; l_s[r] = l_s[r] * alpha + sum; m_s[r] = m_new;
      }
    }
    __syncthreads();
    for (int i = 0; i < 4; i++) { float a = a_s[ty * 4 + i]; for (int j = 0; j < 8; j++) o[i][j] *= a; }
    {
      int acc[4][8];
      for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) acc[i][j] = 0;
      for (int w = 0; w < BN / 4; w++) {        // 4 keys per step: dp2a_lo takes keys 0,1 and dp2a_hi keys 2,3
        int plo[4], phi[4], vb[8];
        for (int i = 0; i < 4; i++) { plo[i] = P16[ty * 4 + i][2 * w]; phi[i] = P16[ty * 4 + i][2 * w + 1]; }
        for (int j = 0; j < 8; j++) vb[j] = Vts[tx + 16 * j][w];
        for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) {
          acc[i][j] = __dp2a_lo(plo[i], vb[j], acc[i][j]);
          acc[i][j] = __dp2a_hi(phi[i], vb[j], acc[i][j]);
        }
      }
      for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) o[i][j] += (float)acc[i][j] * vsc[j];
    }
    __syncthreads();
  }
  for (int i = 0; i < 4; i++) {
    int r = ty * 4 + i, row = q0 + r;
    if (row >= L) continue;
    float inv = 1.f / l_s[r];
    for (int j = 0; j < 8; j++) O[((long)bh * L + row) * HD + tx + 16 * j] = o[i][j] * inv;
  }
}
torch::Tensor attn_dp2a(torch::Tensor Q, torch::Tensor K, torch::Tensor qs, torch::Tensor ks, torch::Tensor Vt,
                        torch::Tensor vs, int64_t L, double sm, torch::Tensor kbias) {
  int BH = Q.size(0);
  auto O = torch::empty({BH, L, HD}, qs.options());
  dim3 grid((L + BM - 1) / BM, BH);
  attn_k3<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(Q.data_ptr<int>(), K.data_ptr<int>(), qs.data_ptr<float>(),
      ks.data_ptr<float>(), Vt.data_ptr<int>(), vs.data_ptr<float>(), O.data_ptr<float>(), (int)L, (int)Vt.size(2), (float)sm,
      kbias.numel() ? kbias.data_ptr<float>() : nullptr);
  return O;
}
// values -> 8 bits, one scale per channel, transposed to [channel][token] and padded to a multiple of 32 tokens
__global__ void prep_v_k(const float* __restrict__ v, const float* __restrict__ vs, int8_t* __restrict__ vt, int L, int Lpad) {
  long bh = blockIdx.y; int d = threadIdx.x; int t0 = blockIdx.x * 32;
  float inv = 127.f / vs[bh * HD + d];
  for (int t = t0; t < t0 + 32; t++) {
    float x = t < L ? v[(bh * L + t) * HD + d] : 0.f;
    vt[(bh * HD + d) * Lpad + t] = (int8_t)__float2int_rn(x * inv);
  }
}
torch::Tensor prep_v(torch::Tensor v, torch::Tensor vamax, int64_t L) {
  int BH = v.size(0); int Lpad = (int)(((L + 31) / 32) * 32);
  auto vt = torch::empty({BH, HD, Lpad}, v.options().dtype(torch::kInt8));
  dim3 grid(Lpad / 32, BH);
  prep_v_k<<<grid, HD, 0, at::cuda::getCurrentCUDAStream()>>>(v.data_ptr<float>(), vamax.data_ptr<float>(), vt.data_ptr<int8_t>(), (int)L, Lpad);
  return vt;
}
torch::Tensor attn_int8(torch::Tensor Q, torch::Tensor K, torch::Tensor qs, torch::Tensor ks, torch::Tensor Vt,
                        torch::Tensor vs, torch::Tensor Vf, int64_t L, double sm, bool pv8, int64_t ver, torch::Tensor kbias) {
  int BH = Q.size(0);
  auto O = torch::empty({BH, L, HD}, qs.options());
  dim3 grid((L + BM - 1) / BM, BH);
  auto st = at::cuda::getCurrentCUDAStream();
  if (ver == 2) {
    if (pv8)
      attn_k2<true><<<grid, 256, 0, st>>>(Q.data_ptr<int>(), K.data_ptr<int>(), qs.data_ptr<float>(), ks.data_ptr<float>(),
          Vt.data_ptr<int>(), vs.data_ptr<float>(), nullptr, O.data_ptr<float>(), (int)L, (int)Vt.size(2), (float)sm,
          kbias.numel() ? kbias.data_ptr<float>() : nullptr);
    else
      attn_k2<false><<<grid, 256, 0, st>>>(Q.data_ptr<int>(), K.data_ptr<int>(), qs.data_ptr<float>(), ks.data_ptr<float>(),
          nullptr, nullptr, Vf.data_ptr<float>(), O.data_ptr<float>(), (int)L, 0, (float)sm,
          kbias.numel() ? kbias.data_ptr<float>() : nullptr);
    return O;
  }
  if (pv8)
    attn_k<true><<<grid, 256, 0, st>>>(Q.data_ptr<int>(), K.data_ptr<int>(), qs.data_ptr<float>(), ks.data_ptr<float>(),
        Vt.data_ptr<int>(), vs.data_ptr<float>(), nullptr, O.data_ptr<float>(), (int)L, (int)Vt.size(2), (float)sm);
  else
    attn_k<false><<<grid, 256, 0, st>>>(Q.data_ptr<int>(), K.data_ptr<int>(), qs.data_ptr<float>(), ks.data_ptr<float>(),
        nullptr, nullptr, Vf.data_ptr<float>(), O.data_ptr<float>(), (int)L, 0, (float)sm);
  return O;
}
"""
_CPP = ("torch::Tensor attn_int8(torch::Tensor Q, torch::Tensor K, torch::Tensor qs, torch::Tensor ks, torch::Tensor Vt, "
        "torch::Tensor vs, torch::Tensor Vf, int64_t L, double sm, bool pv8, int64_t ver, torch::Tensor kbias);\n"
        "std::vector<torch::Tensor> prep_rows(torch::Tensor x, torch::Tensor mean, int64_t L);\n"
        "torch::Tensor attn_dp2a(torch::Tensor Q, torch::Tensor K, torch::Tensor qs, torch::Tensor ks, torch::Tensor Vt, torch::Tensor vs, int64_t L, double sm, torch::Tensor kbias);\n"
        "torch::Tensor prep_v(torch::Tensor v, torch::Tensor vamax, int64_t L);")
_MOD = [None]
STATUS = ["not compiled"]


def load():
    if _MOD[0] is None:
        from torch.utils.cpp_extension import load_inline
        os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
        _MOD[0] = load_inline("p40_int8_attn_v5", _CPP, cuda_sources=_SRC, functions=["attn_int8", "prep_rows", "attn_dp2a", "prep_v"],
                              extra_cuda_cflags=["-O3", "-gencode=arch=compute_61,code=sm_61"], verbose=False)
        STATUS[0] = "compiled"
    return _MOD[0]


def _q8(t, dim):
    a = t.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8)
    return (t * (127.0 / a)).round_().to(torch.int8), (a / 127).squeeze(dim)


def int8_attention_reference(q, k, v, pv8=False, smooth=True, ver=2, smooth_q=False):
    """q, k, v: [B, H, L, 128] float. Returns [B, H, L, 128] float32.
    smooth:   centre K over tokens (exact: softmax ignores a per-query constant)
    smooth_q: centre Q over tokens too; the shift it causes is a per-key constant (mean_q . k_j), added back exactly in
              fp32 inside the kernel (the SageAttention2 idea). Needs ver=2."""
    m = load()
    B, H, L, D = q.shape
    assert D == 128, "head size must be 128"
    sm = 1 / math.sqrt(D)
    q2 = q.reshape(B * H, L, D).float()
    k2 = k.reshape(B * H, L, D).float()
    if smooth: k2 = k2 - k2.mean(dim=1, keepdim=True)
    e = torch.empty(0, device=q.device)
    kbias = e
    if smooth_q:
        qm = q2.mean(dim=1, keepdim=True)                      # [BH, 1, D]
        q2 = q2 - qm
        kbias = (torch.bmm(k2, qm.transpose(1, 2)).squeeze(2) * sm).contiguous()   # [BH, L]
    qi, qs = _q8(q2, 2); ki, ks = _q8(k2, 2)
    Qp = qi.contiguous().view(torch.int32); Kp = ki.contiguous().view(torch.int32)
    v2 = v.reshape(B * H, L, D).float().contiguous()
    if pv8:
        vi, vs = _q8(v2, 1)
        Lpad = -(-L // 32) * 32
        Vt = torch.zeros(B * H, D, Lpad, dtype=torch.int8, device=q.device)
        Vt[:, :, :L] = vi.transpose(1, 2)
        Vt = Vt.view(torch.int32)
        O = m.attn_int8(Qp, Kp, qs.contiguous(), ks.contiguous(), Vt, vs.contiguous(), e, L, sm, True, ver, kbias)
    else:
        O = m.attn_int8(Qp, Kp, qs.contiguous(), ks.contiguous(), e.int(), e, v2, L, sm, False, ver, kbias)
    return O.view(B, H, L, D)


def prep_fused(q, k, v, smooth_q=True):
    """the fused preparation: 2 small means + 1 bias + 2 single-pass squeeze kernels"""
    m = load()
    B, H, L, D = q.shape
    sm = 1 / math.sqrt(D)
    q2 = q.reshape(B * H, L, D).float().contiguous(); k2 = k.reshape(B * H, L, D).float().contiguous()
    km = k2.mean(dim=1)                                        # [BH, D]
    e = torch.empty(0, device=q.device)
    if smooth_q:
        qm = q2.mean(dim=1)
        # bias_j = mean_q . (k_j - mean_k) * scale  (exact correction for centring Q)
        kbias = ((torch.bmm(k2, qm.unsqueeze(2)).squeeze(2) - (km * qm).sum(1, keepdim=True)) * sm).contiguous()
    else:
        qm, kbias = e, e
    qi, qs = m.prep_rows(q2, qm.contiguous() if smooth_q else e, L)
    ki, ks = m.prep_rows(k2, km.contiguous(), L)
    return qi.view(torch.int32), ki.view(torch.int32), qs, ks, v.reshape(B * H, L, D).float().contiguous(), kbias


def int8_attention(q, k, v, pv8=False, smooth=True, ver=2, smooth_q=True):
    """fast path: fused prep + kernel v2 with 16-byte loads (fp32 blend). Other options use the reference path."""
    if pv8 or not smooth or ver != 2:
        return int8_attention_reference(q, k, v, pv8=pv8, smooth=smooth, ver=ver, smooth_q=smooth_q)
    m = load()
    B, H, L, D = q.shape
    Qp, Kp, qs, ks, v2, kbias = prep_fused(q, k, v, smooth_q)
    e = torch.empty(0, device=q.device)
    O = m.attn_int8(Qp, Kp, qs, ks, e.int(), e, v2, L, 1 / math.sqrt(D), False, 2, kbias)
    return O.view(B, H, L, D)


def int8_attention_dp2a(q, k, v, smooth_q=True):
    """round 5: INT8 scores + __dp2a blend (16-bit attention weights x 8-bit values)"""
    m = load()
    B, H, L, D = q.shape
    Qp, Kp, qs, ks, v2, kbias = prep_fused(q, k, v, smooth_q)
    vamax = v2.abs().amax(dim=1).clamp(min=1e-8).contiguous()          # [BH, D], one scale per channel
    Vt = m.prep_v(v2, vamax, L).view(torch.int32)
    O = m.attn_dp2a(Qp, Kp, qs, ks, Vt, vamax, L, 1 / math.sqrt(D), kbias)
    return O.view(B, H, L, D)
