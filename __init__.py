"""Second Wind: run diffusion transformers on the Tesla P40's 8-bit integer math, about 2x faster, same picture.

The Tesla P40 (Pascal, 2016) multiplies 8-bit integers about 4x faster than 32-bit floats, and it is weak at 16-bit
floats, so image models normally run on it in slow 32-bit math. This ComfyUI node swaps a model's linear layers to
INT8 at load time:
  weights  squeezed to 8 bits once (one scale per output row) and cached on the GPU
  inputs   squeezed to 8 bits on every call (one scale per token) by a fused CUDA kernel, multiplied with
           torch._int_mm (INT8 x INT8 -> INT32), and scaled back to float by a second fused kernel
  spikes   the few input channels with huge outliers are found during each layer's first 3 calls and from then on
           go through normal float math (the LLM.int8() decomposition idea), which halves the error in the worst layers
Measured on a Tesla P40 with Z-Image Turbo, 1024x1024, 8 steps: 72.1 s -> 36.1 s per picture, and 97.6% of the lines
land within 3 pixels of the normal-math picture. See README.md.

Author: Khai Bustos (github.com/KhaiB10), developed with Claude (Anthropic). MIT license."""
import os
import sys
import torch

SPLIT_T, SPLIT_MAX = 10.0, 128              # a channel is a "spike" if its peak is > 10x the median channel's peak

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
// one block per row: find the row's peak (skipping masked spike channels), then write the 8-bit row and its scale
__global__ void quant_rows_k(const float* __restrict__ x, int8_t* __restrict__ q, float* __restrict__ s,
                             const unsigned char* __restrict__ mask, int K) {
  const float* xr = x + (long)blockIdx.x * K; int8_t* qr = q + (long)blockIdx.x * K;
  float m = 0.f;
  for (int i = threadIdx.x; i < K; i += blockDim.x) { float v = (mask && mask[i]) ? 0.f : fabsf(xr[i]); m = fmaxf(m, v); }
  __shared__ float red[32];
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    m = (threadIdx.x < (blockDim.x >> 5)) ? red[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if (threadIdx.x == 0) red[0] = m;
  }
  __syncthreads();
  float amax = fmaxf(red[0], 1e-8f), inv = 127.f / amax;
  for (int i = threadIdx.x; i < K; i += blockDim.x) { float v = (mask && mask[i]) ? 0.f : xr[i]; qr[i] = (int8_t)__float2int_rn(v * inv); }
  if (threadIdx.x == 0) s[blockIdx.x] = amax / 127.f;
}
// one pass: int32 result -> float, times row scale, times column scale, plus bias
__global__ void dequant_k(const int* __restrict__ acc, const float* __restrict__ sx, const float* __restrict__ sw,
                          const float* __restrict__ b, float* __restrict__ y, int N) {
  int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= N) return;
  long i = (long)r * N + c;
  float v = (float)acc[i] * sx[r] * sw[c];
  if (b) v += b[c];
  y[i] = v;
}
std::vector<torch::Tensor> quant_rows(torch::Tensor x, torch::Tensor mask) {
  int M = x.size(0), K = x.size(1);
  auto q = torch::empty({M, K}, x.options().dtype(torch::kInt8));
  auto s = torch::empty({M, 1}, x.options().dtype(torch::kFloat));
  const unsigned char* mp = mask.numel() ? mask.data_ptr<unsigned char>() : nullptr;
  quant_rows_k<<<M, 256, 0, at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), q.data_ptr<int8_t>(), s.data_ptr<float>(), mp, K);
  return {q, s};
}
torch::Tensor dequant(torch::Tensor acc, torch::Tensor sx, torch::Tensor sw, torch::Tensor b) {
  int M = acc.size(0), N = acc.size(1);
  auto y = torch::empty({M, N}, acc.options().dtype(torch::kFloat));
  dim3 grid((N + 255) / 256, M);
  dequant_k<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(acc.data_ptr<int>(), sx.data_ptr<float>(), sw.data_ptr<float>(),
                                                                b.numel() ? b.data_ptr<float>() : nullptr, y.data_ptr<float>(), N);
  return y;
}
"""
_CPP = ("std::vector<torch::Tensor> quant_rows(torch::Tensor x, torch::Tensor mask);\n"
        "torch::Tensor dequant(torch::Tensor acc, torch::Tensor sx, torch::Tensor sw, torch::Tensor b);")
K_ = None
KERNEL_STATUS = "not tried"
try:
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")   # ninja lives next to python
    from torch.utils.cpp_extension import load_inline
    K_ = load_inline("p40_int8_fused", _CPP, cuda_sources=_CUDA, functions=["quant_rows", "dequant"],
                     extra_cuda_cflags=["-O3", "-gencode=arch=compute_61,code=sm_61"], verbose=False)
    KERNEL_STATUS = "fused CUDA kernels compiled and loaded"
except Exception as e:
    KERNEL_STATUS = f"fused kernels unavailable, using lean PyTorch prep ({type(e).__name__}: {str(e)[:200]})"
print(f"[Second Wind] {KERNEL_STATUS}")
_EMPTY = {}


def _empty(dev, dtype):
    k = (dev, dtype)
    if k not in _EMPTY: _EMPTY[k] = torch.empty(0, device=dev, dtype=dtype)
    return _EMPTY[k]


# ------------------------------------------------------------------ the math
def _q_rows(t, dim):                                   # symmetric 8-bit: one scale per slice along `dim` (reference)
    s = t.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8) / 127
    return torch.round(t / s).clamp(-127, 127), s


def _prep_weights(m, dev):
    if getattr(m, "_wqT", None) is None or m._wqT.device != dev:
        W = m.weight.to(device=dev, dtype=torch.float32)
        Wq, sw = _q_rows(W, 1)
        m._wqT = Wq.to(torch.int8).t().contiguous(); m._sw = sw.squeeze(1).contiguous(); del W, Wq
        m._bf = m.bias.to(device=dev, dtype=torch.float32).contiguous() if m.bias is not None else None


def _int8_v1(m, x2):                                   # phase 1 (many small GPU steps around the multiply)
    _prep_weights(m, x2.device)
    xq, sx = _q_rows(x2.float(), 1)
    y = torch._int_mm(xq.to(torch.int8), m._wqT).float() * sx * m._sw[None, :]
    if m._bf is not None: y = y + m._bf
    return y


def _spikes(x2):
    col = x2.abs().amax(dim=0)
    thr = SPLIT_T * col.median().clamp(min=1e-8)
    idx = (col > thr).nonzero().squeeze(1)
    if idx.numel() > SPLIT_MAX: idx = col.topk(SPLIT_MAX).indices
    return idx


def _static_spikes(m, xf, calib=3, cap=64):
    """find each layer's spike channels on its first few calls, then reuse them (no per-call search)"""
    n = getattr(m, "_spk_n", 0)
    if n < calib:
        idx = _spikes(xf)
        prev = getattr(m, "_spk", None)
        idx = torch.unique(torch.cat([prev, idx])) if prev is not None else idx
        if idx.numel() > cap:
            idx = idx[xf.abs().amax(0)[idx].topk(cap).indices]
        m._spk, m._spk_n = idx, n + 1
        if m._spk_n == calib and idx.numel():
            m._spk_mask = torch.zeros(xf.shape[1], device=xf.device, dtype=torch.uint8); m._spk_mask[idx] = 1
            try:
                m._spk_W = m.weight[:, idx.to(m.weight.device)].to(device=xf.device, dtype=torch.float32).t().contiguous()
            except Exception:
                m._spk_W = m.weight.to(device=xf.device, dtype=torch.float32)[:, idx].t().contiguous()
        return idx
    return m._spk


def _int8_fast(m, x2, split=False):                    # phase 2b
    dev = x2.device; _prep_weights(m, dev)
    xf = x2.float().contiguous()
    if split == "static":
        idx = _static_spikes(m, xf)
        if m._spk_n >= 3 and getattr(m, "_spk_W", None) is None:
            split = False                                # calibrated, and this layer has no spikes
        elif m._spk_n >= 3:                              # calibrated: the cheap path
            if K_ is not None:
                xq, sx = K_.quant_rows(xf, m._spk_mask)
                y = K_.dequant(torch._int_mm(xq, m._wqT), sx, m._sw, m._bf if m._bf is not None else _empty(dev, torch.float32))
            else:
                xm = xf.masked_fill(m._spk_mask.bool()[None, :], 0)
                amax = xm.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8)
                y = torch._int_mm((xm * (127.0 / amax)).round_().to(torch.int8), m._wqT).float()
                y.mul_(amax / 127); y.mul_(m._sw)
                if m._bf is not None: y.add_(m._bf)
            return y.addmm_(xf[:, m._spk], m._spk_W)
        else:
            split = True                                # still calibrating: per-call split this time
    idx = _spikes(xf) if split else None
    if idx is not None and idx.numel():
        mask = torch.zeros(xf.shape[1], device=dev, dtype=torch.uint8); mask[idx] = 1
    else:
        mask, idx = None, None
    if K_ is not None:
        xq, sx = K_.quant_rows(xf, mask if mask is not None else _empty(dev, torch.uint8))
        y = K_.dequant(torch._int_mm(xq, m._wqT), sx, m._sw, m._bf if m._bf is not None else _empty(dev, torch.float32))
    else:
        xm = xf.masked_fill(mask.bool()[None, :], 0) if mask is not None else xf
        amax = xm.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8)
        xq = (xm * (127.0 / amax)).round_().to(torch.int8)
        y = torch._int_mm(xq, m._wqT).float(); y.mul_(amax / 127); y.mul_(m._sw)
        if m._bf is not None: y.add_(m._bf)
    if idx is not None:                                # the spike channels, in normal math
        # These columns are CACHED on the GPU. They used to be sliced out of m.weight and copied across on
        # EVERY call, and m.weight lives in host memory, so a GPU trace of one picture showed 1,474 pageable
        # host-to-device copies costing 6.9 s out of 38.6 s. The spike channels are fixed once calibration
        # ends - that is what "static" means - so the columns never change and there is nothing to recopy.
        cache = getattr(m, "_wspike", None)
        if cache is None or cache[0] is not idx:
            try:
                W = m.weight[:, idx.to(m.weight.device)].to(device=dev, dtype=torch.float32)
            except Exception:                            # some 8-bit weight formats cannot be indexed directly
                W = m.weight.to(device=dev, dtype=torch.float32)[:, idx]
            m._wspike = cache = (idx, W.t().contiguous())
        y.addmm_(xf[:, idx], cache[1])
    return y




def _make_forward(m, orig):
    def forward(x, *a, **k):
        K, N = m.in_features, m.out_features
        mode = getattr(m, "_p40_mode", "off")
        x2 = x.reshape(-1, K) if x.shape[-1] == K else None
        usable = x2 is not None and x.is_cuda and K % 8 == 0 and N % 8 == 0 and x2.shape[0] > 16
        if mode == "off" or not usable:
            return orig(x, *a, **k)
        y = _int8_fast(m, x2, split="static" if mode == "int8_spike_split" else False)
        return y.reshape(*x.shape[:-1], N).to(x.dtype)
    return forward


class P40Int8:
    """ComfyUI node: MODEL in, MODEL out. Put it between the model loader and the sampler."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "mode": (["int8_spike_split", "int8", "off"],)}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Second Wind"

    def patch(self, model, mode):
        n = 0
        for name, mod in model.model.diffusion_model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                if not getattr(mod, "_p40", False):
                    mod.forward = _make_forward(mod, mod.forward); mod._p40 = True; n += 1
                mod._p40_mode = mode
        print(f"[Second Wind] mode={mode}, {n} layers wrapped, {KERNEL_STATUS}")
        return (model,)


NODE_CLASS_MAPPINGS = {"P40Int8": P40Int8}
NODE_DISPLAY_NAME_MAPPINGS = {"P40Int8": "Second Wind: P40 INT8 (2x faster)"}
