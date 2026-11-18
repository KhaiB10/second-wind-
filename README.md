# Second Wind

**Give an old Tesla P40 a second wind: run diffusion transformers 3× faster, with the same picture.**

The Tesla P40 (Pascal, 2016) is a cheap 24 GB card, but it is weak at 16-bit floats, so image models run on it in slow 32-bit math. Its hidden strength is 8-bit integer math: on the matrix shapes inside Z-Image it measured **37.5–39.7 TOPS of INT8 against 8.8–9.5 TFLOPS of FP32, about 4×**. This ComfyUI node puts that INT8 speed to work.

| Tesla P40, Z-Image Turbo, 1024×1024, 8 steps | Time per picture | Speed |
| --- | --- | --- |
| Normal (FP32 math, FP8-stored weights) | 72.1 s | 1.0× |
| Second Wind, `int8_spike_split` | 36.1 s | 2.0× |
| Second Wind, `int8_spike_split+attention` | 31.6 s | 2.28× |
| **…with the spike columns cached** | **24.1 s** | **2.99×** |

**Same picture:** drawn from the same seed, 97.6% of the INT8 picture's lines fall within 3 pixels of the normal-math picture's lines. Two different seeds score 81%. With the attention kernel on as well, against the INT8 picture: SSIM 0.912 and 98.9% line match on a portrait, 0.815 and 98.4% on a landscape. Caching the spike columns changes no arithmetic at all, so the picture is bit-identical to the 31.6 s version.

![normal math vs INT8 vs INT8 with spike splitting](docs/compare.png)

## How it works

1. **Weights** are squeezed to 8 bits once, with one scale per output row, and cached on the GPU.
2. **Inputs** are squeezed to 8 bits on every call, with one scale per token, by a fused CUDA kernel. They are multiplied with `torch._int_mm` (INT8 × INT8 → INT32) and scaled back to float by a second fused kernel. Before fusing, the eight separate prep steps around each multiply ate most of the gain.
3. **INT8 attention** (`attn.py`). Our own flash-style CUDA kernel for Pascal: one block takes 64 queries, walks the keys 32 at a time, and keeps a running softmax per query, so the full query-by-key score table is never written to memory. Scores are computed with `__dp4a` (four byte-products per instruction). The keys are mean-centred first — softmax ignores a constant shift per query, so subtracting the average key changes nothing mathematically while removing the channel spikes that wreck 8-bit math. On the model's own saved attention inputs: **56.6 ms → 38.3 ms, 1.48×, with 0.73–1.53% error.**
4. **Spike splitting.** A few input channels carry huge outliers, up to 84,000× the average in the feed-forward "down" layers. The node finds them during each layer's first three calls. From then on, those channels go through normal float math and the rest through INT8. This is the LLM.int8() decomposition idea (Dettmers et al., 2022) applied to an image model, and it halves the error in the worst layers.
5. **Those spike columns are cached on the GPU.** They have to be, and for a long time they were not — see below.

## The 6.9 seconds nobody was looking for

Spike splitting sends a handful of channels through float math. Pulling those channels out means slicing columns from the layer's weight matrix — and that slice was being taken from **host** memory and copied to the card on *every single call*, even though the channel list is fixed after the third call and never changes again.

A GPU trace of one picture made it obvious:

```
total GPU kernel time 38.6s
    9.67s  25.0% x1376   maxwell_igemm_int8_128x128_ldg4_nn
    8.40s  21.7% x256    attn_k2
    6.90s  17.9% x1474   Memcpy HtoD (Pageable -> Device)
```

1,474 host-to-device copies against 1,376 linear-layer calls: one per call, 17.9% of all GPU time, carrying bytes that were identical every time. Caching the transposed columns on the GPU on first use removed them completely — **38.6 s → 29.3 s of kernel time, 31.6 s → 24.1 s per picture, for no change in arithmetic and no change in the output.**

The lesson is not about INT8. We had spent days tuning kernels that were already fast, because kernels are the thing you think to look at. A profile found a fifth of the runtime being spent on nothing at all, in code none of us had reread since it was written.

## Where the time goes now (GPU trace, one picture)

| | Normal math | Second Wind |
| --- | --- | --- |
| Matrix layers | 48.8 s | 9.7 s (INT8 kernel) |
| Attention | 13.9 s | 8.4 s (`attn.py`) |
| Layer norm | — | 2.2 s |
| Squeeze and unsqueeze (fused kernels) | none | 2.5 s |
| Spike columns in FP32 (`sgemm`) | — | 1.6 s |
| Copying spike weights from the host | — | 6.9 s → **0** |
| **Total kernel time** | — | **29.3 s** |

**Attention is now the single biggest line that is still ours to fix**, at 28.8% of kernel time. We already know it is *not* arithmetic-bound: putting P·V in INT8 as well saved only 4.9 ms of a predicted 12 ms while costing 6.77% error, which says the kernel is waiting on memory or occupancy, not on multiply throughput. That is the next thing to profile properly.

## Install

1. Copy this folder into `ComfyUI/custom_nodes/second-wind/`.
2. The fused kernels compile on first load. That needs `nvcc` and `ninja` (`pip install ninja` in ComfyUI's Python). If either is missing, the node falls back to a slower PyTorch path and says so in the log.
3. Start ComfyUI with `--force-fp32`, and load the model with `weight_dtype = fp8_e4m3fn`.
4. Put the **Second Wind: P40 INT8** node between the model loader and the sampler. Use mode `int8_spike_split+attention` for the full 3×, or `int8_spike_split` for the linear layers alone.

**Restart ComfyUI after updating this node.** Custom nodes are read once at startup, so copying a new file over the old one changes nothing until the process restarts — which also means a benchmark run after a hot copy is measuring the *old* code.

## What we tried that did not work

- **Token downsampling in attention (ToDo-style 2×2 key/value pooling).** 1.4× faster again, but it **smeared the linework into blocks**. Z-Image tags each token's position with RoPE before attention, and averaging four neighbours mixes up their position codes. The run with downsampling only in late steps scored 0.967 on our line-match metric and still looked bad. Lesson: judge by eye with side-by-side sheets, not by an edge-counting score.
- **Spike search on every call** improved accuracy but was slower (1.5×). Searching only in the first three calls and reusing the result kept the accuracy and the speed.
- **A 16-bit × 8-bit attention blend using `__dp2a`.** The idea was to keep the queries in 16 bits and buy accuracy for free. Measured 42.1 ms against the plain INT8 path's 38.3 ms — *slower* — with 12,607% error and 18% of tokens below 0.99 cosine similarity. Two bugs compounded: a missing divide by 127 made outputs 127× too large (hidden for a while by the model's own normalisation), and per-channel scaling of V broke the per-token softmax weighting. Correcting the scale still left it slower than plain INT8. The code is kept in `attn.py` as `int8_attention_dp2a`, clearly marked, because a negative result is still a result.
- **P·V in INT8.** Predicted to save about 12 ms; measured 4.9 ms, at 6.77% error with 19% of tokens below 0.99 cosine similarity. Rejected — but the number was worth having, because it is what proved the attention kernel is not arithmetic-bound.

## Limits

- Tested on one card (Tesla P40) and one model (Z-Image Turbo, which ComfyUI loads as `NextDiT`). The node patches every `torch.nn.Linear` in the diffusion model, so other DiT models may work, but they are untested.
- INT8 draws a close sibling of the normal picture, not a bit-exact copy.
- The cached INT8 weights take about 6 GB of extra GPU memory, and ComfyUI holds that between runs: on a 24 GB card, expect roughly 8.6 GB resident once warm. The cached spike columns are a small addition on top of that.
- Disabling ECC on the card frees about 1 GB and some memory bandwidth. It needs a cold power cycle, not a reboot.

## Credits

Author: **Khai Bustos** ([github.com/KhaiB10](https://github.com/KhaiB10)), developed with Claude (Anthropic). Spike splitting follows *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* (Dettmers et al., 2022). MIT license.
