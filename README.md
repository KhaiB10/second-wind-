# Second Wind

**Give an old Tesla P40 a second wind: run diffusion transformers about 2× faster, with the same picture.**

The Tesla P40 (Pascal, 2016) is a cheap 24 GB card, but it is weak at 16-bit floats, so image models run on it in slow 32-bit math. Its hidden strength is 8-bit integer math: on the matrix shapes inside Z-Image it measured **37.5–39.7 TOPS of INT8 against 8.8–9.5 TFLOPS of FP32, about 4×**. This ComfyUI node puts that INT8 speed to work.

| Tesla P40, Z-Image Turbo, 1024×1024, 8 steps | Time per picture | Speed |
| --- | --- | --- |
| Normal (FP32 math, FP8-stored weights) | 72.1 s | 1.0× |
| **Second Wind, `int8_spike_split`** | **36.1 s** | **2.0×** |

**Same picture:** drawn from the same seed, 97.6% of the INT8 picture's lines fall within 3 pixels of the normal-math picture's lines. Two different seeds score 81%.

![normal math vs INT8 vs INT8 with spike splitting](docs/compare.png)

## How it works

1. **Weights** are squeezed to 8 bits once, with one scale per output row, and cached on the GPU.
2. **Inputs** are squeezed to 8 bits on every call, with one scale per token, by a fused CUDA kernel. They are multiplied with `torch._int_mm` (INT8 × INT8 → INT32) and scaled back to float by a second fused kernel. Before fusing, the eight separate prep steps around each multiply ate most of the gain.
3. **Spike splitting.** A few input channels carry huge outliers, up to 84,000× the average in the feed-forward "down" layers. The node finds them during each layer's first three calls. From then on, those channels go through normal float math and the rest through INT8. This is the LLM.int8() decomposition idea (Dettmers et al., 2022) applied to an image model, and it halves the error in the worst layers.

## Install

1. Copy this folder into `ComfyUI/custom_nodes/second-wind/`.
2. The fused kernels compile on first load. That needs `nvcc` and `ninja` (`pip install ninja` in ComfyUI's Python). If either is missing, the node falls back to a slower PyTorch path and says so in the log.
3. Start ComfyUI with `--force-fp32`, and load the model with `weight_dtype = fp8_e4m3fn`.
4. Put the **Second Wind: P40 INT8 (2x faster)** node between the model loader and the sampler. Use mode `int8_spike_split`.

## What we tried that did not work

- **Token downsampling in attention (ToDo-style 2×2 key/value pooling).** 1.4× faster again, but it **smeared the linework into blocks**. Z-Image tags each token's position with RoPE before attention, and averaging four neighbours mixes up their position codes. The run with downsampling only in late steps scored 0.967 on our line-match metric and still looked bad. Lesson: judge by eye with side-by-side sheets, not by an edge-counting score.
- **Spike search on every call** improved accuracy but was slower (1.5×). Searching only in the first three calls and reusing the result kept the accuracy and the speed.

## Where the time goes now (GPU trace, one picture)

| | Normal math | Second Wind |
| --- | --- | --- |
| Matrix layers | 48.8 s | 10.1 s (INT8 kernel, 4.8× faster) |
| Attention | 13.9 s | 14.0 s, **now the biggest cost** |
| Squeeze and unsqueeze (fused kernels) | none | 2.5 s |

**Next:** an INT8 attention kernel for Pascal.

## Limits

- Tested on one card (Tesla P40) and one model (Z-Image Turbo, which ComfyUI loads as `NextDiT`). The node patches every `torch.nn.Linear` in the diffusion model, so other DiT models may work, but they are untested.
- INT8 draws a close sibling of the normal picture, not a bit-exact copy.
- The cached INT8 weights take about 6 GB of extra GPU memory.

## Credits

Author: **Khai Bustos** ([github.com/KhaiB10](https://github.com/KhaiB10)), developed with Claude (Anthropic). Spike splitting follows *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* (Dettmers et al., 2022). MIT license.
