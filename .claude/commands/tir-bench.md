Run kernel performance benchmarks to verify codegen changes.

## Kernels to benchmark

All commands use `--warmup 100 --repeat 30` for ~3-minute total runtime with reliable medians. Drop to defaults only when chasing a sub-2% regression.

- **GEMM**: square GEMM at M=N=K in {1024, 2048, 4096, 8192, 16384} for three variants:
  - fp16: `python -m tirx_kernels.bench --kernel fp16_bf16_gemm --warmup 100 --repeat 30`
  - fp8: `python -m tirx_kernels.bench --kernel fp8_blockwise_gemm --warmup 100 --repeat 30`
  - nvfp4: `python -m tirx_kernels.bench --kernel nvfp4_gemm --warmup 100 --repeat 30`
- **FA4** (flash_attention4): all registered configs
  - `python -m tirx_kernels.bench --kernel flash_attention4 --warmup 100 --repeat 30`
- **MQA logits** (fp8 / fp4): all registered configs
  - `python -m tirx_kernels.bench --kernel deepgemm_sm100_fp8_mqa_logits --warmup 100 --repeat 30`
  - `python -m tirx_kernels.bench --kernel deepgemm_sm100_fp4_mqa_logits --warmup 100 --repeat 30`

## Steps

1. Select the least busy GPU:
   ```bash
   export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
   ```

2. Run benchmarks for each kernel using the commands above.

3. Present results in a table: kernel x config, with times in ms.

## When to use

When modifying anything that affects code generation: kernels, op dispatches, lowering passes, codegen, device ops.

## Reference baseline

Captured 2026-05-17 on B200 (sm_100a), GPU 7, `warmup=100 repeat=30`, `timer=proton`.

- `tir`             @ `587f439c4c` (branch `scope-id`, with `feat(exec-scope): infer scope_id extent from sibling defs when omitted` on top of upstream tirx `c9ee147baf`)
- `tirx-kernels`    @ `fdab8ac5` (branch `scope-id`, with `perf(kernel): hoist mqa_fp8 warpgroup index` on top of upstream `ae8673c9`)

All times in us. `baseline/tirx` > 1 means TIRX faster.

### `fp16_bf16_gemm` (baseline=`torch-cublas`)


| config | torch-cublas | tir | baseline/tirx |
|---|---:|---:|---:|
| `fp16_1024x1024x1024` | 5.73us | 16.54us | 0.347 |
| `fp16_2048x2048x2048` | 16.40us | 27.91us | 0.588 |
| `fp16_4096x4096x4096` | 95.19us | 94.34us | 1.009 |
| `fp16_8192x8192x8192` | 823.15us | 843.04us | 0.976 |
| `fp16_16384x16384x16384` | 6093.33us | 6128.95us | 0.994 |
| `bf16_1024x1024x1024` | 5.72us | 16.51us | 0.347 |
| `bf16_2048x2048x2048` | 16.13us | 27.77us | 0.581 |
| `bf16_4096x4096x4096` | 92.25us | 91.35us | 1.010 |
| `bf16_8192x8192x8192` | 756.17us | 781.91us | 0.967 |
| `bf16_16384x16384x16384` | 5823.27us | 5809.98us | 1.002 |

### `fp8_blockwise_gemm` (baseline=`deepgemm`)


| config | deepgemm | tir | baseline/tirx |
|---|---:|---:|---:|
| `smoke_1024x1024x1024` | 6.07us | 5.91us | 1.026 |
| `deepgemm_m4096_n2112_k7168` | 49.86us | 48.96us | 1.018 |
| `deepgemm_m4096_n576_k7168` | 19.12us | 18.84us | 1.015 |
| `deepgemm_m4096_n24576_k1536` | 116.18us | 115.68us | 1.004 |
| `deepgemm_m4096_n32768_k512` | 75.54us | 71.28us | 1.060 |
| `deepgemm_m4096_n7168_k16384` | 320.22us | 329.80us | 0.971 |
| `deepgemm_m4096_n4096_k7168` | 83.19us | 82.69us | 1.006 |
| `deepgemm_m4096_n7168_k2048` | 44.04us | 43.59us | 1.010 |
| `stress_m8192_n7168_k4096` | 159.30us | 159.99us | 0.996 |

### `nvfp4_gemm` (baseline=`flashinfer`)


| config | flashinfer | tir | baseline/tirx |
|---|---:|---:|---:|
| `1024x1024x1024` | 5.13us | 6.59us | 0.778 |
| `2048x2048x2048` | 8.39us | 8.84us | 0.950 |
| `4096x4096x4096` | 32.50us | 30.56us | 1.064 |
| `8192x8192x8192` | 199.24us | 186.39us | 1.069 |
| `16384x16384x16384` | 2128.05us | 1511.81us | 1.408 |

### `flash_attention4` (baseline=`flashattn_sm100`)


| config | flashattn_sm100 | tir | baseline/tirx |
|---|---:|---:|---:|
| `s1024_h32kv4` | 20.34us | 20.80us | 0.978 |
| `s1024_h32kv4_causal` | 19.85us | 19.66us | 1.009 |
| `s1024_h32kv8` | 20.50us | 20.91us | 0.980 |
| `s1024_h32kv8_causal` | 19.85us | 19.75us | 1.005 |
| `s1024_h32kv16` | 20.51us | 21.05us | 0.974 |
| `s1024_h32kv16_causal` | 20.24us | 20.68us | 0.979 |
| `s1024_h32kv32` | 20.75us | 21.18us | 0.980 |
| `s1024_h32kv32_causal` | 21.07us | 22.24us | 0.947 |
| `s2048_h32kv4` | 59.47us | 60.85us | 0.977 |
| `s2048_h32kv4_causal` | 39.40us | 37.51us | 1.050 |
| `s2048_h32kv8` | 60.23us | 61.84us | 0.974 |
| `s2048_h32kv8_causal` | 39.49us | 37.76us | 1.046 |
| `s2048_h32kv16` | 60.60us | 62.83us | 0.965 |
| `s2048_h32kv16_causal` | 39.94us | 38.57us | 1.036 |
| `s2048_h32kv32` | 61.59us | 63.62us | 0.968 |
| `s2048_h32kv32_causal` | 40.29us | 42.38us | 0.951 |
| `s4096_h32kv4` | 203.59us | 204.89us | 0.994 |
| `s4096_h32kv4_causal` | 114.98us | 111.69us | 1.029 |
| `s4096_h32kv8` | 204.46us | 207.67us | 0.985 |
| `s4096_h32kv8_causal` | 116.24us | 112.45us | 1.034 |
| `s4096_h32kv16` | 208.31us | 211.63us | 0.984 |
| `s4096_h32kv16_causal` | 117.59us | 113.66us | 1.035 |
| `s4096_h32kv32` | 211.75us | 216.02us | 0.980 |
| `s4096_h32kv32_causal` | 118.98us | 122.09us | 0.975 |
| `s8192_h32kv4` | 816.39us | 818.33us | 0.998 |
| `s8192_h32kv4_causal` | 429.56us | 420.64us | 1.021 |
| `s8192_h32kv8` | 795.55us | 852.89us | 0.933 |
| `s8192_h32kv8_causal` | 411.97us | 440.47us | 0.935 |
| `s8192_h32kv16` | 779.83us | 841.29us | 0.927 |
| `s8192_h32kv16_causal` | 412.70us | 399.01us | 1.034 |
| `s8192_h32kv32` | 784.06us | 821.54us | 0.954 |
| `s8192_h32kv32_causal` | 459.55us | 420.57us | 1.093 |

### `deepgemm_sm100_fp8_mqa_logits` (baseline=`deepgemm`)


| config | deepgemm | tirx | baseline/tirx |
|---|---:|---:|---:|
| `s2048_skv4096_h64_d128_f32_dense_cp` | 43.80us | 44.49us | 0.984 |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | 58.50us | 58.59us | 0.999 |
| `s2048_skv8192_h64_d128_f32_dense_cp` | 77.25us | 78.07us | 0.990 |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | 118.40us | 118.97us | 0.995 |
| `s4096_skv4096_h64_d128_f32_dense_cp` | 78.02us | 77.94us | 1.001 |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | 77.89us | 78.37us | 0.994 |
| `s4096_skv8192_h64_d128_f32_dense_cp` | 136.98us | 136.12us | 1.006 |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | 196.36us | 202.57us | 0.969 |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | 46.60us | 44.88us | 1.038 |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | 61.46us | 59.54us | 1.032 |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | 81.83us | 78.99us | 1.036 |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | 125.40us | 120.15us | 1.044 |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | 83.89us | 78.42us | 1.070 |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | 83.94us | 78.89us | 1.064 |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | 147.25us | 137.97us | 1.067 |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | 209.79us | 196.89us | 1.066 |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | 44.73us | 44.81us | 0.998 |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | 58.90us | 59.29us | 0.993 |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | 79.48us | 79.03us | 1.006 |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | 121.27us | 121.16us | 1.001 |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | 78.87us | 78.84us | 1.000 |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | 79.02us | 78.66us | 1.005 |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | 139.18us | 138.40us | 1.006 |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | 199.50us | 197.53us | 1.010 |
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | 46.91us | 46.09us | 1.018 |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | 61.15us | 60.29us | 1.014 |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | 82.17us | 80.09us | 1.026 |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | 126.02us | 123.97us | 1.017 |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | 84.10us | 82.16us | 1.024 |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | 83.94us | 82.05us | 1.023 |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | 147.98us | 144.28us | 1.026 |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | 209.74us | 204.18us | 1.027 |

### `deepgemm_sm100_fp4_mqa_logits` (baseline=`deepgemm`)


| config | deepgemm | tirx | baseline/tirx |
|---|---:|---:|---:|
| `s2048_skv4096_h64_d128_f32_dense_cp` | 41.25us | 41.52us | 0.994 |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | 53.67us | 54.10us | 0.992 |
| `s2048_skv8192_h64_d128_f32_dense_cp` | 71.99us | 72.44us | 0.994 |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | 111.41us | 111.13us | 1.003 |
| `s4096_skv4096_h64_d128_f32_dense_cp` | 73.25us | 73.47us | 0.997 |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | 73.21us | 73.52us | 0.996 |
| `s4096_skv8192_h64_d128_f32_dense_cp` | 130.21us | 129.54us | 1.005 |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | 186.20us | 184.96us | 1.007 |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | 45.14us | 42.37us | 1.066 |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | 59.05us | 54.82us | 1.077 |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | 79.09us | 73.69us | 1.073 |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | 122.95us | 113.08us | 1.087 |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | 80.41us | 73.88us | 1.088 |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | 80.32us | 73.81us | 1.088 |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | 144.14us | 131.25us | 1.098 |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | 206.26us | 187.68us | 1.099 |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | 42.24us | 42.51us | 0.994 |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | 55.24us | 55.44us | 0.996 |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | 74.32us | 74.16us | 1.002 |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | 114.28us | 113.84us | 1.004 |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | 74.91us | 74.90us | 1.000 |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | 74.90us | 74.84us | 1.001 |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | 133.11us | 132.55us | 1.004 |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | 190.79us | 189.49us | 1.007 |
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | 44.99us | 45.73us | 0.984 |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | 59.06us | 60.01us | 0.984 |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | 79.27us | 80.35us | 0.987 |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | 122.57us | 123.86us | 0.990 |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | 79.93us | 81.00us | 0.987 |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | 79.78us | 80.97us | 0.985 |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | 142.89us | 144.28us | 0.990 |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | 204.95us | 206.88us | 0.991 |
