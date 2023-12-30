The flash-attention performance on RTX-4090 GPU, with cuda toolkit 12.2

SEQ_LEN is fixed to 2k, All matmul use fp16->fp32 mma, value in TFlops, higher is better.

Flash-Forward
| CASUAL,DIM | Flash_attn | Tvm.tl |
| ---------  | ---------- | ------ |
| False, 32  | 159.79     | 156.82 |
| False, 64  | 168.91     | 166.84 |
| False, 128 | 169.28     | 166.51 |
| False, 256 | 156.15     | 166.77 |
| True, 32   | 126.78     | 142.59 |
| True, 64   | 142.23     | 152.43 |
| True, 128  | 151.19     | 156.30 |
| True, 256  | 144.12     | 151.54 |

Flash-backward
| CASUAL,DIM | Flash_attn | Tvm.tl |
| ---------  | ---------- | ------ |
| False, 32  | 115.12     | 120.03 |
| False, 64  | 124.81     | 130.94 |
| False, 128 | 124.57     | 122.99 |
| True, 32   | 86.48     | 95.66  |
| True, 64   | 96.53     | 106.03 |
| True, 128  | 99.23     | 100.24 |
