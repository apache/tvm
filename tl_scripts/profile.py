import subprocess

configurations = [
  {'m': 2048, 'n': 2048, 'k': 2048, 'bm': 64, 'bn': 64, 'bk': 32},
  {'m': 4096, 'n': 4096, 'k': 4096, 'bm': 128, 'bn': 128, 'bk': 32},
  {'m': 8192, 'n': 8192, 'k': 8192, 'bm': 128, 'bn': 128, 'bk': 32},
  {'m': 16384, 'n': 16384, 'k': 16384, 'bm': 128, 'bn': 128, 'bk': 32},
  {'m': 8192, 'n': 8192, 'k': 1024, 'bm': 128, 'bn': 128, 'bk': 64},
  {'m': 8192, 'n': 8192, 'k': 2048, 'bm': 128, 'bn': 128, 'bk': 64},
  {'m': 8192, 'n': 8192, 'k': 4096, 'bm': 128, 'bn': 128, 'bk': 64},
]

for config in configurations:
    cmd = [
        'python', '/home/t-yucheng/tvm.tl/tl_scripts/gemm_example.py',
        '--m', str(config['m']),
        '--n', str(config['n']),
        '--k', str(config['k']),
    ]
    print(f"M:{config['m']}, N:{config['n']}, K:{config['k']}", flush=True)
    try:
        subprocess.run(cmd)
    except:
        print("Failed.")
    print('-'*100)

batches = [1, 64]
configurations = [
    {'model': 'GPT2', 'h': 12, 'n_ctx': 1024, 'd_head': 64},
    {'model': 'BERT-small', 'h': 8, 'n_ctx': 512, 'd_head': 64},
    {'model': 'BERT-base', 'h': 12, 'n_ctx': 512, 'd_head': 64},
    {'model': 'BERT-large', 'h': 16, 'n_ctx': 512, 'd_head': 64},
    {'model': 'Llamma2-7B', 'h': 32, 'n_ctx': 4096, 'd_head': 128},
    {'model': 'Llamma2-13B', 'h': 40, 'n_ctx': 4096, 'd_head': 128},
    {'model': 'Llamma2-70B', 'h': 64, 'n_ctx': 4096, 'd_head': 128},
    {'model': 'OPT-350M', 'h': 16, 'n_ctx': 2048, 'd_head': 64},
    {'model': 'OPT-13B', 'h': 40, 'n_ctx': 2048, 'd_head': 128},
    {'model': 'OPT-175B', 'h': 96, 'n_ctx': 2048, 'd_head': 128},
    {'model': 'DiT-S-2', 'h': 6, 'n_ctx': 1024, 'd_head': 64},
    {'model': 'DiT-B-2', 'h': 12, 'n_ctx': 1024, 'd_head': 64},
    {'model': 'DiT-L-2', 'h': 16, 'n_ctx': 1024, 'd_head': 64},
]

for batch in batches:
    for config in configurations:
        cmd = [
                'python', '/home/t-yucheng/tvm.tl/tl_scripts/mha_example.py',
                '--batch', str(batch),
                '--h', str(config['h']),
                '--n_ctx', str(config['n_ctx']),
                '--d_head', str(config['d_head'])
            ]
        print(config['model'], flush=True)
        try:
            subprocess.run(cmd)
        except:
            print("Failed.")
        print('-'*100)

# ncu profile:
# sudo -E env PATH=$PATH PYTHONPATH=$PYTHONPATH TMPDIR=~/ncu_tmp ncu --set full -k regex:"main_kernel" --launch-count 1 --launch-skip 500 -f -o reports/gemm_main_kernel_only /home/cy/miniconda3/envs/tl/bin/python tl_scripts/gemm_example.py