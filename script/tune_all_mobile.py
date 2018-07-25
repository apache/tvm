import sys
import argparse
import os
import multiprocessing

networks = [
    'squeezenet',
    'mobilenet',
    'resnet-18',
    'vgg-16',
]

targets = [
    'rk3399-cpu',    #'rk3399-gpu',
    'rpi3b-cpu',     #'pynq-cpu',
    'hikey960-cpu',  #'hikey960-gpu',
    'mate10pro-cpu', #'mate10pro-gpu',
    'pixel2-cpu',    #'rk3399-gpu',
    'p20pro-cpu',     #'pynq-cpu',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default='')
    parser.add_argument("--mode", type=str, default='tune')
    args = parser.parse_args()

    targets = list(filter(lambda x: args.target in x, targets))

    cmds = []
    for network in networks:
        for target in targets:
            cmd = "python3 tune_nnvm.py --network %s --target %s --cache-file %s --n-trial 1000 --mode %s" \
                    % (network, target, network + "." + target + ".log", args.mode)

            print(cmd)
            if args.mode == 'infer':
                cmds.append(cmd)
            else:
                os.system(cmd)

    pool = multiprocessing.Pool()
    pool.map(os.system, cmds)

