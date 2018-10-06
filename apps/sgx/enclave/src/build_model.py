"""Creates a simple TVM modules."""

import argparse
import os
from os import path as osp

import nnvm.compiler
import nnvm.testing
import tvm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out-dir', default='.')
    opts = parser.parse_args()

    # from tutorials/nnvm_quick_start.py
    dshape = (1, 3, 224, 224)
    net, params = nnvm.testing.resnet.get_workload(
        layers=18, batch_size=dshape[0], image_shape=dshape[1:])

    with nnvm.compiler.build_config(opt_level=3):
        graph, lib, params = nnvm.compiler.build(
            net, 'llvm --system-lib', shape={'data': dshape}, params=params)

    build_dir = osp.abspath(opts.out_dir)
    if not osp.isdir(build_dir):
        os.makedirs(build_dir, exist_ok=True)

    lib.save(osp.join(build_dir, 'model.bc'))
    with open(osp.join(build_dir, 'graph.json'), 'w') as f_graph_json:
        f_graph_json.write(graph.json())
        with open(osp.join(build_dir, 'params.bin'), 'wb') as f_params:
            f_params.write(nnvm.compiler.save_param_dict(params))


if __name__ == '__main__':
    main()
