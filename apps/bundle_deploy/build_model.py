"""Creates a simple TVM modules."""

import argparse
import os
import nnvm.compiler
import nnvm.testing
import tvm
import logging


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out-dir', default='.')
    opts = parser.parse_args()

    dshape = (1, 3, 224, 224)
    from mxnet.gluon.model_zoo.vision import get_model
    block = get_model('mobilenet0.25', pretrained=True)
    net, params = nnvm.frontend.from_mxnet(block)
    net = nnvm.sym.softmax(net)

    with nnvm.compiler.build_config(opt_level=3):
        graph, lib, params = nnvm.compiler.build(
            net, 'llvm --system-lib', shape={'data': dshape}, params=params)
    print(graph.symbol().debug_str())
    build_dir = os.path.abspath(opts.out_dir)
    if not os.path.isdir(build_dir):
        os.makedirs(build_dir)

    lib.save(os.path.join(build_dir, 'model.o'))
    with open(os.path.join(build_dir, 'graph.json'), 'w') as f_graph_json:
        f_graph_json.write(graph.json())
    with open(os.path.join(build_dir, 'params.bin'), 'wb') as f_params:
        f_params.write(nnvm.compiler.save_param_dict(params))


if __name__ == '__main__':
    main()
