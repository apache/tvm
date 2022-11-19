import argparse
import os, glob
from tvm_model import *
from tvm_visualize import *

"""
  TODO
    1. Message print part
    2. Eadd - vlength=8 --> unroll
    # 3. tvm_parse.py --> Acc
    4. tvm_parse.py --> Seq --> Delete consecutive max, load
"""
 
def main():
    parser = argparse.ArgumentParser(description='Running TVM to predict NVP latency')
    parser.add_argument('-l', '--layer', required=True, help='Layer name (e.g., relu, leaky_relu, dwconv, ...')
    parser.add_argument('--layout', default='NCHW', help='Layout (e.g., NCHW, NHWC, ...)')
    parser.add_argument('--data', default=(1, 8, 5, 5), nargs='+', type=int, help='Data shape')
    parser.add_argument('--kernel', default=(3, 3), nargs='+', type=int, help='Kernel shape')
    parser.add_argument('--stride', default=(1, 1), nargs='+', type=int, help='Strides')
    parser.add_argument('--swp', default=True, type=int, help='Software Pipeline')
    parser.add_argument('-m', '--model', required=True, help='Model name (e.g., x220, llvm')
    parser.add_argument('-v', '--viz', default=None, help='How to visualize node details (e.g., type, typeNum, name)')
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()
    args.data = tuple(args.data)
    args.kernel = tuple(args.kernel)
    args.stride = tuple(args.stride)

    for png in glob.glob('./graph*.png'):
        try:
            os.remove(png)
        except:
            print("Error while deleting file: %s"%(png))

    Model = VectorProcessor(args.model, args.debug)
    Model.generate_graph(args.layer, args.layout, args.data, args.kernel, args.stride)
    Model.optimize_graph()
    if args.viz: save_graph_viz(Model.graph, args.viz)
    Model.run(args.swp)

    print("#"*40)
    print('{message: <39}'.format(message='# Model:  %s'%(args.model)), end=''), print("#")
    print('{message: <39}'.format(message='# Layer:  %s'%(args.layer)), end=''), print("#")
    print('{message: <39}'.format(message='# Layout: %s'%(args.layout)), end=''), print("#")
    print('{message: <39}'.format(message='# Shape:  %s'%(str(args.data))), end=''), print("#")
    print('{message: <39}'.format(message='# Kernel: %s'%(str(args.kernel))), end=''), print("#")
    print('{message: <39}'.format(message='# Stride: %s'%(str(args.stride))), end=''), print("#")
    print('{message: <39}'.format(message='# SWP:    %s'%('ON' if args.swp==True else 'OFF')), end=''), print("#")
    print('{message: <39}'.format(message='# >> Cycle:  %s'%(str(Model.estimated_cycles))), end=''), print("#")
    print("#"*40)

if __name__ == '__main__':
    main()
