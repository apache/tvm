import argparse
import os, glob
from tvm_model import *
from tvm_visualize import *
 
def main():
    parser = argparse.ArgumentParser(description='Running TVM to predict NVP latency')
    parser.add_argument('-l', '--layer', required=True, help='Layer name (e.g., relu, leaky_relu, dwconv, ...')
    parser.add_argument('--layout', default='NCHW', help='Layout (e.g., NCHW, NHWC, ...)')
    parser.add_argument('--input', default=(1, 8, 5, 5), nargs='+', type=int, help='Input shape')
    parser.add_argument('--kernel', default=(3, 3), nargs='+', type=int, help='Kernel shape')
    parser.add_argument('-m', '--model', required=True, help='Model name (e.g., x220, llvm')
    parser.add_argument('-v', '--viz', default=None, help='How to visualize node details (e.g., type, typeNum, name)')
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()


    for png in glob.glob('./graph*.png'):
        try:
            os.remove(png)
        except:
            print("Error while deleting file: %s"%(png))

    Model = VectorProcessor(args.model, args.debug)
    Model.generate_graph(args.layer, args.layout, args.input, args.kernel)
    Model.optimize_graph()
    if args.viz: save_graph_viz(Model.graph, args.viz)
    Model.run()

    print("#"*40)
    print('{message: <39}'.format(message='# Model:  %s'%(args.model)), end=''), print("#")
    print('{message: <39}'.format(message='# Layer:  %s'%(args.layer)), end=''), print("#")
    print('{message: <39}'.format(message='# Layout: %s'%(args.layout)), end=''), print("#")
    print('{message: <39}'.format(message='# Shape:  %s'%(str(args.input))), end=''), print("#")
    print('{message: <39}'.format(message='# Kernel: %s'%(str(args.kernel))), end=''), print("#")
    print('{message: <39}'.format(message='# >> Cycle:  %s'%(str(Model.estimated_cycles))), end=''), print("#")
    print("#"*40)

if __name__ == '__main__':
    main()
