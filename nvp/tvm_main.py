import argparse
import os
from tvm_layer import *
from tvm_parse import *
from tvm_model import *
from tvm_visualize import *

def Run(args):
    #1. Args --> Relay IR module
    #2. Relay IR module --> nested TIR
    nestedTIR = gen_module(args.model, args.layer, args.layout, args.input, args.kernel)
    print(nestedTIR)

    #3. nested TIR --> Empty Graph
    G = visit_stmts(nestedTIR, args.debug)

    #4. Empty Graph -> Colored Graph
    V = VectorProcessor(model=args.model, graph=G, debug=args.debug)
    V.color_graph()
    V.loop_hoisting() # Hoist loop variables & Remove scalar nodes
    V.remove_tensor_init_nodes(['Input', 'Filter', 'Multiplier', 'Shifter'])
    V.remove_nodes(attr='type', str='Seq')
    V.remove_nodes(attr='name', str='Multiplier')
    V.remove_nodes(attr='name', str='Shifter')
    save_graph_viz(V.graph, args.viz)

    #5. Colored Graph -> Cycle
    V.optimize('vstore')
    V.optimize('filter load') # For kernels that use filter; e.g, Dwconv
    V.run_single_iteration()
    V.get_estimated_cycles()
 
def main():
    parser = argparse.ArgumentParser(description='Running TVM to predict NVP latency')
    parser.add_argument('-l', '--layer', required=True, help='Layer name (e.g., relu, leaky_relu, dwconv, ...')
    parser.add_argument('--layout', default='NCHW', help='Layout (e.g., NCHW, NHWC, ...)')
    parser.add_argument('--input', default=(1, 8, 5, 5), nargs='+', type=int, help='Input shape')
    parser.add_argument('--kernel', default=(3, 3), nargs='+', type=int, help='Kernel shape')
    parser.add_argument('-m', '--model', required=True, help='Model name (e.g., x220, llvm')
    parser.add_argument('-v', '--viz', default='type', help='How to visualize node details (e.g., type, typeNum, name)')
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()

    print("#"*40)
    print('{message: <39}'.format(message='# Layer:  %s'%(args.layer)), end=''), print("#")
    print('{message: <39}'.format(message='# Layout: %s'%(args.layout)), end=''), print("#")
    print('{message: <39}'.format(message='# Shape:  %s'%(str(args.input))), end=''), print("#")
    print('{message: <39}'.format(message='# Kernel: %s'%(str(args.kernel))), end=''), print("#")
    print('{message: <39}'.format(message='# Mdoel:  %s'%(args.model)), end=''), print("#")
    print("#"*40)

    os.system("rm graph*.png")
    Run(args)

if __name__ == '__main__':
    main()
