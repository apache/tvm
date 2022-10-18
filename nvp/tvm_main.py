import argparse
import os
from tvm_layer import *
from tvm_lower import *
from tvm_parse import *
from tvm_model import *
from tvm_visualize import *

def Run(args):
    #1. Args --> Relay IR module
    RelayIRModule = gen_module(args.layer)

    #2. Relay IR module --> nested TIR
    nestedTIR = get_lowered_tir(RelayIRModule, args.model)
    print(nestedTIR)

    #3. nested TIR --> Empty Graph
    G = visit_stmts(nestedTIR, args.debug)

    #4. Empty Graph -> Colored Graph
    V = VectorProcessor(model=args.model, graph=G, debug=args.debug)
    V.color_graph()
    V.loop_hoisting()
    V.remove_nodes(attr='type', str='Seq')
    V.remove_nodes(attr='slot', str='Scalar')
    save_graph_viz(V.graph, args.viz)

    #5. Colored Graph -> Cycle
    V.run_single_iteration()
    V.get_estimated_cycles()
 
def main():
    parser = argparse.ArgumentParser(description='Running TVM to predict NVP latency')
    parser.add_argument('-l', '--layer', required=True, help='Layer name (e.g., relu, leaky_relu, dwconv, ...')
    parser.add_argument('-m', '--model', required=True, help='Model name (e.g., x220, llvm')
    parser.add_argument('-v', '--viz', default='type', help='How to visualize node details (e.g., type, typeNum, name)')
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()

    os.system("rm graph*.png")
    Run(args)

if __name__ == '__main__':
    main()
