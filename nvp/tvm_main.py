import argparse

from tvm_lower import *
from tvm_parse import *
from tvm_visualize import *
from tvm_model import *

def Run(args):
    mod = gen_module(args.layer)
    mod = get_lowered_tir(mod)
    keys = [key for key in mod.functions.keys()]
    print("########## LOWERED TIR ##########")
    print(mod.functions[keys[1]])
    G = visit_stmts(mod.functions[keys[1]])
    save_graph_viz(G, args.viz)

    V = VectorProcessor(model="X220", graph=G)
    V.add_END_node_to_graph()
    V.add_weight_to_graph()
    save_graph_viz(G, args.viz)
    V.run_model()
    
def main():
    parser = argparse.ArgumentParser(description='Running TVM to predict NVP latency')
    parser.add_argument('-l', '--layer', required=True, help='Layer name (e.g., relu, leaky_relu, ...')
    parser.add_argument('-v', '--viz', default='type', help='How to visualize node details (e.g., type, name)')
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()

    Run(args)

if __name__ == '__main__':
    main()
