import nnvm
from tvm.contrib import util


def test_variable_node_parsed():
    sym = nnvm.sym.Variable('data')
    tempdir = util.tempdir()
    json_filename = 'test_nnvm_symbol.json'
    with open(tempdir.relpath(json_filename), 'w') as fo:
        fo.write(nnvm.graph.create(sym).json())
    sym_str = open(tempdir.relpath(json_filename), 'r').read()
    sym = nnvm.graph.load_json(sym_str).symbol()
    sym = nnvm.sym.relu(sym)


if __name__ == '__main__':
    test_variable_node_parsed()
