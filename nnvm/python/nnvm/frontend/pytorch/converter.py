import tvm
import nnvm
import numpy as np
import torch
from . import aten
from . import prim
from .base import _kind, _unique_name, PyTorchGraph
from ..common import get_nnvm_op
from nnvm.symbol import Variable
from collections import OrderedDict

def check(k, v):
    from functools import reduce
    from operator import mul
    if hasattr(v, 'shape') and v.shape and reduce(mul, v.shape) == 357604:
        print(f'FOUND MATCH: {k}')

def operator_map(kind):
    namespace, op_name = kind.split('::')
    module = {
        'aten': aten,
        'prim': prim,
    }[namespace]
    return getattr(module, op_name)


class PyTorchConverter:

    def __init__(self, filename, input_shapes):
        self._load_model(filename, input_shapes)
        self._num_inputs = len(input_shapes)
        self.graph = PyTorchGraph()
        self._parse_inputs(input_shapes)
        self._parse_params()
        self._parse_ops()

    def _load_model(self, filename, input_shapes):
        self._trace = torch.jit.load(filename).float().eval()
        shapes = [input_shapes[k] for k in sorted(input_shapes)]
        inputs = [torch.zeros(shape).float() for shape in shapes]
        self._trace = torch.jit.trace(self._trace, *inputs).float().eval()
        ops = {node.kind() for node in self._trace.graph.nodes()}
#        for k in ops:
#            print(k)
#        print(self._trace.graph)

    @property
    def _ir_tensor_names(self):
        return [i.uniqueName() for i in self._trace.graph.inputs()]

    def _parse_inputs(self, input_shapes):
        input_names = sorted(input_shapes)
        ir_names = self._ir_tensor_names[:self._num_inputs]
        ir_names_set = set(ir_names)
        ir_name_map = dict(zip(input_names, ir_names))
        inv_ir_name_map = dict((v, k) for k, v in ir_name_map.items())
        for i, input in enumerate(self._trace.graph.inputs()):
            if i >= self._num_inputs:
                break
            ir_name = input.uniqueName()
            if ir_name in inv_ir_name_map:
                input.setUniqueName(inv_ir_name_map[ir_name])
        for input_name in sorted(input_shapes):
            input_shape = input_shapes[input_name]
            tensor = np.zeros(input_shape).astype(np.float32)
            ir_name = ir_name_map[input_name]
            for input in self._trace.graph.inputs():
                if input.uniqueName() == ir_name:
                    input.setUniqueName(input_name)
                    break
            self.graph.add_input(input_name, tensor)
            check(input_name, self.graph[input_name])
            

    def _parse_params(self):
        state_dict = self._trace.state_dict()
        state_dict_names = list(state_dict.keys())
        ir_names = self._ir_tensor_names[self._num_inputs:]
        name_map = dict(zip(state_dict_names, ir_names))
        for state_dict_name, param in state_dict.items():
            ir_name = name_map[state_dict_name]
            tensor = param.numpy()
            self.graph.add_param(ir_name, tensor)
            check(ir_name, self.graph[ir_name])

    def _parse_ops(self):
        for node in self._trace.graph.nodes():
            kind = _kind(node)
            name = _unique_name(node)
            self.graph.add_op(name, operator_map(kind)(node, self.graph))
            check(name, self.graph[name])

    def convert(self):
        sym_dict = lambda d: OrderedDict({k: v.as_nnvm() for k, v in d.items()})
        sym = sym_dict(self.graph.inputs)
        sym.update(sym_dict(self.graph.params))
        for name, op in self.graph.ops.items():
            sym.update(op.extras)
            sym[name] = op.as_nnvm()
        params = {k: tvm.nd.array(v.arr) for k, v in self.graph.params.items()}
        return sym[list(sym.keys())[-1]], params


def from_pytorch(filename, input_shapes):
    converter = PyTorchConverter(filename, input_shapes)
    sym, params = converter.convert()
    return sym, params
