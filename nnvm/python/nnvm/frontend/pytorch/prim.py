import tvm
import re
import numpy as np
from .base import PyTorchOp


class PrimOp(PyTorchOp):

    def __init__(self, node, graph):
        super(PrimOp, self).__init__(node, graph)


class Constant(PrimOp):

    def __init__(self, node, graph):
        super(Constant, self).__init__(node, graph)
        output = next(node.outputs())
        type_kind = output.type().kind()
        value = self._parse_value_from_string()
        if type_kind == 'IntType':
            self._retval = int(value)
        elif type_kind == 'FloatType':
            self._retval = value
        elif type_kind == 'BoolType':
            self._retval = bool(value)
        elif type_kind == 'CompleteTensorType' and output.type().sizes() == []:
            shape = output.type().sizes()
            dtype = float
            self._retval = tvm.nd.array(value * np.ones(shape).astype(dtype))
        else:
            msg = 'Only "IntType", "FloatType", "BoolType", and ' \
                  '"CompleteTensorType" type-kinds are supported. For ' \
                  '"CompleteTensorType", type-sizes must be [].'
            raise RuntimeError(msg)

    def _parse_value_from_string(self):
        r'''For some reason, __getitem__ is sometimes stripped from the
        torch._C.Node objects.'''
        pattern = '(?<=value=)[^\]]+'
        string = str(self._node)
        return float(re.findall(pattern, string)[0].strip('{}'))

    def as_json(self):
        return self._retval

    def as_nnvm(self):
        return self._retval


class ListConstruct(PrimOp):

    def __init__(self, node, graph):
        super(ListConstruct, self).__init__(node, graph)

    def as_json(self):
        return [input.as_json() for input in self._inputs]

    def as_nnvm(self):
        return [input.as_nnvm() for input in self._inputs]


class Int(PrimOp):

    def __init__(self, node, graph):
        super(Int, self).__init__(node, graph)
        self._retval = int(self._inputs[0].as_nnvm().asnumpy())

    def as_json(self):
        return self._retval

    def as_nnvm(self):
        return self._retval

    
class NumToTensor(PrimOp):

    def __init__(self, node, graph):
        super(NumToTensor, self).__init__(node, graph)

    def as_nnvm(self):
        shape = []
        dtype = type(self._inputs[0])
        return tvm.nd.array(self._inputs[0].as_json() * np.ones(shape).astype(dtype))

    def as_json(self):
        return self._inputs[0]


class Undefined(PrimOp):

    def __init__(self, node, graph):
        super(Undefined, self).__init__(node, graph)

    def as_json(self):
        return None

    def as_nnvm(self):
        return None
