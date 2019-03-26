from abc import ABC, abstractmethod
from collections import OrderedDict
from nnvm.symbol import Variable
import numpy as np


def _unique_name(node):
    return next(node.outputs()).uniqueName()


def _kind(node):
    return node.kind()


class PyTorchGraph:

    def __init__(self):
        self.inputs = OrderedDict()
        self.params = OrderedDict()
        self.ops = OrderedDict()

    def __getitem__(self, name):
        if name in self.inputs:
            return self.inputs[name]
        elif name in self.params:
            return self.params[name]
        elif name in self.ops:
            return self.ops[name]

    def add_input(self, name, tensor):
        self.inputs[name] = PyTorchInput(name, tensor)
        self.inputs[name].graph = self

    def add_param(self, name, tensor):
        self.params[name] = PyTorchParam(name, tensor.astype('float32'))
        self.params[name].graph = self

    def add_op(self, name, op):
        self.ops[name] = op


class PyTorchNode(ABC):

    @abstractmethod
    def as_nnvm(self):
        raise NotImplementedError

    @abstractmethod
    def as_json(self):
        raise NotImplementedError


class PyTorchConstantTensor(PyTorchNode):

    def __init__(self, name, arr):
        self.name = name
        self.arr = arr
        self.dtype = self.arr.dtype.name

    def as_nnvm(self):
        if not hasattr(self, '_as_nnvm_retval'):
            self._as_nnvm_retval = Variable(name=self.name, shape=self.shape,
                                            dtype=self.dtype)
        return self._as_nnvm_retval

    @property
    def shape(self):
        return list(self.arr.shape)

    def as_json(self):
        return {
            'kind': self.kind,
            'name': self.name,
            'val': self.arr.tolist(),
            'dtype': self.dtype,
            'shape': self.shape,
        }


class PyTorchInput(PyTorchConstantTensor):

    def __init__(self, name, arr):
        super(PyTorchInput, self).__init__(name, arr)
        self.kind = 'input'


class PyTorchParam(PyTorchConstantTensor):

    def __init__(self, name, arr):
        super(PyTorchParam, self).__init__(name, arr)
        self.kind = 'param'


class PyTorchOp(PyTorchNode):

    def __init__(self, node, graph):
        self.graph = graph
        self._inputs = [graph[input.uniqueName()] for input in node.inputs()]
        self._node = node
        self.kind = _kind(node)
        self.name = _unique_name(node)
        self.extras = OrderedDict()

    def _input_name(self, node_index):
        return self._inputs[node_index].name
