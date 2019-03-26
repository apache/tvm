import numpy as np
import tvm
from nnvm.compiler.graph_util import infer_shape
from nnvm.symbol import Variable, Symbol
from nnvm.graph import create
from nnvm.frontend.common import get_nnvm_op
from collections import OrderedDict
from .base import PyTorchOp


class ATenOp(PyTorchOp):

    def __init__(self, node, graph):
        super(ATenOp, self).__init__(node, graph)
        self.dtype = 'float32'

    @property
    def shape(self):
        if not hasattr(self, '_shape'):
            self._shape = list(infer_shape(create(self.as_nnvm()))[1][0])
        return self._shape[:]

    def as_nnvm(self):
        if not hasattr(self, '_as_nnvm_retval'):
            inputs = [self.graph[name].as_nnvm() for name in self.input_names]
            self._as_nnvm_retval = get_nnvm_op(self.topi_name)(*inputs,
                                                               **self.attrs)
        return self._as_nnvm_retval

    def as_json(self):
        if not hasattr(self, '_as_json_retval'):
            attrs = set(dir(self))
            class_name = type(self).__name__
            for k in ['name', 'kind', 'input_names', 'attrs']:
                if k not in attrs:
                    msg = 'Attribute {} of class {} not implemented.'
                    raise RuntimeError(msg.format(k, class_name))
            self._as_json_retval = {
                'topi_name': self.topi_name,
                'kind': self.kind,
                'name': self.name,
                'input_names': self.input_names,
                'attrs': self.attrs,
                'shape': self.shape,
                'dtype': self.dtype,
            }
        return self._as_json_retval


class _convolution(ATenOp):

    def __init__(self, node, graph):
        super(_convolution, self).__init__(node, graph)
        if self._inputs[6].as_json():
            self.topi_name = 'conv2d_transpose'
        else:
            self.topi_name = 'conv2d'
        input_name = self._input_name(0)
        weight_name = self._input_name(1)
        bias_name = self._input_name(2)
        self.input_names = [input_name, weight_name, bias_name]
        if self.graph[bias_name].as_json() is None:
            bias = np.zeros([self._inputs[1].shape[0]]).astype('float32')
            self.graph.add_param(bias_name, bias)
        weight_shape = self._inputs[1].shape
        self.attrs = {
            'channels': weight_shape[0],
            'kernel_size': weight_shape[2:],
            'strides': self._inputs[3].as_json(),
            'padding': self._inputs[4].as_json(),
            'dilation': self._inputs[5].as_json(),
            'groups': self._inputs[8].as_json(),
        }


class threshold_(ATenOp):

#    return tvm.compute(x.shape, lambda *i: tvm.max(x(*i), tvm.const(threshold, x.dtype)))
    def __init__(self, node, graph):
        super(threshold_, self).__init__(node, graph)
        self.topi_name = 'relu'
        self.input_names = [self._input_name(0)]
        self.attrs = {
            'threshold': self._inputs[1].as_json(),
            'constant': self._inputs[2].as_json(),
        }
        if self.attrs['threshold'] != self.attrs['constant']:
            msg = 'For aten::threshold_, threshold != constant is not ' \
                  'implemented.'
            raise RuntimeError(msg)


class constant_pad_nd(ATenOp):

    def __init__(self, node, graph):
        super(constant_pad_nd, self).__init__(node, graph)
        self.topi_name = 'pad'
        self.input_names = [self._input_name(0)]
        self.attrs = {
            'before_padding': self._inputs[1].as_json(),
            'after_padding': [0] * len(self.attrs['before_padding']),
            'val': self._inputs[2].as_json(),
        }


class contiguous(ATenOp):

    def __init__(self, node, graph):
        super(contiguous, self).__init__(node, graph)
        self.topi_name = 'identity'
        self.input_names = [self._input_name(0)]
        self.attrs = {}


class batch_norm(ATenOp):

    def __init__(self, node, graph):
        super(batch_norm, self).__init__(node, graph)
        self.topi_name = 'batch_norm'
        input_name = self._input_name(0)
        weight_name = self._input_name(1)
        bias_name = self._input_name(2)
        mean_name = self._input_name(3)
        std_name = self._input_name(4)
        self.input_names = [input_name, weight_name, bias_name,
                            mean_name, std_name]
        self.attrs = {
            'epsilon': self._inputs[7].as_json(),
        }


class max_pool2d_with_indices(ATenOp):

    def __init__(self, node, graph):
        super(max_pool2d_with_indices, self).__init__(node, graph)
        self.topi_name = 'max_pool2d'
        self.input_names = [self._input_name(0)]
        self.attrs = {
            'pool_size': self._inputs[1].as_json(),
            'strides': self._inputs[2].as_json(),
            'padding': self._inputs[3].as_json(),
            'ceil_mode': self._inputs[5].as_json(),
        }


class cat(ATenOp):

    def __init__(self, node, graph):
        super(cat, self).__init__(node, graph)
        self.topi_name = 'concatenate'
        self.input_names = [self._input_name(0)]
        self.attrs = {
            'axis': self._inputs[1].as_json(),
        }

    def as_nnvm(self):
        if not hasattr(self, '_as_nnvm_retval'):
            self._as_nnvm_retval = get_nnvm_op(self.topi_name)(*self._inputs[0].as_nnvm(), **self.attrs)
        return self._as_nnvm_retval


class t(ATenOp):

    def __init__(self, node, graph):
        super(t, self).__init__(node, graph)
        self.topi_name = 'transpose'
        self.input_names = [self._input_name(0)]
        self.attrs = {
            'axes': [1, 0],
        }


class size(ATenOp):

    def __init__(self, node, graph):
        super(size, self).__init__(node, graph)
        self.input_names = [self._input_name(0)]
        self.attrs =  {}
        self.topi_name = ''
        self._retval = self._inputs[0].shape[self._inputs[1].as_json()]

    def as_nnvm(self):
        return self._retval

    def as_json(self):
        return self._retval


class view(ATenOp):

    def __init__(self, node, graph):
        super(view, self).__init__(node, graph)
        self.topi_name = 'reshape'
        self.input_names = [self._input_name(0)]
        self.attrs = {
            'shape': self._inputs[1].as_json(),
        }


class select(ATenOp):

    def __init__(self, node, graph):
        super(select, self).__init__(node, graph)
        self.topi_name = 'strided_slice'
        self.input_names = [self._input_name(0)]
        self._dim = self._inputs[1].as_json()
        index = self._inputs[2].as_json()
        end = self.graph[self.input_names[0]].shape[:]
        end[self._dim] = index + 1
        begin = [0] * len(end)
        begin[self._dim] = index
        self.attrs = {
            'begin': begin,
            'end': end,
            'stride': 1,
        }

    @property
    def shape(self):
        if not hasattr(self, '_shape'):
            begin = np.array(self.attrs['begin']).astype(int)
            end = np.array(self.attrs['end']).astype(int)
            self._shape = (end - begin).tolist()
        return self._shape[:self._dim] + self._shape[self._dim + 1:]

    def as_nnvm(self):
        if not hasattr(self, '_as_nnvm_retval'):
            inputs = [self.graph[name].as_nnvm() for name in self.input_names]
            extra_name = self.name + '_before_squeeze'
            op = get_nnvm_op(self.topi_name)
            self.extras[extra_name] = op(*inputs, **self.attrs)
            inputs = [self.extras[extra_name]]
            attrs = {
                'axis': self._dim,
            }
            op = get_nnvm_op('squeeze')
            self._as_nnvm_retval = op(*inputs, **attrs)
        return self._as_nnvm_retval
            


class copy(ATenOp):

    def __init__(self, node, graph):
        super(copy, self).__init__(node, graph)
        self.topi_name = 'copy'
        self.input_names = [self._input_name(0)]
        self.attrs = {}


class clone(ATenOp):

    def __init__(self, node, graph):
        super(clone, self).__init__(node, graph)
        self.topi_name = 'copy'
        self.input_names = [self._input_name(0)]
        self.attrs = {}


class relu_(ATenOp):

    def __init__(self, node, graph):
        super(relu_, self).__init__(node, graph)
        self.topi_name = 'relu'
        self.input_names = [self._input_name(0)]
        self.attrs = {}


class sigmoid(ATenOp):

    def __init__(self, node, graph):
        super(sigmoid, self).__init__(node, graph)
        self.topi_name = 'sigmoid'
        self.input_names = [self._input_name(0)]
        self.attrs = {}


class addmm(ATenOp):

    def __init__(self, node, graph):
        super(addmm, self).__init__(node, graph)
        self.topi_name = 'dense'
        bias_name = self._input_name(0)
        data_name = self._input_name(1)
        weight_name = self._input_name(2)
        self.alpha = self._inputs[4].as_json()
        self.beta = self._inputs[3].as_json()
        self.input_names = [data_name, weight_name, bias_name]
        units = self._inputs[2].shape[1]
        self.attrs = {
            'units': units,
        }
        

    def as_nnvm(self):
        if not hasattr(self, '_as_nnvm_retval'):
            inputs = [self.graph[name].as_nnvm() for name in self.input_names]
            inputs[0] *= self.alpha
            inputs[1] = get_nnvm_op('transpose')(self.beta * inputs[1], axes=[1, 0])
            self.extras[self._input_name(1) + '_transposed'] = inputs[1]
            self._as_nnvm_retval = get_nnvm_op(self.topi_name)(*inputs, **self.attrs)
        return self._as_nnvm_retval
 


class avg_pool2d(ATenOp):

    def __init__(self, node, graph):
        super(avg_pool2d, self).__init__(node, graph)
        self.topi_name = 'avg_pool2d'
        self.input_names = [self._input_name(0)]
        self.attrs = {
            'pool_size': self._inputs[1].as_json(),
            'strides': self._inputs[2].as_json(),
            'padding': self._inputs[3].as_json() * 2,
            'ceil_mode': self._inputs[5].as_json(),
        }


class adaptive_avg_pool2d(ATenOp):

    def __init__(self, node, graph):
        super(adaptive_avg_pool2d, self).__init__(node, graph)
        self.topi_name = 'avg_pool2d'
        self.input_names = [self._input_name(0)]
        self.attrs = {
            'pool_size': self._pool_size,
            'strides': [1, 1],
            'padding': [0, 0],
            'ceil_mode': True,
        }

    @property
    def _pool_size(self):
        input_shape = np.array(self._inputs[0].shape).astype(int)
        output_shape = np.array(self.shape).astype(int)
        return (input_shape - output_shape + 1)[-2:].tolist()
        
    @property
    def shape(self):
        if not hasattr(self, '_shape'):
            self._shape = self._inputs[0].shape[:]
            self._shape[-2:] = self._inputs[1].as_json()
        return self._shape


class dropout(ATenOp):

    def __init__(self, node, graph):
        super(dropout, self).__init__(node, graph)
        self.topi_name = 'dropout'
        self.input_names = [self._input_name(0)]
        self.attrs = {
            'rate': self._inputs[1].as_json(),
        }


class slice(ATenOp):

    def __init__(self, node, graph):
        super(slice, self).__init__(node, graph)
        self.topi_name = 'strided_slice'
        self.input_names = [self._input_name(0)]
        end = self._inputs[0].shape[:]
        begin = [0] * len(end)
        dim = self._inputs[1].as_json()
        begin[dim] = self._inputs[2].as_json()
        end[dim] = min(end[dim], self._inputs[3].as_json())
        self.attrs = {
            'begin': begin,
            'end': end,
            'stride': self._inputs[4].as_json(),
        }

#    @property
#    def shape(self):
#        if not hasattr(self, '_shape'):
#            begin = np.array(self.attrs['begin']).astype(int)
#            end = np.array(self.attrs['end']).astype(int)
#            self._shape = (end - begin).tolist()
#        return self._shape[:]
    


class mul(ATenOp):

    def __init__(self, node, graph):
        super(mul, self).__init__(node, graph)
        self.topi_name = 'elemwise_mul'
        input_name0 = self._input_name(0)
        input_name1 = self._input_name(1)
        self.input_names = [input_name0, input_name1]
        self.attrs = {
            'num_args': 2,
        }

    def as_nnvm(self):
        if not hasattr(self, '_as_nnvm_retval'):
            inputs = []
            for name in self.input_names:
                other_name = [k for k in self.input_names if k != name][0]
                extra_name = name + '_tensor'
                if not isinstance(self.graph[name].as_nnvm(), Symbol):
                    val = self.graph[name].as_json()
                    if isinstance(val, tvm.nd.NDArray):
                        val = val.asnumpy()
                    else:
                        shape = self.graph[other_name].shape
                        val *= np.ones(shape).astype('float32')
                    self.graph.add_param(extra_name, val)
                    self.extras[extra_name] = Variable(name=extra_name,
                                                       init=val,
                                                       dtype=val.dtype,
                                                       shape=val.shape)
                if extra_name in self.extras:
                    inputs.append(self.extras[extra_name])
                else:
                    inputs.append(self.graph[name].as_nnvm())
            self._as_nnvm_retval = get_nnvm_op(self.topi_name)(*inputs,
                                                               **self.attrs)
        return self._as_nnvm_retval


class add(ATenOp):

    def __init__(self, node, graph):
        super(add, self).__init__(node, graph)
        self.topi_name = 'elemwise_sum'
        input_name0 = self._input_name(0)
        input_name1 = self._input_name(1)
        self.input_names = [input_name0, input_name1]
        self.attrs = {
            'num_args': 2,
        }

    def as_nnvm(self):
        if not hasattr(self, '_as_nnvm_retval'):
            inputs = []
            #TODO: handle case where both inputs are not tvm arrays
            for name in self.input_names:
                other_name = [k for k in self.input_names if k != name][0]
                extra_name = name + '_tensor'
                if not isinstance(self.graph[name].as_nnvm(), Symbol):
                    val = self.graph[name].as_json()
                    if isinstance(val, tvm.nd.NDArray):
                        val = val.asnumpy()
                    else:
                        shape = self.graph[other_name].shape
                        val *= np.ones(shape).astype('float32')
                    self.graph.add_param(extra_name, val)
                    self.extras[extra_name] = Variable(name=extra_name,
                                                       init=val,
                                                       dtype=val.dtype,
                                                       shape=val.shape)
                if extra_name in self.extras:
                    inputs.append(self.extras[extra_name])
                else:
                    inputs.append(self.graph[name].as_nnvm())
            self._as_nnvm_retval = get_nnvm_op(self.topi_name)(*inputs,
                                                               **self.attrs)
        return self._as_nnvm_retval


class add_(add):
    pass


class unsqueeze(ATenOp):

    def __init__(self, node, graph):
        super(unsqueeze, self).__init__(node, graph)
        self.topi_name = 'expand_dims'
        self.input_names = [self._input_name(0)]
        self.attrs = {
            'axis': self._inputs[1].as_json(),
        }

class expand(ATenOp):

    def __init__(self, node, graph):
        super(expand, self).__init__(node, graph)
        self.topi_name = 'expand'
        self.input_names = [self._input_name(0)]
        self.shape = self._inputs[1].as_json()

#def expand(node, *inputs):
#    op = inputs[0]
#    old_shape = infer_shape(op)
#    new_shape = inputs[1]
#    old_index = 0
#    for new_index in range(len(new_shape)):
#        if old_shape[old_index] != new_shape[new_index]:
#            op = get_nnvm_op('expand_dims', old_index)
#            old_index += 1
#            old_shape = old_shape[:old_index] + [1] + old_shape[old_index:]
#    return op
