# This is taken from WIP PR by AWS

import numpy as np

import tvm
from tvm import relay
from tvm.relay import expr as _expr
from tvm.relay import op as _op
from tvm.relay.frontend.common import get_relay_op
from tvm.relay.frontend.common import infer_shape as _infer_shape
from tvm.relay.frontend.common import infer_value as _infer_value
#from tvm.relay.prelude import Prelude


#mod = relay.Module()
#p = Prelude(mod)


def wrap_const(c):
    if not isinstance(c, _expr.Expr) and not isinstance(c, list):
        return _expr.const(c)
    return c


# operator implementation
def _elemwise(name):
    def _impl(inputs, input_types):
        data0 = wrap_const(inputs[0])
        data1 = wrap_const(inputs[1])

        if not isinstance(data0, (_expr.Call, _expr.TupleGetItem, _expr.Var)):
            temp = data0
            data0 = data1
            data1 = temp

        return get_relay_op(name)(data0, data1)
    return _impl

def _unsqueeze():
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = inputs[1]

        return _op.transform.expand_dims(data, int(axis), 1)
    return _impl

def _concatenate():
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = inputs[1]

        if isinstance(data, (_expr.Call, _expr.TupleGetItem, _expr.Var)):
            data = [data]

        return _op.tensor.concatenate(data, int(axis))
    return _impl

def _slice():
    def _impl(inputs, input_types):
        data = inputs[0]
        strides = []

        inferred_shape = _infer_shape(data)
        end = []
        for infer in inferred_shape:
            end.append(int(infer))
        if isinstance(data, _expr.Var):
            end = _infer_shape(data)
            end = list(end)

        begin = [0]*len(end)
        dim = int(inputs[1])
        begin[dim] = int(inputs[2])
        end[dim] = min(end[dim], inputs[3])

        strides.append(int(inputs[4]))
        return _op.transform.strided_slice(data, begin, end, strides)
    return _impl

def _select():
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = int(inputs[1])
        index = wrap_const(inputs[2])
        return _op.transform.take(data, index, axis=axis)
    return _impl

def _convert_data_type(input_type):
    if input_type == 'double' or input_type == 'torch.float64':
        return 'float64'
    elif input_type == 'float' or input_type == 'torch.float32':
        return 'float32'
    elif input_type == 'half' or input_type == 'torch.float16':
        return 'float16'
    elif input_type == 'long' or input_type == 'torch.int64':
        return 'int64'
    elif input_type == 'int' or input_type == 'torch.int32':
        return 'int32'
    elif input_type == 'short' or input_type == 'torch.int16':
        return 'int16'
    elif input_type == 'char' or input_type == 'torch.int8':
        return 'int8'
    elif input_type == 'byte' or input_type == 'torch.uint8':
        return 'uint8'
    else:
        return input_type

def _ones():
    def _impl(inputs, input_types):
        if isinstance(inputs[0], _expr.Var):
            shape = _infer_shape(inputs[0])
        elif isinstance(inputs[0], (_expr.Call, _expr.TupleGetItem)):
            shape = _infer_shape(inputs[0])
        else:
            shape = inputs[0].shape

        fill_value = _get_fill_value(input_types, 1)

        return get_relay_op('full')(fill_value, shape, dtype="float32")
    return _impl

def _zeros():
    def _impl(inputs, input_types):
        if isinstance(inputs[0], _expr.Var):
            shape = _infer_shape(inputs[0])
        elif isinstance(inputs[0], (_expr.Call, _expr.TupleGetItem)):
            shape = _infer_shape(inputs[0])
        elif isinstance(inputs[0], (list, tuple)):
            shape = inputs[0]
        else:
            shape = inputs[0].shape

        fill_value = _get_fill_value(input_types, 0)
        return _op.full(fill_value, shape, dtype="float32")
    return _impl

def _get_fill_value(input_types, int_val):
    if input_types[0] == 'int':
        fill_value = _expr.const(int_val)
    elif input_types[0] == 'float':
        fill_value = _expr.const(float(int_val))
    else:
        fill_value = _expr.const(float(int_val))

    return fill_value

def _relu():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.nn.relu(data)
    return _impl

def _adaptive_avg_2d():
    def _impl(inputs, input_types):
        data = inputs[0]
        output_size = _infer_shape(inputs[1])

        return _op.contrib.contrib.adaptive_avg_pool2d(
            data,
            output_size=output_size)
    return _impl

def _adaptive_max_2d():
    def _impl(inputs, input_types):
        data = inputs[0]
        output_size = _infer_shape(inputs[1])

        return _op.contrib.contrib.adaptive_max_pool2d(
            data,
            output_size=output_size)
    return _impl

def _maxpool_2d():
    def _impl(inputs, input_types):
        data = inputs[0]

        pool_size = _infer_shape(inputs[1])
        strides = _infer_shape(inputs[2])
        padding = _infer_shape(inputs[3])

        ceil_mode = int(inputs[5])

        return _op.nn.max_pool2d(data, pool_size, strides, padding, "NCHW", ceil_mode)
    return _impl

def _hardtanh():
    def _impl(inputs, input_types):
        a = inputs[0]
        tanh_min = float(inputs[1])
        tanh_max = float(inputs[2])
        return _op.tensor.clip(a, tanh_min, tanh_max)
    return _impl

def _convolution():
    def _impl(inputs, input_types):
        # Use transpose or normal
        use_transpose = False
        if inputs[6] == '1':
            use_transpose = True

        use_bias = False
        if isinstance(inputs[2], _expr.Var):
            use_bias = True

            data = inputs[0]
            weight = inputs[1]
            bias = inputs[2]

            if isinstance(weight, (_expr.Call, _expr.Var, _expr.TupleGetItem)):
                inferred_shape = _infer_shape(weight)
                weight_shape = []
                for infer in inferred_shape:
                    weight_shape.append(infer)
            else:
                weight_shape = weight.shape
            channels = weight_shape[0]

            strides = inputs[3]
            padding = inputs[4]
            dilation = inputs[5]

            kernel_size = weight_shape[2:]

        else:
            data = inputs[0]
            weight = inputs[1]
            bias = inputs[2]

            if isinstance(weight, (_expr.Call, _expr.Var, _expr.TupleGetItem)):
                inferred_shape = _infer_shape(weight)
                weight_shape = []
                for infer in inferred_shape:
                    weight_shape.append(infer)
            else:
                weight_shape = weight.shape
            channels = weight_shape[0]

            strides = inputs[3]
            padding = inputs[4]
            dilation = inputs[5]

            kernel_size = weight_shape[2:]

        if isinstance(strides, _expr.Var):
            strides = _infer_shape(strides)

        if isinstance(padding, _expr.Var):
            padding = _infer_shape(padding)

        if isinstance(dilation, _expr.Var):
            dilation = _infer_shape(dilation)

        groups = int(inputs[8])

        if use_transpose:
            conv_out = _op.nn.conv2d_transpose(data,
                                               weight,
                                               strides=strides,
                                               padding=padding,
                                               dilation=dilation,
                                               groups=groups,
                                               channels=channels,
                                               kernel_size=kernel_size,
                                               data_layout="NCHW",
                                               kernel_layout="OIHW",
                                               out_layout="",
                                               out_dtype="")
        else:
            conv_out = _op.nn.conv2d(data,
                                     weight,
                                     strides=strides,
                                     padding=padding,
                                     dilation=dilation,
                                     groups=groups,
                                     channels=channels,
                                     kernel_size=kernel_size,
                                     data_layout="NCHW",
                                     kernel_layout="OIHW",
                                     out_layout="",
                                     out_dtype="")

        if use_bias:
            return _op.nn.bias_add(conv_out, bias)
        else:
            return conv_out
    return _impl

def _softmax():
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = inputs[1]
        if isinstance(axis, str):
            axis = int(axis)

        return _op.nn.softmax(data, axis=axis)
    return _impl

def _threshold():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.nn.relu(data)
    return _impl

def _contiguous():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.tensor.copy(data)
    return _impl

def _batch_norm():
    def _impl(inputs, input_types):
        data = inputs[0]
        data_type = input_types[0]

        channels = _infer_shape(data)

        if isinstance(inputs[1], _expr.Var) and isinstance(inputs[2], _expr.Var):
            scale = center = True
            weight = inputs[1]
            beta = inputs[2]
        else:
            scale = center = False

        if scale:
            gamma = weight
        else:
            if data_type == 'double':
                gamma = _expr.const(np.ones([int(channels[1])]).astype('float64'))
            elif data_type == 'float':
                gamma = _expr.const(np.ones([int(channels[1])]).astype('float32'))
            elif data_type == 'half':
                gamma = _expr.const(np.ones([int(channels[1])]).astype('float16'))
            elif data_type == 'long':
                gamma = _expr.const(np.ones([int(channels[1])]).astype('int64'))
            elif data_type == 'int':
                gamma = _expr.const(np.ones([int(channels[1])]).astype('int32'))
            elif data_type == 'short':
                gamma = _expr.const(np.ones([int(channels[1])]).astype('int16'))
            elif data_type == 'char':
                gamma = _expr.const(np.ones([int(channels[1])]).astype('int8'))
            elif data_type == 'byte':
                gamma = _expr.const(np.ones([int(channels[1])]).astype('uint8'))

        if center:
            beta = beta
        else:
            if data_type == 'double':
                beta = _expr.const(np.zeros([int(channels[1])]).astype('float64'))
            elif data_type == 'float':
                beta = _expr.const(np.zeros([int(channels[1])]).astype('float32'))
            elif data_type == 'half':
                beta = _expr.const(np.zeros([int(channels[1])]).astype('float16'))
            elif data_type == 'long':
                beta = _expr.const(np.zeros([int(channels[1])]).astype('int64'))
            elif data_type == 'int':
                beta = _expr.const(np.zeros([int(channels[1])]).astype('int32'))
            elif data_type == 'short':
                beta = _expr.const(np.zeros([int(channels[1])]).astype('int16'))
            elif data_type == 'char':
                beta = _expr.const(np.zeros([int(channels[1])]).astype('int8'))
            elif data_type == 'byte':
                beta = _expr.const(np.zeros([int(channels[1])]).astype('uint8'))

        moving_mean = inputs[3]
        moving_var = inputs[4]
        epsilon = float(inputs[7])

        center = center
        scale = scale

        return _op.nn.batch_norm(data,
                                 gamma,
                                 beta,
                                 moving_mean,
                                 moving_var,
                                 axis=1,
                                 epsilon=epsilon,
                                 center=center,
                                 scale=scale)[0]
    return _impl

def _transpose():
    def _impl(inputs, input_types):
        data = inputs[0]

        if isinstance(data, _expr.Var):
            ndims = len(_infer_shape(data))
        elif isinstance(data, (_expr.Call, _expr.TupleGetItem)):
            ndims = _infer_shape(data)
        else:
            ndims = data.shape

        if isinstance(data, tvm.ndarray.NDArray):
            ndims = len(data.shape)
        axes = list(range(ndims))

        num_inputs = len(inputs)

        if num_inputs == 1:
            if ndims >= 2:
                axes[-1] = ndims - 2
                axes[-2] = ndims - 1
            if not isinstance(data, _expr.Var):
                data = _expr.const(data)

        elif num_inputs == 3:
            parse = lambda i: ndims * (i < 0) + i
            src, dst = [parse(int(inputs[i])) for i in [1, 2]]
            axes[src] = dst
            axes[dst] = src
        else:
            axes = inputs[1]
        return _op.transform.transpose(data, axes)
    return _impl

def _flatten():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.nn.batch_flatten(data)
    return _impl

def _dense():
    def _impl(inputs, input_types):
        use_bias = False

        if isinstance(inputs[0], _expr.Var):
            use_bias = True

        data = inputs[1]
        data_type = input_types[1]
        weight = inputs[2]

        beta = inputs[3]
        alpha = inputs[4]

        if not isinstance(alpha, (_expr.Var, _expr.Call, _expr.TupleGetItem)):
            if data_type == 'double':
                alpha = _expr.const(np.float64(alpha), dtype='float64')
            elif data_type == 'float':
                alpha = _expr.const(np.float32(alpha), dtype='float32')
            elif data_type == 'half':
                alpha = _expr.const(np.float16(alpha), dtype='float16')
            elif data_type == 'long':
                alpha = _expr.const(np.int64(alpha), dtype='int64')
            elif data_type == 'int':
                alpha = _expr.const(np.int32(alpha), dtype='int32')
            elif data_type == 'short':
                alpha = _expr.const(np.int16(alpha), dtype='int16')
            elif data_type == 'char':
                alpha = _expr.const(np.int8(alpha), dtype='int8')
            elif data_type == 'byte':
                alpha = _expr.const(np.uint8(alpha), dtype='uint8')
            data *= alpha

        if not isinstance(beta, (_expr.Var, _expr.Call, _expr.TupleGetItem)):
            if data_type == 'double':
                beta = _expr.const(np.float64(beta), dtype='float64')
            elif data_type == 'float':
                beta = _expr.const(np.float32(beta), dtype='float32')
            elif data_type == 'half':
                beta = _expr.const(np.float16(beta), dtype='float16')
            elif data_type == 'long':
                beta = _expr.const(np.int64(beta), dtype='int64')
            elif data_type == 'int':
                beta = _expr.const(np.int32(beta), dtype='int32')
            elif data_type == 'short':
                beta = _expr.const(np.int16(beta), dtype='int16')
            elif data_type == 'char':
                beta = _expr.const(np.int8(beta), dtype='int8')
            elif data_type == 'byte':
                beta = _expr.const(np.uint8(beta), dtype='uint8')
            weight *= beta

        weight_out = _op.transform.transpose(weight, axes=[1, 0])

        units = _infer_shape(weight_out)[0]
        dense_out = _op.nn.dense(data, weight_out, units=units)

        if use_bias:
            bias = inputs[0]
            return _op.nn.bias_add(dense_out, bias)
        else:
            return dense_out
    return _impl

def _size():
    def _impl(inputs, input_types):
        shape = _infer_shape(inputs[0])
        if len(inputs) > 1:
            axis = int(inputs[1])
            return shape[axis]
        return shape
    return _impl

def _numtotensor():
    def _impl(inputs, input_types):
        val = inputs[0]
        dtype = type(val)

        if isinstance(val, tvm.expr.IntImm):
            val = val.__int__()
            dtype = int

        arr = val * np.ones([]).astype(dtype)
        return arr
    return _impl

def _view():
    def _impl(inputs, input_types):
        data = inputs[0]

        if len(inputs) == 3:
            new_shape = [inputs[1], _infer_shape(inputs[2])[0]]
        else:
            if isinstance(inputs[1], list):
                new_shape = inputs[1]
            else:
                new_shape = _infer_shape(inputs[1])

        return _op.transform.reshape(data, new_shape)
    return _impl

def _clone():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.tensor.copy(data)
    return _impl

def _log_softmax():
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = int(inputs[1])
        return _op.nn.log_softmax(data, axis)
    return _impl

def _sigmoid():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.tensor.sigmoid(data)
    return _impl

def _avg_pool2d():
    def _impl(inputs, input_types):
        data = inputs[0]

        pool_size = _infer_shape(inputs[1])
        strides = _infer_shape(inputs[2])
        padding = _infer_shape(inputs[3])

        ceil_mode = int(inputs[4])
        count_include_pad = int(inputs[5])

        return _op.nn.avg_pool2d(data,
                                 pool_size=pool_size,
                                 strides=strides,
                                 padding=padding,
                                 ceil_mode=ceil_mode,
                                 count_include_pad=count_include_pad)
    return _impl

def _dropout():
    def _impl(inputs, input_types):
        data = inputs[0]
        rate = float(inputs[1])

        return _op.nn.dropout(data, rate)
    return _impl

def _reduce(name):
    def _impl(inputs, attrs):
        data = inputs[0]
        return get_relay_op(name)(data)
    return _impl

def _mean():
    def _impl(inputs, input_types):
        data = inputs[0]
        if inputs[1]:
            axis = _infer_shape(inputs[1])
        else:
            axis = None
        if len(inputs) > 2 and inputs[2]:
            keepdims = int(inputs[2])
        else:
            keepdims = False
        if len(inputs) > 3 and inputs[3]:
            exclude = int(inputs[3])
        else:
            exclude = False

        return _op.mean(data, axis, keepdims, exclude)
    return _impl

def _chunk():
    def _impl(inputs, input_types):
        data = inputs[0]

        num_chunks = int(inputs[1])
        axis = int(inputs[2])

        if isinstance(data, _expr.Var):
            inferred_shape = _infer_shape(data)
        elif isinstance(data, (_expr.Call, _expr.TupleGetItem)):
            inferred_shape = _infer_shape(data)

        shape = []
        for infer in inferred_shape:
            shape.append(infer)

        dim = int(shape[axis])

        if dim % num_chunks:
            unif_size = int(dim / (num_chunks - 1))
        else:
            unif_size = int(dim / num_chunks)

        chunks = []
        for i in range(0, dim, unif_size):
            begin = [0] * len(shape)
            end = shape[:]
            begin[axis] = i
            end[axis] = i + unif_size
            stride = [1] * len(shape)

            chunk_out = _op.transform.strided_slice(data, begin, end, stride)
            chunks.append(chunk_out)


        if dim % num_chunks:
            begin = [0] * len(shape)
            end = shape[:]
            begin[axis] = unif_size * (num_chunks - 1)
            end[axis] = dim
            stride = [1] * len(shape)

            chunk_out = _op.transform.strided_slice(data, begin, end, stride)
            chunks.append(chunk_out)

        return chunks
    return _impl

def _matmul():
    def _impl(inputs, input_types):
        data0 = inputs[0]
        data1 = inputs[1]
        data1_t = _op.transpose(data1, axes=(1, 0))

        return _op.nn.dense(data0, data1_t)
    return _impl

def _expand():
    def _impl(inputs, input_types):
        data_in = inputs[0]
        if isinstance(data_in, _expr.Var):
            shape = _infer_shape(data_in)
        elif isinstance(data_in, (_expr.Call, _expr.TupleGetItem)):
            shape = _infer_shape(data_in)

        ndims = len(shape)
        sizes = _infer_shape(inputs[1])
        out = inputs[0]

        for i in range(ndims):
            if sizes[i] in {-1, shape[i]}:
                continue
            data = list()
            for temp in range(sizes[i]):
                data.append(out)
            call = _op.tensor.concatenate(data, i)

        return call
    return _impl

def _int():
    def _impl(inputs, input_types):
        if isinstance(inputs[0], _expr.Call):
            return inputs[0]
        return int(inputs[0])
    return _impl

def _listunpack():
    def _impl(inputs, input_types):
        return inputs[0]
    return _impl

def _to():
    def _impl(inputs, input_types):
        return inputs[0]
    return _impl

def _device():
    def _impl(inputs, input_types):
        return None
    return _impl

def _pad():
    def _impl(inputs, input_types):
        data = inputs[0]
        padding = inputs[1]
        pad_width = list(zip(padding, padding))
        pad_value = inputs[2]
        return _op.nn.pad(data, pad_width, pad_value)
    return _impl

def _sqrt():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.tensor.sqrt(data)
    return _impl


def _neg():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.tensor.negative(data)
    return _impl


def _tanh():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.tensor.tanh(data)
    return _impl


def _gt():
    def _impl(inputs, input_types):
        assert len(inputs) == 2
        lhs = wrap_const(inputs[0])
        rhs = wrap_const(inputs[1])
        return _op.tensor.greater(lhs, rhs)
    return _impl


def _lt():
    def _impl(inputs, input_types):
        assert len(inputs) == 2
        lhs = wrap_const(inputs[0])
        rhs = wrap_const(inputs[1])
        return _op.tensor.less(lhs, rhs)
    return _impl


def _Bool():
    def _impl(inputs, input_types):
        assert len(inputs) == 1
        return inputs[0]
    return _impl


def _Float():
    def _impl(inputs, input_types):
        assert len(inputs) == 1
        return _op.cast(inputs[0], "float")
    return _impl


def _stack():
    def _impl(inputs, input_types):
        if isinstance(inputs[0], list):
            return _op.tensor.stack(inputs[0], 0)
        else:
            return wrap_const(1)
            assert False
    return _impl


def _mm():
    def _impl(inputs, input_types):
        print("mm input:", inputs)
        return _op.nn.dense(inputs[0], inputs[1])
    return _impl


def _empty_list():
    def _impl(inputs, input_types):
        return p.nil()
    return _impl


def _cons_list():
    def _impl(inputs, input_types):
        tensor2 = p.get_var('tensor2', "float32")
        return p.cons(tensor2(inputs[0]), inputs[1])
    return _impl


def _rev_list():
    def _impl(inputs, input_types):
        return p.rev(inputs[0])
    return _impl


def _tensor_array_stack():
    def _impl(inputs, input_types):
        stack = p.get_var('tensor_array_stack', "float32")
        stacked = stack(inputs[0])
        get_tensor_func = p.get_var("get_tensor2", "float32")
        return get_tensor_func(stacked)
    return _impl


def _upsample(method):
    def _impl(inputs, input_types):
        if isinstance(inputs[1], _expr.Var):
            out_size = _infer_shape(inputs[1])
        elif isinstance(inputs[1], list):
            infer_res = [_infer_value(size, {}) for size in inputs[1]]
            out_size = [np.asscalar(res.asnumpy().astype(np.int)) for res in infer_res]
        data = inputs[0]

        align_corners = inputs[2]
        if align_corners:
            coord_trans = "align_corners"
        else:
            coord_trans = "half_pixel"
        # read that we should actually start to use interpolate(..., mode='bilinear', align_corners=True) instead of upsample
        return _op.image.resize(data, out_size, "NCHW", "bilinear", coord_trans)
    return _impl


# Operator mappings
convert_map = {
    'aten::device'                          : _device(),
    'aten::add'                             : _elemwise('add'),
    'aten::add_'                            : _elemwise('add'),
    'aten::sub'                             : _elemwise('subtract'),
    'aten::sub_'                            : _elemwise('subtract'),
    'aten::max'                             : _elemwise('maximum'),
    'aten::min'                             : _elemwise('minimum'),
    'aten::mul'                             : _elemwise('multiply'),
    'aten::mul_'                            : _elemwise('multiply'),
    'aten::pow'                             : _elemwise('power'),
    'aten::div'                             : _elemwise('divide'),
    'aten::div_'                            : _elemwise('divide'),
    'aten::ones'                            : _ones(),
    'aten::zeros'                           : _zeros(),
    'aten::to'                              : _to(),
    'aten::unsqueeze'                       : _unsqueeze(),
    'aten::cat'                             : _concatenate(),
    'aten::slice'                           : _slice(),
    'aten::select'                          : _select(),
    'aten::relu'                            : _relu(),
    'aten::relu_'                           : _relu(),
    'aten::adaptive_avg_pool2d'             : _adaptive_avg_2d(),
    'aten::adaptive_max_pool2d'             : _adaptive_max_2d(),
    'aten::max_pool2d'                      : _maxpool_2d(),
    'aten::max_pool2d_with_indices'         : _maxpool_2d(),
    'aten::hardtanh'                        : _hardtanh(),
    'aten::hardtanh_'                       : _hardtanh(),
    'aten::_convolution'                    : _convolution(),
    'aten::softmax'                         : _softmax(),
    'aten::threshold'                       : _threshold(),
    'aten::threshold_'                      : _threshold(),
    'aten::contiguous'                      : _contiguous(),
    'aten::batch_norm'                      : _batch_norm(),
    'aten::transpose'                       : _transpose(),
    'aten::transpose_'                      : _transpose(),
    'aten::t'                               : _transpose(),
    'aten::flatten'                         : _flatten(),
    'aten::addmm'                           : _dense(),
    'aten::size'                            : _size(),
    'aten::view'                            : _view(),
    'aten::clone'                           : _clone(),
    'aten::log_softmax'                     : _log_softmax(),
    'aten::sigmoid'                         : _sigmoid(),
    'aten::avg_pool2d'                      : _avg_pool2d(),
    'aten::dropout'                         : _dropout(),
    'aten::dropout_'                        : _dropout(),
    'aten::mean'                            : _mean(),
    'aten::chunk'                           : _chunk(),
    'aten::matmul'                          : _matmul(),
    'aten::expand'                          : _expand(),
    'aten::Int'                             : _int(),
    'prim::NumToTensor'                     : _numtotensor(),
    'aten::constant_pad_nd'                 : _pad(),
    'aten::permute'                         : _transpose(),
    'aten::sum'                             : _reduce('sum'),
    'aten::prod'                            : _reduce('prod'),
    'aten::sqrt'                            : _sqrt(),
    'aten::lt'                              : _lt(),
    'aten::gt'                              : _gt(),
    'aten::Bool'                            : _Bool(),
    'aten::Float'                           : _Float(),
    'aten::neg'                             : _neg(),
    'aten::tanh'                            : _tanh(),
    'aten::stack'                           : _stack(),
    'aten::mm'                              : _matmul(),
    'relay::empty_list'                     : _empty_list(),
    'relay::cons_list'                      : _cons_list(),
    'relay::rev_list'                       : _rev_list(),
    'relay::tensor_array_stack'             : _tensor_array_stack(),
    'aten::upsample_bilinear2d'             : _upsample("bilinear"),
}
