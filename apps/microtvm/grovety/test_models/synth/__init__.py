def _generate_params(relay_mod):
    import numpy as np

    id = 1
    params = {}
    for _, function in relay_mod.functions.items():
        for param in function.params[1:]:
            name = f'_param_{id}'
            dtype = param.type_annotation.dtype
            shape = param.type_annotation.shape
            low = -3 if len(shape) == 1 else -30
            high = 3 if len(shape) == 1 else 30
            generate_func = np.random.randint if 'int' in dtype else np.random.uniform
            params[name] = generate_func(low, high, size=[int(x) for x in shape], dtype=dtype)
            id += 1
    return params


def read_relay_mod_from_txt(file_name):
    import tvm

    with open(file_name, "r") as f:
        data = f.read()

    relay_mod = tvm.parser.fromtext(data)

    return relay_mod, _generate_params(relay_mod)


def maxpool_1d_relay_mod():
    import tvm
    from tvm import relay

    shape = (1, 64, 64)
    input = relay.var("input", relay.TensorType(shape, 'int8'))
    requantize = relay.qnn.op.requantize(input, relay.const(0.00392157), relay.const(0), relay.const(0.00392157), relay.const(-128))
    max_pool1d = relay.nn.max_pool1d(requantize, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    max_pool1d = relay.nn.max_pool1d(max_pool1d, pool_size=3, layout="NWC")
    requantize = relay.qnn.op.requantize(max_pool1d, relay.const(0.00392157), relay.const(-128), relay.const(0.00392157), relay.const(0), out_dtype='uint8')

    relay_mod = tvm.IRModule.from_expr(relay.Function([input], requantize))
    input = ('input', shape, 'int8')
    output = ('output', (1, 4, 64), 'uint8')
    return (relay_mod, _generate_params(relay_mod), input, output)


def avgpool_1d_relay_mod():
    import tvm
    from tvm import relay

    shape = (1, 64, 64)
    input = relay.var("input", relay.TensorType(shape, 'int8'))
    requantize = relay.qnn.op.requantize(input, relay.const(0.00392157), relay.const(0), relay.const(0.00392157), relay.const(-128))
    cast = relay.cast(requantize, 'int16')
    avg_pool1d = relay.nn.avg_pool1d(cast, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    cast2 = relay.cast(avg_pool1d, 'int32')
    requantize = relay.qnn.op.requantize(cast2, relay.const(0.133801), relay.const(37), relay.const(0.133801), relay.const(165), out_dtype='uint8')

    relay_mod = tvm.IRModule.from_expr(relay.Function([input], requantize))
    input = ('input', shape, 'int8')
    output = ('output', (1, 64, 4), 'uint8')
    return (relay_mod, _generate_params(relay_mod), input, output)


def avgpool_2d_relay_mod():
    import tvm
    from tvm import relay

    shape = (1, 1, 64, 64)
    input = relay.var("input", relay.TensorType(shape, 'int8'))
    requantize = relay.qnn.op.requantize(input, relay.const(0.00392157), relay.const(0), relay.const(0.00392157), relay.const(-128))
    cast = relay.cast(requantize, 'int16')
    avg_pool2d = relay.nn.avg_pool2d(cast, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    avg_pool2d = relay.nn.avg_pool2d(avg_pool2d, pool_size=(3, 3), layout="NCHW")
    cast2 = relay.cast(avg_pool2d, 'int32')
    requantize = relay.qnn.op.requantize(cast2, relay.const(0.00392157), relay.const(-128), relay.const(0.00392157), relay.const(0), out_dtype='uint8')

    relay_mod = tvm.IRModule.from_expr(relay.Function([input], requantize))
    input = ('input', shape, 'int8')
    output = ('output', (1, 1, 4, 4), 'uint8')
    return (relay_mod, _generate_params(relay_mod), input, output)


def conv_1d_relay_mod():
    import tvm
    from tvm import relay
    import numpy as np

    in_channels = 1
    input = relay.var("input", relay.TensorType((1, 28, 28, in_channels), "int8"))
    v1 = relay.reshape(input, (1, -1, in_channels))
    v2 = relay.const(np.ones((3, in_channels, 12), dtype="int8"))
    conv1 = relay.op.nn.conv1d(v1, v2, kernel_size=3, data_layout="NWC", kernel_layout="WIO", out_dtype="int32", channels=12)
    v3 = relay.cast(conv1, dtype="int16")
    v4 = relay.const(np.ones((3, 12, 24), dtype="int16"))
    conv2 = relay.op.nn.conv1d(v3, v4, kernel_size=3, data_layout="NWC", kernel_layout="WIO", out_dtype="int32", channels=24)
    relay_mod = tvm.IRModule.from_expr(relay.Function([input], conv2))

    input = ('input', (1, 28, 28, in_channels), 'int8')
    output = ('output', (1, 780, 24), 'int16')
    return (relay_mod, _generate_params(relay_mod), input, output)


def conv_1d_relay_mod_2():
    import tvm
    from tvm import relay
    import numpy as np

    in_channels = 1
    input = relay.var("input", relay.TensorType((1, 28, 28, in_channels), "int8"))
    v1 = relay.reshape(input, (1, -1, in_channels))
    v2 = relay.const(np.ones((3, 12, in_channels), dtype="int8"))
    conv1 = relay.op.nn.conv1d(v1, v2, kernel_size=3, data_layout="NWC", kernel_layout="WOI", out_dtype="int32", channels=12)
    v3 = relay.cast(conv1, dtype="int16")
    v4 = relay.const(np.ones((3, 24, 12), dtype="int16"))
    conv2 = relay.op.nn.conv1d(v3, v4, kernel_size=3, data_layout="NWC", kernel_layout="WOI", out_dtype="int32", channels=24)
    v5 = relay.cast(conv2, dtype="int16")
    relay_mod = tvm.IRModule.from_expr(relay.Function([input], v5))

    input = ('input', (1, 28, 28, in_channels), 'int8')
    output = ('output', (1, 780, 24), 'int16')
    return (relay_mod, _generate_params(relay_mod), input, output)


def complex_test_relay_mod():
    import numpy as np
    import tvm
    import tvm.relay as relay

    in_C = 1
    in_N = 1
    conv2d_1_C = 12
    conv2d_2_C = 12
    conv1d_1_C = 12
    conv1d_2_C = 24

    def rand_data(shape, range, dtype):
        return np.random.randint(*range, size=shape, dtype=dtype)

    input = relay.var("input", relay.TensorType((in_N, 28, 28, in_C), "int8"))

    # conv2d int16
    conv2d_1_f = relay.cast(input, dtype="int16")
    conv2d_1_w = relay.const(rand_data((3, 3, conv2d_1_C, in_C), dtype="int16", range=(-10, 10)))
    conv2d_1 = relay.op.nn.conv2d(conv2d_1_f, conv2d_1_w, kernel_size=(3, 3), data_layout="NHWC", kernel_layout="HWOI",
                               out_dtype="int32", channels=conv2d_1_C, out_layout="NCHW")

    # avgpool_2d
    ap2d_f = relay.cast(conv2d_1, dtype="int16")
    ap2d = relay.op.nn.avg_pool2d(ap2d_f, (3, 3), layout="NCHW", strides=(2, 2))
    ap2d = relay.op.transpose(ap2d, (0, 2, 3, 1))

    # conv2d int8
    conv2d_2_f = relay.cast(ap2d, dtype="int8")
    conv2d_2_w = relay.const(rand_data((3, 3, conv2d_2_C, conv2d_1_C), dtype="int8", range=(-10, 10)))
    conv2d_2 = relay.op.nn.conv2d(conv2d_2_f, conv2d_2_w, kernel_size=(3, 3), data_layout="NHWC", kernel_layout="HWOI",
                               out_dtype="int32", channels=conv2d_2_C)

    # maxpool_2d
    mp2d_f = relay.cast(conv2d_2, dtype="int8")
    mp2d = relay.op.nn.max_pool2d(mp2d_f, (3, 3), layout="NHWC")

    # conv1d int8
    conv1d_1_f = relay.cast(relay.reshape(mp2d, (in_N, -1, conv2d_2_C)), dtype="int8")
    conv1d_1_w = relay.const(rand_data((3, conv1d_1_C, conv2d_2_C), dtype="int8", range=(-10, 10)))
    conv1d_1 = relay.op.nn.conv1d(conv1d_1_f, conv1d_1_w, kernel_size=3, data_layout="NWC", kernel_layout="WOI",
                               out_dtype="int32",
                               channels=conv1d_1_C)

    # maxpool_1d
    mp1d_f = relay.cast(conv1d_1, dtype="int8")
    mp1d = relay.op.nn.max_pool1d(mp1d_f, (3,), layout="NWC", strides=2)

    # conv1d int16
    conv1d_2_f = relay.cast(mp1d, dtype="int16")
    conv1d_2_w = relay.const(rand_data((3, conv1d_2_C, conv1d_1_C), dtype="int16", range=(-10, 10)))
    conv1d_2 = relay.op.nn.conv1d(conv1d_2_f, conv1d_2_w, kernel_size=3, data_layout="NWC", kernel_layout="WOI",
                               out_dtype="int32",
                               channels=conv1d_2_C, out_layout="NCW")

    # avgpool_1d
    ap1d_f = relay.cast(conv1d_2, dtype="int16")
    ap1d = relay.op.nn.avg_pool1d(ap1d_f, (3,), layout="NCW", strides=2)

    # dense
    dense_f = relay.op.nn.batch_flatten(ap1d)
    dense_w = relay.const(rand_data((10, 13 * conv1d_2_C), dtype="int16", range=(-10, 10)))
    dense = relay.op.nn.dense(dense_f, dense_w, units=10, out_dtype="int32")

    relay_mod = tvm.IRModule.from_expr(relay.Function([input], dense))

    input = ('input', (1, 28, 28, 1), 'int8')
    output = ('output', (1, 10), 'int32')
    return (relay_mod, _generate_params(relay_mod), input, output)


def get_data(input_shape, input_dtype='int8'):
    import numpy as np

    dataset = []
    for i in range(10):
        data = np.random.randint(low=-128, high=127, size=input_shape, dtype=input_dtype)
        dataset.append((str(i), data))

    return dataset


def read_relay_mod_from_txt(file_name):
    import tvm

    with open(file_name, "r") as f:
        data = f.read()

    relay_mod = tvm.parser.fromtext(data)

    return relay_mod, _generate_params(relay_mod)

