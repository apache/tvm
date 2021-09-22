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

