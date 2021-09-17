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


def _maxpool_1d_relay_func():
    from tvm import relay

    input = relay.var("input", relay.TensorType((1, 28, 28), 'int8'))
    requantize = relay.qnn.op.requantize(input,
        relay.const(0.00392157), relay.const(0), relay.const(0.00392157), relay.const(-128))
    max_pool1d = relay.nn.max_pool1d(requantize, pool_size=3, layout="NWC")
    requantize = relay.qnn.op.requantize(max_pool1d,
        relay.const(0.00392157), relay.const(-128), relay.const(0.00392157), relay.const(0),
        out_dtype='uint8')

    return relay.Function([input], requantize)


def read_relay_mod_from_txt(file_name):
    import tvm

    with open(file_name, "r") as f:
        data = f.read()

    relay_mod = tvm.parser.fromtext(data)

    return relay_mod, _generate_params(relay_mod)


def _avgpool_1d_layer(shape):
    from tvm import relay

    input = relay.var("input", relay.TensorType(shape, 'int8'))
    requantize = relay.qnn.op.requantize(input, relay.const(0.00392157), relay.const(0), relay.const(0.00392157), relay.const(-128))
    cast = relay.cast(requantize, 'int16')
    avg_pool1d = relay.nn.avg_pool1d(cast, pool_size=3, layout="NCW")
    avg_pool1d_2 = relay.nn.avg_pool1d(avg_pool1d, pool_size=3, layout="NCW")
    avg_pool1d_3 = relay.nn.avg_pool1d(avg_pool1d_2, pool_size=6, layout="NCW")
    avg_pool1d_4 = relay.nn.avg_pool1d(avg_pool1d_3, pool_size=7, layout="NCW")
    avg_pool1d_5 = relay.nn.avg_pool1d(avg_pool1d_4, pool_size=8, layout="NCW")
    cast2 = relay.cast(avg_pool1d_5, 'int32')
    requantize = relay.qnn.op.requantize(cast2, relay.const(0.133801), relay.const(37), relay.const(0.133801), relay.const(165), out_dtype='uint8')

    return relay.Function([input], requantize)


def maxpool_1d_relay_mod():
    import tvm


    relay_mod = tvm.IRModule({tvm.relay.GlobalVar('main'): _maxpool_1d_relay_func()})
    input = ('input', (1, 28, 28), 'int8')
    output = ('output', (1, 4), 'uint8')
    return (relay_mod, _generate_params(relay_mod), input, output)


def avgpool_1d_relay_mod():
    import tvm

    shape = (1, 128, 128)
    relay_mod = tvm.IRModule.from_expr(_avgpool_1d_layer(shape))
    input = ('input', shape, 'int8')
    output = ('output', (1, 128, 106), 'uint8')
    return (relay_mod, _generate_params(relay_mod), input, output)


def get_data():
    import numpy as np

    dataset = []
    for i in range(10):
        data = np.random.randint(low=-128, high=127, size=(1, 128, 128), dtype='int8')
        dataset.append((str(i), data))

    return dataset
