import numpy as np
import tvm
from tvm import relay

# convert(convert(a, tg), tg) = convert(a, tg)
def convert(a, ctx):
    while True:
        if isinstance(a, int):
            a = np.array(a, dtype='int32')
        elif isinstance(a, np.ndarray):
            a = tvm.nd.array(a, ctx)
        elif isinstance(a, tvm.runtime.NDArray):
            return a
        elif isinstance(a, relay.Call):
            assert isinstance(a.op, relay.Constructor)
            a = (a.op, *a.args)
        elif isinstance(a, tuple):
            assert isinstance(a[0], relay.Constructor)
            a = relay.backend.interpreter.ConstructorValue(a[0].tag, [convert(arg, ctx) for arg in a[1:]], a[0])
        elif isinstance(a, relay.backend.interpreter.ConstructorValue):
            return a
        else:
            raise Exception(a, type(a))
