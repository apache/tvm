from tvm.relay import var, const, create_executor
from tvm.relay.op import debug


def test_debug():
    exec = create_executor()
    x = var('x', shape=(), dtype='int32')
    hit = False
    def did_exec(x):
        global hit
        hit = True
    prog = debug(x, debug_func=did_exec)
    exec.evaluate(prog, { x: const(1) })
    assert hit
