from tvm.relay import var, const, create_executor
from tvm.relay.op import debug


_test_debug_hit = False

def test_debug():
    global _test_debug_hit
    exec = create_executor()
    x = var('x', shape=(), dtype='int32')
    _test_debug_hit = False
    def did_exec(x):
        global _test_debug_hit
        _test_debug_hit = True
    prog = debug(x, debug_func=did_exec)
    exec.evaluate(prog, { x: const(1) })
    assert _test_debug_hit
