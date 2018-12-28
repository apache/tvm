from tvm.relay import var, const, create_executor
from tvm.relay.op import debug


_test_debug_hit = False

def test_debug():
    global _test_debug_hit
    ex = create_executor()
    x = var('x', shape=(), dtype='int32')
    _test_debug_hit = False
    def did_exec(x):
        global _test_debug_hit
        _test_debug_hit = True
    prog = debug(x, debug_func=did_exec)
    result = ex.evaluate(prog, { x: const(1, 'int32') })
    assert _test_debug_hit
    assert result.asnumpy() == 1


def test_debug_with_expr():
    global _test_debug_hit
    _test_debug_hit = False
    ex = create_executor()
    x = var('x', shape=(), dtype='int32')
    _test_debug_hit = False
    def did_exec(x):
        global _test_debug_hit
        _test_debug_hit = True
    prog = debug(x + x * x, debug_func=did_exec)
    result = ex.evaluate(prog, { x: const(2, 'int32') })
    assert _test_debug_hit
    assert result.asnumpy() == 6
