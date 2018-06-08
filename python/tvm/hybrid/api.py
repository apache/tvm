"""APIs of lowering the Python subset to HalideIR"""
import types
import decorator
from .parser import parse_python
from ._internal import _enter_hybrid_runtime, _restore_runtime, _is_tvm_arg_types, _pruned_source

@decorator.decorator
def script(func, *args):
    """If the arguments are tvm types, compile it to HalideIR.
    O.W. return the python emulated result"""
    if _is_tvm_arg_types(args):
        return parse(func, args)
    else:
        intersect = _enter_hybrid_runtime(func)
        func(*args)
        _restore_runtime(func, intersect)
    return func

def parse(func, args):
    """Parse a subset of Python to HalideIR

    Parameters
    ----------
    func : str or types.FunctionType
        If it is a string, parse the source code
        If it is a function, parse the function

    args : list of Buffer or Tensor or Var
        The argument lists to the function.
        Leave it None if no buffer is related to the function to be parsed

    Returns
    -------
    (halide_ir, parser) : (Stmt, PyAST2HalideIR)
        The result Halide IR and the parser class instance.
        TODO: Later we deprecate this return value, use a dedicated OP node type instead
    """
    if isinstance(func, str):
        src = func
    else:
        assert isinstance(func, types.FunctionType)
        src = _pruned_source(func)
    return parse_python(src, args)
