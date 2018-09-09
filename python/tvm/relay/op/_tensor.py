#pylint: disable=invalid-name
"""Backend compiler related feature regsitration"""
from topi import add
from .op import register
from ..type import FuncType, TensorType
from ...schedule import create_schedule
from ...api import placeholder

def type_to_placeholder(name, ty):
    """Convert a single type into the correct placeholder."""
    if isinstance(ty, TensorType):
        return placeholder(ty.shape, name=name, dtype=ty.dtype)
    else:
        raise Exception("can only pass Tensor values to TVM operators")

def func_ty_to_placeholders(func_ty):
    """Build input placeholders based on a function type."""
    if isinstance(func_ty, FuncType):
        arg_types = func_ty.arg_types
        ret_type = func_ty.ret_type
        args = []
        var = 0
        for arg in arg_types:
            var += 1
            args.append(type_to_placeholder(f"Input{var}", arg))
        return args, ret_type
    else:
        raise Exception("error")

# def lookup_in_topi(name):
#     try:
#         f = eval(f"topi.{name}")
#     except:
#         f = eval(f"topi.nn.{name}")

#     return f

# @tvm.register_func("nnvm.relay._default_op_compiler")
# def _default_op_compile(op_name: str, func_ty: ir.Type, attrs: ir.Attributes=None) -> Any:
#     Inputs, ret_ty = func_ty_to_placeholders(func_ty)
#     op = lookup_in_topi(op_name)
#     Output = op(*Inputs)

#     if Output.dtype == 'uint1':
#         import pdb; pdb.set_trace()
#         Output = Output.astype('uint8')

#     schedule = tvm.create_schedule(Output.op)
#     return [schedule, Inputs + [Output]]

#pylint: disable=duplicate-argument-name
def add_compiler(_, func_type, *__):
    """The compilation code for the TVM compiler."""
    inputs, _ = func_ty_to_placeholders(func_type)
    # op = lookup_in_topi(op_name)
    output = add(*inputs)
    schedule = create_schedule(output.op)
    return [schedule, inputs + [output]]

register("add", "FRelayOpCompiler", add_compiler)
