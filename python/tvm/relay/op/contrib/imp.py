"""IMP library supported operators.
There are two ways to registering a function for an op to indicate if it is
supported by IMP        .

- The first and simplest way is to use the helper so that
users only need to provide the operator name and a boolean value to indicate if
it is supported. For example:

    .. code-block:: python

      add = _register_external_op_helper("add")
      add = _register_external_op_helper("add", True)
      add = _register_external_op_helper("add", False)

- The other way is to implement the function by themselves to
check the attributes of the op and decide if it should be offloaded to DNNL.
"""
import tvm.ir


# def _register_external_op_helper(op_name, supported=True):
#     """The helper function to indicate that a given operator can be supported
#     by IMP.

#     Paramters
#     ---------
#     op_name : Str
#         The name of operator that will be registered.

#     Returns
#     -------
#     f : callable
#         A function that returns if the operator is supported by DNNL.
#     """
#     @tvm.ir.register_op_attr(op_name, "target.imp")
#     def _func_wrapper(attrs, args):
#         return supported

#     return _func_wrapper


# # _register_external_op_helper("nn.batch_norm")
# # _register_external_op_helper("nn.conv2d")
# # _register_external_op_helper("nn.dense")
# # _register_external_op_helper("nn.relu")
# _register_external_op_helper("add")
# # _register_external_op_helper("subtract")
# _register_external_op_helper("multiply")


@tvm.ir.register_op_attr("add", "target.imp")
def _imp_add_wrapper(attrs, args):
  return True

@tvm.ir.register_op_attr("multiply", "target.imp")
def _imp_multiply_wrapper(attrs, args):
  return True
