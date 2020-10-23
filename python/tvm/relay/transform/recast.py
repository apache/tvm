"""Relay Downcast from Full-precision to Half-precision floating-point Pass"""
import tvm
from tvm import relay
from tvm.relay import ExprVisitor, ExprMutator, Call, Var, Constant, TupleGetItem, Function
import tvm.relay.transform as _transform
from tvm.relay.frontend.common import infer_type
from tvm.relay.analysis import count_layers


def recast(func, dtype, out_dtype, ops=['nn.conv2d'], skip_layers=[]):
    # pylint: disable=line-too-long
    """Downcast mutator
    Parameters
    ---------
    function: Function
        The original function that will have its type changed.
    dtype: str
        The target type to cast to.
    out_dtype: str
        The output type to cast to.
    ops: List[str]
        A list of operations that should have their type changed,
        others will be left as is.
    skip_layers: List[int]
        A list of integers indicating operations that should
        not have their type changed, counted starting with the
        first valid operation encountered. Negative indices are
        allowed and indicate starting at the last layer.
    Returns
    -------
    The graph after downcasting to the specified datatype.
    """
    # Collect valid relay ops that should be recast.
    valid_ops = [relay.op.get(op) for op in ops]

    class RecastMutator(ExprMutator):
        """Cast operations to the target type."""
        def __init__(self, valid_op_count):
            self.layer_count = 0
            self.valid_op_count = valid_op_count
            self.skip_layers = skip_layers
            # Convert negative indices to positive ones.
            for i, layer in enumerate(skip_layers):
                if layer < 0:
                    skip_layers[i] = self.valid_op_count + layer
            super().__init__()

        def set_attr_dtype(self, attrs):
            new_attr_dict= {}
            for attr in attrs.keys():
                attr_value = attrs[attr]
                if isinstance(attr_value, tvm.ir.container.Array):
                    attr_value = tuple(attr_value)
                new_attr_dict[str(attr)] = attr_value
            new_attr_dict['out_dtype'] = out_dtype
            attr_type = str(attrs).split('(')[0]
            return tvm.ir.make_node(attr_type, **new_attr_dict)

        def visit_call(self, call):
            if call.op in valid_ops:
                layer_count = self.valid_op_count - self.layer_count - 1
                self.layer_count += 1
                print(layer_count)
                print(call)
                print("\n\n\n")
                if layer_count in skip_layers:
                    return super().visit_call(call)

                # Otherwise recast its inputs.
                new_fn = self.visit(call.op)
                args = [self.visit(arg) for arg in call.args]
                self.layer_count = 0
                new_args = list()
                for arg in args:
                    new_args.append(relay.cast(arg, dtype=dtype))
                new_attrs = self.set_attr_dtype(call.attrs)
                # Recast the output for compatibility with other graph operations.
                return relay.cast(Call(new_fn, new_args, new_attrs), infer_type(args[0]).checked_type.dtype)
                    
            return super().visit_call(call)

    layer_depth = count_layers.count_layers(func, ['nn.conv2d', 'nn.dense'])
    print(layer_depth)
    exit()
    recast_pass = RecastMutator(count_pass.valid_op_count)
    func = recast_pass.visit(func)
    return tvm.IRModule.from_expr(func)