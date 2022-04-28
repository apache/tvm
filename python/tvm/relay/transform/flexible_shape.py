from tvm import relay


def override_shape(ty, dim, value):
    """Change a value in a tensor shape."""
    new_dims = list(ty.shape)
    new_dims[dim] = value
    return relay.TensorType(new_dims, ty.dtype)


def specialize_body(mod, fn, dim, value):
    """Create a subgraph to handle a specific input shape"""
    data = fn.params[0]
    flex_ty = override_shape(data.type_annotation, dim, value)
    dyn_data = relay.Var(data.name_hint, type_annotation=flex_ty)
    new_params = [dyn_data] + fn.params[1:]
    new_body = relay.expr.bind(fn.body, {data: dyn_data})
    new_ret_ty = override_shape(fn.ret_type, dim, value)
    gvar = relay.GlobalVar("main_" + str(value))
    mod[gvar] = relay.Function(new_params, new_body, new_ret_ty, fn.type_params, fn.attrs)
    return gvar, dyn_data.type_annotation


def flexible_dispatch(mod, dim=0, buckets=[]):
    """
    Implement a batching transform for Relay.

    Constructs a tree which splits on the batch dimension dispatching
    to specialized versions of Relay code which execute for each batch
    dimension.

    This enables us to ship a single Relay model which supports performance
    specialization for a batch dimension without needing Relax and defaults
    to a slow but reliable performance path.

    This also allows for us to tune all batch sizes in a single program
    increasing effectiveness of TRS or other caching by filling/hitting.
    """
    main_fn = mod["main"]
    data = main_fn.params[0]
    dyn_shape = override_shape(data.type_annotation, dim, relay.Any())
    dyn_data = relay.Var(data.name_hint, type_annotation=dyn_shape)
    rt_sh = relay.op.shape_of(dyn_data)
    flex_value = relay.op.take(rt_sh, relay.const(dim))
    if_exprs = []

    for bucket in buckets:
        spec_call, spec_ty = specialize_body(mod, main_fn, dim, bucket)
        spec_data = relay.op.reshape(dyn_data, spec_ty.shape)
        call_args = [spec_data] + main_fn.params[1:]
        if_exprs.append((relay.op.equal(flex_value, relay.const(bucket)), spec_call(*call_args)))

    default_dyn_call, _ = specialize_body(mod, main_fn, dim, relay.Any())
    call_args = [dyn_data] + main_fn.params[1:]

    new_body = default_dyn_call(*call_args)

    for cond, true_branch in if_exprs:
        new_body = relay.If(cond, true_branch, new_body)

    new_params = [dyn_data] + main_fn.params[1:]
    dyn_ret_type = override_shape(main_fn.ret_type, dim, relay.Any())
    new_main = relay.Function(
        new_params, new_body, dyn_ret_type, main_fn.type_params, main_fn.attrs
    )
    mod["main"] = new_main
    mod = relay.transform.InferType()(mod)
    return mod


class FlexibleShapeDispatch(object):
    """Enable inference of multiple shaped inputs in one module.

    This transformation adds a handler around a module that
    checks input shapes and dispatches to a subgraph specialized
    to handle the specific shapes of that input. If no exactly matching
    subgraph is available, the input will be run using full dynamism.
    For best performance, specify all the sizes the module will
    be likely to see using the buckets argument.

    Parameters
    ----------
    dim: int
        The dimension of the input that should be made flexible. This will
        most often be used for the batch dimension.
    buckets: list[int]
        The sizes of the input dimension that should be explicitly handled.
        Each value in buckets will have a corresponding subgraph constructed to
        handle it.

    Returns
    -------
    ret : FlexibleShapeDispatch
        A pass that can be applied to a module to add flexible shape handling.
    """

    def __init__(self, dim=0, buckets=[]):
        self.dim = dim
        self.buckets = buckets
        super(FlexibleShapeDispatch, self).__init__()

    def __call__(self, mod):
        return flexible_dispatch(mod, self.dim, self.buckets)
