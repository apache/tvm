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


def flexible_dispatch(mod, dim=0, buckets=[], auto_pad=False, pad_value=0):
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

    for i, bucket in enumerate(buckets):
        input_data = dyn_data
        check_dim = flex_value

        # Apply automatic padding if specified.
        if auto_pad:
            # Construct padding expression for inputs.
            pad_width = relay.const(bucket) - flex_value
            rank = len(data.type_annotation.shape)
            pads = relay.zeros([rank, 2], 'int32')
            pads = relay.scatter_nd(pads, relay.const([dim, 1]), pad_width)
            padded_value = relay.nn.pad(dyn_data, pads, pad_value)

            # Determine if this is the proper bucket to pad to. Do this by checking if the
            # input shape is between this bucket and the previous.
            if i == 0:
                padded_value = relay.If(relay.op.less_equal(flex_value, relay.const(bucket)), padded_value, dyn_data)
            else:
                padded_value = relay.If(relay.op.logical_and(relay.op.less_equal(flex_value, relay.const(bucket)), relay.op.greater(flex_value, relay.const(buckets[i - 1]))), padded_value, dyn_data)
            # Update input value and test dimension to reflect possible padding.
            input_data = padded_value
            check_dim = relay.op.take(relay.op.shape_of(input_data), relay.const(dim))

        # Create a specialized subgraph for the current bucket.
        spec_call, spec_ty = specialize_body(mod, main_fn, dim, bucket)
        spec_data = relay.op.reshape(input_data, spec_ty.shape)

        # Create a dispatch statement for the current specialized graph.
        call_args = [spec_data] + main_fn.params[1:]
        if_exprs.append((relay.op.equal(check_dim, relay.const(bucket)), spec_call(*call_args)))

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
    auto_pad: Optional[bool]
        If True, then padding will be inserted to values that don't match one of
        the provided buckets.
    pad_value: Optional[float]
        When auto_pad is true, padding will be done with this value.

    Returns
    -------
    ret : FlexibleShapeDispatch
        A pass that can be applied to a module to add flexible shape handling.
    """

    def __init__(self, dim=0, buckets=[], auto_pad=False, pad_value=0):
        self.dim = dim
        self.buckets = buckets
        self.auto_pad = auto_pad
        self.pad_value = pad_value
        super(FlexibleShapeDispatch, self).__init__()

    def __call__(self, mod):
        return flexible_dispatch(mod, self.dim, self.buckets, self.auto_pad, self.pad_value)
