# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Relay functions for wrapping a module with flexible shape dispatch."""
import tvm
from tvm import relay


def override_shape(tensor_type, axis, dim):
    """Change a dimension in a tensor shape."""
    # Handle multiple tensors by overriding the shape of each.
    if isinstance(tensor_type, relay.TupleType):
        tensor_type = tensor_type.fields
    else:
        tensor_type = [tensor_type]

    # Create new tensortypes for each input.
    new_types = []
    for t_type in tensor_type:
        new_dims = list(t_type.shape)
        new_dims[axis] = dim
        new_types.append(relay.TensorType(new_dims, t_type.dtype))

    # Dont return a tuple if there is a single tensor.
    if len(new_types) == 1:
        return new_types[0]
    return relay.TupleType(tvm.runtime.convert(new_types))


def specialize_body(mod, function, axis, dim, input_indices, affects_output=True):
    """
    Create a subgraph to handle specific input shapes

    This function takes in a module and one of it's functions and creates a
    similar function with a specific input shape. It then attaches the new function
    to the module. Calling this function multiple times results in a module that
    contains several similar functions each specialized to a specific input shape.
    This allows a dispatch handler to be built on top of the module to deal with
    flexible shapes.

    There are a few modes to this function. When the specialized function has multiple
    flexible inputs, the index of those inputs must be provided to the input_indices argument.
    In this case, the axis of the flexible dimension for each of those inputs must be the same.

    By default, this function assumes that the output shape is dependent on the input
    shape (as is the case in dynamic batching) and will also specialize the output type
    accordingly. If this is not true, the affects_output argument must be set to False.

    Parameters
    ----------
    mod: IRModule
        The module that contains specialized functions and the dispatcher.
    function: Function
        The original non-specialized function that will be transformed.
    axis: int
        Which axis the flexible shape is on.
    dim: int
        The shape to specialize the new subgraph for along the axis dim.
    input_indices: List[int]
        Which inputs should be dispatched dynamically, provided by index. All inputs
        must share the same dynamic axis.
    affects_output: Optional[bool]
        Whether the change in input shape has a corresponding effect on the output shape.
        Batching for example effects both the input and output whereas changing sequence
        length in an NLP model typically does not.

    Returns
    -------
    gvar : GlobalVar
        The new variable for the specialized subgraph.
    spec_types : List[TensorType]
        A list of the new specialized types for each input in the graph.
    """
    # Iterate through specified inputs and construct specialized shapes for each.
    new_params = list(function.params)
    data_binding = {}
    dyn_data_array = []
    for inp in input_indices:
        data = function.params[inp]
        flex_ty = override_shape(data.type_annotation, axis, dim)
        dyn_data = relay.Var(data.name_hint, type_annotation=flex_ty)
        new_params[inp] = dyn_data
        data_binding[data] = dyn_data
        dyn_data_array.append(dyn_data)

    # Create a new function body for the modified shapes.
    new_body = relay.expr.bind(function.body, data_binding)
    # Only change the output shape if the input shape affects it.
    if affects_output:
        new_ret_ty = override_shape(function.ret_type, axis, dim)
    else:
        new_ret_ty = function.ret_type
    gvar = relay.GlobalVar("main_" + str(dim))
    # Add the new function to the main IRModule.
    mod[gvar] = relay.Function(
        new_params, new_body, new_ret_ty, function.type_params, function.attrs
    )
    return gvar, [d.type_annotation for d in dyn_data_array]


def flexible_dispatch(
    mod, buckets, axis=0, auto_pad=False, pad_value=0, input_indices=None, affects_output=True
):
    """
    Enable inference of multiple shaped inputs in one module.

    This transformation adds a handler around a module that
    checks input shapes and dispatches to a subgraph specialized
    to handle the specific shapes of that input. If no exactly matching
    subgraph is available, the input will be run using full dynamism.
    For best performance, specify all the sizes the module will
    be likely to see using the buckets argument.

    By default, this function will dispatch shapes that exactly match one
    of the buckets to a corresponding subgraph. All non-matching shapes
    use the same fully dynamic fallback. This can be detrimental to performance
    for those non-matching shapes. Setting auto_pad to True causes this
    function to round-up the shape of non-matching inputs to the closest
    bucket. This allows them to use the tuned kernels of bucket shapes
    which can improve performance.

    Functions that have multiple inputs sharing a dynamic axis, which
    is common for batch size or sequence length dynamism, are supported
    through the input_indices argument.

    Many types of dynamism such as batching affect both the input and output
    shape, however this is not always the case. If the output shape
    is independent of the input, the affects_output argument of this
    function must be set to False.

    Parameters
    ----------
    buckets: list[int]
        The sizes of the input dimension that should be explicitly handled.
        Each value in buckets will have a corresponding subgraph constructed to
        handle it.
    axis: int
        The dimension of the input that should be made flexible. This will
        most often be used for the batch dimension.
    auto_pad: Optional[bool]
        If True, then padding will be inserted to values that don't match one of
        the provided buckets.
    pad_value: Optional[float]
        When auto_pad is true, padding will be done with this value.
    input_indices: Optional[List[int]]
        Which inputs should be dispatched dynamically, provided by index. All inputs
        must share the same dynamic axis.
    affects_output: Optional[bool]
        Whether the change in input shape has a corresponding effect on the output shape.
        Batching for example effects both the input and output whereas changing sequence
        length in an NLP model typically does not.

    Returns
    -------
    mod : IRModule
        The new module wrapped with a flexible shape dispatch handler.
    """
    main_fn = mod["main"]

    # Default to single input if not specified.
    if input_indices is None:
        input_indices = [0]

    # Extract all input data and create a new dynamic variable for each.
    data = []
    dyn_data = []
    for i in input_indices:
        data.append(main_fn.params[i])
        dyn_shape = override_shape(data[i].type_annotation, axis, relay.Any())
        dyn_data.append(relay.Var(data[i].name_hint, type_annotation=dyn_shape))

    # Extract the dynamic shape value from one of the inputs.
    rt_sh = relay.op.shape_of(dyn_data[0])
    flex_value = relay.op.take(rt_sh, relay.const(axis))

    if_exprs = []

    for i, bucket in enumerate(buckets):
        input_data = dyn_data
        check_dim = flex_value

        # Apply automatic padding if specified.
        if auto_pad:
            input_data = []
            # Construct padding expression for inputs.
            for j, inp in enumerate(dyn_data):
                pad_width = relay.const(bucket) - flex_value
                rank = len(data[j].type_annotation.shape)
                pads = relay.zeros([rank, 2], "int32")
                pads = relay.scatter_nd(pads, relay.const([axis, 1]), pad_width)
                padded_value = relay.nn.pad(inp, pads, pad_value)

                # Determine if this is the proper bucket to pad to. Do this by checking if the
                # input shape is between this bucket and the previous.
                if i == 0:
                    padded_value = relay.If(
                        relay.op.less_equal(flex_value, relay.const(bucket)), padded_value, inp
                    )
                else:
                    padded_value = relay.If(
                        relay.op.logical_and(
                            relay.op.less_equal(flex_value, relay.const(bucket)),
                            relay.op.greater(flex_value, relay.const(buckets[i - 1])),
                        ),
                        padded_value,
                        inp,
                    )
                # Update input value and test dimension to reflect possible padding.
                input_data.append(padded_value)
            # Grab the new possibly padded shape for checking bucket size.
            check_dim = relay.op.take(relay.op.shape_of(input_data[0]), relay.const(axis))

        # Create a specialized subgraph for the current bucket.
        spec_call, spec_ty = specialize_body(
            mod, main_fn, axis, bucket, input_indices=input_indices, affects_output=affects_output
        )
        # Apply hard casting to shape to create statically typed graphs.
        spec_data = []
        for j, inp in enumerate(input_data):
            spec_data.append(relay.op.reshape(inp, spec_ty[j].shape))

        # Create a dispatch statement for the current specialized graph.
        call_args = list(main_fn.params)
        for j, inp in enumerate(input_indices):
            call_args[inp] = spec_data[j]
        new_call = spec_call(*call_args)

        # Remove meaningless padded outputs if applicable.
        if auto_pad and affects_output:
            new_call = relay.take(
                new_call,
                relay.arange(start=relay.const(0), stop=flex_value, dtype="int32"),
                axis=axis,
            )

        # Add this new case to the dispatch handler.
        if_exprs.append((relay.op.equal(check_dim, relay.const(bucket)), new_call))

    # Create a subgraph to handle all other shapes.
    default_dyn_call, _ = specialize_body(
        mod, main_fn, axis, relay.Any(), input_indices=input_indices, affects_output=affects_output
    )
    call_args = list(main_fn.params)
    for j, inp in enumerate(input_indices):
        call_args[inp] = dyn_data[j]
    new_body = default_dyn_call(*call_args)

    # Create an If chain to dispatch shapes to the appropriate specialized subgraph.
    for cond, true_branch in if_exprs:
        new_body = relay.If(cond, true_branch, new_body)

    # Assign new parameters to the function.
    new_params = list(main_fn.params)
    for j, inp in enumerate(input_indices):
        new_params[inp] = dyn_data[j]

    # Update the output shape to be dynamic if needed.
    if affects_output:
        dyn_ret_type = override_shape(main_fn.ret_type, axis, relay.Any())
    else:
        dyn_ret_type = main_fn.ret_type

    # Assign the handler as the new entrypoint in the module.
    new_main = relay.Function(
        new_params, new_body, dyn_ret_type, main_fn.type_params, main_fn.attrs
    )
    mod["main"] = new_main
    # Do type inference to make sure everything worked.
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

    By default, this pass will dispatch shapes that exactly match one
    of the buckets to a corresponding subgraph. All non-matching shapes
    use the same fully dynamic fallback. This can be detrimental to performance
    for those non-matching shapes. Setting auto_pad to True causes this
    pass to round-up the shape of non-matching inputs to the closest
    bucket. This allows them to use the tuned kernels of bucket shapes
    which can improve performance.

    Models that have multiple inputs sharing a dynamic axis, which
    is common for batch size or sequence length dynamism, are supported
    through the input_indices argument.

    Many types of dynamism such as batching affect both the input and output
    shape, however this is not always the case. If the output shape
    is independent of the input, the affects_output argument of this
    pass must be set to False.

    Parameters
    ----------
    buckets: list[int]
        The sizes of the input dimension that should be explicitly handled.
        Each value in buckets will have a corresponding subgraph constructed to
        handle it.
    axis: int
        The dimension of the input that should be made flexible. This will
        most often be used for the batch dimension.
    auto_pad: Optional[bool]
        If True, then padding will be inserted to values that don't match one of
        the provided buckets.
    pad_value: Optional[float]
        When auto_pad is true, padding will be done with this value.
    input_indices: Optional[List[int]]
        Which inputs should be dispatched dynamically, provided by index. All inputs
        must share the same dynamic axis.
    affects_output: Optional[bool]
        Whether the change in input shape has a corresponding effect on the output shape.
        Batching for example effects both the input and output whereas changing sequence
        length in an NLP model typically does not.

    Returns
    -------
    ret : FlexibleShapeDispatch
        A pass that can be applied to a module to add flexible shape handling.
    """

    def __init__(
        self,
        buckets,
        axis=0,
        auto_pad=False,
        pad_value=0,
        input_indices=None,
        affects_output=True,
    ):
        self.axis = axis
        self.buckets = buckets
        self.auto_pad = auto_pad
        self.pad_value = pad_value
        self.input_indices = input_indices
        self.affects_output = affects_output
        super(FlexibleShapeDispatch, self).__init__()

    def __call__(self, mod):
        # Shape information is required for this pass.
        mod = relay.transform.InferType()(mod)
        return flexible_dispatch(
            mod,
            self.buckets,
            self.axis,
            self.auto_pad,
            self.pad_value,
            self.input_indices,
            self.affects_output,
        )
