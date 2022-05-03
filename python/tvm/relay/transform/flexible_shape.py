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
from tvm import relay


def override_shape(ty, dim, value):
    """Change a value in a tensor shape."""
    new_dims = list(ty.shape)
    new_dims[dim] = value
    return relay.TensorType(new_dims, ty.dtype)


def specialize_body(mod, fn, dim, value, input_indices=[0], affects_output=True):
    """Create a subgraph to handle a specific input shape"""
    # Iterate through specified inputs and construct specialized shapes.
    new_params = list(fn.params)
    data_binding = {}
    dyn_data_array = []
    for inp in input_indices:
        data = fn.params[inp]
        flex_ty = override_shape(data.type_annotation, dim, value)
        dyn_data = relay.Var(data.name_hint, type_annotation=flex_ty)
        new_params[inp] = dyn_data
        data_binding[data] = dyn_data
        dyn_data_array.append(dyn_data)

    new_body = relay.expr.bind(fn.body, data_binding)
    # Only change the output shape if the input shape affects it.
    if affects_output:
        new_ret_ty = override_shape(fn.ret_type, dim, value)
    else:
        new_ret_ty = fn.ret_type
    gvar = relay.GlobalVar("main_" + str(value))
    mod[gvar] = relay.Function(new_params, new_body, new_ret_ty, fn.type_params, fn.attrs)
    return gvar, [d.type_annotation for d in dyn_data_array]


def flexible_dispatch(
    mod, dim=0, buckets=[], auto_pad=False, pad_value=0, input_indices=[0], affects_output=True
):
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

    # Extract all input data and create a new dynamic variable for each.
    data = []
    dyn_data = []
    for i in input_indices:
        data.append(main_fn.params[i])
        dyn_shape = override_shape(data[i].type_annotation, dim, relay.Any())
        dyn_data.append(relay.Var(data[i].name_hint, type_annotation=dyn_shape))

    # Extract the dynamic shape value from one of the inputs.
    rt_sh = relay.op.shape_of(dyn_data[0])
    flex_value = relay.op.take(rt_sh, relay.const(dim))

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
                pads = relay.scatter_nd(pads, relay.const([dim, 1]), pad_width)
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
            check_dim = relay.op.take(relay.op.shape_of(input_data[0]), relay.const(dim))

        # Create a specialized subgraph for the current bucket.
        spec_call, spec_ty = specialize_body(
            mod, main_fn, dim, bucket, input_indices=input_indices, affects_output=affects_output
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
                axis=dim,
            )

        # Add this new case to the dispatch handler.
        if_exprs.append((relay.op.equal(check_dim, relay.const(bucket)), new_call))

    # Create a subgraph to handle all other shapes.
    default_dyn_call, _ = specialize_body(
        mod, main_fn, dim, relay.Any(), input_indices=input_indices, affects_output=affects_output
    )
    call_args = list(main_fn.params)
    for j, inp in enumerate(input_indices):
        call_args[inp] = dyn_data[j]
    new_body = default_dyn_call(*call_args)

    for cond, true_branch in if_exprs:
        new_body = relay.If(cond, true_branch, new_body)

    new_params = list(main_fn.params)
    for j, inp in enumerate(input_indices):
        new_params[inp] = dyn_data[j]

    if affects_output:
        dyn_ret_type = override_shape(main_fn.ret_type, dim, relay.Any())
    else:
        dyn_ret_type = main_fn.ret_type

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
        self, dim=0, buckets=[], auto_pad=False, pad_value=0, input_indices=[0], affects_output=True
    ):
        self.dim = dim
        self.buckets = buckets
        self.auto_pad = auto_pad
        self.pad_value = pad_value
        self.input_indices = input_indices
        self.affects_output = affects_output
        super(FlexibleShapeDispatch, self).__init__()

    def __call__(self, mod):
        mod = relay.transform.InferType()(mod)
        return flexible_dispatch(
            mod,
            self.dim,
            self.buckets,
            self.auto_pad,
            self.pad_value,
            self.input_indices,
            self.affects_output,
        )
