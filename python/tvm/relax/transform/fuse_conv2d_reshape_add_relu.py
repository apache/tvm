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

"""This module provides a TVM Relax pass for fusing Conv2d-Reshape-Add-ReLU pattern."""

import tvm
from tvm import IRModule, relax
from tvm.relax.dpl.pattern import is_op, wildcard

# Define a TVM module pass for fusing specific operations.
# @tvm.transform.module_pass decorates a class to turn it into a TVM IRModule pass.
# opt_level=0 means this pass can be run at any optimization level.
# name="FuseConv2dReshapeAddRelu" gives a descriptive name to the pass.


@tvm.transform.module_pass(opt_level=0, name="FuseConv2dReshapeAddRelu")
class FuseConv2dReshapeAddRelu:
    """A Relax pass that fuses the Conv2d-Reshape-Add-ReLU pattern into a composite function."""

    # The main transformation method that applies the pass to an IRModule.
    # mod: The input IRModule to be transformed.
    # _ctx: PassContext (unused in this specific pass but required by the decorator).
    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Transforms the input IRModule by applying the Conv2d-Reshape-Add-ReLU fusion.

        Parameters
        ----------
        mod : IRModule
            The input IRModule to be transformed.
        _ctx : tvm.transform.PassContext
            The pass context (unused in this specific pass but required by the decorator).

        Returns
        -------
        IRModule
            The transformed IRModule with the fused pattern.
        """
        # Apply the FuseOpsByPattern transformation.
        # This pass identifies specific operator patterns in the IRModule
        # and fuses them into a single composite function.
        mod = relax.transform.FuseOpsByPattern(
            # Define the patterns to fuse. It's a list of tuples:
            # ("composite_function_name", pattern_root, annotations, check_function)
            # "dnnl.conv2d_reshape_add_relu" is the name given to the fused operation,
            # indicating it's suitable for DNNL backend.
            [("dnnl.conv2d_reshape_add_relu", *_conv2d_reshape_add_relu_pattern())],
            # bind_constants=False means that constants in the pattern (like shapes)
            # are not treated as part of the pattern to be matched, allowing for more flexibility.
            bind_constants=False,
        )(mod)

        # Return the transformed IRModule.
        return mod


# Helper function to define the operator fusion pattern for Conv2d-Reshape-Add-ReLU.
# This function uses TVM's declarative pattern language (DPL).
def _conv2d_reshape_add_relu_pattern():
    # Define wildcard placeholders for the input tensors.
    # 'wildcard()' matches any Relax expression.
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    shape = wildcard()  # Wildcard for the target shape of the reshape operation

    # Define the sequence of operations in the pattern:
    # 1. Convolution (relax.nn.conv2d)
    #    varg_default_wildcard=True means that any variadic arguments (like strides, padding)
    #    will also be matched by wildcards, making the pattern more general.
    conv_out = is_op("relax.nn.conv2d")(data, weight, varg_default_wildcard=True)
    # 2. Reshape (relax.reshape)
    #    This matches a reshape operation applied to the 'bias' tensor with any 'shape'.
    reshaped_bias = is_op("relax.reshape")(bias, shape)
    # 3. Addition (relax.add)
    #    This matches an add operation where 'conv_out' and 'reshaped_bias' are inputs.
    add_out = is_op("relax.add")(conv_out, reshaped_bias)
    # 4. ReLU (relax.nn.relu)
    #    This matches a ReLU operation applied to the output of the add operation.
    relu_out = is_op("relax.nn.relu")(add_out)

    # Define annotations for the pattern.
    # These map internal names (keys) to the matched Relax expressions (values).
    # This is useful for debugging and for custom check functions.
    annotations = {
        "conv_out": conv_out,
        "reshaped_bias": reshaped_bias,
        "add_out": add_out,
        "relu_out": relu_out,
    }

    # Define a custom check function for the pattern.
    # This function is executed after a potential match is found.
    # It can be used to add more specific conditions for the fusion.
    # In this case, 'return True' means it always matches if the structure is found.
    def _check(_context):
        """A check function for the pattern (currently always returns True)."""
        return True

    # Return the root of the pattern, the annotations, and the check function.
    # The 'relu_out' is the final output of the sequence being matched.
    return relu_out, annotations, _check
