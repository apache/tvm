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
# specific language
# pylint: disable=unused-argument
"""
TVMC Graph Transforms
"""

from tvm import relay, transform
from tvm.driver.tvmc import TVMCException

# ToMixedPrecision
ACC_DTYPE = "float32"


def mixed_precision_rule(call_node: "relay.Call", mixed_precision_type: str):
    global ACC_DTYPE
    return [
        relay.transform.mixed_precision.MIXED_PRECISION_ALWAYS,
        ACC_DTYPE,
        mixed_precision_type,
    ]


class MixedPrecision(object):
    """Temporarily changes attr of ops to enable required precision."""

    def __init__(self, ops):
        """Saves the required info for RAII pattern usage.

        Parameters
        ----------
        ops : list
            list of operators
        """
        self.older_attr = {}
        self.ops = ops
        self.attr_key = "FTVMMixedPrecisionConversionType"

    def __enter__(self):
        for op_name in self.ops:
            op = relay.op.get(op_name)
            self.older_attr[op_name] = op.get_attr(self.attr_key)
            op.reset_attr(self.attr_key)
            op.set_attr(self.attr_key, mixed_precision_rule)
        return self

    def __exit__(self, ptype, value, trace):
        for op_name in self.ops:
            op = relay.op.get(op_name)
            op.reset_attr(self.attr_key)
            if self.older_attr[op_name]:
                op.set_attr(self.attr_key, self.older_attr[op_name])


def convert_to_mixed_precision(
    mod, ops="nn.conv2d,nn.dense", input_type="float16", out_type="float16"
):
    """Converts the operator datatypes

    Parameters
    ----------
    mod : tvm.IRModule
        The relay module to convert.
    ops : str
        List of operators to be precision converted.
    input_type: str
        Input precision to be used.
    output_type: str
        Output or accumulation precision to be used.

    Returns
    -------
    mod : tvm.IRModule
        The converted module.
    """

    global ACC_DTYPE
    ACC_DTYPE = out_type

    with MixedPrecision(ops.split(",")):
        seq = transform.Sequential(
            [relay.transform.InferType(), relay.transform.ToMixedPrecision()]
        )
        with transform.PassContext(
            config={"relay.ToMixedPrecision.keep_orig_output_dtype": True}, opt_level=3
        ):
            try:
                return seq(mod)
            except Exception as err:
                raise TVMCException("Error converting mixed precision : {0}".format(str(err)))


def convert_graph_layout(mod, desired_layout, ops="nn.conv2d,nn.conv2d_transpose,qnn.conv2d"):
    """Alter the layout of the input graph.

    Parameters
    ----------
    mod : tvm.IRModule
        The relay module to convert.
    desired_layout : str
        The layout to convert to.
    ops : str
        List of operators to be layout converted.

    Returns
    -------
    mod : tvm.IRModule
        The converted module.
    """

    desired_layouts = {op: [desired_layout, "default"] for op in ops.split(",")}

    # Convert the layout of the graph where possible.
    seq = transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout(desired_layouts),
            relay.transform.FoldConstant(),
        ]
    )

    try:
        return seq(mod)
    except Exception as err:
        raise TVMCException("Error converting layout to {0}: {1}".format(desired_layout, str(err)))


def apply_graph_transforms(mod, args):
    """Alter the layout of the input graph.

    Parameters
    ----------
    mod : tvm.IRModule
        The relay module to convert.
    args : dict
        The transform arguments.

    Returns
    -------
    mod : tvm.IRModule
        The converted module.
    """
    if not args:
        return mod

    # AlterLayout
    if args.get("desired_layout", False):
        mod = convert_graph_layout(mod, args["desired_layout"])

    # ToMixedPrecision
    if args.get("mixed_precision", False):
        mod = convert_to_mixed_precision(
            mod,
            args.get("mixed_precision_ops", "nn.conv2d,nn.dense"),
            args.get("mixed_precision_input", "float16"),
            args.get("mixed_precision_output", "float16"),
        )
    return mod


def parse_graph_transform_args(args):
    """Parse incoming options for graph transform arguments.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments from command line parser.

    Returns
    -------
    transform_args : dict
        Graph transform arguments
    """

    args_dict = vars(args)

    transform_args = [
        "desired_layout",
        "desired_layout_ops",
        "mixed_precision",
        "mixed_precision_ops",
        "mixed_precision_input",
        "mixed_precision_output",
    ]
    transform_args = {key: args_dict.get(key, None) for key in transform_args}
    return transform_args


def generate_transform_args(parser):
    """Add graph transform related args"""

    # AlterLayout
    parser.add_argument(
        "--desired-layout",
        choices=["NCHW", "NHWC"],
        default=None,
        help="Change the data layout of the whole graph.",
    )
    parser.add_argument(
        "--desired-layout-ops",
        default="nn.conv2d,nn.conv2d_transpose,qnn.conv2d",
        help="List of operators to be layout converted.",
    )

    # ToMixedPrecision
    parser.add_argument(
        "--mixed-precision",
        help="Enable mixed precision conversion",
        action="store_true",
    )
    parser.add_argument(
        "--mixed-precision-ops",
        default="nn.conv2d,nn.dense",
        help="List of operators to be converted to mixed precision",
    )
    parser.add_argument(
        "--mixed-precision-input",
        choices=["float16", "float32"],
        default="float16",
        help="Input precision type",
    )
    parser.add_argument(
        "--mixed-precision-output",
        choices=["float16", "float32"],
        default="float16",
        help="Output or accumulator precision type",
    )
