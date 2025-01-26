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


def generate_mixed_precision_rule(acc_dtype):
    def _mixed_precision_rule(call_node: "relay.Call", mixed_precision_type: str):
        return [
            relay.transform.mixed_precision.MIXED_PRECISION_ALWAYS,
            acc_dtype,
            mixed_precision_type,
        ]

    return _mixed_precision_rule


class MixedPrecision(object):
    """Temporarily changes attr of ops to enable required precision."""

    def __init__(self, ops, acc_type):
        """Saves the required info for RAII pattern usage.

        Parameters
        ----------
        ops : list
            list of operators
        acc_type: str
            Output or accumulation precision to be used.
        """
        self.older_attr = {}
        self.ops = ops
        self.acc_type = acc_type
        self.attr_key = "FTVMMixedPrecisionConversionType"

    def __enter__(self):
        for op_name in self.ops:
            op = relay.op.get(op_name)
            self.older_attr[op_name] = op.get_attr(self.attr_key)
            op.reset_attr(self.attr_key)
            op.set_attr(self.attr_key, generate_mixed_precision_rule(self.acc_type))
        return self

    def __exit__(self, ptype, value, trace):
        for op_name in self.ops:
            op = relay.op.get(op_name)
            op.reset_attr(self.attr_key)
            if self.older_attr[op_name]:
                op.set_attr(self.attr_key, self.older_attr[op_name])


def convert_to_mixed_precision(mod, ops=None, calculation_type="float16", acc_type="float16"):
    """Converts the operator datatypes

    Parameters
    ----------
    mod : tvm.IRModule
        The relay module to convert.
    ops : list
        List of operators to be precision converted.
    calculation_type: str
        Input precision to be used.
    acc_type: str
        Output or accumulation precision to be used.

    Returns
    -------
    mod : tvm.IRModule
        The converted module.
    """

    if ops is None:
        ops = ["nn.conv2d", "nn.dense"]

    with MixedPrecision(ops, acc_type):
        seq = transform.Sequential(
            [relay.transform.InferType(), relay.transform.ToMixedPrecision(calculation_type)]
        )
        with transform.PassContext(
            config={"relay.ToMixedPrecision.keep_orig_output_dtype": True}, opt_level=3
        ):
            try:
                return seq(mod)
            except Exception as err:
                raise TVMCException("Error converting mixed precision : {0}".format(str(err)))


def convert_graph_layout(mod, desired_layouts, ops=None):
    """Alter the layout of the input graph.

    Parameters
    ----------
    mod : tvm.IRModule
        The relay module to convert.
    desired_layouts : list[str]
        The layouts to convert to.
        Expects either a single element or one str per operator.
        Can be only data layouts or combination of both, e.g. NHWC:HWIO
    ops : list
        List of operators to be layout converted.

    Returns
    -------
    mod : tvm.IRModule
        The converted module.
    """
    if ops is None:
        ops = ["nn.conv2d", "nn.conv2d_transpose", "qnn.conv2d"]

    if not isinstance(desired_layouts, list):
        # For backwards compatibility
        assert isinstance(desired_layouts, str)
        desired_layouts = [desired_layouts]

    if len(desired_layouts) != len(ops):
        if len(desired_layouts) != 1:
            raise TVMCException(
                "Expected 1 or {} layouts but got {}".format(len(ops), len(desired_layouts))
            )
        desired_layouts = desired_layouts * len(ops)

    def layout_helper(layout):
        if ":" in layout:
            data_layout, kernel_layout = layout.split(":", 1)
        else:
            data_layout = layout
            kernel_layout = "default"
        return [data_layout, kernel_layout]

    desired_layouts = {op: layout_helper(desired_layouts[i]) for i, op in enumerate(ops)}

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
        raise TVMCException("Error converting layouts: {}".format(str(err)))


def apply_graph_transforms(mod, args, params=None):
    """Alter the layout of the input graph.

    Parameters
    ----------
    mod : tvm.IRModule
        The relay module to convert.
    args : dict
        The transform arguments.
    params: dict
        Module params

    Returns
    -------
    mod : tvm.IRModule
        The converted module.
    """
    if not args:
        return mod

    # AlterLayout
    if args.get("desired_layout", None):
        mod = convert_graph_layout(
            mod, args["desired_layout"], args.get("desired_layout_ops", None)
        )

    # ToMixedPrecision
    if args.get("mixed_precision", False):
        mod = relay.quantize.prerequisite_optimize(mod, params)
        mod = convert_to_mixed_precision(
            mod,
            args.get("mixed_precision_ops"),
            args.get("mixed_precision_calculation_type"),
            args.get("mixed_precision_acc_type"),
        )
    return mod


def parse_graph_transform_args(args):
    """Parse incoming options for graph transform arguments.

    Parameters
    ----------
    args: argparse.Namespace or dict
        Arguments.

    Returns
    -------
    transform_args : dict
        Graph transform arguments
    """

    if not isinstance(args, dict):
        args = vars(args)

    transform_args = [
        "desired_layout",
        "desired_layout_ops",
        "mixed_precision",
        "mixed_precision_ops",
        "mixed_precision_calculation_type",
        "mixed_precision_acc_type",
    ]
    transform_args = {key: args.get(key, None) for key in transform_args}
    return transform_args


def generate_transform_args(parser):
    """Add graph transform related args"""

    # AlterLayout
    parser.add_argument(
        "--desired-layout",
        nargs="+",
        help="Change the data/kernel layout of the graph. (i.e. NCHW or NHWC:HWIO)"
        "This option can be provided multiple times to specify per-operator layouts, "
        "e.g. '--desired-layout NHWC:HWIO' (Apply same layout for every operator)."
        "e.g. '--desired-layout-ops nn.conv2d nn.avg_pool2d --desired-layout NCHW NHWC'.",
    )
    parser.add_argument(
        "--desired-layout-ops",
        default=["nn.conv2d", "nn.conv2d_transpose", "qnn.conv2d"],
        nargs="+",
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
        default=["nn.conv2d", "nn.dense"],
        nargs="+",
        help="List of operators to be converted to mixed precision",
    )
    parser.add_argument(
        "--mixed-precision-calculation-type",
        choices=["float16", "float32"],
        default="float16",
        help="Calculation precision type",
    )
    parser.add_argument(
        "--mixed-precision-acc-type",
        choices=["float16", "float32"],
        default="float16",
        help="Accumulator precision type",
    )
