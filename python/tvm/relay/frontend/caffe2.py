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
# pylint: disable=import-self, invalid-name, line-too-long, unused-argument
"""Caffe2 frontend"""
import tvm
from tvm.ir import IRModule

from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from ... import nd as _nd
from .common import AttrCvt, Renamer
from .common import get_relay_op, new_var, infer_channels

__all__ = ["from_caffe2"]


def dimension_picker(prefix, surfix=""):
    def _impl(attr):
        kernel = attr["kernel_shape"]
        if len(kernel) == 2:
            return prefix + "2d" + surfix
        raise tvm.error.OpAttributeUnImplemented(
            "Non-2D kernels are not supported for operator {}2d".format(prefix)
        )

    return _impl


def revert_caffe2_pad(pads):
    """Caffe2 requires two times the normal padding."""
    if len(pads) == 4:
        pads = pads[:2]
    elif len(pads) == 2:
        pass
    else:
        raise tvm.error.OpAttributeInvalid("Number of pads must equal 2 or 4.")
    return pads


def dimension_constraint():
    def _dim_check(args):
        if len(args["kernel_shape"]) == 2:
            return True
        return False

    return _dim_check, "Only 2d kernel supported."


def _clean_up_pool_args(args):
    """A helper function to clean up common arguments in conv and pooling ops."""
    assert isinstance(args, dict)

    if "stride_h" in args and "stride_w" in args:
        assert "stride" not in args and "strides" not in args
        args["strides"] = [args["stride_h"], args["stride_w"]]
        args.pop("stride_h")
        args.pop("stride_w")
    elif "stride" in args:
        args["strides"] = [args["stride"], args["stride"]]
        args.pop("stride")

    # rename 'kernel', 'kernels', to 'kernel_shape'
    if "kernel_h" in args and "kernel_w" in args:
        assert "kernel" not in args and "kernels" not in args
        args["kernel_shape"] = [args["kernel_h"], args["kernel_w"]]
        args.pop("kernel_h")
        args.pop("kernel_w")
    elif "kernel" in args:
        args["kernel_shape"] = [args["kernel"], args["kernel"]]
        args.pop("kernel")
    elif "kernels" in args:
        args["kernel_shape"] = args["kernels"]
        args.pop("kernels")

    if "pad_t" in args and "pad_l" in args and "pad_b" in args and "pad_r" in args:
        assert "pad" not in args and "pads" not in args
        args["pads"] = [args["pad_t"], args["pad_l"], args["pad_b"], args["pad_r"]]
        for pad in ["pad_t", "pad_l", "pad_b", "pad_r"]:
            args.pop(pad)
    elif "pad" in args:
        args["pads"] = [args["pad"], args["pad"]]
        args.pop("pad")

    if "dilation_h" in args and "dilation_w" in args:
        assert "dilation" not in args and "dilations" not in args
        args["dilations"] = [args["dilation_h"], args["dilation_w"]]
        args.pop("dilation_h")
        args.pop("dilation_w")
    elif "dilation" in args:
        args["dilations"] = [args["dilation"], args["dilation"]]
        args.pop("dilation")

    return args


class Caffe2OpConverter(object):
    """A helper class for holding Caffe2 op converters."""

    @classmethod
    def get_converter(cls):
        """Get converter.

        :return: converter, which should be `_impl`.
        """

        if hasattr(cls, "_impl"):
            return getattr(cls, "_impl")
        raise tvm.error.OpNotImplemented(
            "Operator {} is not supported in frontend Caffe2.".format(cls.__name__)
        )


_caffe2_internal_args = [
    # nnpack args
    "algo",
    "convolution_transform_strategy",
    "float16_compute",
    "shared_buffer",
    # training args
    "init_params",
    "cudnn_exhaustive_search",
    "exhaustive_search",
    # training args
    "adj",
    "hwgq",
    # args that we don't care
    "legacy_pad",
]


class Elemwise(Caffe2OpConverter):
    """A helper class for elemwise op converters."""

    name = ""

    @classmethod
    def _impl(cls, inputs, args, params):
        assert len(inputs) == 2, "Math op take 2 inputs, {} given".format(len(inputs))
        op_name = cls.name
        conv_ops = ["conv2d", "conv2d_transpose"]
        if args.get("broadcast", 0) and any(x in str(inputs[0]) for x in conv_ops):
            # TODO(zhreshold): remove hard coded infershape
            axis = int(args.get("axis", 0))
            inputs[1] = _op.expand_dims(inputs[1], axis=axis, num_newaxis=2)
        return get_relay_op(op_name)(*inputs)


class Add(Elemwise):
    """Operator converter for Add."""

    name = "add"


class Mul(Elemwise):
    """Operator converter for Mul."""

    name = "multiply"


class Pool(Caffe2OpConverter):
    """A helper class for pool op converters."""

    name = ""

    @classmethod
    def _impl(cls, inputs, args, params):
        _clean_up_pool_args(args)
        if "global_pooling" in args and args["global_pooling"] == 1:
            op_name = dimension_picker("global_" + cls.name)
            return get_relay_op(op_name(args))(*inputs)

        return AttrCvt(
            op_name=dimension_picker(cls.name),
            transforms={
                "kernel_shape": "pool_size",
                "pads": ("padding", (0, 0), revert_caffe2_pad),
                "strides": "strides",
            },
            ignores=["dilations", "order", "legacy_pad", "global_pooling"],
            extras={"ceil_mode": False},
            custom_check=dimension_constraint(),
        )(inputs, args, params)


class AveragePool(Pool):
    name = "avg_pool"


class MaxPool(Pool):
    name = "max_pool"


class Conv(Caffe2OpConverter):
    """Operator converter for Conv."""

    @classmethod
    def _impl(cls, inputs, args, params):
        # get number of channels
        channels = infer_channels(inputs[1])
        args["channels"] = channels
        _clean_up_pool_args(args)
        out = AttrCvt(
            op_name=dimension_picker("conv"),
            transforms={
                "group": ("groups", 1),
                "kernel_shape": "kernel_size",
                "pads": ("padding", (0, 0), revert_caffe2_pad),
                "strides": "strides",
                "dilations": ("dilation", (1, 1)),
                "order": (
                    "data_layout",
                    ("NCHW"),
                    lambda x: x if isinstance(x, str) else x.decode("UTF-8"),
                ),
            },
            excludes=[],
            ignores=_caffe2_internal_args,
            custom_check=dimension_constraint(),
        )(inputs[:2], args, params)
        use_bias = len(inputs) == 3
        if use_bias:
            out = _op.nn.bias_add(out, inputs[2])
        return out


class ConvTranspose(Caffe2OpConverter):
    """Operator converter for ConvTranspose."""

    @classmethod
    def _impl(cls, inputs, args, params):
        # get number of channels
        channels = infer_channels(inputs[1], True)
        args["channels"] = channels
        _clean_up_pool_args(args)
        out = AttrCvt(
            op_name=dimension_picker("conv", "_transpose"),
            transforms={
                "kernel_shape": "kernel_size",
                "pads": ("padding", (0, 0), revert_caffe2_pad),
                "dilations": ("dilation", (1, 1)),
                "order": (
                    "data_layout",
                    ("NCHW"),
                    lambda x: x if isinstance(x, str) else x.decode("UTF-8"),
                ),
            },
            excludes=[],
            ignores=_caffe2_internal_args,
            custom_check=dimension_constraint(),
        )(inputs[:2], args, params)
        use_bias = len(inputs) == 3
        if use_bias:
            out = _op.nn.bias_add(out, inputs[2])
        return out


class Concat(Caffe2OpConverter):
    """Operator converter for Concat."""

    @classmethod
    def _impl(cls, inputs, args, params):
        def _get_axis_from_order_str(order):
            order = order if isinstance(order, str) else order.decode("UTF-8")
            if order == "NCHW":
                return 1
            if order == "NHWC":
                return 3
            raise tvm.error.OpAttributeUnImplemented(
                "Order {} is not supported in operator Concat.".format(order)
            )

        return AttrCvt(
            op_name="concatenate",
            transforms={
                "order": ("axis", (1), _get_axis_from_order_str),
            },
            excludes=["add_axis"],
        )((inputs,), args, params)


class NormalizePlanarYUV(Caffe2OpConverter):
    """Operator converter for NormalizePlanarYUV.
    caffe2 definition: https://github.com/pytorch/pytorch/blob/master/caffe2/operators/norm_planar_yuv_op.cc
    """

    @classmethod
    def _impl(cls, inputs, args, params):
        assert len(inputs) == 3
        mean = _op.expand_dims(inputs[1], axis=2, num_newaxis=2)
        std = _op.expand_dims(inputs[2], axis=2, num_newaxis=2)

        return _op.divide(_op.subtract(inputs[0], mean), std)


class ResizeNearest(Caffe2OpConverter):
    """Operator converter for Upsample (nearest mode)."""

    @classmethod
    def _impl(cls, inputs, args, params):
        width_scale = args["width_scale"] if "width_scale" in args else 1
        height_scale = args["height_scale"] if "height_scale" in args else 1
        assert width_scale == height_scale

        return _op.nn.upsampling(
            inputs[0], scale_h=int(width_scale), scale_w=int(width_scale), method="NEAREST_NEIGHBOR"
        )


class Sum(Caffe2OpConverter):
    """Operator converter for Sum."""

    @classmethod
    def _impl(cls, inputs, args, params):
        # Sum Operator
        for in_index in range(len(inputs) - 1):
            inputs[in_index + 1] = _op.add(inputs[in_index], inputs[in_index + 1])

        return inputs[len(inputs) - 1]


class Softmax(Caffe2OpConverter):
    """Operator converter for Softmax."""

    @classmethod
    def _impl(cls, inputs, args, params):
        # set default value when axis is not set in the model
        if "axis" not in args:
            args["axis"] = 1
        return AttrCvt("softmax", transforms={"axis": ("axis", args["axis"])})(inputs, args, params)


class FC(Caffe2OpConverter):
    """Operator converter for FC."""

    @classmethod
    def _impl(cls, inputs, args, params):
        inputs[0] = _op.nn.batch_flatten(inputs[0])
        units = infer_channels(inputs[1])
        res = _op.nn.dense(inputs[0], inputs[1], units=units)
        use_bias = len(inputs) == 3
        if use_bias:
            res = _op.nn.bias_add(res, inputs[2])
        return res


class SpatialBN(Caffe2OpConverter):
    """Operator converter for SpatialBN."""

    @classmethod
    def _impl(cls, inputs, args, params):
        return AttrCvt(
            op_name="batch_norm",
            disables=["momentum"],
            ignores=["order", "spatial", "is_test", "consumed_inputs", "num_batches"],
        )(inputs, args, params)


# compatible operators that do NOT require any conversion.
_identity_list = []

# _convert_map defines maps of name to converter functor(callable)
# for 1 to 1 mapping, use Renamer if nothing but name is different
# use AttrCvt if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping, currently not supported(?)

# Minimal set of ops for squeezenet and resnet50
def _get_convert_map():
    return {
        # caffe2 common operators
        "Add": Add.get_converter(),
        "Sum": Sum.get_converter(),
        "Mul": Mul.get_converter(),
        "Softmax": Softmax.get_converter(),
        # nn
        "AveragePool": AveragePool.get_converter(),
        "MaxPool": MaxPool.get_converter(),
        "Conv": Conv.get_converter(),
        "ConvTranspose": ConvTranspose.get_converter(),
        "Concat": Concat.get_converter(),
        "FC": FC.get_converter(),
        "SpatialBN": SpatialBN.get_converter(),
        "ResizeNearest": ResizeNearest.get_converter(),
        "Relu": AttrCvt("relu", {}, ignores=["order"]),
        "Sigmoid": Renamer("sigmoid"),
        "Dropout": AttrCvt("dropout", {"ratio": "rate"}, ignores=["is_test"]),
        # c2 image preprocessing ops
        "NormalizePlanarYUV": NormalizePlanarYUV.get_converter(),
    }


class Caffe2NetDef(object):
    """A helper class for handling Relay expression copying from pb2.GraphProto.
    Definition: https://github.com/pytorch/pytorch/blob/master/caffe2/proto/caffe2.proto
    """

    def __init__(self, shape, dtype):
        self._nodes = {}
        self._params = {}
        self._visited_nodes = set()
        self._ops = {}
        self._shape = shape
        self._dtype = dtype
        self._mod = IRModule({})

    def from_caffe2(self, init_net, predict_net):
        """Construct Relay expression from caffe2 graph.

        Parameters
        ----------
        init_net : protobuf object
        predict_net : protobuf object

        Returns
        -------
        mod : tvm.IRModule
            The module that optimizations will be performed on.

        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """
        # pylint: disable=import-outside-toplevel
        from caffe2.python import workspace

        workspace.RunNetOnce(init_net)

        # Input
        input_name = predict_net.op[0].input[0]

        # Params
        self._params = {}
        used_blobs = set()
        for c2_op in predict_net.op:
            for i in c2_op.input:
                used_blobs.add(i)
        for blob in workspace.Blobs():
            if blob in used_blobs and blob != input_name:
                self._params[blob] = _nd.array(workspace.FetchBlob(blob))

        # Variables
        self._nodes = {}
        for blob in predict_net.external_input:
            if blob in self._params:
                self._nodes[blob] = new_var(
                    blob, shape=self._params[blob].shape, dtype=self._params[blob].dtype
                )
            else:
                shape = self._shape[blob] if blob in self._shape else ()
                if isinstance(self._dtype, dict) and blob in self._dtype:
                    dtype = str(self._dtype[blob])
                elif isinstance(self._dtype, str):
                    dtype = self._dtype
                else:
                    dtype = "float32"
                self._nodes[blob] = new_var(blob, shape=shape, dtype=dtype)

        # Ops
        for c2_op in predict_net.op:
            for blob in c2_op.output:
                self._ops[blob] = c2_op

        for c2_op in predict_net.op:
            self._process_op(c2_op)

        # Outputs
        out = []
        for blob in predict_net.external_output:
            out.append(self._nodes[blob])

        if len(out) > 1:
            outputs = _expr.Tuple(out)
        else:
            outputs = out[0]

        func = _function.Function(analysis.free_vars(outputs), outputs)
        self._mod["main"] = func

        return self._mod, self._params

    def _get_node(self, blob):
        """Get the Symbol of blob and detect cyclic dependency in the graph."""
        if blob in self._nodes:
            return self._nodes[blob]

        assert blob not in self._visited_nodes, "Cyclic dependency in the graph (in {})".format(
            blob
        )
        self._visited_nodes.add(blob)

        self._process_op(self._ops[blob])
        return self._nodes[blob]

    def _process_op(self, c2_op):
        op_type = c2_op.type
        args = self._parse_arg(c2_op.arg)
        inputs = [self._get_node(i) for i in c2_op.input]
        tvm_op = self._convert_operator(op_type, inputs, args)

        if not isinstance(tvm_op, _expr.TupleWrapper):
            self._nodes[c2_op.output[0]] = tvm_op
        else:
            for k, i in zip(list(c2_op.output), range(len(tvm_op))):
                self._nodes[k] = tvm_op[i]

    def _parse_arg(self, arg):
        """Convert a list of Argument to a dict, with names as keys."""
        args = {}
        for a in arg:
            for f in ["f", "i", "s"]:
                if a.HasField(f):
                    args[a.name] = getattr(a, f)
            for f in ["floats", "ints", "strings"]:
                if list(getattr(a, f)):
                    assert a.name not in args, "Only one type of attr is allowed"
                    args[a.name] = tuple(getattr(a, f))
            for f in ["n"]:
                if a.HasField(f):
                    raise NotImplementedError("Field {} is not supported in relay.".format(f))
            for f in ["nets"]:
                if list(getattr(a, f)):
                    raise NotImplementedError("Field {} is not supported in relay.".format(f))
            if a.name not in args:
                raise ValueError("Cannot parse attribute: \n{}\n.".format(a))
        return args

    def _convert_operator(self, op_type, inputs, args, identity_list=None, convert_map=None):
        """Convert from Caffe2 operator to Relay operator.
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        op_type : str
            Operator name, such as Convolution, FullyConnected
        inputs : list of tvm.relay.function.Function
            List of input inputs.
        args : dict
            Dict of operator attributes
        identity_list : list
            List of operators that don't require conversion
        convert_map : dict
            Dict of name : callable, where name is the op's name that
            require conversion to relay, callable are functions which
            take args and return (new_op_type, new_args)

        Returns
        -------
        func : tvm.relay.function.Function
            Converted relay function
        """
        identity_list = identity_list if identity_list else _identity_list
        convert_map = convert_map if convert_map else _get_convert_map()
        if op_type in identity_list:
            func = get_relay_op(op_type)(*inputs, **args)
        elif op_type in convert_map:
            # Add a sanitizing step to convert all byte strings in args to strings
            func = convert_map[op_type](inputs, args, self._params)
        else:
            raise tvm.error.OpNotImplemented(
                "Operator {} is not supported in frontend Caffe2.".format(op_type)
            )
        return func


def from_caffe2(init_net, predict_net, shape=None, dtype="float32"):
    """Load caffe2 graph which contains init_net and predict_net into Relay Function.

    Parameters
    ----------
    init_net : protobuf object
        Caffe2 NetDef containing the weights

    predict_net : protobuf object
        Caffe2 NetDef containing the graph

    shape : dict of str to tuple
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph

    Returns
    -------
    mod : tvm.IRModule
        The module that optimizations will be performed on.

    params : dict of str to tvm.nd.NDArray
        Dict of converted parameters stored in tvm.nd.NDArray format
    """

    caffe2 = Caffe2NetDef(shape, dtype)
    return caffe2.from_caffe2(init_net, predict_net)
