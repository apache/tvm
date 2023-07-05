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
# pylint: disable=unused-argument, bad-chained-comparison
"""A Relay implementation of graph packing."""

import tvm
from tvm import relay
from tvm.relay import op, transform
from tvm.relay import ExprMutator


def run_opt_pass(expr, opt_pass):
    """Exectue a relay pass."""
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def _to_shape(shape):
    """convert shape into tuple."""
    return tuple(int(sh) for sh in shape)


def _pack_batch_channel(data, dshape, bfactor, cfactor):
    """Pack the data channel dimension."""
    assert int(dshape[0]) % bfactor == 0
    assert int(dshape[1]) % cfactor == 0
    data = op.reshape(
        data,
        newshape=(
            int(dshape[0]) // bfactor,
            bfactor,
            int(dshape[1]) // cfactor,
            cfactor,
            int(dshape[2]),
            int(dshape[3]),
        ),
    )
    data = op.transpose(data, axes=(0, 2, 4, 5, 1, 3))
    return data


def _unpack_batch_channel(data, old_shape, unpack_transpose=False):
    """Unpack the data channel dimension."""
    if unpack_transpose:
        data = op.transpose(data, axes=(0, 4, 1, 5, 2, 3))
    data = op.reshape(data, newshape=old_shape)
    return data


def _channel_const_match(channel_length, cfactor_out):
    """Round the channel const variant if the value not divisible by cfactor_out"""
    diff = int(channel_length) % cfactor_out
    if diff != 0:
        diff = cfactor_out - diff
        channel_length = channel_length + diff

    return diff, channel_length


def _const_shape_match(data, dshape, cfactor_out):
    """Pad the constant if the shape[0] not divisible by cfactor_out."""
    assert len(dshape) == 3
    pad_width = int(dshape[0]) % cfactor_out
    if pad_width != 0:
        pad_width = cfactor_out - pad_width
        data = op.nn.pad(data, [[0, pad_width], [0, 0], [0, 0]])
        dshape = tuple([dshape[0] + pad_width, dshape[1], dshape[2]])
    return data, dshape


def _weight_shape_match(data, dshape, channels, cfactor_out, transpose=False):
    """Pad the weight if the shape[0] not divisible by cfactor_out."""
    assert len(dshape) == 4
    pad_width = int(dshape[0]) % cfactor_out
    channels_pad = int(channels) % cfactor_out
    if pad_width != 0:
        pad_width = cfactor_out - pad_width
        data = op.nn.pad(data, [[0, pad_width], [0, 0], [0, 0], [0, 0]])
        dshape = tuple([dshape[0] + pad_width, dshape[1], dshape[2], dshape[3]])

    if channels_pad != 0:
        channels = channels + (cfactor_out - channels_pad)

    return data, dshape, channels


def _weight_shape_match_transpose(data, dshape, channels, cfactor_out):
    """Pad the weight if the shape[1] not divisible by cfactor_out."""
    assert len(dshape) == 4
    pad_width = int(dshape[1]) % cfactor_out
    channels_pad = int(channels) % cfactor_out
    if pad_width != 0:
        pad_width = cfactor_out - pad_width
        data = op.nn.pad(data, [[0, 0], [0, pad_width], [0, 0], [0, 0]])
        dshape = tuple(dshape[0], [dshape[1] + pad_width, dshape[2], dshape[3]])

    if channels_pad != 0:
        channels = channels + (cfactor_out - channels_pad)

    return data, dshape, channels


def _pack_weight(data, dshape, cfactor):
    """Pack the weight into packed format."""
    assert len(dshape) == 4
    assert int(dshape[0]) % cfactor == 0
    assert int(dshape[1]) % cfactor == 0
    data = op.reshape(
        data,
        newshape=(
            int(dshape[0]) // cfactor,
            cfactor,
            int(dshape[1]) // cfactor,
            cfactor,
            int(dshape[2]),
            int(dshape[3]),
        ),
    )
    data = op.transpose(data, axes=(0, 2, 4, 5, 1, 3))
    return data


def _pack_weight_conv2d_transpose(data, dshape, cfactor):
    """Pack the weight into packed format."""
    dshape = _to_shape(dshape)
    assert len(dshape) == 4
    assert dshape[0] % cfactor == 0
    assert dshape[1] % cfactor == 0
    data = op.reshape(
        data,
        newshape=(
            dshape[0] // cfactor,
            cfactor,
            dshape[1] // cfactor,
            cfactor,
            dshape[2],
            dshape[3],
        ),
    )
    data = op.transpose(data, axes=(2, 0, 4, 5, 3, 1))
    return data


def _pack_const(data, dshape, dtype, bfactor, cfactor):
    """Pack a constant parameter."""
    dshape = _to_shape(dshape)
    assert len(dshape) == 3
    assert dshape[0] % cfactor == 0
    data = op.reshape(data, newshape=(dshape[0] // cfactor, cfactor, dshape[1], dshape[2], 1))
    data = op.transpose(data, axes=(0, 2, 3, 4, 1))

    # broadcast batch dimension to bfactor
    data = op.broadcast_to(
        data, shape=(dshape[0] // cfactor, dshape[1], dshape[2], bfactor, cfactor)
    )
    return data


def _get_tensor_shape(node):
    """Get node shape."""
    if isinstance(node.checked_type, relay.ty.TensorType):
        return _to_shape(node.checked_type.shape)
    return []


def _get_tensor_type(node):
    """Get node type."""
    if isinstance(node.checked_type, relay.ty.TensorType):
        return node.checked_type.dtype
    return "float32"


def _operator_idx_inc(expr, count_meta, operator_current_idx):
    """Increase operator index"""
    if isinstance(expr, relay.expr.Constant):
        operator_current_idx = operator_current_idx + 1 if count_meta else operator_current_idx
    else:
        operator_current_idx = operator_current_idx + 1
    return operator_current_idx


class ExprDeviceAnnot(ExprMutator):
    """Visitor to perform graph annotation on an AST.

    Parameters
    ----------
    start: int
        the start location to mark run on vta (inclusive)
    end: int
        the end location to mark run on vta (exclusive)

    Returns
    ---------
    None
    """

    def __init__(self, start=-1, end=-1):
        self.ext_dev = tvm.device("ext_dev")
        self.cpu_dev = tvm.device("cpu")
        self.cast = op.op.get("cast")
        self.counter = -1
        self.start = start
        self.end = end
        super().__init__()

    def visit_call(self, call):
        """Visit the children."""
        # First visit the children.
        args = [self.visit(arg) for arg in call.args]

        self.counter += 1
        if self.counter == self.start:
            ret = relay.Call(call.op, args, call.attrs)
            ret = relay.annotation.on_device(ret, self.ext_dev)
            return ret

        if self.counter == self.end:
            ret = relay.Call(call.op, args, call.attrs)
            ret = relay.annotation.on_device(ret, self.cpu_dev)
            return ret

        if self.counter > self.start and self.counter < self.end:
            ret = relay.Call(call.op, args, call.attrs)

            # skip the float op, i.e., float->int cast
            if self.is_float_op(call):
                return ret

            return relay.annotation.on_device(ret, self.ext_dev)

        return relay.Call(self.visit(call.op), args, call.attrs)

    def is_float_op(self, call):
        """check if this op belongs to a float op
        in general, float op's odtype is float;
        a special case is float->int cast, which follow this op sequence:
        multiply(float) -> round(float) -> clip(float) -> cast(int);
        """
        args = call.args
        odtype = _get_tensor_type(call)

        if odtype == "float32":
            return True

        if call.op == self.cast:
            idtype = _get_tensor_type(args[0])
            if idtype == "float32":
                return True

        return False


class ExprLocator(ExprMutator):
    """Visitor to locate op on an AST."""

    def __init__(self):
        self.counter = -1
        self.op2nodes = {}
        super().__init__()

    def visit_call(self, call):
        """Visit the children."""
        # First visit the children.
        args = [self.visit(arg) for arg in call.args]

        odtype = _get_tensor_type(call)
        self.counter += 1
        if (call.op, odtype) in self.op2nodes:
            self.op2nodes[(call.op, odtype)].append(self.counter)
        else:
            self.op2nodes[(call.op, odtype)] = [self.counter]

        return relay.Call(self.visit(call.op), args, call.attrs)


class ExprPack(ExprMutator):
    """Visitor to perform graph packing on an AST."""

    def __init__(self, bfactor, cfactor, weight_bits):
        self.bfactor = bfactor
        self.cfactor = cfactor
        self.weight_bits = weight_bits
        self.start_pack = False
        # Cache Operator the algorithm matches against.
        self.bitpack_start = op.op.get("annotation.bitpack_start")
        self.bitpack_end = op.op.get("annotation.bitpack_end")
        self.conv2d = op.op.get("nn.conv2d")
        self.conv2d_transpose = op.op.get("nn.conv2d_transpose")
        self.add = op.op.get("add")
        self.multiply = op.op.get("multiply")
        self.bias_add = op.op.get("nn.bias_add")
        self.pad = op.op.get("nn.pad")
        self.upsampling = op.op.get("nn.upsampling")
        self.reshape = op.op.get("reshape")
        self.number_of_conv2d = 0
        self.unpack_transpose = True
        super().__init__()

    def visit_call(self, call):
        """Visit the children."""
        # First visit the children.
        oshape = _get_tensor_shape(call)
        odtype = _get_tensor_type(call)
        input_types = [arg.checked_type for arg in call.args]
        args = [self.visit(arg) for arg in call.args]

        # Start and stop cases.
        if call.op == self.bitpack_start:
            assert not self.start_pack
            self.start_pack = True
            return _pack_batch_channel(args[0], oshape, self.bfactor, self.cfactor)
        if call.op == self.bitpack_end:
            if self.start_pack:
                self.start_pack = False
                data = args[0]
                data_shape = _get_tensor_shape(call.args[0])
                return _unpack_batch_channel(data, data_shape, self.unpack_transpose)
        if self.start_pack:
            # Operator cases
            if call.op == self.conv2d and odtype == "int32":
                self.number_of_conv2d += 1
                assert 8 % self.weight_bits == 0
                w_lanes = 8 // self.weight_bits
                data_layout = "NCHW%dn%dc" % (self.bfactor, self.cfactor)
                kernel_layout = "OIHW%do%di" % (self.cfactor, self.cfactor)
                data, weight = args
                data_shape = _to_shape(input_types[0].shape)
                kernel_shape = _to_shape(input_types[1].shape)
                channels = call.attrs.channels
                weight, kernel_shape, channels = _weight_shape_match(
                    weight, kernel_shape, channels, self.cfactor
                )
                kernel = _pack_weight(weight, kernel_shape, self.cfactor)
                # insert bit packing when necessary
                if w_lanes != 1:
                    assert 8 % w_lanes == 0
                    kernel = op.bitpack(kernel, lanes=w_lanes)

                conv2d = op.nn.conv2d(
                    data,
                    kernel,
                    strides=call.attrs.strides,
                    padding=call.attrs.padding,
                    dilation=call.attrs.dilation,
                    groups=call.attrs.groups,
                    channels=channels,
                    kernel_size=call.attrs.kernel_size,
                    data_layout=data_layout,
                    kernel_layout=kernel_layout,
                    out_dtype=call.attrs.out_dtype,
                )
                return conv2d

            if call.op == self.conv2d_transpose and odtype == "int32":
                self.number_of_conv2d += 1
                assert 8 % self.weight_bits == 0
                w_lanes = 8 // self.weight_bits
                if self.start_pack:
                    data_layout = "NCHW%dn%dc" % (self.bfactor, self.cfactor)
                    kernel_layout = "IOHW%di%do" % (self.cfactor, self.cfactor)
                    data, weight = args
                    data_shape = _to_shape(input_types[0].shape)
                    kernel_shape = _to_shape(input_types[1].shape)
                    channels = call.attrs.channels
                    weight, kernel_shape, channels = _weight_shape_match_transpose(
                        weight, kernel_shape, channels, self.cfactor
                    )
                    kernel = _pack_weight_conv2d_transpose(weight, kernel_shape, self.cfactor)
                    conv2d = op.nn.conv2d_transpose(
                        data,
                        kernel,
                        strides=call.attrs.strides,
                        padding=call.attrs.padding,
                        dilation=call.attrs.dilation,
                        groups=call.attrs.groups,
                        channels=call.attrs.channels,
                        kernel_size=call.attrs.kernel_size,
                        data_layout=data_layout,
                        kernel_layout=kernel_layout,
                        output_padding=call.attrs.output_padding,
                        out_dtype=call.attrs.out_dtype,
                    )
                return conv2d
            if call.op == self.add and tuple(input_types[0].shape) == tuple(input_types[1].shape):
                pass
            elif call.op == self.add and len(input_types[1].shape) == 3:
                data, const = args
                const, input_shape = _const_shape_match(const, input_types[1].shape, self.cfactor)
                const = _pack_const(
                    const, _to_shape(input_shape), input_types[1].dtype, self.bfactor, self.cfactor
                )
                return relay.Call(self.add, [data, const])
            elif call.op == self.multiply and tuple(input_types[0].shape) == tuple(
                input_types[1].shape
            ):
                pass
            elif call.op == self.multiply and len(input_types[1].shape) == 3:
                data, const = args
                const = _pack_const(
                    const,
                    _to_shape(input_types[1].shape),
                    input_types[1].dtype,
                    self.bfactor,
                    self.cfactor,
                )
                return relay.Call(self.multiply, [data, const])
            elif self.start_pack and call.op == self.bias_add:
                data, bias = args
                bias = _pack_const(
                    bias,
                    _to_shape(input_types[1].shape),
                    input_types[1].dtype,
                    self.bfactor,
                    self.cfactor,
                )
                return relay.Call(self.add, [data, bias])
            elif (
                self.start_pack and call.op == op.op.get("cast") and input_types[0].dtype == "int32"
            ):
                cast = relay.Call(op.op.get("cast"), [args[0]], call.attrs)
                return cast
            elif call.op == self.pad:
                pad_width = call.attrs.pad_width
                if len(pad_width) == 6:
                    pass
                elif len(pad_width) == 4:
                    (data, pad_value) = args
                    new_pad_width = []
                    new_pad_width.extend(pad_width)
                    for _ in range(2):
                        new_pad_width.append([0, 0])
                    return op.nn.pad(data, pad_value=pad_value, pad_width=new_pad_width)
            elif call.op == self.upsampling:
                (data,) = args
                scale_h = call.attrs.scale_h
                scale_w = call.attrs.scale_w
                data_layout = "NCHW%dn%dc" % (self.bfactor, self.cfactor)
                method = call.attrs.method
                align_corners = call.attrs.align_corners
                return op.nn.upsampling(data, scale_h, scale_w, data_layout, method, align_corners)
            elif call.op == self.reshape and len(input_types[0].shape) == 4:
                (data,) = args
                self.unpack_transpose = False
                data = op.transpose(data, axes=(0, 4, 1, 5, 2, 3))
                new_shape = [int(x) for x in input_types[0].shape]
                # Check if the reshape match with such shape after pad
                pad, new_shape[1] = _channel_const_match(new_shape[1], self.cfactor)
                data = op.reshape(data, new_shape)
                # remove pad data
                if pad != 0:
                    new_pad_width = [[0, 0], [0, -pad], [0, 0], [0, 0]]
                    data = op.nn.pad(data, pad_width=new_pad_width)
                return data

        return relay.Call(self.visit(call.op), args, call.attrs)


class BT(Exception):
    pass


def get_subgraph(expr, start_name, stop_name, start_name_idx, stop_name_idx, count_meta):
    """We assume stop_name only appears once for simplicity.
    This constraint will be lifted in the future.
    bitpack_start and bitpack_end are both inclusive.
    """
    bitpack_start = op.op.get("annotation.bitpack_start")
    bitpack_end = op.op.get("annotation.bitpack_end")
    anf = run_opt_pass(expr, transform.ToANormalForm())
    operator_current_idx = 0

    def _recursion(anf, start_found, stop_found, operator_current_idx):
        """Helper to obtain the subgraph."""
        if isinstance(anf, relay.Function):
            return relay.Function(
                anf.params,
                _recursion(anf.body, start_found, stop_found, operator_current_idx),
                anf.ret_type,
                anf.type_params,
                anf.attrs,
            )
        if isinstance(anf, relay.expr.Let):
            value = anf.value
            if isinstance(value, relay.expr.Call):
                if isinstance(value.op, tvm.ir.Op):
                    if value.op.name == start_name and not start_found:
                        if operator_current_idx == start_name_idx or start_name_idx is None:
                            value = relay.expr.Call(bitpack_start, [value])
                            start_found = True
                    elif value.op.name == stop_name:
                        if operator_current_idx == stop_name_idx or stop_name_idx is None:
                            raise BT()

            operator_current_idx = _operator_idx_inc(value, count_meta, operator_current_idx)

            try:
                return relay.expr.Let(
                    anf.var,
                    value,
                    _recursion(anf.body, start_found, stop_found, operator_current_idx),
                )
            except BT:
                assert start_found
                assert not stop_found
                stop_found = True
                value = relay.expr.Call(bitpack_end, [value])
                # todo: check anf.body has no more stop_name beside that one
                return relay.expr.Let(anf.var, value, anf.body)
        else:
            assert start_found
            assert stop_found
            return anf

    annotated = _recursion(anf, False, False, operator_current_idx)
    return run_opt_pass(annotated, transform.ToGraphNormalForm())


def graph_pack(
    expr,
    bfactor,
    cfactor,
    weight_bits,
    start_name="nn.max_pool2d",
    stop_name="nn.global_avg_pool2d",
    start_name_idx=None,
    stop_name_idx=None,
    count_meta=False,
    device_annot=False,
    annot_start_name="nn.conv2d",
    annot_end_name="annotation.stop_fusion",
):
    """Pack the graph into batch&channel packed format.

    Parameters
    ----------
    expr : relay.Expr
       The input program.

    bfactor : int
       The packing factor in batch

    cfactor : int
       The packing factor in channel

    weight_bits: int
        The bit-width of the weights.

    start_name: str, optional
       Start packing from certain known node when start_name_idx is None.

    stop_name: str, optional
       Stop packing from certain known node when stop_name_idx is None.

    start_name_idx: int, optional
        When start_name_idx not None, start packing only when node name equal start_name
        and node idx equals start_name_idx.

    stop_name_idx: int, optional
        When stop_name_idx not None, stop packing only when node name equal stop_name
        and node index equals stop_name_idx.

    count_meta:boolean, optional
        When count_meta is False, the operator increase logic would not count the meta that have
        the type 'relay.expr.Constant', start_name_idx and stop_name_idx follow the index from
        'expr.astext(show_meta_data=False)'. When count_meta is True, the operator increase
        logic would count the meta.

    device_annot: boolean, optional
        if we want to annoate the device_type

    annot_start_name: str, optional
        device annotation start node, from which we mark the nodes as `ext_dev`

    annot_end_name: str, optional
        device annotation end node, after which we mark the nodes as 'cpu'

    Returns
    -------
    expr : Expr
        The transformed expression.
    """
    assert isinstance(expr, relay.Function)
    assert (
        (start_name != stop_name)
        or (start_name_idx is None != stop_name_idx is None)
        or (not (start_name_idx is None and stop_name_idx is None))
        or (start_name_idx < stop_name_idx)
    )
    expr = get_subgraph(expr, start_name, stop_name, start_name_idx, stop_name_idx, count_meta)
    expr = run_opt_pass(expr, transform.InferType())
    packer = ExprPack(bfactor, cfactor, weight_bits)
    expr = packer.visit(expr)
    assert not packer.start_pack
    expr = run_opt_pass(expr, transform.InferType())

    if device_annot:
        expr_locator = ExprLocator()
        expr_locator.visit(expr)

        annot_start = op.op.get(annot_start_name)
        start = expr_locator.op2nodes[(annot_start, "int32")][0]

        annot_end = op.op.get(annot_end_name)
        # we mark the next op to the last stop_fusion on cpu device
        end = expr_locator.op2nodes[(annot_end, "int8")][-1] + 1

        device_annot = ExprDeviceAnnot(start=start, end=end)
        expr = device_annot.visit(expr)
        return run_opt_pass(expr, transform.InferType())

    return expr
