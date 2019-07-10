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
# pylint: disable=unused-argument
"""A Relay implementation of graph packing."""

from tvm import relay
from tvm.relay import op, transform
from tvm.relay import ExprMutator

def run_opt_pass(expr, opt_pass):
    """Exectue a relay pass."""
    assert isinstance(opt_pass, transform.Pass)
    mod = relay.Module.from_expr(expr)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body

def _to_shape(shape):
    return tuple(int(sh) for sh in shape)

def _pack_batch_channel(data, dshape, bfactor, cfactor):
    """Pack the data channel dimension.
    """
    assert int(dshape[0]) % bfactor == 0
    assert int(dshape[1]) % cfactor == 0
    data = op.reshape(data,
                      newshape=(int(dshape[0]) // bfactor, bfactor,
                                int(dshape[1]) // cfactor, cfactor,
                                int(dshape[2]), int(dshape[3])))
    data = op.transpose(
        data, axes=(0, 2, 4, 5, 1, 3))
    return data


def _unpack_batch_channel(data, old_shape):
    """Unpack the data channel dimension.
    """
    data = op.transpose(data, axes=(0, 4, 1, 5, 2, 3))
    data = op.reshape(data, newshape=old_shape)
    return data


def _pack_weight(data, dshape, cfactor):
    """Pack the weight into packed format.
    """
    assert len(dshape) == 4
    assert int(dshape[0]) % cfactor == 0
    assert int(dshape[1]) % cfactor == 0
    data = op.reshape(data,
                      newshape=(int(dshape[0]) // cfactor, cfactor,
                                int(dshape[1]) // cfactor, cfactor,
                                int(dshape[2]), int(dshape[3])))
    data = op.transpose(
        data, axes=(0, 2, 4, 5, 1, 3))
    return data


def _pack_weight_conv2d_transpose(data, dshape, cfactor):
    """Pack the weight into packed format.
    """
    dshape = _to_shape(dshape)
    assert len(dshape) == 4
    assert dshape[0] % cfactor == 0
    assert dshape[1] % cfactor == 0
    data = op.reshape(data,
                      newshape=(dshape[0] // cfactor, cfactor,
                                dshape[1] // cfactor, cfactor,
                                dshape[2], dshape[3]))
    data = op.transpose(
        data, axes=(2, 0, 4, 5, 3, 1))
    return data


def _pack_bias(data, dshape, dtype, bfactor, cfactor):
    """Pack the bias parameter.
    """
    dshape = _to_shape(dshape)
    assert len(dshape) == 3
    assert dshape[0] % cfactor == 0
    data = op.reshape(data,
                      newshape=(dshape[0] // cfactor,
                                cfactor, dshape[1],
                                dshape[2], 1))
    data = op.transpose(
        data, axes=(0, 2, 3, 4, 1))

    # broadcast batch dimension to bfactor
    data = op.broadcast_to(
        data,
        shape=(dshape[0] // cfactor, dshape[1], dshape[2], bfactor, cfactor))
    return data


def _get_shape(node):
    """Get the shape of a node.
    """
    return _to_shape(node.checked_type.shape)

class ExprPack(ExprMutator):
    """Visitor to perform graph packing on an AST.
    """
    def __init__(self, bfactor, cfactor, weight_bits):
        self.bfactor = bfactor
        self.cfactor = cfactor
        self.weight_bits = weight_bits
        self.start_pack = False
        # Cache Operator the algorithm matches against.
        self.bitpack_start = op.op.get('annotation.bitpack_start')
        self.bitpack_end = op.op.get('annotation.bitpack_end')
        self.conv2d = op.op.get("nn.conv2d")
        self.conv2d_transpose = op.op.get("nn.conv2d_transpose")
        self.add = op.op.get("add")
        self.bias_add = op.op.get("nn.bias_add")
        self.number_of_conv2d = 0
        super().__init__()

    def visit_call(self, call):
        """ Visit the children. """
        # First visit the children.
        oshape = _get_shape(call)
        odtype = call.checked_type.dtype
        input_types = [arg.checked_type for arg in call.args]
        args = [self.visit(arg) for arg in call.args]

        # Start and stop cases.
        if call.op == self.bitpack_start:
            assert not self.start_pack
            self.start_pack = True
            return _pack_batch_channel(args[0], oshape, self.bfactor, self.cfactor)
        elif call.op == self.bitpack_end:
            if self.start_pack:
                self.start_pack = False
                data = args[0]
                data_shape = _get_shape(call.args[0])
                return _unpack_batch_channel(data, data_shape)
            else:
                pass
        if self.start_pack:
            # Operator cases
            if call.op == self.conv2d and odtype == 'int32':
                self.number_of_conv2d += 1
                assert 8 % self.weight_bits == 0
                w_lanes = 8 // self.weight_bits
                data_layout = "NCHW%dn%dc" % (self.bfactor, self.cfactor)
                kernel_layout = "OIHW%do%di" % (self.cfactor, self.cfactor)
                data, weight = args
                data_shape = _to_shape(input_types[0].shape)
                kernel_shape = _to_shape(input_types[1].shape)
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
                    channels=call.attrs.channels,
                    kernel_size=call.attrs.kernel_size,
                    data_layout=data_layout,
                    kernel_layout=kernel_layout,
                    out_dtype=call.attrs.out_dtype)
                return conv2d
            elif call.op == self.conv2d_transpose and odtype == 'int32':
                self.number_of_conv2d += 1
                assert 8 % self.weight_bits == 0
                w_lanes = 8 // self.weight_bits
                if self.start_pack:
                    data_layout = "NCHW%dn%dc" % (self.bfactor, self.cfactor)
                    kernel_layout = "IOHW%di%do" % (self.cfactor, self.cfactor)
                    data, weight = args
                    data_shape = _to_shape(input_types[0].shape)
                    kernel_shape = _to_shape(input_types[1].shape)
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
                        out_dtype=call.attrs.out_dtype)
                return conv2d
            elif call.op == self.add and tuple(input_types[0].shape) == tuple(input_types[1].shape):
                pass
            elif call.op == self.add and len(input_types[1].shape) == 3:
                data, bias = args
                bias = _pack_bias(bias,
                                  _to_shape(input_types[1].shape),
                                  input_types[1].dtype,
                                  self.bfactor,
                                  self.cfactor)
                return relay.Call(self.add, [data, bias])
            elif self.start_pack and call.op == self.bias_add:
                data, bias = args
                bias = _pack_bias(bias,
                                  _to_shape(input_types[1].shape),
                                  input_types[1].dtype,
                                  self.bfactor,
                                  self.cfactor)
                return relay.Call(self.add, [data, bias])
            elif self.start_pack and call.op == op.op.get('cast') and \
                    input_types[0].dtype == 'int32':
                cast = relay.Call(op.op.get('cast'), [args[0]], call.attrs)
                return relay.Call(op.op.get('copy'), [cast])

        return relay.Call(
            self.visit(call.op),
            args,
            call.attrs)

class BT(Exception):
    pass
def get_subgraph(expr, start_name, stop_name):
    """ We assume stop_name only appears once for simplicity.
        This constraint will be lifted in the future.
        bitpack_start and bitpack_end are both inclusive.
    """
    bitpack_start = op.op.get('annotation.bitpack_start')
    bitpack_end = op.op.get('annotation.bitpack_end')
    anf = run_opt_pass(expr, transform.ToANormalForm())
    def _recursion(anf, start_found, stop_found):
        """ Helper to obtain the subgraph.
        """
        if isinstance(anf, relay.expr.Function):
            return relay.expr.Function(anf.params,
                                       _recursion(anf.body, start_found, stop_found),
                                       anf.ret_type, anf.type_params, anf.attrs)
        elif isinstance(anf, relay.expr.Let):
            value = anf.value
            if isinstance(value, relay.expr.Call):
                if isinstance(value.op, relay.op.Op):
                    if value.op.name == start_name and not start_found:
                        value = relay.expr.Call(bitpack_start, [value])
                        start_found = True
                    elif value.op.name == stop_name:
                        raise BT()
            try:
                return relay.expr.Let(anf.var, value, _recursion(anf.body, start_found, stop_found))
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
    annotated = _recursion(anf, False, False)
    return run_opt_pass(annotated, transform.ToGraphNormalForm())

def graph_pack(expr,
               bfactor,
               cfactor,
               weight_bits,
               start_name="nn.max_pool2d",
               stop_name="nn.global_avg_pool2d"):
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
       Start packing from certain known node.

    stop_name: str, optional
       Stop packing from certain known node.

    Returns
    -------
    expr : Expr
        The transformed expression.
    """
    assert isinstance(expr, relay.Function)
    expr = get_subgraph(expr, start_name, stop_name)
    expr = run_opt_pass(expr, transform.InferType())
    packer = ExprPack(
        bfactor, cfactor,
        weight_bits)
    expr = packer.visit(expr)
    assert not packer.start_pack
    return run_opt_pass(expr, transform.InferType())
