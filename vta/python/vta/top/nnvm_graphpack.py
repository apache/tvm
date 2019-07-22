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

"""An NNVM implementation of graph packing."""

import nnvm
from nnvm.compiler import graph_attr, graph_util

def _pack_batch_channel(data, dshape, bfactor, cfactor):
    """Pack the data channel dimension.
    """
    assert dshape[0] % bfactor == 0
    assert dshape[1] % cfactor == 0
    data = nnvm.sym.reshape(data,
                            shape=(dshape[0] // bfactor, bfactor,
                                   dshape[1] // cfactor, cfactor,
                                   dshape[2], dshape[3]))
    data = nnvm.sym.transpose(
        data, axes=(0, 2, 4, 5, 1, 3))
    return data


def _unpack_batch_channel(data, old_shape):
    """Unpack the data channel dimension.
    """
    data = nnvm.sym.transpose(data, axes=(0, 4, 1, 5, 2, 3))
    data = nnvm.sym.reshape(data, shape=old_shape)
    return data


def _pack_weight(data, dshape, cfactor):
    """Pack the weight into packed format.
    """
    assert len(dshape) == 4
    assert dshape[0] % cfactor == 0
    assert dshape[1] % cfactor == 0
    data = nnvm.sym.reshape(data,
                            shape=(dshape[0] // cfactor, cfactor,
                                   dshape[1] // cfactor, cfactor,
                                   dshape[2], dshape[3]))
    data = nnvm.sym.transpose(
        data, axes=(0, 2, 4, 5, 1, 3))
    return data


def _pack_weight_conv2d_transpose(data, dshape, cfactor):
    """Pack the weight into packed format.
    """
    assert len(dshape) == 4
    assert dshape[0] % cfactor == 0
    assert dshape[1] % cfactor == 0
    data = nnvm.sym.reshape(data,
                            shape=(dshape[0] // cfactor, cfactor,
                                   dshape[1] // cfactor, cfactor,
                                   dshape[2], dshape[3]))
    data = nnvm.sym.transpose(
        data, axes=(2, 0, 4, 5, 3, 1))
    return data


def _pack_bias(data, dshape, bfactor, cfactor):
    """Pack the bias parameter.
    """
    assert len(dshape) == 3
    assert dshape[0] % cfactor == 0
    data = nnvm.sym.reshape(data,
                            shape=(dshape[0] // cfactor,
                                   cfactor, dshape[1],
                                   dshape[2], 1))
    data = nnvm.sym.transpose(
        data, axes=(0, 2, 3, 4, 1))
    # broadcast batch dimension to bfactor
    data = nnvm.sym.broadcast_to(
        data,
        shape=(dshape[0] // cfactor, dshape[1], dshape[2], bfactor, cfactor))
    return data


def _get_shape(sym, shape_dict):
    """Get the shape of a node.
    """
    return graph_util.infer_shape(
        nnvm.graph.create(sym), **shape_dict)[1][0]


def nnvm_graph_pack(graph,
                    shape_dict,
                    bfactor,
                    cfactor,
                    weight_bits,
                    start_name="max_pool2d0",
                    stop_name="global_avg_pool2d0"):
    """Pack the graph into batch&channel packed format.

    Parameters
    ----------
    graph : Graph
       The input graph.

    shape_dict : dict of str to shape
       The input shape.

    bfactor : int
       The packing factor in batch

    cfactor : int
       The packing factor in channel

    start_name: str, optional
       Start packing from certain known node.

    start_name: str, optional
       Stop packing from certain known node.

    Returns
    -------
    graph : Graph
        The transformed graph.
    """
    graph = graph_attr.set_shape_inputs(graph, shape_dict)
    graph = graph.apply("InferShape")
    shape = graph.json_attr("shape")
    gidx = graph.index
    node_map = {}
    dset = set()
    start_pack = False

    for nid, node in enumerate(gidx.nodes):
        children = [node_map[e[0]] for e in node["inputs"]]
        ishape = [shape[gidx.entry_id(e)] for e in node["inputs"]]
        oshape = shape[gidx.entry_id(nid, 0)]
        attrs = node.get("attrs", {})
        node_name = node["name"]
        op_name = node["op"]
        get_clone = lambda c, o_n, n_n, a: getattr(nnvm.symbol, o_n)(
            *c, name=n_n, **a)
        if op_name == "null":
            new_node = nnvm.symbol.Variable(node_name)
            if start_name and node_name == start_name:
                start_pack = True
                new_node = _pack_batch_channel(new_node, oshape, bfactor, cfactor)
            if start_pack and "_begin_state_" in node_name: # RNN -> CNN, pack
                new_node = _pack_batch_channel(new_node, oshape, bfactor, cfactor)
        elif node_name == start_name:
            assert not start_pack
            start_pack = True
            new_node = get_clone(children, op_name, node_name, attrs)
            new_node = _pack_batch_channel(new_node, oshape, bfactor, cfactor)
        elif node_name == stop_name:
            if start_pack:
                start_pack = False
                children[0] = _unpack_batch_channel(children[0], ishape[0])
                new_node = getattr(nnvm.symbol, op_name)(
                    *children, name=node_name, **attrs)
            else:
                new_node = get_clone(children, op_name, node_name, attrs)
        elif op_name == "conv2d" and attrs.get("out_dtype", None) == "int32":
            assert 8 % weight_bits == 0
            w_lanes = 8 // weight_bits
            if start_pack:
                attrs["layout"] = "NCHW%dn%dc" % (bfactor, cfactor)
                attrs["kernel_layout"] = "OIHW%do%di%dp" % (cfactor, cfactor, w_lanes)
                data, weight = children
                weight = _pack_weight(weight, ishape[1], cfactor)
                # insert bit packing when necessary
                if w_lanes != 1:
                    assert 8 % w_lanes == 0
                    weight = nnvm.sym.bitpack(weight, lanes=w_lanes)
                new_node = nnvm.sym.conv2d(
                    data, weight, name=node_name, **attrs)
            else:
                new_node = get_clone(children, op_name, node_name, attrs)
        elif op_name == "conv2d_transpose" and attrs.get("out_dtype", None) == "int32":
            assert 8 % weight_bits == 0
            w_lanes = 8 // weight_bits
            if start_pack:
                attrs["layout"] = "NCHW%dn%dc" % (bfactor, cfactor)
                attrs["kernel_layout"] = "IOHW%di%do%dp" % (cfactor, cfactor, w_lanes)
                data, weight = children
                weight = _pack_weight_conv2d_transpose(weight, ishape[1], cfactor)
                new_node = nnvm.sym.conv2d_transpose(
                    data, weight, name=node_name, **attrs)
            else:
                new_node = get_clone(children, op_name, node_name, attrs)
        elif op_name.startswith("broadcast_") and tuple(ishape[0]) == tuple(ishape[1]):
            new_node = get_clone(children, op_name, node_name, attrs)
        elif op_name.startswith("broadcast") and len(ishape[1]) == 3:
            if start_pack:
                children[1] = _pack_bias(children[1], ishape[1], bfactor, cfactor)
                new_node = getattr(nnvm.symbol, op_name)(
                    *children, name=node_name, **attrs)
            else:
                new_node = get_clone(children, op_name, node_name, attrs)
        elif op_name.startswith("elementwise_add"):
            new_node = get_clone(children, op_name, node_name, attrs)
        else:
            new_node = get_clone(children, op_name, node_name, attrs)
            dset.add(op_name)
        node_map[nid] = new_node

    assert len(graph.index.output_entries) == 1
    ret = node_map[graph.index.output_entries[0][0]]
    if start_pack:
        oshape = shape[graph.index.output_entries[0][0]]
        ret = _unpack_batch_channel(ret, oshape)
    graph = nnvm.graph.create(ret)
    graph = graph_attr.set_shape_inputs(graph, shape_dict)
    graph = graph.apply("InferShape")
    return graph
