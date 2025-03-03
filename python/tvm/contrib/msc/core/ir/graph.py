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
"""tvm.contrib.msc.core.ir.graph"""

from typing import Dict, Tuple, List, Optional, Union, Iterable, Any
import numpy as np

import tvm
from tvm.runtime import Object
from tvm.contrib.msc.core import _ffi_api
from tvm.contrib.msc.core import utils as msc_utils


@tvm._ffi.register_object("msc.core.MSCTensor")
class MSCTensor(Object):
    """Tensor in MSCGraph

    Parameters
    ----------
    name: string
        The name of the tensor.
    dtype: string or np.dtype or DataType
        The data type the tensor.
    layout: string
        The layout of the tensor.
    shape: list<int>
        The shape of the tensor.
    alias: string
        The alias of the tensor.
    prims: list<str>
        The prims of the tensor.
    """

    def __init__(
        self,
        name: str,
        dtype: Union[str, np.dtype, tvm.DataType],
        layout: str,
        shape: List[int],
        alias: Optional[str] = None,
        prims: List[str] = None,
    ):
        if not isinstance(dtype, tvm.DataType):
            dtype = tvm.DataType(dtype)
        self.__init_handle_by_constructor__(
            _ffi_api.MSCTensor, name, dtype, layout, shape, alias or "", prims or []
        )

    def get_shape(self, with_prims: bool = False) -> List[Union[int, str]]:
        """Get shape of the tensor

        Parameters
        -------
        with_prims: bool
            Whether get shape with prims.

        Returns
        -------
        shape: list<str|int>
            The shape of tensor.
        """

        if not self.prims or not with_prims:
            return [int(i) for i in self.shape]
        return [int(p) if p.isdigit() else p for p in self.prims]

    def get_size(self) -> int:
        return int(_ffi_api.MSCTensorGetSize(self))

    def dim_at(self, axis: Union[int, str]) -> int:
        if isinstance(axis, int):
            return int(self.shape[axis])
        return int(_ffi_api.MSCTensorDimAt(self, axis))

    def layout_of(self, axis: str) -> int:
        return self.layout.index_of(axis)

    def set_alias(self, alias: str):
        """Set alis for the tensor

        Parameters
        -------
        alias: str
            The alias.
        """

        _ffi_api.MSCTensorSetAlias(self, alias)

    def equal(self, other: Object) -> bool:
        """A fast method to check if two nodes are same.

        Parameters
        -------
        other: MSCTensor
            The tensor to be compared.

        Returns
        -------
        equal: bool
            Whether two tensors are the same.
        """

        if not isinstance(other, MSCTensor):
            return False
        if self.get_shape(True) != other.get_shape(True):
            return False
        if self.dtype != other.dtype:
            return False
        return True

    def to_json(self) -> str:
        """Dump the tensor to json.

        Returns
        -------
        tensor_json: string
            The tensor in json format.
        """

        return _ffi_api.MSCTensorToJson(self)

    def inspect(self) -> dict:
        """Extract important info of the tensor.

        Returns
        -------
        tensor_des: dict
            The tensor description in json format.
        """

        tensor_des = {"name": self.alias, "shape": self.get_shape(True), "dtype": self.dtype_name}
        tensor_des["layout"] = self.layout.name if self.layout else ""
        return tensor_des

    @classmethod
    def from_json(cls, json_str: str, **options) -> object:
        """Load the tensor from json.

        Parameters
        ----------
        json_str: string
            The file_path or json string.
        options: dict
            The items to be changed.

        Returns
        -------
        tensor: MSCTensor
            The tensor.
        """

        dict_obj = msc_utils.load_dict(json_str)
        dict_obj.update(options)
        return _ffi_api.MSCTensorFromJson(msc_utils.dump_dict(dict_obj))

    def clone(self, **options) -> object:
        """Clone the tensor.

        Parameters
        ----------
        json_str: string
            The file_path or json string.
        options: dict
            The items to be changed.

        Returns
        -------
        new_tensor: MSCTensor
            The cloned tensor.
        """

        return MSCTensor.from_json(self.to_json(), **options)

    @property
    def dtype_name(self) -> str:
        return _ffi_api.MSCTensorDTypeName(self)

    @property
    def ndim(self) -> int:
        return len(self.shape)


class BaseJoint(Object):
    """Base class of all MSC Nodes."""


@tvm._ffi.register_object("msc.core.MSCJoint")
class MSCJoint(BaseJoint):
    """Node in MSCGraph

    Parameters
    ----------
    index: int
        The index of the node.
    name: string
        The name of the node.
    shared_ref: string
        The share reference of the node.
    optype: string
        The optype of the node.
    attrs: dict<string, string>
        The attributes of the node.
    inputs: list<tuple<MSCJoint, int>>
        The inputs of the node in format <parent,out_idx>.
    outputs: list<MSCTensor>
        The outputs of the node.
    weights: dict<string, MSCTensor>
        The weights of the node.
    """

    def __init__(
        self,
        index: int,
        name: str,
        shared_ref: str,
        optype: str,
        attrs: Dict[str, str],
        inputs: List[Tuple[BaseJoint, int]],
        outputs: List[MSCTensor],
        weights: Dict[str, MSCTensor],
    ):
        parents = [i[0] for i in inputs]
        out_indices = [i[1] for i in inputs]
        self.__init_handle_by_constructor__(
            _ffi_api.MSCJoint,
            index,
            name,
            shared_ref,
            optype,
            attrs,
            parents,
            out_indices,
            outputs,
            weights,
        )

    def input_at(self, idx: int) -> MSCTensor:
        """Get input at idx.

        Parameters
        ----------
        idx: int
            The index of input.

        Returns
        -------
        input: MSCTensor
            The input Tensor.
        """

        return _ffi_api.MSCJointInputAt(self, idx)

    def output_at(self, idx: int) -> MSCTensor:
        """Get output at idx.

        Parameters
        ----------
        idx: int
            The index of output.

        Returns
        -------
        output: MSCTensor
            The output Tensor.
        """

        return _ffi_api.MSCJointOutputAt(self, idx)

    def weight_at(self, wtype: str) -> MSCTensor:
        """Get weight from reference.

        Parameters
        ----------
        wtype: str
            The type of weight.

        Returns
        -------
        weight: MSCTensor
            The weight Tensor.
        """

        return _ffi_api.MSCJointWeightAt(self, wtype)

    def weight_type(self, name: str) -> str:
        """Get the weight type of weight

        Parameters
        ----------
        name: str
            The name of weight.

        Returns
        -------
        wtype: str
            The type of weight.
        """

        for w_type, weight in self.get_weights().items():
            if weight.name == name:
                return w_type
        raise Exception("Can not find weight type for " + name)

    def get_inputs(self) -> List[MSCTensor]:
        """Get all the inputs.

        Returns
        -------
        inputs: list<MSCJoint>
            The input Tensors.
        """

        return _ffi_api.MSCJointGetInputs(self)

    def get_outputs(self) -> List[MSCTensor]:
        """Get all the outputs.

        Returns
        -------
        outputs: list<MSCJoint>
            The output Tensors.
        """

        return _ffi_api.MSCJointGetOutputs(self)

    def get_weights(self) -> Dict[str, MSCTensor]:
        """Get all the weights.

        Returns
        -------
        weights: dict<str, MSCJoint>
            The weight Tensors.
        """

        src_weights = _ffi_api.MSCJointGetWeights(self)
        return {wtype: src_weights[wtype] for wtype in src_weights}

    def get_attrs(self) -> Dict[str, str]:
        """Get all the attributes from node

        Returns
        -------
        attributes: dict<str, str>
            The attributes of node.
        """

        return _ffi_api.MSCJointGetAttrs(self)

    def get_attr(self, key: str, default: Optional[Any] = None) -> str:
        """Get the attribute of key from node

        Parameters
        -------
        key: str
            The key of the attribute.
        default: Any
            The default value when key is missing.

        Returns
        -------
        attribute: str
            The attributes of node.
        """

        return self.get_attrs().get(key, default)

    def has_attr(self, key: str) -> bool:
        """Check if key in attributes

        Parameters
        -------
        key: str
            The key of the attribute.

        Returns
        -------
        has_attr: bool
            Whether the key in the attributes.
        """

        return bool(_ffi_api.MSCJointHasAttr(self, key))

    def equal(self, other: BaseJoint) -> bool:
        """A fast method to check if two nodes are same.

        Parameters
        -------
        other: MSCJoint
            The node to be compared.

        Returns
        -------
        equal: bool
            Whether two nodes are the same.
        """

        if not isinstance(other, MSCJoint):
            return False
        if len(self.get_inputs()) != len(other.get_inputs()):
            return False
        if len(self.get_inputs()) != len(other.get_inputs()):
            return False
        for s_i, o_i in zip(self.get_inputs(), other.get_inputs()):
            if not s_i.equal(o_i):
                return False
        for s_o, o_o in zip(self.get_inputs(), other.get_inputs()):
            if not s_o.equal(o_o):
                return False
        return msc_utils.dict_equal(self.get_attrs(), other.get_attrs())


@tvm._ffi.register_object("msc.core.MSCPrim")
class MSCPrim(BaseJoint):
    """Prim in MSCGraph

    Parameters
    ----------
    index: int
        The index of the prim.
    name: string
        The name of the prim.
    optype: string
        The optype of the prim.
    attrs: dict<string, string>
        The attributes of the node.
    parents: list<MSCPrim>
        The parents of the prim.
    """

    def __init__(
        self, index: int, name: str, optype: str, attrs: Dict[str, str], parents: List[BaseJoint]
    ):
        self.__init_handle_by_constructor__(_ffi_api.MSCPrim, index, name, optype, attrs, parents)


@tvm._ffi.register_object("msc.core.WeightJoint")
class WeightJoint(BaseJoint):
    """Node in WeightGraph

    Parameters
    ----------
    index: int
        The index of the node.
    name: string
        The name of the node.
    shared_ref: string
        The share reference of the node.
    optype: string
        The optype of the node.
    wtype: string
        The weight type of the node.
    strategy: string
        The prune strategy of the node.
    weight: MSCTensor
        The weight of the node.
    attrs: dict<string, string>
        The attributes of the node.
    parents: list<WeightJoint>
        The parents of the node.
    friends: list<WeightJoint>
        The friends of the node.
    """

    def __init__(
        self,
        index: int,
        name: str,
        shared_ref: str,
        optype: str,
        wtype: str,
        strategy: str,
        weight: MSCTensor,
        attrs: Dict[str, str],
        parents: List[BaseJoint],
        friends: List[BaseJoint],
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.WeightJoint,
            index,
            name,
            shared_ref,
            optype,
            wtype,
            strategy,
            weight,
            attrs,
            parents,
            friends,
        )

    def set_attr(self, key: str, value: str):
        """Set attribute to node

        Parameters
        -------
        key: str
            The key of the attribute.
        value: str
            The value.
        """

        _ffi_api.WeightJointSetAttr(self, key, value)

    def get_attrs(self) -> Dict[str, str]:
        """Get all the attributes from node

        Returns
        -------
        attributes: dict<str, str>
            The attributes of node.
        """

        return _ffi_api.WeightJointGetAttrs(self)

    def get_attr(self, key: str, default: Optional[Any] = None) -> str:
        """Get the attribute of key from node

        Parameters
        -------
        key: str
            The key of the attribute.
        default: Any
            The default value when key is missing.

        Returns
        -------
        attribute: str
            The attributes of node.
        """

        return self.get_attrs().get(key, default)

    def has_attr(self, key: str) -> bool:
        """Check if key in attributes

        Parameters
        -------
        key: str
            The key of the attribute.

        Returns
        -------
        has_attr: bool
            Whether the key in the attributes.
        """

        return bool(_ffi_api.WeightJointHasAttr(self, key))


class BaseGraph(Object):
    """Base class of all MSC Graphs."""


@tvm._ffi.register_object("msc.core.MSCGraph")
class MSCGraph(BaseGraph):
    """The MSCGraph

    Parameters
    ----------
    name: string
        The name of the graph.
    nodes: list<MSCJoint>
        The nodes of the graph.
    input_names: list<str>
        The input names of the graph.
    output_names: list<str>
        The output names of the graph.
    """

    def __init__(
        self,
        name: str,
        nodes: List[MSCJoint],
        input_names: List[str],
        output_names: List[str],
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.MSCGraph,
            name,
            nodes,
            input_names,
            output_names,
        )

    def has_node(self, name: str) -> bool:
        """Check if node in the graph.

        Parameters
        ----------
        name: string
            The name of the node.

        Returns
        -------
        has_node: bool
            Whether the node is in the graph
        """

        return bool(_ffi_api.MSCGraphHasNode(self, name))

    def find_node(self, name: str) -> MSCJoint:
        """Find node by name.

        Parameters
        ----------
        name: string
            The name of the node.

        Returns
        -------
        node: MSCJoint
            The found node.
        """

        return _ffi_api.MSCGraphFindNode(self, name)

    def find_prim(self, name: str) -> MSCPrim:
        """Find prim by name.

        Parameters
        ----------
        name: string
            The name of the prim.

        Returns
        -------
        prim: MSCPrim
            The found prim.
        """

        return _ffi_api.MSCGraphFindPrim(self, name)

    def has_tensor(self, name: str) -> bool:
        """Check if tensor in the graph.

        Parameters
        ----------
        name: string
            The name of the tensor.

        Returns
        -------
        has_tensor: bool
            Whether the tensor is in the graph
        """

        return bool(_ffi_api.MSCGraphHasTensor(self, name))

    def find_tensor(self, name: str) -> MSCTensor:
        """Find tensor by name.

        Parameters
        ----------
        name: string
            The name of the tensor.

        Returns
        -------
        node: MSCTensor
            The found tensor.
        """

        return _ffi_api.MSCGraphFindTensor(self, name)

    def set_tensor_alias(self, tensor: MSCTensor, alias: str):
        """Set alis for the tensor

        Parameters
        -------
        tensor: MSCTensor
            The tensor.
        alias: str
            The alias.
        """

        _ffi_api.MSCGraphSetTensorAlias(self, tensor, alias)

    def find_producer(self, ref: Union[str, MSCTensor]) -> MSCJoint:
        """Find producer by tensor_name or tensor.

        Parameters
        ----------
        ref: string or MSCTensor
            The name of the tensor or tensor.

        Returns
        -------
        node: MSCJoint
            The found prducer.
        """

        if isinstance(ref, MSCTensor):
            return _ffi_api.MSCGraphFindProducer(self, ref.name)
        return _ffi_api.MSCGraphFindProducer(self, ref)

    def find_consumers(self, ref: Union[str, MSCTensor]) -> List[MSCJoint]:
        """Find consumers by tensor_name or tensor.

        Parameters
        ----------
        ref: string or MSCTensor
            The name of the tensor or tensor.

        Returns
        -------
        node: list<MSCJoint>
            The found consumers.
        """

        if isinstance(ref, MSCTensor):
            return _ffi_api.MSCGraphFindConsumers(self, ref.name)
        return _ffi_api.MSCGraphFindConsumers(self, ref)

    def get_nodes(self) -> Iterable[MSCJoint]:
        """Get all the nodes in the graph.

        Returns
        -------
        nodes: generator<MSCJoint>
            The generator of nodes.
        """

        for n in self.node_names:
            yield self.find_node(n)

    def get_prims(self) -> Iterable[MSCPrim]:
        """Get all the prims in the graph.

        Returns
        -------
        prims: generator<MSCPrim>
            The generator of prims.
        """

        for n in self.prim_names:
            yield self.find_prim(n)

    def get_weights(self) -> Iterable[MSCTensor]:
        """Get all the weights in the graph.

        Returns
        -------
        weights: generator<MSCTensor>
            The generator of weights.
        """

        for node in self.get_nodes():
            for weight in node.get_weights().values():
                yield weight

    def input_at(self, idx: int) -> MSCTensor:
        """Get input at idx.

        Parameters
        ----------
        idx: int
            The index of input.

        Returns
        -------
        input: MSCTensor
            The input Tensor.
        """

        return _ffi_api.MSCGraphInputAt(self, idx)

    def output_at(self, idx: int) -> MSCTensor:
        """Get output at idx.

        Parameters
        ----------
        idx: int
            The index of output.

        Returns
        -------
        output: MSCTensor
            The output Tensor.
        """

        return _ffi_api.MSCGraphOutputAt(self, idx)

    def get_inputs(self) -> List[MSCTensor]:
        """Get all the inputs.

        Returns
        -------
        inputs: list<MSCJoint>
            The input Tensors.
        """

        return _ffi_api.MSCGraphGetInputs(self)

    def get_outputs(self) -> List[MSCTensor]:
        """Get all the outputs.

        Returns
        -------
        outputs: list<MSCJoint>
            The output Tensors.
        """

        return _ffi_api.MSCGraphGetOutputs(self)

    def get_tensors(self) -> List[MSCTensor]:
        """Get all the tensors.

        Returns
        -------
        tensors: list<MSCJoint>
            The Tensors.
        """

        for node in self.get_nodes():
            for t_input in node.get_inputs():
                yield t_input
            for weight in node.get_weights().values():
                yield weight
        for t_output in self.get_outputs():
            yield t_output

    def to_json(self) -> str:
        """Dump the graph to json.

        Returns
        -------
        graph_json: string
            The graph in json format.
        """

        return _ffi_api.MSCGraphToJson(self)

    def inspect(self) -> dict:
        """Extract important info of the graph.

        Returns
        -------
        graph_des: dict
            The graph description in json format.
        """

        graph_des = {
            "inputs": [i.inspect() for i in self.get_inputs()],
            "outputs": [o.inspect() for o in self.get_outputs()],
            "nodes": {"total": 0},
        }
        for node in self.get_nodes():
            graph_des["nodes"].setdefault(node.optype, 0)
            graph_des["nodes"]["total"] += 1
            graph_des["nodes"][node.optype] += 1
        prims = {"total": 0}
        for prim in self.get_prims():
            prims.setdefault(prim.optype, 0)
            prims["total"] += 1
            prims[prim.optype] += 1
        if prims["total"] > 0:
            graph_des["prims"] = prims
        return graph_des

    @classmethod
    def from_json(cls, json_str: str) -> BaseGraph:
        """Load the graph from json.

        Parameters
        ----------
        json_str: string
            The file_path or json string.

        Returns
        -------
        graph: MSCgraph
            The graph.
        """

        dict_obj = msc_utils.load_dict(json_str)
        return _ffi_api.MSCGraphFromJson(msc_utils.dump_dict(dict_obj))

    def clone(self) -> BaseGraph:
        """Clone the graph.

        Returns
        -------
        new_graph: MSCGraph
            The cloned graph.
        """

        return MSCGraph.from_json(self.to_json())

    def equal(self, other: BaseGraph) -> bool:
        """A fast method to check if two graphs are same.

        Parameters
        -------
        other: MSCGraph
            The graph to be compared.

        Returns
        -------
        equal: bool
            Whether two graphs are the same.
        """

        if not isinstance(other, MSCGraph):
            return False
        if len(self.input_names) != len(other.input_names):
            return False
        if len(self.output_names) != len(other.output_names):
            return False
        if len(self.node_names) != len(other.node_names):
            return False
        for s_i, o_i in zip(self.get_inputs(), other.get_inputs()):
            if not s_i.equal(o_i):
                return False
        for s_o, o_o in zip(self.get_outputs(), other.get_outputs()):
            if not s_o.equal(o_o):
                return False
        for s_n, o_n in zip(self.get_nodes(), other.get_nodes()):
            if not s_n.equal(o_n):
                return False
        return True

    def visualize(self, path: Optional[str] = None) -> str:
        """Dump the graph to prototxt format.

        Parameters
        ----------
        path: string
            The file_path to save prototxt.

        Returns
        -------
        graph_proto: string
            The graph in prototxt format.
        """

        graph_proto = _ffi_api.MSCGraphToPrototxt(self)
        if path:
            with open(path, "w") as f:
                f.write(graph_proto)
        return graph_proto


@tvm._ffi.register_object("msc.core.WeightGraph")
class WeightGraph(Object):
    """The WeightGraph

    Parameters
    ----------
    name: string
        The name of the graph.
    nodes: list<WeightJoint>
        The nodes of the graph.
    """

    def __init__(
        self,
        name: str,
        nodes: List[WeightJoint],
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.WeightGraph,
            name,
            nodes,
        )

    def has_node(self, name: str) -> bool:
        """Check if weight node in the graph.

        Parameters
        ----------
        name: string
            The name of the node.

        Returns
        -------
        has_node: bool
            Whether the node is in the graph
        """

        return bool(_ffi_api.WeightGraphHasNode(self, name))

    def find_node(self, name: str) -> WeightJoint:
        """Find weight node by name.

        Parameters
        ----------
        name: string
            The name of the node.

        Returns
        -------
        node: MSCJoint
            The found node.
        """

        return _ffi_api.WeightGraphFindNode(self, name)

    def get_nodes(self) -> Iterable[WeightJoint]:
        """Get all the weight nodes in the graph.

        Returns
        -------
        nodes: generator<WeightJoint>
            The generator of nodes.
        """

        for n in self.node_names:
            yield self.find_node(n)

    def to_json(self) -> str:
        """Dump the graph to json.

        Returns
        -------
        graph_json: string
            The graph in json format.
        """

        return _ffi_api.WeightGraphToJson(self)

    def inspect(self) -> dict:
        """Extract important info of the graph.

        Returns
        -------
        graph_des: dict
            The graph description in json format.
        """

        graph_des = {
            "nodes": {"total": 0},
        }
        for node in self.get_nodes():
            graph_des["nodes"]["total"] += 1
            if node.weight_type not in graph_des["nodes"]:
                graph_des["nodes"][node.weight_type] = 1
            else:
                graph_des["nodes"][node.weight_type] += 1
        return graph_des

    @classmethod
    def from_json(cls, json_str: str) -> BaseGraph:
        """Load the graph from json.

        Parameters
        ----------
        json_str: string
            The file_path or json string.

        Returns
        -------
        graph: WeightGraph
            The graph.
        """

        dict_obj = msc_utils.load_dict(json_str)
        return _ffi_api.WeightGraphFromJson(msc_utils.dump_dict(dict_obj))

    def clone(self) -> BaseGraph:
        """Clone the graph.

        Returns
        -------
        new_graph: MSCGraph
            The cloned graph.
        """

        return MSCGraph.from_json(self.to_json())

    def visualize(self, path: Optional[str] = None) -> str:
        """Dump the graph to prototxt format.

        Parameters
        ----------
        path: string
            The file_path to save prototxt.

        Returns
        -------
        graph_proto: string
            The graph in prototxt format.
        """

        graph_proto = _ffi_api.WeightGraphToPrototxt(self)
        if path:
            with open(path, "w") as f:
                f.write(graph_proto)
        return graph_proto
