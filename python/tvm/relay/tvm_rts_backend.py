"""A compiler from Relay programs to TVM's graph runtime.
"""
import json
from typing import Dict, Any, List, Tuple

import attr

from relay.frontend import get_env
from . import ir
from .tyck import get_checked_type
from .opt import AbstractExprVisitor, compile_ops_to_module
from ._make import Operator_is_generic


@attr.s(auto_attribs=True)
class NodeRef:
    ident: int
    index: int = 0
    version: int = 0

    def to_json(self) -> Any:
        return [self.ident, self.index, self.version]


@attr.s(auto_attribs=True)
class Node():
    name: str
    attrs: Dict[str, Any]
    is_output: bool

    def to_json(self) -> Any:
        raise Exception("Abstract method, please implement me.")


@attr.s(auto_attribs=True)
class InputNode(Node):
    """An input node in the graph representation we lower to before NNVM's graph."""
    is_output: bool = False

    def to_json(self):
        return {
            "op": "null",
            "name": self.name,
            "inputs": []
        }


@attr.s(auto_attribs=True)
class OpNode(Node):
    """An operator node in the graph representation we lower to before NNVM's graph."""
    op_name: str
    inputs: List[NodeRef]
    op_attrs: Dict[str, Any]
    is_output: bool = False

    def to_json(self) -> Any:
        attrs = dict.copy(self.op_attrs)
        # Extend ops with extra info.
        attrs['func_name'] = self.op_name
        # When do we flatten?
        attrs['flatten_data'] = "0"
        # Fix me!
        attrs['num_inputs'] = str(len(self.inputs))
        attrs['num_outputs'] = "1"

        return {
            "op": "tvm_op",
            "name": self.name,
            "attrs": attrs,
            "inputs": self.inputs
        }


def from_tensor(typ: ir.TensorType) -> Tuple[str, List[int]]:
    dtype = typ.dtype.dtype
    shape = typ.shape
    dims = []
    for dim in shape.shapes:
        dims.append(dim.value)
    return dtype, dims


class TVMRTSCompiler(AbstractExprVisitor[NodeRef]):
    """The compiler from Relay to the TVM runtime system."""
    nodes: List[Node]
    id_map: Dict[ir.LocalId, NodeRef]

    def __init__(self) -> None:
        self.nodes = []
        self.id_map = {}

    def add_node(self, node: Node) -> NodeRef:
        self.nodes.append(node)
        ident = len(self.nodes) - 1
        return NodeRef(ident)

    def add_binding(self, ident: ir.LocalId, ref: NodeRef) -> None:
        self.id_map[ident] = ref

    def let_bind(self, ident: ir.LocalId, node: Node) -> NodeRef:
        ref = self.add_node(node)
        self.add_binding(ident, ref)
        return ref

    def get_node(self, ref: NodeRef) -> Node:
        return self.nodes[ref.ident]

    def lookup(self, ident: ir.LocalId) -> NodeRef:
        return self.id_map[ident]

    def compile(self, func: ir.Function) -> None:
        """Compile a single function into a graph."""
        # TODO: (@jroesch) Restore me
        # assert len(fn.ty_params) == 0

        # First we convert all the parameters into input nodes.
        params = func.params

        for param in params:
            dtype, shape = from_tensor(param.type)
            node = InputNode(f"{param.id.name}", {
                "shape": shape,
                "dtype": dtype,
            })
            self.let_bind(param.id, node)

        # Then we compile the body into a graph which can depend
        # on input variables.
        output_ref = self.visit(func.body)

        # Finally we retreive return value of program, which will
        # become our output node.
        self.get_node(output_ref).is_output = True

    def visit_let(self, let: ir.Let) -> NodeRef:
        """Visit the Let binding, by first traversing its value,
           then setting the metadata on the returned NodeRef.

           Finally visit the body, and return the NodeRef corresponding
           to it.
        """
        ident = let.id
        val = let.value
        body = let.body

        # Need to add type info?
        val_ref = self.visit(val)
        dtype, shape = from_tensor(get_checked_type(val))
        val_node = self.get_node(val_ref)
        val_node.attrs["dtype"] = dtype
        val_node.attrs["shape"] = shape
        self.add_binding(ident, val_ref)
        return self.visit(body)

    def visit_local_id(self, ident: ir.LocalId) -> NodeRef:
        return self.lookup(ident)

    def visit_call(self, call: ir.Call) -> NodeRef:
        inputs = []
        for arg in call.args:
            inputs.append(self.visit(arg).to_json())

        # need to deal with name mangle
        op_name = call.fn.name
        op_node = OpNode("call_name", {}, op_name, inputs, {})
        return self.add_node(op_node)

    def to_json(self) -> str:
        """Convert the sequence of nodes stored by the compiler into the
           JSON format defined in: https://docs.tvm.ai/dev/nnvm_json_spec.html.
        """
        nodes = []
        # First we compute "nodes" field.
        for node in self.nodes:
            nodes.append(node.to_json())

        arg_nodes = []
        heads = []
        # Compute "arg_nodes" and "heads" fields.
        for i, node in enumerate(self.nodes):
            if isinstance(node, InputNode):
                arg_nodes.append(i)

            if node.is_output:
                # Need to fix this.
                heads.append(NodeRef(i).to_json())

        # Compute "node_row_ptr".
        # TODO

        # Compute "attrs" field.
        attrs = {}

        # A
        shapes = []
        storage_ids = []
        dtype = []
        dltype = []

        for i, node in enumerate(self.nodes):
            storage_ids.append(i)
            shapes.append(node.attrs['shape'])
            if node.attrs['dtype'] == 'float32':
                dtype.append(0)
                dltype.append('float32')

        attrs["shape"] = ["list_shape", shapes]
        attrs["storage_id"] = ["list_int", storage_ids]
        attrs["dtype"] = ["list_int", dtype]
        attrs["dltype"] = ["list_str", dltype]

        json_dict = {
            "nodes": nodes,
            "arg_nodes": arg_nodes,
            "heads": heads,
            "attrs": attrs
        }

        return json.dumps(json_dict)


def compile_to_tvm(func):
    """Compile a single function to the components needed by the
       TVM RTS.
    """
    env = get_env()
    iids = []

    # Why do I need to call items?
    for op in env.operators():
        if not Operator_is_generic(op):
            iids.append(op.id)

    # TODO(@jroesch): Need to write test case for this
    mod = compile_ops_to_module(env, iids)
    comp = TVMRTSCompiler()
    comp.compile(func)
    graph_json = comp.to_json()
    return graph_json, mod, None  # params currently isn't supported by API
