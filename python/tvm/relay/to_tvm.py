"""A compiler from Relay programs to TVM's graph runtime.
"""
import json
from typing import Dict, Any, List, Tuple, Set

import attr
from .ir_pass import AbstractExprVisitor
from .op import compile_ops, Op
from .type import TensorType
from .expr import LocalVar, Function, Let, Call


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


def shape_to_json(shape):
    return [str(sh.value) for sh in shape]

def from_tensor(typ: TensorType) -> Tuple[str, List[int]]:
    return (typ.dtype, shape_to_json(typ.shape))


class TVMRTSCompiler(AbstractExprVisitor[NodeRef]):
    """The compiler from Relay to the TVM runtime system."""
    nodes: List[Node]
    id_map: Dict[LocalVar, NodeRef]
    all_ops: Set[Op]

    def __init__(self) -> None:
        self.nodes = []
        self.id_map = {}
        self.all_ops = set()

    def add_node(self, node: Node) -> NodeRef:
        self.nodes.append(node)
        ident = len(self.nodes) - 1
        return NodeRef(ident)

    def add_binding(self, ident: LocalVar, ref: NodeRef) -> None:
        self.id_map[ident] = ref

    def let_bind(self, ident: LocalVar, node: Node) -> NodeRef:
        ref = self.add_node(node)
        self.add_binding(ident, ref)
        return ref

    def get_node(self, ref: NodeRef) -> Node:
        return self.nodes[ref.ident]

    def lookup(self, ident: LocalVar) -> NodeRef:
        return self.id_map[ident]

    def compile(self, func: Function) -> None:
        """Compile a single function into a graph."""
        # TODO: (@jroesch) Restore me
        # assert len(fn.ty_params) == 0

        # First we convert all the parameters into input nodes.
        params = func.params

        for param in params:
            dtype, shape = from_tensor(param.type)
            node = InputNode(f"{param.var.name_hint}", {
                "shape": shape,
                "dtype": dtype,
            })
            self.let_bind(param.var, node)

        # Then we compile the body into a graph which can depend
        # on input variables.
        output_ref = self.visit(func.body)

        # Finally we retreive return value of program, which will
        # become our output node.
        self.get_node(output_ref).is_output = True

    def visit_let(self, let: Let) -> NodeRef:
        """Visit the Let binding, by first traversing its value,
           then setting the metadata on the returned NodeRef.

           Finally visit the body, and return the NodeRef corresponding
           to it.
        """
        ident = let.var
        val = let.value
        body = let.body

        # Need to add type info?
        val_ref = self.visit(val)
        dtype, shape = from_tensor(val.checked_type())
        val_node = self.get_node(val_ref)
        val_node.attrs["dtype"] = dtype
        val_node.attrs["shape"] = shape
        self.add_binding(ident, val_ref)
        return self.visit(body)

    def visit_local_var(self, ident: LocalVar) -> NodeRef:
        return self.lookup(ident)

    def visit_call(self, call: Call) -> NodeRef:
        inputs = []
        for arg in call.args:
            inputs.append(self.visit(arg).to_json())

        assert isinstance(call.op, Op)
        self.all_ops.add(call.op.name)

        op_name = call.op.name
        attrs = { 'shape': shape_to_json(call.checked_type().shape),
                  'dtype': call.checked_type().dtype }
        op_node = OpNode("call_name", attrs, op_name, inputs, {})
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


def compile(func):
    """Compile a single function to the components needed by the
       TVM RTS.
    """
    comp = TVMRTSCompiler()
    comp.compile(func)
    op_names = list(comp.all_ops)
    mod = compile_ops(op_names)
    graph_json = comp.to_json()
    try:
        import nnvm
        graph = nnvm.graph.load_json(graph_json)
    except Exception as e:
        import traceback
        traceback.print_tb(e.__traceback__)
        import pdb; pdb.set_trace()
    return graph, mod, None  # params currently isn't supported by API
