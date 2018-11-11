"""
A compiler from a Relay expression to TVM's graph runtime.

The compiler is built from a few pieces.

First we define a compiler from a single Relay expression to the
graph langauge. We require the expression to be a function.
The function's parameters correpond to the placeholder/inputs
and model parameters found in the computation graph representation.
The body of the function represents the computation graph.

The compiler's output is a program in the graph language, which is composed of
graph langauge is composed of Node, NodeRef, InputNode, OpNode.
This "little language" represents programs in TVM's graph format.

To connect to the graph runtime, we use a printer that converts our graph format
into TVM's JSON format. The resulting string can be loaded by
contrib.graph_runtime or any other TVM runtime comptatible system.
"""

from __future__ import absolute_import
import json
import attr
from . import compile_engine
from ..op import Op
from ..expr import Function, GlobalVar, ExprFunctor
from ..ty import TupleType, TensorType


@attr.s
class NodeRef(object):
    """A reference to a node, used for constructing the graph."""
    ident = attr.ib()
    index = attr.ib(default=0)
    version = attr.ib(default=0)

    def to_json(self):
        return [self.ident, self.index, self.version]


@attr.s
class Node(object):
    """The base class for nodes in the TVM runtime system graph input."""
    name = attr.ib()
    attrs = attr.ib()

    def to_json(self):
        raise Exception("Abstract method, please implement me.")


@attr.s
class InputNode(Node):
    """An input node in the TVM runtime system graph input."""
    name = attr.ib()
    attrs = attr.ib()

    def to_json(self):
        return {
            "op": "null",
            "name": self.name,
            "inputs": []
        }


@attr.s
class OpNode(Node):
    """An operator node in the TVM runtime system"s graph input."""
    op_name = attr.ib()
    inputs = attr.ib()
    op_attrs = attr.ib()
    num_outputs = attr.ib(default=1)

    def to_json(self):
        attrs = dict.copy(self.op_attrs)
        # Extend ops with extra info.
        attrs["func_name"] = self.op_name
        attrs["flatten_data"] = "0"
        attrs["num_inputs"] = str(len(self.inputs))
        attrs["num_outputs"] = str(self.num_outputs)

        return {
            "op": "tvm_op",
            "name": self.name,
            "attrs": attrs,
            "inputs": self.inputs
        }


def shape_to_json(shape):
    """Convert symbolic shape to json compatible forma."""
    return [sh.value for sh in shape]


class GraphRuntimeCodegen(ExprFunctor):
    """The compiler from Relay to the TVM runtime system."""
    nodes = attr.ib()
    var_map = attr.ib()

    def __init__(self, mod, target):
        ExprFunctor.__init__(self)
        self.mod = mod
        self.target = target
        self.nodes = []
        self.var_map = {}
        self.compile_engine = compile_engine.get()
        self.lowered_funcs = set()
        self._name_map = {}

    def add_node(self, node, checked_type):
        """
        Add a node to the graph.

        Parameters
        ----------
        node: Node
            The node to add to the graph.

        checked_type: Type
            The type of the node.

        Returns
        -------
        node_ref: Union[NodeRef, List[NodeRef]]
            A reference to the node.
        """
        node_id = len(self.nodes)
        self.nodes.append(node)
        # Tuple return value, flatten as tuple
        if isinstance(checked_type, TupleType):
            ret = []
            shape = []
            dtype = []
            for i, typ in enumerate(checked_type.fields):
                if not isinstance(typ, TensorType):
                    raise RuntimeError("type %s not supported" % typ)
                ret.append(NodeRef(node_id, i))
                shape.append(shape_to_json(typ.shape))
                dtype.append(typ.dtype)
            node.attrs["shape"] = shape
            node.attrs["dtype"] = dtype
            assert isinstance(node, OpNode)
            node.num_outputs = len(checked_type.fields)
            return tuple(ret)
        # Normal tensor return type
        if not isinstance(checked_type, TensorType):
            raise RuntimeError("type %s not supported" % checked_type)
        node.attrs["shape"] = [shape_to_json(checked_type.shape)]
        node.attrs["dtype"] = [checked_type.dtype]
        node.num_outputs = 1
        return NodeRef(node_id, 0)

    def visit_tuple(self, vtuple):
        fields = []
        for field in vtuple.fields:
            ref = self.visit(field)
            assert isinstance(ref, NodeRef)
            fields.append(ref)
        return tuple(fields)

    def visit_tuple_getitem(self, op):
        vtuple = self.visit(op.tuple_value)
        assert isinstance(vtuple, tuple)
        return vtuple[op.index]

    def visit_constant(self, _):
        raise RuntimeError("constant not supported")

    def visit_function(self, _):
        raise RuntimeError("function not supported")

    def visit_if(self, _):
        raise RuntimeError("if not supported")

    def visit_global_var(self, _):
        raise RuntimeError()

    def visit_let(self, let):
        """
        Visit the let binding, by first traversing its value,
        then setting the metadata on the returned NodeRef.

        Finally visit the body, and return the NodeRef corresponding
        to it.

        Parameters
        ----------
        let: tvm.relay.Expr
            The let binding to transform.

        Returns
        -------
        ref: NodeRef
            The node reference to the body.
        """
        assert let.var not in self.var_map
        self.var_map[let.var] = self.visit(let.value)
        return self.visit(let.body)

    def visit_var(self, rvar):
        return self.var_map[rvar]

    def visit_call(self, call):
        """Transform a ::tvm.relay.Call into an operator in the TVM graph."""
        if isinstance(call.op, Op):
            raise Exception(
                "Operators should be transformed away; try applying" +
                "the fuse_ops transformation to the expression.")
        elif isinstance(call.op, GlobalVar):
            func = self.mod[call.op]
        elif isinstance(call.op, Function):
            func = call.op
        else:
            raise Exception(
                "TVM runtime does not support calls to {0}".format(type(call.op)))
        if int(func.attrs.Primitive) != 1:
            raise Exception(
                "TVM only support calls to primitive functions " +
                "(i.e functions composed of fusable operator invocations)")

        cached_func = self.compile_engine.lower(func, self.target)
        for loweredf in cached_func.funcs:
            self.lowered_funcs.add(loweredf)

        inputs = []
        tuple_arg_count = 0
        for arg in call.args:
            if isinstance(arg.checked_type, TupleType):
                tuple_arg_count += 1
            inputs.append(self.visit(arg))
        # We need to specially handle tuple inputs and
        # tuple output cases.
        # Tuple input function(e.g. concat)
        if tuple_arg_count:
            assert len(call.args) == 1
            assert isinstance(inputs[0], tuple)
            inputs = list(inputs[0])

        inputs = [x.to_json() for x in inputs]
        op_name = cached_func.func_name
        op_node = OpNode(self._get_unique_name(op_name), {},
                         op_name, inputs, {})
        return self.add_node(op_node, call.checked_type)

    def _get_json(self):
        """
        Convert the sequence of nodes stored by the compiler into the
        TVM graph runtime format.

        Returns
        -------
        graph_json : str
            The generated JSON as a string.
        """
        nodes = []
        # First we compute "nodes" field.
        for node in self.nodes:
            nodes.append(node.to_json())

        arg_nodes = []
        # Compute "arg_nodes" and "heads" fields.
        for i, node in enumerate(self.nodes):
            if isinstance(node, InputNode):
                arg_nodes.append(i)

        heads = self.heads
        heads = heads if isinstance(heads, tuple) else [heads]
        heads = [x.to_json() for x in heads]

        # Compute "node_row_ptr" and entry attributes.
        num_entry = 0
        shapes = []
        storage_ids = []
        dltypes = []
        node_row_ptr = [0]
        for node in self.nodes:
            assert node.num_outputs == len(node.attrs["shape"])
            shapes += node.attrs["shape"]
            dltypes += node.attrs["dtype"]
            for i in range(node.num_outputs):
                storage_ids.append(i + num_entry)
            num_entry += node.num_outputs
            node_row_ptr.append(num_entry)

        # Compute "attrs" field.
        attrs = {}
        attrs["shape"] = ["list_shape", shapes]
        attrs["storage_id"] = ["list_int", storage_ids]
        attrs["dltype"] = ["list_str", dltypes]

        json_dict = {
            "nodes": nodes,
            "arg_nodes": arg_nodes,
            "heads": heads,
            "attrs": attrs,
            "node_row_ptr":  node_row_ptr
        }

        return json.dumps(json_dict, indent=2)

    def codegen(self, func):
        """Compile a single function into a graph.

        Parameters
        ----------
        func: tvm.relay.Expr
            The function to compile.

        Returns
        -------
        graph_json : str
            The graph json that can be consumed by runtime.

        lowered_funcs : List[tvm.LoweredFunc]
            The lowered functions.
        """
        # First we convert all the parameters into input nodes.
        for param in func.params:
            node = InputNode(param.name_hint, {})
            self.var_map[param] = self.add_node(
                node, param.type_annotation)

        # Then we compile the body into a graph which can depend
        # on input variables.
        self.heads = self.visit(func.body)
        graph_json = self._get_json()
        lowered_funcs = list(self.lowered_funcs)
        return graph_json, lowered_funcs

    def _get_unique_name(self, name):
        if name not in self._name_map:
            self._name_map[name] = 1
            return name
        index = self._name_map[name]
        self._name_map[name] += 1
        return self.get_unique_name(name + str(index))
