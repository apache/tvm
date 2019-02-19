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
from collections import defaultdict
import attr
from . import _backend
from . import compile_engine
from ..op import Op
from ..expr import Function, GlobalVar
from ..expr_functor import ExprFunctor
from ..ty import TupleType, TensorType
from ... import target as _target


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
        self.params = {}
        self.storage_device_map = None
        self.compile_engine = compile_engine.get()
        self.lowered_funcs = defaultdict(set)
        self._name_map = {}

    def add_node(self, node, expr):
        """
        Add a node to the graph.

        Parameters
        ----------
        node: Node
            The node to add to the graph.

        expr: tvm.relay.Expr
            The corresponding expression.

        Returns
        -------
        node_ref: Union[NodeRef, List[NodeRef]]
            A reference to the node.
        """
        checked_type = expr.checked_type
        # setup storage ids
        assert expr in self.storage_device_map
        storage_device_info = self.storage_device_map[expr]
        assert len(storage_device_info) == 2
        node.attrs["storage_id"] = [x.value for x in storage_device_info[0]]
        device_types = [x.value for x in storage_device_info[1]]
        num_unknown_devices = device_types.count(0)
        if num_unknown_devices != 0 and num_unknown_devices != len(device_types):
            raise RuntimeError("The graph contains not annotated nodes for "
                               "heterogeneous execution. All nodes must be "
                               "annotated.")

        # Add the `device_index` attribute when the graph is annotated.
        if num_unknown_devices == 0:
            node.attrs["device_index"] = device_types

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

    def visit_constant(self, op):
        index = len(self.params)
        name = "p%d" % index
        self.params[name] = op.data
        node = InputNode(name, {})
        return self.add_node(node, op)

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

        assert call in self.storage_device_map
        device_types = self.storage_device_map[call][1]
        call_dev_type = device_types[0].value
        if isinstance(self.target, (str, _target.Target)):
            # homogeneous execution.
            cached_func = self.compile_engine.lower(func, self.target)
            self.target = {0: str(self.target)}
        elif isinstance(self.target, dict):
            # heterogeneous execution.
            if call_dev_type not in self.target:
                raise Exception("No target is provided for device " +
                                "{0}".format(call_dev_type))
            cached_func = self.compile_engine.lower(func,
                                                    self.target[call_dev_type])
        else:
            raise ValueError("self.target must be the type of str," +
                             "tvm.target.Target, or dict of int to str")
        for loweredf in cached_func.funcs:
            self.lowered_funcs[self.target[call_dev_type]].add(loweredf)

        inputs = []
        # flatten tuple in the call.
        for arg in call.args:
            res = self.visit(arg)
            if isinstance(arg.checked_type, TupleType):
                assert isinstance(res, tuple)
                inputs += res
            else:
                inputs.append(res)

        inputs = [x.to_json() for x in inputs]
        op_name = cached_func.func_name
        op_node = OpNode(self._get_unique_name(op_name), {},
                         op_name, inputs, {})
        return self.add_node(op_node, call)

    def visit_op(self, _):
        raise Exception("can not compile op in non-eta expanded form")

    def visit_ref_create(self, _):
        raise RuntimeError("reference not supported")

    def visit_ref_read(self, _):
        raise RuntimeError("reference not supported")

    def visit_ref_write(self, _):
        raise RuntimeError("reference not supported")

    def visit_constructor(self, _):
        raise Exception("ADT constructor case not yet implemented")

    def visit_match(self, _):
        raise Exception("match case not yet implemented")

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
        device_types = []
        dltypes = []
        node_row_ptr = [0]
        for node in self.nodes:
            assert node.num_outputs == len(node.attrs["shape"])
            shapes += node.attrs["shape"]
            dltypes += node.attrs["dtype"]
            storage_ids += node.attrs["storage_id"]
            if "device_index" in node.attrs:
                device_types += node.attrs["device_index"]
            num_entry += node.num_outputs
            node_row_ptr.append(num_entry)

        # Compute "attrs" field.
        attrs = {}
        attrs["shape"] = ["list_shape", shapes]
        attrs["storage_id"] = ["list_int", storage_ids]
        if device_types:
            attrs["device_index"] = ["list_int", device_types]
        attrs["dltype"] = ["list_str", dltypes]

        json_dict = {
            "nodes": nodes,
            "arg_nodes": arg_nodes,
            "heads": heads,
            "attrs": attrs,
            "node_row_ptr":  node_row_ptr
        }

        return json.dumps(json_dict, indent=2)

    def debug_dump_memory_plan(self, func):
        """Debug function to dump memory plan."""
        def _annotate(expr):
            if expr in self.storage_device_map:
                storage_device_info = self.storage_device_map[expr]
                assert len(storage_device_info) == 2
                return str(storage_device_info[0])
            return ""
        return func.astext(show_meta_data=False, annotate=_annotate)

    def debug_dump_device_annotation(self, func):
        """Debug function to dump device annotation result."""
        def _annotate(expr):
            if expr in self.storage_device_map:
                storage_device_info = self.storage_device_map[expr]
                assert len(storage_device_info) == 2
                return str(storage_device_info[1])
            return ""
        return func.astext(show_meta_data=False, annotate=_annotate)


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

        lowered_funcs : List[tvm.LoweredFunc] or Dict[str, List[tvm.LoweredFunc]]
            The lowered functions.

        params : Dict[str, tvm.nd.NDArray]
            Additional constant parameters.
        """
        self.storage_device_map = _backend.GraphPlanMemory(func)
        # First we convert all the parameters into input nodes.
        for param in func.params:
            node = InputNode(param.name_hint, {})
            self.var_map[param] = self.add_node(node, param)

        # Then we compile the body into a graph which can depend
        # on input variables.
        self.heads = self.visit(func.body)
        graph_json = self._get_json()

        # Return the lowered functions as a list for homogeneous compilation.
        # Otherwise, for heterogeneous compilation, a dictionary containing
        # the device id to a list of lowered functions is returned. Both forms
        # are acceptable to tvm.build.
        if not isinstance(self.target, dict):
            lowered_funcs = list(list(self.lowered_funcs.values())[0])
        else:
            lowered_funcs = {k: list(v) for k, v in self.lowered_funcs.items()}
        return graph_json, lowered_funcs, self.params

    def _get_unique_name(self, name):
        if name not in self._name_map:
            self._name_map[name] = 1
            return name
        index = self._name_map[name]
        self._name_map[name] += 1
        return self._get_unique_name(name + str(index))
