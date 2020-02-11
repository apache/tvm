import itertools
import numpy as np
import torch
import tvm
from tvm import relay
from tvm.relay import expr as _expr
from tvm.relay import analysis as _analysis
from tvm.relay.loops import while_loop
from tvm.relay import op as _op

from relay_op_conversion import convert_map, wrap_const


def is_int_seq(seq):
    return len(seq) > 0 and all([isinstance(i, int) for i in seq])


def parse_inputs(graph_inputs, input_shapes, input_types):
    ir_inputs = list(graph_inputs)
    ir_names = [i.debugName() for i in ir_inputs]
    input_vars = {}
    num_ir_inputs = len(ir_inputs)
    input_names = list(input_shapes.keys()) + list(input_types.keys())
    assert len(input_names) == len(ir_inputs) - 1

    for i in range(1, num_ir_inputs):
        iname = input_names[i-1]
        ir_inputs[i].setDebugName(iname)

        if i-1 >= len(input_shapes):
            itype = input_types[iname]
            input_vars[iname] = _expr.var(iname, type_annotation=itype)
        else:
            ishape = input_shapes[iname]
            assert ishape and is_int_seq(ishape)
            input_vars[iname] = _expr.var(iname, shape=ishape)

    # Add self (first input of a PyTorch graph) to inputs
    input_shape = [3]
    tensor = tvm.nd.array(np.zeros(input_shape).astype(np.float32))
    input_name = ir_names[0]  # self.1
    input_vars[input_name] = tensor

    return input_vars


def get_tensor_and_var(torch_tensor, name):
    tensor = tvm.nd.array(torch_tensor.cpu().numpy())
    var = _expr.var(name, shape=tensor.shape)
    return tensor, var


def get_output_name(node):
    assert node.outputsSize() == 1
    return node.output().debugName()


def get_output_names(node):
    return [output.debugName() for output in node.outputs()]


def get_input_names(node):
    return [inp.debugName() for inp in node.inputs()]


def getattr_attr_name(node):
    attribute_names = node.attributeNames()
    assert(len(attribute_names) == 1)
    attr_name = node.s(attribute_names[0])
    return attr_name


def get_use_chains(root_node, terminate=lambda _: False):
    def concat_lists(lists):
        return itertools.chain.from_iterable(lists)

    def inner(current, accum):
        users = []
        for output in current.outputs():
            users += [use.user for use in output.uses()]

        if not users or terminate(users):
            return [accum]

        return concat_lists([inner(nxt, accum + [nxt]) for nxt in users])

    return inner(root_node, [root_node])


def get_attr_chains(root_getattr_node):
    """Returns chains of attribute access starting from root_getattr_node

    For example, given attribute "block", as in "self.block" when "self" points
    to the top level torch.nn.Module, it returns lists of attribute "chains",
    e.g. ['block', '2'], ['block', '1'], ['block', '0', '_packed_params']

    These sets of attributes form full attribute accessors. For example,
    "self.block.1", "self.block.2" will return the second and third submodule,
    and "self.block.0._packed_params" will return the parameters of the first
    submodule.
    """
    def terminate(users):
        next_attrs = [user for user in users if user.kind() == "prim::GetAttr"]
        return len(next_attrs) == 0

    return get_use_chains(root_getattr_node, terminate)


def get_full_attr_name(getattrs):
    return ".".join([getattr_attr_name(node) for node in getattrs])


def parse_params(graph, state_dict):
    getattr_nodes = graph.findAllNodes("prim::GetAttr", recurse=True)
    params = {}
    param_tensors = {}
    seen = set()

    for node in getattr_nodes:
        if get_output_name(node) in seen:
            continue

        for getattrs in get_attr_chains(node):
            seen.update(map(get_output_name, getattrs))

            full_attr = get_full_attr_name(getattrs)
            full_attr_node_name = get_output_name(getattrs[-1])

            if full_attr in state_dict:
                torch_tensor = state_dict[full_attr]
                tensor, var = get_tensor_and_var(torch_tensor,
                                                 full_attr_node_name)
                param_tensors[full_attr_node_name] = tensor
                params[full_attr_node_name] = var

    return params, param_tensors


def get_input_types(op_node):
    input_list_types = []
    for input_node in op_node.inputs():
        in_ty = input_node.type()
        input_node_kind = in_ty.kind()
        if input_node_kind == 'TensorType':
            if in_ty.scalarType() is None:
                input_list_types.append('float')
            else:
                input_list_types.append(in_ty.scalarType().lower())
        elif input_node_kind == 'ListType':
            input_list_types.append(str(in_ty.getElementType()).lower())
        elif input_node_kind in ['IntType', 'FloatType', 'BoolType',
                                 'StringType', 'OptionalType']:
            input_list_types.append(str(in_ty).lower())
        else:
            input_list_types.append('UnsupportedType')

    if op_node.kind() in ['aten::ones', 'aten::zeros']:
        node_type = op_node.output().type()
        scalar_type = node_type.scalarType()
        if scalar_type:
            input_list_types[0] = scalar_type.lower()

    return input_list_types


def get_constant(node):
    attribute_names = node.attributeNames()
    num_attributes = len(attribute_names)

    if num_attributes == 1:
        attr_name = attribute_names[0]
        ty = node.output().type().kind()

        if ty == "IntType" or ty == "BoolType":
            return node.i(attr_name)
        elif ty == "FloatType":
            return node.f(attr_name)
        elif ty == "TensorType":
            tensor = node.t(attr_name)
            if len(tensor.shape) == 0:  # tensor(0.1)
                return float(tensor)
            return tensor
        elif ty == "DeviceObjType":
            return node.s(attr_name)
        elif ty == "FunctionType":
            return None
        else:
            print(ty, node)
            assert False  # TODO: handle other types
    else:
        assert num_attributes == 0
        return None


def parse_ops(nodes):
    ops = {}
    # Traverse nodes and add to graph
    for node in nodes:
        if node.outputsSize() > 1:
            node_name = "_".join(get_output_names(node))
        else:
            node_name = get_output_name(node)

        if node.kind() != "prim::GetAttr":
            ops[node_name] = node

    return ops


def get_input_node_names(op_node, output_index_map):
    return [output_index_map[name] for name in get_input_names(op_node)]


def get_op_inputs(op_node, outputs, output_index_map):
    input_names = get_input_node_names(op_node, output_index_map)
    return [outputs[name] for name in input_names]


def run_jit_passes(graph):
    torch._C._jit_pass_inline(graph)


def update_outputs_from_pairs(name_output_pairs, outputs, output_index_map):
    for output_name, output in name_output_pairs:
        output_index_map[output_name] = len(outputs)
        outputs.append(output)


def parse_block(block, outputs, output_index_map):
    ops = parse_ops(block.nodes())
    ret_name = get_input_names(block.returnNode())[0]
    return parse_operators(ops, outputs, output_index_map, ret_name)


def parse_loop(op_node, outputs, output_index_map):

    def get_input(index):
        inode = op_node.inputsAt(index).node()
        if inode.kind() == "prim::Constant":
            return _expr.const(get_constant(inode))
        var_name = op_node.inputsAt(index).debugName()
        assert var_name in output_index_map
        output_ind = output_index_map[var_name]
        out = outputs[output_ind]
        if isinstance(out, _expr.Expr):  # TODO: remove this condition
            return out
        if isinstance(out, list):
            return out
        return _expr.const(out)

    max_loop_count = get_input(0)
    init_cond = get_input(1)
    num_loop_var = len(list(op_node.inputs())) - 2
    init_vals = [get_input(i + 2) for i in range(num_loop_var)]

    is_for_loop = isinstance(init_cond, _expr.Constant)

    if is_for_loop:
        loop_iter_dtype = "int32"
        init_loop_iter_val = _expr.const(0, dtype="int32")
    else:
        loop_iter_dtype = "bool"
        init_loop_iter_val = init_cond

    body_block = list(op_node.blocks())[0]
    inames = get_input_names(body_block)
    loop_input_vals = [init_loop_iter_val] + init_vals
    name_val_pairs = list(zip(inames, loop_input_vals))
    update_outputs_from_pairs(name_val_pairs, outputs, output_index_map)

    def get_outputs(outputs, output_index_map, names):
        return [wrap_const(outputs[output_index_map[name]])
                for name in names]

    def cond(*current_vals):
        i = current_vals[0]

        if is_for_loop:
            return _op.less(i, max_loop_count)

        return _op.equal(i, _expr.const(True, 'bool'))

    def body(*current_vals):
        for (i, iname) in enumerate(inames):
            outputs[output_index_map[iname]] = current_vals[i]

        parse_block(body_block, outputs, output_index_map)

        block_output_names = get_output_names(body_block)
        block_outputs = get_outputs(outputs, output_index_map,
                                    block_output_names)
        if is_for_loop:
            incr = _expr.const(1, dtype="int32")
            block_outputs[0] = current_vals[0] + incr

        return block_outputs

    def get_var(name, val):
        if isinstance(val, _expr.Constant):
            return _expr.var(name, shape=(), dtype=val.data.dtype)
        if isinstance(val, _expr.Var):
            return _expr.var(name, type_annotation=val.type_annotation)
        if isinstance(val, list):
            assert False
        return _expr.var(name)

    loop_iter_var = _expr.var(inames[0], shape=(), dtype=loop_iter_dtype)
    loop_vars = [get_var(name, val) for name, val in name_val_pairs[1:]]
    loop = while_loop(cond, [loop_iter_var] + loop_vars, body)
    loop_val = loop(init_loop_iter_val, *init_vals)

    return [_expr.TupleGetItem(loop_val, i+1) for i in range(num_loop_var)]


def handle_unpack(op_node, outputs, output_index_map, inp):
    def unpack_and_update(tup, num_fields):
        assert num_fields == len(unpacked_names)
        unpacked = [_expr.TupleGetItem(tup, i) for i in range(num_fields)]
        update_outputs_from_pairs(zip(unpacked_names, unpacked),
                                  outputs, output_index_map)
    unpacked_names = get_output_names(op_node)

    if isinstance(inp, list):
        update_outputs_from_pairs(zip(unpacked_names, inp),
                                  outputs, output_index_map)
    else:
        if isinstance(inp, relay.Tuple):
            unpack_and_update(inp, len(inp.fields))
        elif isinstance(inp.type_annotation, relay.TupleType):
            fields = inp.type_annotation.fields
            unpack_and_update(inp, len(fields))
        else:
            assert False


def parse_operators(operators, outputs, output_index_map, ret_name):
    for node_name, op_node in operators.items():
        operator = op_node.kind()
        inputs = get_op_inputs(op_node, outputs, output_index_map)

        if operator == "prim::Constant":
            output_index_map[node_name] = len(outputs)
            outputs.append(get_constant(op_node))
        elif operator == 'prim::ListConstruct' and is_int_seq(inputs):
            output_index_map[node_name] = len(outputs)
            outputs.append(_expr.var(node_name, shape=inputs))
        elif operator == 'prim::ListConstruct':
            output_index_map[node_name] = len(outputs)
            outputs.append(inputs)
        elif operator == 'prim::TupleConstruct':
            output_index_map[node_name] = len(outputs)
            outputs.append(relay.Tuple(inputs))
        elif operator in ["prim::ListUnpack", 'prim::TupleUnpack']:
            assert len(inputs) == 1
            handle_unpack(op_node, outputs, output_index_map, inputs[0])
        elif operator == "prim::If":
            cond = outputs[output_index_map[op_node.inputsAt(0).debugName()]]
            blocks = list(op_node.blocks())
            true_branch = parse_block(blocks[0], outputs, output_index_map)
            false_branch = parse_block(blocks[1], outputs, output_index_map)
            output_index_map[node_name] = len(outputs)
            outputs.append(_expr.If(cond, true_branch, false_branch))
        elif operator == "prim::Loop":
            loop = parse_loop(op_node, outputs, output_index_map)
            unpacked_names = get_output_names(op_node)
            assert len(loop) == len(unpacked_names)
            update_outputs_from_pairs(zip(unpacked_names, loop),
                                      outputs, output_index_map)
        else:
            output_index_map[node_name] = len(outputs)
            relay_op = convert_map[operator]
            outputs.append(relay_op(inputs, get_input_types(op_node)))

    ret = outputs[output_index_map[ret_name]]

    if isinstance(ret, list):
        ret = _expr.Tuple(ret)
    else:
        ret = wrap_const(ret)
    return ret


def get_all_op_names(graph):
    nodes = list(graph.nodes())
    prim_with_blocks = ["prim::If", "prim::Loop"]
    for prim in prim_with_blocks:
        prim_nodes = graph.findAllNodes(prim, recurse=True)
        for prim_node in prim_nodes:
            for block in prim_node.blocks():
                nodes += block.nodes()
    return set([node.kind() for node in nodes])


def report_missing_conversion(graph):
    known_ops = ["prim::Constant", "prim::GetAttr",
                 "prim::ListConstruct", "prim::ListUnpack",
                 "prim::TupleConstruct", "prim::TupleUnpack",
                 "prim::If", "prim::Loop"]
    # ops added during rewrite
    known_ops += ["relay::empty_list",
                  "relay::cons_list",
                  "relay::rev_list",
                  "relay::tensor_array_stack"]

    known_ops += list(convert_map.keys())

    missing = [op_name for op_name in get_all_op_names(graph)
               if op_name not in known_ops]

    if missing:
        msg = "The following operators are not implemented: {}".format(missing)
        raise NotImplementedError(msg)


def rewrite_for_tensor_array(graph):
    def has_kind(chain, kind):
        return any([node.kind() == kind for node in chain])

    def needs_rewrite(chain):
        return has_kind(chain, "aten::stack") and has_kind(chain, "prim::Loop")

    def get_node(node_list, kind, filter_func=lambda node: True):
        for node in node_list:
            if node.kind() == kind and filter_func(node):
                return node
        assert False
        return None

    def node_type(node):
        return str(node.output().type())

    list_construct_ops = graph.findAllNodes("prim::ListConstruct")
    tensor_list_ops = [op for op in list_construct_ops
                       if node_type(op) == "List[Tensor]"]
    chains = []
    for tensor_list_op in tensor_list_ops:
        chains += get_use_chains(tensor_list_op)

    for chain in [chain for chain in chains if needs_rewrite(chain)]:
        tensor_list_op = chain[0]
        loop_op = get_node(chain, "prim::Loop")

        empty_list_node = graph.create("relay::empty_list")
        empty_list_node.insertBefore(loop_op)
        tensor_list_op.replaceAllUsesWith(empty_list_node)
        tensor_list_op.destroy()

        rev_list_node = graph.create("relay::rev_list",
                                     [loop_op.outputsAt(0)])
        rev_list_node.insertAfter(loop_op)

        stack_op = get_node(chain, "aten::stack")
        tarray_stack_node = graph.create("relay::tensor_array_stack",
                                         [rev_list_node.output()])
        tarray_stack_node.insertBefore(stack_op)
        stack_op.replaceAllUsesWith(tarray_stack_node)
        stack_op.destroy()

        loop_block = list(loop_op.blocks())[0]
        loop_nodes = list(loop_block.nodes())

        add_op = get_node(loop_nodes, "aten::add_",
                          lambda node: node_type(node) == "List[Tensor]")

        list_singlton_op = add_op.inputsAt(1).node()
        list_singlton_op_input = list_singlton_op.inputsAt(0)
        list_singlton_op.output().replaceAllUsesWith(list_singlton_op_input)
        list_singlton_op.destroy()

        cons_list_node = graph.create("relay::cons_list",
                                      list(reversed(list(add_op.inputs()))))
        cons_list_node.insertBefore(add_op)
        add_op.replaceAllUsesWith(cons_list_node)
        add_op.destroy()


def parse_script_module(script_module, input_shapes, input_types={}):
    graph = script_module.graph.copy()
    rewrite_for_tensor_array(graph)
    run_jit_passes(graph)
    # print(graph)
    report_missing_conversion(graph)

    params = script_module.state_dict()
    input_vars = parse_inputs(graph.inputs(), input_shapes, input_types)
    param_vars, tensors = parse_params(graph, params)

    input_vars.update(param_vars)
    outputs = list(input_vars.values())
    output_index_map = dict(zip(input_vars.keys(), range(len(outputs))))
    ret_name = get_input_names(graph.return_node())[0]

    body = parse_operators(parse_ops(graph.nodes()), outputs,
                           output_index_map, ret_name)
    tvm_params = {k: tvm.nd.array(v) for k, v in tensors.items()}

    #from relay_op_conversion import mod
    mod["main"] = tvm.relay.Function(_analysis.free_vars(body), body)

    return mod, tvm_params
