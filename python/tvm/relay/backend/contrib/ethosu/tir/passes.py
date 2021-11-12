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
# pylint: disable=invalid-name, unused-argument
"""The TIR passes to be run on Arm(R) Ethos(TM)-U NPU TIR Compiler."""
import numpy as np  # type: ignore

import tvm
from tvm.relay.backend.contrib.ethosu import vela_api
from .convolution import get_conv2d_params
from .depthwise import get_depthwise_conv2d_params
from .pooling import get_pooling_params
from .binary_elementwise import get_binary_elementwise_params
from .transform import get_copy_params
from .utils import get_weights_pointer, get_scale_bias_pointer


def RemoveZeroStores():
    """This pass removes stores which just store zero to initialise buffers.

    We don't codegen these stores and they otherwise considerably reduce
    the simplicity of the static traversal of convolution."""

    def _remove_zero_store(stmt):
        if isinstance(stmt.value, tvm.tir.IntImm) and int(stmt.value) == 0:
            return tvm.tir.Evaluate(tvm.tir.IntImm("uint8", 0))
        return stmt

    def _ftransform(f, mod, ctx):
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(f.body, _remove_zero_store, None, ["tir.Store"])
        )

    return tvm.tir.transform.prim_func_pass(
        _ftransform, opt_level=0, name="tir.ethosu.remove_zero_stores"
    )


def ReplaceOperators():
    """Replace operators represented as explicit loop nests with call_externs
    to NPU operators."""
    op_map = {
        "ethosu_conv2d": get_conv2d_params,
        "ethosu_copy": get_copy_params,
        "ethosu_depthwise_conv2d": get_depthwise_conv2d_params,
        "ethosu_pooling": get_pooling_params,
        "ethosu_binary_elementwise": get_binary_elementwise_params,
    }
    pointer_to_producer = {}
    pointer_to_consumer = {}
    replace_output_pointer = {}
    pointer_to_extents = {}

    def _resolve_pointers(stmt):
        """This pass determines information about the pointers present in the IR.
        In particular, it associates pointers with both the operations that
        produce them and the operations that consume them through the
        pointer_to_producer and pointer_to_consumer dicts.

        Additionally, it determines the extent (size/shape) of each pointer which
        is required for the _replace_pointers pass which runs later."""
        loads = []

        def _get_loads(stmt):
            if isinstance(stmt, tvm.tir.Load):
                loads.append(stmt.buffer_var)

        if isinstance(stmt, tvm.tir.Allocate):
            pointer_to_extents[stmt.buffer_var] = stmt.extents
            if isinstance(stmt.body[0], tvm.tir.AttrStmt):
                if stmt.body[0].attr_key == "pragma_op":
                    pointer_to_producer[stmt.buffer_var] = stmt.body[0]

        elif isinstance(stmt, tvm.tir.AttrStmt):
            if stmt.attr_key == "pragma_op":
                tvm.tir.stmt_functor.post_order_visit(stmt, _get_loads)
                for load_buffer in loads:
                    pointer_to_consumer[load_buffer] = stmt

    def _replace_operator(stmt):
        """Replace operators with call_externs, having derived the parameters
        from the relevant TIR expressions/statements.

        Note the complexity of this pass is mostly from the concept of 'replace
        pointers'. A call_extern may in principle require information from several
        loop nests in TIR (each corresponding to a different TE compute op). For
        example, a convolution operator will have other TE compute ops before and
        after corresponding to the input/output DMA functionality. Therefore, when
        the 'central' convolution op is replaced with a call_extern, the memory
        from the final DMA output op must be hoisted to the location/scope of
        the call_extern.

        The is done by replacing the pointer corresponding to the current operation
        with the 'true' output operator through the replace_output_pointer dict.
        Because of this, the param_func must provide a replace_pointer if the op
        isn't the true output but instead a no_compile op is."""
        if isinstance(stmt, tvm.tir.AttrStmt):
            op_name = stmt.value.value
            if stmt.attr_key == "pragma_op" and op_name in op_map:
                # Get the parameters for the extern call
                param_func = op_map[op_name]
                info, output_pointer, replace_pointer = param_func(
                    stmt, pointer_to_producer, pointer_to_consumer
                )
                if replace_pointer is not None:
                    replace_output_pointer[output_pointer] = replace_pointer
                # Make the extern call
                irb = tvm.tir.ir_builder.create()
                irb.emit(tvm.tir.call_extern("handle", op_name, *info))
                return irb.get()
        return None

    def _remove_no_compile(stmt):
        """Certain operators are marked as 'no compile' operators. This means they
        should be removed from the IR as they are compiled as part of other operators.
        The IFM DMA operations are an example of this, as they don't get compiled
        independently but instead get compiled into the operator they're associated with,
        e.g. a conv2d.

        There are potentially 3 parts to remove for an operator: the memory scope, the
        allocate for its output and the compute nest itself. For the memory scope and
        allocate, we can check if the pointer they reference is produced by a 'no compile'
        operator. For the compute nest, we can just check the op pragma."""
        if isinstance(stmt, tvm.tir.AttrStmt):
            # Remove memory scopes
            if stmt.node in pointer_to_producer:
                producer_attr = pointer_to_producer[stmt.node]
                if (
                    producer_attr.attr_key == "pragma_op"
                    and producer_attr.value.value not in op_map
                ):
                    return stmt.body

            # Remove compute nests
            if stmt.attr_key == "pragma_op" and stmt.value.value not in op_map:
                return tvm.tir.Evaluate(0)

        if isinstance(stmt, tvm.tir.Allocate):
            # Remove allocates
            if stmt.buffer_var in pointer_to_producer:
                op_attr = pointer_to_producer[stmt.buffer_var]
                if op_attr.attr_key == "pragma_op" and op_attr.value.value not in op_map:
                    return stmt.body
        return None

    def _replace_pointers(stmt):
        if isinstance(stmt, tvm.tir.AttrStmt):
            # If the attribute references a pointer that needs replacing
            if stmt.node in replace_output_pointer:
                replace_pointer = replace_output_pointer[stmt.node]
                # If the pointer doesn't have an extent registered to it,
                # this means the pointer is to a Buffer. In this case, we
                # just want to delete the memory scope attribute
                if replace_pointer not in pointer_to_extents:
                    return stmt.body
                # Otherwise, rewrite the memory scope attribute with the new pointer
                return tvm.tir.AttrStmt(
                    replace_output_pointer[stmt.node], stmt.attr_key, stmt.value, stmt.body
                )

        if isinstance(stmt, tvm.tir.Allocate):
            # If the allocate allocates a pointer that needs replacing
            if stmt.buffer_var in replace_output_pointer:
                replace_pointer = replace_output_pointer[stmt.buffer_var]
                # If the pointer doesn't have an extent registered to it,
                # this means the pointer is to a Buffer. In this case, we
                # just want to delete the allocation statement
                if replace_pointer not in pointer_to_extents:
                    return stmt.body
                # Otherwise, rewrite the allocation statement with the new pointer
                # and the new extent
                replace_type = replace_pointer.type_annotation.element_type.dtype
                replace_extents = pointer_to_extents[replace_pointer]
                return tvm.tir.Allocate(
                    replace_pointer, replace_type, replace_extents, stmt.condition, stmt.body
                )
        return None

    def _post_transform(stmt):
        # Replace operators with call_externs
        result = _replace_operator(stmt)
        # Remove operators that don't need compiling
        result = result or _remove_no_compile(stmt)
        # Replace necessary pointers that were removed in the previous step
        return result or _replace_pointers(stmt)

    def _ftransform(f, mod, ctx):
        tvm.tir.stmt_functor.post_order_visit(f.body, _resolve_pointers)
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(
                f.body, None, _post_transform, ["tir.AttrStmt", "tir.Allocate"]
            )
        )

    return tvm.tir.transform.prim_func_pass(
        _ftransform, opt_level=0, name="tir.ethosu.replace_operators"
    )


def DivideConstants(const_dict):
    """This pass rewrites the IR and constant dict such that all constant
    accesses are at 0 offset and full length (i.e. they read the whole buffer).

    Where necessary, new constants are created in order to ensure the rewrite
    can take place. As an example, if a convolution is tiled along the channels
    axis, the accesses to the weights will need to be offset. This pass will
    create new constants consisting of 'slices' of the weights so each tile
    of the compute can access one of these 'slices'.

    The purpose of this pass is to transform the IR into a form we can apply
    constant encoding to (which will compress weights and encode biases)."""
    buffer_to_const = {}  # type: ignore
    new_buffers = []
    new_consts = []
    keep_buffers = set()
    new_const_dict = {}

    def _visit(stmt):
        new_args = []
        for i, arg in enumerate(stmt.args):
            if isinstance(arg, tvm.tir.expr.Load):
                # If we're trying to load a buffer that maps to a constant
                if arg.buffer_var in buffer_to_const:
                    const = buffer_to_const[arg.buffer_var]
                    offset = int(arg.index)
                    # Note by convention the arg after a constant read is the length of the read
                    length = int(stmt.args[i + 1])
                    # If it's anything other than a full read, create a new buffer
                    if offset != 0 or len(const) != length:
                        new_consts.append(const[offset : offset + length])
                        new_buffer = tvm.tir.decl_buffer((length,), arg.dtype)
                        new_buffers.append(new_buffer)
                        new_args.append(tvm.tir.expr.Load(new_buffer.dtype, new_buffer.data, 0))
                        continue
                    keep_buffers.add(arg.buffer_var)

            new_args.append(arg)

        return tvm.tir.Call(stmt.dtype, stmt.op, new_args, stmt.span)

    def _ftransform(f, mod, ctx):
        for i, param in enumerate(f.params):
            if i in const_dict:
                buffer_to_const[param] = const_dict[i].flatten()
                buffer_to_const[f.buffer_map[param].data] = const_dict[i].flatten()

        new_body = tvm.tir.stmt_functor.ir_transform(f.body, _visit, None, ["tir.Call"])
        # Both the params and buffer map need updating for the newly introduced buffers
        new_params = []  # type: ignore
        new_buffer_map = {}
        for i, param in enumerate(f.params):
            buffer = f.buffer_map[param]
            pointer = buffer.data
            if pointer in buffer_to_const:
                if pointer not in keep_buffers:
                    continue
                new_const_dict[len(new_params)] = const_dict[i]
            new_params.append(param)
            new_buffer_map[param] = buffer

        for i, new_buffer in enumerate(new_buffers):
            handle = tvm.tir.Var("placeholder", "handle")
            new_params.append(handle)
            new_buffer_map[handle] = new_buffer
            new_const_dict[len(new_params) - 1] = new_consts[i]

        new_f = tvm.tir.PrimFunc(new_params, new_body, f.ret_type, new_buffer_map, f.attrs, f.span)
        return new_f

    def _divide_constants(mod):
        transform_func = tvm.tir.transform.prim_func_pass(
            _ftransform, opt_level=0, name="tir.ethosu.divide_constants"
        )
        new_func = transform_func(mod)
        return new_func, new_const_dict

    return _divide_constants


def EncodeConstants(const_dict):
    """the NPU requires that weights are compressed and bias/scales are 'encoded', both
    of which are performed by this pass.

    This pass modifies both the constant dict to contain the post-encoding values of the
    constants and the IR to adjust buffer types/sizes/accesses so they align with the
    encoded constants. Calls to the Vela API are made to perform the actual compression/
    encoding.

    """
    new_const_dict = {}
    buffer_to_const = {}
    pointer_to_buffer = {}
    rewrite_buffer = {}
    rewrite_pointer = {}
    accel_config = vela_api.get_accelerator_config()

    def _align_scale_bias(tir_extern_call, bias):
        """Align the scale_bias to 16 bytes."""
        value_bytes = bytearray()
        value_bytes.extend(bias.tobytes())
        # Align to 16
        remainder = (len(value_bytes)) % 16
        if remainder > 0:
            value_bytes.extend(bytearray(16 - remainder))
        value = np.frombuffer(value_bytes, dtype="uint8")
        return value

    def _encode_weights(tir_extern_call, weights):
        """Encode the weights for a TIR extern call."""
        value_bytes = vela_api.encode_weights(tir_extern_call, weights, accel_config)
        value = np.frombuffer(value_bytes, dtype="uint8")
        return value

    def _new_buffer(old_buffer, new_value):
        """Create a new buffer and add the old buffer and its pointer to the
        rewriting maps."""
        new_buffer = tvm.tir.decl_buffer((len(new_value),), str(new_value.dtype))
        pointer_to_buffer[new_buffer.data] = new_buffer
        rewrite_buffer[old_buffer] = new_buffer
        rewrite_pointer[old_buffer.data] = new_buffer.data
        buffer_to_const[new_buffer] = new_value

    def _visit_encode_pre(stmt):
        if isinstance(stmt, tvm.tir.Call):
            # Handle copies as a special-case by propagating the buffer information
            # from the read to the write pointer.
            if stmt.args[0] == "ethosu_copy":
                read_pointer = stmt.args[1].buffer_var
                if read_pointer in pointer_to_buffer:
                    write_pointer = stmt.args[3].buffer_var
                    # Assert writing to the base of the write_var (pre-StorageRewrite)
                    assert stmt.args[3].index == 0
                    assert stmt.args[1].index == 0
                    pointer_to_buffer[write_pointer] = pointer_to_buffer[read_pointer]
            else:
                # Encode the weights
                weights_pointer = get_weights_pointer(stmt)
                if weights_pointer is not None:
                    assert weights_pointer in pointer_to_buffer
                    weights_buffer = pointer_to_buffer[weights_pointer]
                    weights_value = buffer_to_const[weights_buffer]
                    new_weights_value = _encode_weights(stmt, weights_value)
                    _new_buffer(weights_buffer, new_weights_value)
                # Align the scale_bias to 16 bytes
                scale_bias_pointer = get_scale_bias_pointer(stmt)
                if scale_bias_pointer is not None:
                    assert scale_bias_pointer in pointer_to_buffer
                    scale_bias_buffer = pointer_to_buffer[scale_bias_pointer]
                    scale_bias_value = buffer_to_const[scale_bias_buffer]
                    new_scale_bias_value = _align_scale_bias(stmt, scale_bias_value)
                    _new_buffer(scale_bias_buffer, new_scale_bias_value)

    def _visit_encode_post(stmt):
        # Because encoding may change the data type (e.g. bias to uint8) and type information
        # is stored in pointer vars, it's necessary to rewrite all the pointers which point
        # to encoded data.
        if isinstance(stmt, tvm.tir.Allocate):
            allocate_pointer = stmt.buffer_var
            if allocate_pointer in pointer_to_buffer:
                buffer = pointer_to_buffer[allocate_pointer]
                if buffer in rewrite_buffer:  # If the pointer needs rewriting
                    # Create a new pointer var with the type of the new buffer
                    new_buffer = rewrite_buffer[buffer]
                    storage_type = tvm.ir.PrimType(new_buffer.dtype)
                    new_pointer = tvm.tir.Var(
                        allocate_pointer.name,
                        tvm.ir.PointerType(storage_type, buffer.scope()),
                        allocate_pointer.span,
                    )
                    # Set the new pointer to resolve to the new buffer
                    pointer_to_buffer[new_pointer] = new_buffer
                    # Add the old pointer to the pointer rewriting dict
                    rewrite_pointer[allocate_pointer] = new_pointer

    def _visit_rewrite(stmt):
        if isinstance(stmt, tvm.tir.Call):
            # For extern calls, we need to rewrite pairs of arguments corresponding to
            # base address load and the length of the load.
            new_args = [stmt.args[0]]
            for i in range(1, len(stmt.args)):
                # If the previous argument was a load, the current should be a length
                if isinstance(stmt.args[i - 1], tvm.tir.Load):
                    load = stmt.args[i - 1]
                    pointer = load.buffer_var
                    if pointer in pointer_to_buffer:
                        new_args.append(np.prod(list(pointer_to_buffer[pointer].shape)))
                        continue
                new_args.append(stmt.args[i])

            return tvm.tir.Call(stmt.dtype, stmt.op, new_args, stmt.span)
        if isinstance(stmt, tvm.tir.Allocate):
            # Where a pointer needs rewriting, the allocate for it must be rewritten
            allocate_pointer = stmt.buffer_var
            if allocate_pointer in pointer_to_buffer:
                if pointer_to_buffer[allocate_pointer] in rewrite_buffer:
                    new_buffer = rewrite_buffer[pointer_to_buffer[allocate_pointer]]
                    new_pointer = rewrite_pointer[allocate_pointer]
                    return tvm.tir.Allocate(
                        new_pointer,
                        new_buffer.dtype,
                        new_buffer.shape,
                        stmt.condition,
                        stmt.body,
                        stmt.span,
                    )
        # The following rewrites would be better expressed by just rewriting the Vars, however
        # ir_transform doesn't seem to visit Vars. So instead we do the next best thing and rewrite
        # the nodes which contain the Vars.
        if isinstance(stmt, tvm.tir.Load):
            load_pointer = stmt.buffer_var
            if load_pointer in rewrite_pointer:
                new_pointer = rewrite_pointer[load_pointer]
                element_type = new_pointer.type_annotation.element_type.dtype
                return tvm.tir.Load(
                    element_type, new_pointer, stmt.index, stmt.predicate, stmt.span
                )
        if isinstance(stmt, tvm.tir.AttrStmt):
            node_pointer = stmt.node
            if node_pointer in rewrite_pointer:
                return tvm.tir.AttrStmt(
                    rewrite_pointer[node_pointer], stmt.attr_key, stmt.value, stmt.body, stmt.span
                )
        return None

    def _ftransform(f, mod, ctx):
        for i, param in enumerate(f.params):
            if i in const_dict:
                buffer_to_const[f.buffer_map[param]] = const_dict[i].flatten()
                pointer_to_buffer[f.buffer_map[param].data] = f.buffer_map[param]

        # First analyse what needs to be rewritten
        new_body = tvm.tir.stmt_functor.ir_transform(
            f.body, _visit_encode_pre, _visit_encode_post, ["tir.Call", "tir.Allocate"]
        )
        # Then perform the rewrites
        new_body = tvm.tir.stmt_functor.ir_transform(
            f.body, None, _visit_rewrite, ["tir.Call", "tir.Allocate", "tir.Load", "tir.AttrStmt"]
        )
        new_buffer_map = {}
        # Rewrite the buffer map and const dict to instead use the encoded versions
        for i, param in enumerate(f.params):
            buffer = f.buffer_map[param]
            if buffer in rewrite_buffer:
                new_buffer = rewrite_buffer[buffer]
                new_buffer_map[param] = new_buffer
                new_value = buffer_to_const[new_buffer]
                new_const_dict[i] = new_value
            elif buffer in buffer_to_const:
                new_const_dict[i] = buffer_to_const[buffer]
                new_buffer_map[param] = buffer
            else:
                new_buffer_map[param] = buffer

        new_f = tvm.tir.PrimFunc(f.params, new_body, f.ret_type, new_buffer_map, f.attrs, f.span)
        return new_f

    def _encode_constants(mod):
        mod, divided_const_dict = DivideConstants(const_dict)(mod)
        const_dict.clear()
        for key, value in divided_const_dict.items():
            const_dict[key] = value
        transform_func = tvm.tir.transform.prim_func_pass(
            _ftransform, opt_level=0, name="tir.ethosu.encode_constants"
        )
        new_func = transform_func(mod)
        return new_func, new_const_dict

    return _encode_constants
