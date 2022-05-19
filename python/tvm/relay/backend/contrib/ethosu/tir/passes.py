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
# pylint: disable=invalid-name, unused-argument, no-else-return, inconsistent-return-statements, too-many-nested-blocks
"""The TIR passes to be run on Arm(R) Ethos(TM)-U NPU TIR Compiler."""
from collections import namedtuple
from typing import Optional
import numpy as np  # type: ignore

import tvm
from tvm.relay.backend.contrib.ethosu import vela_api
from tvm.relay.backend.contrib.ethosu import tir_to_cs_translator as tirtocs
from ethosu.vela import api as vapi
from .convolution import get_conv2d_params
from .depthwise import get_depthwise_conv2d_params
from .pooling import get_pooling_params
from .binary_elementwise import get_binary_elementwise_params
from .identity import get_identity_params
from .unary_elementwise import get_unary_elementwise_params
from .transform import get_copy_params
from .producers_consumers import ProducersConsumers

from .. import _ffi_api


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
        _ftransform, opt_level=0, name="tir.contrib.ethos-u.remove_zero_stores"
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
        "ethosu_identity": get_identity_params,
        "ethosu_unary_elementwise": get_unary_elementwise_params,
    }
    producers_consumers = ProducersConsumers()
    replace_output_pointer = {}
    pointer_to_extents = {}

    ReplaceInfo = namedtuple("ReplaceInfo", ["pointer", "reallocate"])

    def _find_pointer_to_extent(stmt):
        if isinstance(stmt, tvm.tir.Allocate):
            pointer_to_extents[stmt.buffer_var] = stmt.extents

    def _resolve_pointers(stmt):
        """This pass determines information about the pointers present in the IR.
        In particular, it associates pointers with both the operations that
        produce them and the operations that consume them through the
        pointer_to_producer and pointer_to_consumer dicts.

        Additionally, it determines the extent (size/shape) of each pointer which
        is required for the _replace_pointers pass which runs later."""
        loads = []

        def _get_loads(stmt):
            if isinstance(stmt, tvm.tir.BufferLoad):
                loads.append(stmt.buffer.data)

        buffer_var = None

        def _get_buffer_var(stmt):
            if isinstance(stmt, tvm.tir.BufferStore):
                nonlocal buffer_var
                buffer_var = stmt.buffer.data

        if isinstance(stmt, tvm.tir.AttrStmt):
            if stmt.attr_key == "pragma_op":
                tvm.tir.stmt_functor.post_order_visit(stmt, _get_buffer_var)
                producers_consumers.add_producer(buffer_var, stmt)

                tvm.tir.stmt_functor.post_order_visit(stmt, _get_loads)
                for load_pointer in loads:
                    if load_pointer != buffer_var:
                        producers_consumers.add_consumer(load_pointer, stmt)

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
                info, output_pointer, replace_pointer, is_allocator = param_func(
                    stmt, producers_consumers
                )
                if replace_pointer is not None:
                    replace_output_pointer[output_pointer] = ReplaceInfo(
                        replace_pointer, is_allocator
                    )
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

        There are potentially 2 parts to remove for an operator:
        the allocate for its output and the compute nest itself. For the
        allocate, we can check if the pointer they reference is produced by a 'no compile'
        operator. For the compute nest, we can just check the op pragma."""
        if isinstance(stmt, tvm.tir.AttrStmt):
            # Remove compute nests
            if stmt.attr_key == "pragma_op" and stmt.value.value not in op_map:
                return tvm.tir.Evaluate(0)

        if isinstance(stmt, tvm.tir.Allocate):
            # Remove allocates
            producer = producers_consumers.get_last_producer(stmt.buffer_var)
            if producer:
                if producer.attr_key == "pragma_op" and producer.value.value not in op_map:
                    return stmt.body

        return None

    def _replace_pointers(stmt):
        if isinstance(stmt, tvm.tir.Allocate):
            # If the allocate allocates a pointer that needs replacing
            if stmt.buffer_var in replace_output_pointer:
                replace_pointer, reallocate = replace_output_pointer[stmt.buffer_var]
                if not reallocate:
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
        tvm.tir.stmt_functor.post_order_visit(f.body, _find_pointer_to_extent)
        tvm.tir.stmt_functor.post_order_visit(f.body, _resolve_pointers)
        producers_consumers.add_allocate_variables(pointer_to_extents.keys())
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(
                f.body, None, _post_transform, ["tir.AttrStmt", "tir.Allocate"]
            )
        )

    return tvm.tir.transform.prim_func_pass(
        _ftransform, opt_level=0, name="tir.contrib.ethos-u.replace_operators"
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
        # We don't want to divide the constant that will be executed on two cores in parallel
        is_u65_conv2d = (
            vela_api.get_accelerator_config() == vapi.NpuAccelerator.Ethos_U65_512
            and stmt.args[0] == "ethosu_conv2d"
        )
        for i, arg in enumerate(stmt.args):
            if isinstance(arg, tvm.tir.expr.BufferLoad):
                # If we're trying to load a buffer that maps to a constant
                if arg.buffer.data in buffer_to_const:
                    const = buffer_to_const[arg.buffer.data]
                    flattened_const_shape = np.prod(const.shape)

                    offset = int(arg.indices[0])
                    # Note by convention the arg after a constant read is the length of the read
                    length = int(stmt.args[i + 1])
                    # If it's anything other than a full read, create a new buffer
                    if (offset != 0 or flattened_const_shape != length) and not is_u65_conv2d:
                        out_channels = const.shape[0]
                        offset_channels = int((offset * out_channels) / flattened_const_shape)
                        length_channels = int((length * out_channels) / flattened_const_shape)
                        # split the constant up across channels
                        split_const = np.split(const, out_channels, axis=0)
                        # create a new const out of the channels we want to keep
                        new_const = np.concatenate(
                            split_const[offset_channels : offset_channels + length_channels], axis=0
                        )
                        new_consts.append(new_const)
                        new_buffer = tvm.tir.decl_buffer(
                            (length,), arg.dtype, scope=arg.buffer.scope()
                        )
                        new_buffers.append(new_buffer)
                        new_args.append(tvm.tir.expr.BufferLoad(new_buffer, [0]))
                        continue
                    keep_buffers.add(arg.buffer.data)

            new_args.append(arg)

        return tvm.tir.Call(stmt.dtype, stmt.op, new_args, stmt.span)

    def _ftransform(f, mod, ctx):
        for i, param in enumerate(f.params):
            if i in const_dict:
                buffer_to_const[param] = const_dict[i]
                buffer_to_const[f.buffer_map[param].data] = const_dict[i]

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

        new_f = tvm.tir.PrimFunc(
            new_params,
            new_body,
            f.ret_type,
            new_buffer_map,
            f.preflattened_buffer_map,
            f.attrs,
            f.span,
        )
        return new_f

    def _divide_constants(mod):
        transform_func = tvm.tir.transform.prim_func_pass(
            _ftransform, opt_level=0, name="tir.contrib.ethos-u.divide_constants"
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

    def collect_encoding_definitions(stmt, old_buffer_to_const):
        # Map from copy destination to copy source.
        copy_map = {}
        # List of buffer copies that occurred
        copied_buffers = []
        # List of encoded buffer information
        constant_buffer_replacements = []

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

        accel_config = vela_api.get_accelerator_config()

        def _encode_weights(tir_extern_call, weights):
            """Encode the weights for a TIR extern call."""
            value_bytes = vela_api.encode_weights(tir_extern_call, weights, accel_config)
            value = np.frombuffer(value_bytes, dtype="uint8")
            return value

        def _declare_constant_buffer(old_buffer, encoded_constants, split_idx):
            """Create a new buffer and add the old buffer and its pointer to the
            rewriting maps."""
            new_buffer = tvm.tir.decl_buffer(
                shape=[len(encoded_constants)],
                dtype=str(encoded_constants.dtype),
                name=old_buffer.name + "_encoded",
                scope=old_buffer.scope(),
            )

            constant_buffer_replacements.append(
                {
                    "old_buffer": old_buffer,
                    "new_buffer": new_buffer,
                    "encoded_constants": encoded_constants,
                    "split_idx": split_idx,
                }
            )

        def _encode_weights_or_bias(buffer1, buffer2, stmt, encode_func):
            """Encode the weights or align the bias either for one or two cores,
            depending on the variant."""
            constant = old_buffer_to_const[buffer1]

            # If we have just one core, encode the whole constant
            if buffer2 is None:
                new_const = encode_func(stmt, constant)
                return new_const, None

            # Assume that the constant tensor has not been flattened yet
            assert len(constant.shape) != 1
            channels = constant.shape[0]
            split_const = np.split(constant, channels, axis=0)

            const_list = [split_const[i] for i in range(channels) if i % 2 == 0]
            const_to_encode = np.concatenate(const_list, axis=0)

            new_const = encode_func(stmt, const_to_encode)
            split_idx = len(new_const)

            # Encode half of the constant separately for the other core if it exists
            assert buffer1.same_as(buffer2)
            const2_list = [split_const[i] for i in range(channels) if i % 2 == 1]
            const2_to_encode = np.concatenate(const2_list, axis=0)

            new_const2 = encode_func(stmt, const2_to_encode)
            new_const = np.append(new_const, new_const2).astype("uint8")

            return new_const, split_idx

        def _visit(stmt):
            if isinstance(stmt, tvm.tir.Call):
                op = str(stmt.args[0].value)
                # Handle copies as a special-case by propagating the buffer information
                # from the read to the write pointer.
                if op == "ethosu_copy":
                    read_buffer = stmt.args[1].buffer
                    write_buffer = stmt.args[3].buffer
                    # Assert writing to the base of the write_var (pre-StorageRewrite)
                    assert list(stmt.args[3].indices) == [0]
                    assert list(stmt.args[1].indices) == [0]
                    copied_buffers.append({"source": read_buffer, "dest": write_buffer})
                    copy_map[write_buffer] = read_buffer

                ops_with_weights = {
                    "ethosu_conv2d": tirtocs.translate_ethosu_conv2d,
                    "ethosu_depthwise_conv2d": tirtocs.translate_ethosu_depthwise_conv2d,
                }
                if op in ops_with_weights:
                    npu_op, _ = ops_with_weights[op](stmt)

                    # Encode the weights
                    weights_buffer = npu_op.weights[0].address.buffer
                    if weights_buffer in copy_map:
                        weights_buffer = copy_map[weights_buffer]

                    # In case of U65 512 mac variant the weights are split across two cores
                    # and need to be encoded separately
                    weights2_buffer = (
                        npu_op.weights[1].address.buffer
                        if accel_config == vapi.NpuAccelerator.Ethos_U65_512
                        else None
                    )
                    if weights2_buffer in copy_map:
                        weights2_buffer = copy_map[weights2_buffer]

                    new_weights, split_idx = _encode_weights_or_bias(
                        weights_buffer, weights2_buffer, stmt, _encode_weights
                    )
                    _declare_constant_buffer(weights_buffer, new_weights, split_idx)

                    # Align the scale_bias to 16 bytes
                    scale_bias_buffer = npu_op.biases[0].address.buffer
                    if scale_bias_buffer in copy_map:
                        scale_bias_buffer = copy_map[scale_bias_buffer]
                    scale_bias2_buffer = (
                        npu_op.biases[1].address.buffer
                        if accel_config == vapi.NpuAccelerator.Ethos_U65_512
                        else None
                    )
                    if scale_bias2_buffer in copy_map:
                        scale_bias2_buffer = copy_map[scale_bias2_buffer]

                    new_scale_bias, split_idx = _encode_weights_or_bias(
                        scale_bias_buffer, scale_bias2_buffer, stmt, _align_scale_bias
                    )

                    _declare_constant_buffer(scale_bias_buffer, new_scale_bias, split_idx)

        tvm.tir.stmt_functor.post_order_visit(stmt, _visit)

        return {
            "copied_buffers": copied_buffers,
            "constant_buffer_replacements": constant_buffer_replacements,
        }

    def transform_stmt(
        stmt, buf_remap, var_remap, pointer_to_buffer, new_buffer_to_const, new_buffer_to_split_idx
    ):
        def _visit_rewrite(stmt):
            if isinstance(stmt, tvm.tir.Call):
                # For extern calls, we need to rewrite pairs of arguments corresponding to
                # base address load and the length of the load.
                old_args = list(stmt.args)

                new_args = [stmt.args[0]]
                for prev_arg, arg in zip(old_args[:-1], old_args[1:]):
                    # If the previous argument was a load from an
                    # encoded buffer, the current should be a length.
                    if (
                        isinstance(prev_arg, tvm.tir.BufferLoad)
                        and prev_arg.buffer in new_buffer_to_const
                    ):
                        buffer_size = np.prod(list(prev_arg.buffer.shape))
                        arg = buffer_size
                        # We have to check for split weights/bias for conv2d and depthwise_conv2d
                        if old_args[0] in ("ethosu_conv2d", "depthwise_conv2d"):
                            # We have split weights/bias
                            if prev_arg.buffer in new_buffer_to_split_idx:
                                split_idx = new_buffer_to_split_idx[prev_arg.buffer]
                                # The first half of the split buffer
                                if prev_arg.indices[0] == 0:
                                    arg = split_idx
                                # the second half of the split buffer
                                else:
                                    arg = buffer_size - split_idx

                    new_args.append(arg)

                return tvm.tir.Call(stmt.dtype, stmt.op, new_args, stmt.span)

            if isinstance(stmt, tvm.tir.Allocate):
                # Where a pointer needs rewriting, the allocate for it must be rewritten
                allocate_pointer = stmt.buffer_var
                if allocate_pointer in var_remap:
                    new_allocate_pointer = var_remap[allocate_pointer]
                    new_buffer = pointer_to_buffer[new_allocate_pointer]

                    return tvm.tir.Allocate(
                        new_buffer.data,
                        new_buffer.dtype,
                        new_buffer.shape,
                        stmt.condition,
                        stmt.body,
                        stmt.span,
                    )

            # The following rewrites would be better expressed by just
            # rewriting the Buffers. However ir_transform doesn't
            # visit Buffers, so instead we do the next best thing and
            # rewrite the nodes which contain the Buffers.
            if isinstance(stmt, tvm.tir.BufferLoad):
                if stmt.buffer in buf_remap:
                    new_buffer = buf_remap[stmt.buffer]
                    new_indices = stmt.indices
                    offset = new_indices[0]
                    if offset != 0 and new_buffer in new_buffer_to_split_idx:
                        offset = new_buffer_to_split_idx[new_buffer]
                    return tvm.tir.BufferLoad(buf_remap[stmt.buffer], [offset], stmt.span)

            if isinstance(stmt, tvm.tir.AttrStmt):
                node_pointer = stmt.node
                if node_pointer in var_remap:
                    return tvm.tir.AttrStmt(
                        var_remap[node_pointer],
                        stmt.attr_key,
                        stmt.value,
                        stmt.body,
                        stmt.span,
                    )

            return None

        return tvm.tir.stmt_functor.ir_transform(
            stmt,
            None,
            _visit_rewrite,
            ["tir.Call", "tir.Allocate", "tir.BufferLoad", "tir.AttrStmt"],
        )

    def _ftransform(f, mod, ctx):
        # Step 0: Unpack the constant dictionary in terms of the
        # functions buffers.
        old_buffer_to_const = {}
        for i, param in enumerate(f.params):
            if i in const_dict:
                old_buffer_to_const[f.buffer_map[param]] = const_dict[i]

        # Step 1: Collect information on the buffers that will be
        # replaced by encodings.
        buffer_information = collect_encoding_definitions(f.body, old_buffer_to_const)

        # Step 2: Generate variable/buffer remaps, based on the
        # collected information.
        buf_remap = {}
        new_buffer_to_const = {}
        new_buffer_to_split_idx = {}

        # Any encoded buffers must be replaced
        for info in buffer_information["constant_buffer_replacements"]:
            buf_remap[info["old_buffer"]] = info["new_buffer"]
            new_buffer_to_const[info["new_buffer"]] = info["encoded_constants"]

            if info["split_idx"]:
                new_buffer_to_split_idx[info["new_buffer"]] = info["split_idx"]

        # Any buffers that are copied into from an encoded buffer must
        # be replaced.
        for info in buffer_information["copied_buffers"]:
            copy_source = info["source"]
            while copy_source in buf_remap:
                copy_source = buf_remap[copy_source]

            copy_dest = info["dest"]

            if copy_source.shape != copy_dest.shape or copy_source.dtype != copy_dest.dtype:
                new_dest = tvm.tir.decl_buffer(
                    shape=copy_source.shape,
                    dtype=copy_source.dtype,
                    name=copy_dest.name,
                    scope=copy_dest.scope(),
                )
                buf_remap[copy_dest] = new_dest
                if copy_source in new_buffer_to_const:
                    new_buffer_to_const[new_dest] = new_buffer_to_const[copy_source]

                if copy_source in new_buffer_to_split_idx:
                    new_buffer_to_split_idx[new_dest] = new_buffer_to_split_idx[copy_source]

        # Define additional dependent lookup tables.
        var_remap = {old.data: new.data for (old, new) in buf_remap.items()}
        pointer_to_buffer = {
            buf.data: buf for (old, new) in buf_remap.items() for buf in [old, new]
        }

        # Step 3: Then perform the rewrites
        new_body = transform_stmt(
            f.body,
            buf_remap,
            var_remap,
            pointer_to_buffer,
            new_buffer_to_const,
            new_buffer_to_split_idx,
        )

        # Step 4: Rewrite the buffer map and const dict to instead use the encoded versions
        new_buffer_map = {}
        for i, param in enumerate(f.params):
            buffer = f.buffer_map[param]
            if buffer in buf_remap:
                buffer = buf_remap[buffer]

            if buffer in new_buffer_to_const:
                new_const_dict[i] = new_buffer_to_const[buffer].flatten()
            elif buffer in old_buffer_to_const:
                new_const_dict[i] = old_buffer_to_const[buffer].flatten()

            new_buffer_map[param] = buffer

        new_f = tvm.tir.PrimFunc(
            f.params,
            new_body,
            f.ret_type,
            new_buffer_map,
            f.preflattened_buffer_map,
            f.attrs,
            f.span,
        )
        return new_f

    def _encode_constants(mod):
        mod, divided_const_dict = DivideConstants(const_dict)(mod)
        const_dict.clear()
        for key, value in divided_const_dict.items():
            const_dict[key] = value
        transform_func = tvm.tir.transform.prim_func_pass(
            _ftransform, opt_level=0, name="tir.contrib.ethos-u.encode_constants"
        )
        new_func = transform_func(mod)
        return new_func, new_const_dict

    return _encode_constants


# This need to be kept in sync with kDisableLowerTVMBuiltin in include/tvm/tir/transform.h
DISABLE_LOWER_BUILTIN = "disable_lower_builtin"


def AnnotateAllocates():
    """
    This is pass to annotate all allocate
    nodes of the PrimFuncs of the microNPU
    to be not lowered to built-ins.
    """

    def _post_transform(allocate):
        return tvm.tir.Allocate(
            buffer_var=allocate.buffer_var,
            dtype=allocate.dtype,
            extents=allocate.extents,
            condition=allocate.condition,
            body=allocate.body,
            annotations={DISABLE_LOWER_BUILTIN: True},
        )

    def _ftransform(f, mod, ctx):
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(f.body, None, _post_transform, ["tir.Allocate"])
        )

    return tvm.tir.transform.prim_func_pass(
        _ftransform, opt_level=0, name="tir.contrib.ethos-u.annotate_allocates"
    )


def RemoveConcatenates():
    """Remove concatenate operators by modifying the input buffers to write directly into
    the concatenated buffer with the appropriate offset.

    This pass works in two stages. The first finds every concatenate operation (marked by
    pragma_op = ethosu_concatenate) and it performs the following analysis. For each buffer
    that is concatenated, the buffer is marked that it is to be replaced with the concat
    buffer and the axis along which it is concatenated as well as the offset along that
    axis is recorded in 'ReplaceInfo'. Once this analysis is completed, the concatenate
    loop nest along with its buffer realization statements are removed.

    In the second stage, the input buffers to the concatenate operators are rewritten
    to use the concat buffer directly. This means applying the correct offset to the
    concatenation axis where ever the buffer is loaded or stored. Additionally, as the
    realization statements for the concat buffers were removed in the first stage, they
    are rewritten in place of the input buffer realization with the earliest liveness."""

    in_concat = [False]  # Whether the visitor is currently inside a concatenate operator
    concat_buffers = []  # The buffers produced by concatenate operators
    buffer_replace_map = {}  # A map of buffers to be replaced with the concat buffer
    attrs_by_buffer = {}  # AttrStmts by the buffer they reference
    realizes_by_buffer = {}  # BufferRealize statements by the buffer they reference
    first_replacements = {}  # The first buffers to be replaced by a given concat buffer

    ReplaceInfo = namedtuple("ReplaceInfo", ["buffer", "axis", "offset"])

    def _get_replace_info(buffer_load, concat_buffer):
        axis = 0
        offset = 0
        dmap = dict()

        for i, index in enumerate(buffer_load.indices):
            if isinstance(index, tvm.tir.Sub):
                axis = i
                dmap = {}

                def _visit(stmt):
                    if isinstance(stmt, tvm.tir.Var):
                        dmap[stmt] = tvm.arith.IntervalSet(0, 0)

                tvm.tir.stmt_functor.post_order_visit(index, _visit)
                offset = abs(int(tvm.arith.Analyzer().int_set(index, dmap).max_value))
        return ReplaceInfo(concat_buffer, axis, offset)

    def _pre_remove(stmt):
        if isinstance(stmt, tvm.tir.BufferRealize):
            # Record the realize statements by buffer as we need to hoist some of these
            realizes_by_buffer[stmt.buffer] = stmt
        if isinstance(stmt, tvm.tir.AttrStmt):
            if stmt.attr_key == "realize_scope" and isinstance(stmt.node, tvm.tir.Buffer):
                # Record the realize_scope attrs by buffer as we need to hoist some of these
                attrs_by_buffer[stmt.node] = stmt
            if stmt.attr_key == "pragma_op" and stmt.value.value == "ethosu_concatenate":
                # Record that we're entering a concatenate loop nest
                in_concat[0] = True
        if isinstance(stmt, tvm.tir.BufferLoad) and in_concat[0]:
            # Any buffer loaded inside a concat is a buffer we intend to replace with this pass.
            # The buffer_replace_map keeps track of which buffers need replacing with the
            # concat buffer.
            replace_info = _get_replace_info(stmt, concat_buffers[-1])
            buffer_replace_map[stmt.buffer] = replace_info
        if isinstance(stmt, tvm.tir.BufferStore) and in_concat[0]:
            # If we're inside a concat, the BufferStore indicates what the concat buffer is
            concat_buffers.append(stmt.buffer)

    def _post_remove(stmt):
        if isinstance(stmt, tvm.tir.AttrStmt):
            if isinstance(stmt.node, tvm.tir.Buffer) and stmt.node in concat_buffers:
                return stmt.body
            if stmt.attr_key == "pragma_op" and stmt.value.value == "ethosu_concatenate":
                # When we leave a concatenate operator, record it and then remove the loop nest
                in_concat[0] = False
                return tvm.tir.Evaluate(0)
        if isinstance(stmt, tvm.tir.BufferRealize):
            if stmt.buffer in concat_buffers:
                return stmt.body
        return None

    def _pre_replace(stmt):
        if isinstance(stmt, (tvm.tir.BufferLoad, tvm.tir.BufferStore)):
            # The first buffer referenced that needs replacing with a concat buffer shall
            # be the one that the concat buffer realize is hoisted to.
            if stmt.buffer in buffer_replace_map:
                concat_buffer = buffer_replace_map[stmt.buffer].buffer
                if concat_buffer not in first_replacements:
                    first_replacements[concat_buffer] = stmt.buffer

    def _post_replace(stmt):
        if isinstance(stmt, tvm.tir.BufferStore):
            if stmt.buffer in buffer_replace_map:
                # Replace the original buffer store with a new one into the concat buffer
                # and adjust the indices accordingly to account for the offset
                replace_info = buffer_replace_map[stmt.buffer]
                concat_buffer = replace_info.buffer
                new_indices = list(stmt.indices)
                new_indices[replace_info.axis] += replace_info.offset
                # The new buffer store node that stores the tensor directly into the concat buffer
                new_store = tvm.tir.BufferStore(concat_buffer, stmt.value, new_indices, stmt.span)
                return new_store
        if isinstance(stmt, tvm.tir.BufferLoad):
            if stmt.buffer in buffer_replace_map:
                # Replace the original buffer load with a new one into the concat buffer
                # and adjust the indices accordingly to account for the offset
                replace_info = buffer_replace_map[stmt.buffer]
                concat_buffer = replace_info.buffer
                new_indices = list(stmt.indices)
                new_indices[replace_info.axis] += replace_info.offset
                new_load = tvm.tir.BufferLoad(concat_buffer, new_indices, stmt.span)
                return new_load
        if isinstance(stmt, tvm.tir.BufferRealize):
            if stmt.buffer in buffer_replace_map:
                concat_buffer = buffer_replace_map[stmt.buffer].buffer
                # If this isn't the first buffer replaced, don't hoist the realize
                if first_replacements[concat_buffer] != stmt.buffer:
                    return stmt.body
                # Otherwise, do hoist it
                else:
                    concat_realize = realizes_by_buffer[concat_buffer]
                    new_realize = tvm.tir.BufferRealize(
                        concat_realize.buffer,
                        concat_realize.bounds,
                        concat_realize.condition,
                        stmt.body,
                        stmt.span,
                    )
                    return new_realize
        if isinstance(stmt, tvm.tir.AttrStmt):
            if isinstance(stmt.node, tvm.tir.Buffer) and stmt.node in buffer_replace_map:
                concat_buffer = buffer_replace_map[stmt.node].buffer
                # If this isn't the first buffer replaced, don't hoist the attrstmt
                if first_replacements[concat_buffer] != stmt.node:
                    return stmt.body
                # Otherwise, do hoist it
                else:
                    concat_attr = attrs_by_buffer[concat_buffer]
                    new_attr = tvm.tir.AttrStmt(
                        concat_attr.node,
                        concat_attr.attr_key,
                        concat_attr.value,
                        stmt.body,
                        stmt.span,
                    )
                    return new_attr

    def _ftransform(f, mod, ctx):
        f = f.with_body(
            tvm.tir.stmt_functor.ir_transform(
                f.body,
                _pre_remove,
                _post_remove,
                ["tir.AttrStmt", "tir.BufferLoad", "tir.BufferStore", "tir.BufferRealize"],
            )
        )
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(
                f.body,
                _pre_replace,
                _post_replace,
                ["tir.AttrStmt", "tir.BufferLoad", "tir.BufferStore", "tir.BufferRealize"],
            )
        )

    return tvm.tir.transform.prim_func_pass(
        _ftransform, opt_level=0, name="tir.contrib.ethos-u.remove_concatenates"
    )


def CreatePrimFuncWithoutConstants(const_dict):
    """
    This pass will remove arguments that are constants
    from PrimFunc Args. These should be replaced properly
    with tir.allocate_const when it becomes available.

    It also modifies the constant dictionary to
    rewrite the keys as the actual tir.Vars that are params
    rather than the index because this pass removes PrimFunc
    arguments that represent constants.
    """

    new_const_dict = dict()

    def _ftransform(f, mod, ctx):
        new_params = list()
        new_buffer_map = dict()
        new_preflattened_buffer_map = dict()
        for param_idx in const_dict.keys():
            # We are using buffer_var to key the constants as
            # PrimFunc params of constants will be removed.
            new_const_dict[f.buffer_map[f.params[param_idx]].data] = const_dict[param_idx]
        for i, param in enumerate(f.params):
            if i not in const_dict.keys():
                new_params.append(param)
                new_buffer_map[param] = f.buffer_map[param]
                if param in f.preflattened_buffer_map:
                    new_preflattened_buffer_map[param] = f.preflattened_buffer_map[param]
        return tvm.tir.PrimFunc(
            new_params,
            f.body,
            f.ret_type,
            new_buffer_map,
            new_preflattened_buffer_map,
            f.attrs,
            f.span,
        )

    def _create_primfunc_without_constants(mod):
        transform_func = tvm.tir.transform.prim_func_pass(
            _ftransform, opt_level=0, name="tir.contrib.ethos-u.CreatePrimFuncWithoutConstants"
        )
        mod = transform_func(mod)
        return mod, new_const_dict

    return _create_primfunc_without_constants


def HoistAllocates() -> tvm.IRModule:
    """
    Hoist allocate nodes up to the top of the body of the main function.

    Returns
    -------
    tvm.IRModule
        The new module with hoisted allocate nodes.
    """
    return _ffi_api.HoistAllocates()


def CopyComputeReordering(max_copy_movements: Optional[int] = None) -> tvm.IRModule:
    """
    Reorders copy and compute nodes in such a way that independent DMA copies,
    and computes happen in parallel.
    Copies to buffers with local scope are not reordered, indeed they copy LUT
    into the SHRAM which already happens in parallel with copying weights into
    the weights encoder.

    Parameters
    ----------
    max_copy_movements: Optional[int]
        The maximum number of movements allowed for a copy.
        If None, the pass context option
        tir.contrib.ethos-u.copy_compute_reordering_max_copy_movements
        is used if provided, otherwise the default value will be 1.

    Returns
    -------
    tvm.IRModule
        The new module with copy and compute nodes reordered.
    """
    return _ffi_api.CopyComputeReordering(max_copy_movements)
