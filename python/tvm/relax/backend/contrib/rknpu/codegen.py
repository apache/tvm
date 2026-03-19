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

"""RKNPU Python codegen — generates real NPU register command blobs.

Registered as ``relax.ext.rknpu``. Walks composite Relax functions, extracts
shapes from struct_info, creates AbstractMatmulTasks, and calls the vendored
regcmd generator to produce hardware register command sequences.

The regcmd blob, relocation table, and shape metadata are serialized into a
binary format and passed to the C++ runtime. At init time, the runtime patches
placeholder DMA addresses with real ones and submits the regcmds to the NPU.
"""

import json
import logging
import struct

import numpy as np

import tvm
from tvm import relax
from tvm.relax.analysis import get_var2val

from .npu_core.abstract import AbstractMatmulTask, AbstractConv2DTask, AbstractElementwiseTask, AbstractMaxPoolTask
from .npu_core.handles import TensorHandle
from .npu_core.regcmd_gen import RegCmdGenerator, compute_n_tile
from .npu_core.alignment import align_up, pad_m

logger = logging.getLogger(__name__)


# Placeholder DMA addresses used during regcmd generation.
# The runtime will patch these with real DMA addresses at init time.
PLACEHOLDER_INPUT = 0xAAAA0000
PLACEHOLDER_WEIGHT = 0xBBBB0000
PLACEHOLDER_OUTPUT = 0xCCCC0000
PLACEHOLDER_BIAS = 0xDDDD0000
PLACEHOLDER_RESIDUAL = 0xEEEE0000
PLACEHOLDER_INTERMEDIATE = 0xFFFF0000

# Relocation type codes (must match rknpu_runtime.cc).
RELOC_INPUT = 0
RELOC_WEIGHT = 1
RELOC_OUTPUT = 2
RELOC_BIAS = 3
RELOC_RESIDUAL = 4
RELOC_INTERMEDIATE = 5

# Regcmd binary format magic/version (must match rknpu_runtime.cc).
REGCMD_MAGIC = 0x504E4B52  # "RKNP" in little-endian
REGCMD_VERSION = 1     # v1: 40-byte header (matmul only)
REGCMD_VERSION_V2 = 2  # v2: 64-byte header (all op types)
REGCMD_VERSION_V3 = 3  # v3: 40-byte header + tile metadata (M-tiled matmul)
REGCMD_VERSION_V4 = 4  # v4: 40-byte header + 2D tile metadata (M+N tiled matmul)
REGCMD_VERSION_V5 = 5  # v5: 64-byte header + 2D tile metadata (N-tiled conv2d)
REGCMD_VERSION_V6 = 6  # v6: multi-task sequential (e.g. GELU = LUT + EW multiply)
REGCMD_VERSION_V7 = 7  # v7: multi-task with intermediates, constants, PC-chaining

# Operation type codes (must match rknpu_runtime.cc kOp* constants).
OP_MATMUL = 0
OP_CONV2D = 1
OP_ELEMENTWISE = 2
OP_MAXPOOL = 3
OP_DEPTHWISE_CONV2D = 4
OP_AVGPOOL = 5
OP_LUT = 6
OP_GELU = 7
OP_LAYER_NORM = 8

# Relocation types for V7 multi-intermediate/constant
RELOC_INTERMEDIATE_BASE = 5   # type = 5 + buffer_index
RELOC_CONSTANT_BASE = 16      # type = 16 + constant_index


def _get_matmul_shapes(func):
    """Extract M, K, N from a composite matmul function's struct_info.

    The composite function has two parameters (lhs, rhs) with shapes
    (..., M, K) and (..., K, N) respectively.

    Returns (M, K, N) or raises ValueError.
    """
    params = func.params
    if len(params) < 2:
        raise ValueError(f"Expected >=2 params for matmul, got {len(params)}")

    lhs_sinfo = params[0].struct_info
    rhs_sinfo = params[1].struct_info

    lhs_shape = [int(s) for s in lhs_sinfo.shape]
    rhs_shape = [int(s) for s in rhs_sinfo.shape]

    if len(lhs_shape) < 2 or len(rhs_shape) < 2:
        raise ValueError(
            f"Matmul requires 2D+ tensors, got lhs={lhs_shape}, rhs={rhs_shape}"
        )

    M = lhs_shape[-2]
    K = lhs_shape[-1]
    N = rhs_shape[-1]

    if rhs_shape[-2] != K:
        raise ValueError(f"Matmul K mismatch: lhs K={K}, rhs K={rhs_shape[-2]}")

    return M, K, N


def _get_conv2d_shapes(func):
    params = func.params
    if len(params) < 2:
        raise ValueError(f"Expected >=2 params for conv2d, got {len(params)}")

    inp_sinfo = params[0].struct_info
    wt_sinfo = params[1].struct_info

    inp_shape = [int(s) for s in inp_sinfo.shape]
    wt_shape = [int(s) for s in wt_sinfo.shape]

    if len(inp_shape) != 4 or len(wt_shape) != 4:
        raise ValueError(f"Conv2d requires 4D tensors, got input={inp_shape}, weight={wt_shape}")

    batch, C, H, W = inp_shape
    N, C_in, kH, kW = wt_shape

    # Extract stride and padding from the composite function body
    # Walk through the body to find the conv2d call and its attributes
    stride = 1
    pad_top = pad_bottom = pad_left = pad_right = 0

    bindings = get_var2val(func)
    for var, expr in bindings.items():
        if isinstance(expr, tvm.relax.Call):
            if hasattr(expr.op, 'name') and 'conv2d' in str(expr.op.name):
                if expr.attrs is not None:
                    strides = expr.attrs.strides
                    if strides:
                        stride = int(strides[0])
                    padding_attr = expr.attrs.padding
                    if padding_attr:
                        pads = [int(p) for p in padding_attr]
                        if len(pads) == 4:
                            pad_top, pad_left, pad_bottom, pad_right = pads
                        elif len(pads) == 2:
                            pad_top = pad_bottom = pads[0]
                            pad_left = pad_right = pads[1]
        elif isinstance(expr, tvm.relax.Function):
            # Inner composite function — recurse into its body
            inner_bindings = get_var2val(expr)
            for _, inner_expr in inner_bindings.items():
                if isinstance(inner_expr, tvm.relax.Call):
                    if hasattr(inner_expr.op, 'name') and 'conv2d' in str(inner_expr.op.name):
                        if inner_expr.attrs is not None:
                            strides = inner_expr.attrs.strides
                            if strides:
                                stride = int(strides[0])
                            padding_attr = inner_expr.attrs.padding
                            if padding_attr:
                                pads = [int(p) for p in padding_attr]
                                if len(pads) == 4:
                                    pad_top, pad_left, pad_bottom, pad_right = pads
                                elif len(pads) == 2:
                                    pad_top = pad_bottom = pads[0]
                                    pad_left = pad_right = pads[1]

    return C, H, W, N, kH, kW, stride, pad_top, pad_bottom, pad_left, pad_right


def _pack_regcmds(regcmds):
    """Serialize a list of regcmds to a bytes blob."""
    return b"".join(struct.pack("<Q", cmd.to_u64()) for cmd in regcmds)


def _pack_relocations(relocations):
    """Serialize relocation entries to a bytes blob."""
    return b"".join(struct.pack("<II", idx, rtype) for idx, rtype in relocations)


def _pack_tiled_body(tiles_data, relocations, include_n_fields=False):
    """Pack tile metadata + regcmd blobs + relocation table for tiled formats."""
    tile_meta = b""
    for tile in tiles_data:
        if include_n_fields:
            tile_meta += struct.pack("<IIIII",
                tile['m_offset'], tile['M_tile'],
                tile['n_offset'], tile['N_tile'],
                tile['num_regcmds'])
        else:
            tile_meta += struct.pack("<III",
                tile['m_offset'], tile['M_tile'], tile['num_regcmds'])
    regcmd_blob = b""
    for tile in tiles_data:
        regcmd_blob += _pack_regcmds(tile['regcmds'])
    reloc_blob = _pack_relocations(relocations)
    return tile_meta + regcmd_blob + reloc_blob


def _build_relocation_table(task):
    """Scan regcmds for placeholder DMA addresses and build a relocation table.

    Returns a list of (regcmd_index, reloc_type) tuples indicating which
    regcmds contain placeholder addresses that need runtime patching.
    """
    relocations = []
    for i, cmd in enumerate(task.regcmds):
        val = cmd.value & 0xFFFFFFFF
        if val == (PLACEHOLDER_INPUT & 0xFFFFFFFF):
            relocations.append((i, RELOC_INPUT))
        elif val == (PLACEHOLDER_WEIGHT & 0xFFFFFFFF):
            relocations.append((i, RELOC_WEIGHT))
        elif val == (PLACEHOLDER_OUTPUT & 0xFFFFFFFF):
            relocations.append((i, RELOC_OUTPUT))
        elif val == (PLACEHOLDER_BIAS & 0xFFFFFFFF):
            relocations.append((i, RELOC_BIAS))
        elif val == (PLACEHOLDER_INTERMEDIATE & 0xFFFFFFFF):
            relocations.append((i, RELOC_INTERMEDIATE))
    return relocations


def _serialize_regcmd_data(hw_task, relocations, M, K, N):
    """Serialize regcmd blob + relocation table + metadata into binary format.

    Binary layout (all little-endian):
        Header (40 bytes):
            [4] magic = 0x504E4B52 ("RKNP")
            [4] version = 1
            [4] M, [4] K, [4] N
            [4] num_regcmds
            [4] enable_mask, [4] int_mask, [4] regcfg_amount
            [4] num_relocations
        Body:
            [num_regcmds * 8] regcmd blob (uint64 LE)
            [num_relocations * 8] relocation entries:
                [4] regcmd_index (uint32), [4] type (uint32)
    """
    num_regcmds = len(hw_task.regcmds)
    num_relocations = len(relocations)

    # regcfg_amount = num_regcmds - 4 (the kernel driver adds 4 PC tail commands)
    regcfg_amount = num_regcmds - 4

    # Header
    header = struct.pack(
        "<IIIIIIIIII",
        REGCMD_MAGIC,
        REGCMD_VERSION,
        M, K, N,
        num_regcmds,
        hw_task.enable_mask,
        hw_task.int_mask,
        regcfg_amount,
        num_relocations,
    )

    return header + _pack_regcmds(hw_task.regcmds) + _pack_relocations(relocations)


def _serialize_regcmd_data_v2(hw_task, relocations, op_type, M, K, N,
                               C=0, H=0, W=0, H_out=0, W_out=0):
    """Serialize regcmd blob with v2 header (64 bytes) for non-matmul ops.

    Binary layout (all little-endian):
        Header (64 bytes):
            [4] magic = 0x504E4B52 ("RKNP")
            [4] version = 2
            [4] op_type (0=matmul, 1=conv2d, 2=elementwise, 3=maxpool)
            [4] M, [4] K, [4] N
            [4] num_regcmds
            [4] enable_mask, [4] int_mask, [4] regcfg_amount
            [4] num_relocations
            [4] C, [4] H, [4] W
            [4] H_out, [4] W_out
        Body:
            [num_regcmds * 8] regcmd blob (uint64 LE)
            [num_relocations * 8] relocation entries:
                [4] regcmd_index (uint32), [4] type (uint32)
    """
    num_regcmds = len(hw_task.regcmds)
    num_relocations = len(relocations)
    regcfg_amount = num_regcmds - 4

    header = struct.pack(
        "<IIIIIIIIIIIIIIII",
        REGCMD_MAGIC,
        REGCMD_VERSION_V2,
        op_type,
        M, K, N,
        num_regcmds,
        hw_task.enable_mask,
        hw_task.int_mask,
        regcfg_amount,
        num_relocations,
        C, H, W,
        H_out, W_out,
    )

    return header + _pack_regcmds(hw_task.regcmds) + _pack_relocations(relocations)


def _serialize_regcmd_data_v3(tiles_data, relocations, M_full, K, N,
                               enable_mask, int_mask, regcfg_amount):
    """Serialize multi-tile regcmd blob with v3 header for M-tiled matmul.

    Binary layout (all little-endian):
        Header (40 bytes = 10 uint32):
            [4] magic = 0x504E4B52 ("RKNP")
            [4] version = 3
            [4] M_full, [4] K, [4] N
            [4] num_tiles
            [4] enable_mask, [4] int_mask, [4] regcfg_amount
            [4] num_relocations
        Tile metadata (num_tiles × 12 bytes):
            Per tile: [4] m_offset, [4] M_tile, [4] num_regcmds
        Regcmd blobs (concatenated):
            Per tile: [num_regcmds × 8 bytes]
        Shared relocation table:
            [num_relocations × 8 bytes]
    """
    num_tiles = len(tiles_data)
    num_relocations = len(relocations)

    header = struct.pack(
        "<IIIIIIIIII",
        REGCMD_MAGIC,
        REGCMD_VERSION_V3,
        M_full, K, N,
        num_tiles,
        enable_mask, int_mask, regcfg_amount,
        num_relocations,
    )

    return header + _pack_tiled_body(tiles_data, relocations, include_n_fields=False)


def _serialize_regcmd_data_v4(tiles_data, relocations, M_full, K, N_full,
                               enable_mask, int_mask, regcfg_amount):
    """Serialize multi-tile regcmd blob with v4 header for M+N tiled matmul.

    Binary layout (all little-endian):
        Header (40 bytes = 10 uint32):
            [4] magic = 0x504E4B52 ("RKNP")
            [4] version = 4
            [4] M_full, [4] K, [4] N_full
            [4] num_tiles
            [4] enable_mask, [4] int_mask, [4] regcfg_amount
            [4] num_relocations
        Tile metadata (num_tiles × 20 bytes):
            Per tile: [4] m_offset, [4] M_tile, [4] n_offset, [4] N_tile, [4] num_regcmds
        Regcmd blobs (concatenated):
            Per tile: [num_regcmds × 8 bytes]
        Shared relocation table:
            [num_relocations × 8 bytes]
    """
    num_tiles = len(tiles_data)
    num_relocations = len(relocations)

    header = struct.pack(
        "<IIIIIIIIII",
        REGCMD_MAGIC,
        REGCMD_VERSION_V4,
        M_full, K, N_full,
        num_tiles,
        enable_mask, int_mask, regcfg_amount,
        num_relocations,
    )

    return header + _pack_tiled_body(tiles_data, relocations, include_n_fields=True)


def _serialize_regcmd_data_v5(tiles_data, relocations, op_type, M_full, K, N_full,
                               enable_mask, int_mask, regcfg_amount,
                               C=0, H=0, W=0, H_out=0, W_out=0):
    """Serialize multi-tile regcmd blob with v5 header for N-tiled conv2d.

    V5 = V2's 64-byte header (with op_type + spatial dims) + V4's tile metadata.
    The num_regcmds field position in V2 is repurposed as num_tiles.

    Binary layout (all little-endian):
        Header (64 bytes = 16 uint32):
            [4] magic, [4] version=5, [4] op_type
            [4] M_full, [4] K, [4] N_full
            [4] num_tiles
            [4] enable_mask, [4] int_mask, [4] regcfg_amount
            [4] num_relocations
            [4] C, [4] H, [4] W
            [4] H_out, [4] W_out
        Tile metadata (num_tiles × 20 bytes):
            Per tile: [4] m_offset, [4] M_tile, [4] n_offset, [4] N_tile, [4] num_regcmds
        Regcmd blobs (concatenated):
            Per tile: [num_regcmds × 8 bytes]
        Shared relocation table:
            [num_relocations × 8 bytes]
    """
    num_tiles = len(tiles_data)
    num_relocations = len(relocations)

    header = struct.pack(
        "<IIIIIIIIIIIIIIII",
        REGCMD_MAGIC,
        REGCMD_VERSION_V5,
        op_type,
        M_full, K, N_full,
        num_tiles,
        enable_mask, int_mask, regcfg_amount,
        num_relocations,
        C, H, W,
        H_out, W_out,
    )

    return header + _pack_tiled_body(tiles_data, relocations, include_n_fields=True)


def _build_json_graph(composite_name, input_shapes, input_dtypes, output_shape, output_dtype):
    """Build a minimal JSON graph string for JSONRuntimeBase.

    This produces a valid JSON graph that JSONRuntimeBase can parse,
    with input nodes and a kernel node for the composite operation.

    Each node needs shape/dtype attrs so that JSONRuntimeBase can correctly
    count input/output entries.
    """
    num_inputs = len(input_shapes)
    nodes = []
    arg_nodes = []
    node_row_ptr = [0]

    # Input nodes — each needs shape/dtype attrs for JSONRuntimeBase to count entries
    for i in range(num_inputs):
        nodes.append({
            "op": "input",
            "name": f"input_{i}",
            "attrs": {
                "shape": [list(input_shapes[i])],
                "dtype": [input_dtypes[i]],
            },
        })
        arg_nodes.append(i)
        node_row_ptr.append(i + 1)

    # Kernel node
    input_entries = [[i, 0, 0] for i in range(num_inputs)]
    nodes.append({
        "op": "kernel",
        "name": composite_name,
        "inputs": input_entries,
        "attrs": {
            "num_inputs": num_inputs,
            "num_outputs": 1,
            "shape": [list(output_shape)],
            "dtype": [output_dtype],
        },
    })
    kernel_idx = len(nodes) - 1
    node_row_ptr.append(kernel_idx + 1)

    # Graph output (head)
    heads = [[kernel_idx, 0, 0]]

    graph = {
        "nodes": nodes,
        "arg_nodes": arg_nodes,
        "heads": heads,
        "node_row_ptr": node_row_ptr,
    }
    return json.dumps(graph)


def _parse_matmul_composite(composite_name):
    """Parse composite name to determine relu and bias flags.

    Returns (has_relu, has_bias).
    """
    has_relu = "relu" in composite_name
    has_bias = "bias" in composite_name
    return has_relu, has_bias


def _compile_matmul(func, composite_name, constant_names):
    """Compile a single matmul composite function to regcmd blob + runtime module.

    Supports: rknpu.matmul, rknpu.matmul_relu, rknpu.matmul_bias,
    rknpu.matmul_bias_relu.

    Returns a TVM Module wrapping the RKNPURuntime.
    """
    has_relu, has_bias = _parse_matmul_composite(composite_name)
    M, K, N = _get_matmul_shapes(func)
    num_inputs = 3 if has_bias else 2

    logger.info(
        "RKNPU codegen: %s M=%d K=%d N=%d (relu=%s, bias=%s)",
        composite_name, M, K, N, has_relu, has_bias,
    )

    # Create abstract task (also serves as probe for tiling check)
    task = AbstractMatmulTask(
        op_name=composite_name,
        M=M,
        K=K,
        N=N,
        precision="float16",
        relu=has_relu,
        has_bias=has_bias,
        output_fp16=True,
    )

    # Check if tiling is needed
    m_tile_size = task.compute_m_tile_size()
    n_tile_size = compute_n_tile(N)
    needs_m_tiling = M > m_tile_size
    needs_n_tiling = N > n_tile_size

    if needs_n_tiling:
        return _compile_matmul_tiled_v4(func, composite_name, M, K, N, has_relu, has_bias,
                                         m_tile_size if needs_m_tiling else M,
                                         n_tile_size)
    if needs_m_tiling:
        return _compile_matmul_tiled(func, composite_name, M, K, N, has_relu, has_bias,
                                     m_tile_size)

    # Create placeholder tensor handles
    K_aligned = align_up(K, 32)
    N_aligned = align_up(N, 16)
    Mp = pad_m(M)

    input_handle = TensorHandle(
        name="input",
        shape=(M, K),
        dtype="float16",
        size_bytes=Mp * K_aligned * 2,
        dma_addr=PLACEHOLDER_INPUT,
    )
    weight_handle = TensorHandle(
        name="weight",
        shape=(K, N),
        dtype="float16",
        size_bytes=K_aligned * N_aligned * 2,
        dma_addr=PLACEHOLDER_WEIGHT,
    )
    output_handle = TensorHandle(
        name="output",
        shape=(M, N),
        dtype="float16",
        size_bytes=Mp * N_aligned * 2,
        dma_addr=PLACEHOLDER_OUTPUT,
    )

    # Create bias handle if needed (DPU_RDMA reads FP32 bias data)
    bias_handle = None
    if has_bias:
        bias_handle = TensorHandle(
            name="bias",
            shape=(N,),
            dtype="float32",
            size_bytes=N_aligned * 4,  # FP32 per-channel
            dma_addr=PLACEHOLDER_BIAS,
        )

    # Generate real register commands
    gen = RegCmdGenerator()
    hw_task = gen.generate_matmul(task, input_handle, weight_handle, output_handle,
                                  bias_handle=bias_handle)

    # Build relocation table
    relocations = _build_relocation_table(hw_task)

    logger.info(
        "RKNPU codegen: generated %d regcmds, %d relocations",
        len(hw_task.regcmds),
        len(relocations),
    )
    logger.info(
        "RKNPU codegen: enable_mask=0x%04X, int_mask=0x%04X",
        hw_task.enable_mask,
        hw_task.int_mask,
    )

    # Serialize regcmd binary data
    regcmd_data = _serialize_regcmd_data(hw_task, relocations, M, K, N)

    logger.info(
        "RKNPU codegen: serialized %d bytes (%d regcmds, %d relocations)",
        len(regcmd_data),
        len(hw_task.regcmds),
        len(relocations),
    )

    # Build JSON graph for JSONRuntimeBase
    input_shapes = [(M, K), (K, N)]
    input_dtypes = ["float16", "float16"]
    if has_bias:
        input_shapes.append((N,))
        input_dtypes.append("float16")
    graph_json = _build_json_graph(
        composite_name,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shape=(M, N),
        output_dtype="float16",
    )

    # No constant names for now (all inputs are runtime args)
    const_names = tvm.runtime.convert([])

    # Create runtime module via the C++ factory with regcmd data
    create_fn = tvm.get_global_func("runtime.rknpu_runtime_create")
    func_name = str(func.attrs["global_symbol"])
    runtime_mod = create_fn(func_name, graph_json, const_names, regcmd_data)

    return runtime_mod


def _compile_matmul_tiled(func, composite_name, M, K, N, has_relu, has_bias, m_tile_size):
    """Compile a tiled matmul for when M exceeds CBUF capacity.

    Splits the M dimension into tiles that each fit in CBUF. Each tile
    generates regcmds with standard placeholder DMA addresses; the C++
    runtime adds m_offset-based offsets when patching each tile.
    """
    K_aligned = align_up(K, 32)
    N_aligned = align_up(N, 16)

    gen = RegCmdGenerator()
    tiles_data = []
    shared_relocations = None
    shared_enable_mask = None
    shared_int_mask = None
    shared_regcfg_amount = None

    m_offset = 0
    while m_offset < M:
        M_tile = min(m_tile_size, M - m_offset)

        tile_task = AbstractMatmulTask(
            op_name=composite_name, M=M_tile, K=K, N=N,
            precision="float16", relu=has_relu, has_bias=has_bias, output_fp16=True,
        )
        tile_task.is_mtile = True
        tile_task.M_tile = M_tile
        tile_task.M_full = M
        tile_task.m_offset = 0  # Clean placeholders — actual offset in binary metadata

        Mp_tile = pad_m(M_tile)
        input_handle = TensorHandle(
            name="input", shape=(M_tile, K), dtype="float16",
            size_bytes=Mp_tile * K_aligned * 2, dma_addr=PLACEHOLDER_INPUT,
        )
        weight_handle = TensorHandle(
            name="weight", shape=(K, N), dtype="float16",
            size_bytes=K_aligned * N_aligned * 2, dma_addr=PLACEHOLDER_WEIGHT,
        )
        output_handle = TensorHandle(
            name="output", shape=(M_tile, N), dtype="float16",
            size_bytes=Mp_tile * N_aligned * 2, dma_addr=PLACEHOLDER_OUTPUT,
        )
        bias_handle = None
        if has_bias:
            bias_handle = TensorHandle(
                name="bias", shape=(N,), dtype="float32",
                size_bytes=N_aligned * 4, dma_addr=PLACEHOLDER_BIAS,
            )

        hw_task = gen.generate_matmul(tile_task, input_handle, weight_handle, output_handle,
                                       bias_handle=bias_handle)

        # Build relocations from first tile (same for all tiles)
        if shared_relocations is None:
            shared_relocations = _build_relocation_table(hw_task)
            shared_enable_mask = hw_task.enable_mask
            shared_int_mask = hw_task.int_mask
            shared_regcfg_amount = len(hw_task.regcmds) - 4

        tiles_data.append({
            'm_offset': m_offset,
            'M_tile': M_tile,
            'num_regcmds': len(hw_task.regcmds),
            'regcmds': hw_task.regcmds,
        })

        m_offset += m_tile_size

    logger.info(
        "RKNPU codegen: M-tiled matmul %d tiles (m_tile_size=%d, M=%d)",
        len(tiles_data), m_tile_size, M,
    )

    # Serialize with V3 format
    regcmd_data = _serialize_regcmd_data_v3(
        tiles_data, shared_relocations,
        M_full=M, K=K, N=N,
        enable_mask=shared_enable_mask,
        int_mask=shared_int_mask,
        regcfg_amount=shared_regcfg_amount,
    )

    # Build JSON graph — use full M (all tiles share buffers)
    input_shapes = [(M, K), (K, N)]
    input_dtypes = ["float16", "float16"]
    if has_bias:
        input_shapes.append((N,))
        input_dtypes.append("float16")
    graph_json = _build_json_graph(
        composite_name,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shape=(M, N),
        output_dtype="float16",
    )

    const_names = tvm.runtime.convert([])
    create_fn = tvm.get_global_func("runtime.rknpu_runtime_create")
    func_name = str(func.attrs["global_symbol"])
    runtime_mod = create_fn(func_name, graph_json, const_names, regcmd_data)
    return runtime_mod


def _compile_matmul_tiled_v4(func, composite_name, M, K, N, has_relu, has_bias,
                              m_tile_size, n_tile_size):
    """Compile a 2D-tiled matmul for when N (and optionally M) exceeds hardware limits.

    Splits both M and N dimensions into tiles. Each tile generates regcmds with
    standard placeholder DMA addresses; the C++ runtime adds m_offset and n_offset
    based offsets when patching each tile.
    """
    K_aligned = align_up(K, 32)

    gen = RegCmdGenerator()
    tiles_data = []
    shared_relocations = None
    shared_enable_mask = None
    shared_int_mask = None
    shared_regcfg_amount = None

    for m_off in range(0, M, m_tile_size):
        M_tile = min(m_tile_size, M - m_off)
        for n_off in range(0, N, n_tile_size):
            N_tile = min(n_tile_size, N - n_off)
            N_tile_aligned = align_up(N_tile, 16)
            Mp_tile = pad_m(M_tile)

            tile_task = AbstractMatmulTask(
                op_name=composite_name, M=M_tile, K=K, N=N_tile,
                precision="float16", relu=has_relu, has_bias=has_bias, output_fp16=True,
            )
            # Set M-tiling attrs if M is tiled
            if M_tile < M:
                tile_task.is_mtile = True
                tile_task.M_tile = M_tile
                tile_task.M_full = M
                tile_task.m_offset = 0
            # Set N-tiling attrs
            tile_task.is_ntile = True
            tile_task.N_tile = N_tile

            input_handle = TensorHandle(
                name="input", shape=(M_tile, K), dtype="float16",
                size_bytes=Mp_tile * K_aligned * 2, dma_addr=PLACEHOLDER_INPUT,
            )
            weight_handle = TensorHandle(
                name="weight", shape=(K, N_tile), dtype="float16",
                size_bytes=K_aligned * N_tile_aligned * 2, dma_addr=PLACEHOLDER_WEIGHT,
            )
            output_handle = TensorHandle(
                name="output", shape=(M_tile, N_tile), dtype="float16",
                size_bytes=Mp_tile * N_tile_aligned * 2, dma_addr=PLACEHOLDER_OUTPUT,
            )
            bias_handle = None
            if has_bias:
                bias_handle = TensorHandle(
                    name="bias", shape=(N_tile,), dtype="float32",
                    size_bytes=N_tile_aligned * 4, dma_addr=PLACEHOLDER_BIAS,
                )

            hw_task = gen.generate_matmul(tile_task, input_handle, weight_handle, output_handle,
                                           bias_handle=bias_handle)

            if shared_relocations is None:
                shared_relocations = _build_relocation_table(hw_task)
                shared_enable_mask = hw_task.enable_mask
                shared_int_mask = hw_task.int_mask
                shared_regcfg_amount = len(hw_task.regcmds) - 4

            tiles_data.append({
                'm_offset': m_off,
                'M_tile': M_tile,
                'n_offset': n_off,
                'N_tile': N_tile,
                'num_regcmds': len(hw_task.regcmds),
                'regcmds': hw_task.regcmds,
            })

    logger.info(
        "RKNPU codegen: V4 tiled matmul %d tiles (m_tile=%d, n_tile=%d, M=%d, N=%d)",
        len(tiles_data), m_tile_size, n_tile_size, M, N,
    )

    # Serialize with V4 format
    regcmd_data = _serialize_regcmd_data_v4(
        tiles_data, shared_relocations,
        M_full=M, K=K, N_full=N,
        enable_mask=shared_enable_mask,
        int_mask=shared_int_mask,
        regcfg_amount=shared_regcfg_amount,
    )

    # Build JSON graph — use full M and N (all tiles share buffers)
    input_shapes = [(M, K), (K, N)]
    input_dtypes = ["float16", "float16"]
    if has_bias:
        input_shapes.append((N,))
        input_dtypes.append("float16")
    graph_json = _build_json_graph(
        composite_name,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shape=(M, N),
        output_dtype="float16",
    )

    const_names = tvm.runtime.convert([])
    create_fn = tvm.get_global_func("runtime.rknpu_runtime_create")
    func_name = str(func.attrs["global_symbol"])
    runtime_mod = create_fn(func_name, graph_json, const_names, regcmd_data)
    return runtime_mod


def _compile_conv2d_tiled_v5(func, composite_name, C, H, W, N, kH, kW, stride,
                              pad_top, pad_bottom, pad_left, pad_right,
                              has_relu, has_bias, n_tile_size):
    """Compile an N-tiled conv2d using V5 format.

    Splits the N (output channels) dimension into tiles. Each tile generates
    its own regcmds via generate_conv_mode0. Conv2d doesn't need external
    M-tiling because fill_conv_mode0_descriptors handles CBUF overflow
    internally via automatic H-tiling.
    """
    C_al = align_up(C, 32)

    gen = RegCmdGenerator()
    tiles_data = []
    shared_relocations = None
    shared_enable_mask = None
    shared_int_mask = None
    shared_regcfg_amount = None

    # Compute full-size task properties for header
    full_task = AbstractConv2DTask(
        op_name=composite_name, C=C, H=H, W=W, N=N,
        kH=kH, kW=kW, stride=stride,
        pad_top=pad_top, pad_bottom=pad_bottom,
        pad_left=pad_left, pad_right=pad_right,
        relu=has_relu, has_bias=has_bias,
    )
    H_out = full_task.H_out
    W_out = full_task.W_out
    M_full = full_task.M
    K_eff = full_task.K_eff

    for n_off in range(0, N, n_tile_size):
        N_tile = min(n_tile_size, N - n_off)
        N_tile_al = align_up(N_tile, 16)
        K_eff_al = align_up(K_eff, 32)

        tile_task = AbstractConv2DTask(
            op_name=composite_name, C=C, H=H, W=W, N=N_tile,
            kH=kH, kW=kW, stride=stride,
            pad_top=pad_top, pad_bottom=pad_bottom,
            pad_left=pad_left, pad_right=pad_right,
            relu=has_relu, has_bias=has_bias,
        )

        Mp_tile = tile_task.M_padded

        input_handle = TensorHandle(
            name="input", shape=(C, H, W), dtype="float16",
            size_bytes=C_al * H * W * 2, dma_addr=PLACEHOLDER_INPUT,
        )
        weight_handle = TensorHandle(
            name="weight", shape=(N_tile, C, kH, kW), dtype="float16",
            size_bytes=K_eff_al * N_tile_al * 2, dma_addr=PLACEHOLDER_WEIGHT,
        )
        output_handle = TensorHandle(
            name="output", shape=(N_tile, H_out, W_out), dtype="float16",
            size_bytes=N_tile_al * Mp_tile * 2, dma_addr=PLACEHOLDER_OUTPUT,
        )
        bias_handle = None
        if has_bias:
            bias_handle = TensorHandle(
                name="bias", shape=(N_tile,), dtype="float32",
                size_bytes=N_tile_al * 4, dma_addr=PLACEHOLDER_BIAS,
            )

        hw_task = gen.generate_conv_mode0(tile_task, input_handle, weight_handle, output_handle,
                                           bias_handle=bias_handle)

        if shared_relocations is None:
            shared_relocations = _build_relocation_table(hw_task)
            shared_enable_mask = hw_task.enable_mask
            shared_int_mask = hw_task.int_mask
            shared_regcfg_amount = len(hw_task.regcmds) - 4

        tiles_data.append({
            'm_offset': 0,
            'M_tile': M_full,
            'n_offset': n_off,
            'N_tile': N_tile,
            'num_regcmds': len(hw_task.regcmds),
            'regcmds': hw_task.regcmds,
        })

    logger.info(
        "RKNPU codegen: V5 tiled conv2d %d tiles (n_tile=%d, N=%d)",
        len(tiles_data), n_tile_size, N,
    )

    # Serialize with V5 format
    regcmd_data = _serialize_regcmd_data_v5(
        tiles_data, shared_relocations,
        op_type=OP_CONV2D,
        M_full=M_full, K=K_eff, N_full=N,
        enable_mask=shared_enable_mask,
        int_mask=shared_int_mask,
        regcfg_amount=shared_regcfg_amount,
        C=C, H=H, W=W, H_out=H_out, W_out=W_out,
    )

    # Build JSON graph — use full N
    input_shapes = [(1, C, H, W), (N, C, kH, kW)]
    input_dtypes = ["float16", "float16"]
    if has_bias:
        input_shapes.append((N,))
        input_dtypes.append("float16")

    graph_json = _build_json_graph(
        composite_name,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shape=(1, N, H_out, W_out),
        output_dtype="float16",
    )

    const_names = tvm.runtime.convert([])
    create_fn = tvm.get_global_func("runtime.rknpu_runtime_create")
    func_name = str(func.attrs["global_symbol"])
    runtime_mod = create_fn(func_name, graph_json, const_names, regcmd_data)
    return runtime_mod


def _compile_conv2d_common(func, composite_name, constant_names, is_depthwise=False):
    """Compile a conv2d or depthwise conv2d subgraph."""
    has_relu = "relu" in composite_name
    has_bias = "bias" in composite_name
    C, H, W, N, kH, kW, stride, pad_top, pad_bottom, pad_left, pad_right = _get_conv2d_shapes(func)

    if is_depthwise:
        assert N == C, f"Depthwise conv2d requires N == C, got N={N}, C={C}"
        op_type = OP_DEPTHWISE_CONV2D
        task_kwargs = dict(is_depthwise=True, conv_mode=3, groups=C)
        label = "depthwise conv2d"
    else:
        op_type = OP_CONV2D
        task_kwargs = {}
        label = "conv2d"

    logger.info(
        "RKNPU codegen: %s %s C=%d H=%d W=%d N=%d k=%dx%d s=%d (relu=%s, bias=%s)",
        label, composite_name, C, H, W, N, kH, kW, stride, has_relu, has_bias,
    )

    # Check if N-tiling is needed (only for standard conv2d)
    if not is_depthwise:
        n_tile_size = compute_n_tile(N)
        if N > n_tile_size:
            return _compile_conv2d_tiled_v5(
                func, composite_name, C, H, W, N, kH, kW, stride,
                pad_top, pad_bottom, pad_left, pad_right,
                has_relu, has_bias, n_tile_size,
            )

    task = AbstractConv2DTask(
        op_name=composite_name,
        C=C, H=H, W=W, N=N,
        kH=kH, kW=kW, stride=stride,
        pad_top=pad_top, pad_bottom=pad_bottom,
        pad_left=pad_left, pad_right=pad_right,
        relu=has_relu,
        has_bias=has_bias,
        **task_kwargs,
    )

    H_out = task.H_out
    W_out = task.W_out
    Mp = task.M_padded
    K_eff = task.K_eff
    K_eff_al = align_up(K_eff, 32)
    C_al = align_up(C, 32)

    # Alignment and sizes differ for depthwise
    if is_depthwise:
        N_al = C_al  # 32-aligned for depthwise
        weight_size = K_eff_al * 2  # flat layout
        weight_shape = (C, 1, kH, kW)
    else:
        N_al = align_up(N, 16)
        weight_size = K_eff_al * N_al * 2
        weight_shape = (N, C, kH, kW)

    input_handle = TensorHandle(
        name="input",
        shape=(C, H, W),
        dtype="float16",
        size_bytes=C_al * H * W * 2,
        dma_addr=PLACEHOLDER_INPUT,
    )
    weight_handle = TensorHandle(
        name="weight",
        shape=weight_shape,
        dtype="float16",
        size_bytes=weight_size,
        dma_addr=PLACEHOLDER_WEIGHT,
    )
    output_handle = TensorHandle(
        name="output",
        shape=(N, H_out, W_out),
        dtype="float16",
        size_bytes=N_al * Mp * 2,
        dma_addr=PLACEHOLDER_OUTPUT,
    )

    bias_handle = None
    if has_bias:
        bias_al = N_al  # C_al for depthwise, N_al(16) for standard
        bias_handle = TensorHandle(
            name="bias",
            shape=(N,),
            dtype="float32",
            size_bytes=bias_al * 4,
            dma_addr=PLACEHOLDER_BIAS,
        )

    gen = RegCmdGenerator()
    hw_task = gen.generate_conv_mode0(task, input_handle, weight_handle, output_handle,
                                       bias_handle=bias_handle)

    relocations = _build_relocation_table(hw_task)

    logger.info(
        "RKNPU codegen: generated %d regcmds, %d relocations",
        len(hw_task.regcmds), len(relocations),
    )

    regcmd_data = _serialize_regcmd_data_v2(
        hw_task, relocations,
        op_type=op_type,
        M=task.M, K=K_eff, N=N,
        C=C, H=H, W=W, H_out=H_out, W_out=W_out,
    )

    # Build JSON graph
    input_shapes = [(1, C, H, W), weight_shape]
    input_dtypes = ["float16", "float16"]
    if has_bias:
        input_shapes.append((N,))
        input_dtypes.append("float16")

    graph_json = _build_json_graph(
        composite_name,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shape=(1, N, H_out, W_out),
        output_dtype="float16",
    )

    const_names = tvm.runtime.convert([])
    create_fn = tvm.get_global_func("runtime.rknpu_runtime_create")
    func_name = str(func.attrs["global_symbol"])
    runtime_mod = create_fn(func_name, graph_json, const_names, regcmd_data)
    return runtime_mod


def _compile_elementwise(func, composite_name, constant_names):
    params = func.params
    if len(params) < 1:
        raise ValueError(f"Expected >=1 params for elementwise, got {len(params)}")

    n_inputs = len(params)
    lhs_sinfo = params[0].struct_info
    lhs_shape = [int(s) for s in lhs_sinfo.shape]

    op_type = "add" if "add" in composite_name else "mul"

    logger.info("RKNPU codegen: elementwise %s shape=%s n_inputs=%d", op_type, lhs_shape, n_inputs)

    # Treat as (H, C) = (shape[0], shape[1]) for 2D tensors
    if len(lhs_shape) == 2:
        H, C = lhs_shape
    elif len(lhs_shape) == 1:
        H, C = 1, lhs_shape[0]
    else:
        raise ValueError(f"Elementwise expects 1D or 2D, got shape={lhs_shape}")

    task = AbstractElementwiseTask(
        op_name=composite_name,
        op_type=op_type,
        n_inputs=2,
        shape=tuple(lhs_shape),
    )

    C_al = align_up(C, 16)
    Mp = pad_m(H)
    buf_size = C_al * Mp * 2

    input_a = TensorHandle(name="input_a", shape=tuple(lhs_shape), dtype="float16",
                           size_bytes=buf_size, dma_addr=PLACEHOLDER_INPUT)
    input_b = TensorHandle(name="input_b", shape=tuple(lhs_shape), dtype="float16",
                           size_bytes=buf_size, dma_addr=PLACEHOLDER_WEIGHT)
    output = TensorHandle(name="output", shape=tuple(lhs_shape), dtype="float16",
                          size_bytes=buf_size, dma_addr=PLACEHOLDER_OUTPUT)

    gen = RegCmdGenerator()
    hw_task = gen.generate_elementwise(task, input_a, input_b, output)
    relocations = _build_relocation_table(hw_task)

    # Use v2 header: M=H, K=C, N=C (all three buffers share same layout)
    regcmd_data = _serialize_regcmd_data_v2(
        hw_task, relocations,
        op_type=OP_ELEMENTWISE,
        M=H, K=C, N=C,
    )

    # n_inputs may be 1 when both operands are the same Var (e.g., multiply(x, x)).
    # The runtime handles this by scattering the single input to both buffers.
    input_shapes = [tuple(lhs_shape)] * n_inputs
    input_dtypes = ["float16"] * n_inputs
    graph_json = _build_json_graph(
        composite_name,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shape=tuple(lhs_shape),
        output_dtype="float16",
    )

    const_names = tvm.runtime.convert([])
    create_fn = tvm.get_global_func("runtime.rknpu_runtime_create")
    func_name = str(func.attrs["global_symbol"])
    runtime_mod = create_fn(func_name, graph_json, const_names, regcmd_data)
    return runtime_mod


# LUT function name → (le_table, lo_table, lut_params) mapping
_LUT_FUNCTIONS = None  # Lazy-loaded to avoid import cost

def _get_lut_config(func_name):
    """Get LUT tables and activation-specific params for a given function."""
    global _LUT_FUNCTIONS
    if _LUT_FUNCTIONS is None:
        from .npu_core.lut_tables import (
            EXP_LE_TABLE, EXP_LO_TABLE, EXP_LUT_PARAMS,
            SIGMOID_LE_TABLE, SIGMOID_LO_TABLE, SIGMOID_LUT_PARAMS,
            RSQRT_LE_TABLE, RSQRT_LO_TABLE, RSQRT_LUT_PARAMS,
            RECIPROCAL_LE_TABLE, RECIPROCAL_LO_TABLE, RECIPROCAL_LUT_PARAMS,
        )
        _LUT_FUNCTIONS = {
            "exp": (EXP_LE_TABLE, EXP_LO_TABLE, EXP_LUT_PARAMS),
            "sigmoid": (SIGMOID_LE_TABLE, SIGMOID_LO_TABLE, SIGMOID_LUT_PARAMS),
            "rsqrt": (RSQRT_LE_TABLE, RSQRT_LO_TABLE, RSQRT_LUT_PARAMS),
            "reciprocal": (RECIPROCAL_LE_TABLE, RECIPROCAL_LO_TABLE, RECIPROCAL_LUT_PARAMS),
        }
    return _LUT_FUNCTIONS[func_name]


def _compile_lut(func, composite_name, constant_names):
    """Compile a LUT-based activation function (exp, sigmoid, etc.).

    Generates a single combined task that first uploads LUT tables to DPU SRAM,
    then processes input data through the LUT. The PC processes upload regcmds
    (filling SRAM) then eval regcmds (data processing) sequentially. Only the
    eval triggers the DPU completion interrupt.
    """
    params = func.params
    if len(params) < 1:
        raise ValueError(f"Expected >=1 params for LUT op, got {len(params)}")

    inp_sinfo = params[0].struct_info
    inp_shape = [int(s) for s in inp_sinfo.shape]

    # Determine which LUT function
    if "exp" in composite_name:
        lut_name = "exp"
    elif "sigmoid" in composite_name:
        lut_name = "sigmoid"
    elif "rsqrt" in composite_name:
        lut_name = "rsqrt"
    elif "reciprocal" in composite_name:
        lut_name = "reciprocal"
    else:
        raise ValueError(f"Unsupported LUT composite: {composite_name}")

    le_table, lo_table, lut_params = _get_lut_config(lut_name)

    logger.info("RKNPU codegen: LUT %s shape=%s", lut_name, inp_shape)

    if len(inp_shape) == 2:
        H, C = inp_shape
    elif len(inp_shape) == 1:
        H, C = 1, inp_shape[0]
    else:
        raise ValueError(f"LUT expects 1D or 2D, got shape={inp_shape}")

    gen = RegCmdGenerator()

    # Generate combined upload+eval task with placeholder DMA addresses
    hw_task = gen.generate_lut_combined_task(
        le_table, lo_table,
        shape=tuple(inp_shape),
        src_dma=PLACEHOLDER_INPUT,
        dst_dma=PLACEHOLDER_OUTPUT,
        lut_params=lut_params,
    )

    relocations = _build_relocation_table(hw_task)

    # Use V2 format — same as elementwise (single task, same scatter/gather)
    regcmd_data = _serialize_regcmd_data_v2(
        hw_task, relocations,
        op_type=OP_LUT,
        M=H, K=C, N=C,
    )

    graph_json = _build_json_graph(
        composite_name,
        input_shapes=[tuple(inp_shape)],
        input_dtypes=["float16"],
        output_shape=tuple(inp_shape),
        output_dtype="float16",
    )

    const_names = tvm.runtime.convert([])
    create_fn = tvm.get_global_func("runtime.rknpu_runtime_create")
    func_name = str(func.attrs["global_symbol"])
    runtime_mod = create_fn(func_name, graph_json, const_names, regcmd_data)
    return runtime_mod


def _serialize_regcmd_data_v6(sub_tasks, M, K, N):
    """Serialize multi-task sequential regcmd data (V6 format).

    Used for GELU (LUT sigmoid + EW multiply) and other multi-step ops.

    Binary layout (all little-endian):
        Header (28 bytes = 7 uint32):
            [4] magic = "RKNP"
            [4] version = 6
            [4] op_type (7=GELU)
            [4] M, [4] K, [4] N
            [4] num_sub_tasks
        Per sub-task:
            [4] num_regcmds
            [4] enable_mask, [4] int_mask, [4] regcfg_amount
            [4] num_relocations
            [num_regcmds * 8] regcmd blob
            [num_relocations * 8] relocation entries
    """
    header = struct.pack(
        "<IIIIIII",
        REGCMD_MAGIC,
        REGCMD_VERSION_V6,
        OP_GELU,
        M, K, N,
        len(sub_tasks),
    )

    body = b""
    for hw_task, relocations in sub_tasks:
        num_regcmds = len(hw_task.regcmds)
        num_relocations = len(relocations)
        regcfg_amount = num_regcmds - 4  # PC adds 4

        task_header = struct.pack(
            "<IIIII",
            num_regcmds,
            hw_task.enable_mask,
            hw_task.int_mask,
            regcfg_amount,
            num_relocations,
        )
        body += task_header
        body += _pack_regcmds(hw_task.regcmds)
        body += _pack_relocations(relocations)

    return header + body


def _serialize_regcmd_data_v7(sub_tasks, M, K, N, op_type,
                              intermediate_sizes=None, constants=None):
    """Serialize multi-task regcmd data with intermediates and constants (V7 format).

    Used for LayerNorm and other complex composites with PC-chaining.

    Binary layout (all little-endian):
        Header (32 bytes = 8 uint32):
            [4] magic = "RKNP"
            [4] version = 7
            [4] op_type
            [4] M, [4] K, [4] N
            [4] num_sub_tasks
            [4] num_intermediates
        Intermediate buffer table (num_intermediates * 4 bytes):
            [4] buffer_size_bytes each
        Constant data section:
            [4] num_constants
            Per constant: [4] size, [size] data
        Per sub-task (same as V6):
            [4] num_regcmds
            [4] enable_mask, [4] int_mask, [4] regcfg_amount
            [4] num_relocations
            [num_regcmds * 8] regcmd blob
            [num_relocations * 8] relocation entries
    """
    if intermediate_sizes is None:
        intermediate_sizes = []
    if constants is None:
        constants = []

    num_intermediates = len(intermediate_sizes)
    num_constants = len(constants)

    header = struct.pack(
        "<IIIIIIII",
        REGCMD_MAGIC,
        REGCMD_VERSION_V7,
        op_type,
        M, K, N,
        len(sub_tasks),
        num_intermediates,
    )

    # Intermediate buffer sizes.
    intermediate_data = b""
    for size in intermediate_sizes:
        intermediate_data += struct.pack("<I", size)

    # Constant data section.
    const_data = struct.pack("<I", num_constants)
    for const_bytes in constants:
        const_data += struct.pack("<I", len(const_bytes))
        const_data += const_bytes

    # Sub-tasks (same format as V6).
    body = b""
    for hw_task, relocations in sub_tasks:
        num_regcmds = len(hw_task.regcmds)
        num_relocations = len(relocations)
        regcfg_amount = num_regcmds - 4  # PC adds 4

        task_header = struct.pack(
            "<IIIII",
            num_regcmds,
            hw_task.enable_mask,
            hw_task.int_mask,
            regcfg_amount,
            num_relocations,
        )
        body += task_header
        body += _pack_regcmds(hw_task.regcmds)
        body += _pack_relocations(relocations)

    return header + intermediate_data + const_data + body


def _compile_gelu(func, composite_name, constant_names):
    """Compile GELU = x * sigmoid(1.702 * x) as two NPU tasks.

    Task 1: Combined LUT (sigmoid with GELU prescale) → intermediate buffer
    Task 2: Elementwise multiply (x, intermediate) → output buffer
    """
    params = func.params
    if len(params) < 1:
        raise ValueError(f"Expected >=1 params for GELU, got {len(params)}")

    inp_sinfo = params[0].struct_info
    inp_shape = [int(s) for s in inp_sinfo.shape]

    logger.info("RKNPU codegen: GELU shape=%s", inp_shape)

    if len(inp_shape) == 2:
        H, C = inp_shape
    elif len(inp_shape) == 1:
        H, C = 1, inp_shape[0]
    else:
        raise ValueError(f"GELU expects 1D or 2D, got shape={inp_shape}")

    from .npu_core.lut_tables import (
        SIGMOID_LE_TABLE, SIGMOID_LO_TABLE, GELU_LUT_PARAMS,
    )

    gen = RegCmdGenerator()

    # Task 1: LUT sigmoid(1.702*x) → intermediate buffer
    # Uses GELU_LUT_PARAMS with bn_mul_cfg=0x6C500000 (absorbs 1.702 factor)
    lut_task = gen.generate_lut_combined_task(
        SIGMOID_LE_TABLE, SIGMOID_LO_TABLE,
        shape=tuple(inp_shape),
        src_dma=PLACEHOLDER_INPUT,
        dst_dma=PLACEHOLDER_INTERMEDIATE,
        lut_params=GELU_LUT_PARAMS,
    )
    lut_relocs = _build_relocation_table(lut_task)

    # Task 2: EW multiply x * sigmoid(1.702*x) → output
    ew_task_obj = AbstractElementwiseTask(
        op_name="gelu_mul",
        op_type="mul",
        n_inputs=2,
        shape=tuple(inp_shape),
    )

    C_al = align_up(C, 16)
    Mp = pad_m(H)
    buf_size = C_al * Mp * 2

    input_a = TensorHandle(name="input_a", shape=tuple(inp_shape), dtype="float16",
                           size_bytes=buf_size, dma_addr=PLACEHOLDER_INPUT)
    input_b = TensorHandle(name="input_b", shape=tuple(inp_shape), dtype="float16",
                           size_bytes=buf_size, dma_addr=PLACEHOLDER_INTERMEDIATE)
    output = TensorHandle(name="output", shape=tuple(inp_shape), dtype="float16",
                          size_bytes=buf_size, dma_addr=PLACEHOLDER_OUTPUT)

    ew_task = gen.generate_elementwise(ew_task_obj, input_a, input_b, output)
    ew_relocs = _build_relocation_table(ew_task)

    # Serialize as V6 (two sequential tasks)
    regcmd_data = _serialize_regcmd_data_v6(
        [(lut_task, lut_relocs), (ew_task, ew_relocs)],
        M=H, K=C, N=C,
    )

    graph_json = _build_json_graph(
        composite_name,
        input_shapes=[tuple(inp_shape)],
        input_dtypes=["float16"],
        output_shape=tuple(inp_shape),
        output_dtype="float16",
    )

    const_names = tvm.runtime.convert([])
    create_fn = tvm.get_global_func("runtime.rknpu_runtime_create")
    func_name = str(func.attrs["global_symbol"])
    runtime_mod = create_fn(func_name, graph_json, const_names, regcmd_data)
    return runtime_mod


def _get_pool_shapes(func, pool_op_name):
    """Extract pool parameters from a composite pool function.

    Returns (batch, C, H, W, kH, kW, stride_h, stride_w,
             pad_top, pad_bottom, pad_left, pad_right).
    """
    params = func.params
    if len(params) < 1:
        raise ValueError(f"Expected >=1 params for pool, got {len(params)}")

    inp_sinfo = params[0].struct_info
    inp_shape = [int(s) for s in inp_sinfo.shape]

    if len(inp_shape) != 4:
        raise ValueError(f"Pool requires 4D input, got shape={inp_shape}")

    batch, C, H, W = inp_shape

    kH = kW = 2
    stride_h = stride_w = 2
    pad_top = pad_bottom = pad_left = pad_right = 0

    bindings = get_var2val(func)
    for var, expr in bindings.items():
        if isinstance(expr, tvm.relax.Function):
            inner_bindings = get_var2val(expr)
            for _, inner_expr in inner_bindings.items():
                if isinstance(inner_expr, tvm.relax.Call):
                    if hasattr(inner_expr.op, 'name') and pool_op_name in str(inner_expr.op.name):
                        if inner_expr.attrs is not None:
                            pool_size = inner_expr.attrs.pool_size
                            if pool_size:
                                kH, kW = int(pool_size[0]), int(pool_size[1])
                            strides = inner_expr.attrs.strides
                            if strides:
                                stride_h, stride_w = int(strides[0]), int(strides[1])
                            padding_attr = inner_expr.attrs.padding
                            if padding_attr:
                                pads = [int(p) for p in padding_attr]
                                if len(pads) == 4:
                                    pad_top, pad_left, pad_bottom, pad_right = pads
                                elif len(pads) == 2:
                                    pad_top = pad_bottom = pads[0]
                                    pad_left = pad_right = pads[1]
        elif isinstance(expr, tvm.relax.Call):
            if hasattr(expr.op, 'name') and pool_op_name in str(expr.op.name):
                if expr.attrs is not None:
                    pool_size = expr.attrs.pool_size
                    if pool_size:
                        kH, kW = int(pool_size[0]), int(pool_size[1])
                    strides = expr.attrs.strides
                    if strides:
                        stride_h, stride_w = int(strides[0]), int(strides[1])
                    padding_attr = expr.attrs.padding
                    if padding_attr:
                        pads = [int(p) for p in padding_attr]
                        if len(pads) == 4:
                            pad_top, pad_left, pad_bottom, pad_right = pads
                        elif len(pads) == 2:
                            pad_top = pad_bottom = pads[0]
                            pad_left = pad_right = pads[1]

    return batch, C, H, W, kH, kW, stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right


def _compile_pool(func, composite_name, constant_names, is_avg=False):
    """Compile a max or avg pool2d subgraph.

    Max pool uses the PPU hardware directly (method=1).
    Avg pool is implemented as a depthwise conv2d with constant weights
    (all 1/(kH*kW)) since the PPU only supports max and min pooling.
    """
    pool_op_name = 'avg_pool' if is_avg else 'max_pool'
    (batch, C, H, W, kH, kW, stride_h, stride_w,
     pool_pad_top, pool_pad_bottom, pool_pad_left, pool_pad_right) = _get_pool_shapes(func, pool_op_name)

    label = "avgpool" if is_avg else "maxpool"
    logger.info("RKNPU codegen: %s C=%d H=%d W=%d k=%dx%d s=%dx%d",
                label, C, H, W, kH, kW, stride_h, stride_w)

    if is_avg:
        # Avg pool = depthwise conv2d with uniform weights 1/(kH*kW).
        # PPU hardware only supports max/min, so we use the CNA/DPU path.
        return _compile_avgpool_as_dwconv(
            func, composite_name, batch, C, H, W, kH, kW,
            stride_h, pool_pad_top, pool_pad_bottom, pool_pad_left, pool_pad_right,
        )

    # Max pool uses PPU hardware
    task = AbstractMaxPoolTask(
        op_name=composite_name,
        C=C, H=H, W=W,
        kH=kH, kW=kW,
        stride_h=stride_h, stride_w=stride_w,
        pad_top=pool_pad_top, pad_bottom=pool_pad_bottom,
        pad_left=pool_pad_left, pad_right=pool_pad_right,
    )

    C_al = align_up(C, 8)
    H_out = task.H_out
    W_out = task.W_out

    input_handle = TensorHandle(
        name="input", shape=(C, H, W), dtype="float16",
        size_bytes=C_al * H * W * 2, dma_addr=PLACEHOLDER_INPUT,
    )
    output_handle = TensorHandle(
        name="output", shape=(C, H_out, W_out), dtype="float16",
        size_bytes=C_al * H_out * W_out * 2, dma_addr=PLACEHOLDER_OUTPUT,
    )

    gen = RegCmdGenerator()
    hw_task = gen.generate_pool(task, input_handle, output_handle)
    relocations = _build_relocation_table(hw_task)

    M_out = H_out * W_out
    regcmd_data = _serialize_regcmd_data_v2(
        hw_task, relocations,
        op_type=OP_MAXPOOL,
        M=M_out, K=0, N=C,
        C=C, H=H, W=W, H_out=H_out, W_out=W_out,
    )

    graph_json = _build_json_graph(
        composite_name,
        input_shapes=[(batch, C, H, W)],
        input_dtypes=["float16"],
        output_shape=(batch, C, H_out, W_out),
        output_dtype="float16",
    )

    const_names = tvm.runtime.convert([])
    create_fn = tvm.get_global_func("runtime.rknpu_runtime_create")
    func_name = str(func.attrs["global_symbol"])
    runtime_mod = create_fn(func_name, graph_json, const_names, regcmd_data)
    return runtime_mod


def _compile_avgpool_as_dwconv(func, composite_name, batch, C, H, W, kH, kW,
                                stride, pad_top, pad_bottom, pad_left, pad_right):
    """Compile avg_pool2d as a depthwise conv2d with constant uniform weights.

    The PPU hardware only supports max/min pooling. Average pooling is
    mathematically equivalent to depthwise conv2d with weight = 1/(kH*kW).
    The C++ runtime generates the constant weights at init time when it
    sees op_type = OP_AVGPOOL.
    """
    task = AbstractConv2DTask(
        op_name=composite_name,
        C=C, H=H, W=W, N=C,
        kH=kH, kW=kW, stride=stride,
        pad_top=pad_top, pad_bottom=pad_bottom,
        pad_left=pad_left, pad_right=pad_right,
        relu=False, has_bias=False,
        is_depthwise=True, conv_mode=3, groups=C,
    )

    H_out = task.H_out
    W_out = task.W_out
    Mp = task.M_padded
    K_eff = task.K_eff
    K_eff_al = align_up(K_eff, 32)
    C_al = align_up(C, 32)

    input_handle = TensorHandle(
        name="input", shape=(C, H, W), dtype="float16",
        size_bytes=C_al * H * W * 2, dma_addr=PLACEHOLDER_INPUT,
    )
    weight_handle = TensorHandle(
        name="weight", shape=(C, 1, kH, kW), dtype="float16",
        size_bytes=K_eff_al * 2, dma_addr=PLACEHOLDER_WEIGHT,
    )
    output_handle = TensorHandle(
        name="output", shape=(C, H_out, W_out), dtype="float16",
        size_bytes=C_al * Mp * 2, dma_addr=PLACEHOLDER_OUTPUT,
    )

    gen = RegCmdGenerator()
    hw_task = gen.generate_conv_mode0(task, input_handle, weight_handle, output_handle)
    relocations = _build_relocation_table(hw_task)

    # Use OP_AVGPOOL so the C++ runtime knows to generate constant weights
    regcmd_data = _serialize_regcmd_data_v2(
        hw_task, relocations,
        op_type=OP_AVGPOOL,
        M=task.M, K=K_eff, N=C,
        C=C, H=H, W=W, H_out=H_out, W_out=W_out,
    )

    # Graph has only 1 input (data) — no weight input (runtime generates it)
    graph_json = _build_json_graph(
        composite_name,
        input_shapes=[(batch, C, H, W)],
        input_dtypes=["float16"],
        output_shape=(batch, C, H_out, W_out),
        output_dtype="float16",
    )

    const_names = tvm.runtime.convert([])
    create_fn = tvm.get_global_func("runtime.rknpu_runtime_create")
    func_name = str(func.attrs["global_symbol"])
    runtime_mod = create_fn(func_name, graph_json, const_names, regcmd_data)
    return runtime_mod


# ---------------------------------------------------------------------------
# V8: Graph-level compilation
# ---------------------------------------------------------------------------

REGCMD_VERSION_V8 = 8

# V8 scatter type codes (how to prepare external inputs)
V8_SCATTER_FEATURE = 0     # [M, K] → NPU feature layout [K_al/8, M_pad, 8]
V8_SCATTER_WEIGHT = 1      # [K, N] → NPU weight layout [N_al/16, K_al/32, 16, 32]
V8_SCATTER_BIAS_FP32 = 2   # [N] FP16 → [N_al] FP32 for DPU_RDMA

# V8 gather type codes (how to extract external outputs)
V8_GATHER_FEATURE = 0      # NPU feature layout → [M, N] row-major

# V8 segment type codes
V8_SEG_NPU = 0
V8_SEG_CPU = 1

# V8 CPU operation codes
V8_CPU_MAX_REDUCE = 0      # Row-wise max of [M, K] → [M, 1]
V8_CPU_RECIPROCAL = 1      # Element-wise 1/x


def _graph_buf_size_2d(M, K):
    """Compute DMA buffer size for a 2D [M, K] tensor in NPU feature layout.

    Uses 32-alignment (strictest requirement, needed for matmul input K).
    """
    K_al = align_up(K, 32)
    Mp = pad_m(M)
    return Mp * K_al * 2  # FP16


def _weight_buf_size(K, N):
    """Compute DMA buffer size for a [K, N] weight in NPU weight layout."""
    K_al = align_up(K, 32)
    N_al = align_up(N, 16)
    return K_al * N_al * 2


def _bias_buf_size(N):
    """Compute DMA buffer size for [N] bias in FP32."""
    N_al = align_up(N, 16)
    return N_al * 4


# Composite parameter roles: "feature", "weight", or "bias"
_PARAM_ROLES = {
    "rknpu.matmul": ["feature", "weight"],
    "rknpu.matmul_bias": ["feature", "weight", "bias"],
    "rknpu.matmul_relu": ["feature", "weight"],
    "rknpu.matmul_bias_relu": ["feature", "weight", "bias"],
    "rknpu.add": ["feature", "feature"],
    "rknpu.multiply": ["feature", "feature"],
    "rknpu.gelu": ["feature"],
    "rknpu.exp": ["feature"],
    "rknpu.sigmoid": ["feature"],
    "rknpu.rsqrt": ["feature"],
    "rknpu.softmax": ["feature"],
    "rknpu.layer_norm": ["feature", "weight_1d", "weight_1d"],
}


def _var_shape(var):
    """Get the shape of a Relax variable as a list of ints."""
    return [int(s) for s in var.struct_info.shape]


def _remap_relocations(std_relocs, reloc_map):
    """Remap standard relocations (type-based) to V8 relocations (buf_idx + offset).

    Parameters
    ----------
    std_relocs : list of (regcmd_idx, reloc_type)
        Standard relocations from _build_relocation_table.
    reloc_map : dict of {reloc_type: (buf_idx, byte_offset)}
        Mapping from relocation types to buffer indices.

    Returns
    -------
    list of (regcmd_idx, buf_idx, byte_offset)
    """
    v8_relocs = []
    for regcmd_idx, reloc_type in std_relocs:
        if reloc_type in reloc_map:
            buf_idx, byte_offset = reloc_map[reloc_type]
            v8_relocs.append((regcmd_idx, buf_idx, byte_offset))
        else:
            raise ValueError(f"Unmapped V8 relocation type {reloc_type}")
    return v8_relocs


def _gen_matmul_tasks_v8(comp_name, comp_func, arg_bufs, result_buf):
    """Generate V8 NPU tasks for a matmul composite.

    Handles M-tiling and N-tiling automatically.

    Returns list of (hw_task, v8_relocations).
    """
    has_relu = "relu" in comp_name
    has_bias = "bias" in comp_name

    # Get shapes from the composite function's params
    params = comp_func.params
    lhs_shape = _var_shape(params[0])
    rhs_shape = _var_shape(params[1])
    M, K, N = lhs_shape[0], lhs_shape[1], rhs_shape[1]

    task = AbstractMatmulTask(
        op_name=comp_name, M=M, K=K, N=N,
        precision="float16", relu=has_relu, has_bias=has_bias, output_fp16=True,
    )
    m_tile_size = task.compute_m_tile_size()
    n_tile_size = compute_n_tile(N)
    needs_m_tiling = M > m_tile_size
    needs_n_tiling = N > n_tile_size

    K_aligned = align_up(K, 32)
    gen = RegCmdGenerator()
    tasks = []
    dma_row_bytes = 8 * 2  # 16 bytes per row in feature layout

    for m_off in range(0, M, m_tile_size if needs_m_tiling else M):
        M_tile = min(m_tile_size if needs_m_tiling else M, M - m_off)
        for n_off in range(0, N, n_tile_size if needs_n_tiling else N):
            N_tile = min(n_tile_size if needs_n_tiling else N, N - n_off)
            N_tile_al = align_up(N_tile, 16)
            Mp_tile = pad_m(M_tile)

            tile_task = AbstractMatmulTask(
                op_name=comp_name, M=M_tile, K=K, N=N_tile,
                precision="float16", relu=has_relu, has_bias=has_bias, output_fp16=True,
            )
            if needs_m_tiling:
                tile_task.is_mtile = True
                tile_task.M_tile = M_tile
                tile_task.M_full = M
                tile_task.m_offset = 0
            if needs_n_tiling:
                tile_task.is_ntile = True
                tile_task.N_tile = N_tile

            input_h = TensorHandle(
                name="input", shape=(M_tile, K), dtype="float16",
                size_bytes=Mp_tile * K_aligned * 2, dma_addr=PLACEHOLDER_INPUT,
            )
            weight_h = TensorHandle(
                name="weight", shape=(K, N_tile), dtype="float16",
                size_bytes=K_aligned * N_tile_al * 2, dma_addr=PLACEHOLDER_WEIGHT,
            )
            output_h = TensorHandle(
                name="output", shape=(M_tile, N_tile), dtype="float16",
                size_bytes=Mp_tile * N_tile_al * 2, dma_addr=PLACEHOLDER_OUTPUT,
            )
            bias_h = None
            if has_bias:
                bias_h = TensorHandle(
                    name="bias", shape=(N_tile,), dtype="float32",
                    size_bytes=N_tile_al * 4, dma_addr=PLACEHOLDER_BIAS,
                )

            hw_task = gen.generate_matmul(tile_task, input_h, weight_h, output_h, bias_handle=bias_h)
            std_relocs = _build_relocation_table(hw_task)

            # Compute byte offsets for tiling
            in_byte_off = m_off * dma_row_bytes
            wt_byte_off = n_off * K_aligned * 2
            out_byte_off = m_off * dma_row_bytes + n_off * pad_m(M) * 2
            bias_byte_off = n_off * 4

            reloc_map = {
                RELOC_INPUT: (arg_bufs[0], in_byte_off),
                RELOC_WEIGHT: (arg_bufs[1], wt_byte_off),
                RELOC_OUTPUT: (result_buf, out_byte_off),
            }
            if has_bias:
                reloc_map[RELOC_BIAS] = (arg_bufs[2], bias_byte_off)

            v8_relocs = _remap_relocations(std_relocs, reloc_map)
            tasks.append((hw_task, v8_relocs))

    return tasks


def _gen_elementwise_tasks_v8(comp_name, comp_func, arg_bufs, result_buf):
    """Generate V8 NPU tasks for an add/multiply composite."""
    lhs_shape = _var_shape(comp_func.params[0])
    op_type = "add" if "add" in comp_name else "mul"

    if len(lhs_shape) == 2:
        H, C = lhs_shape
    else:
        H, C = 1, lhs_shape[0]

    task = AbstractElementwiseTask(
        op_name=comp_name, op_type=op_type, n_inputs=2, shape=tuple(lhs_shape),
    )

    C_al = align_up(C, 16)
    Mp = pad_m(H)
    buf_size = C_al * Mp * 2

    input_a = TensorHandle(name="input_a", shape=tuple(lhs_shape), dtype="float16",
                           size_bytes=buf_size, dma_addr=PLACEHOLDER_INPUT)
    input_b = TensorHandle(name="input_b", shape=tuple(lhs_shape), dtype="float16",
                           size_bytes=buf_size, dma_addr=PLACEHOLDER_WEIGHT)
    output = TensorHandle(name="output", shape=tuple(lhs_shape), dtype="float16",
                          size_bytes=buf_size, dma_addr=PLACEHOLDER_OUTPUT)

    gen = RegCmdGenerator()
    hw_task = gen.generate_elementwise(task, input_a, input_b, output)
    std_relocs = _build_relocation_table(hw_task)

    reloc_map = {
        RELOC_INPUT: (arg_bufs[0], 0),
        RELOC_WEIGHT: (arg_bufs[1], 0),
        RELOC_OUTPUT: (result_buf, 0),
    }
    v8_relocs = _remap_relocations(std_relocs, reloc_map)
    return [(hw_task, v8_relocs)]


def _gen_lut_tasks_v8(comp_name, comp_func, arg_bufs, result_buf):
    """Generate V8 NPU tasks for a LUT activation (exp, sigmoid, rsqrt)."""
    inp_shape = _var_shape(comp_func.params[0])
    if "exp" in comp_name:
        lut_name = "exp"
    elif "sigmoid" in comp_name:
        lut_name = "sigmoid"
    elif "rsqrt" in comp_name:
        lut_name = "rsqrt"
    else:
        raise ValueError(f"Unsupported LUT: {comp_name}")

    le_table, lo_table, lut_params = _get_lut_config(lut_name)
    gen = RegCmdGenerator()

    hw_task = gen.generate_lut_combined_task(
        le_table, lo_table, shape=tuple(inp_shape),
        src_dma=PLACEHOLDER_INPUT, dst_dma=PLACEHOLDER_OUTPUT,
        lut_params=lut_params,
    )
    std_relocs = _build_relocation_table(hw_task)

    reloc_map = {
        RELOC_INPUT: (arg_bufs[0], 0),
        RELOC_OUTPUT: (result_buf, 0),
    }
    v8_relocs = _remap_relocations(std_relocs, reloc_map)
    return [(hw_task, v8_relocs)]


def _gen_gelu_tasks_v8(comp_func, arg_bufs, result_buf, alloc_buf):
    """Generate V8 NPU tasks for GELU = x * sigmoid(1.702 * x).

    Creates an internal intermediate buffer for the sigmoid output.
    """
    inp_shape = _var_shape(comp_func.params[0])
    if len(inp_shape) == 2:
        H, C = inp_shape
    else:
        H, C = 1, inp_shape[0]

    C_al = align_up(C, 16)
    Mp = pad_m(H)
    buf_size = C_al * Mp * 2

    # Allocate internal intermediate buffer
    int_buf = alloc_buf(buf_size, "gelu_internal")

    from .npu_core.lut_tables import SIGMOID_LE_TABLE, SIGMOID_LO_TABLE, GELU_LUT_PARAMS
    gen = RegCmdGenerator()

    # Task 1: LUT sigmoid(1.702*x) → internal buffer
    lut_task = gen.generate_lut_combined_task(
        SIGMOID_LE_TABLE, SIGMOID_LO_TABLE,
        shape=tuple(inp_shape),
        src_dma=PLACEHOLDER_INPUT, dst_dma=PLACEHOLDER_INTERMEDIATE,
        lut_params=GELU_LUT_PARAMS,
    )
    lut_relocs = _build_relocation_table(lut_task)
    lut_reloc_map = {
        RELOC_INPUT: (arg_bufs[0], 0),
        RELOC_INTERMEDIATE: (int_buf, 0),
    }
    v8_lut_relocs = _remap_relocations(lut_relocs, lut_reloc_map)

    # Task 2: EW multiply x * sigmoid(1.702*x) → output
    ew_task_obj = AbstractElementwiseTask(
        op_name="gelu_mul", op_type="mul", n_inputs=2, shape=tuple(inp_shape),
    )
    input_a = TensorHandle(name="input_a", shape=tuple(inp_shape), dtype="float16",
                           size_bytes=buf_size, dma_addr=PLACEHOLDER_INPUT)
    input_b = TensorHandle(name="input_b", shape=tuple(inp_shape), dtype="float16",
                           size_bytes=buf_size, dma_addr=PLACEHOLDER_INTERMEDIATE)
    output = TensorHandle(name="output", shape=tuple(inp_shape), dtype="float16",
                          size_bytes=buf_size, dma_addr=PLACEHOLDER_OUTPUT)

    ew_task = gen.generate_elementwise(ew_task_obj, input_a, input_b, output)
    ew_relocs = _build_relocation_table(ew_task)
    ew_reloc_map = {
        RELOC_INPUT: (arg_bufs[0], 0),
        RELOC_INTERMEDIATE: (int_buf, 0),
        RELOC_OUTPUT: (result_buf, 0),
    }
    v8_ew_relocs = _remap_relocations(ew_relocs, ew_reloc_map)

    return [(lut_task, v8_lut_relocs), (ew_task, v8_ew_relocs)]


def _gen_layer_norm_tasks_v8(comp_func, arg_bufs, result_buf, alloc_buf, alloc_const):
    """Generate V8 NPU tasks for layer_norm decomposition (11 tasks).

    Internally decomposes layer_norm(x, gamma, beta) into:
        1. neg_mean = x @ W_neg_mean           [M,K]x[K,R] → [M,R]
        2. neg_mean_bc = neg_mean @ W_tile      [M,R]x[R,K] → [M,K]
        3. centered = x + neg_mean_bc           [M,K]
        4. sq = centered * centered             [M,K]
        5. var = sq @ W_pos_mean                [M,K]x[K,R] → [M,R]
        6. var_eps = var + eps_const            [M,R]
        7. inv_std = rsqrt(var_eps)             [M,R]
        8. inv_std_bc = inv_std @ W_tile        [M,R]x[R,K] → [M,K]
        9. normed = centered * inv_std_bc       [M,K]
       10. scaled = normed * gamma_bc           [M,K]
       11. output = scaled + beta_bc            [M,K]

    gamma_bc and beta_bc are gamma/beta reshaped to [1,K] for broadcasting.
    """
    inp_shape = _var_shape(comp_func.params[0])
    M, K = inp_shape[0], inp_shape[1]
    R = 16  # Reduction dimension for broadcast-via-matmul

    # arg_bufs: [input_buf, gamma_buf, beta_buf]
    input_buf = arg_bufs[0]
    gamma_buf = arg_bufs[1]
    beta_buf = arg_bufs[2]
    output_buf = result_buf

    # Allocate intermediate buffers
    buf_MK = _graph_buf_size_2d(M, K)
    buf_MR = _graph_buf_size_2d(M, R)
    neg_mean_buf = alloc_buf(buf_MR, "ln_neg_mean")
    neg_mean_bc_buf = alloc_buf(buf_MK, "ln_neg_mean_bc")
    centered_buf = alloc_buf(buf_MK, "ln_centered")
    sq_buf = alloc_buf(buf_MK, "ln_sq")
    var_buf = alloc_buf(buf_MR, "ln_var")
    var_eps_buf = alloc_buf(buf_MR, "ln_var_eps")
    inv_std_buf = alloc_buf(buf_MR, "ln_inv_std")
    inv_std_bc_buf = alloc_buf(buf_MK, "ln_inv_std_bc")
    normed_buf = alloc_buf(buf_MK, "ln_normed")
    scaled_buf = alloc_buf(buf_MK, "ln_scaled")
    gamma_bc_buf = alloc_buf(buf_MK, "ln_gamma_bc")
    beta_bc_buf = alloc_buf(buf_MK, "ln_beta_bc")

    # Create constant weight matrices
    # W_neg_mean: [K, R] = -1/K per element
    W_neg_mean = np.full((K, R), -1.0 / K, dtype=np.float16)
    # W_pos_mean: [K, R] = +1/K per element
    W_pos_mean = np.full((K, R), 1.0 / K, dtype=np.float16)
    # W_tile: [R, K] = 1/R per element (broadcast tile, 1/R compensates for R-way sum)
    W_tile = np.full((R, K), 1.0 / R, dtype=np.float16)

    # Extract epsilon from the layer_norm attrs
    eps = 1e-5
    bindings = get_var2val(comp_func)
    for _, expr in bindings.items():
        if isinstance(expr, tvm.relax.Call):
            if hasattr(expr.op, 'name') and 'layer_norm' in str(expr.op.name):
                if expr.attrs is not None:
                    eps = float(expr.attrs.epsilon)

    # eps_const: [M, R] = epsilon (broadcast via matmul)
    eps_const = np.full((M, R), eps, dtype=np.float16)

    # ones_col: [M, 1] constant for broadcasting via matmul [M,1] @ [1,K] → [M,K]
    ones_col = np.ones((M, 1), dtype=np.float16)

    # Allocate constant buffers
    w_neg_mean_buf = alloc_const(_weight_buf_size(K, R), "ln_w_neg_mean",
                                  W_neg_mean.tobytes(), V8_SCATTER_WEIGHT, K, R)
    w_pos_mean_buf = alloc_const(_weight_buf_size(K, R), "ln_w_pos_mean",
                                  W_pos_mean.tobytes(), V8_SCATTER_WEIGHT, K, R)
    w_tile_buf = alloc_const(_weight_buf_size(R, K), "ln_w_tile",
                              W_tile.tobytes(), V8_SCATTER_WEIGHT, R, K)
    eps_const_buf = alloc_const(buf_MR, "ln_eps",
                                 eps_const.tobytes(), V8_SCATTER_FEATURE, M, R)
    ones_col_buf = alloc_const(_graph_buf_size_2d(M, 1), "ln_ones_col",
                                ones_col.tobytes(), V8_SCATTER_FEATURE, M, 1)

    gen = RegCmdGenerator()
    all_tasks = []

    def _matmul_task(M_, K_, N_, in_buf, wt_buf, out_buf, name=""):
        """Generate a single matmul task with V8 relocations."""
        t = AbstractMatmulTask(
            op_name=name, M=M_, K=K_, N=N_,
            precision="float16", relu=False, has_bias=False, output_fp16=True,
        )
        m_tile = t.compute_m_tile_size()
        K_al = align_up(K_, 32)
        N_al = align_up(N_, 16)
        Mp = pad_m(M_)

        # Check if M-tiling needed
        needs_tiling = M_ > m_tile
        tile_tasks = []
        for m_off in range(0, M_, m_tile if needs_tiling else M_):
            Mt = min(m_tile if needs_tiling else M_, M_ - m_off)
            Mpt = pad_m(Mt)

            tile_t = AbstractMatmulTask(
                op_name=name, M=Mt, K=K_, N=N_,
                precision="float16", relu=False, has_bias=False, output_fp16=True,
            )
            if needs_tiling:
                tile_t.is_mtile = True
                tile_t.M_tile = Mt
                tile_t.M_full = M_
                tile_t.m_offset = 0

            ih = TensorHandle(name="input", shape=(Mt, K_), dtype="float16",
                              size_bytes=Mpt * K_al * 2, dma_addr=PLACEHOLDER_INPUT)
            wh = TensorHandle(name="weight", shape=(K_, N_), dtype="float16",
                              size_bytes=K_al * N_al * 2, dma_addr=PLACEHOLDER_WEIGHT)
            oh = TensorHandle(name="output", shape=(Mt, N_), dtype="float16",
                              size_bytes=Mpt * N_al * 2, dma_addr=PLACEHOLDER_OUTPUT)

            hw = gen.generate_matmul(tile_t, ih, wh, oh)
            sr = _build_relocation_table(hw)
            dma_row = 8 * 2
            rmap = {
                RELOC_INPUT: (in_buf, m_off * dma_row),
                RELOC_WEIGHT: (wt_buf, 0),
                RELOC_OUTPUT: (out_buf, m_off * dma_row),
            }
            tile_tasks.append((hw, _remap_relocations(sr, rmap)))
        return tile_tasks

    def _ew_task(op, shape, a_buf, b_buf, out_buf, name=""):
        """Generate an elementwise task."""
        H_, C_ = (shape[0], shape[1]) if len(shape) == 2 else (1, shape[0])
        C_al_ = align_up(C_, 16)
        Mp_ = pad_m(H_)
        bs = C_al_ * Mp_ * 2
        t = AbstractElementwiseTask(op_name=name, op_type=op, n_inputs=2, shape=tuple(shape))
        a = TensorHandle(name="a", shape=tuple(shape), dtype="float16",
                         size_bytes=bs, dma_addr=PLACEHOLDER_INPUT)
        b = TensorHandle(name="b", shape=tuple(shape), dtype="float16",
                         size_bytes=bs, dma_addr=PLACEHOLDER_WEIGHT)
        o = TensorHandle(name="o", shape=tuple(shape), dtype="float16",
                         size_bytes=bs, dma_addr=PLACEHOLDER_OUTPUT)
        hw = gen.generate_elementwise(t, a, b, o)
        sr = _build_relocation_table(hw)
        rmap = {
            RELOC_INPUT: (a_buf, 0),
            RELOC_WEIGHT: (b_buf, 0),
            RELOC_OUTPUT: (out_buf, 0),
        }
        return [(hw, _remap_relocations(sr, rmap))]

    def _rsqrt_task(shape, in_buf, out_buf):
        """Generate an rsqrt LUT task."""
        le, lo, params = _get_lut_config("rsqrt")
        hw = gen.generate_lut_combined_task(
            le, lo, shape=tuple(shape),
            src_dma=PLACEHOLDER_INPUT, dst_dma=PLACEHOLDER_OUTPUT,
            lut_params=params,
        )
        sr = _build_relocation_table(hw)
        rmap = {
            RELOC_INPUT: (in_buf, 0),
            RELOC_OUTPUT: (out_buf, 0),
        }
        return [(hw, _remap_relocations(sr, rmap))]

    MK = [M, K]
    MR = [M, R]

    # 1. neg_mean = x @ W_neg_mean
    all_tasks.extend(_matmul_task(M, K, R, input_buf, w_neg_mean_buf, neg_mean_buf, "ln_1"))
    # 2. neg_mean_bc = neg_mean @ W_tile
    all_tasks.extend(_matmul_task(M, R, K, neg_mean_buf, w_tile_buf, neg_mean_bc_buf, "ln_2"))
    # 3. centered = x + neg_mean_bc
    all_tasks.extend(_ew_task("add", MK, input_buf, neg_mean_bc_buf, centered_buf, "ln_3"))
    # 4. sq = centered * centered
    all_tasks.extend(_ew_task("mul", MK, centered_buf, centered_buf, sq_buf, "ln_4"))
    # 5. var = sq @ W_pos_mean
    all_tasks.extend(_matmul_task(M, K, R, sq_buf, w_pos_mean_buf, var_buf, "ln_5"))
    # 6. var_eps = var + eps_const
    all_tasks.extend(_ew_task("add", MR, var_buf, eps_const_buf, var_eps_buf, "ln_6"))
    # 7. inv_std = rsqrt(var_eps)
    all_tasks.extend(_rsqrt_task(MR, var_eps_buf, inv_std_buf))
    # 8. inv_std_bc = inv_std @ W_tile
    all_tasks.extend(_matmul_task(M, R, K, inv_std_buf, w_tile_buf, inv_std_bc_buf, "ln_8"))
    # 9. normed = centered * inv_std_bc
    all_tasks.extend(_ew_task("mul", MK, centered_buf, inv_std_bc_buf, normed_buf, "ln_9"))
    # 10. gamma_bc = ones_col @ gamma  [M,1]x[1,K] → [M,K] (broadcast gamma)
    all_tasks.extend(_matmul_task(M, 1, K, ones_col_buf, gamma_buf, gamma_bc_buf, "ln_10_bc"))
    # 11. scaled = normed * gamma_bc
    all_tasks.extend(_ew_task("mul", MK, normed_buf, gamma_bc_buf, scaled_buf, "ln_11"))
    # 12. beta_bc = ones_col @ beta  [M,1]x[1,K] → [M,K] (broadcast beta)
    all_tasks.extend(_matmul_task(M, 1, K, ones_col_buf, beta_buf, beta_bc_buf, "ln_12_bc"))
    # 13. output = scaled + beta_bc
    all_tasks.extend(_ew_task("add", MK, scaled_buf, beta_bc_buf, output_buf, "ln_13"))

    return all_tasks


def _gen_softmax_tasks_v8(comp_func, arg_bufs, result_buf, alloc_buf, alloc_const):
    """Generate V8 tasks for softmax decomposition.

    Softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    Decomposed as:
        CPU: row_max = max(x, axis=-1)     [M, K] → [M, 1]  (CPU segment)
        NPU: neg_max_bc = row_max @ W_bc   [M, 1] × [1, K] → [M, K]  (broadcast)
        NPU: shifted = x + neg_max_bc      [M, K]  (actually subtract via negated max)
        NPU: exp_shifted = exp(shifted)     [M, K]
        NPU: row_sum = exp_shifted @ W_sum  [M, K] × [K, 1] → [M, 1]
        CPU: inv_sum = 1 / row_sum          [M, 1]  (CPU segment)
        NPU: inv_sum_bc = inv_sum @ W_bc_K  [M, 1] × [1, K] → [M, K]
        NPU: output = exp_shifted * inv_bc  [M, K]

    Returns list of (seg_type, hw_task_or_cpu_op, v8_relocs_or_params)
    """
    inp_shape = _var_shape(comp_func.params[0])
    M, K = inp_shape[0], inp_shape[1]

    input_buf = arg_bufs[0]
    output_buf = result_buf

    # Intermediate buffers
    buf_MK = _graph_buf_size_2d(M, K)
    buf_M1 = _graph_buf_size_2d(M, 1)
    row_max_buf = alloc_buf(buf_M1, "sm_row_max")
    neg_max_bc_buf = alloc_buf(buf_MK, "sm_neg_max_bc")
    shifted_buf = alloc_buf(buf_MK, "sm_shifted")
    exp_buf = alloc_buf(buf_MK, "sm_exp")
    row_sum_buf = alloc_buf(buf_M1, "sm_row_sum")
    inv_sum_buf = alloc_buf(buf_M1, "sm_inv_sum")
    inv_sum_bc_buf = alloc_buf(buf_MK, "sm_inv_sum_bc")

    # Constants
    # W_neg_ones: [K, 1] = -1.0 (for max-reduce, we negate so we can add)
    # Actually, row_max is computed on CPU, then negated before broadcast.
    # W_bc: [1, K] = 1.0 (broadcast [M,1] → [M,K] via matmul)
    W_bc = np.ones((1, K), dtype=np.float16)
    # W_sum: [K, 1] = 1.0 (sum reduction via matmul)
    W_sum = np.ones((K, 1), dtype=np.float16)

    w_bc_buf = alloc_const(_weight_buf_size(1, K), "sm_w_bc",
                            W_bc.tobytes(), V8_SCATTER_WEIGHT, 1, K)
    w_sum_buf = alloc_const(_weight_buf_size(K, 1), "sm_w_sum",
                             W_sum.tobytes(), V8_SCATTER_WEIGHT, K, 1)

    gen = RegCmdGenerator()
    segments = []  # list of (seg_type, tasks_or_ops)

    def _matmul_task(M_, K_, N_, in_buf, wt_buf, out_buf, name=""):
        t = AbstractMatmulTask(
            op_name=name, M=M_, K=K_, N=N_,
            precision="float16", relu=False, has_bias=False, output_fp16=True,
        )
        m_tile = t.compute_m_tile_size()
        K_al = align_up(K_, 32)
        N_al = align_up(N_, 16)
        tile_tasks = []
        for m_off in range(0, M_, m_tile if M_ > m_tile else M_):
            Mt = min(m_tile if M_ > m_tile else M_, M_ - m_off)
            Mpt = pad_m(Mt)
            tt = AbstractMatmulTask(
                op_name=name, M=Mt, K=K_, N=N_,
                precision="float16", relu=False, has_bias=False, output_fp16=True,
            )
            if Mt < M_:
                tt.is_mtile = True
                tt.M_tile = Mt
                tt.M_full = M_
                tt.m_offset = 0
            ih = TensorHandle(name="i", shape=(Mt, K_), dtype="float16",
                              size_bytes=Mpt * K_al * 2, dma_addr=PLACEHOLDER_INPUT)
            wh = TensorHandle(name="w", shape=(K_, N_), dtype="float16",
                              size_bytes=K_al * N_al * 2, dma_addr=PLACEHOLDER_WEIGHT)
            oh = TensorHandle(name="o", shape=(Mt, N_), dtype="float16",
                              size_bytes=Mpt * N_al * 2, dma_addr=PLACEHOLDER_OUTPUT)
            hw = gen.generate_matmul(tt, ih, wh, oh)
            sr = _build_relocation_table(hw)
            rmap = {
                RELOC_INPUT: (in_buf, m_off * 16),
                RELOC_WEIGHT: (wt_buf, 0),
                RELOC_OUTPUT: (out_buf, m_off * 16),
            }
            tile_tasks.append((hw, _remap_relocations(sr, rmap)))
        return tile_tasks

    def _ew_task(op, shape, a_buf, b_buf, out_buf):
        H_, C_ = shape if len(shape) == 2 else (1, shape[0])
        C_al_ = align_up(C_, 16)
        Mp_ = pad_m(H_)
        bs = C_al_ * Mp_ * 2
        t = AbstractElementwiseTask(op_name="", op_type=op, n_inputs=2, shape=tuple(shape))
        a = TensorHandle(name="a", shape=tuple(shape), dtype="float16",
                         size_bytes=bs, dma_addr=PLACEHOLDER_INPUT)
        b = TensorHandle(name="b", shape=tuple(shape), dtype="float16",
                         size_bytes=bs, dma_addr=PLACEHOLDER_WEIGHT)
        o = TensorHandle(name="o", shape=tuple(shape), dtype="float16",
                         size_bytes=bs, dma_addr=PLACEHOLDER_OUTPUT)
        hw = gen.generate_elementwise(t, a, b, o)
        sr = _build_relocation_table(hw)
        rmap = {
            RELOC_INPUT: (a_buf, 0),
            RELOC_WEIGHT: (b_buf, 0),
            RELOC_OUTPUT: (out_buf, 0),
        }
        return [(hw, _remap_relocations(sr, rmap))]

    def _exp_task(shape, in_buf, out_buf):
        le, lo, params = _get_lut_config("exp")
        hw = gen.generate_lut_combined_task(
            le, lo, shape=tuple(shape),
            src_dma=PLACEHOLDER_INPUT, dst_dma=PLACEHOLDER_OUTPUT,
            lut_params=params,
        )
        sr = _build_relocation_table(hw)
        rmap = {RELOC_INPUT: (in_buf, 0), RELOC_OUTPUT: (out_buf, 0)}
        return [(hw, _remap_relocations(sr, rmap))]

    MK = [M, K]
    M1 = [M, 1]

    # CPU segment 1: max-reduce, negate
    segments.append((V8_SEG_CPU, [(V8_CPU_MAX_REDUCE, input_buf, row_max_buf, M, K)]))

    # NPU segment 1: broadcast neg_max, add, exp, sum
    npu_tasks_1 = []
    npu_tasks_1.extend(_matmul_task(M, 1, K, row_max_buf, w_bc_buf, neg_max_bc_buf, "sm_bc1"))
    npu_tasks_1.extend(_ew_task("add", MK, input_buf, neg_max_bc_buf, shifted_buf))
    npu_tasks_1.extend(_exp_task(MK, shifted_buf, exp_buf))
    npu_tasks_1.extend(_matmul_task(M, K, 1, exp_buf, w_sum_buf, row_sum_buf, "sm_sum"))
    segments.append((V8_SEG_NPU, npu_tasks_1))

    # CPU segment 2: reciprocal
    segments.append((V8_SEG_CPU, [(V8_CPU_RECIPROCAL, row_sum_buf, inv_sum_buf, M, 1)]))

    # NPU segment 2: broadcast inv_sum, multiply
    npu_tasks_2 = []
    npu_tasks_2.extend(_matmul_task(M, 1, K, inv_sum_buf, w_bc_buf, inv_sum_bc_buf, "sm_bc2"))
    npu_tasks_2.extend(_ew_task("mul", MK, exp_buf, inv_sum_bc_buf, output_buf))
    segments.append((V8_SEG_NPU, npu_tasks_2))

    return segments


def _serialize_v8(buffers, ext_inputs, ext_outputs, constants, segments, total_npu_regcmds):
    """Serialize a V8 graph-level binary.

    Parameters
    ----------
    buffers : list of int
        Buffer sizes in bytes.
    ext_inputs : list of (param_idx, buf_idx, scatter_type, dim0, dim1)
    ext_outputs : list of (output_idx, buf_idx, gather_type, dim0, dim1)
    constants : list of (buf_idx, scatter_type, dim0, dim1, data_bytes)
    segments : list of (seg_type, tasks)
        For NPU segments: tasks = [(hw_task, v8_relocs)]
        For CPU segments: tasks = [(op_type, in_buf, out_buf, M, K)]
    total_npu_regcmds : int

    Returns
    -------
    bytes
    """
    # Header (32 bytes = 8 uint32)
    header = struct.pack(
        "<IIIIIIII",
        REGCMD_MAGIC,
        REGCMD_VERSION_V8,
        len(buffers),
        len(segments),
        len(ext_inputs),
        len(ext_outputs),
        len(constants),
        total_npu_regcmds,
    )

    # Buffer table
    buf_table = b""
    for size in buffers:
        buf_table += struct.pack("<I", size)

    # External input table
    ext_in_data = b""
    for param_idx, buf_idx, scatter_type, dim0, dim1 in ext_inputs:
        ext_in_data += struct.pack("<IIIII", param_idx, buf_idx, scatter_type, dim0, dim1)

    # External output table
    ext_out_data = b""
    for output_idx, buf_idx, gather_type, dim0, dim1 in ext_outputs:
        ext_out_data += struct.pack("<IIIII", output_idx, buf_idx, gather_type, dim0, dim1)

    # Constants
    const_data = b""
    for buf_idx, scatter_type, dim0, dim1, data_bytes in constants:
        const_data += struct.pack("<IIII", buf_idx, scatter_type, dim0, dim1)
        const_data += struct.pack("<I", len(data_bytes))
        const_data += data_bytes

    # Segments
    seg_data = b""
    for seg_type, tasks in segments:
        seg_data += struct.pack("<II", seg_type, len(tasks))
        if seg_type == V8_SEG_NPU:
            for hw_task, v8_relocs in tasks:
                num_regcmds = len(hw_task.regcmds)
                num_relocs = len(v8_relocs)
                regcfg_amount = num_regcmds - 4

                seg_data += struct.pack(
                    "<IIIII",
                    num_regcmds,
                    hw_task.enable_mask,
                    hw_task.int_mask,
                    regcfg_amount,
                    num_relocs,
                )
                # Regcmd blob
                seg_data += _pack_regcmds(hw_task.regcmds)
                # V8 relocations (12 bytes each: regcmd_idx, buf_idx, byte_offset)
                for rcmd_idx, buf_idx, byte_off in v8_relocs:
                    seg_data += struct.pack("<III", rcmd_idx, buf_idx, byte_off)
        else:
            # CPU segment
            for op_type, in_buf, out_buf, M, K in tasks:
                seg_data += struct.pack("<IIIII", op_type, in_buf, out_buf, M, K)

    return header + buf_table + ext_in_data + ext_out_data + const_data + seg_data


def _compile_graph(func, constant_names):
    """Compile a merged function with multiple composites as one V8 binary.

    Walks the merged function's dataflow bindings in topological order,
    generates NPU/CPU tasks for each composite, assigns shared DMA buffers
    for intermediates (persistent NPU layout), and serializes as V8.
    """
    bindings = get_var2val(func)
    params = func.params

    # Step 1: Find composite definitions and composite calls
    composite_defs = {}  # Var → (composite_name, composite_func)
    composite_calls = []  # [(result_var, composite_name, composite_func, args)]

    for var, expr in bindings.items():
        if isinstance(expr, tvm.relax.Function):
            comp_attr = expr.attrs.get("Composite") if expr.attrs else None
            if comp_attr is not None:
                composite_defs[var] = (str(comp_attr), expr)
        elif isinstance(expr, tvm.relax.Call):
            op = expr.op
            if isinstance(op, tvm.relax.Var) and op in composite_defs:
                name, cfunc = composite_defs[op]
                composite_calls.append((var, name, cfunc, list(expr.args)))

    logger.info("V8 graph: %d composites found", len(composite_calls))

    # Step 2: Buffer assignment
    buffers = []       # List of buffer sizes
    var_to_buf = {}    # relax.Var → buffer index
    ext_inputs = []    # (param_idx, buf_idx, scatter_type, dim0, dim1)
    ext_outputs = []   # (output_idx, buf_idx, gather_type, dim0, dim1)
    constants = []     # (buf_idx, scatter_type, dim0, dim1, data_bytes)

    def alloc_buf(size_bytes, name=""):
        """Allocate a new buffer, return its index."""
        idx = len(buffers)
        buffers.append(size_bytes)
        logger.debug("V8 buf %d: %d bytes (%s)", idx, size_bytes, name)
        return idx

    def alloc_const(size_bytes, name, data_bytes, scatter_type, dim0, dim1):
        """Allocate a constant buffer with embedded data."""
        idx = alloc_buf(size_bytes, name)
        constants.append((idx, scatter_type, dim0, dim1, data_bytes))
        return idx

    # Track which function params are used and what role they play.
    # TVM Var identity uses same_as() rather than Python id(), because
    # different Python wrappers can reference the same underlying TVM node.
    def _find_param_idx(var):
        """Return param index if var is a function parameter, else -1."""
        for idx, p in enumerate(params):
            if var.same_as(p):
                return idx
        return -1

    # Var-to-buffer mapping using TVM object identity
    var_buf_list = []  # [(var, buf_idx)] for lookup via same_as()

    def _get_buf(var):
        """Get buffer index for a var, or None."""
        for v, b in var_buf_list:
            if var.same_as(v):
                return b
        return None

    def _set_buf(var, buf_idx):
        """Set buffer index for a var."""
        var_buf_list.append((var, buf_idx))

    # First pass: determine buffer for each composite arg and result
    for result_var, comp_name, comp_func, args in composite_calls:
        roles = _PARAM_ROLES.get(comp_name, ["feature"] * len(args))

        for i, arg in enumerate(args):
            if _get_buf(arg) is not None:
                continue  # Already assigned

            role = roles[i] if i < len(roles) else "feature"
            arg_shape = _var_shape(arg)
            pidx = _find_param_idx(arg)

            if pidx >= 0:
                # External input
                if role == "weight" and len(arg_shape) == 2:
                    K, N = arg_shape
                    buf_idx = alloc_buf(_weight_buf_size(K, N), f"ext_wt_{pidx}")
                    ext_inputs.append((pidx, buf_idx,
                                       V8_SCATTER_WEIGHT, K, N))
                elif role == "weight_1d" and len(arg_shape) == 1:
                    # 1D param scattered as weight [1, K] for matmul broadcast
                    K_dim = arg_shape[0]
                    buf_idx = alloc_buf(_weight_buf_size(1, K_dim),
                                        f"ext_wt1d_{pidx}")
                    ext_inputs.append((pidx, buf_idx,
                                       V8_SCATTER_WEIGHT, 1, K_dim))
                elif role == "bias":
                    N = arg_shape[0]
                    buf_idx = alloc_buf(_bias_buf_size(N), f"ext_bias_{pidx}")
                    ext_inputs.append((pidx, buf_idx,
                                       V8_SCATTER_BIAS_FP32, N, 0))
                else:
                    # Feature input
                    if len(arg_shape) == 2:
                        M, K = arg_shape
                    elif len(arg_shape) == 1:
                        M, K = 1, arg_shape[0]
                    else:
                        raise ValueError(f"Unsupported external input shape: {arg_shape}")
                    buf_idx = alloc_buf(_graph_buf_size_2d(M, K),
                                        f"ext_feat_{pidx}")
                    ext_inputs.append((pidx, buf_idx,
                                       V8_SCATTER_FEATURE, M, K))
                _set_buf(arg, buf_idx)
            else:
                # Intermediate (output of previous composite)
                # Should already have a buffer from being a result_var
                if _get_buf(arg) is None:
                    # This shouldn't normally happen, but handle gracefully
                    if len(arg_shape) == 2:
                        M, K = arg_shape
                    else:
                        M, K = 1, arg_shape[0]
                    _set_buf(arg, alloc_buf(
                        _graph_buf_size_2d(M, K), "intermediate"))

        # Assign result buffer
        if _get_buf(result_var) is None:
            out_shape = _var_shape(result_var)
            if len(out_shape) == 2:
                M, K = out_shape
            else:
                M, K = 1, out_shape[0]
            _set_buf(result_var, alloc_buf(
                _graph_buf_size_2d(M, K), f"result_{comp_name}"))

    # Identify function output
    # The function body's output is the last binding's result
    if composite_calls:
        last_result = composite_calls[-1][0]
        out_shape = _var_shape(last_result)
        if len(out_shape) == 2:
            M_out, K_out = out_shape
        else:
            M_out, K_out = 1, out_shape[0]
        ext_outputs.append((0, _get_buf(last_result),
                            V8_GATHER_FEATURE, M_out, K_out))

    # Step 3: Generate tasks per composite
    all_segments = []  # [(seg_type, tasks)]
    current_npu_tasks = []
    total_npu_regcmds = 0

    def flush_npu_tasks():
        nonlocal current_npu_tasks
        if current_npu_tasks:
            all_segments.append((V8_SEG_NPU, current_npu_tasks))
            current_npu_tasks = []

    for result_var, comp_name, comp_func, args in composite_calls:
        arg_bufs = [_get_buf(a) for a in args]
        res_buf = _get_buf(result_var)

        if comp_name.startswith("rknpu.matmul"):
            tasks = _gen_matmul_tasks_v8(comp_name, comp_func, arg_bufs, res_buf)
            for hw, relocs in tasks:
                current_npu_tasks.append((hw, relocs))
                total_npu_regcmds += len(hw.regcmds)

        elif comp_name in ("rknpu.add", "rknpu.multiply"):
            tasks = _gen_elementwise_tasks_v8(comp_name, comp_func, arg_bufs, res_buf)
            for hw, relocs in tasks:
                current_npu_tasks.append((hw, relocs))
                total_npu_regcmds += len(hw.regcmds)

        elif comp_name in ("rknpu.exp", "rknpu.sigmoid", "rknpu.rsqrt"):
            tasks = _gen_lut_tasks_v8(comp_name, comp_func, arg_bufs, res_buf)
            for hw, relocs in tasks:
                current_npu_tasks.append((hw, relocs))
                total_npu_regcmds += len(hw.regcmds)

        elif comp_name == "rknpu.gelu":
            tasks = _gen_gelu_tasks_v8(comp_func, arg_bufs, res_buf, alloc_buf)
            for hw, relocs in tasks:
                current_npu_tasks.append((hw, relocs))
                total_npu_regcmds += len(hw.regcmds)

        elif comp_name == "rknpu.layer_norm":
            tasks = _gen_layer_norm_tasks_v8(comp_func, arg_bufs, res_buf,
                                              alloc_buf, alloc_const)
            for hw, relocs in tasks:
                current_npu_tasks.append((hw, relocs))
                total_npu_regcmds += len(hw.regcmds)

        elif comp_name == "rknpu.softmax":
            # Softmax creates CPU segments, so flush current NPU tasks first
            flush_npu_tasks()
            sm_segments = _gen_softmax_tasks_v8(comp_func, arg_bufs, res_buf,
                                                 alloc_buf, alloc_const)
            for seg_type, tasks in sm_segments:
                if seg_type == V8_SEG_NPU:
                    for hw, relocs in tasks:
                        current_npu_tasks.append((hw, relocs))
                        total_npu_regcmds += len(hw.regcmds)
                else:
                    flush_npu_tasks()
                    all_segments.append((seg_type, tasks))

        else:
            raise ValueError(f"Unsupported V8 composite: {comp_name}")

    flush_npu_tasks()

    logger.info("V8 graph: %d buffers, %d segments, %d ext_inputs, %d ext_outputs, "
                "%d constants, %d total NPU regcmds",
                len(buffers), len(all_segments), len(ext_inputs), len(ext_outputs),
                len(constants), total_npu_regcmds)

    # Step 4: Serialize V8 binary
    regcmd_data = _serialize_v8(
        buffers, ext_inputs, ext_outputs, constants,
        all_segments, total_npu_regcmds,
    )

    # Debug: verify binary header and buffer table
    # Build JSON graph for JSONRuntimeBase
    input_shapes = []
    input_dtypes = []
    for param in params:
        sh = _var_shape(param)
        input_shapes.append(tuple(sh))
        input_dtypes.append(str(param.struct_info.dtype))

    out_shape = _var_shape(composite_calls[-1][0])
    graph_json = _build_json_graph(
        "rknpu.graph",
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shape=tuple(out_shape),
        output_dtype="float16",
    )

    const_names_arr = tvm.runtime.convert([])
    create_fn = tvm.get_global_func("runtime.rknpu_runtime_create")
    func_name = str(func.attrs["global_symbol"])
    runtime_mod = create_fn(func_name, graph_json, const_names_arr, regcmd_data)
    return runtime_mod


@tvm.register_global_func("relax.ext.rknpu")
def rknpu_compiler(functions, options, constant_names):
    """RKNPU Python codegen entry point.

    Called by ``relax.transform.RunCodegen()`` for functions annotated with
    ``codegen = "rknpu"``. Generates real NPU register command blobs using
    the vendored regcmd generator.

    Parameters
    ----------
    functions : Array[Function]
        Relax functions to compile.
    options : Map[String, Any]
        Compilation options (unused).
    constant_names : Map[Constant, String]
        Mapping from constant values to names.

    Returns
    -------
    Array[Module]
        Compiled runtime modules.
    """
    compiled = []
    for func in functions:
        assert isinstance(func, tvm.relax.Function)

        # Find all composite functions inside this partitioned function
        bindings = get_var2val(func)
        composite_names = []

        for var, expr in bindings.items():
            if isinstance(expr, tvm.relax.Function):
                composite_opt = expr.attrs.get("Composite") if expr.attrs else None
                if composite_opt is not None:
                    composite_names.append(str(composite_opt))

        if len(composite_names) > 1:
            # Graph-level: multiple composites merged into one function
            logger.info("RKNPU codegen: graph-level compilation (%d composites: %s)",
                        len(composite_names), composite_names)
            mod = _compile_graph(func, constant_names)
        elif len(composite_names) == 1:
            composite_name = composite_names[0]
            logger.info("RKNPU codegen: compiling %s", composite_name)

            if composite_name.startswith("rknpu.matmul"):
                mod = _compile_matmul(func, composite_name, constant_names)
            elif composite_name.startswith("rknpu.depthwise_conv2d"):
                mod = _compile_conv2d_common(func, composite_name, constant_names, is_depthwise=True)
            elif composite_name.startswith("rknpu.conv2d"):
                mod = _compile_conv2d_common(func, composite_name, constant_names)
            elif composite_name in ("rknpu.add", "rknpu.multiply"):
                mod = _compile_elementwise(func, composite_name, constant_names)
            elif composite_name == "rknpu.max_pool2d":
                mod = _compile_pool(func, composite_name, constant_names, is_avg=False)
            elif composite_name == "rknpu.avg_pool2d":
                mod = _compile_pool(func, composite_name, constant_names, is_avg=True)
            elif composite_name in ("rknpu.exp", "rknpu.sigmoid", "rknpu.rsqrt"):
                mod = _compile_lut(func, composite_name, constant_names)
            elif composite_name == "rknpu.gelu":
                mod = _compile_gelu(func, composite_name, constant_names)
            elif composite_name in ("rknpu.softmax", "rknpu.layer_norm"):
                # Single high-level composite — compile as graph with 1 composite
                logger.info("RKNPU codegen: graph-level for single %s", composite_name)
                mod = _compile_graph(func, constant_names)
            else:
                raise ValueError(f"Unsupported RKNPU composite: {composite_name}")
        else:
            raise ValueError("No composite functions found in partitioned function")

        compiled.append(mod)

    return compiled
