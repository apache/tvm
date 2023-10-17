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
# pylint: disable=invalid-name, unused-wildcard-import, wildcard-import, pointless-exception-statement

"""DeviceGemmXdlCShuffle Kernels."""
from . import factory
from . import library as gemm
from ... import library


def create_device_gemm_xdl_c_shuffle_f16_f16_f16_rrr_kernels():
    a_element_desc = library.TensorDesc(library.DataType.f16, library.LayoutType.RowMajor)
    b_element_desc = library.TensorDesc(library.DataType.f16, library.LayoutType.RowMajor)
    c_element_desc = library.TensorDesc(library.DataType.f16, library.LayoutType.RowMajor)
    element_op = library.TensorOperation.PassThrough

    tile_descriptions = [
        gemm.TileDesc(256, 256, 128, 32, 8, 2, 32, 32, 4, 2),
        gemm.TileDesc(256, 256, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 256, 32, 8, 2, 32, 32, 2, 4),
        gemm.TileDesc(256, 128, 256, 32, 8, 8, 32, 32, 2, 4),
        gemm.TileDesc(128, 128, 128, 32, 8, 2, 32, 32, 4, 2),
        gemm.TileDesc(128, 128, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 128, 32, 8, 2, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 32, 8, 2, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 8, 2, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 64, 32, 8, 2, 32, 32, 2, 1),
        gemm.TileDesc(256, 128, 64, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(256, 64, 128, 32, 8, 2, 32, 32, 1, 2),
        gemm.TileDesc(256, 64, 128, 32, 8, 8, 32, 32, 1, 2),
    ]

    b_block_descriptions = [
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1),
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1),
        gemm.BlockTransferDesc([8, 16, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 8, 1),
        gemm.BlockTransferDesc([16, 16, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 1, 8, 1),
        gemm.BlockTransferDesc([8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 2, 0),
        gemm.BlockTransferDesc([4, 64, 1], [0, 2, 1], [0, 2, 1], 1, 2, 8, 1),
    ]
    a_block_descriptions = []
    c_block_descriptions = []
    for t in tile_descriptions:
        a_block_transfer = -1
        c_block_transfer = -1
        if t.block_size == 256:
            a_block_transfer = [4, 64, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8)
        if t.block_size == 128:
            a_block_transfer = [4, 32, 1]
            if t.n_per_block == 128:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 8], 8)
            if t.n_per_block == 64:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 4], 8)

        assert (
            a_block_transfer != -1
            and c_block_transfer != -1
            and "Cannot determine block_transfer_size with block_size " + str(t.block_size)
        )
        a_block_descriptions.append(
            gemm.BlockTransferDesc(a_block_transfer, [1, 0, 2], [1, 0, 2], 2, 8, 8, 1)
        )
        c_block_descriptions.append(c_block_transfer)

    gemm_specialization = [
        gemm.GemmSpecialization.GemmDefault,
        gemm.GemmSpecialization.MNKPadding,
    ]
    operations = []
    for gemm_spec in gemm_specialization:
        for tile_desc, a_block_desc, b_block_desc, c_block_desc in zip(
            tile_descriptions,
            a_block_descriptions,
            b_block_descriptions,
            c_block_descriptions,
        ):
            new_operation = gemm.GemmOperation(
                gemm_type=gemm.GemmType.DeviceGemmXdl_CShuffle,
                A=a_element_desc,
                B=b_element_desc,
                C=c_element_desc,
                a_elem_op=element_op,
                b_elem_op=element_op,
                c_elem_op=element_op,
                gemm_specialization=gemm_spec,
                tile_desc=tile_desc,
                a_block_transfer=a_block_desc,
                b_block_transfer=b_block_desc,
                c_block_transfer=c_block_desc,
            )
            operations.append(new_operation)
    return operations


factory.register_instances(
    factory.DeviceGemmInstanceKey(
        a_dtype=library.DataType.f16,
        b_dtype=library.DataType.f16,
        c_dtype=library.DataType.f16,
        a_layout=library.LayoutType.RowMajor,
        b_layout=library.LayoutType.RowMajor,
        c_layout=library.LayoutType.RowMajor,
        batched=False,
    ),
    create_device_gemm_xdl_c_shuffle_f16_f16_f16_rrr_kernels(),
)


def create_device_gemm_xdl_c_shuffle_f16_f16_f16_rcr_kernels():
    a_element_desc = library.TensorDesc(library.DataType.f16, library.LayoutType.RowMajor)
    b_element_desc = library.TensorDesc(library.DataType.f16, library.LayoutType.ColumnMajor)
    c_element_desc = library.TensorDesc(library.DataType.f16, library.LayoutType.RowMajor)
    element_op = library.TensorOperation.PassThrough

    tile_descriptions = [
        gemm.TileDesc(256, 256, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 256, 32, 8, 8, 32, 32, 2, 4),
        gemm.TileDesc(128, 128, 128, 32, 8, 8, 32, 32, 4, 2),
        gemm.TileDesc(256, 128, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 128, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(128, 64, 128, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(64, 64, 64, 32, 8, 8, 32, 32, 2, 2),
        gemm.TileDesc(256, 128, 64, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(256, 64, 128, 32, 8, 8, 32, 32, 1, 2),
        gemm.TileDesc(128, 128, 32, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(128, 32, 128, 32, 8, 8, 32, 32, 1, 2),
        gemm.TileDesc(64, 64, 32, 32, 8, 8, 32, 32, 2, 1),
        gemm.TileDesc(64, 32, 64, 32, 8, 8, 32, 32, 1, 2),
    ]

    block_descriptions = []
    c_block_descriptions = []
    for t in tile_descriptions:
        block_transfer = -1
        c_block_transfer = -1
        if t.block_size == 256:
            block_transfer = [4, 64, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8)
        if t.block_size == 128:
            block_transfer = [4, 32, 1]
            if t.n_per_block == 128:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 8], 8)
            else:
                c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 4], 8)
        if t.block_size == 64:
            block_transfer = [4, 16, 1]
            c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 4], 8)

        assert (
            block_transfer != -1
            and c_block_transfer != -1
            and "Cannot determine block_transfer_size with block_size " + str(t.block_size)
        )
        block_descriptions.append(
            gemm.BlockTransferDesc(block_transfer, [1, 0, 2], [1, 0, 2], 2, 8, 8, 1)
        )
        c_block_descriptions.append(c_block_transfer)
    gemm_specialization = [
        gemm.GemmSpecialization.GemmDefault,
        gemm.GemmSpecialization.MNKPadding,
    ]
    operations = []
    for gemm_spec in gemm_specialization:
        for tile_desc, block_desc, c_block_desc in zip(
            tile_descriptions, block_descriptions, c_block_descriptions
        ):
            new_operation = gemm.GemmOperation(
                gemm_type=gemm.GemmType.DeviceGemmXdl_CShuffle,
                A=a_element_desc,
                B=b_element_desc,
                C=c_element_desc,
                a_elem_op=element_op,
                b_elem_op=element_op,
                c_elem_op=element_op,
                gemm_specialization=gemm_spec,
                tile_desc=tile_desc,
                a_block_transfer=block_desc,
                b_block_transfer=block_desc,
                c_block_transfer=c_block_desc,
            )
            operations.append(new_operation)
    return operations


factory.register_instances(
    factory.DeviceGemmInstanceKey(
        a_dtype=library.DataType.f16,
        b_dtype=library.DataType.f16,
        c_dtype=library.DataType.f16,
        a_layout=library.LayoutType.RowMajor,
        b_layout=library.LayoutType.ColumnMajor,
        c_layout=library.LayoutType.RowMajor,
        batched=False,
    ),
    create_device_gemm_xdl_c_shuffle_f16_f16_f16_rcr_kernels(),
)
