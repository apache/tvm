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

"""Gemm data structures that mirrors ComposableKernel definitions."""

from dataclasses import dataclass
from copy import deepcopy
from enum import auto, Enum
from typing import List
import jinja2

from tvm.contrib.composable_kernel import library


################################################################################
# Gemm Template Parameter
################################################################################
class GemmSpecialization(Enum):
    GemmDefault = auto()
    MNKPadding = auto()
    MNPadding = auto()
    MNOPadding = auto()
    MNKOPadding = auto()


GemmSpecializationTag = {
    GemmSpecialization.GemmDefault: "ck::tensor_operation::device::GemmSpecialization::Default",
    GemmSpecialization.MNKPadding: "ck::tensor_operation::device::GemmSpecialization::MNKPadding",
    GemmSpecialization.MNPadding: "ck::tensor_operation::device::GemmSpecialization::MNPadding",
    GemmSpecialization.MNOPadding: "ck::tensor_operation::device::GemmSpecialization::MNOPadding",
    GemmSpecialization.MNKOPadding: "ck::tensor_operation::device::GemmSpecialization::MNKOPadding",
}


class GemmBaseType(Enum):
    DeviceGemm = auto()
    DeviceBatchedGemm = auto()


GemmBaseTypeTag = {
    GemmBaseType.DeviceGemm: "ck::tensor_operation::device::DeviceGemm",
    GemmBaseType.DeviceBatchedGemm: "ck::tensor_operation::device::DeviceBatchedGemm",
}


class GemmType(Enum):
    DeviceGemmXdl_CShuffle = auto()
    DeviceBatchedGemmXdl = auto()


GemmTypeTag = {
    GemmType.DeviceGemmXdl_CShuffle: "ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle",
    GemmType.DeviceBatchedGemmXdl: "ck::tensor_operation::device::DeviceBatchedGemmXdl",
}

GemmTypeShortName = {
    GemmType.DeviceGemmXdl_CShuffle: "gemm_xdl_cshuffle",
    GemmType.DeviceBatchedGemmXdl: "batched_gemm_xdl",
}

GemmTypeHeader = {
    GemmType.DeviceGemmXdl_CShuffle: "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp",
    GemmType.DeviceBatchedGemmXdl: "ck/tensor_operation/gpu/device/impl/device_batched_gemm_xdl.hpp",
}


GemmType2GemmBaseType = {
    GemmType.DeviceGemmXdl_CShuffle: GemmBaseType.DeviceGemm,
    GemmType.DeviceBatchedGemmXdl: GemmBaseType.DeviceBatchedGemm,
}


@dataclass
class TileDesc:
    block_size: int
    m_per_block: int
    n_per_block: int
    k_per_block: int
    ak1: int
    bk1: int
    m_per_xdl: int
    n_per_xdl: int
    m_xdl_per_wave: int
    n_xdl_per_wave: int

    def __str__(self) -> str:
        values = list(self.__dict__.values())
        return "_".join([str(x) for x in values])

    def emit(self) -> str:
        args = deepcopy(self.__dict__)
        template = jinja2.Template(
            """
{%for key, value in param.items() %}
{% if value!=0 %}   {{value}}, // {{key}}
    {% endif %}
{% endfor %}
""",
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return template.render(param=args)


@dataclass
class BlockTransferDesc:
    thread_cluster_length: List[int]
    thread_cluster_arrange_order: List[int]
    src_access_order: List[int]
    src_vector_dim: int
    src_scalar_per_vector: int
    dst_scalar_per_vector: int
    add_extra_dim: int
    add_extra_dim_flag: bool = False

    def __str__(self) -> str:
        args = deepcopy(self.__dict__)
        args["thread_cluster_length"] = [str(x) for x in self.thread_cluster_length]
        args["thread_cluster_arrange_order"] = [str(x) for x in self.thread_cluster_arrange_order]
        args["src_access_order"] = [str(x) for x in self.src_access_order]

        template = jinja2.Template(
            """S{{thread_cluster_length|join('_')}}S
            _S{{thread_cluster_arrange_order|join('_')}}S
            _S{{src_access_order|join('_')}}S
            _{{src_vector_dim}}
            _{{src_scalar_per_vector}}
            _{{dst_scalar_per_vector}}
            _{{add_extra_dim}}
            """,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return template.render(**args).replace("\n", "").replace(" ", "")

    def emit(self) -> str:
        args = deepcopy(self.__dict__)
        args["thread_cluster_length"] = [str(x) for x in self.thread_cluster_length]
        args["thread_cluster_arrange_order"] = [str(x) for x in self.thread_cluster_arrange_order]
        args["src_access_order"] = [str(x) for x in self.src_access_order]
        template = jinja2.Template(
            """
    ck::Sequence<{{thread_cluster_length|join(',')}}>, // thread_cluster_length
    ck::Sequence<{{thread_cluster_arrange_order|join(',')}}>, // thread_cluster_arrange_order
    ck::Sequence<{{src_access_order|join(',')}}>, // src_access_order
    {{src_vector_dim}}, // src_vector_dim
    {{src_scalar_per_vector}}, // src_scalar_per_vector
    {{dst_scalar_per_vector}}, // dst_scalar_per_vector
{% if add_extra_dim_flag %}
    {% if add_extra_dim==1 %}true, {% else %}false,{% endif %} //add_extra_dim
{% else %}
    {{add_extra_dim}}, // add_extra_dim
{% endif %}
"""
        )
        return template.render(**args)


@dataclass
class CBlockTransferDesc:
    m_xdl_per_wave: int
    n_xdl_per_wave: int
    m_n_block_wave_per_xdl: List[int]
    scalar_per_vector: int

    def __str__(self) -> str:
        args = deepcopy(self.__dict__)
        args["m_n_block_wave_per_xdl"] = [str(x) for x in self.m_n_block_wave_per_xdl]
        template = jinja2.Template(
            """
        {{m_xdl_per_wave}}
        _{{n_xdl_per_wave}}
        {{m_n_block_wave_per_xdl|join('_')}}S
        {{scalar_per_vector}}
        """,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return template.render(**args).replace("\n", "").replace(" ", "")

    def emit(self) -> str:
        args = deepcopy(self.__dict__)
        args["m_n_block_wave_per_xdl"] = [str(x) for x in self.m_n_block_wave_per_xdl]

        template = jinja2.Template(
            """
    {{m_xdl_per_wave}}, // m_xdl_per_wave
    {{n_xdl_per_wave}}, // n_xdl_per_wave
    ck::Sequence<{{m_n_block_wave_per_xdl|join(',')}}>, // m_n_block_wave_per_xdl
    {{scalar_per_vector}} // scalar_per_vector
    """,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return template.render(**args)


GemmTemplateMap = {
    GemmType.DeviceGemmXdl_CShuffle: jinja2.Template(
        """
using {{name}} = {{gemm_type}}<
    {{ALayout}},
    {{BLayout}},
    {{CLayout}},
    {{ADType}},
    {{BDType}},
    {{CDType}},
    {{AccDType}},
    {{CShuffleDType}},
    {{AElementwiseOperation}},
    {{BElementwiseOperation}},
    {{CElementwiseOperation}},
    {{GemmSpecialization}},
    1,
    {{tile_config}}
    {{a_block_transfer}}
    {{b_block_transfer}}
    {{c_block_transfer}}
    >;
"""
    ),
    GemmType.DeviceBatchedGemmXdl: jinja2.Template(
        """
using {{name}} = {{gemm_type}}<
    {{ADType}},
    {{BDType}},
    {{CDType}},
    {{AccDType}},
    {{ALayout}},
    {{BLayout}},
    {{CLayout}},
    {{AElementwiseOperation}},
    {{BElementwiseOperation}},
    {{CElementwiseOperation}},
    {{tile_config}}
    {{a_block_transfer}}
    {{b_block_transfer}}
    7, // CThreadTransferSrcDstVectorDim
    1  // CThreadTransferDstScalarPerVector
    >;
"""
    ),
}


@dataclass
class GemmOperation:
    gemm_type: GemmType
    A: library.TensorDesc
    B: library.TensorDesc
    C: library.TensorDesc
    a_elem_op: library.TensorOperation
    b_elem_op: library.TensorOperation
    c_elem_op: library.TensorOperation
    gemm_specialization: GemmSpecialization
    tile_desc: TileDesc
    a_block_transfer: BlockTransferDesc
    b_block_transfer: BlockTransferDesc
    c_block_transfer: CBlockTransferDesc = None

    def __str__(self) -> str:
        io_name = "{gemm_type}_{gemm_specialization}_{a_dtype}{b_dtype}{c_dtype}_{a_layout}{b_layout}{c_layout}".format(
            gemm_type=GemmTypeShortName[self.gemm_type],
            gemm_specialization=self.gemm_specialization.value,
            a_dtype=library.ShortDataTypeNames[self.A.element],
            b_dtype=library.ShortDataTypeNames[self.B.element],
            c_dtype=library.ShortDataTypeNames[self.C.element],
            a_layout=library.ShortLayoutTypeNames[self.A.layout],
            b_layout=library.ShortLayoutTypeNames[self.B.layout],
            c_layout=library.ShortLayoutTypeNames[self.C.layout],
        )
        extra_tile = ""
        if self.c_block_transfer is not None:
            if self.c_block_transfer.scalar_per_vector == 4:
                extra_tile = "_C4"
            elif self.c_block_transfer.scalar_per_vector == 1:
                extra_tile = "_C1"

        tile_name = str(self.tile_desc) + extra_tile
        return "{io_name}_{tile_name}_{c_elem_op}".format(
            io_name=io_name,
            tile_name=tile_name,
            c_elem_op=library.ShortTensorOperationNames[self.c_elem_op],
        )

    def accumulator_type(self):
        return library.DataType.f32

    def header(self):
        return GemmTypeHeader[self.gemm_type]

    def emit(self) -> str:
        template = GemmTemplateMap[self.gemm_type]
        return template.render(
            name=self.__str__(),
            gemm_type=GemmTypeTag[self.gemm_type],
            ALayout=library.LayoutTag[self.A.layout],
            BLayout=library.LayoutTag[self.B.layout],
            CLayout=library.LayoutTag[self.C.layout],
            ADType=library.DataTypeTag[self.A.element],
            BDType=library.DataTypeTag[self.B.element],
            CDType=library.DataTypeTag[self.C.element],
            AccDType=library.DataTypeTag[library.DataType.f32],
            CShuffleDType=library.DataTypeTag[self.C.element],
            AElementwiseOperation=library.TensorOperationTag[self.a_elem_op],
            BElementwiseOperation=library.TensorOperationTag[self.b_elem_op],
            CElementwiseOperation=library.TensorOperationTag[self.c_elem_op],
            GemmSpecialization=GemmSpecializationTag[self.gemm_specialization],
            tile_config=self.tile_desc.emit(),
            a_block_transfer=self.a_block_transfer.emit(),
            b_block_transfer=self.b_block_transfer.emit(),
            c_block_transfer=self.c_block_transfer.emit()
            if self.c_block_transfer is not None
            else "",
        )
