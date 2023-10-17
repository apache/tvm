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
"""Templates for ComposableKernel GEMM kernels."""
import enum
from copy import deepcopy
from dataclasses import dataclass
from enum import auto as enum_auto
from typing import List
from jinja2 import Template

from tvm.contrib.composable_kernel import library
from .kernel_factory.gemm import library as gemm


@dataclass
class GemmOpAttributes:
    a_data_type: library.DataType
    b_data_type: library.DataType
    c_data_type: library.DataType
    a_layout: library.LayoutType
    b_layout: library.LayoutType
    c_layout: library.LayoutType
    a_element_op: library.TensorOperation
    b_element_op: library.TensorOperation
    c_element_op: library.TensorOperation

    def emit(self):
        args = deepcopy(self.__dict__)
        template = Template(
            """
using ADataType = {{ADataType}};
using BDataType = {{BDataType}};
using CDataType = {{CDataType}};

using ALayout = {{ALayout}};
using BLayout = {{BLayout}};
using CLayout = {{CLayout}};

using AElementOp = {{AElementOp}};
using BElementOp = {{BElementOp}};
using CElementOp = {{CElementOp}};
		"""
        )
        return template.render(
            ADataType=library.DataTypeTag[self.a_data_type],
            BDataType=library.DataTypeTag[self.b_data_type],
            CDataType=library.DataTypeTag[self.c_data_type],
            ALayout=library.LayoutTag[self.a_layout],
            BLayout=library.LayoutTag[self.b_layout],
            CLayout=library.LayoutTag[self.c_layout],
            AElementOp=library.TensorOperationTag[self.a_element_op],
            BElementOp=library.TensorOperationTag[self.b_element_op],
            CElementOp=library.TensorOperationTag[self.c_element_op],
        )


@dataclass
class GemmOpArgument:
    M: int
    N: int
    K: int
    B: int
    a_stride: int
    b_stride: int
    c_stride: int
    a_batch_stride: int
    b_batch_stride: int
    c_batch_stride: int
    batched: bool

    def emit(self, gemm_name: str, lhs_arg: str, rhs_arg: str, out_arg: str):
        args = deepcopy(self.__dict__)
        template = Template(
            """{{gemm_name}}MakeArgumentPointer(
	(ADataType*){{lhs_arg}}, // p_a
	(BDataType*){{rhs_arg}}, // p_b
	(CDataType*){{out_arg}}, // p_c
	{{M}}, // M
	{{N}}, // N
	{{K}}, // K
	{{a_stride}}, // StrideA
	{{b_stride}}, // StrideB
	{{c_stride}}, // StrideC
	{% if batched %}
	{{a_batch_stride}}, // BatchStrideA
	{{b_batch_stride}}, // BatchStrideB
	{{c_batch_stride}}, // BatchStrideC
	{{B}}, // Batch
	{% endif %}
	AElementOp(), // a_element_op
	BElementOp(), // b_element_op
	CElementOp() // c_element_op
);"""
        )
        return template.render(
            gemm_name=gemm_name, lhs_arg=lhs_arg, rhs_arg=rhs_arg, out_arg=out_arg, **args
        )


class GemmOpEmitter(object):
    """Emit C++ source code ComposableKernel GEMM kernels."""

    def __init__(self):
        self.template = Template(
            """
	{{gemm_attributes}}

    {{instance_def}}

    auto gemm = {{instance_name}}();
    auto argument_ptr = {{gemm_argument}}
    auto invoker = gemm.MakeInvoker();
    invoker.Run(argument_ptr.get(), StreamConfig{nullptr, false});
    """
        )

    def emit(
        self,
        lhs_arg: str,
        rhs_arg: str,
        gemm_attributes: GemmOpAttributes,
        gemm_argument: GemmOpArgument,
        instance: gemm.GemmOperation,
    ):
        src = self.template.render(
            gemm_attributes=gemm_attributes.emit(),
            instance_def=instance.emit(),
            instance_name=str(instance),
            gemm_argument=gemm_argument.emit(
                "gemm.",
                f"{lhs_arg}->data",
                f"{rhs_arg}->data",
                "out0->data",
            ),
        )
        return src


class GemmInstanceEmitter(object):
    """Emit C++ source for declaring ComposableKernel GEMM kernels."""

    def __init__(self):
        self.header_template = Template(
            """
#include "ck/ck.hpp"
{% for header in instance_headers %}
#include "{{header}}"
{% endfor %}

/************************************************************************************************/
/* Gemm Op Attributes Definitions                                                               */
/************************************************************************************************/
{{gemm_attributes}}

/************************************************************************************************/
/* Gemm Instance Factory                                                                        */
/************************************************************************************************/
using InstanceList = std::vector<std::unique_ptr<{{gemm_base_type}}<
        ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AElementOp,
        BElementOp, CElementOp>>>;

{% for name in instance_names %}void get_{{name}}(InstanceList &op_instances);
{% endfor %}
		"""
        )

        self.cc_template = Template(
            """
#include "{{header_path}}"

{{instance_def}}

void get_{{instance_name}}(InstanceList &op_instances) {
    op_instances.push_back(std::make_unique<{{instance_name}}>());
}
		"""
        )

    def emit_header(self, gemm_attributes: GemmOpAttributes, instances: List[gemm.GemmOperation]):
        src = self.header_template.render(
            gemm_base_type=gemm.GemmBaseTypeTag[gemm.GemmType2GemmBaseType[instances[0].gemm_type]],
            gemm_attributes=gemm_attributes.emit(),
            instance_headers=list(set([inst.header() for inst in instances])),
            instance_names=[str(inst) for inst in instances],
        )
        return src

    def emit_cc(self, instance, header_path):
        src = self.cc_template.render(
            instance_def=instance.emit(), instance_name=str(instance), header_path=header_path
        )
        return src


class GemmProfilerEmitter(object):
    """Emit a C++ source for profiling ComposableKernel kernels."""

    def __init__(self):
        self.template = Template(
            """
#include <iostream>
#include <limits>

#include "ck/ck.hpp"
#include <hip/hip_runtime.h>
#include "{{instance_header_path}}"

/************************************************************************************************/
/* Helper Functions and Macros.                                                                 */
/************************************************************************************************/
#define HIP_CHECK(status)                                                                        \\
  {                                                                                              \\
    hipError_t error = status;                                                                   \\
    if (error != hipSuccess) {                                                                   \\
      std::cerr << "HIP runtime error: " << hipGetErrorString(error) << ". " << __FILE__ << ": " \\
                << __LINE__ << "in function: " << __func__;                                      \\
      exit(EXIT_FAILURE);                                                                        \\
    }                                                                                            \\
  }

template <typename DataType>
hipError_t AllocateMatrix(DataType** matrix, int ldm, int rows, int columns) {
  size_t sizeof_matrix = sizeof(DataType) * ldm * rows * columns;
  return hipMalloc((void**)matrix, sizeof_matrix);
}

/************************************************************************************************/
/* Gemm Instance Definitions.                                                                   */
/************************************************************************************************/
void GetInstances(InstanceList &op_instances) {
{% for name in instance_names %}	get_{{name}}(op_instances);
{% endfor %}}

/************************************************************************************************/
/* Gemm Benchmark Entry Function                                                                */
/************************************************************************************************/
void BenchComposableKernelGemm(int M, int N, int K, int LD) {
  ADataType* A;
  BDataType* B;
  CDataType* C;

  // 1. Prepare input, output data buffer
  HIP_CHECK(AllocateMatrix<ADataType>(&A, LD, M, K));

  auto result = AllocateMatrix<BDataType>(&B, LD, K, N);
  if (result != hipSuccess)
    HIP_CHECK(hipFree(A));
  HIP_CHECK(result);

  result = AllocateMatrix<CDataType>(&C, LD, M, N);
  if (result != hipSuccess) {
    HIP_CHECK(hipFree(A));
    HIP_CHECK(hipFree(B));
  }
  HIP_CHECK(result);
  
  // 2. Initialize metrics
  float bestCandLatency = std::numeric_limits<float>::max();
  int bestCandIndex = -1;

  // 3. Fetch all instances
  InstanceList instances;
  GetInstances(instances);

  // 4. Benchmark all instances
  for (int i = 0; i < instances.size(); i++) {
    auto& inst = instances[i];

    auto argPtr = {{gemm_argument}}

    auto invokerPtr = inst->MakeInvokerPointer();

    if (!inst->IsSupportedArgument(argPtr.get())) {
      continue;
    }

    float latency = invokerPtr->Run(argPtr.get(), StreamConfig{nullptr, true});

    {% if debug %}
    std::cout<< latency << "ms | "<< inst->GetTypeString()<<std::endl;
    {% endif %}

    if (latency < bestCandLatency) {
      bestCandLatency = latency;
      bestCandIndex = i;
    }
  }

  HIP_CHECK(hipFree(A));
  HIP_CHECK(hipFree(B));
  HIP_CHECK(hipFree(C));

  {% if debug %}
  std::cout<< "Best latency: " << bestCandLatency << "ms | "<< 
  instances[bestCandIndex]->GetTypeString() <<std::endl;
  {% endif %}

  std::cout<< bestCandIndex <<std::endl;

  return;
}

int main(int argc, char* argv[]) {
  BenchComposableKernelGemm({{M}}, {{N}}, {{K}}, {{LD}});
}
"""
        )

    def emit(
        self,
        gemm_argument: GemmOpArgument,
        instance_header_path: str,
        instances: List[gemm.GemmOperation],
    ):
        src = self.template.render(
            gemm_argument=gemm_argument.emit("inst->", "A", "B", "C"),
            instance_header_path=instance_header_path,
            instance_names=[str(instance) for instance in instances],
            M=gemm_argument.M,
            N=gemm_argument.N,
            K=gemm_argument.K,
            LD=gemm_argument.B,
            debug=True,
        )
        return src
