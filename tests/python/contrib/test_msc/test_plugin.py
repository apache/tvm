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

""" Test Plugin in MSC. """

import numpy as np

import torch
from torch import nn

import tvm.testing
from tvm import relax
from tvm.relax.transform import BindParams
from tvm.script import relax as R
from tvm.contrib.msc.plugin import build_plugins
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


def _get_externs_header():
    """Get the header source for externs"""

    return """#ifndef EXTERNS_H_
#define EXTERNS_H_

#include "plugin_base.h"

#ifdef PLUGIN_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace tvm {
namespace contrib {
namespace msc {
namespace plugin {

template <typename TAttr>
std::vector<MetaTensor> my_relu_infer(const std::vector<MetaTensor>& inputs, const TAttr& attrs,
                                      bool is_runtime) {
  std::vector<MetaTensor> outputs;
  outputs.push_back(MetaTensor(inputs[0].shape(), inputs[0].data_type(), inputs[0].layout()));
  return outputs;
}

template <typename T>
void my_relu_cpu_kernel(const DataTensor<T>& input, DataTensor<T>& output, T max_val);

template <typename T, typename TAttr>
void my_relu_cpu_compute(const DataTensor<T>& input, DataTensor<T>& output, const TAttr& attrs) {
  my_relu_cpu_kernel(input, output, T(attrs.max_val));
}

#ifdef PLUGIN_ENABLE_CUDA
template <typename T>
void my_relu_cuda_kernel(const DataTensor<T>& input, DataTensor<T>& output, T max_val,
                         const cudaStream_t& stream);

template <typename T, typename TAttr>
void my_relu_cuda_compute(const DataTensor<T>& input, DataTensor<T>& output, const TAttr& attrs,
                          const cudaStream_t& stream) {
  my_relu_cuda_kernel(input, output, T(attrs.max_val), stream);
}
#endif

}  // namespace plugin
}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // EXTERNS_H_
"""


def _get_externs_cc():
    """Get externs cc source"""
    return """#include "externs.h"

namespace tvm {
namespace contrib {
namespace msc {
namespace plugin {

template <typename T>
void my_relu_cpu_kernel(const DataTensor<T>& input, DataTensor<T>& output, T max_val) {
  const T* input_data = input.const_data();
  T* output_data = output.data();
  for (size_t i = 0; i < output.size(); i++) {
    if (input_data[i] >= max_val) {
      output_data[i] = max_val;
    } else if (input_data[i] <= 0) {
      output_data[i] = 0;
    } else {
      output_data[i] = input_data[i];
    }
  }
}

template void my_relu_cpu_kernel<float>(const DataTensor<float>& input, DataTensor<float>& output,
                                        float max_val);

}  // namespace plugin
}  // namespace msc
}  // namespace contrib
}  // namespace tvm
"""


def _get_externs_cu():
    """Get externs cu source"""

    return """#include "externs.h"

#define CU1DBLOCK 256
#define KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)

namespace tvm {
namespace contrib {
namespace msc {
namespace plugin {

inline int n_blocks(int size, int block_size) {
  return size / block_size + (size % block_size == 0 ? 0 : 1);
}

template <typename T>
__global__ static void _my_relu(const T* src, T* dst, T max_val, int n) {
  KERNEL_LOOP(i, n) {
    if (src[i] >= max_val) {
      dst[i] = max_val;
    } else if (src[i] <= 0) {
      dst[i] = 0;
    } else {
      dst[i] = src[i];
    }
  }
}

template <typename T>
void my_relu_cuda_kernel(const DataTensor<T>& input, DataTensor<T>& output, T max_val,
                         const cudaStream_t& stream) {
  const T* input_data = input.const_data();
  T* output_data = output.data();
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(output.size(), CU1DBLOCK));
  _my_relu<<<Gr, Bl, 0, stream>>>(input_data, output_data, max_val, output.size());
}

template void my_relu_cuda_kernel<float>(const DataTensor<float>& input, DataTensor<float>& output,
                                         float max_val, const cudaStream_t& stream);

}  // namespace plugin
}  // namespace msc
}  // namespace contrib
}  // namespace tvm
"""


def _create_plugin(externs_dir):
    """Create sources under source folder"""
    with open(externs_dir.relpath("externs.h"), "w") as f:
        f.write(_get_externs_header())
    with open(externs_dir.relpath("externs.cc"), "w") as f:
        f.write(_get_externs_cc())
    with open(externs_dir.relpath("externs.cu"), "w") as f:
        f.write(_get_externs_cu())
    return {
        "MyRelu": {
            "inputs": [{"name": "input", "dtype": "T"}],
            "outputs": [{"name": "output", "dtype": "T"}],
            "attrs": [{"name": "max_val", "type": "float"}],
            "support_dtypes": {"T": ["float"]},
            "externs": {
                "infer_output": {"name": "my_relu_infer", "header": "externs.h"},
                "cpu_compute": {
                    "name": "my_relu_cpu_compute",
                    "header": "externs.h",
                    "source": "externs.cc",
                },
                "cuda_compute": {
                    "name": "my_relu_cuda_compute",
                    "header": "externs.h",
                    "source": "externs.cu",
                },
            },
        }
    }


def _get_torch_model(torch_manager):
    """Build model with plugin"""

    class MyModel(nn.Module):
        """Test model with plugin"""

        def __init__(self):
            super(MyModel, self).__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)
            self.relu = torch_manager.MyRelu(max_val=0.5)
            self.maxpool = nn.MaxPool2d(kernel_size=[1, 1])

        def forward(self, data):
            data = self.conv(data)
            data = self.relu(data)
            return self.maxpool(data)

    return MyModel()


def _get_tvm_model(tvm_manager):
    """Build model with plugin"""

    block_builder = relax.BlockBuilder()
    weights = np.random.rand(6, 3, 7, 7).astype("float32")
    data = relax.Var("data", R.Tensor((1, 3, 224, 224), "float32"))
    weight = relax.Var("weight", R.Tensor(weights.shape, weights.dtype.name))
    inputs = [data, weight]
    with block_builder.function(name="main", params=inputs.copy()):
        with block_builder.dataflow():
            data = relax.op.nn.conv2d(data, weight)
            data = block_builder.emit(data, "conv2d")
            data = tvm_manager.MyRelu(data, max_val=0.5)
            data = block_builder.emit(data, "relu")
            data = relax.op.nn.max_pool2d(data)
            data = block_builder.emit(data, "max_pool2d")
            data = block_builder.emit_output(data)
        block_builder.emit_func_output(data)
    mod = block_builder.finalize()
    return BindParams("main", {"weight": tvm.nd.array(weights)})(mod)


def _build_plugin(frameworks, plugin_root):
    externs_dir = plugin_root.create_dir("externs")
    install_dir = plugin_root.create_dir("install")
    plugin = _create_plugin(externs_dir)
    managers = build_plugins(plugin, frameworks, install_dir, externs_dir=externs_dir)
    return managers


def _run_relax(relax_mod, target_name, data):
    target = tvm.target.Target(target_name)
    relax_mod = tvm.relax.transform.LegalizeOps()(relax_mod)
    if target_name == "cuda":
        with target:
            relax_mod = tvm.tir.transform.DefaultGPUSchedule()(relax_mod)
        device = tvm.cuda()
    else:
        device = tvm.cpu()
    with tvm.transform.PassContext(opt_level=3):
        relax_exec = tvm.relax.build(relax_mod, target)
        runnable = tvm.relax.VirtualMachine(relax_exec, device)
    data = tvm.nd.array(data, device)
    return runnable["main"](data).asnumpy()


def _test_tvm_plugin(manager, target):
    """Test plugin in tvm"""

    model = _get_tvm_model(manager)
    data = np.random.rand(1, 3, 224, 224).astype("float32")
    outputs = _run_relax(model, target, data)
    assert outputs.min() >= 0 and outputs.max() <= 0.5


def _test_torch_plugin(manager):
    """Test plugin in torch"""

    model = _get_torch_model(manager)
    torch_data = torch.from_numpy(np.random.rand(1, 3, 224, 224).astype("float32"))
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda:0"))
        torch_data = torch_data.to(torch.device("cuda:0"))
    outputs = model(torch_data)
    assert outputs.min() >= 0 and outputs.max() <= 0.5


def test_plugin():
    """Test the plugins"""

    frameworks = [MSCFramework.TORCH, MSCFramework.TVM]
    if tvm.get_global_func("relax.ext.tensorrt", True) is not None:
        frameworks.append(MSCFramework.TENSORRT)
    plugin_root = msc_utils.msc_dir("msc_plugin")
    managers = _build_plugin(frameworks, plugin_root)

    # test the plugin load
    _test_tvm_plugin(managers[MSCFramework.TVM], "llvm")
    if tvm.cuda().exist:
        _test_tvm_plugin(managers[MSCFramework.TVM], "cuda")
    _test_torch_plugin(managers[MSCFramework.TORCH])

    plugin_root.destory()


if __name__ == "__main__":
    tvm.testing.main()
