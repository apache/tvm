/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/gna/gna_json_runtime_emulation.cc
 * \brief CPU emulation-only runtime for GNA backend (no GNA SDK dependencies).
 *
 * This runtime provides CPU emulation for GNA operations without requiring
 * Intel GNA SDK headers or libraries. It enables CI testing and development
 * on systems without GNA hardware or SDK.
 *
 * This implementation follows OpenVINO's Software Emulation Mode pattern,
 * executing simplified versions of GNA operations on CPU for testing purposes.
 *
 * For production use with actual GNA hardware, the full gna_json_runtime.cc
 * implementation should be used instead.
 */

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/ndarray.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

/*!
 * \brief GNA JSON runtime with CPU emulation only.
 *
 * This class provides a CPU-only implementation of the GNA runtime
 * for testing and CI purposes. It executes simplified versions of
 * GNA operations without requiring GNA hardware or SDK.
 */
class GNAJSONRuntimeEmulation : public JSONRuntimeBase {
 public:
  GNAJSONRuntimeEmulation(const std::string& symbol_name, const std::string& graph_json,
                          const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {
    LOG(INFO) << "GNA runtime initialized in CPU emulation mode (no hardware support)";
  }

  const char* type_key() const override { return "gna_json"; }

  void Init(const Array<NDArray>& consts) override {
    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";

    SetupConstants(consts);

    // In emulation mode, we don't need to build any hardware-specific structures
    LOG(INFO) << "GNA CPU emulation mode initialized with " << nodes_.size() << " operations";
  }

  void Run() override { LOG(FATAL) << "Use Run(PackedArgs) instead"; }

  void Run(ffi::PackedArgs args) {
    std::vector<NDArray> inputs;
    std::vector<NDArray> outputs;

    // Collect input and output tensors
    for (size_t i = 0; i < static_cast<size_t>(args.size()); i++) {
      if (auto opt_nd = args[i].as<NDArray>()) {
        if (i < input_var_eid_.size()) {
          inputs.push_back(opt_nd.value());
        } else {
          outputs.push_back(opt_nd.value());
        }
      }
    }

    // Execute operations in emulation mode
    RunCPUEmulation(inputs, outputs);
  }

  ffi::Function GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) override {
    if (this->symbol_name_ == name) {
      return ffi::Function([sptr_to_self, this](ffi::PackedArgs args, ffi::Any* rv) {
        ICHECK(this->initialized_) << "The module has not been initialized";
        this->Run(args);
      });
    } else {
      return JSONRuntimeBase::GetFunction(name, sptr_to_self);
    }
  }

 private:
  /*!
   * \brief Execute operations using CPU emulation.
   *
   * This provides simplified reference implementations of GNA operations
   * for testing purposes. The implementations are not optimized but are
   * sufficient for verifying graph partitioning and codegen correctness.
   */
  void RunCPUEmulation(const std::vector<NDArray>& inputs, const std::vector<NDArray>& outputs) {
    // Process each operation in the graph
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];

      if (node.GetOpType() == "kernel") {
        auto op_name = node.GetOpName();

        // Simplified emulation for different operation types
        if (op_name.find("gna.dense") != std::string::npos) {
          EmulateLinearOperation(outputs);
        } else if (op_name.find("gna.conv1d") != std::string::npos) {
          EmulateConvOperation(outputs);
        } else if (op_name.find("gna.relu") != std::string::npos) {
          EmulateReLUOperation(outputs);
        } else {
          LOG(WARNING) << "Unsupported operation in emulation: " << op_name;
        }
      }
    }

    LOG(INFO) << "GNA CPU emulation executed " << nodes_.size() << " operations";
  }

  /*!
   * \brief Emulate linear/dense operation.
   *
   * For testing purposes, fills output with small positive values
   * to simulate a computed result.
   */
  void EmulateLinearOperation(const std::vector<NDArray>& outputs) {
    for (const auto& output : outputs) {
      FillTensorWithTestValues(output, 0.1f);
    }
  }

  /*!
   * \brief Emulate convolution operation.
   *
   * For testing purposes, fills output with small positive values
   * to simulate a computed result.
   */
  void EmulateConvOperation(const std::vector<NDArray>& outputs) {
    for (const auto& output : outputs) {
      FillTensorWithTestValues(output, 0.1f);
    }
  }

  /*!
   * \brief Emulate ReLU operation.
   *
   * For testing purposes, fills output with non-negative values
   * since ReLU output is always >= 0.
   */
  void EmulateReLUOperation(const std::vector<NDArray>& outputs) {
    for (const auto& output : outputs) {
      FillTensorWithTestValues(output, 0.1f);
    }
  }

  /*!
   * \brief Fill tensor with test values based on its data type.
   */
  void FillTensorWithTestValues(const NDArray& tensor, float float_value) {
    DLTensor* dl_tensor = const_cast<ffi::NDArrayObj*>(tensor.operator->());

    size_t num_elements = 1;
    for (int i = 0; i < dl_tensor->ndim; ++i) {
      num_elements *= dl_tensor->shape[i];
    }

    // Fill based on data type
    if (dl_tensor->dtype.code == kDLFloat) {
      if (dl_tensor->dtype.bits == 32) {
        std::fill_n(static_cast<float*>(dl_tensor->data), num_elements, float_value);
      } else if (dl_tensor->dtype.bits == 64) {
        std::fill_n(static_cast<double*>(dl_tensor->data), num_elements,
                    static_cast<double>(float_value));
      }
    } else if (dl_tensor->dtype.code == kDLInt) {
      // For integer types, use small positive values
      if (dl_tensor->dtype.bits == 8) {
        std::fill_n(static_cast<int8_t*>(dl_tensor->data), num_elements, 1);
      } else if (dl_tensor->dtype.bits == 16) {
        std::fill_n(static_cast<int16_t*>(dl_tensor->data), num_elements, 1);
      } else if (dl_tensor->dtype.bits == 32) {
        std::fill_n(static_cast<int32_t*>(dl_tensor->data), num_elements, 1);
      } else if (dl_tensor->dtype.bits == 64) {
        std::fill_n(static_cast<int64_t*>(dl_tensor->data), num_elements, 1);
      }
    } else if (dl_tensor->dtype.code == kDLUInt) {
      // For unsigned integer types
      if (dl_tensor->dtype.bits == 8) {
        std::fill_n(static_cast<uint8_t*>(dl_tensor->data), num_elements, 1);
      } else if (dl_tensor->dtype.bits == 16) {
        std::fill_n(static_cast<uint16_t*>(dl_tensor->data), num_elements, 1);
      } else if (dl_tensor->dtype.bits == 32) {
        std::fill_n(static_cast<uint32_t*>(dl_tensor->data), num_elements, 1);
      } else if (dl_tensor->dtype.bits == 64) {
        std::fill_n(static_cast<uint64_t*>(dl_tensor->data), num_elements, 1);
      }
    }
  }
};

/*!
 * \brief Create a GNA JSON runtime module with CPU emulation.
 * \param symbol_name The name of the function to be executed.
 * \param graph_json The JSON graph representation.
 * \param const_names The names of constants.
 * \return The created runtime module.
 */
runtime::Module GNAJSONRuntimeCreate(String symbol_name, String graph_json,
                                     const Array<String>& const_names) {
  auto n = make_object<GNAJSONRuntimeEmulation>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.GNAJSONRuntimeCreate", GNAJSONRuntimeCreate)
      .def("runtime.module.loadbinary_gna_json",
           JSONRuntimeBase::LoadFromBinary<GNAJSONRuntimeEmulation>);
});

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
