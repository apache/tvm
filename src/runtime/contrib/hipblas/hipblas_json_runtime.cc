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
 * \file src/runtime/contrib/hipblas/hipblas_json_runtime.cc
 * \brief A simple JSON runtime for HIPBLAS.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"
#include "hipblas_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {
using namespace tvm::runtime;
using namespace tvm::runtime::json;
class HipblasJSONRuntime : public JSONRuntimeBase {
 public:
  HipblasJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                     const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  void Init(const Array<NDArray>& consts) override {}

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) override {
    // JSONRuntimeBase::SetInputOutputBuffers(...) is not thread safe. Since HipblasJSONRuntime
    // can be used by multiple GPUs running on different threads, we avoid using that function
    // and directly call hipBLAS on the inputs from TVMArgs.
    if (this->symbol_name_ == name) {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK(this->initialized_) << "The module has not been initialized";
        this->Run(args);
      });
    } else {
      return JSONRuntimeBase::GetFunction(name, sptr_to_self);
    }
  }

  const char* type_key() const override { return "hipblas_json"; }  // May be overridden

  void Run(TVMArgs args) {
    auto* entry_ptr = tvm::contrib::HipBlasLtThreadEntry::ThreadLocal();

    auto func = tvm::runtime::Registry::Get("runtime.get_rocm_stream");
    ICHECK(func != nullptr);
    hipStream_t stream = static_cast<hipStream_t>((*func)().operator void*());

    std::vector<const DLTensor*> dl_tensors(NumEntries());

    for (size_t i = 0; i < static_cast<size_t>(args.size()); i++) {
      auto eid = i < input_var_eid_.size() ? input_var_eid_[i]
                                           : EntryID(outputs_[i - input_var_eid_.size()]);
      ICHECK(args[i].type_code() == kTVMNDArrayHandle || args[i].type_code() == kTVMDLTensorHandle)
          << "Expect NDArray or DLTensor as inputs";

      const DLTensor* arg;
      if (args[i].IsObjectRef<NDArray>()) {
        NDArray arr = args[i];
        arg = arr.operator->();
      } else {
        arg = args[i].operator DLTensor*();
      }

      dl_tensors[eid] = arg;
    }

    auto get_input = [this, &dl_tensors](const JSONGraphNode& node, int idx) {
      ICHECK_LT(idx, node.GetInputs().size());
      auto eid = EntryID(node.GetInputs()[idx]);
      ICHECK(eid < dl_tensors.size());
      return dl_tensors[eid];
    };

    auto get_inputs = [=](const JSONGraphNode& node, bool has_bias) {
      const DLTensor* bias = nullptr;
      if (has_bias) {
        bias = get_input(node, 2);
      }
      return std::make_tuple(get_input(node, 0), get_input(node, 1), bias);
    };

    for (size_t i = 0; i < nodes_.size(); ++i) {
      const auto& node = nodes_[i];
      if (node.GetOpType() == "kernel") {
        auto op_name = node.GetOpName();
        uint32_t output_eid = EntryID(outputs_[0]);
        auto out_ptr = dl_tensors[output_eid];
        bool transa = false;
        bool transb = false;
        hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;

        if (op_name.find("transposed") != std::string::npos) {
          transb = true;
        }

        if (op_name.find("relu") != std::string::npos) {
          epilogue = HIPBLASLT_EPILOGUE_RELU_BIAS;
        } else if (op_name.find("gelu") != std::string::npos) {
          epilogue = HIPBLASLT_EPILOGUE_GELU_BIAS;
        } else if (op_name.find("bias") != std::string::npos) {
          epilogue = HIPBLASLT_EPILOGUE_BIAS;
        }

        auto [a_ptr, b_ptr, bias_ptr] = get_inputs(node, epilogue != HIPBLASLT_EPILOGUE_DEFAULT);

        tvm::contrib::CallHipblasLt(entry_ptr->handle, stream, entry_ptr->matmul_pref_desc, a_ptr,
                                    b_ptr, bias_ptr, out_ptr, transa, transb,
                                    entry_ptr->workspace_ptr, entry_ptr->workspace_size, epilogue);
      }
    }
  }

  void Run() override { LOG(FATAL) << "Unreachable"; }
};

runtime::Module HipblasJSONRuntimeCreate(String symbol_name, String graph_json,
                                         const Array<String>& const_names) {
  auto n = make_object<HipblasJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.HipblasJSONRuntimeCreate").set_body_typed(HipblasJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_hipblas_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<HipblasJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
