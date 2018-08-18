 /*!
 *  Copyright (c) 2017 by Contributors
 *  Implementation of basic API functions
 * \file api_base.cc
 */
#include <tvm/expr.h>
#include <tvm/tensor.h>
#include <tvm/api_registry.h>

namespace tvm {
TVM_REGISTER_API("_format_str")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    CHECK(args[0].type_code() == kNodeHandle);
    std::ostringstream os;
    os << args[0].operator NodeRef();
    *ret = os.str();
  });

TVM_REGISTER_API("_raw_ptr")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    CHECK(args[0].type_code() == kNodeHandle);
    *ret = reinterpret_cast<int64_t>(
        args[0].node_sptr().get());
  });

TVM_REGISTER_API("_save_json")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    *ret = SaveJSON(args[0]);
  });

TVM_REGISTER_API("_load_json")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    *ret = LoadJSON<NodeRef>(args[0]);
  });

TVM_REGISTER_API("_nop")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
  });

// internal fucntion used for debug and testing purposes
TVM_REGISTER_API("_ndarray_use_count")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    runtime::NDArray nd = args[0];
    // substract the current one
    *ret = (nd.use_count() - 1);
  });

TVM_REGISTER_API("_TVMSetStream")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    TVMSetStream(args[0], args[1], args[2]);
  });

TVM_DLL::dmlc::Registry<::tvm::NodeFactoryReg > * GetTVMNodeFactoryRegistry() {
  return ::dmlc::Registry<::tvm::NodeFactoryReg>::Get();
}
}  // namespace tvm
