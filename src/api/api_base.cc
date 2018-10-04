 /*!
 *  Copyright (c) 2017 by Contributors
 *  Implementation of basic API functions
 * \file api_base.cc
 */
#include <dmlc/memory_io.h>
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
.set_body_typed<std::string(NodeRef)>(SaveJSON);

TVM_REGISTER_API("_load_json")
.set_body_typed<NodeRef(std::string)>(LoadJSON<NodeRef>);

TVM_REGISTER_API("_TVMSetStream")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    TVMSetStream(args[0], args[1], args[2]);
  });
TVM_REGISTER_API("_save_param_dict")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    CHECK_EQ(args.size() % 2, 0u);
    constexpr uint64_t TVMNDArrayListMagic = 0xF7E58D4F05049CB7;
    size_t num_params = args.size() / 2;
    std::vector<std::string> names;
    names.reserve(num_params);
    std::vector<DLTensor*> arrays;
    arrays.reserve(num_params);
    for (size_t i = 0; i < num_params * 2; i += 2) {
      names.emplace_back(args[i].operator std::string());
      arrays.emplace_back(args[i + 1].operator DLTensor*());
    }
    std::string bytes;
    dmlc::MemoryStringStream strm(&bytes);
    dmlc::Stream* fo = &strm;
    uint64_t header = TVMNDArrayListMagic, reserved = 0;
    fo->Write(header);
    fo->Write(reserved);
    fo->Write(names);
    {
      uint64_t sz = static_cast<uint64_t>(arrays.size());
      fo->Write(sz);
      for (size_t i = 0; i < sz; ++i) {
        tvm::runtime::SaveDLTensor(fo, arrays[i]);
      }
    }
    TVMByteArray arr;
    arr.data = bytes.c_str();
    arr.size = bytes.length();
    *rv = arr;
  });

}  // namespace tvm
