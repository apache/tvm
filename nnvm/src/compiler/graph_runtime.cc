/*!
 * Copyright (c) 2017 by Contributors
 * \file graph_runtime.cc
 * \brief Interface code with TVM graph runtime.
*/
#include <dmlc/memory_io.h>
#include "graph_runtime.h"

namespace nnvm {
namespace compiler {

using tvm::runtime::TVMArgs;
using tvm::runtime::TVMRetValue;
using tvm::runtime::PackedFunc;

DMLC_REGISTER_PARAMETER(TVMOpParam);

// parser
inline void TVMOpParamParser(nnvm::NodeAttrs* attrs) {
  TVMOpParam param;
  param.Init(attrs->dict);
  attrs->parsed = std::move(param);
}

NNVM_REGISTER_OP(tvm_op)
.set_attr_parser(TVMOpParamParser)
.set_num_inputs([](const NodeAttrs& attrs) {
    const TVMOpParam& param = nnvm::get<TVMOpParam>(attrs.parsed);
    return param.num_inputs;
  })
.set_num_outputs([](const NodeAttrs& attrs) {
    const TVMOpParam& param = nnvm::get<TVMOpParam>(attrs.parsed);
    return param.num_outputs;
  });


TVM_REGISTER_GLOBAL("nnvm.compiler._save_param_dict")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    CHECK_EQ(args.size() % 2, 0u);
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
    uint64_t header = kTVMNDArrayListMagic, reserved = 0;
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


TVM_REGISTER_GLOBAL("nnvm.compiler._load_param_dict")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    std::string bytes = args[0];
    std::vector<std::string> names;
    dmlc::MemoryStringStream memstrm(&bytes);
    dmlc::Stream* strm = &memstrm;
    uint64_t header, reserved;
    CHECK(strm->Read(&header))
        << "Invalid parameters file format";
    CHECK(header == kTVMNDArrayListMagic)
        << "Invalid parameters file format";
    CHECK(strm->Read(&reserved))
        << "Invalid parameters file format";
    CHECK(strm->Read(&names))
        << "Invalid parameters file format";
    uint64_t sz;
    strm->Read(&sz, sizeof(sz));
    size_t size = static_cast<size_t>(sz);
    CHECK(size == names.size())
        << "Invalid parameters file format";
    tvm::Array<NDArrayWrapper> ret;
    for (size_t i = 0; i < size; ++i) {
      tvm::runtime::NDArray temp;
      temp.Load(strm);
      auto n = tvm::make_node<NDArrayWrapperNode>();
      n->name = std::move(names[i]);
      n->array = temp;
      ret.push_back(NDArrayWrapper(n));
    }
    *rv = ret;
  });

TVM_REGISTER_NODE_TYPE(NDArrayWrapperNode);
}  // namespace compiler
}  // namespace nnvm
