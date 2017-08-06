/*!
 *  Copyright (c) 2017 by Contributors
 * \file graph_executor_ext.cc
 */
#include "./graph_executor.h"

namespace tvm {
namespace contrib {

bool SaveDLTensor(dmlc::Stream* strm, DLTensor* tensor) {
    uint64_t header = kTVMNDArrayMagic, reserved = 0;
    strm->Write(&header, sizeof(header));
    strm->Write(&reserved, sizeof(reserved));

    strm->Write(&tensor->ctx, sizeof(tensor->ctx));
    strm->Write(&tensor->ndim, sizeof(tensor->ndim));
    strm->Write(&tensor->dtype, sizeof(tensor->dtype));

    int ndim = tensor->ndim;
    strm->Write(tensor->shape, sizeof(int64_t) * ndim);

    int type_size = tensor->dtype.bits / 8;
    int64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
      size *= tensor->shape[i];
    }
    int64_t data_byte_size = type_size * size;
    strm->Write(&data_byte_size, sizeof(data_byte_size));
    strm->Write(tensor->data, data_byte_size);
    return true;
}

TVM_REGISTER_GLOBAL("tvm_graph._save_param_dict")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    std::string fname = args[0];
    int num_params = args[1];
    std::vector<std::string> names;
    names.reserve(num_params);
    std::vector<DLTensor*> arrays;
    arrays.reserve(num_params);
    for (int i = 2; i < (2 + 2*num_params); i += 2) {
      names.emplace_back(args[i].operator std::string());
      arrays.emplace_back(args[i+1].operator DLTensor*());
    }

    std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname.c_str(), "w"));
    uint64_t header = kTVMNDArrayListMagic, reserved = 0;
    fo->Write(&header, sizeof(header));
    fo->Write(&reserved, sizeof(reserved));

    fo->Write(names);
    {
      uint64_t sz = static_cast<uint64_t>(arrays.size());
      fo->Write(&sz, sizeof(sz));
      for (size_t i = 0; i < sz; ++i) {
        SaveDLTensor(fo.get(), arrays[i]);
      }
    }
  });

// Create executor
tvm::runtime::Module CreateExecutor(nnvm::Graph g, TVMContext ctx) {
  std::shared_ptr<GraphExecutor> exec =
      std::make_shared<GraphExecutor>();
  exec->Init(g, ctx);
  return tvm::runtime::Module(exec);
}

TVM_REGISTER_GLOBAL("tvm_graph._create_executor")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    void* graph_handle = args[0];
    int device_type = args[1];
    int device_id = args[2];
    TVMContext ctx{static_cast<DLDeviceType>(device_type), device_id};
    nnvm::Graph g = static_cast<nnvm::Graph*>(graph_handle)[0];
    *rv = CreateExecutor(g, ctx);
  });


TVM_REGISTER_GLOBAL("tvm_graph._get_module_from_graph")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    void* graph_handle = args[0];
    nnvm::Graph* g = static_cast<nnvm::Graph*>(graph_handle);
    *rv = g->MoveCopyAttr<tvm::runtime::Module>("module");
  });
}  // namespace contrib
}  // namespace tvm
