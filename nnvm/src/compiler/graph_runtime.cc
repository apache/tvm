/*!
 * Copyright (c) 2017 by Contributors
 * \file graph_runtime.cc
 * \brief Interface code with TVM graph runtime.
*/
#include <dmlc/memory_io.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/serializer.h>
#include "./graph_runtime.h"

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

bool SaveDLTensor(dmlc::Stream* strm, DLTensor* tensor) {
  uint64_t header = kTVMNDArrayMagic, reserved = 0;
  strm->Write(header);
  strm->Write(reserved);
  strm->Write(tensor->ctx);
  strm->Write(tensor->ndim);
  strm->Write(tensor->dtype);
  int ndim = tensor->ndim;
  strm->WriteArray(tensor->shape, ndim);

  int type_bytes = tensor->dtype.bits / 8;
  int64_t num_elems = 1;
  for (int i = 0; i < ndim; ++i) {
    num_elems *= tensor->shape[i];
  }
  int64_t data_byte_size = type_bytes * num_elems;
  strm->Write(data_byte_size);
  // handle endianness of data correctly.
  if (DMLC_IO_NO_ENDIAN_SWAP) {
    strm->Write(tensor->data, data_byte_size);
  } else {
    uint8_t* dptr = reinterpret_cast<uint8_t*>(tensor->data);
    std::vector<uint8_t> bytes(dptr, dptr + data_byte_size);
    dmlc::ByteSwap(dmlc::BeginPtr(bytes), type_bytes, num_elems);
    strm->Write(dmlc::BeginPtr(bytes), data_byte_size);
  }
  return true;
}

DLTensor* LoadDLTensor(dmlc::Stream* strm) {
  uint64_t header, reserved;
  CHECK(strm->Read(&header))
      << "Invalid DLTensor file format";
  CHECK(strm->Read(&reserved))
      << "Invalid DLTensor file format";
  CHECK(header == kTVMNDArrayMagic)
      << "Invalid DLTensor file format";
  DLTensor tensor;
  CHECK(strm->Read(&(tensor.ctx)))
      << "Invalid DLTensor file format";
  CHECK(strm->Read(&(tensor.ndim)))
      << "Invalid DLTensor file format";
  CHECK(strm->Read(&(tensor.dtype)))
      << "Invalid DLTensor file format";
  std::vector<int64_t> shape(tensor.ndim);
  if (tensor.ndim != 0) {
    CHECK(strm->ReadArray(&shape[0], tensor.ndim))
        << "Invalid DLTensor file format";
  }
  DLTensor* ret;
  CHECK_EQ(TVMArrayAlloc(shape.data(),
                         tensor.ndim,
                         tensor.dtype.code,
                         tensor.dtype.bits,
                         tensor.dtype.lanes,
                         static_cast<int>(tensor.ctx.device_type),
                         tensor.ctx.device_id,
                         &ret), 0) << TVMGetLastError();
  int64_t num_elems = 1;
  int elem_bytes = (ret->dtype.bits + 7) / 8;
  for (int i = 0; i < ret->ndim; ++i) {
    num_elems *= ret->shape[i];
  }
  int64_t data_byte_size;
  CHECK(strm->Read(&data_byte_size))
      << "Invalid DLTensor file format";
  CHECK(data_byte_size == num_elems * elem_bytes)
      << "Invalid DLTensor file format";
  CHECK(strm->Read(ret->data, data_byte_size))
      << "Invalid DLTensor file format";
  if (!DMLC_IO_NO_ENDIAN_SWAP) {
    dmlc::ByteSwap(ret->data, elem_bytes, num_elems);
  }
  return ret;
}

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
        SaveDLTensor(fo, arrays[i]);
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
    std::vector<DLTensor*> data;
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
    for (size_t i = 0; i < size; ++i) {
      data.push_back(LoadDLTensor(strm));
    }
    auto packed = [data, names](TVMArgs args, TVMRetValue* rv) {
      int code = args[0];
      if (code == 0) {
        *rv = static_cast<int64_t>(data.size());
      } else if (code == 1) {
        int index = args[1];
        *rv = names[index];
      } else {
        CHECK_EQ(code, 2);
        int index = args[1];
        *rv = static_cast<void*>(data[index]);
      }
    };
    *rv = PackedFunc(packed);
  });
}  // namespace compiler
}  // namespace nnvm
