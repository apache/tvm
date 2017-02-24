/*!
 *  Copyright (c) 2017 by Contributors
 * \file meta_data.h
 * \brief Meta data related utilities
 */
#ifndef TVM_RUNTIME_META_DATA_H_
#define TVM_RUNTIME_META_DATA_H_

#include <dmlc/json.h>
#include <string>
#include <vector>
#include "./runtime_base.h"

extern "C" {
// Function signature for generated packed function in shared library
typedef int (*BackendPackedCFunc)(void* args,
                                  int* type_codes,
                                  int num_args);
}  // extern "C"

namespace tvm {
namespace runtime {

/*! \brief function information needed by device */
struct FunctionInfo {
  std::string name;
  std::vector<TVMType> arg_types;
  std::vector<std::string> thread_axis_tags;

  void Save(dmlc::JSONWriter *writer) const {
    std::vector<std::string> sarg_types(arg_types.size());
    for (size_t i = 0; i < arg_types.size(); ++i) {
      sarg_types[i] = TVMType2String(arg_types[i]);
    }
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("arg_types", sarg_types);
    writer->WriteObjectKeyValue("thread_axis_tags", thread_axis_tags);
    writer->EndObject();
  }

  void Load(dmlc::JSONReader *reader) {
    dmlc::JSONObjectReadHelper helper;
    std::vector<std::string> sarg_types;
    helper.DeclareField("name", &name);
    helper.DeclareField("arg_types", &sarg_types);
    helper.DeclareField("thread_axis_tags", &thread_axis_tags);
    helper.ReadAllFields(reader);
    arg_types.resize(sarg_types.size());
    for (size_t i = 0; i < arg_types.size(); ++i) {
      arg_types[i] = String2TVMType(sarg_types[i]);
    }
  }
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_META_DATA_H_
