/*!
 * Copyright (c) 2017 by Contributors
 * \file graph_runtime.h
 * \brief Interface code with TVM graph runtime.
*/
#ifndef NNVM_COMPILER_GRAPH_RUNTIME_H_
#define NNVM_COMPILER_GRAPH_RUNTIME_H_

#include <nnvm/graph.h>
#include <vector>
#include <string>

namespace nnvm {
namespace compiler {

/*! \brief Magic number for NDArray file */
constexpr uint64_t kTVMNDArrayMagic = 0xDD5E40F096B4A13F;
/*! \brief Magic number for NDArray list file  */
constexpr uint64_t kTVMNDArrayListMagic = 0xF7E58D4F05049CB7;

struct TVMOpParam : public dmlc::Parameter<TVMOpParam> {
  std::string func_name;
  uint32_t num_inputs;
  uint32_t num_outputs;
  uint32_t flatten_data;

  DMLC_DECLARE_PARAMETER(TVMOpParam) {
    DMLC_DECLARE_FIELD(func_name);
    DMLC_DECLARE_FIELD(num_inputs).set_default(1);
    DMLC_DECLARE_FIELD(num_outputs).set_default(1);
    DMLC_DECLARE_FIELD(flatten_data).set_default(0);
  }
};

}  // namespace compiler
}  // namespace nnvm
#endif   // NNVM_COMPILER_GRAPH_RUNTIME_H_
