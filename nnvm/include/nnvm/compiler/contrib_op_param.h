/*!
 *  Copyright (c) 2017 by Contributors
 * \file contrib_op_param.h
 * \brief Additional parameters for compiler optimized operators.
 */
#ifndef NNVM_COMPILER_CONTRIB_OP_PARAM_H_
#define NNVM_COMPILER_CONTRIB_OP_PARAM_H_

#include <dmlc/parameter.h>
#include <string>

namespace nnvm {
namespace compiler {

/*! \brief Parameters of layout transform operator */
struct LayoutTransformParam : public dmlc::Parameter<LayoutTransformParam> {
  std::string src_layout;
  std::string dst_layout;

  DMLC_DECLARE_PARAMETER(LayoutTransformParam) {
    DMLC_DECLARE_FIELD(src_layout);
    DMLC_DECLARE_FIELD(dst_layout);
  }
};
}  // namespace compiler
}  // namespace nnvm

#endif  // NNVM_COMPILER_CONTRIB_OP_PARAM_H_
