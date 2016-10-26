/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions
 * \file c_api_impl.cc
 */
#include <tvm/expr.h>
#include "./c_api_registry.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::tvm::APIFunctionReg);
}  // namespace dmlc

namespace tvm {

using ArgStack = const std::vector<APIVariantValue>;
using RetValue = APIVariantValue;

}  // namespace tvm
