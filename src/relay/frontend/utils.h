/*!
 *  Copyright (c) 2018 by Contributors
 * \file relay/backend/compile_engine.h
 * \brief Internal compilation engine handle function cache.
 *  and interface to low level code generation.
 */
#ifndef TVM_RELAY_FRONTEND_UTILS_H_
#define TVM_RELAY_FRONTEND_UTILS_H_

#include <dmlc/json.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/pass.h>
#include <tvm/relay/type.h>
#include <tvm/tvm.h>

#include <typeinfo>
#include <string>

namespace tvm {
namespace relay {
namespace frontend {
/*!
 * \brief Get the Pakced Func
 *
 * \param func_name
 * \return const PackedFunc*
 */
inline const PackedFunc* GetPakcedFunc(const std::string& func_name) {
  return tvm::runtime::Registry::Get(func_name);
}
/*!
 * \brief Convert type to string
 *
 * \param typ
 * \return std::string string format of type
 */
std::string DType2String(const tvm::Type typ) {
  std::ostringstream os;
  auto tvm_type = Type2TVMType(typ);
  if (tvm_type.code == kDLFloat) {
    os << "float";
  } else {
    LOG(FATAL) << "Not implemented";
  }
  os << typ.bits();
  return os.str();
}

}  // namespace frontend
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_FRONTEND_UTILS_H_
