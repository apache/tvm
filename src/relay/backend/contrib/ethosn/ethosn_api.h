/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_ETHOSN_ETHOSN_API_H_
#define TVM_RELAY_BACKEND_CONTRIB_ETHOSN_ETHOSN_API_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ethosn_support_library/Support.hpp"
#include "ethosn_support_library/SupportQueries.hpp"

namespace tvm {
namespace relay {
namespace contrib {
namespace ethosn {

namespace sl = ::ethosn::support_library;

struct ConcatenateParams {
  sl::QuantizationInfo qInfo;
  sl::ConcatenationInfo concat_info = sl::ConcatenationInfo(1, qInfo);
  std::vector<sl::TensorInfo> input_infos;
};

struct SplitParams {
  sl::SplitInfo split_info = sl::SplitInfo(0, {});
  sl::TensorInfo input_info;
};

class ErrStrm {
 public:
  template <typename T>
  ErrStrm& operator<<(const T& val) {  // NOLINT(*)
    stream_ << val;
    return *this;
  }

 private:
  std::stringstream stream_;
  friend class EthosnError;
};

class EthosnError {
 public:
  EthosnError() {}
  explicit EthosnError(const Array<String>& msgs) : msgs(msgs) {}
  explicit EthosnError(const String& msg) { msgs.push_back(msg); }
  explicit EthosnError(const ErrStrm& err) : EthosnError(err.stream_.str()) {}

  explicit operator bool() const { return !msgs.empty(); }

  EthosnError& operator+=(const EthosnError& other) {
    msgs.insert(msgs.end(), other.msgs.begin(), other.msgs.end());
    return *this;
  }

  Array<String> msgs;
};

class EthosnAPI {
 public:
  static std::unique_ptr<sl::CompiledNetwork> Compile(std::shared_ptr<sl::Network> network,
                                                      const sl::CompilationOptions& options);

  static sl::CompilationOptions CreateOptions();

  static bool IsEthosnOp(const Call& call, const std::string& op_name);

  static EthosnError Concatenate(const Expr& expr, ConcatenateParams* params);
  static EthosnError Split(const Expr& expr, SplitParams* params);

 private:
  static EthosnError Tvm2Npu(const Array<IndexExpr>& shape, sl::TensorShape* npu_shape);
  static EthosnError Tvm2Npu(const tvm::DataType& dtype, sl::DataType* data_type);

  // Convert an array of IntImmNodes into ValueT
  // IndexT type of Array indexing variable
  // ValueT type of resulting value
  template <typename IndexT, typename ValueT>
  static EthosnError AsArray(const Array<IndexT>& arr, std::array<ValueT, 4>* v) {
    if (arr.size() > 4)
      return EthosnError(ErrStrm() << "dimensions=" << arr.size() << ", dimensions must be <= 4");
    for (size_t i = 0; i < std::min(arr.size(), 4ul); i++) {
      const PrimExpr& a = arr[i];
      const auto* intImm = a.as<IntImmNode>();
      if (intImm->value > std::numeric_limits<ValueT>::max()) {
        return EthosnError(ErrStrm() << "axis size=" << intImm->value << ", axis size must be <= "
                                     << std::numeric_limits<ValueT>::max());
      }
      (*v)[i] = static_cast<ValueT>(intImm->value);
    }
    return EthosnError();
  }

  // Get a T from a constant represented by a NDArray.
  template <typename T>
  static EthosnError AsConstant(const Expr& expr, T* out) {
    if (!expr->IsInstance<ConstantNode>()) {
      return EthosnError("expected constant data");
    }
    runtime::NDArray data = Downcast<Constant>(expr)->data;
    *out = *static_cast<T*>(data.operator->()->data);
    return EthosnError();
  }
};

}  // namespace ethosn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_ETHOSN_ETHOSN_API_H_
