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

/*!
 * \file src/relay/transforms/simplify_expr.h
 * \brief Utility data structures for simplifying Relay expressions.
 */
#ifndef TVM_RELAY_TRANSFORMS_SIMPLIFY_EXPR_H_
#define TVM_RELAY_TRANSFORMS_SIMPLIFY_EXPR_H_

#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr.h>

#include <vector>

namespace tvm {
namespace relay {

/*! \brief Defines a static function `RewriteType::Get()` that returns a statically initialized
 * instance of RewriteType. */
#define TVM_DF_PATTERN_REWRITE_GETTER(RewriteType) \
  static DFPatternRewrite* Get() {                 \
    static RewriteType rw;                         \
    return &rw;                                    \
  }

/*! \brief A wrapper class defining a rewrite matching a specific pattern. */
class DFPatternRewrite {
 public:
  /*! \brief Returns the rewritten expression. */
  virtual Expr Callback(const Expr& pre, const Expr& post,
                        const Map<DFPattern, Array<Expr>>& node_map) const = 0;

  /*! \brief Returns the pattern to be used for matching and rewriting. */
  inline DFPattern Pattern() const { return pattern_; }

  inline bool RequireType() const { return require_type_; }

 protected:
  /*! \brief The pattern for matching and rewriting. */
  DFPattern pattern_;
  bool require_type_;
};

/*! \brief Returns an array of DFPatternCallbacks using the given rewrites. */
inline Array<DFPatternCallback> MakeCallbacks(const std::vector<DFPatternRewrite*>& rewrites) {
  Array<DFPatternCallback> callbacks;
  for (const auto& rewrite : rewrites) {
    auto func = [rewrite](TVMArgs args, TVMRetValue* rv) {
      Expr pre = args[0];
      Expr post = args[1];
      Map<DFPattern, Array<Expr>> node_map = args[2];
      *rv = rewrite->Callback(pre, post, node_map);
    };
    callbacks.push_back(
        DFPatternCallback(rewrite->Pattern(), PackedFunc(func), rewrite->RequireType()));
  }
  return callbacks;
}

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_TRANSFORMS_SIMPLIFY_EXPR_H_
