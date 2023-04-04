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
#include "utils.h"

#include <tvm/relax/analysis.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/expr.h>

#include <optional>

#include "../pattern_registry.h"

namespace tvm {
namespace relax {
namespace backend {

Map<String, IntImm> ExtractArgIdx(String pattern_name, Function f) {
  Map<String, IntImm> arg_idx;
  auto pattern = backend::GetPattern(pattern_name);
  ICHECK(pattern) << "Unsupported op_type " << pattern_name;

  auto bindings = AnalyzeVar2Value(f);
  auto inner_body = Downcast<SeqExpr>(f->body)->body;
  auto matched_expr = relax::ExtractMatchedExpr(pattern.value()->pattern, inner_body, bindings);
  ICHECK(matched_expr);

  auto find_index = [](const Array<Var>& params, Var v) -> std::optional<size_t> {
    for (size_t i = 0; i < params.size(); ++i) {
      if (params[i] == v) {
        return i;
      }
    }
    return std::nullopt;
  };

  for (const auto& [name, pat] : pattern.value()->annotation_patterns) {
    auto exp = matched_expr.value()[pat];
    if (auto arg_var = exp.as<VarNode>()) {
      if (auto idx = find_index(f->params, GetRef<Var>(arg_var))) {
        arg_idx.Set(name, IntImm(DataType::Int(64), *idx));
      }
    }
  }

  return arg_idx;
}

TVM_REGISTER_GLOBAL("relax.contrib.extract_arg_idx").set_body_typed(ExtractArgIdx);

}  // namespace backend
}  // namespace relax
}  // namespace tvm
