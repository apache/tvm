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
 * \file src/relay/transforms/merge_composite.cc
 * \brief Merges expressions matching patterns into functions marked
 * as 'composite'. This is primarily intended to be used alongside the
 * external codegen infrastructure to support the case where multiple
 * Relay operators map to a single external operator.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/te/operation.h>

namespace tvm {
namespace relay {
namespace merge_composite {

Function InferType(const Function& expr, const IRModule& m) {
  IRModule mod(m);
  mod->Update(mod->GetGlobalVar("main"), expr);
  mod = transform::InferType()(mod);
  return Downcast<Function>(mod->Lookup("main"));
}

Expr MergeComposite(const Function& func, const Array<runtime::String>& pattern_names,
                    const Array<DFPattern>& patterns, const std::vector<PackedFunc>& checks,
                    const IRModule& m) {
  ICHECK_EQ(pattern_names.size(), patterns.size());
  Function merged_func = func;
  // merge the patterns one-by-one in order
  for (size_t i = 0; i < patterns.size(); i++) {
    Map<String, ObjectRef> attrs;
    attrs.Set("Composite", pattern_names[i]);
    merged_func = Downcast<Function>(PartitionPattern(patterns[i], merged_func, attrs, checks[i]));
    merged_func = InferType(merged_func, m);
  }
  return std::move(merged_func);
}

}  // namespace merge_composite

namespace transform {

Pass MergeComposite(const tvm::Array<runtime::String>& pattern_names,
                    const tvm::Array<DFPattern>& patterns, const std::vector<PackedFunc>& checks) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(
            relay::merge_composite::MergeComposite(f, pattern_names, patterns, checks, m));
      };
  auto func_pass = CreateFunctionPass(pass_func, 0, "MergeComposite", {});
  return func_pass;
}

TVM_REGISTER_GLOBAL("relay._transform.MergeComposite").set_body([](TVMArgs args, TVMRetValue* rv) {
  tvm::Array<runtime::String> pattern_names = args[0];
  tvm::Array<DFPattern> patterns = args[1];
  std::vector<PackedFunc> checks;
  for (int i = 2; i < args.size(); i++) {
    checks.push_back(args[i]);
  }
  *rv = MergeComposite(pattern_names, patterns, checks);
});

}  // namespace transform

}  // namespace relay
}  // namespace tvm
