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
 * \file tl/ir.cc
 * \brief Extension for the tvm script frontend.
 *
 */

#include <tvm/script/ir_builder/tir/ir.h>

namespace tvm {
namespace script {
namespace ir_builder {
namespace tir {

ForFrame ParallelFor(Array<PrimExpr> extents) {
  using namespace tvm::tir;
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  n->vars.reserve(extents.size());
  n->doms.reserve(extents.size());
  for (const auto& extent : extents) {
    DataType dtype = extent.dtype();
    n->vars.push_back(Var("v", extent.dtype()));
    n->doms.push_back(Range(make_const(dtype, 0), extent));
  }
  n->f_make_for_loop = [](Array<Var> vars, Array<Range> doms, Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    int n = vars.size();
    for (int i = n - 1; i >= 0; --i) {
      Range dom = doms[i];
      Var var = vars[i];
      body = For(var, dom->min, dom->extent, ForKind::kParallel, std::move(body),
                 /*thread_binding=*/NullOpt, /*annotations=*/{});
    }
    return body;
  };
  return ForFrame(n);
}

ForFrame PipelinedFor(PrimExpr start, PrimExpr stop, int num_stages) {
  using namespace tvm::tir;
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  DataType dtype = stop.dtype();
  n->vars.push_back(Var("v", dtype));
  n->doms.push_back(Range(make_const(dtype, 0), stop));
  n->f_make_for_loop = [=] (Array<Var> vars, Array<Range> doms, Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    int n = vars.size();
    ICHECK(n == 1);
    Map<String, ObjectRef> anno;
    if (num_stages > 0) anno.Set("num_stages", PrimExpr(num_stages));
    body = For(vars[0], doms[0]->min, doms[0]->extent, ForKind::kSerial, std::move(body),
                /*thread_binding=*/NullOpt, /*annotations=*/anno);
    return body;
  };
  return ForFrame(n);
}

TVM_REGISTER_GLOBAL("tl.Parallel").set_body_typed(ParallelFor);
TVM_REGISTER_GLOBAL("tl.Pipelined").set_body_typed(PipelinedFor);

}
}
}
}
