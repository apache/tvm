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

#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/node/structural_equal.h>
#include <tvm/te/operation.h>

TEST(Expr, Basic) {
  using namespace tvm;
  using namespace tvm::tir;
  Var x("x");
  auto z = max(x + 1 + 2, 100);
  ObjectRef tmp = z;
  PrimExpr zz = Downcast<PrimExpr>(tmp);
  std::ostringstream os;
  os << z;
  ICHECK(zz.same_as(z));
  ICHECK(os.str() == "T.max(x + 1 + 2, 100)");
}

TEST(Expr, VarTypeAnnotation) {
  using namespace tvm;
  using namespace tvm::tir;
  Var x("x", DataType::Float(32));
  Var y("y", PrimType(DataType::Float(32)));
  StructuralEqual checker;
  ICHECK(checker(x->dtype, y->dtype));
  ICHECK(checker(x->type_annotation, y->type_annotation));
}

TEST(ExprNodeRef, Basic) {
  using namespace tvm;
  using namespace tvm::tir;
  Var x("x");
  PrimExpr z = max(x + 1 + 2, 100);
  const tir::MaxNode* op = z.as<tir::MaxNode>();
  ICHECK(GetRef<ObjectRef>(op).same_as(z));
}
