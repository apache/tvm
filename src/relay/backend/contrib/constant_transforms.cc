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

#include "constant_transforms.h"

#include <string>

#include "../../transforms/fold_constant.h"
#include "../../transforms/pattern_utils.h"
#include "../../transforms/simplify_expr.h"

/*!
 * \file src/relay/backend/contrib/constant_transforms.cc
 * \brief Transforms applied to constant operations during codegen for BYOC backends.
 */

namespace tvm {
namespace relay {
namespace contrib {

Constant TransposeWeights(const Constant& data, const std::string& source_layout,
                          const std::string& target_layout) {
  Array<Integer> transpose_matrix;
  for (const char& c : target_layout) {
    int pos = source_layout.find(c);
    transpose_matrix.push_back(pos);
  }
  Expr transpose = MakeTranspose(data, transpose_matrix);
  transpose = InferType(transform::FoldConstantExpr(transpose));
  Constant transposed_data = Downcast<Constant>(transpose);
  return transposed_data;
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
