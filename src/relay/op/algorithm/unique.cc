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
 * \file unique.cc
 * \brief The unique operator
 */
#include <dlpack/dlpack.h>
#include <tvm/relay/attrs/algorithm.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/data_type.h>

namespace tvm {
namespace relay {

bool UniqueRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  // types: [data, result]
  ICHECK_EQ(types.size(), 2) << "Unique: expect 2 types but " << types.size() << " provided";
  ICHECK_EQ(num_inputs, 1) << "Unique: expect 1 inputs but " << num_inputs << " provided";
  auto data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "Unique: expect input type to be TensorType but get " << types[0];
    return false;
  }
  std::vector<Type> fields;
  fields.push_back(TensorType(data->shape, data->dtype));
  fields.push_back(TensorType(data->shape, DataType::Int(32)));
  fields.push_back(TensorType(data->shape, DataType::Int(32)));
  fields.push_back(TensorType(Array<PrimExpr>{1}, DataType::Int(32)));
  reporter->Assign(types[1], TupleType(Array<Type>(fields)));
  return true;
}

Expr MakeUnique(Expr data) {
  static const Op& op = Op::Get("unique");
  return Call(op, {data}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.unique").set_body_typed(MakeUnique);

RELAY_REGISTER_OP("unique")
    .describe(
        R"code(This operation returns a tensor **output** containing all of the unique elements of **data**
        sorted in the same order that they occur in **data**; **data** does not need to be sorted.
        This operation also returns a tensor **inverse_indices** contains the index of each value of **data** in the unique output **output**.
        In other words: output[inverse_indices[i]] = data[i] for i in [0, 1,..., len(data) - 1].
        This operation also returns a 0-D tensor **num_unique_elements** contains the number of unique elements in **data**.
        Please note **output** and **counts** have the same size of **data** and only items [0, 1,..., num_unique_elements[0]-1] are valid.

        - **data**: A 1-D tensor of integers

        - **output**: A 1-D tensor containing the unique elements of **data**

        - **inverse_indices**: A 1-D tensor containing the index of each value of **data** in **output**

        - **counts**: A 1-D tensor containing the count of each element of **output** in **data**

        - **num_unique_elements**: A 0-D tensor containing the number of unique elements

        Example::
        -  [y, idx, counts, n] = unique([1, 1, 2, 4, 4, 4, 7, 8, 8])
        -       y     =  [1, 2, 4, 7, 8, ?, ?, ?, ?]
        -       idx   =  [0, 0, 1, 2, 2, 2, 3, 4, 4]
        -       count =  [2, 1, 3, 1, 2, ?, ?, ?, ?]
        -       n     =  [5]
    )code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .add_type_rel("unique", UniqueRel)
    .set_support_level(6);

template <typename T>
void calc_unique(DLTensor* input, DLTensor* output, DLTensor* inverse_indices, DLTensor* counts,
                 DLTensor* num_unique_elements) {
  std::unordered_map<T, int>
      unique_map;  // map to record the idx of each unique element in the output tensor
  auto input_ptr = static_cast<T*>(input->data);
  auto output_ptr = static_cast<T*>(output->data);
  auto inverse_indices_ptr = static_cast<int32_t*>(inverse_indices->data);
  auto counts_ptr = static_cast<int32_t*>(counts->data);
  auto num_unique_ptr = static_cast<int32_t*>(num_unique_elements->data);

  int unique_counter = 0;
  for (int i = 0; i < input->shape[0]; i++) {
    if (unique_map.count(input_ptr[i]) == 0) {
      unique_map[input_ptr[i]] = unique_counter;
      output_ptr[unique_counter] = input_ptr[i];
      counts_ptr[unique_counter] = 0;
      unique_counter++;
    }
    inverse_indices_ptr[i] = unique_map[input_ptr[i]];
    counts_ptr[inverse_indices_ptr[i]]++;
  }

  num_unique_ptr[0] = unique_counter;
}

// The unique operator
TVM_REGISTER_GLOBAL("tvm.contrib.algorithm.unique").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* input = args[0];
  DLTensor* output = args[1];
  DLTensor* inverse_indices = args[2];
  DLTensor* counts = args[3];
  DLTensor* num_unique_elements = args[4];

  ICHECK_EQ(input->ndim, 1) << "The input tensor must be 1-D";
  ICHECK((output->ndim) == 1 && (inverse_indices->ndim) == 1 && (counts->ndim == 1) &&
         (num_unique_elements->ndim == 1))
      << "The output,inverse_indices,counts,num_unique_elements tensors must be 1-D";
  ICHECK((input->shape[0] == output->shape[0]) && (input->shape[0] == inverse_indices->shape[0]) &&
         (input->shape[0] == counts->shape[0]))
      << "The input,output,inverse_indices,counts tensors must have the "
         "same size";
  ICHECK_EQ(num_unique_elements->shape[0], 1) << "The num_unique_elements tensor must have size 1";

  auto data_dtype = tvm::runtime::DLDataType2String(input->dtype);

  if (data_dtype == "int32") {
    calc_unique<int32_t>(input, output, inverse_indices, counts, num_unique_elements);
  } else if (data_dtype == "int64") {
    calc_unique<int64_t>(input, output, inverse_indices, counts, num_unique_elements);
  } else {
    LOG(FATAL) << "Unsupported input dtype: " << data_dtype;
  }
});

}  // namespace relay
}  // namespace tvm
