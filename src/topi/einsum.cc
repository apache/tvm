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
 * \file topi/einsum.cc
 * \brief Einstein summation op
 */
#include <tvm/topi/broadcast.h>
#include <tvm/topi/einsum.h>

namespace tvm {
namespace topi {

EinsumEquation EinsumEquation::FromString(const std::string& equation) {
  EinsumEquation result;
  Subscript current;
  bool has_arrow = false;
  bool has_ellipsis = false;

  for (int i = 0, n = equation.size(); i < n; ++i) {
    switch (equation[i]) {
      case ' ':
        // Ignore spaces
        break;
      case '-':
        // Arrow
        CHECK(!has_arrow) << "Equation can only have one arrow";
        CHECK(i + 1 < n && equation[i + 1] == '>')
            << "Cannot parse the Einsum equation: invalid arrow";
        i++;
        has_arrow = true;
        [[fallthrough]];
      case ',':
        // Delimiter between inputs, push current and start a new one
        result.inputs.emplace_back(current);
        current.clear();
        has_ellipsis = false;
        break;
      case '.':
        // Ellipsis
        CHECK(!has_ellipsis) << "Ellipsis can only appear once for each input and output";
        CHECK(i + 2 < n && equation[i + 1] == '.' && equation[i + 2] == '.')
            << "Cannot parse the Einsum equation: invalid ellipsis";
        current.push_back(kEllipsis);
        has_ellipsis = true;
        i += 2;
        break;
      default:
        // Default case: current character is a subscript label
        CHECK(std::isalpha(equation[i])) << "Cannot parse the Einsum equation: invalid character "
                                         << equation[i] << " in equation " << equation;
        current.emplace_back(equation[i]);
        break;
    }
  }

  if (has_arrow) {
    // If there is an arrow, the last subscript is the output
    result.output = current;
  } else {
    // Otherwise, the equation is in implicit mode, and the last subscript is an input
    result.inputs.emplace_back(current);
  }

  // Convert the equation to explicit mode if it is in implicit mode
  if (!has_arrow) {
    // The output of the implicit mode is all repeated labels sorted in alphabetical order and the
    // ellipsis in the leftmost if it exists in the inputs.
    std::map<char, int> label_counts;
    for (const Subscript& subscript : result.inputs) {
      for (char label : subscript) {
        label_counts[label]++;
      }
    }
    for (auto [label, count] : label_counts) {
      if (label == kEllipsis || count == 1) {
        result.output.emplace_back(label);
      }
    }
  }
  return result;
}

PrimExpr GetBroadcastedExtent(const PrimExpr& extent1, const PrimExpr& extent2) {
  const IntImmNode* extent1_imm = extent1.as<IntImmNode>();
  const IntImmNode* extent2_imm = extent2.as<IntImmNode>();
  if (extent1_imm != nullptr && extent2_imm != nullptr) {
    if (extent1_imm->value == extent2_imm->value) {
      return extent1;
    } else if (extent1_imm->value == 1 || extent2_imm->value == 1) {
      return Integer(std::max(extent1_imm->value, extent2_imm->value));
    }
    LOG(FATAL) << "Cannot broadcast extents " << extent1 << " and " << extent2;
    throw;
  } else if (extent1_imm != nullptr) {
    return extent2;
  } else if (extent2_imm != nullptr) {
    return extent1;
  } else {
    return max(extent1, extent2);
  }
}

PrimExpr GetIndexForBroadcastedDim(const Var& index, const PrimExpr& extent,
                                   const PrimExpr& broadcasted_extent) {
  // Check if current dimension is being broadcasted to `broadcasted_extent` (symbolic shape is
  // handled)
  if (is_one(extent) && !is_one(broadcasted_extent)) {
    return make_zero(index.dtype());
  }
  return index;
}

/*! \brief The compute builder for Einsum */
class EinsumBuilder {
 public:
  /*!
   * \brief The constructor
   * \param equation The Einsum equation
   * \param input_shapes The shapes of the input tensors
   */
  EinsumBuilder(EinsumEquation equation, Array<Array<PrimExpr>> input_shapes)
      : equation_(equation), input_shapes_(input_shapes) {}

  /*!
   * \brief Run the shape inference
   * \return The inferred shape of the output
   */
  Array<PrimExpr> InferShape() {
    CHECK_EQ(equation_.inputs.size(), input_shapes_.size())
        << "Number of operands does not match the "
           "equation";

    std::vector<Array<PrimExpr>>
        ellipis_shapes;  // the sub-shape covered by the ellipsis for each operand

    // Step 1: Collect the broadcasted extent for each label
    for (int operand_index = 0; operand_index < static_cast<int>(input_shapes_.size());
         ++operand_index) {
      const EinsumEquation::Subscript subscript = equation_.inputs[operand_index];
      const Array<PrimExpr>& input_shape = input_shapes_[operand_index];

      int current_dim = 0;
      for (auto label : subscript) {
        if (label == EinsumEquation::kEllipsis) {
          // Find the sub-shape covered by the ellipsis
          int ellipsis_ndim =
              static_cast<int>(input_shape.size()) - static_cast<int>(subscript.size()) + 1;
          ellipis_shapes.emplace_back(input_shape.begin() + current_dim,
                                      input_shape.begin() + current_dim + ellipsis_ndim);
          current_dim += ellipsis_ndim;
        } else {
          const PrimExpr& extent = input_shape[current_dim++];
          auto it = label_to_extent_.find(label);
          if (it == label_to_extent_.end()) {
            label_to_extent_[label] = extent;
          } else {
            it->second = GetBroadcastedExtent(it->second, extent);
          }
        }
      }
      ICHECK_EQ(current_dim, input_shape.size());
    }

    // Step 2: Infer the shape of the ellipsis if exists
    // The ellipsis may cover different number of dimensions for each operand, these sub-shapes
    // need to be broadcasted to the shape with the maximum number of dimensions
    Array<PrimExpr> ellipsis_shape;
    if (ellipis_shapes.size()) {
      ellipsis_shape = *std::max_element(
          ellipis_shapes.begin(), ellipis_shapes.end(),
          [](const Array<PrimExpr>& a, const Array<PrimExpr>& b) { return a.size() < b.size(); });
      for (const Array<PrimExpr>& shape : ellipis_shapes) {
        auto common_shape = detail::BroadcastShape(ellipsis_shape, shape).common_shape;
        ellipsis_shape = Array<PrimExpr>(common_shape.begin(), common_shape.end());
      }
    }

    // Step 3: Infer output shape based on infered extent for each label
    for (auto label : equation_.output) {
      if (label == EinsumEquation::kEllipsis) {
        output_shape_.insert(output_shape_.end(), ellipsis_shape.begin(), ellipsis_shape.end());
      } else {
        output_shape_.push_back(label_to_extent_[label]);
      }
    }
    ellipsis_shape_ = std::move(ellipsis_shape);
    return output_shape_;
  }

  PrimExpr BuildOutputExpr(const Array<Tensor> inputs, const Array<Var>& indices) {
    std::unordered_map<EinsumEquation::Label, Var> label_to_index;
    Array<Var> ellipsis_indices;
    Array<IterVar> reduce_axes;

    PrepareOutputIndicesMapping(indices, &label_to_index, &ellipsis_indices);
    PrepareReductionIndicesMapping(indices, &label_to_index, &ellipsis_indices, &reduce_axes);

    auto zero = make_zero(inputs[0]->dtype);

    PrimExpr result = zero;
    for (int i = 0, n = static_cast<int>(inputs.size()); i < n; ++i) {
      auto term = inputs[i](GetIndicesForOperand(i, label_to_index, ellipsis_indices));
      if (i == 0) {
        result = term;
      } else {
        result = result * term;
      }
    }
    if (reduce_axes.size() > 0) {
      result = sum(result, reduce_axes, {zero});
    }
    return result;
  }

 private:
  /*!
   * \brief Prepare mapping from label (including ellipsis) to the output indices
   */
  void PrepareOutputIndicesMapping(const Array<Var>& indices,
                                   std::unordered_map<EinsumEquation::Label, Var>* label_to_index,
                                   Array<Var>* ellipsis_indices) {
    int i = 0;
    for (auto label : equation_.output) {
      if (label == EinsumEquation::kEllipsis) {
        auto ellipsis_ndim = ellipsis_shape_.value().size();
        *ellipsis_indices = Array<Var>(indices.begin() + i, indices.begin() + i + ellipsis_ndim);
        i += ellipsis_ndim;
      } else {
        label_to_index->emplace(label, indices[i++]);
      }
    }
    ICHECK_EQ(i, indices.size());
  }

  /*!
   * \brief Create reduction axes and prepare mapping from reduction label (including ellipsis if
   * necessary) to the reduction axes
   */
  void PrepareReductionIndicesMapping(
      const Array<Var>& indices, std::unordered_map<EinsumEquation::Label, Var>* label_to_index,
      Array<Var>* ellipsis_indices, Array<IterVar>* reduction_axes) {
    // Collect labels that need to be reduced, which is the union(input_labels) - output_labels
    std::set<char> reduction_labels;
    for (const EinsumEquation::Subscript& subscript : equation_.inputs) {
      reduction_labels.insert(subscript.begin(), subscript.end());
    }
    for (auto label : equation_.output) {
      reduction_labels.erase(label);
    }

    // Create reduction axes.The order of the reduction axes is not specified in the Einsum
    // equation. Here we sort them alphabetically, with the ellipsis axes at the
    // beginning if exists.
    for (auto label : reduction_labels) {
      if (label == EinsumEquation::kEllipsis) {
        // Ellipsis
        auto ellipsis_shape = ellipsis_shape_.value();
        for (int i = 0; i < static_cast<int>(ellipsis_shape.size()); ++i) {
          reduction_axes->push_back(
              IterVar(Range(0, ellipsis_shape[i]), Var("k"), IterVarType::kCommReduce));
          ellipsis_indices->push_back(reduction_axes->back()->var);
        }
      } else {
        // Normal label
        reduction_axes->push_back(IterVar(
            Range(0, label_to_extent_[label]),
            Var(std::string(1, label), label_to_extent_[label].dtype()), IterVarType::kCommReduce));
        label_to_index->emplace(label, reduction_axes->back()->var);
      }
    }
  }

  Array<PrimExpr> GetIndicesForOperand(
      int operand_index, const std::unordered_map<EinsumEquation::Label, Var>& label_to_index,
      const Array<Var>& ellipsis_indices) {
    const EinsumEquation::Subscript& subscript = equation_.inputs[operand_index];
    Array<PrimExpr> indices;  // the indices for the operand
    const Array<PrimExpr> input_shape = input_shapes_[operand_index];

    int i = 0;  // index of the operand shape
    for (char label : subscript) {
      if (label == EinsumEquation::kEllipsis) {
        // Ellipsis
        Array<PrimExpr> ellipsis_shape = ellipsis_shape_.value();
        int ellipsis_ndim =
            static_cast<int>(input_shape.size()) - static_cast<int>(subscript.size()) + 1;
        // use last 'ellipsis_ndim' axes
        for (int j = static_cast<int>(ellipsis_indices.size()) - ellipsis_ndim;
             j < static_cast<int>(ellipsis_indices.size()); ++j) {
          indices.push_back(
              GetIndexForBroadcastedDim(ellipsis_indices[j], input_shape[i++], ellipsis_shape[j]));
        }
      } else {
        // Normal label
        indices.push_back(GetIndexForBroadcastedDim(label_to_index.at(label), input_shape[i++],
                                                    label_to_extent_.at(label)));
      }
    }
    ICHECK_EQ(i, input_shape.size());
    ICHECK_EQ(indices.size(), input_shape.size());
    return indices;
  }

  EinsumEquation equation_;
  Array<Array<PrimExpr>> input_shapes_;

  // intermediate results of shape inference

  // The output shape
  Array<PrimExpr> output_shape_;
  // The extent of each label with broadcast rules applied
  std::unordered_map<EinsumEquation::Label, PrimExpr> label_to_extent_;
  // The shape of the ellipsis if ellipsis is used. The shape covered by the
  // ellipsis in each operand might be different from this, this is the common
  // shape among them according to the broadcast rules.
  Optional<Array<PrimExpr>> ellipsis_shape_;
};

Tensor einsum(const std::string& subscripts_str, const Array<Tensor> inputs, std::string name,
              std::string tag) {
  EinsumEquation equation = EinsumEquation::FromString(subscripts_str);
  Array<Array<PrimExpr>> input_shapes;
  for (const Tensor& input : inputs) {
    input_shapes.push_back(input->shape);
  }
  EinsumBuilder einsum_builder = EinsumBuilder(equation, input_shapes);
  auto output_shape = einsum_builder.InferShape();
  return te::compute(
      output_shape,
      [&](const Array<Var>& indices) { return einsum_builder.BuildOutputExpr(inputs, indices); },
      name, tag);
}

Array<PrimExpr> InferEinsumShape(const std::string& subscripts,
                                 const std::vector<Array<PrimExpr>>& operands) {
  EinsumEquation equation = EinsumEquation::FromString(subscripts);
  EinsumBuilder einsum_builder = EinsumBuilder(equation, operands);
  return einsum_builder.InferShape();
}

TVM_REGISTER_GLOBAL("topi.einsum").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = einsum(args[0], args[1]);
});

}  // namespace topi
}  // namespace tvm
