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
 *  Copyright (c) 2017 by Contributors
 * \file op_common.h
 * \brief Common operator utilities
 */
#ifndef NNVM_TOP_OP_COMMON_H_
#define NNVM_TOP_OP_COMMON_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <nnvm/top/tensor.h>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>
#include <unordered_set>

namespace nnvm {
namespace top {
/*!
 * \brief Parse keyword arguments as PType arguments and save to parsed
 * \tparam PType the parameter type.
 * \param attrs The attributes.
 */
template<typename PType>
inline void ParamParser(nnvm::NodeAttrs* attrs) {
  PType param;
  try {
    param.Init(attrs->dict);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
  attrs->parsed = std::move(param);
}

/*!
 * \brief Parse keyword arguments as PType arguments and save to parsed
 * \tparam PType the arameter type.
 * \param attrs The attributes.
 */
template<typename PType>
inline std::unordered_map<std::string, std::string>
ParamGetAttrDict(const nnvm::NodeAttrs& attrs) {
  std::unordered_map<std::string, std::string> dict = attrs.dict;
  nnvm::get<PType>(attrs.parsed).UpdateDict(&dict);
  return dict;
}

/*! \brief check if shape is empty or contains unkown (0) dim. */
inline bool shape_is_none(const TShape& x) {
  return x.ndim() == 0 || x.Size() == 0;
}

/*! \brief check if type is none (-1) */
inline bool type_is_none(const int& x) {
  return x == -1;
}

/*! \brief check if shape is scalar({1}). */
inline bool shape_is_scalar(const TShape& x) {
  return x.ndim() == 1 && x.Size() == 1;
}

/*! \brief get string representation of shape */
inline std::string shape_string(const TShape& x) {
  std::ostringstream os;
  os << x;
  return os.str();
}

/*! \brief get string representation of shape */
inline std::string type_string(const int& x) {
  return std::to_string(x);
}

/*!
 * \brief Assign x to y. Checks for compatiblity when y is not empty.
 *  Allow missing dim in both x and y (as 0).
 * \param y target shape.
 * \param x source shape.
 * \return whether x and y are compatible.
 */
inline bool shape_assign(TShape *y, const TShape& x) {
  if (y->ndim() == 0) {
    *y = x;
    return true;
  } else if (y->ndim() != x.ndim()) {
    return x.ndim() == 0;
  } else {
    for (size_t i = 0; i < y->ndim(); ++i) {
      if ((*y)[i] == 0) {
        (*y)[i] = x[i];
      } else if ((*y)[i] != x[i] && x[i] != 0) {
        return false;
      }
    }
    return true;
  }
}

/*!
 * \brief Assign x to y. Checks for compatiblity when y is not -1.
 * \param y target type.
 * \param x source type.
 * \return whether x and y are compatible.
 */
inline bool type_assign(int *y, const int& x) {
  if (*y == -1) {
    *y = x;
    return true;
  } else if (*y != x && x != -1) {
    return false;
  }
  return true;
}

template<typename AttrType>
inline std::string attr_assign_error_msg(const NodeAttrs& attrs,
                                         int index, bool is_input,
                                         const AttrType& expected,
                                         const AttrType& actual,
                                         const char* attr_name) {
  static const auto& flist_inputs = Op::GetAttr<FListInputNames>("FListInputNames");
  static const auto& flist_outputs = Op::GetAttr<FListOutputNames>("FListOutputNames");
  const auto& flist = is_input ? flist_inputs : flist_outputs;
  std::string name;
  if (flist.count(attrs.op)) {
    name = flist[attrs.op](attrs)[index];
  } else {
    name = (is_input ? "data" : "output") + std::to_string(index);
  }
  std::ostringstream msg;
  msg << "Operator " << attrs.op->name << "(";
  for (const auto& kv : attrs.dict) msg << kv.first << "=" << kv.second << ", ";
  msg << "name=" << attrs.name << ") expects " << name << "\'s " << attr_name
      << " to be " << expected << ", but got " << actual << ".";
  return msg.str();
}

/*!
 * \brief macro assign shape to input if out is unknown otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param inputs the shape array to store the result
 * \param index the index of in the array
 * \param shape the inferred shape
 */
#define NNVM_ASSIGN_INPUT_SHAPE(attrs, inputs, index, shape)             \
  {                                                                      \
    if (!shape_assign(&(inputs)[index], TShape(shape))) {                \
      LOG(FATAL) << attr_assign_error_msg(attrs, index, true, shape,     \
                                          (inputs)[index], "shape");     \
    }                                                                    \
  }

/*!
 * \brief macro assign shape to out if out is unknown otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param inputs the shape array to store the result
 * \param index the index of in the array
 * \param shape the inferred shape
 */
#define NNVM_ASSIGN_OUTPUT_SHAPE(attrs, outputs, index, shape)           \
  {                                                                      \
    if (!shape_assign(&(outputs)[index], TShape(shape))) {               \
      LOG(FATAL) << attr_assign_error_msg(attrs, index, false, shape,    \
                                          (outputs)[index], "shape");    \
    }                                                                    \
  }

/*!
 * \brief macro assign type to out if out is unknown (-1) otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param inputs the type array to store the result
 * \param index the index of in the array
 * \param type the inferred type
 */
#define NNVM_ASSIGN_INPUT_TYPE(attrs, inputs, index, type)               \
  {                                                                      \
    if (!type_assign(&(inputs)[index], type)) {                          \
      LOG(FATAL) << attr_assign_error_msg(attrs, index, true, type,      \
                                          (inputs)[index], "type");      \
    }                                                                    \
  }

/*!
 * \brief macro assign type to out if out is unknown (-1) otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param inputs the type array to store the result
 * \param index the index of in the array
 * \param type the inferred type
 */
#define NNVM_ASSIGN_OUTPUT_TYPE(attrs, outputs, index, type)             \
  {                                                                      \
    if (!type_assign(&(outputs)[index], type)) {                         \
      LOG(FATAL) << attr_assign_error_msg(attrs, index, false, type,     \
                                          (outputs)[index], "type");     \
    }                                                                    \
  }

#define NNVM_ASSIGN_LAYOUT(outputs, index, layout)                       \
  {                                                                      \
    if (layout.defined()) {                                              \
      (outputs)[index] = layout;                                         \
    }                                                                    \
  }

/*!
 * \brief macro assign rhs shape to lhs
 *  Use macro so we can see the error file more clearly
 * \param lhs lhs shape
 * \param rhs rhs shape
 */
#define SHAPE_ASSIGN(lhs, rhs)                                \
  if ((lhs).ndim() == 0) (lhs) = (rhs);                       \
  else                                                        \
    CHECK_EQ(lhs, rhs) << "shape inference inconsistent";     \

/*!
 * \brief macro assign rhs type to lhs
 *  Use macro so we can see the error file more clearly
 * \param lhs lhs type
 * \param rhs rhs type
 */
#define DTYPE_ASSIGN(lhs, rhs)                                \
  if ((lhs) == -1) (lhs) = (rhs);                             \
  else                                                        \
    CHECK_EQ(lhs, rhs) << "type inference inconsistent";     \

// simply return the shape as same
inline bool SameShape(const NodeAttrs& attrs,
                      std::vector<TShape> *ishape,
                      std::vector<TShape> *oshape) {
  if (ishape->size() == 0 || (*ishape)[0].ndim() == 0) return false;
  for (TShape& pshape : *oshape) {
    pshape = (*ishape)[0];
  }
  for (TShape& pshape : *ishape) {
    pshape = (*ishape)[0];
  }
  return true;
}

// return shape from node attrs
template<typename PType>
inline bool ZeroShape(const NodeAttrs& attrs,
                      std::vector<TShape> *ishape,
                      std::vector<TShape> *oshape) {
  const TShape& ts = dmlc::get<PType>(attrs.parsed).shape;
  if (ts.ndim() != 0) {
    SHAPE_ASSIGN(oshape->at(0), ts);
    return true;
  } else {
    return false;
  }
}

// do not infer layout
inline bool ZeroLayout(const NodeAttrs& attrs,
                       std::vector<Layout> *in_layouts,
                       const std::vector<Layout> *last_in_layouts,
                       std::vector<Layout> *out_layouts) {
  return true;
}

// simply assign output shape or type from input
template<typename AttrType, int in_index, int out_index>
inline bool AssignOutputAttr(const NodeAttrs& attrs,
                              std::vector<AttrType> *in_attrs,
                              std::vector<AttrType> *out_attrs) {
  CHECK_LT(in_index, in_attrs->size());
  CHECK_LT(out_index, out_attrs->size());
  const TShape &dshape = in_attrs->at(in_index);
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, out_index, dshape);
  return true;
}

// return type from node attrs
template<typename PType>
inline bool ZeroType(const NodeAttrs& attrs,
                     std::vector<int> *iattr,
                     std::vector<int> *oattr) {
  int dtype = dmlc::get<PType>(attrs.parsed).dtype;
  DTYPE_ASSIGN(oattr->at(0), dtype);
  return true;
}

// Make zero grad node
inline std::vector<NodeEntry> MakeZeroGradNodes(
  const NodePtr& n,
  const std::vector<NodeEntry>& ograds) {
  std::vector<NodeEntry> ret;
  for (uint32_t i = 0; i < n->num_inputs(); ++i) {
    std::ostringstream os;
    ret.push_back(MakeNode("zeros_like", n->attrs.name + "_zero_grad",
                           {n->inputs[i]}));
  }
  return ret;
}

// Helper to make gradient node
inline std::vector<NodeEntry> MakeGradNode(
  const char* op_name,
  const NodePtr& n,
  std::vector<NodeEntry> inputs,
  std::unordered_map<std::string, std::string> attr = {{}}) {
  NodePtr p = Node::Create();
  p->attrs.op = nnvm::Op::Get(op_name);
  p->attrs.name = n->attrs.name + "_grad";
  p->inputs = std::move(inputs);
  p->attrs.dict = std::move(attr);
  if (p->attrs.op->attr_parser) {
    p->attrs.op->attr_parser(&p->attrs);
  }
  std::vector<NodeEntry> ret;
  for (uint32_t i = 0; i < p->num_outputs(); ++i) {
    ret.emplace_back(NodeEntry{p, i, 0});
  }
  return ret;
}


}  // namespace top
}  // namespace nnvm

#endif  // NNVM_TOP_OP_COMMON_H_
