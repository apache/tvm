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
 * \file src/contrib/msc/core/utils.cc
 */

#include "utils.h"

#include <string>
namespace tvm {
namespace contrib {
namespace msc {

size_t CommonUtils::GetIndex(int index, size_t max_size) {
  size_t v_index;
  if (index < 0) {
    v_index = index + max_size;
  } else {
    v_index = index;
  }
  ICHECK_LT(v_index, max_size) << "Index " << index << " out of range " << max_size;
  return v_index;
}

std::vector<size_t> CommonUtils::GetIndices(const std::vector<int>& indices, size_t max_size) {
  std::vector<size_t> v_indices;
  for (const auto& i : indices) {
    v_indices.push_back(GetIndex(i, max_size));
  }
  return v_indices;
}

const Array<String> StringUtils::Split(const String& src_string, const String& sep) {
  Array<String> sub_strings;
  if (src_string.size() == 0) {
    return sub_strings;
  }
  std::string src_cstring = src_string;
  const std::string& csep = sep;
  int pos = src_cstring.find(csep);
  while (pos >= 0) {
    if (pos > 0) {
      sub_strings.push_back(src_cstring.substr(0, pos));
    }
    src_cstring = src_cstring.substr(pos + csep.size());
    pos = src_cstring.find(csep);
  }
  if (src_cstring.size() > 0) {
    sub_strings.push_back(src_cstring);
  }
  return sub_strings;
}

const String StringUtils::Join(const Array<String>& sub_strings, const String& joint) {
  String join_str = "";
  for (size_t i = 0; i < sub_strings.size(); i++) {
    join_str = join_str + sub_strings[i] + (i == sub_strings.size() - 1 ? "" : joint);
  }
  return join_str;
}

const String StringUtils::Replace(const String& src_string, const String& old_str,
                                  const String& new_str) {
  String new_string;
  const auto& sub_strings = Split(src_string, old_str);
  for (size_t i = 0; i < sub_strings.size(); i++) {
    new_string = new_string + sub_strings[i] + (i == sub_strings.size() - 1 ? "" : new_str);
  }
  return new_string;
}

const std::tuple<String, String> StringUtils::SplitOnce(const String& src_string, const String& sep,
                                                        bool from_left) {
  if (src_string.size() == 0) {
    return std::make_tuple(String(), String());
  }
  std::string src_cstring = src_string;
  const std::string& csep = sep;
  int pos = from_left ? src_cstring.find(csep) : src_cstring.rfind(csep);
  if (pos >= 0) {
    return std::make_tuple(src_cstring.substr(0, pos), src_cstring.substr(pos + csep.size()));
  }
  return std::make_tuple(src_string, String());
}

const Array<String> StringUtils::GetClosures(const String& src_string, const String& left,
                                             const String& right) {
  Array<String> tokens;
  if (src_string.size() == 0) {
    return tokens;
  }
  String token = "start";
  String left_str = src_string;
  while (token.size() > 0) {
    std::tie(token, left_str) = StringUtils::SplitOnce(left_str, left);
    if (left_str.size() > 0) {
      std::tie(token, left_str) = StringUtils::SplitOnce(left_str, right);
    } else {
      token = "";
    }
    if (token.size() > 0) {
      tokens.push_back(token);
    }
  }
  return tokens;
}

const String StringUtils::GetClosureOnce(const String& src_string, const String& left,
                                         const String& right, bool from_left) {
  if (src_string.size() == 0) {
    return "";
  }
  String val = std::get<1>(SplitOnce(src_string, left, from_left));
  if (val.size() > 0) {
    val = std::get<0>(StringUtils::SplitOnce(val, right, from_left));
  }
  return val;
}

const String StringUtils::ToString(const runtime::ObjectRef& obj) {
  String obj_string;
  if (!obj.defined()) {
    obj_string = "";
  } else if (obj.as<StringObj>()) {
    obj_string = Downcast<String>(obj);
  } else if (const auto* n = obj.as<IntImmNode>()) {
    obj_string = std::to_string(n->value);
  } else if (const auto* n = obj.as<FloatImmNode>()) {
    obj_string = std::to_string(n->value);
  } else if (const auto* n = obj.as<ArrayNode>()) {
    for (size_t i = 0; i < n->size(); i++) {
      obj_string = obj_string + ToString((*n)[i]);
      if (n->size() == 1 || i < n->size() - 1) {
        obj_string = obj_string + ",";
      }
    }
  } else {
    std::ostringstream obj_des;
    obj_des << obj;
    obj_string = obj_des.str();
  }
  return obj_string;
}

bool StringUtils::CompareArrays(const Array<String>& left, const Array<String>& right, int size) {
  if (left.size() == right.size() && left.size() == 0) {
    return true;
  }
  if (size == -1 && left.size() != right.size()) {
    return false;
  }
  if (left.size() == 0 || right.size() == 0) {
    return false;
  }
  size = left.size();
  ICHECK_GT(size, 0) << "Positive size should be given, get " << size;
  if (size > static_cast<int>(left.size()) || size > static_cast<int>(right.size())) {
    return false;
  }
  for (size_t i = 0; i < static_cast<size_t>(size); i++) {
    if (left[i] != right[i]) {
      return false;
    }
  }
  return true;
}

const Span SpanUtils::SetAttr(const Span& span, const String& key, const String& value) {
  if (value.size() == 0) {
    return span;
  }
  String new_source;
  Array<String> tokens{"<" + key + ">", "</" + key + ">"};
  if (span.defined() && span->source_name.defined()) {
    const String& source_str = span->source_name->name;
    String left = std::get<0>(StringUtils::SplitOnce(source_str, tokens[0]));
    String right = std::get<1>(StringUtils::SplitOnce(source_str, tokens[1]));
    if (left.size() > 0) {
      new_source = left + tokens[0] + value + tokens[1] + right;
    } else {
      new_source = source_str + tokens[0] + value + tokens[1];
    }
  } else {
    new_source = tokens[0] + value + tokens[1];
  }
  if (span.defined()) {
    return Span(SourceName::Get(new_source), span->line, span->end_line, span->column,
                span->end_column);
  }
  return Span(SourceName::Get(new_source), 0, 0, 0, 0);
}

const String SpanUtils::GetAttr(const Span& span, const String& key) {
  if (span.defined() && span->source_name.defined()) {
    Array<String> tokens{"<" + key + ">", "</" + key + ">"};
    return StringUtils::GetClosureOnce(span->source_name->name, tokens[0], tokens[1]);
  }
  return "";
}

const Map<String, String> SpanUtils::GetAttrs(const Span& span) {
  Map<String, String> attrs;
  for (const auto& key : StringUtils::GetClosures(span->source_name->name, "</", ">")) {
    attrs.Set(key, GetAttr(span, key));
  }
  return attrs;
}

const Array<String> ExprUtils::GetInputTypes(const String& optype, size_t inputs_num,
                                             bool as_relax) {
  Array<String> input_types;
  if (as_relax && (optype == "broadcast_to" || optype == "reshape")) {
    input_types.push_back("input");
    input_types.push_back("shape");
  } else if (optype == "clip" && as_relax) {
    input_types.push_back("input");
    input_types.push_back("min");
    input_types.push_back("max");
  } else if (optype == "full" && as_relax) {
    input_types.push_back("shape");
    input_types.push_back("input");
  } else if (optype == "trilu") {
    input_types.push_back("input");
    input_types.push_back("k");
  } else if (optype == "image.resize2d" && as_relax) {
    input_types.push_back("input");
    input_types.push_back("size");
  } else if (optype == "nn.conv1d" || optype == "nn.conv2d" || optype == "nn.conv3d") {
    input_types.push_back("input");
    input_types.push_back("weight");
  } else if (optype == "nn.batch_norm") {
    input_types.push_back("input");
    input_types.push_back("gamma");
    input_types.push_back("beta");
    input_types.push_back("mean");
    input_types.push_back("var");
  } else if (optype == "nn.layer_norm" || optype == "nn.group_norm") {
    input_types.push_back("input");
    input_types.push_back("gamma");
    input_types.push_back("beta");
  } else if (optype == "msc.linear") {
    if (as_relax) {
      input_types.push_back("weight");
      input_types.push_back("input");
    } else {
      input_types.push_back("input");
      input_types.push_back("weight");
    }
  } else if (optype == "msc.conv1d_bias" || optype == "msc.conv2d_bias") {
    input_types.push_back("input");
    input_types.push_back("weight");
    input_types.push_back("bias");
    if (as_relax) {
      input_types.push_back("expand_bias");
    }
  } else if (optype == "msc.linear_bias") {
    if (as_relax) {
      input_types.push_back("weight");
      input_types.push_back("input");
    } else {
      input_types.push_back("input");
      input_types.push_back("weight");
    }
    input_types.push_back("bias");
  } else if (optype == "msc.embedding" && inputs_num == 2) {
    input_types.push_back("input");
    input_types.push_back("weight");
  } else if (optype == "msc.embedding" && inputs_num == 4) {
    input_types.push_back("input");
    input_types.push_back("reduce_in");
    input_types.push_back("weight");
    input_types.push_back("expand_out");
  } else if (optype == "msc.gelu") {
    input_types.push_back("input");
    input_types.push_back("factor_1");
    input_types.push_back("factor_2");
    input_types.push_back("factor_3");
  } else {
    for (size_t i = 0; i < inputs_num; i++) {
      input_types.push_back("input");
    }
  }
  ICHECK_EQ(input_types.size(), inputs_num)
      << "Optype " << optype << " get input types " << input_types << " and inputs_num "
      << inputs_num << " mismatch";
  return input_types;
}

const Array<String> ExprUtils::GetInputTypes(const RelaxCall& call) {
  const String& optype = StringUtils::Replace(Downcast<Op>(call->op)->name, "relax.", "");
  return GetInputTypes(optype, call->args.size(), true);
}

const Array<String> ExprUtils::GetInputTypes(const RelayCall& call) {
  const String& optype = StringUtils::Replace(Downcast<Op>(call->op)->name, "relay.", "");
  return GetInputTypes(optype, call->args.size(), false);
}

TVM_REGISTER_GLOBAL("msc.core.SpanGetAttr").set_body_typed(SpanUtils::GetAttr);

TVM_REGISTER_GLOBAL("msc.core.SpanGetAttrs").set_body_typed(SpanUtils::GetAttrs);

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
