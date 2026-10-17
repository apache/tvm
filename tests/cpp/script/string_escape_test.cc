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

#include <gtest/gtest.h>
#include <tvm/ffi/function.h>
#include <tvm/script/printer/config.h>
#include <tvm/script/printer/doc.h>

namespace tvm {
namespace script {
namespace printer {
namespace {

ffi::String Render(const Doc& doc, const PrinterConfig& config = PrinterConfig()) {
  return (*ffi::Function::GetGlobal("script.printer.DocToPythonScript"))(doc, config)
      .cast<ffi::String>();
}

TEST(StringEscape, UnicodeExpressionAndNestedQuotes) {
  EXPECT_EQ(Render(ExprStringDoc(IdDoc("é中😀"), std::nullopt)), "\"\\u00e9\\u4e2d\\U0001f600\"");
  ExprStringDoc inner(IdDoc("é"), std::nullopt);
  EXPECT_EQ(Render(ExprStringDoc(inner, std::nullopt)), "\"\\\"\\\\u00e9\\\"\"");
}

TEST(StringEscape, ExpressionUnderlineUsesEscapedOffsets) {
  auto path = AccessPath::Root()->Attr("value");
  IdDoc value("é中");
  value->source_paths.push_back(path);
  ExprStringDoc quoted(ListDoc({IdDoc("prefix"), value}), std::nullopt);
  PrinterConfig config(
      {{"path_to_underline", ffi::Array<AccessPath>{path}}, {"render_invisible_path_info", false}});
  EXPECT_EQ(Render(quoted, config), "\"[prefix, \\u00e9\\u4e2d]\"\n          ^^^^^^^^^^^^");
}

TEST(StringEscape, NestedExpressionUnderlineUsesEscapedOffsets) {
  auto path = AccessPath::Root()->Attr("value");
  IdDoc value("é");
  value->source_paths.push_back(path);
  ExprStringDoc inner(value, std::nullopt);
  ExprStringDoc outer(inner, std::nullopt);
  PrinterConfig config(
      {{"path_to_underline", ffi::Array<AccessPath>{path}}, {"render_invisible_path_info", false}});
  EXPECT_EQ(Render(outer, config), "\"\\\"\\\\u00e9\\\"\"\n   ^^^^^^^");
}

}  // namespace
}  // namespace printer
}  // namespace script
}  // namespace tvm
