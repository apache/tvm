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
#include <tvm/ir/base_expr.h>
#include <tvm/script/printer/config.h>
#include <tvm/script/printer/printer.h>

#include "../../src/script/printer/visible_path.h"

namespace tvm {
namespace script {
namespace printer {

namespace refl = ffi::reflection;

TEST(ScriptPrinter, RenderInvisiblePathInfoHiddenPath) {
  refl::AccessPath visible_path = refl::AccessPath::Root()->Attr("body");
  refl::AccessPath requested_path = visible_path->Attr("ty");

  ffi::String result =
      RenderInvisiblePathInfo("T.evaluate(x)\n           ^\n", {requested_path}, {visible_path});

  EXPECT_EQ(std::string(result),
            "Access path: <root>.body.ty\n"
            "Note: The underlined object is the nearest visible parent of this path.\n\n"
            "T.evaluate(x)\n"
            "           ^\n");
}

TEST(ScriptPrinter, ScriptRendersInvisiblePathInfoAsString) {
  refl::AccessPath requested_path = refl::AccessPath::Root()->Attr("dtype");
  PrinterConfig config({
      {"path_to_underline", ffi::Array<refl::AccessPath>{requested_path}},
      {"render_invisible_path_info", true},
  });

  std::string result = Script(PrimType::Int(32), config);

  EXPECT_EQ(result,
            "Access path: <root>.dtype\n"
            "Note: The underlined object is the nearest visible parent of this path.\n\n"
            "T.int32\n"
            "^^^^^^^");
}

TEST(ScriptPrinter, RenderInvisiblePathInfoExactPath) {
  refl::AccessPath requested_path = refl::AccessPath::Root()->Attr("body");

  ffi::String result =
      RenderInvisiblePathInfo("T.evaluate(x)\n^^^^^^^^^^^^^\n", {requested_path}, {requested_path});

  EXPECT_EQ(std::string(result),
            "Access path: <root>.body\n\n"
            "T.evaluate(x)\n"
            "^^^^^^^^^^^^^\n");
}

TEST(ScriptPrinter, RenderInvisiblePathInfoWithoutVisiblePath) {
  refl::AccessPath requested_path = refl::AccessPath::Root()->Attr("missing");

  ffi::String result = RenderInvisiblePathInfo("T.evaluate(x)\n", {requested_path},
                                               {ffi::Optional<refl::AccessPath>(std::nullopt)});

  EXPECT_EQ(std::string(result),
            "Access path: <root>.missing\n"
            "Note: No visible object for this path is rendered in TVMScript.\n\n"
            "T.evaluate(x)\n");
}

TEST(ScriptPrinter, RenderInvisiblePathInfoConfig) {
  PrinterConfig default_config;
  EXPECT_FALSE(default_config->render_invisible_path_info);

  PrinterConfig direct_config({{"render_invisible_path_info", true}});
  EXPECT_TRUE(direct_config->render_invisible_path_info);

  ffi::Map<ffi::String, ffi::Any> extra_config{{"render_invisible_path_info", true}};
  PrinterConfig forwarded_config({{"extra_config", extra_config}});
  EXPECT_TRUE(forwarded_config->render_invisible_path_info);
}

}  // namespace printer
}  // namespace script
}  // namespace tvm
