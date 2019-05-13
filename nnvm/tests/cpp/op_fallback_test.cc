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
#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nnvm/graph.h>
#include <nnvm/op.h>
#include <nnvm/pass.h>
#include <nnvm/symbolic.h>
#include <tvm/tvm.h>

#include <string>
#include <utility>
#include <vector>

namespace nnvm {

extern nnvm::Graph AnnotateGraph(nnvm::Graph g);
extern nnvm::Graph InsertDataCopy(nnvm::Graph g);

namespace cpptest {
using StringVector = std::vector<std::string>;
using IntVector = std::vector<int>;

enum class AnnotationType : int {
  kTarget = 1,
  kDeivceTarget = 2
};

NNVM_REGISTER_OP(add)
  .describe("addtion operation")
  .set_num_inputs(2)
  .set_num_outputs(1);

NNVM_REGISTER_OP(sub)
  .describe("subtract operation")
  .set_num_inputs(2)
  .set_num_outputs(1)
  .set_fallback_device(true);

// Register a simple form of the copy operation for testing purpose.
NNVM_REGISTER_OP(device_copy_op)
  .describe("Copy data across devices." NNVM_ADD_FILELINE)
  .set_num_inputs(1)
  .set_num_outputs(1);

nnvm::Graph GetGraph() {
  const auto* add = nnvm::Op::Get("add");
  nnvm::NodePtr add_node = nnvm::Node::Create();
  add_node->attrs.op = add;
  add_node->attrs.name = "add";
  nnvm::Symbol add_sym;
  add_sym.outputs.push_back(nnvm::NodeEntry{add_node, 0, 0});

  const auto* sub = nnvm::Op::Get("sub");

  nnvm::NodePtr sub_node = nnvm::Node::Create();
  sub_node->attrs.op = sub;
  sub_node->attrs.name = "sub";
  sub_node->inputs.push_back(add_sym.outputs[0]);
  nnvm::Symbol sub_sym;
  sub_sym.outputs.push_back(nnvm::NodeEntry{sub_node, 0, 0});

  nnvm::Symbol sym;
  sym.outputs.insert(sym.outputs.end(), add_sym.outputs.begin(),
                     add_sym.outputs.end());
  sym.outputs.insert(sym.outputs.end(), sub_sym.outputs.begin(),
                     sub_sym.outputs.end());

  nnvm::Graph g;
  g.outputs = sym.outputs;
  return g;
}

TEST(NodeAttrTest, DefaultValueForNodes) {
  nnvm::Graph g = GetGraph();
  const auto& idx = g.indexed_graph();
  const auto& add = idx[0U];
  const auto& sub = idx[1U];
  EXPECT_EQ(add.source->attrs.device_type, 0);
  EXPECT_EQ(sub.source->attrs.device_type, 0);
  EXPECT_TRUE(sub.source->attrs.op->fallback);
}

TEST(TargetAnnotationTest, AnnotateNodesWithTarget) {
  nnvm::Graph g = GetGraph();
  StringVector targets{"llvm"};
  IntVector devices{1};
  // Setup required attributes.
  g.attrs["annotation_type"] = std::make_shared<nnvm::any>(
      static_cast<int>(AnnotationType::kTarget));
  g.attrs["target"] = std::make_shared<nnvm::any>(targets);
  g.attrs["device_type"] = std::make_shared<nnvm::any>(std::move(devices));
  g = nnvm::AnnotateGraph(g);
  const auto& idx = g.indexed_graph();
  const auto& add = idx[0U];
  const auto& sub = idx[1U];
  EXPECT_EQ(g.indexed_graph().num_nodes(), 2);
  EXPECT_TRUE(add.source->attrs.dict.count("target"));
  EXPECT_TRUE(sub.source->attrs.dict.count("target"));
  EXPECT_EQ(add.source->attrs.dict.at("target"), targets[0]);
  EXPECT_EQ(sub.source->attrs.dict.at("target"), targets[0]);
}


// Both add and sub are explicitly specified to device type 2. However, sub is
// registered with fallback. It, therefore, should be annotated with device
// type 1.
TEST(DeviceFallbackTest, SubOpFallbackToOne) {
  nnvm::Graph g = GetGraph();
  int fallback_device = 1;
  StringVector op_names{"add", "sub"};
  IntVector op_devices{2, 2};
  StringVector targets{"llvm", "cuda"};
  IntVector devices{1, 2};
  // Setup required attributes.
  g.attrs["annotation_type"] = std::make_shared<nnvm::any>(
      static_cast<int>(AnnotationType::kDeivceTarget));
  g.attrs["target"] = std::make_shared<nnvm::any>(std::move(targets));
  g.attrs["device_type"] = std::make_shared<nnvm::any>(std::move(devices));
  g.attrs["op_name"] = std::make_shared<nnvm::any>(std::move(op_names));
  g.attrs["op_device"] = std::make_shared<nnvm::any>(std::move(op_devices));
  g.attrs["fallback"] = std::make_shared<nnvm::any>(std::move(fallback_device));
  g = nnvm::AnnotateGraph(g);
  const auto& idx = g.indexed_graph();
  const auto& add = idx[0U];
  const auto& sub = idx[1U];
  EXPECT_EQ(g.indexed_graph().num_nodes(), 2);
  // add should be annotated with device type 2
  EXPECT_EQ(add.source->attrs.device_type, 2);
  // sub should have been scheduled to device type 1
  EXPECT_EQ(sub.source->attrs.device_type, 1);
}

// No device information is explicitly specified for add. It should be
// annotatedc with the fallback device.
TEST(DeviceFallbackTest, AddOpFallbackToOne) {
  nnvm::Graph g = GetGraph();
  int fallback_device = 1;
  StringVector op_names{"sub"};
  std::vector<int> op_devices{2};
  StringVector targets{"llvm", "cuda"};
  std::vector<int> devices{1, 2};
  // Setup required attributes.
  g.attrs["annotation_type"] = std::make_shared<nnvm::any>(
      static_cast<int>(AnnotationType::kDeivceTarget));
  g.attrs["target"] = std::make_shared<nnvm::any>(std::move(targets));
  g.attrs["device_type"] = std::make_shared<nnvm::any>(std::move(devices));
  g.attrs["op_name"] = std::make_shared<nnvm::any>(std::move(op_names));
  g.attrs["op_device"] = std::make_shared<nnvm::any>(std::move(op_devices));
  g.attrs["fallback"] = std::make_shared<nnvm::any>(std::move(fallback_device));
  g = nnvm::AnnotateGraph(g);
  const auto& idx = g.indexed_graph();
  const auto& add = idx[0U];
  const auto& sub = idx[1U];
  EXPECT_EQ(g.indexed_graph().num_nodes(), 2);
  // add should be annotated with device type 2
  EXPECT_EQ(add.source->attrs.device_type, 1);
  // sub should have been scheduled to device type 1
  EXPECT_EQ(sub.source->attrs.device_type, 1);
}

TEST(CopyNodeInsertionTest, CopyNodeInsertedIsAndAnnotated) {
  nnvm::Graph g = GetGraph();
  int fallback_device = 1;
  StringVector op_names{"add"};
  IntVector op_devices{2};
  StringVector targets{"llvm", "cuda"};
  IntVector devices{1, 2};
  // Setup required attributes.
  g.attrs["annotation_type"] = std::make_shared<nnvm::any>(
      static_cast<int>(AnnotationType::kDeivceTarget));
  g.attrs["target"] = std::make_shared<nnvm::any>(targets);
  g.attrs["device_type"] = std::make_shared<nnvm::any>(std::move(devices));
  g.attrs["op_name"] = std::make_shared<nnvm::any>(std::move(op_names));
  g.attrs["op_device"] = std::make_shared<nnvm::any>(std::move(op_devices));
  g.attrs["fallback"] = std::make_shared<nnvm::any>(std::move(fallback_device));
  g = nnvm::AnnotateGraph(g);
  g = nnvm::InsertDataCopy(g);
  const auto& idx = g.indexed_graph();
  const auto& add = idx[0U];
  const auto& copy = idx[1U];
  const auto& sub = idx[2U];
  // A copy node should be inserted.
  EXPECT_EQ(g.indexed_graph().num_nodes(), 3);
  EXPECT_EQ(add.source->attrs.device_type, 2);
	// Both copy node and sub should have the same device type, which is 1.
  EXPECT_EQ(copy.source->attrs.device_type, 1);
  EXPECT_EQ(sub.source->attrs.device_type, 1);

  // Check annotated target for each node.
  EXPECT_TRUE(add.source->attrs.dict.count("target"));
  EXPECT_FALSE(copy.source->attrs.dict.count("target"));
  EXPECT_TRUE(sub.source->attrs.dict.count("target"));
  EXPECT_EQ(add.source->attrs.dict.at("target"), targets[1]);
  EXPECT_EQ(sub.source->attrs.dict.at("target"), targets[0]);

  // Check device index array
  EXPECT_TRUE(g.HasAttr("device_index"));
  const auto& device_vec = g.MoveCopyAttr<IntVector>("device_index");
  EXPECT_THAT(device_vec, testing::ElementsAre(2, 1, 1));
}

}  // namespace cpptest
}  // namespace nnvm

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
