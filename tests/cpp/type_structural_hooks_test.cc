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
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/type.h>
#include <tvm/relax/distributed/type.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/tirx/buffer_region.h>
#include <tvm/tirx/var.h>

namespace {

template <typename TNode>
void ExpectStructuralHooks() {
  namespace refl = tvm::ffi::reflection;
  for (const char* attr_name :
       {refl::type_attr::kStructuralVisit, refl::type_attr::kStructuralMutate,
        refl::type_attr::kStructuralMaybeInplaceMutate}) {
    refl::TypeAttrColumn column(attr_name);
    EXPECT_EQ(column[TNode::RuntimeTypeIndex()].type_index(), tvm::ffi::TypeIndex::kTVMFFIOpaquePtr)
        << TNode::_type_key << " is missing " << attr_name;
  }
}

TEST(TypeStructuralHooks, EveryConcreteOpenTypeHasExplicitHooks) {
  using namespace tvm;
  ExpectStructuralHooks<PointerTypeNode>();
  ExpectStructuralHooks<TupleTypeNode>();
  ExpectStructuralHooks<FuncTypeNode>();
  ExpectStructuralHooks<TensorMapTypeNode>();
  ExpectStructuralHooks<relax::PackedFuncTypeNode>();
  ExpectStructuralHooks<relax::AnyTypeNode>();
  ExpectStructuralHooks<relax::ShapeTypeNode>();
  ExpectStructuralHooks<relax::TensorTypeNode>();
  ExpectStructuralHooks<relax::FuncTypeNode>();
  ExpectStructuralHooks<relax::distributed::DTensorTypeNode>();
  ExpectStructuralHooks<tirx::BufferRegionTypeNode>();
}

TEST(TypeStructuralHooks, RelaxFuncTypeParametersUsePatternDefinitionRegion) {
  using namespace tvm;
  tirx::PrimVar symbolic_extent("n", PrimType::Int(64));
  relax::TensorType tensor_type(relax::ShapeExpr(ffi::Array<PrimExpr>{symbolic_extent}),
                                PrimType::Float(32));
  relax::FuncType input({tensor_type}, tensor_type, true);
  std::vector<TVMFFIDefRegionKind> observed_regions;
  auto observe_var = [&](const VarNode*,
                         TVMFFIDefRegionKind region) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    observed_regions.push_back(region);
    return ffi::Unchanged();
  };

  relax::FuncType mapped =
      ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(input, observe_var).cast<relax::FuncType>();

  EXPECT_TRUE(mapped.same_as(input));
  ASSERT_FALSE(observed_regions.empty());
  EXPECT_EQ(observed_regions.front(), kTVMFFIDefRegionKindPattern);
}

TEST(TypeStructuralHooks, StructuralMapDescendsThroughTypeFields) {
  using namespace tvm;
  Type input = TupleType(
      {PointerType(PrimType::Float(32), "global"), relax::TensorType(PrimType::Float(32), 2)});
  Type mapped = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
                    input,
                    [](const PrimType& type) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
                      if (!type.MatchesElementType(DLDataTypeCode::kDLFloat, 32)) {
                        return ffi::Unchanged();
                      }
                      return ffi::Any(PrimType::Float(64));
                    })
                    .cast<Type>();

  const auto* tuple = mapped.as<TupleTypeNode>();
  ASSERT_NE(tuple, nullptr);
  EXPECT_TRUE(tuple->fields[0].as<PointerTypeNode>()->element_type.as<PrimTypeNode>()->dtype ==
              PrimType::Float(64)->dtype);
  EXPECT_TRUE(tuple->fields[1].as<relax::TensorTypeNode>()->dtype.value()->dtype ==
              PrimType::Float(64)->dtype);
}

TEST(TypeStructuralHooks, StructuralEqualAndHashStillUseAllReflectedFields) {
  using namespace tvm;
  ffi::StructuralEqual equal;
  ffi::StructuralHash hash;

  PointerType global(PrimType::Float(32), "global");
  PointerType shared(PrimType::Float(32), "shared");
  EXPECT_FALSE(equal(global, shared));
  EXPECT_NE(hash(global), hash(shared));

  relax::ShapeType rank_one(1);
  relax::ShapeType rank_two(2);
  EXPECT_FALSE(equal(rank_one, rank_two));
  EXPECT_NE(hash(rank_one), hash(rank_two));

  relax::TensorType f32(PrimType::Float(32), 2);
  relax::TensorType f64(PrimType::Float(64), 2);
  EXPECT_FALSE(equal(f32, f64));
  EXPECT_NE(hash(f32), hash(f64));

  relax::FuncType pure({}, relax::AnyType(), true);
  relax::FuncType impure({}, relax::AnyType(), false);
  EXPECT_FALSE(equal(pure, impure));
  EXPECT_NE(hash(pure), hash(impure));
}

}  // namespace
