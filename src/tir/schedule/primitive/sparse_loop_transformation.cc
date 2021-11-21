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
#include "../utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Check whether the new iterators are valid. We say they are valid if the new order is a
 * permutation of the old order
 * \param new_order The new iterator order to be checked
 * \param old_order The old order of the iterators
 * \throw ScheduleError If the iterators in the new order are not valid
 */
void CheckValidInputIterators(const ScheduleState self, const Array<SpIterVar>& new_order,
                              const Array<SpIterVar>& old_order) {
  class LengthNotEqualError : public ScheduleError {
   public:
    explicit LengthNotEqualError(IRModule mod, Array<SpIterVar> old_order,
                                 Array<SpIterVar> new_order)
        : mod_(std::move(mod)), old_order_(std::move(old_order)), new_order_(std::move(new_order)) {
      ICHECK_NE(new_order_.size(), old_order_.size());
    }

    String FastErrorString() const final {
      return "ScheduleError: The number of iterators in the new order does not equal to the "
             "number of iterators in the old order";
    }

    String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "ScheduleError: The new order has " << new_order_.size() << " iterators" << new_order_
         << ", while the old order has " << old_order_.size() << " iterators" << old_order_
         << ". They are supposed to have the same set of iterators";
      return os.str();
    }

    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {}; }

    IRModule mod_;
    Array<SpIterVar> old_order_;
    Array<SpIterVar> new_order_;
  };

  class IterNotAppearError : public ScheduleError {
   public:
    explicit IterNotAppearError(IRModule mod, SpIterVar iter, Array<SpIterVar> old_order)
        : mod_(std::move(mod)), iter_(std::move(iter)), old_order_(std::move(old_order)) {}

    String FastErrorString() const final {
      return "ScheduleError: An iterator in the new order does not appear in the old order";
    }

    String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "ScheduleError: Iterator " << iter_
         << " appears in the new order. However, it does not appear in the old order "
         << old_order_;
      return os.str();
    }

    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {}; }

    IRModule mod_;
    SpIterVar iter_;
    Array<SpIterVar> old_order_;
  };

  if (new_order.size() != old_order.size()) {
    throw LengthNotEqualError(self->mod, new_order, old_order);
  }
  for (const SpIterVar& sp_iter : new_order) {
    if (std::find(old_order.begin(), old_order.end(), sp_iter) == old_order.end()) {
      throw IterNotAppearError(self->mod, sp_iter, old_order);
    }
  }
}

SparseBlock SparseReorder(ScheduleState self, const SparseBlock& block,
                          const Array<SpIterVar>& new_order) {
  // Step 1. Check whether the iterators in `new_order` are the same as `block`'s iterators.
  CheckValidInputIterators(self, new_order, block->sp_iter_vars);

  // Step 2. Check whether the new order does not break the iterator dependency.
  CheckDependency(self, block, new_order);

  // Step 3. Create the new SparseBlock.
  ObjectPtr<SparseBlockNode> p_new_block = make_object<SparseBlockNode>(*block.get());
  p_new_block->sp_iter_vars = new_order;
  SparseBlock new_block(p_new_block);

  // Step 4. Create the new IRModule. (The following lines are from Schedule::Replace(...))
  const PrimFuncNode* g_func = nullptr;
  GlobalVar g_var;
  g_func = GetPrimFuncFromSparseBlock(self->mod, block.get(), &g_var);

  IRModuleNode* new_mod = self->mod.CopyOnWrite();
  MapNode* new_map = new_mod->functions.CopyOnWrite();
  PrimFunc ref_new_func = Downcast<PrimFunc>(std::move(new_map->at(g_var)));
  ICHECK(ref_new_func.get() == g_func);
  PrimFuncNode* new_func = ref_new_func.CopyOnWrite();

  new_func->body = new_block;
  new_map->at(g_var) = std::move(ref_new_func);
  self->mod = GetRef<IRModule>(new_mod);

  return new_block;
}

}  // namespace tir
}  // namespace tvm
