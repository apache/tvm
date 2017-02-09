/*!
 *  Copyright (c) 2016 by Contributors
 * \file auto_inline_elem_wise.cc
 */
#include <tvm/schedule_pass.h>
#include <tvm/ir_pass.h>

namespace tvm {
namespace schedule {

void AutoInlineElemWise(Schedule sch) {
  for (Stage s : sch->stages) {
    if (!s.is_scheduled() && ir::IsElemWise(s->op)) {
      bool is_root = false;
      for (auto r : sch->roots) {
        if (r == s->op) {
          is_root = true;
          break;
        }
      }
      if (!is_root)
        s.compute_inline();
    }
  }
}

}  // namespace schedule
}  // namespace tvm
