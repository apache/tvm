/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule.cc
 */
#include <tvm/schedule_pass.h>
#include <tvm/ir_pass.h>

namespace tvm {
namespace schedule {

namespace {
inline bool is_stage_scheduled(const Stage& s) {
  return !(s->relations.empty() && s->attach_type == kNone);
}
}

void AutoFuseElemWise(Schedule sch) {
  for (Stage s : sch->stages) {
    if (!is_stage_scheduled(s) && ir::IsElemWise(s->op)) {
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
