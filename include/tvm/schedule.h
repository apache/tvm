/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule.h
 * \brief Define a schedule.
 */
#ifndef TVM_SCHEDULE_H_
#define TVM_SCHEDULE_H_

#include <string>
#include "./base.h"
#include "./split.h"
#include "./operation.h"

namespace tvm {

// Node container for Schedule
class ScheduleNode;
// Node container for AttachSpec
class AttachSpecNode;

/*! \brief the attachment type */
enum AttachType : int {
  kRoot = 0,
  kInline = 1,
  kSplit = 2
};

/*! \brief schedule container */
class Schedule : public NodeRef {
 public:
  Schedule() {}
  explicit Schedule(std::shared_ptr<Node> n) : NodeRef(n) {}
  Schedule(Operation op, std::string scope);
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const ScheduleNode* operator->() const;
};

// defintion of node containers
/*! \brief represents the schedule of the tensor */
class ScheduleNode : public Node {
 public:
  /*! \brief The operation to be scheduled */
  Operation op;
  /*! \brief The thread scope level of the schedule */
  std::string scope;
  /*! \brief Splits over iteration domains */
  Array<Split> splits;
  /*! \brief The attachment type of the schedule */
  AttachType attach_type;
  /*!
   * \brief The attach point of this schedule, if it is a split
   * \note This is not a cyclic dependency,
   *  because split do not refer back to parent schedule.
   */
  Split attach_parent;
  /*! \brief the schedules that this schedule depend on */
  Array<Schedule> children;
  // the type key
  const char* type_key() const final {
    return "Schedule";
  }
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("scope", &scope);
    v->Visit("op", &op);
    v->Visit("splits", &splits);
    v->Visit("attach_type", &attach_type);
    v->Visit("attach_parent", &attach_parent);
    v->Visit("children", &children);
  }
};

// implementations
inline const ScheduleNode* Schedule::operator->() const {
  return static_cast<const ScheduleNode*>(node_.get());
}

}  // namespace tvm
#endif  // TVM_SCHEDULE_H_
