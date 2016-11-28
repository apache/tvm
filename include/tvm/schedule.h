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

/*! \brief schedule container */
class AttachSpec : public NodeRef {
 public:
  AttachSpec() {}
  explicit AttachSpec(std::shared_ptr<Node> n) : NodeRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const AttachSpecNode* operator->() const;
};

// defintion of node containers

/*! \brief The attach specification of each subschedule */
class AttachSpecNode : public Node {
 public:
  /*! \brief The attachment type */
  AttachType attach_type;
  /*!
   * \brief The split to be attached to,
   *  only valid when attach_type is kRoot
   */
  Split attach_split;
  /*! \brief the child schedule to be attached. */
  Schedule schedule;
  const char* type_key() const final {
    return "AttachSpec";
  }
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("attach_type", &attach_type);
    v->Visit("attach_split", &attach_split);
    v->Visit("schedule", &schedule);
  }
};

/*! \brief represents the schedule of the tensor */
class ScheduleNode : public Node {
 public:
  /*! \brief The operation to be scheduled */
  Operation op;
  /*! \brief The thread scope level of the schedule */
  std::string scope;
  /*! \brief Splits over iteration domains */
  Array<Split> splits;
  /*! \brief attach specifications */
  Array<AttachSpec> attachs;
  const char* type_key() const final {
    return "Schedule";
  }
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("scope", &scope);
    v->Visit("op", &op);
    v->Visit("splits", &splits);
    v->Visit("attachs", &attachs);
  }
};

// implementations
inline const ScheduleNode* Schedule::operator->() const {
  return static_cast<const ScheduleNode*>(node_.get());
}

inline const AttachSpecNode* AttachSpec::operator->() const {
  return static_cast<const AttachSpecNode*>(node_.get());
}

}  // namespace tvm
#endif  // TVM_SCHEDULE_H_
