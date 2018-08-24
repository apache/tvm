/*!
 * Copyright (c) 2018 by Contributors
 * \file graph_annotate.h
 * \brief Define rules to annotate a graph. The annotation rules/properties is
 * implemented similarly to the selection of subgraph nodes in mxnet.
 */
#ifndef NNVM_GRAPH_ANNOTATE_H_
#define NNVM_GRAPH_ANNOTATE_H_

#include <dlpack/dlpack.h>
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <tvm/runtime/device_api.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nnvm {
namespace op {

// TODO(chzhi) Use a config file to store the operator whitelist.
// static constexpr const char* kOpDeviceConfigFile = "op_devive_config.json";

/*
 * This class provides criteria for annotating nodes in a graph. It is an
 * abstract class that can be derived by other class to implement different
 * annotation rules. Rules could be designed either simply based on a vendor
 * provided whitelist (e.g. which node is best fitted to which device) or  using
 * a more intelligent scheme where complicated algorithm is designed to guide
 * annotation (future work).
 */
class AnnotationOpSelector {
 public:
  AnnotationOpSelector() = default;
  virtual ~AnnotationOpSelector() {}
  // Determine the device that the node should be scheduled to. This is a pure
  // virtual function that will be implmented by the children classes for
  // different annotation strategies.
  virtual DLDeviceType Select(const nnvm::Node* n) const = 0;
};

using AnnotationOpSelectorPtr = std::shared_ptr<AnnotationOpSelector>;

/*!
 * \brief This provides a set of properties to annotate nodes.
 */
class AnnotationOpProperty {
  using op_device_map_t_ =
      std::unordered_map<DLDeviceType, std::unordered_set<std::string>,
                         tvm::runtime::DLDeviceTypeHash>;

 public:
  AnnotationOpProperty() = default;

  // Create the rule to annotate a graph.
  virtual AnnotationOpSelectorPtr CreateAnnotationOpSelector() const = 0;

 private:
  op_device_map_t_ op_device_map_;
};

using AnnotationOpPropertyPtr = std::shared_ptr<AnnotationOpProperty>;

/*
 * This returns the a suitable device for nodes in a graph if the contained
 * operator is in the given set.
 */
class ContainOpSelector : public AnnotationOpSelector {
 public:
  explicit ContainOpSelector(
      std::shared_ptr<const std::unordered_set<std::string>> op_names) {
    op_names_ = op_names;
  }

  // TODO(chzhi) Make a config file contain <op name, device_name> pairs
  // Currently, we assume the default device is opencl when heterogeneous
  // execution is invoked. Users can specify some operators to CPU at the Python
  // frontend for annotation. Set the default as the fallback device (CPU) in
  // the future, and annotate according to the whitelist.
  DLDeviceType Select(const nnvm::Node* n) const final {
    if (n->is_variable()) return tvm::runtime::kDLDefaultDevice;

    if (op_names_->count(n->op()->name)) return kDLCPU;

    // Inference simplification will unpack batch_norm into an array of ops
    // starting with "batch_norm". All these operators should be annotated with
    // the same device as the bn operator. For example, all unpacked nodes are
    // annotated with CPU if batch_norm is specified to be scheduled to CPU.
    if (n->attrs.name.rfind("batch_norm") == 0 &&
        op_names_->count("batch_norm")) {
      return kDLCPU;
    }

    return tvm::runtime::kDLDefaultDevice;
  }

 private:
  std::shared_ptr<const std::unordered_set<std::string>> op_names_;
};

/*
 * This default property finds nodes with operators in a set.
 */
class DefaultAnnotationOpProperty : public AnnotationOpProperty {
 public:
  explicit DefaultAnnotationOpProperty(
      const std::unordered_set<std::string>& op_names)
      : op_names_(std::make_shared<std::unordered_set<std::string>>(op_names)) {
  }

  virtual AnnotationOpSelectorPtr CreateAnnotationOpSelector() const {
    return std::make_shared<ContainOpSelector>(op_names_);
  }

 private:
  std::shared_ptr<const std::unordered_set<std::string>> op_names_;
};

}  // namespace op
}  // namespace nnvm

#endif  // NNVM_GRAPH_ANNOTATE_H_
