/*!
 * Copyright (c) 2018 by Contributors
 * \file graph_annotate.cc
 * \brief NNVM pass to annotate a graph according to certain rules.
 */

#include <nnvm/graph.h>
#include <nnvm/graph_annotate.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <tvm/runtime/device_api.h>

#include <queue>
#include <stack>
#include <unordered_set>

namespace nnvm {
namespace op {

// Annotate nodes with the compilation target for homogeneous execution.
//
// Save the build target information. It will be used duirng compilatio
// since FTVMCompute and FTVMSchedule will need target to get correct
// compute and schedule.
nnvm::Graph AnnotateHomogeneousGraph(nnvm::Graph g) {
  DFSVisit(g.outputs, [&g](const nnvm::NodePtr &node) {
    // Annotate the compilation target on the node if it hasn't been added.
    node->attrs.dict["target"] = g.GetAttr<std::string>("target");
  });
  return g;
}

// Annotate graph nodes using vendor provided whitelist.
nnvm::Graph AnnotateHeterogeneousGraph(nnvm::Graph g) {
  const AnnotationOpPropertyPtr &annotate_prop =
      g.MoveCopyAttr<AnnotationOpPropertyPtr>("annotation_property");
  DFSVisit(g.outputs, [&annotate_prop, &g](const nnvm::NodePtr &node) {
    const auto &selector = annotate_prop->CreateAnnotationOpSelector();
    DLDeviceType device = selector->Select(node.get());
    const auto &device_name = tvm::runtime::DeviceName(device);
    const auto &target_ctx = "target" + device_name;
    CHECK(g.HasAttr(target_ctx))
        << device_name << " target hasn't been attached to the graph yet!";

    // Annotate device only when necessary. For instance, all nodes in a graph
    // should be scheduled to the same devcie (e.g. cpu) at the precompute
    // pruning pass.
    if (g.HasAttr("annotate_device")) {
      node->attrs.device = device;
    }
    node->attrs.dict["target"] = g.GetAttr<std::string>(target_ctx);
  });

  g.attrs["annotated"] = std::make_shared<any>("annotated");
  return g;
}

// Adjust nodes' device info in an annotated graph. The device info of inputs,
// like weights, of an annotated node is changed to be the same as the node when
// necessary.
// TODO(chzhi) Handle the case where an input is shared to by multiple nodes
// that are annotated with different device attributes.
nnvm::Graph AdjustAnnotation(nnvm::Graph g) {
  CHECK(g.HasAttr("annotated")) << "Graph has not been annotated. Apply "
                                   "Annotation pass before adjustment.";

  DFSVisit(g.outputs, [](const nnvm::NodePtr& node) {
    if (node->is_variable()) return;

    for (const auto& e : node->inputs) {
      if (e.node->op()) continue;

      if (e.node->attrs.device != node->attrs.device) {
        e.node->attrs.device = node->attrs.device;
      }
    }
  });

  const auto& idx = g.indexed_graph();
  DeviceVector device_vec(idx.num_nodes(), -1);
  for (size_t i = 0; i < idx.num_nodes(); i++) {
    device_vec[i] = static_cast<int>(idx[i].source->attrs.device);
  }
  g.attrs["device"] = std::make_shared<any>(std::move(device_vec));
  return g;
}

nnvm::Graph AnnotateGraph(nnvm::Graph&& g) {
  // The graph should always have a "target" attribute or multiple
  // "target+device_name" ("targetcpu")  attributes. The former indicates that
  // the graph will be compiled and executed on the same device, and the
  // latter requires to mark nondes with different device information, e.g.
  // device type and compilation target.

  if (g.HasAttr("target")) {
    g = AnnotateHomogeneousGraph(g);
  } else {
    CHECK(g.HasAttr("annotation_property"))
        << "The graph cannot be annotated because it has no"
           "annotation_property or target attribute attached.";
    g = AnnotateHeterogeneousGraph(g);
    // Adjust the annotated graph and Insert data copy nodes only when device
    // information is annotated.
    if (g.HasAttr("annotate_device")) {
      g = AdjustAnnotation(g);
      g = nnvm::ApplyPass(g, "PlaceDataCopy");
    }
  }

  nnvm::Graph ret;
  ret.outputs = g.outputs;
  return ret;
}

NNVM_REGISTER_PASS(AnnotateGraph)
    .describe(
        "Annotate the nodes  in a graph to indicate where it should be "
        "executed.")
    .set_body(AnnotateGraph)
    .set_change_graph(true);

}  // namespace op
}  // namespace nnvm
