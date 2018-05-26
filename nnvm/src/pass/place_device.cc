/*!
 *  Copyright (c) 2016 by Contributors
 * \file place_device.cc
 * \brief Inference the device of each operator given known information.
 *  Insert a copy node automatically when there is a cross device.
 */
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

namespace nnvm {
namespace pass {

// simply logic to place device according to device_group hint
// insert copy node when there is
Graph PlaceDevice(Graph src) {
  CHECK_NE(src.attrs.count("device_group_attr_key"), 0)
      << "Need graph attribute \"device_group_attr_key\" in PlaceDevice";
  CHECK_NE(src.attrs.count("device_assign_map"), 0)
      << "Need graph attribute \"device_assign_map\" in PlaceDevice";
  CHECK_NE(src.attrs.count("device_copy_op"), 0)
      << "Need graph attribute \"device_copy_op\" in PlaceDevice";

  std::string device_group_attr_key = src.GetAttr<std::string>("device_group_attr_key");
  const Op* copy_op = Op::Get(src.GetAttr<std::string>("device_copy_op"));
  auto& device_assign_map = src.GetAttr<DeviceAssignMap>("device_assign_map");
  const IndexedGraph& idx = src.indexed_graph();

  DeviceVector device(idx.num_nodes(), -1);
  // forward pass
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    auto it = inode.source->attrs.dict.find(device_group_attr_key);
    if (it != inode.source->attrs.dict.end()) {
      const std::string& device_group = it->second;
      auto dit = device_assign_map.find(device_group);
      CHECK_NE(dit, device_assign_map.end())
          << "The device assignment not found for group " << device_group;
      device[nid] = dit->second;
    } else {
      for (const IndexedGraph::NodeEntry& e : inode.inputs) {
        if (device[e.node_id] != -1) {
          device[nid] = device[e.node_id]; break;
        }
      }
    }
  }
  // backward pass
  for (uint32_t i = idx.num_nodes(); i != 0; --i) {
    uint32_t nid = i - 1;
    const auto& inode = idx[nid];
    if (device[nid] == -1) continue;
    for (const IndexedGraph::NodeEntry& e : inode.inputs) {
      if (device[e.node_id] == -1) device[e.node_id] = device[nid];
    }
  }

  int num_dev = 1, other_dev_id = -1;
  for (int& dev : device) {
    if (dev == -1) dev = 0;
    if (dev != other_dev_id) {
      if (other_dev_id != -1) ++num_dev;
      other_dev_id = dev;
    }
  }

  if (num_dev == 1) {
    src.attrs.erase("device_group_attr_key");
    src.attrs.erase("device_assign_map");
    src.attrs.erase("device_copy_op");
    src.attrs["device"] = std::make_shared<any>(std::move(device));
    return src;
  }

  std::map<std::tuple<uint32_t, uint32_t, int>, NodePtr> copy_map;
  std::vector<NodePtr> new_node_map(idx.num_nodes(), nullptr);
  std::unordered_map<const Node*, int> new_device_map;

  // insert copy node
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    int dev_id = device[nid];
    const auto& inode = idx[nid];
    // check if mutation is needed
    bool need_mutate = false;
    for (const IndexedGraph::NodeEntry& e : inode.inputs) {
      if (new_node_map[e.node_id] != nullptr || dev_id != device[e.node_id]) {
        need_mutate = true; break;
      }
    }
    if (!need_mutate) {
      for (const uint32_t cid : inode.control_deps) {
        if (new_node_map[cid] != nullptr)  {
          need_mutate = true; break;
        }
      }
    }
    if (need_mutate) {
      NodePtr new_node = Node::Create();
      new_node->attrs = inode.source->attrs;
      new_node->inputs.reserve(inode.inputs.size());
      for (size_t i = 0; i < inode.inputs.size(); ++i) {
        const IndexedGraph::NodeEntry& e = inode.inputs[i];
        if (dev_id != device[e.node_id]) {
          auto copy_key = std::make_tuple(e.node_id, e.index, dev_id);
          auto it = copy_map.find(copy_key);
          if (it != copy_map.end() && it->first == copy_key) {
            new_node->inputs.emplace_back(
                NodeEntry{it->second, 0, 0});
          } else {
            NodePtr copy_node = Node::Create();
            copy_node->op = copy_op;
            std::ostringstream os;
            os << inode.source->inputs[i].node->attrs.name << "_" << e.index <<"_copy";
            copy_node->attrs.name = os.str();
            copy_node->inputs.push_back(inode.source->inputs[i]);
            copy_map[copy_key] = copy_node;
            new_device_map[copy_node.get()] = dev_id;
            new_node->inputs.emplace_back(
                NodeEntry{std::move(copy_node), 0, 0});
          }
        } else {
          if (new_node_map[e.node_id] != nullptr) {
            new_node->inputs.emplace_back(
                NodeEntry{new_node_map[e.node_id], e.index, 0});
        } else {
            new_node->inputs.push_back(inode.source->inputs[i]);
          }
        }
      }
      new_node->control_deps.reserve(inode.control_deps.size());
      for (size_t i = 0; i < inode.control_deps.size(); ++i) {
        uint32_t cid = inode.control_deps[i];
        if (new_node_map[cid] != nullptr) {
          new_node->control_deps.push_back(new_node_map[cid]);
        } else {
          new_node->control_deps.push_back(inode.source->control_deps[i]);
        }
      }
      new_device_map[new_node.get()] = dev_id;
      new_node_map[nid] = std::move(new_node);
    } else {
      new_device_map[inode.source] = dev_id;
    }
  }

  // make the new graph
  Graph ret;
  for (const NodeEntry& e : src.outputs) {
    if (new_node_map[idx.node_id(e.node.get())] != nullptr) {
      ret.outputs.emplace_back(
          NodeEntry{new_node_map[idx.node_id(e.node.get())], e.index, e.version});
    } else {
      ret.outputs.emplace_back(e);
    }
  }
  DeviceVector new_device_vec(ret.indexed_graph().num_nodes());
  for (uint32_t nid = 0; nid < ret.indexed_graph().num_nodes(); ++nid) {
    if (new_device_map.count(ret.indexed_graph()[nid].source) == 0) {
      LOG(INFO) << "canot find " << ret.indexed_graph()[nid].source->attrs.name;
    }
    new_device_vec[nid] = new_device_map.at(ret.indexed_graph()[nid].source);
  }
  ret.attrs["device"] = std::make_shared<any>(std::move(new_device_vec));
  return ret;
}

NNVM_REGISTER_PASS(PlaceDevice)
.describe("Infer the device type of each operator."\
          "Insert a copy node when there is cross device copy")
.set_body(PlaceDevice)
.set_change_graph(true)
.provide_graph_attr("device")
.depend_graph_attr("device_group_attr_key")
.depend_graph_attr("device_assign_map")
.depend_graph_attr("device_copy_op");

DMLC_JSON_ENABLE_ANY(DeviceAssignMap, dict_str_int);

}  // namespace pass
}  // namespace nnvm
