/*!
 *  Copyright (c) 2016 by Contributors
 * \file plan_memory.cc
 * \brief Assign memory tag to each of the data entries.
 */
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/op_attr_types.h>
#include <memory>
#include "./graph_algorithm.h"

namespace nnvm {
namespace pass {
namespace {

// simple graph based allocator.
class GraphAllocator {
 public:
  // storage id equals integer.
  using StorageID = int;

  // bad storage id
  static const StorageID kBadStorageID = -1;
  // external storage id
  static const StorageID kExternalStorageID = -2;

  // request a free storage
  StorageID Request(int dev_id, int dtype, TShape shape, uint32_t node_id) {
    if (shape.ndim() == 0) return kBadStorageID;
    // search memory block in [size / match_range_, size * match_range_)
    // TODO(tqchen) add size of the dtype, assume 4 bytes for now
    size_t size = shape.Size() * 4;
    if (match_range_ == 0) return this->Alloc(dev_id, size);
    auto begin = free_.lower_bound(size / match_range_);
    auto mid = free_.lower_bound(size);
    auto end = free_.upper_bound(size * match_range_);
    // search for memory blocks larger than requested
    for (auto it = mid; it != end; ++it) {
      StorageEntry *e = it->second;
      if (e->device_id != dev_id) continue;
      if (node_color_.size() != 0 &&
          node_color_[e->released_by_node] != node_color_[node_id]) continue;
      // Use exect matching strategy
      e->max_bytes = std::max(size, e->max_bytes);
      // find a exact match, erase from map and return
      free_.erase(it);
      return e->id;
    }
    // then search for memory blocks smaller than requested space
    for (auto it = mid; it != begin;) {
      --it;
      StorageEntry *e = it->second;
      if (e->device_id != dev_id) continue;
      if (node_color_.size() != 0 &&
          node_color_[e->released_by_node] != node_color_[node_id]) continue;
      // Use exect matching strategy
      e->max_bytes = std::max(size, e->max_bytes);
      // erase from map and return
      free_.erase(it);
      return e->id;
    }
    // cannot find anything return a new one.
    return this->Alloc(dev_id, size);
  }
  // release a memory space.
  void Release(StorageID id, uint32_t node_id) {
    CHECK_NE(id, kBadStorageID);
    if (id == kExternalStorageID) return;
    StorageEntry *e = data_[id].get();
    e->released_by_node = node_id;
    free_.insert({e->max_bytes, e});
  }

  // totoal number of bytes allocated
  size_t TotalAllocBytes() const {
    size_t total = 0;
    for (auto &p : data_) {
      total += p->max_bytes;
    }
    return total;
  }

  // constructor
  explicit GraphAllocator(const IndexedGraph* idx, const size_t match_range) : idx_(idx) {
    this->Init(match_range, dmlc::GetEnv("NNVM_EXEC_NUM_TEMP", 1));
  }

 private:
  // initialize the graph allocator
  void Init(const size_t match_range, const uint32_t num_match_color) {
    match_range_ = match_range;
    num_match_color_ = num_match_color;
    if (num_match_color_ > 1) {
      std::vector<uint32_t> importance(idx_->num_nodes(), 0);
      for (uint32_t nid = 0; nid < idx_->num_nodes(); ++nid) {
        if ((*idx_)[nid].source->is_variable()) continue;
        importance[nid] = 1;
      }
      num_match_color_ = pass::ColorNodeGroup(
          *idx_, importance, num_match_color_, &node_color_);
    }
  }

  StorageID Alloc(int dev_id, size_t size) {
    StorageID id = static_cast<StorageID>(data_.size());
    std::unique_ptr<StorageEntry> ptr(new StorageEntry());
    ptr->id = id;
    ptr->device_id = dev_id;
    ptr->max_bytes = size;
    data_.emplace_back(std::move(ptr));
    return id;
  }
  // internal storage entry
  struct StorageEntry {
    // the id of the entry.
    StorageID id;
    // the device id of the storage.
    int device_id;
    // maximum size of storage requested.
    size_t max_bytes{0};
    // node index that released it last time
    uint32_t released_by_node{0};
  };
  // scale used for rough match
  size_t match_range_;
  // whether use color based match algorithm
  uint32_t num_match_color_{1};
  // the size of each dtype
  std::vector<size_t> dtype_size_dict_;
  // free list of storage entry
  std::multimap<size_t, StorageEntry*> free_;
  // all the storage resources available
  std::vector<std::unique_ptr<StorageEntry> > data_;
  // color of nodes in the graph, used for auxiliary policy making.
  std::vector<uint32_t> node_color_;
  // internal indexed graph
  const IndexedGraph* idx_;
};

/*
 * Internal method to perform the memory allocation for a graph
 * */
size_t AllocMemory(const Graph& ret, const IndexedGraph& idx, StorageVector* storage_ptr,
                   std::vector<int>* storage_inplace_index_ptr, std::vector<uint32_t> ref_count,
                   GraphAllocator* allocator) {
  // Get reference
  auto &storage = *storage_ptr;
  auto &storage_inplace_index = *storage_inplace_index_ptr;

  // Get attributes from the graph
  const ShapeVector& shape_vec = ret.GetAttr<ShapeVector>("shape");
  const DTypeVector& dtype_vec = ret.GetAttr<DTypeVector>("dtype");
  const DeviceVector* device_vec = nullptr;
  static auto& finplace_option = Op::GetAttr<FInplaceOption>("FInplaceOption");

  if (ret.attrs.count("device") != 0) {
    device_vec = &(ret.GetAttr<DeviceVector>("device"));
  }
  size_t num_not_allocated = 0;

  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    // check inplace option
    if (finplace_option.count(inode.source->op()) != 0) {
      auto inplace_pairs = finplace_option[inode.source->op()](inode.source->attrs);
      for (auto& kv : inplace_pairs) {
        uint32_t eid_out = idx.entry_id(nid, kv.second);
        uint32_t eid_in = idx.entry_id(inode.inputs[kv.first]);
        if (ref_count[eid_in] == 1 &&
            ref_count[eid_out] != 0 &&
            storage[eid_out] == GraphAllocator::kBadStorageID &&
            storage[eid_in] >= 0 &&
            shape_vec[eid_out].Size() == shape_vec[eid_in].Size() &&
            dtype_vec[eid_out] == dtype_vec[eid_in]) {
          // inplace optimization
          storage[eid_out] = storage[eid_in];
          ref_count[eid_in] = 0;
          storage_inplace_index[eid_out] = kv.first;
        }
      }
    }
    // normal allocation
    const int dev_id = (device_vec != nullptr) ? device_vec->at(nid) : 0;
    // sort output nodes based on size before allocating output
    std::multimap<size_t, uint32_t> eids;
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
      uint32_t eid = idx.entry_id(nid, index);
      if (storage[eid] == GraphAllocator::kBadStorageID) {
        auto &eshape = shape_vec[eid];
        size_t esize = 0;
        if (eshape.ndim() != 0) esize = eshape.Size();
        eids.insert(std::make_pair(esize, eid));
      }
    }
    for (auto rit = eids.rbegin(); rit != eids.rend(); ++rit) {
        uint32_t eid = rit->second;
        storage[eid] = allocator->Request(dev_id, dtype_vec[eid], shape_vec[eid], nid);
    }

    // check if certain inputs is ignored.
    static auto& fignore_inputs = Op::GetAttr<FIgnoreInputs>("FIgnoreInputs");
    std::vector<uint32_t> ignore_inputs;
    if (fignore_inputs.count(inode.source->op()) != 0) {
      ignore_inputs = fignore_inputs[inode.source->op()](inode.source->attrs);
      std::sort(ignore_inputs.begin(), ignore_inputs.end());
    }
    // then free inputs
    for (size_t i = 0; i < inode.inputs.size(); ++i) {
      // ref counter of ignored input is already decreased.
      if (std::binary_search(ignore_inputs.begin(), ignore_inputs.end(), i)) continue;
      const auto& e = inode.inputs[i];
      uint32_t eid = idx.entry_id(e);
      // temp_ref_count == 0 means it is taken by inplace op
      if (ref_count[eid] == 0) continue;
      // if we decrease it to zero, means we are ready to relase
      --ref_count[eid];
      if (ref_count[eid] == 0 && storage[eid] != GraphAllocator::kBadStorageID) {
        allocator->Release(storage[eid], nid);
      }
    }
    // check if there are outputs that can be freeded immediately
    // these output are not referenced by any operator.
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
      uint32_t eid = idx.entry_id(nid, index);
      if (ref_count[eid] == 0 && storage[eid] != GraphAllocator::kBadStorageID) {
        allocator->Release(storage[eid], nid);
        // use -2 to indicate that the node was never touched.
        storage_inplace_index[eid] = -2;
      }
      if (storage[eid] == GraphAllocator::kBadStorageID) {
        ++num_not_allocated;
      }
    }
  }
  return num_not_allocated;
}


// function to plan memory
Graph PlanMemory(Graph ret) {
  // setup ref counter
  const IndexedGraph& idx = ret.indexed_graph();
  static auto& fignore_inputs = Op::GetAttr<FIgnoreInputs>("FIgnoreInputs");
  // reference counter of each node
  std::vector<uint32_t> ref_count(idx.num_node_entries(), 0);
  // step 1: initialize reference count
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    for (const auto& e : inode.inputs) {
      ++ref_count[idx.entry_id(e)];
    }
    // no dataflow dependency is needed for those are ignored.
    // revoke the dependency counter.
    if (fignore_inputs.count(inode.source->op()) != 0) {
      auto ignore_inputs = fignore_inputs[inode.source->op()](inode.source->attrs);
      for (uint32_t i : ignore_inputs) {
        --ref_count[idx.entry_id(inode.inputs[i])];
      }
    }
  }
  for (const auto& e : idx.outputs()) {
    ++ref_count[idx.entry_id(e)];
  }
  // step 2: allocate memory.
  StorageVector storage;
  if (ret.attrs.count("storage") != 0) {
    storage = ret.MoveCopyAttr<StorageVector>("storage");
  } else {
    storage.resize(idx.num_node_entries(), -1);
  }

  // Search the best NNVM_EXEC_MATCH_RANGE parameter. This is turned off by default
  size_t min_allocated_bytes = -1;
  size_t max_match_range = dmlc::GetEnv("NNVM_EXEC_MATCH_RANGE", 16);
  size_t min_match_range =
         dmlc::GetEnv("NNVM_AUTO_SEARCH_MATCH_RANGE", false) ? 1 : max_match_range;
  for (size_t match_range = min_match_range; match_range <= max_match_range; match_range *= 2) {
    // Make a copy of related fields
    StorageVector storage_vec(storage);
    std::vector<int> storage_inplace_index(idx.num_node_entries(), -1);

    // the allocator
    GraphAllocator allocator(&idx, match_range);

    // number of entries that are not statically allocated.
    size_t storage_num_not_allocated =
      AllocMemory(ret, idx, &storage_vec, &storage_inplace_index, ref_count, &allocator);
    size_t storage_allocated_bytes = allocator.TotalAllocBytes();
    // Choose the plan which leads to minimal memory usage
    if (min_allocated_bytes > storage_allocated_bytes) {
      ret.attrs["storage_id"] = std::make_shared<any>(std::move(storage_vec));
      ret.attrs["storage_inplace_index"] = std::make_shared<any>(std::move(storage_inplace_index));
      ret.attrs["storage_allocated_bytes"] = std::make_shared<any>(storage_allocated_bytes);
      ret.attrs["storage_num_not_allocated"] = std::make_shared<any>(storage_num_not_allocated);
      min_allocated_bytes = storage_allocated_bytes;
    }
  }
  return ret;
}

NNVM_REGISTER_PASS(PlanMemory)
.describe("Plan the memory allocation of each node entries.")
.set_body(PlanMemory)
.set_change_graph(false)
.depend_graph_attr("dtype")
.depend_graph_attr("shape")
.provide_graph_attr("storage_id")
.provide_graph_attr("storage_inplace_index");

}  // namespace
}  // namespace pass
}  // namespace nnvm
