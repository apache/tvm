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

#ifndef TVM_RUNTIME_CONTRIB_DNNL_DNNL_NODE_HELPER_H_
#define TVM_RUNTIME_CONTRIB_DNNL_DNNL_NODE_HELPER_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../json/json_runtime.h"
#include "dnnl.hpp"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace dnnl;

namespace utils {

/** Converter helper for shape objects */
inline static dnnl::memory::dims convert2dnnl(std::vector<int64_t> shape) {
  if (shape.empty()) return {1};  // DNNL scalar representation
  return shape;
}

/** Converter helper for data type objects */
inline static dnnl::memory::data_type convert2dnnl(DLDataType dtype) {
  if (dtype.code == DLDataTypeCode::kDLInt) {
    if (dtype.bits == 8) return dnnl::memory::data_type::s8;
    if (dtype.bits == 32) return dnnl::memory::data_type::s32;
  } else if (dtype.code == DLDataTypeCode::kDLUInt) {
    if (dtype.bits == 8) return dnnl::memory::data_type::u8;
  } else if (dtype.code == DLDataTypeCode::kDLFloat) {
    if (dtype.bits == 16) return dnnl::memory::data_type::f16;
    if (dtype.bits == 32) return dnnl::memory::data_type::f32;
  } else if (dtype.code == DLDataTypeCode::kDLBfloat) {
    if (dtype.bits == 16) return dnnl::memory::data_type::bf16;
  }
  LOG(FATAL) << "Data type is not supported";
  return dnnl::memory::data_type::undef;
}

/** Converter of primitive types to corresponding DNNL data type */
template <typename T>
dnnl::memory::data_type dnnlDType();
template <>
dnnl::memory::data_type dnnlDType<int>() {
  return dnnl::memory::data_type::s32;
}
template <>
dnnl::memory::data_type dnnlDType<float>() {
  return dnnl::memory::data_type::f32;
}

/** Generator of dnnl format_tag for plain version of tensor */
inline static dnnl::memory::format_tag plainLayout(uint32_t rank) {
  switch (rank) {
    case 0:
    case 1:
      return dnnl::memory::format_tag::a;
    case 2:
      return dnnl::memory::format_tag::ab;
    case 3:
      return dnnl::memory::format_tag::abc;
    case 4:
      return dnnl::memory::format_tag::abcd;
    case 5:
      return dnnl::memory::format_tag::abcde;
    case 6:
      return dnnl::memory::format_tag::abcdef;
    default:
      LOG(FATAL) << "Unsupported data tensor rank: " << rank;
      break;
  }
  return dnnl::memory::format_tag::undef;
}

inline static dnnl::memory::desc makePlainTDesc(const std::vector<int64_t>& shape,
                                                const DLDataType& dtype) {
  return {convert2dnnl(shape), convert2dnnl(dtype), plainLayout(shape.size())};
}

/** Builder of dnnl memory on top of provided DLTensor */
dnnl::memory convert2dnnl(const DLTensor* dl_tensor, const dnnl::engine engine) {
  // TODO(apeskov): assume that data is always in plain format, check if it's true
  ICHECK(dl_tensor->strides == nullptr);
  ICHECK_EQ(dl_tensor->device.device_type, kDLCPU);

  std::vector<int64_t> dl_dims(dl_tensor->shape, dl_tensor->shape + dl_tensor->ndim);
  dnnl::memory::desc desc{convert2dnnl(dl_dims), convert2dnnl(dl_tensor->dtype),
                          plainLayout(dl_dims.size())};

  desc.data.offset0 = dl_tensor->byte_offset;
  return {desc, engine, dl_tensor->data};
}

/** Converter helper for Eltwise op name proper dnnl::algorithm value */
dnnl::algorithm convert2dnnl_activation(std::string name) {
  if (name == "nn.relu")
    return dnnl::algorithm::eltwise_relu;
  else if (name == "clip")
    return dnnl::algorithm::eltwise_clip;
  else if (name == "gelu")
    return dnnl::algorithm::eltwise_gelu;
  else if (name == "tanh")
    return dnnl::algorithm::eltwise_tanh;
  else if (name == "sqrt")
    return dnnl::algorithm::eltwise_sqrt;
  else if (name == "sigmoid")
    return dnnl::algorithm::eltwise_logistic;
  else
    LOG(FATAL) << "Unknown activation name";

  return dnnl::algorithm::undef;
}

/** Find a permutation of chars in src string to achieve a ref string version */
inline static std::vector<int> permutation(const std::string& src, const std::string& ref) {
  std::set<char> chars(src.begin(), src.end());
  ICHECK_EQ(chars.size(), src.size()) << "\"" << src << "\" has a duplicate symbols";

  std::vector<int> perm;
  for (const auto& c : src) {
    auto found = ref.find(c);
    ICHECK_NE(found, std::string::npos) << "\"" << src << "\" is not a permutation of "
                                        << "\"" << ref << "\"";
    perm.push_back(found);
  }
  return perm;
}

/** Data copy function */
void copy_now(const dnnl::memory& src, const dnnl::memory& dst) {
  auto reorder = dnnl::reorder(src, dst);
  auto stream = dnnl::stream(src.get_engine());
  // DNNL api requires non const ref for src. Have to use const_cast
  auto src_non_const = const_cast<dnnl::memory&>(src);
  auto dst_non_const = const_cast<dnnl::memory&>(dst);
  reorder.execute(stream, src_non_const, dst_non_const);
}

}  // namespace utils

/**
 * Helper object to simplify handling of tensor
 *
 * Allow to specify tensor in future and actions which should be applied to it.
 * Can be treated couple of future tensor source reference and list of action which should be
 * applied to this tensor. Finally TensorRequisite object should be registered in TensorRegistry.
 *
 * @note: Empty TR object allow any manipulation. Empty TR will be returned.
 *
 * Like:
 *   source - input tensor on position 3
 *   actions - reinterpret like a plain 1D tensor
 *
 * Example:
 *   auto tr = node.getInput(3);  // source is node input #3
 *   tr = tr.permute({1, 2, 0});  // permute axes chw -> hwc
 *   tr = tr.crop({128, 128, 1}, {0, 0, 0}); // extract first channel
 *   tr = tr.squeeze();
 *
 *   submit(prim, {DNNL_ARG_SRC, tr});
 */
class TensorRequisite {
 public:
  static constexpr uint32_t INVALID_EID = std::numeric_limits<uint32_t>::max() - 1;

  TensorRequisite() {}

  /** return shape of tensor */
  dnnl::memory::dims dims() const { return t_desc_.dims(); }
  /** return tensor desc */
  dnnl::memory::desc desc() const { return t_desc_; }

  /** Produce tensor with permuted axes */
  TensorRequisite permute(const std::vector<int>& permutation) const {
    if (!defined()) return *this;  // nothing for empty TR

    auto orig = std::make_shared<TensorRequisite>(*this);
    // reinterpret memory buffer with new strides
    auto desc = t_desc_.permute_axes(permutation);
    return {desc, orig, true, {}, INVALID_EID, reverse_data_flow_};
  }

  /** Produce tensor with reinterpret data of original tr */
  TensorRequisite reshape(const dnnl::memory::dims& shape) const {
    if (!defined()) return *this;  // nothing for empty TR
    if (t_desc_.dims() == shape) return *this;

    auto orig = std::make_shared<TensorRequisite>(*this);
    // reinterpret memory buffer with new strides
    auto desc = t_desc_.reshape(shape);
    return {desc, orig, true, {}, INVALID_EID, reverse_data_flow_};
  }

  /** Produce tensor with broadcasted values */
  TensorRequisite broadcast(const dnnl::memory::dims& shape) const {
    if (!defined()) return *this;  // nothing for empty TR
    if (t_desc_.dims() == shape) return *this;
    ICHECK(!reverse_data_flow_);

    auto orig = std::make_shared<TensorRequisite>(*this);

    // numpy like broadcast
    auto extended_dims = t_desc_.dims();
    auto one_filled = dnnl::memory::dims(shape.size()-extended_dims.size(), 1);
    extended_dims.insert(extended_dims.begin(), one_filled.begin(), one_filled.end());
    auto desc = t_desc_.reshape(extended_dims);
    for (int i = 0; i < extended_dims.size(); i++) {
      if (extended_dims[i] == shape[i]) continue;
      ICHECK(extended_dims[i] == 1);
      ICHECK(desc.data.dims[i] == desc.data.padded_dims[i]);

      desc.data.dims[i] = shape[i];
      desc.data.padded_dims[i] = shape[i];
      desc.data.format_desc.blocking.strides[i] = 0;
    }

    // reinterpret memory buffer with new strides
    return {desc, orig, true, {}, INVALID_EID, reverse_data_flow_};
  }

  /** Produce tensor with sub memory view (ROI) */
  TensorRequisite crop(const dnnl::memory::dims& shape, const dnnl::memory::dims& offset) const {
    if (!defined()) return *this;  // nothing for empty TR

    auto orig = std::make_shared<TensorRequisite>(*this);
    // reinterpret memory buffer with new strides
    auto desc = t_desc_.submemory_desc(shape, offset);
    return {desc, orig, true, {}, INVALID_EID, reverse_data_flow_};
  }

  /** Produce tensor with squeeze shape */
  TensorRequisite squeeze(const dnnl::memory::dims& dims_to_squeeze = {}) const {
    if (!defined()) return *this;  // nothing for empty TR

    dnnl::memory::dims squeezed_dims;
    if (dims_to_squeeze.empty()) {
      for (auto d : t_desc_.dims())
        if (d != 1) squeezed_dims.push_back(d);
    } else {
      for (size_t i = 0; i < t_desc_.dims().size(); i++)
        if (std::find(dims_to_squeeze.begin(), dims_to_squeeze.end(), i) == dims_to_squeeze.end())
          squeezed_dims.push_back(t_desc_.dims()[i]);
    }

    auto orig = std::make_shared<TensorRequisite>(*this);
    // reinterpret memory buffer with new strides
    auto desc = t_desc_.reshape(squeezed_dims);
    return {desc, orig, true, {}, INVALID_EID, reverse_data_flow_};
  }

  /** Produce tensor with specified layout */
  TensorRequisite requestLayout(dnnl::memory::desc desc) const {
    if (!defined()) return *this;  // nothing for empty TR

    // If it's the same desc just return self
    if (desc == t_desc_) return *this;

    ICHECK(t_desc_.dims() == desc.dims()) << "Requested layout is not compatible with "
                                             "presented shape";

    auto orig = std::make_shared<TensorRequisite>(*this);
    return {desc, orig, false, {}, INVALID_EID, reverse_data_flow_};
  }

  /**
   * Produce tensor with unspecified layout
   * Cannot be registered in TensorRegistry. Only for querying DNNL for preferred layouts.
   */
  TensorRequisite layoutAny() const {
    auto orig = std::make_shared<TensorRequisite>(*this);
    // Recreate tensor desc with layout 'any'
    dnnl::memory::desc any_desc{t_desc_.dims(), t_desc_.data_type(), dnnl::memory::format_tag::any};
    return {any_desc, orig, false, {}, INVALID_EID, reverse_data_flow_};
  }

  /** Check is tensor is constant */
  bool isConstant() const {
    if (orig_) return orig_->isConstant();
    return mem_.operator bool();
  }

  /** Check is tensor is scalar */
  bool isScalar() const { return t_desc_.dims().size() == 1 && t_desc_.dims()[0] == 1; }

  /** Produce const data memory object with proper content */
  dnnl::memory getConstData() const {
    if (reverse_data_flow_ || eid_ != INVALID_EID) return {};
    if (mem_) return mem_;

    ICHECK(orig_);
    if (auto orig_const_data = orig_->getConstData()) {
      if (reinterpret_) {
        return {t_desc_, orig_const_data.get_engine(), orig_const_data.get_data_handle()};
      } else {
        auto res = dnnl::memory{t_desc_, orig_const_data.get_engine()};
        utils::copy_now(orig_const_data, res);
        return res;
      }
    }
    return {};
  }

  /**
   * Same as getConstData but in form of std::vector
   * Useful for 1D constant tensor like zero_point or per_channel_scale
   *
   * @tparam T desired data type
   * @return resulting data
   */
  template <typename T>
  std::vector<T> getConstDataLikeVec() const {
    auto const_data = getConstData();
    auto desc = const_data.get_desc();
    ICHECK(desc.data_type() == utils::dnnlDType<T>());
    ICHECK(desc.dims().size() == 1);

    auto size = desc.get_size() / sizeof(T);
    auto ptr = static_cast<T*>(const_data.get_data_handle());

    return std::vector<T>(ptr, ptr + size);
  }

  /**
   * Produce value of constant scalar tensor
   * @tparam T desired scalar type
   * @return resulting value of type T
   */
  template <typename T>
  T getConstScalarData() const {
    ICHECK(isConstant());
    ICHECK(isScalar());
    auto const_data = getConstData();
    auto desc = const_data.get_desc();
    ICHECK(desc.data_type() == utils::dnnlDType<T>());

    auto ptr = static_cast<T*>(const_data.get_data_handle());
    return *ptr;
  }

  /** Check if tensor is not empty */
  bool defined() const { return !t_desc_.is_zero(); }

  /** Same as defined() */
  operator bool() const { return defined(); }

  /** Check if tensor represent a reversed action queue (aka is a dst) */
  bool isReversed() const { return reverse_data_flow_; }

 private:
  TensorRequisite(const dnnl::memory::desc& t_desc, const std::shared_ptr<TensorRequisite>& orig,
                  bool reinterpret, const dnnl::memory& const_mem, uint32_t eid,
                  bool reverse_data_flow)
      : t_desc_(t_desc),
        orig_(orig),
        reinterpret_(reinterpret),
        mem_(const_mem),
        eid_(eid),
        reverse_data_flow_(reverse_data_flow) {}

  /** Descriptor of PT */
  dnnl::memory::desc t_desc_ = {};
  /** Original PT to relay in operation */
  std::shared_ptr<TensorRequisite> orig_ = {};
  /** Flag to specify reinterpret orig or do reordering */
  bool reinterpret_ = false;
  /** Const memory object if available */
  dnnl::memory mem_ = {};
  /** Entry ID of tensor if it available */
  uint32_t eid_ = INVALID_EID;

  /**
   * Flag to describe reverse data flow case
   * All operation on queue will be executed in reverse order. Actual for dst tensor description
   */
  bool reverse_data_flow_ = false;

  friend class TensorRegistry;
  friend class NodeHelper;
};

class TensorRegistry {
 private:
  enum ArgReqFlag {
    UNKNOWN,      /// < Undefined type of args. Cannot be matched to real tensor
    CONST,        /// < Constant tensor. ExecutionCTX independent
    TMP_STORAGE,  /// < Intermediate tensors. Stored inside TensorRegistry. Inaccessible outside
    EXT_EID,      /// < External data. Input or Output.
    SCRATCHPAD    /// < Scratchpad tensor. May overlap with other Scratchpad buffers.
  };

 public:
  struct ArgReq {
    TensorRegistry::ArgReqFlag flag_;
    uint32_t idx_;
  };
  using ArgReqSet = std::unordered_map<int, ArgReq>;
  using Action = std::tuple<dnnl::primitive, ArgReqSet>;
  using ActionQue = std::vector<Action>;
  using ExtDataProvider = std::function<void*(uint32_t)>;

  TensorRegistry() = default;
  TensorRegistry(const dnnl::engine& eng, const std::set<uint32_t>& ext_eid_set)
      : ext_eid_(ext_eid_set), eng_(eng) {}

  /**
   * Register a TensorRequisite
   *
   * As result corresponding ArgReq and related action which should be executed before
   * (or after in case of reverse data flow) usage of this tensor.
   * @param tr TensorRequisite to register
   * @return Associated ArgReq ar list of actions
   */
  std::pair<ArgReq, ActionQue> registerTR(const TensorRequisite& tr) {
    // 1) Constant tensor. Direct reference
    if (auto const_data = tr.getConstData()) {
      auto idx = const_mem_collection_.size();
      const_mem_collection_.push_back(const_data);
      auto arg_req = makeArgReq(ArgReqFlag::CONST, static_cast<uint32_t>(idx));
      return {arg_req, {}};
    }

    // 2) EID mapped tensor. Direct reference
    if (tr.eid_ != TensorRequisite::INVALID_EID) {
      if (isTempEID(tr.eid_)) {
        if (eid2idx_tmp_.count(tr.eid_)) {
          auto idx = eid2idx_tmp_.at(tr.eid_);
          auto arg_req = makeArgReq(ArgReqFlag::TMP_STORAGE, idx);
          return {arg_req, {}};
        } else {
          // register himself
          auto mem = dnnl::memory{tr.t_desc_, eng_};
          auto idx = tmp_mem_collection_.size();
          tmp_mem_collection_.push_back(mem);
          eid2idx_tmp_[tr.eid_] = idx;
          auto arg_req = makeArgReq(ArgReqFlag::TMP_STORAGE, static_cast<uint32_t>(idx));
          return {arg_req, {}};
        }
      } else {
        auto idx = ext_mem_collection_.size();
        ext_mem_collection_.push_back({tr.eid_, tr.t_desc_});
        auto arg_req = makeArgReq(ArgReqFlag::EXT_EID, static_cast<uint32_t>(idx));
        return {arg_req, {}};
      }
    }

    // 3) Tensors with transform actions
    if (tr.orig_) {
      ArgReq arg_req;
      ActionQue actions;

      // recursive register of orig TR
      std::tie(arg_req, actions) = registerTR(*tr.orig_);
      if (tr.reinterpret_) {
        arg_req = register_reinterp(arg_req, tr.t_desc_);
      } else {
        ActionQue reorder_act;
        std::tie(arg_req, reorder_act) =
            register_reorder(arg_req, tr.t_desc_, tr.reverse_data_flow_);

        actions.insert(tr.reverse_data_flow_ ? actions.begin() : actions.end(), reorder_act.begin(),
                       reorder_act.end());
      }
      return {arg_req, actions};
    }

    // 4) Scratchpad
    ICHECK(!tr.orig_ && !tr.mem_ && tr.eid_ == TensorRequisite::INVALID_EID);
    auto scratchpad_ar = register_scratchpad(tr.t_desc_);
    return {scratchpad_ar, {}};
  }

  std::unordered_map<int, dnnl::memory> solve(const ArgReqSet& args,
                                              const ExtDataProvider& ext_provider) const {
    std::unordered_map<int, dnnl::memory> res;
    for (const auto& kvp : args) res[kvp.first] = solve(kvp.second, ext_provider);
    return res;
  }

  /**
   * Find a proper memory object associated with provided ArgReq
   * @param ar ArgReq to
   * @param ext_provider
   * @return
   */
  dnnl::memory solve(const ArgReq& ar, const ExtDataProvider& ext_provider) const {
    switch (ar.flag_) {
      case CONST:
        return const_mem_collection_.at(ar.idx_);
      case TMP_STORAGE:
        return tmp_mem_collection_.at(ar.idx_);
      case EXT_EID: {
        auto eid_and_desc = ext_mem_collection_.at(ar.idx_);
        auto eid = eid_and_desc.first;
        auto desc = eid_and_desc.second;

        auto hdl = ext_provider(eid);
        return dnnl::memory{desc, eng_, hdl};
      }
      case SCRATCHPAD: {
        auto desc = scratchpad_desc_collection_.at(ar.idx_);
        // TODO(@apeskov): make it thread local and avoid recreation each time
        return dnnl::memory(desc, eng_);
      }
      default:
        LOG(FATAL) << "Unknown case";
    }
    return {};
  }

  /** Finalize registry. Should be called before any call of solve() method */
  void finalize() {
    // calc total scratchpad size
    dnnl::memory::dim scratchpad_size = 0;
    for (const auto& scr_desc : scratchpad_desc_collection_) {
      dnnl::memory::dim size = scr_desc.get_size();
      scratchpad_size = std::max(scratchpad_size, size);
    }
    scratchpad_mem_ = dnnl::memory::desc({scratchpad_size}, dnnl::memory::data_type::u8,
                                         dnnl::memory::format_tag::a);
  }

 private:
  ArgReq register_reinterp(ArgReq src_ar, const dnnl::memory::desc& desc) {
    switch (src_ar.flag_) {
      case CONST: {
        LOG(FATAL) << "Unreachable case";
        return {};
      }
      case TMP_STORAGE: {
        auto src = tmp_mem_collection_[src_ar.idx_];
        auto dst = dnnl::memory{desc, src.get_engine(), src.get_data_handle()};
        auto idx = tmp_mem_collection_.size();
        tmp_mem_collection_.push_back(dst);
        return makeArgReq(TMP_STORAGE, idx);
      }
      case EXT_EID: {
        auto ext_req = ext_mem_collection_[src_ar.idx_];
        auto idx = ext_mem_collection_.size();
        ext_mem_collection_.push_back({ext_req.first, desc});
        return makeArgReq(EXT_EID, idx);
      }
      default:
        LOG(FATAL) << "Unknown case";
    }
    return {};
  }

  std::pair<ArgReq, ActionQue> register_reorder(ArgReq src_ar, const dnnl::memory::desc& desc,
                                                bool reverse_data_flow) {
    switch (src_ar.flag_) {
      case CONST: {
        LOG(FATAL) << "Unreachable case";
        return {};
      }
      case TMP_STORAGE: {
        auto src = tmp_mem_collection_[src_ar.idx_];

        auto dst = dnnl::memory{desc, eng_};
        auto idx = tmp_mem_collection_.size();
        tmp_mem_collection_.push_back(dst);
        auto dst_ar = makeArgReq(TMP_STORAGE, idx);

        // Action
        Action res_action;
        if (reverse_data_flow) {
          res_action = {dnnl::reorder(dst, src), {{DNNL_ARG_FROM, dst_ar}, {DNNL_ARG_TO, src_ar}}};
        } else {
          res_action = {dnnl::reorder(src, dst), {{DNNL_ARG_FROM, src_ar}, {DNNL_ARG_TO, dst_ar}}};
        }

        return {dst_ar, {res_action}};
      }
      case EXT_EID: {
        auto src_desc = ext_mem_collection_[src_ar.idx_].second;

        auto dst = dnnl::memory{desc, eng_};
        auto idx = tmp_mem_collection_.size();
        tmp_mem_collection_.push_back(dst);
        auto dst_ar = makeArgReq(TMP_STORAGE, idx);

        // Action
        Action res_action;
        if (reverse_data_flow) {
          auto reorder_pd = dnnl::reorder::primitive_desc(eng_, desc, eng_, src_desc);
          auto reorder = dnnl::reorder(reorder_pd);
          res_action = {reorder, {{DNNL_ARG_FROM, dst_ar}, {DNNL_ARG_TO, src_ar}}};
        } else {
          auto reorder_pd = dnnl::reorder::primitive_desc(eng_, src_desc, eng_, desc);
          auto reorder = dnnl::reorder(reorder_pd);
          res_action = {reorder, {{DNNL_ARG_FROM, src_ar}, {DNNL_ARG_TO, dst_ar}}};
        }

        return {dst_ar, {res_action}};
      }
      default:
        LOG(FATAL) << "Unknown case";
    }
    return {};
  }

  ArgReq register_scratchpad(const dnnl::memory::desc& desc) {
    auto idx = scratchpad_desc_collection_.size();
    scratchpad_desc_collection_.push_back(desc);
    return makeArgReq(SCRATCHPAD, idx);
  }

  ArgReq makeArgReq(ArgReqFlag flag, uint32_t idx) { return {flag, idx}; }

  bool isTempEID(uint32_t eid) { return ext_eid_.count(eid) == 0; }

  /** Collection of const memory objects. */
  std::vector<dnnl::memory> const_mem_collection_;

  /** Collection of intermediate memory objects. */
  std::vector<dnnl::memory> tmp_mem_collection_;

  /** Map of eid to index of temp buffer in tmp_mem_collection_ */
  std::unordered_map<uint32_t, size_t> eid2idx_tmp_;

  /** Collection of external_intermediate memory objects.
   *  first  - eid of external buffer to ask
   *  second - t_desc describes how to treat external buffer */
  std::vector<std::pair<uint32_t, dnnl::memory::desc>> ext_mem_collection_;

  /** Scratchpad collection */
  std::vector<dnnl::memory::desc> scratchpad_desc_collection_;

  /** Overall scratchpad memory obj */
  dnnl::memory::desc scratchpad_mem_;

  /** List of external eid */
  std::set<uint32_t> ext_eid_;

  /** Engine of all tensors returned form this registry */
  dnnl::engine eng_;
};

/**
 * GraphExplorer is a list of fields of original JSONRuntimeBase which allows
 * to travers through the graph.
 *
 * Essentially that is a WA for access of protected fields of JSONRuntimeBase.
 */
struct GraphExplorer {
  GraphExplorer(const std::vector<json::JSONGraphNode>& nodes,
                const std::vector<const DLTensor*>& data_entry,
                const std::vector<uint32_t>& node_row_ptr, const dnnl::engine& engine)
      : nodes_(nodes),
        data_entry_(data_entry),
        node_row_ptr_(node_row_ptr),
        engine_(engine),
        gen_eid_offset(data_entry.size()) {}

  const std::vector<json::JSONGraphNode>& nodes_;
  const std::vector<const DLTensor*>& data_entry_;
  const std::vector<uint32_t>& node_row_ptr_;

  const dnnl::engine& engine_;

  uint32_t gen_eid_offset;

  uint32_t generateUniqueEID() { return gen_eid_offset++; }
};

class NodeHelper {
 public:
  NodeHelper(const uint32_t& nid, const GraphExplorer& graph_explorer)
      : nid_(nid), node_(graph_explorer.nodes_[nid]), graph_explorer_(graph_explorer) {}

  template <typename T>
  typename std::enable_if<std::is_integral<T>::value, T>::type convert(
      std::vector<std::string> val) {
    ICHECK_EQ(val.size(), 1);
    return std::stol(val[0]);
  }

  template <typename T>
  typename std::enable_if<std::is_floating_point<T>::value, T>::type convert(
      std::vector<std::string> val) {
    ICHECK_EQ(val.size(), 1);
    return std::stof(val[0]);
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, std::vector<std::string>>::value, T>::type convert(
      std::vector<std::string> val) {
    T res;
    for (const auto& el : val) res.push_back(convert<std::string>({el}));
    return res;
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, std::string>::value, T>::type convert(
      std::vector<std::string> val) {
    ICHECK_EQ(val.size(), 1);
    return val[0];
  }

  // TODO(apeskov): enhance to any vector type, not only int
  template <typename T>
  typename std::enable_if<std::is_same<T, std::vector<int>>::value, T>::type convert(
      std::vector<std::string> val) {
    T res;
    for (const auto& el : val) res.push_back(convert<int>({el}));
    return res;
  }

  template <typename T>
  const T getAttr(std::string name, std::vector<std::string> def = {}) {
    auto attr = node_.HasAttr(name) ? node_.GetAttr<std::vector<std::string>>(name) : def;
    return convert<T>(attr);
  }

  TensorRequisite getInput(int idx) {
    if (idx == -1) return {};  // unavailable input

    ICHECK_LT(idx, node_.GetInputs().size());
    auto data_entry = node_.GetInputs()[idx];

    auto shape = graph_explorer_.nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    auto dtype = graph_explorer_.nodes_[data_entry.id_].GetOpDataType()[data_entry.index_];
    auto eid = graph_explorer_.node_row_ptr_[data_entry.id_] + data_entry.index_;
    auto dl_tensor = graph_explorer_.data_entry_[eid];

    auto desc = utils::makePlainTDesc(shape, dtype);

    dnnl::memory mem = {};
    if (dl_tensor) {
      eid = TensorRequisite::INVALID_EID;
      mem = utils::convert2dnnl(dl_tensor, graph_explorer_.engine_);
      ICHECK(mem.get_desc() == desc);
    }

    return {desc, nullptr, false, mem, eid, false};
  }

  TensorRequisite getOutput(int idx) {
    ICHECK_LT(idx, node_.GetNumOutput());

    auto shape = node_.GetOpShape()[idx];
    auto dtype = node_.GetOpDataType()[idx];
    auto eid = graph_explorer_.node_row_ptr_[nid_] + static_cast<uint32_t>(idx);
    auto dl_tensor = graph_explorer_.data_entry_[eid];

    auto desc = utils::makePlainTDesc(shape, dtype);

    ICHECK(!dl_tensor) << "Output of operation node cannot be constant";
    return {desc, nullptr, true, {}, eid, true};
  }

  TensorRequisite makeTemp(const dnnl::memory::desc& desc, uint32_t eid) {
    return {desc, nullptr, false, {}, eid, true};
  }

  TensorRequisite makeScratchpad(const dnnl::memory::desc& desc) {
    return {desc, nullptr, false, {}, TensorRequisite::INVALID_EID, true};
  }

 private:
  const uint32_t nid_;
  const json::JSONGraphNode& node_;
  const GraphExplorer& graph_explorer_;
};

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_DNNL_DNNL_NODE_HELPER_H_
