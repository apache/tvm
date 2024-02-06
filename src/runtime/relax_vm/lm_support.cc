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
/*!
 * \file src/runtime/relax_vm/lm_support.cc
 * \brief Runtime to support language model related task
 *
 * Including inplace attention kv cache for runtime and simple sampler.
 *
 * This file provides a simple implementation of inplace attention
 * kv cache for relax runtime. The main goal here is to help us enable
 * auto-regressive decoding quickly in relax.
 *
 * This is not the only way to support attention kv-cache.
 * Our support of attention kv-cache can subject to future
 * changes as we build more LM verticals.
 *
 * We will keep the impact minimum by puting it as a private
 * runtime builtin provide as in this file.
 *
 * We can evolve this implementation as we build more LM verticals.
 */
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/relax_vm/vm.h>

#include <cmath>

namespace tvm {
namespace runtime {
namespace relax_vm {

//-------------------------------------------
// We keep the implementation private as
// they may subject to future changes.
//
// Users can interact with it through the
// runtime API function calls
//-------------------------------------------
/*!
 * \brief An object representing an attention kv cache.
 */
class AttentionKVCacheObj : public Object {
 public:
  /*!
   * \brief Underlying support data.
   */
  NDArray data;

  /*!
   * \brief number of slots already filled.
   */
  int64_t fill_count{0};

  /*!
   * \brief current cache position (windowed kv cache only).
   */
  int64_t window_attention_current_pos{0};

  /*!
   * \brief View all current cached values as one array.
   * \param shape The cached values.
   */
  NDArray View(const ShapeTuple& shape) {
    CHECK_EQ(shape[0], fill_count) << "Requested shape do not match the filled count";
    for (int i = 1; i < this->data->ndim; ++i) {
      CHECK_EQ(shape[i], data->shape[i]) << "Dimension " << i << " mismatch";
    }
    return data.CreateView(shape, data->dtype);
  }

  /** Clear the cache */
  void Clear() {
    this->fill_count = 0;
    this->window_attention_current_pos = 0;
  }

  /** pop n entries */
  void PopN(size_t n) {
    ICHECK_LE(n, fill_count);
    this->fill_count -= n;
  }

  void Update(NDArray value) {
    CHECK(data.DataType() == value.DataType()) << "dtype mismatch";
    CHECK_EQ(value->shape[0], fill_count) << "Requested shape do not match the filled count";
    ICHECK(data.IsContiguous());
    ICHECK(value.IsContiguous());

    DLTensor copy_dst = *(data.operator->());
    copy_dst.byte_offset = 0;
    copy_dst.shape = value->shape;
    NDArray::CopyFromTo(value.operator->(), &copy_dst);
    this->fill_count = value->shape[0];
  }

  /*!
   * \brief Append value to the cache, overrides if full.
   * \param value The value to override previous elements.
   * \param max_cache_size max size of the cache.
   * \param num_attention_sinks number of sinks to store (https://arxiv.org/abs/2309.17453).
   */
  void WindowOverride(NDArray value, int64_t max_cache_size, int64_t num_attention_sinks = 0) {
    CHECK(data.DataType() == value.DataType()) << "dtype mismatch";
    CHECK_LE(value->shape[0], max_cache_size - num_attention_sinks) << "dim 0 of value too large";
    // reallocate cache
    if (fill_count + value->shape[0] <= max_cache_size) {
      int64_t reserved_slots = data->shape[0];
      while (fill_count + value->shape[0] > reserved_slots) {
        reserved_slots *= 2;
      }
      if (reserved_slots != data->shape[0]) {
        std::vector<int64_t> new_shape(data->shape, data->shape + data->ndim);
        new_shape[0] = reserved_slots;
        NDArray new_data = NDArray::Empty(new_shape, data->dtype, data->device);
        new_data.CreateView(data.Shape(), data->dtype).CopyFrom(data);
        this->data = new_data;
      }
    }
    // copy into the current position.
    ICHECK(data.IsContiguous());

    int64_t num_elements_to_copy =
        std::min(value->shape[0], max_cache_size - window_attention_current_pos);
    int64_t num_elements_p_entry = 1;
    std::vector<int64_t> shape;
    shape.push_back(num_elements_to_copy);
    for (int i = 1; i < data->ndim; ++i) {
      CHECK_EQ(value->shape[i], data->shape[i]) << "Dimension " << i << " mismatch";
      num_elements_p_entry *= data->shape[i];
      shape.push_back(data->shape[i]);
    }
    int64_t num_filled_elements = window_attention_current_pos * num_elements_p_entry;
    this->fill_count = std::min(this->fill_count + value->shape[0], max_cache_size);
    this->window_attention_current_pos =
        std::min(this->window_attention_current_pos + value->shape[0], max_cache_size);

    if (num_elements_to_copy > 0) {
      DLTensor copy_dst = *(data.operator->());
      copy_dst.byte_offset = num_filled_elements * ((data->dtype.bits * data->dtype.lanes + 7) / 8);
      copy_dst.shape = &shape[0];

      DLTensor copy_src = *(value.operator->());
      copy_src.byte_offset = 0;
      copy_src.shape = &shape[0];

      NDArray::CopyFromTo(&copy_src, &copy_dst);
    }

    // copy the remainder to the beginning of the cache
    if (num_elements_to_copy < value->shape[0]) {
      ICHECK_EQ(this->fill_count, max_cache_size);
      ICHECK_EQ(this->fill_count, this->window_attention_current_pos);

      shape[0] = value->shape[0] - num_elements_to_copy;
      num_filled_elements = num_elements_to_copy * num_elements_p_entry;

      DLTensor copy_dst = *(data.operator->());
      copy_dst.byte_offset = (num_attention_sinks * num_elements_p_entry) *
                             ((data->dtype.bits * data->dtype.lanes + 7) / 8);
      copy_dst.shape = &shape[0];

      DLTensor copy_src = *(value.operator->());
      copy_src.byte_offset =
          num_filled_elements * ((value->dtype.bits * value->dtype.lanes + 7) / 8);
      copy_src.shape = &shape[0];

      NDArray::CopyFromTo(&copy_src, &copy_dst);
      this->window_attention_current_pos =
          value->shape[0] - num_elements_to_copy + num_attention_sinks;
    }
  }

  /*!
   * \brief Append value to the cache.
   * \param value The value to be appended.
   */
  void Append(NDArray value) {
    CHECK(data.DataType() == value.DataType()) << "dtype mismatch";
    // reallocate cache
    int64_t reserved_slots = data->shape[0];
    while (fill_count + value->shape[0] > reserved_slots) {
      reserved_slots *= 2;
    }
    if (reserved_slots != data->shape[0]) {
      std::vector<int64_t> new_shape(data->shape, data->shape + data->ndim);
      new_shape[0] = reserved_slots;
      NDArray new_data = NDArray::Empty(new_shape, data->dtype, data->device);
      new_data.CreateView(data.Shape(), data->dtype).CopyFrom(data);
      this->data = new_data;
    }
    // copy into the fill count position.
    ICHECK_LE(fill_count + value->shape[0], data->shape[0]);
    ICHECK(data.IsContiguous());

    int64_t num_filled_elements = fill_count;
    for (int i = 1; i < data->ndim; ++i) {
      CHECK_EQ(value->shape[i], data->shape[i]) << "Dimension " << i << " mismatch";
      num_filled_elements *= data->shape[i];
    }
    // create a view of copy dest to copy the value into it.
    DLTensor copy_dst = *(data.operator->());
    copy_dst.byte_offset = num_filled_elements * ((data->dtype.bits * data->dtype.lanes + 7) / 8);
    copy_dst.shape = value->shape;
    NDArray::CopyFromTo(value.operator->(), &copy_dst);
    this->fill_count += value->shape[0];
  }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.vm.AttentionKVCache";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttentionKVCacheObj, Object);
};

/*! \brief reference to closure. */
class AttentionKVCache : public ObjectRef {
 public:
  /*!
   * \brief Create the attention kv cache.
   * \param init_data The initial reserved.
   */
  static AttentionKVCache Create(NDArray init_data, ShapeTuple reserve_shape, int init_fill_count) {
    auto n = make_object<AttentionKVCacheObj>();
    n->data = NDArray::Empty(reserve_shape, init_data->dtype, init_data->device);
    n->fill_count = 0;
    n->Append(init_data);
    if (init_fill_count >= 0) {
      n->fill_count = init_fill_count;
      n->window_attention_current_pos = init_fill_count;  // window attention only
    }
    return AttentionKVCache(n);
  }

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(AttentionKVCache, ObjectRef, AttentionKVCacheObj);
};

TVM_REGISTER_OBJECT_TYPE(AttentionKVCacheObj);

//-------------------------------------------------
//  Register runtime functions
//-------------------------------------------------
TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_create")
    .set_body_typed(AttentionKVCache::Create);

AttentionKVCache AttentionKVCacheUpdate(AttentionKVCache cache, NDArray value) {
  cache->Update(value);
  return cache;
}

TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_update").set_body_typed(AttentionKVCacheUpdate);

AttentionKVCache AttentionKVCacheAppend(AttentionKVCache cache, NDArray value) {
  cache->Append(value);
  return cache;
}

TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_append").set_body_typed(AttentionKVCacheAppend);

AttentionKVCache AttentionKVCacheWindowOverride(AttentionKVCache cache, NDArray value,
                                                int64_t max_cache_size) {
  cache->WindowOverride(value, max_cache_size);
  return cache;
}

TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_window_override")
    .set_body_typed(AttentionKVCacheWindowOverride);

AttentionKVCache AttentionKVCacheWindowOverrideWithSinks(AttentionKVCache cache, NDArray value,
                                                         int64_t max_cache_size,
                                                         int64_t num_attention_sinks) {
  cache->WindowOverride(value, max_cache_size, num_attention_sinks);
  return cache;
}

TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_window_override_with_sinks")
    .set_body_typed(AttentionKVCacheWindowOverrideWithSinks);

NDArray AttentionKVCacheView(AttentionKVCache cache, ShapeTuple shape) {
  return cache->View(shape);
}

TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_view")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      CHECK(args.size() == 1 || args.size() == 2)
          << "ValueError: `vm.builtin.attention_kv_cache_view` expects 1 or 2 arguments, but got "
          << args.size() << ".";
      AttentionKVCache cache = args[0];
      if (args.size() == 2) {
        ShapeTuple shape = args[1];
        *rv = cache->View(shape);
      } else {
        std::vector<ShapeTuple::index_type> shape;
        shape.push_back(cache->fill_count);
        for (int i = 1; i < cache->data->ndim; ++i) {
          shape.push_back(cache->data->shape[i]);
        }
        *rv = cache->View(ShapeTuple(shape));
      }
    });

void AttentionKVCacheArrayPopN(Array<AttentionKVCache> caches, int64_t n) {
  for (AttentionKVCache cache : caches) {
    cache->PopN(static_cast<size_t>(n));
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_array_popn")
    .set_body_typed(AttentionKVCacheArrayPopN);

void AttentionKVCacheArrayClear(Array<AttentionKVCache> caches) {
  for (AttentionKVCache cache : caches) {
    cache->Clear();
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_array_clear")
    .set_body_typed(AttentionKVCacheArrayClear);

// NOTE this is a built-in highly related to LM so we put it here.
int SampleTopPFromLogits(NDArray logits, double temperature, double top_p, double uniform_sample) {
  ICHECK(logits.IsContiguous());
  ICHECK(logits.DataType() == DataType::Float(32));

  if (logits->device.device_type != kDLCPU) {
    logits = logits.CopyTo(DLDevice{kDLCPU, 0});
  }

  ICHECK(logits->device.device_type == kDLCPU);

  for (int i = 0; i < logits->ndim - 1; ++i) {
    ICHECK_EQ(logits->shape[i], 1) << "The leading dimensions of logits must be 1";
  }

  std::vector<std::pair<float, int>> data;
  data.resize(logits->shape[logits->ndim - 1]);
  const float* plogits = static_cast<float*>(logits->data);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = std::make_pair(plogits[i], static_cast<int>(i));
  }

  auto fcmp = [](const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) {
    return lhs.first > rhs.first;
  };
  // sort by logits from largest to smallest
  std::sort(data.begin(), data.end(), fcmp);

  // argmax
  if (temperature < 1e-6f) {
    return data[0].second;
  }

  // compute expf scaled by temp
  float sum = 0.0f, logit_scale = 1.0f / temperature;
  float max_value = data[0].first;
  for (auto it = data.begin(); it != data.end(); ++it) {
    it->first = expf((it->first - max_value) * logit_scale);
    sum += it->first;
  }

  // do a cumsum in order of data
  float cum_sum_prob = 0.0f;
  float top_p_sum = 0.0f;
  for (auto it = data.begin(); it != data.end(); ++it) {
    float prob = it->first / sum;
    if (cum_sum_prob < top_p) {
      top_p_sum += prob;
    }
    cum_sum_prob += prob;
    it->first = cum_sum_prob;
  }
  // pick a number based on random in (0, 1)
  for (auto it = data.begin(); it != data.end(); ++it) {
    if (uniform_sample < it->first / top_p_sum) {
      return it->second;
    }
  }
  ICHECK_LE(uniform_sample, data[0].first);
  return data[0].second;
}

TVM_REGISTER_GLOBAL("vm.builtin.sample_top_p_from_logits").set_body_typed(SampleTopPFromLogits);

int SampleTopPFromProb(NDArray prob, double top_p, double uniform_sample) {
  ICHECK(prob.IsContiguous());
  ICHECK(prob.DataType() == DataType::Float(32));

  if (prob->device.device_type != kDLCPU) {
    prob = prob.CopyTo(DLDevice{kDLCPU, 0});
  }

  ICHECK(prob->device.device_type == kDLCPU);

  for (int i = 0; i < prob->ndim - 1; ++i) {
    ICHECK_EQ(prob->shape[i], 1) << "The leading dimensions of logits must be 1";
  }

  // Key observation: when we are doing top_p sampling
  // usually we only need to preserve some of the elements with
  // high probablities before we do sort
  std::vector<std::pair<float, int>> data;
  int64_t ndata = prob->shape[prob->ndim - 1];
  const float* p_prob = static_cast<float*>(prob->data);

  auto sample_top_p_with_filter = [&](float cuttoff) -> int64_t {
    data.clear();
    // filter the data with cuttoff
    for (int64_t i = 0; i < ndata; ++i) {
      if (p_prob[i] >= cuttoff) {
        data.emplace_back(std::make_pair(p_prob[i], static_cast<int>(i)));
      }
    }
    if (data.size() == 0) return -1;
    auto fcmp = [](const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) {
      return lhs.first > rhs.first;
    };
    std::sort(data.begin(), data.end(), fcmp);

    // short cut, if we know that
    // uniform sample < p[0] / top_p
    // we know that unform_sample < p[0] / top_p_sum
    // because top_p_sum gaurantees to be smaller than top_p
    // so we can simply return the argmax sample
    // without computing anything
    if (uniform_sample < data[0].first / top_p) return data[0].second;

    // compute top_p_sum
    float cum_sum_prob = 0.0f;
    float top_p_sum = 0.0f;
    for (auto it = data.begin(); it != data.end(); ++it) {
      float prob = it->first;
      if (cum_sum_prob < top_p) {
        top_p_sum += prob;
      } else {
        // we get to the right cutoff pt
        break;
      }
      cum_sum_prob += prob;
      it->first = cum_sum_prob;
    }
    // we find that the current total sum by the given cutoff
    // is not sufficient to cover everything
    // this means we might need to retry a smaller cutoff pt.
    if (cum_sum_prob < top_p && cuttoff != 0.0f) return -1;

    for (auto it = data.begin(); it != data.end(); ++it) {
      if (uniform_sample < it->first / top_p_sum) {
        return it->second;
      }
    }
    return data[data.size() - 1].second;
  };

  auto is_all_nan = [&]() -> bool {
    return std::all_of(p_prob, p_prob + ndata, [](float x) { return std::isnan(x); });
  };

  if (top_p < 1) {
    // sample through cutoff by a number
    // by pigeonhole principle we will get at most 1024 elements
    // usually it is much less by applying this filtering(order of 10 - 20)
    data.reserve(128);
    int64_t sampled_index = sample_top_p_with_filter(top_p / 1024);
    if (sampled_index >= 0) return sampled_index;
  }
  // fallback via full prob, rare case
  data.reserve(ndata);
  int64_t sampled_index = sample_top_p_with_filter(0.0f);
  if (sampled_index < 0 && is_all_nan()) {
    LOG(FATAL) << "The output probabilities are all NaNs, can not sample from it";
  } else if (sampled_index < 0) {
    LOG(FATAL) << "Cannot sample from the given probability distribution due to unknown reason";
  }
  return sampled_index;
}

TVM_REGISTER_GLOBAL("vm.builtin.sample_top_p_from_prob").set_body_typed(SampleTopPFromProb);

// This is an inplace operation.
void ApplyRepetitionPenalty(NDArray logits, NDArray token_ids, double penalty) {
  ICHECK(logits.IsContiguous());
  ICHECK(token_ids.IsContiguous());
  ICHECK(logits.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
  ICHECK(token_ids.DataType() == DataType::Int(32)) << "token ids must be int32!";
  ICHECK(logits->device.device_type == kDLCPU) << "logits device must be CPU!";
  ICHECK(token_ids->device.device_type == kDLCPU) << "token_ids device must be CPU!";
  float* logits_raw_data = static_cast<float*>(logits->data);
  int* token_ids_data = static_cast<int*>(token_ids->data);
  size_t num_token_ids = token_ids->shape[token_ids->ndim - 1];
  for (size_t i = 0; i < num_token_ids; ++i) {
    int token_id = token_ids_data[i];
    if (logits_raw_data[token_id] <= 0) {
      logits_raw_data[token_id] *= penalty;
    } else {  // logits > 0
      logits_raw_data[token_id] /= penalty;
    }
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.apply_repetition_penalty").set_body_typed(ApplyRepetitionPenalty);

/*!
 * \brief Apply presence and frequency penalty. This is an inplace operation.
 * \param logits The input logits before penalty.
 * \param token_ids The appeared token ids.
 * \param token_freqs The number of times each token has appeared since last PrefillStep.
 * token_freqs[i] is the frequency of token_ids[i], for all i. And all token_freqs should be >= 1.
 * \param presence_penalty The penalty factor, applied if a token appeared in an one-off manner.
 * \param frequency_penalty The penalty factor, contributes more the more frequent a token appears.
 */
void ApplyPresenceAndFrequencyPenalty(NDArray logits, NDArray token_ids, NDArray token_freqs,
                                      double presence_penalty, double frequency_penalty) {
  // See https://platform.openai.com/docs/guides/text-generation/frequency-and-presence-penalties
  ICHECK(logits.IsContiguous());
  ICHECK(token_ids.IsContiguous());
  ICHECK(token_freqs.IsContiguous());
  ICHECK(logits.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
  ICHECK(token_ids.DataType() == DataType::Int(32)) << "token ids must be int32!";
  ICHECK(token_freqs.DataType() == DataType::Int(32)) << "token freqs must be int32!";
  ICHECK(logits->device.device_type == kDLCPU) << "logits device must be CPU!";
  ICHECK(token_ids->device.device_type == kDLCPU) << "token_ids device must be CPU!";
  ICHECK(token_freqs->device.device_type == kDLCPU) << "token_ids device must be CPU!";

  float* logits_raw_data = static_cast<float*>(logits->data);
  int* token_ids_data = static_cast<int*>(token_ids->data);
  int* token_freqs_data = static_cast<int*>(token_freqs->data);
  size_t num_token_ids = token_ids->shape[token_ids->ndim - 1];
  for (size_t i = 0; i < num_token_ids; ++i) {
    int token_id = token_ids_data[i];
    int token_freq = token_freqs_data[i];
    logits_raw_data[token_id] -= (token_freq * frequency_penalty + presence_penalty);
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.apply_presence_and_frequency_penalty")
    .set_body_typed(ApplyPresenceAndFrequencyPenalty);

// This is an inplace operation.
void ApplySoftmaxWithTemperature(NDArray logits, double temperature) {
  ICHECK(logits.IsContiguous());
  ICHECK(logits.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
  ICHECK(logits->device.device_type == kDLCPU) << "logits device must be CPU!";
  int vocab_size = logits->shape[logits->ndim - 1];
  float* logits_raw_data = static_cast<float*>(logits->data);
  float inv_temp = 1.0f / temperature;
  float m = std::numeric_limits<float>::min();
  double d = 0.0f;
  for (int i = 0; i < vocab_size; ++i) {
    float x = logits_raw_data[i] * inv_temp;
    float m_prev = m;
    m = std::max(m, x);
    d = d * std::exp(m_prev - m) + std::exp(x - m);
  }
  for (int i = 0; i < vocab_size; ++i) {
    float x = logits_raw_data[i] * inv_temp;
    logits_raw_data[i] = std::exp(x - m) / d;
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.apply_softmax_with_temperature")
    .set_body_typed(ApplySoftmaxWithTemperature);

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
