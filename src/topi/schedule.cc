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
 * \brief Registration of TVM schedules
 * \file schedule.cc
 */

#include <tvm/ir/expr.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/generic_func.h>
#include <tvm/topi/cuda/dense.h>
#include <tvm/topi/cuda/injective.h>
#include <tvm/topi/cuda/pooling.h>
#include <tvm/topi/cuda/reduction.h>
#include <tvm/topi/cuda/softmax.h>
#include <tvm/topi/detail/tensor_utils.h>
#include <tvm/topi/generic/default.h>
#include <tvm/topi/generic/extern.h>
#include <tvm/topi/generic/injective.h>
#include <tvm/topi/rocm/dense.h>
#include <tvm/topi/rocm/injective.h>
#include <tvm/topi/rocm/pooling.h>
#include <tvm/topi/rocm/reduction.h>
#include <tvm/topi/rocm/softmax.h>
#include <tvm/topi/x86/bnn.h>
#include <tvm/topi/x86/default.h>
#include <tvm/topi/x86/injective.h>

namespace tvm {
namespace topi {

using namespace tvm;
using namespace tvm::runtime;

TVM_REGISTER_GLOBAL("topi.TEST_create_target").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = tvm::Target(args[0].operator String());
});

/* Generic schedules */
TVM_REGISTER_GLOBAL("topi.generic.default_schedule").set_body([](TVMArgs args, TVMRetValue* rv) {
  if (args[2]) {
    *rv = topi::generic::default_schedule_auto_inline(args[0], args[1]);
  } else {
    *rv = topi::generic::default_schedule(args[0], args[1]);
  }
});

TVM_REGISTER_GLOBAL("topi.generic.schedule_extern").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::generic::schedule_extern(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.generic.schedule_injective").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::generic::schedule_injective(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.generic.schedule_injective_from_existing")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = topi::generic::schedule_injective_from_existing(args[0], args[1]);
    });

/* x86 schedules */
TVM_REGISTER_GLOBAL("topi.x86.schedule_binarize_pack").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::x86::schedule_binarize_pack(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.x86.schedule_binary_dense").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::x86::schedule_binary_dense(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.x86.default_schedule").set_body([](TVMArgs args, TVMRetValue* rv) {
  if (args[2]) {
    *rv = topi::x86::default_schedule_auto_inline(args[0], args[1]);
  } else {
    *rv = topi::x86::default_schedule(args[0], args[1]);
  }
});

TVM_REGISTER_GLOBAL("topi.x86.schedule_injective").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::x86::schedule_injective(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.x86.schedule_injective_from_existing")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = topi::x86::schedule_injective_from_existing(args[0], args[1]);
    });

/* ROCm schedules */
TVM_REGISTER_GLOBAL("topi.rocm.dense_cuda").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = rocm::dense_rocm(args[0], args[1], args[2], args[3], args[4]);
});

TVM_REGISTER_GLOBAL("topi.rocm.schedule_dense").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::rocm::schedule_dense(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.rocm.schedule_injective").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::rocm::schedule_injective(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.rocm.schedule_injective_from_existing")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = topi::rocm::schedule_injective_from_existing(args[0], args[1]);
    });

TVM_REGISTER_GLOBAL("topi.rocm.schedule_pool").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::rocm::schedule_pool(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.rocm.schedule_global_pool").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::rocm::schedule_global_pool(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.rocm.schedule_reduce").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::rocm::schedule_reduce(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.rocm.schedule_softmax").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::rocm::schedule_softmax(args[0], args[1]);
});

/* CUDA schedules */
TVM_REGISTER_GLOBAL("topi.cuda.dense_cuda").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = cuda::dense_cuda(args[0], args[1], args[2], args[3], args[4]);
});

TVM_REGISTER_GLOBAL("topi.cuda.schedule_dense").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::cuda::schedule_dense(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.cuda.schedule_injective").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::cuda::schedule_injective(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.cuda.schedule_injective_from_existing")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = topi::cuda::schedule_injective_from_existing(args[0], args[1]);
    });

TVM_REGISTER_GLOBAL("topi.cuda.schedule_pool").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::cuda::schedule_pool(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.cuda.schedule_global_pool").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::cuda::schedule_global_pool(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.cuda.schedule_reduce").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::cuda::schedule_reduce(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.cuda.schedule_softmax").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::cuda::schedule_softmax(args[0], args[1]);
});

/* Utility functions */
TVM_REGISTER_GLOBAL("topi.utils.is_empty_shape").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = topi::detail::is_empty_shape(args[0]);
});

TVM_REGISTER_GLOBAL("topi.utils.bilinear_sample_nchw").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = detail::bilinear_sample_nchw(args[0], args[1], args[2], args[3]);
});

TVM_REGISTER_GLOBAL("topi.utils.bilinear_sample_nhwc").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = detail::bilinear_sample_nhwc(args[0], args[1], args[2], args[3]);
});

/*! \brief Builder function for instantiating schedules. */
using FTVMScheduleBuilder = std::function<tvm::te::Schedule(
    const tvm::Target& target, const tvm::Array<tvm::te::Tensor>& outs)>;

/*!
 * \brief Helper function for registering generic functions matching the
 * FTVMScheduleBuilder signature. The schedule builder function is wrapped
 * with a PackedFunc suitable for passing to a tvm::GenericFunc.
 *
 * \param builder The schedule builder to wrap.
 *
 * \return The wrapped schedule builder
 */
inline PackedFunc WrapSchedule(FTVMScheduleBuilder builder) {
  return PackedFunc([builder](TVMArgs args, TVMRetValue* ret) {
    auto target = Target::Current(false);
    Array<Tensor> outs;
    ObjectRef argNodeRef = args[0];
    if (argNodeRef->type_index() == outs->type_index()) {
      outs = args[0];
    } else {
      outs = Array<Tensor>{args[0]};
    }

    *ret = builder(target, outs);
  });
}

TVM_REGISTER_GENERIC_FUNC(schedule_injective)
    .set_default(WrapSchedule(topi::generic::schedule_injective))
    .register_func({"cpu"}, WrapSchedule(topi::x86::schedule_injective))
    .register_func({"cuda", "gpu"}, WrapSchedule(topi::cuda::schedule_injective));

TVM_REGISTER_GENERIC_FUNC(schedule_softmax)
    .set_default(WrapSchedule(topi::generic::default_schedule))
    .register_func({"cpu"}, WrapSchedule(topi::x86::default_schedule))
    .register_func({"cuda", "gpu"}, WrapSchedule(topi::cuda::schedule_softmax));

TVM_REGISTER_GENERIC_FUNC(schedule_dense)
    .set_default(WrapSchedule(topi::generic::default_schedule))
    .register_func({"cuda", "gpu"}, WrapSchedule(topi::cuda::schedule_dense))
    .register_func({"rocm"}, WrapSchedule(topi::rocm::schedule_dense));

TVM_REGISTER_GENERIC_FUNC(schedule_batch_matmul)
    .set_default(WrapSchedule(topi::generic::default_schedule));

TVM_REGISTER_GENERIC_FUNC(schedule_batch_norm)
    .set_default(WrapSchedule(topi::generic::default_schedule));

TVM_REGISTER_GENERIC_FUNC(schedule_pool)
    .set_default(WrapSchedule(topi::generic::default_schedule))
    .register_func({"cpu"}, WrapSchedule(topi::x86::default_schedule))
    .register_func({"cuda", "gpu"}, WrapSchedule(topi::cuda::schedule_pool));

TVM_REGISTER_GENERIC_FUNC(schedule_global_pool)
    .set_default(WrapSchedule(topi::generic::default_schedule))
    .register_func({"cpu"}, WrapSchedule(topi::x86::default_schedule))
    .register_func({"cuda", "gpu"}, WrapSchedule(topi::cuda::schedule_global_pool));

TVM_REGISTER_GENERIC_FUNC(schedule_reduce)
    .set_default(WrapSchedule(topi::generic::default_schedule_auto_inline))
    .register_func({"cpu"}, WrapSchedule(topi::x86::default_schedule_auto_inline))
    .register_func({"cuda", "gpu"}, WrapSchedule(topi::cuda::schedule_reduce));

TVM_REGISTER_GENERIC_FUNC(schedule_binarize_pack)
    .set_default(WrapSchedule(topi::generic::default_schedule))
    .register_func({"cpu"}, WrapSchedule(topi::x86::schedule_binarize_pack));

TVM_REGISTER_GENERIC_FUNC(schedule_binary_dense)
    .set_default(WrapSchedule(topi::generic::default_schedule))
    .register_func({"cpu"}, WrapSchedule(topi::x86::schedule_binary_dense));

/*! \brief Builder function for instantiating schedules from existing schedules. */
using FTVMScheduleFromExistingBuilder =
    std::function<tvm::te::Schedule(tvm::te::Schedule sch, const tvm::te::Tensor& out)>;

/*!
 * \brief Helper function for registering generic functions matching the
 * FTVMScheduleFromExistingBuilder signature. The schedule builder function is wrapped
 * with a PackedFunc suitable for passing to a tvm::GenericFunc.
 *
 * \param builder The schedule builder to wrap.
 *
 * \return The wrapped schedule builder
 */
inline PackedFunc WrapScheduleFromExisting(FTVMScheduleFromExistingBuilder builder) {
  return PackedFunc(
      [builder](TVMArgs args, TVMRetValue* ret) { *ret = builder(args[0], args[1]); });
}

TVM_REGISTER_GENERIC_FUNC(schedule_injective_from_existing)
    .set_default(WrapScheduleFromExisting(topi::generic::schedule_injective_from_existing))
    .register_func({"cpu"}, WrapScheduleFromExisting(topi::x86::schedule_injective_from_existing))
    .register_func({"cuda", "gpu"},
                   WrapScheduleFromExisting(topi::cuda::schedule_injective_from_existing));

/*! \brief Builder function for instantiating dense ops. */
using FTVMDenseOpBuilder = std::function<tvm::te::Tensor(
    const Target& target, const tvm::te::Tensor& data, const tvm::te::Tensor& weight,
    const tvm::te::Tensor& bias, const DataType& out_dtype)>;

/*!
 * \brief Helper function for registering dense ops matching the
 * FTVMDenseOpBuilder signature. The op builder function is wrapped
 * with a PackedFunc suitable for passing to a tvm::GenericFunc.
 *
 * \param builder The op builder to wrap.
 *
 * \return The wrapped op builder
 */
inline PackedFunc WrapDenseOp(FTVMDenseOpBuilder builder) {
  return PackedFunc([builder](TVMArgs args, TVMRetValue* ret) {
    auto target = Target::Current(false);
    Tensor data = args[0];
    Tensor weight = args[1];
    Tensor bias = args[2];
    DataType out_dtype = args[3];

    *ret = builder(target, data, weight, bias, out_dtype);
  });
}

TVM_REGISTER_GENERIC_FUNC(dense)
    .set_default(WrapDenseOp([](const Target& target, const tvm::te::Tensor& data,
                                const tvm::te::Tensor& weight, const tvm::te::Tensor& bias,
                                const DataType& out_dtype) {
      return topi::nn::dense(data, weight, bias, out_dtype);
    }))
    .register_func({"cuda", "gpu"}, WrapDenseOp(topi::cuda::dense_cuda))
    .register_func({"rocm"}, WrapDenseOp(topi::rocm::dense_rocm));

}  // namespace topi
}  // namespace tvm
