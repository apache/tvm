/*!
*  Copyright (c) 2017 by Contributors
* \brief Registration of TVM operators and schedules
* \file topi.cc
*/
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/packed_func_ext.h>
#include <tvm/build_module.h>

#include <topi/broadcast.h>
#include <topi/elemwise.h>
#include <topi/nn.h>
#include <topi/reduction.h>
#include <topi/transform.h>

#include <topi/nn/batch_norm.h>
#include <topi/nn/bnn.h>
#include <topi/nn/dense.h>
#include <topi/nn/dilate.h>
#include <topi/nn/flatten.h>
#include <topi/nn/mapping.h>
#include <topi/nn/pooling.h>
#include <topi/nn/softmax.h>

#include <topi/generic/default.h>
#include <topi/generic/extern.h>
#include <topi/generic/injective.h>

#include <topi/cuda/dense.h>
#include <topi/cuda/extern.h>
#include <topi/cuda/injective.h>
#include <topi/cuda/pooling.h>
#include <topi/cuda/reduction.h>
#include <topi/cuda/softmax.h>

#include <topi/x86/bnn.h>
#include <topi/x86/default.h>
#include <topi/x86/injective.h>

#include <topi/rocm/dense.h>

namespace tvm {
namespace runtime {
template<>
struct extension_class_info<tvm::Target> {
  static const int code = 28;
};
}  // namespace tvm
} // namespace runtime

namespace topi {

using namespace tvm;
using namespace tvm::runtime;

TVM_REGISTER_EXT_TYPE(tvm::Target);

/*! \brief Canonicalize an argument that may be Array<Expr> or int to Array<Expr> */
Array<Expr> ArrayOrInt(TVMArgValue arg) {
  if (arg.type_code() == kDLInt || arg.type_code() == kDLUInt) {
    Array<Expr> result;
    result.push_back(arg.operator int());
    return result;
  } else {
    return arg;
  }
}

TVM_REGISTER_GLOBAL("topi.TEST_create_target")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = tvm::Target::create(args[0]);
  });

/* Ops from broadcast.h */
TVM_REGISTER_GLOBAL("topi.broadcast_to")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = broadcast_to(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.broadcast_add")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = broadcast_add(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.broadcast_sub")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = broadcast_sub(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.broadcast_mul")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = broadcast_mul(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.broadcast_div")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = broadcast_div(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.broadcast_maximum")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = broadcast_maximum(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.broadcast_minimum")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = broadcast_minimum(args[0], args[1]);
   });

TVM_REGISTER_GLOBAL("topi.broadcast_pow")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = broadcast_pow(args[0], args[1]);
  });

/* Ops from elemwise.h */
TVM_REGISTER_GLOBAL("topi.exp")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = exp(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.tanh")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = tanh(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.sigmoid")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = sigmoid(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.sqrt")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = sqrt(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.log")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = log(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.identity")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = identity(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.negative")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = negative(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.pow")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = pow(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.left_shift")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  Tensor lhs = args[0];
  Expr rhs = args[1];
  *rv = lhs >> rhs;
  });

TVM_REGISTER_GLOBAL("topi.right_shift")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  Tensor lhs = args[0];
  Expr rhs = args[1];
  *rv = lhs << rhs;
  });

TVM_REGISTER_GLOBAL("topi.clip")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = clip(args[0], args[1], args[2]);
  });

TVM_REGISTER_GLOBAL("topi.cast")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = cast(args[0], args[1]);
  });

/* Ops from nn.h */
TVM_REGISTER_GLOBAL("topi.nn.relu")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = relu<float>(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.nn.leaky_relu")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = leaky_relu<float>(args[0]);
  });

TVM_REGISTER_GLOBAL("topi.nn.prelu")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = prelu<float>(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.nn.pad")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = pad(args[0], args[1], args[2], args[3]);
  });

/* Ops from reduction.h */
TVM_REGISTER_GLOBAL("topi.sum")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = topi::sum(args[0], ArrayOrInt(args[1]), args[2]);
  });

TVM_REGISTER_GLOBAL("topi.min")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = topi::min(args[0], ArrayOrInt(args[1]), args[2]);
  });

TVM_REGISTER_GLOBAL("topi.max")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = topi::max(args[0], ArrayOrInt(args[1]), args[2]);
  });

TVM_REGISTER_GLOBAL("topi.argmin")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = topi::argmin(args[0], ArrayOrInt(args[1]), args[2]);
  });

TVM_REGISTER_GLOBAL("topi.argmax")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = topi::argmax(args[0], ArrayOrInt(args[1]), args[2]);
  });

/* Ops from transform.h */
TVM_REGISTER_GLOBAL("topi.expand_dims")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = expand_dims(args[0], args[1], args[2]);
  });

TVM_REGISTER_GLOBAL("topi.transpose")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = transpose(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.reshape")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = reshape(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.squeeze")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = squeeze(args[0], ArrayOrInt(args[1]));
  });

TVM_REGISTER_GLOBAL("topi.concatenate")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = concatenate(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.split")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  if (args[1].type_code() == kDLInt || args[1].type_code() == kDLUInt) {
    *rv = split_sections(args[0], args[1], args[2]);
  } else {
    *rv = split(args[0], args[1], args[2]);
  }
  });

/* Ops from nn/batch_norm.h */
TVM_REGISTER_GLOBAL("topi.nn.batch_norm_inference")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = nn::batch_norm_inference(args[0],
                                 args[1],
                                 args[2],
                                 args[3],
                                 args[4],
                                 static_cast<double>(args[5]),
                                 args[6]);
  });

/* Ops from nn/bnn.h */
TVM_REGISTER_GLOBAL("topi.nn.binarize_pack")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = nn::binarize_pack(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.nn.binary_dense")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = nn::binary_dense(args[0], args[1]);
  });

/* Ops from nn/dense.h */
TVM_REGISTER_GLOBAL("topi.nn.dense")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = nn::dense(args[0], args[1], args[2]);
  });

/* Ops from nn/dilate.h */
TVM_REGISTER_GLOBAL("topi.nn.dilate")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = nn::dilate(args[0], args[1]);
  });

/* Ops from nn/flatten.h */
TVM_REGISTER_GLOBAL("topi.nn.flatten")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = nn::flatten(args[0]);
  });

/* Ops from nn/mapping.h */
TVM_REGISTER_GLOBAL("topi.nn.scale_shift_nchw")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = nn::scale_shift_nchw(args[0], args[1], args[2]);
  });

TVM_REGISTER_GLOBAL("topi.nn.scale_shift_nhwc")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = nn::scale_shift_nhwc(args[0], args[1], args[2]);
  });

/* Ops from nn/pooling.h */
TVM_REGISTER_GLOBAL("topi.nn.pool")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = nn::pool(args[0], args[1], args[2], args[3],
                 static_cast<nn::PoolType>(static_cast<int>(args[4])),
                 args[5], args[6]);
  });

TVM_REGISTER_GLOBAL("topi.nn.global_pool")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = nn::global_pool(args[0],
                        static_cast<nn::PoolType>(static_cast<int>(args[1])));
  });

/* Ops from nn/softmax.h */
TVM_REGISTER_GLOBAL("topi.nn.softmax")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = nn::softmax(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.nn.log_softmax")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = nn::log_softmax(args[0]);
  });

/* Generic schedules */
TVM_REGISTER_GLOBAL("topi.generic.default_schedule")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  if (args[2]) {
    *rv = topi::generic::default_schedule_auto_inline(args[0], args[1]);
  } else {
    *rv = topi::generic::default_schedule(args[0], args[1]);
  }
  });

TVM_REGISTER_GLOBAL("topi.generic.schedule_extern")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = topi::generic::schedule_extern(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.generic.schedule_injective")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = topi::generic::schedule_injective(args[0], args[1]);
  });

/* x86 schedules */
TVM_REGISTER_GLOBAL("topi.x86.schedule_binarize_pack")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = topi::x86::schedule_binarize_pack(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.x86.schedule_binary_dense")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = topi::x86::schedule_binary_dense(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.x86.default_schedule")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  if (args[2]) {
    *rv = topi::x86::default_schedule_auto_inline(args[0], args[1]);
  } else {
    *rv = topi::x86::default_schedule(args[0], args[1]);
  }
  });

TVM_REGISTER_GLOBAL("topi.x86.schedule_injective")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = topi::x86::schedule_injective(args[0], args[1]);
  });

/* ROCm schedules */
TVM_REGISTER_GLOBAL("topi.rocm.dense_cuda")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = rocm::dense_rocm(args[0], args[1], args[2], args[3]);
  });

TVM_REGISTER_GLOBAL("topi.rocm.schedule_dense")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = topi::rocm::schedule_dense(args[0], args[1]);
  });

/* CUDA schedules */
TVM_REGISTER_GLOBAL("topi.cuda.dense_cuda")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = cuda::dense_cuda(args[0], args[1], args[2], args[3]);
  });

TVM_REGISTER_GLOBAL("topi.cuda.schedule_dense")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = topi::cuda::schedule_dense(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.cuda.schedule_extern")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = topi::cuda::schedule_extern(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.cuda.schedule_injective")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = topi::cuda::schedule_injective(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.cuda.schedule_pool")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = topi::cuda::schedule_pool(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.cuda.schedule_global_pool")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = topi::cuda::schedule_global_pool(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.cuda.schedule_reduce")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = topi::cuda::schedule_reduce(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.cuda.schedule_softmax")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = topi::cuda::schedule_softmax(args[0], args[1]);
  });

/*! \brief Builder function for instantiating schedules. */
using FTVMScheduleBuilder = std::function<
  tvm::Schedule(const tvm::Target& target, const tvm::Array<tvm::Tensor>& outs)>;

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
    auto target = Target::current_target(false);
    Array<Tensor> outs;
    NodeRef argNodeRef = args[0];
    if (argNodeRef->type_index() == outs->type_index()) {
      outs = args[0];
    } else {
      outs = Array<Tensor> { args[0] };
    }

    *ret = builder(target, outs);
  });
}

TVM_REGISTER_GENERIC_FUNC(schedule_injective)
.set_default(WrapSchedule(topi::generic::schedule_injective))
.register_func({ "cpu" }, WrapSchedule(topi::x86::schedule_injective))
.register_func({ "cuda", "gpu" }, WrapSchedule(topi::cuda::schedule_injective));

TVM_REGISTER_GENERIC_FUNC(schedule_softmax)
.set_default(WrapSchedule(topi::generic::default_schedule))
.register_func({ "cpu" }, WrapSchedule(topi::x86::default_schedule))
.register_func({ "cuda", "gpu" }, WrapSchedule(topi::cuda::schedule_softmax));

TVM_REGISTER_GENERIC_FUNC(schedule_dense)
.set_default(WrapSchedule(topi::generic::default_schedule))
.register_func({ "cuda", "gpu" }, WrapSchedule(topi::cuda::schedule_dense))
.register_func({ "rocm" }, WrapSchedule(topi::rocm::schedule_dense));

TVM_REGISTER_GENERIC_FUNC(schedule_pool)
.set_default(WrapSchedule(topi::generic::default_schedule))
.register_func({ "cpu" }, WrapSchedule(topi::x86::default_schedule))
.register_func({ "cuda", "gpu" }, WrapSchedule(topi::cuda::schedule_pool));

TVM_REGISTER_GENERIC_FUNC(schedule_global_pool)
.set_default(WrapSchedule(topi::generic::default_schedule))
.register_func({ "cpu" }, WrapSchedule(topi::x86::default_schedule))
.register_func({ "cuda", "gpu" }, WrapSchedule(topi::cuda::schedule_global_pool));

TVM_REGISTER_GENERIC_FUNC(schedule_reduce)
.set_default(WrapSchedule(topi::generic::default_schedule_auto_inline))
.register_func({ "cpu" }, WrapSchedule(topi::x86::default_schedule_auto_inline))
.register_func({ "cuda", "gpu" }, WrapSchedule(topi::cuda::schedule_reduce));

TVM_REGISTER_GENERIC_FUNC(schedule_binarize_pack)
.set_default(WrapSchedule(topi::generic::default_schedule))
.register_func({ "cpu" }, WrapSchedule(topi::x86::schedule_binarize_pack));

TVM_REGISTER_GENERIC_FUNC(schedule_binary_dense)
.set_default(WrapSchedule(topi::generic::default_schedule))
.register_func({ "cpu" }, WrapSchedule(topi::x86::schedule_binary_dense));

/*! \brief Builder function for instantiating dense ops. */
using FTVMDenseOpBuilder = std::function<tvm::Tensor(const Target& target,
                                                     const tvm::Tensor& data,
                                                     const tvm::Tensor& weight,
                                                     const tvm::Tensor& bias)>;

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
    auto target = Target::current_target(false);
    Tensor data = args[0];
    Tensor weight = args[1];
    Tensor bias = args[2];

    *ret = builder(target, data, weight, bias);
  });
}

TVM_REGISTER_GENERIC_FUNC(dense)
.set_default(WrapDenseOp([](const Target& target,
                            const tvm::Tensor& data,
                            const tvm::Tensor& weight,
                            const tvm::Tensor& bias) {
  return topi::nn::dense(data, weight, bias);
}))
.register_func({ "cuda", "gpu" }, WrapDenseOp(topi::cuda::dense_cuda))
.register_func({ "rocm" }, WrapDenseOp(topi::rocm::dense_rocm));

}  // namespace topi
