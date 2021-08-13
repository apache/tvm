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
 * \file src/relay/analysis/context_analysis.cc
 * \brief A pass for analyzing device attribute of each IR node.
 *
 * We use union-find data structures to analyze the context information of each
 * sub-expression in a Relay program in this pass. Only the device copy node in
 * Relay directly contains bidiretional device information. We use it to
 * bidirectionally propagate the device info of its inputs and outputs.
 *
 * However, to support dynamism (e.g dynamic inputs), Relay introduces several
 * concepts to compute the shape of tensors and operators at runtime, i.e.
 * shape_of, shape_func, and reshape_tensor. These nodes are also referred to as
 * VM dialects as we have native VM instructions for them. These dialects are
 * intrinsically CPU friendly, therefore, they are only designed to be
 * executed on CPU. We, hence, unify their inputs and outputs to CPU as well.
 * Note the input of shape_of is a tensor and we only need the tensor shape.
 * Therefore, the input could be sitting on GPU as well since no real data is
 * needed. The context of the input would be propagated from its other
 * consumers or fallback to the default device.
 *
 * Another type of dialect is used fo memory allocation, namely, alloc_storage
 * and alloc_tensor. alloc_storage contains a context field to indicate where
 * the chunk of memory is allocated. Therefore, we unify the context of
 * alloc_storage with the context field. Other inputs, such as size and
 * alignment, are left on CPU.
 *
 * Based on the above rules, we keep unifying the connected expressions and
 * propagating their device information. An error will be raised whenever there
 * is a unification conflict. All IR nodes that are not propagated with device
 * context will fallback to the specified device.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/attrs/memory.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/object.h>

namespace tvm {
namespace relay {

using PackedAnalysisResultMap = Map<Expr, Array<Integer>>;
using AnalysisResultMap =
    std::unordered_map<Expr, Device, runtime::ObjectPtrHash, runtime::ObjectPtrEqual>;

namespace analysis {

// Cache ops
static const Op& device_copy_op = Op::Get("device_copy");
static const Op& alloc_storage_op = Op::Get("memory.alloc_storage");
static const Op& alloc_tensor_op = Op::Get("memory.alloc_tensor");
static const Op& shape_of_op = Op::Get("vm.shape_of");
static const Op& invoke_tvm_op = Op::Get("vm.invoke_tvm_op");
static const Op& shape_func_of = Op::Get("vm.shape_func");
static const Op& reshape_tensor_op = Op::Get("vm.reshape_tensor");

class DeviceDomain;
using DeviceDomainPtr = std::shared_ptr<DeviceDomain>;

/*
 * \brief A class to represent the device of a domain, i.e. a segment of relay program.
 */
class DeviceDomain {
 public:
  // Construct an empty domain.
  DeviceDomain() {
    device_.device_type = static_cast<DLDeviceType>(-1);
    device_.device_id = -1;
  }

  // Construct a domain based on a given context.
  explicit DeviceDomain(const Device& dev) : device_(dev) {}

  // Check if the current domain is empty.
  bool IsEmptyDomain() const {
    return static_cast<int>(device_.device_type) == -1 && device_.device_id == -1;
  }

  // Check if the current domain equals the other one.
  bool operator==(const DeviceDomain& other) const {
    return device_.device_type == other.device_.device_type &&
           device_.device_id == other.device_.device_id;
  }

  bool operator!=(const DeviceDomain& other) const { return !(*this == other); }

 private:
  // Create a hash for a domain.
  struct Hash {
    size_t operator()(const DeviceDomainPtr& domain) const {
      if (domain->IsEmptyDomain()) {
        return static_cast<size_t>(reinterpret_cast<uintptr_t>(domain.get()));
      } else {
        size_t const h1(std::hash<int>()(static_cast<int>(domain->device_.device_type)));
        size_t const h2(std::hash<int>()(domain->device_.device_id));
        return h1 ^ (h2 << 1);
      }
    }
  };

  // Create an equality for domains.
  struct Equal {
   public:
    bool operator()(const DeviceDomainPtr& lhs, const DeviceDomainPtr& rhs) const {
      // We compare the pointer for empty domains.
      if (lhs->IsEmptyDomain() && rhs->IsEmptyDomain()) return lhs.get() == rhs.get();

      // Otherwise device type and id are used to check equality.
      return (*lhs.get() == *rhs.get());
    }
  };

  /* \brief The device to be assigned to the current domain. */
  Device device_;

  friend DeviceDomainPtr Join(const DeviceDomainPtr& lhs, const DeviceDomainPtr& rhs);
  friend class ContextAnalyzer;
};

// Join two domains.
DeviceDomainPtr Join(const DeviceDomainPtr& lhs, const DeviceDomainPtr& rhs) {
  if (lhs->IsEmptyDomain() && rhs->IsEmptyDomain()) {
    return lhs;
  } else if (lhs->IsEmptyDomain()) {
    return rhs;
  } else if (rhs->IsEmptyDomain()) {
    return lhs;
  } else {
    ICHECK(*lhs.get() == *rhs.get()) << "All expressions must have a singular device to unify";
    return lhs;
  }
}

/*
 * \brief Compute on which device each sub-expression will execute. A union find
 * algorithm is used to assign and merge the context domains.
 */
class ContextAnalyzer : public MixedModeVisitor {
 public:
  ContextAnalyzer(const IRModule& mod, const GlobalVar& current_func,
                  const Device& default_device)
      : MixedModeVisitor(9),  // the number of repeated visits a node can perform
        mod_(mod),
        current_func_(current_func),
        default_device_(default_device) {
    cpu_dev_.device_type = kDLCPU;
    cpu_dev_.device_id = 0;
  }

  // Create an empty domain.
  // This usually happens when we enter a new scope, i.e. Function.
  DeviceDomainPtr Bottom() { return std::make_shared<DeviceDomain>(DeviceDomain()); }

  // Create a domain with the given device context.
  DeviceDomainPtr DeviceType(const Device& dev) {
    return std::make_shared<DeviceDomain>(DeviceDomain(dev));
  }

  // Find the root of a device.
  DeviceDomainPtr Lookup(DeviceDomainPtr device) {
    while (device_uf_.count(device) && device != device_uf_[device]) {
      // Path compression
      if (device_uf_.count(device_uf_[device])) {
        device_uf_[device] = device_uf_[device_uf_[device]];
      }
      device = device_uf_[device];
    }
    return device;
  }

  // Unify two domains.
  DeviceDomainPtr Unify(DeviceDomainPtr lhs, DeviceDomainPtr rhs) {
    lhs = Lookup(lhs);
    rhs = Lookup(rhs);
    auto unified_device = Join(lhs, rhs);
    if (lhs != unified_device) {
      device_uf_[lhs] = unified_device;
    }

    if (rhs != unified_device) {
      device_uf_[rhs] = unified_device;
    }

    return unified_device;
  }

  // Unify the domain for two IR nodes.
  DeviceDomainPtr UnifyExpr(const Expr& lhs, const Expr& rhs) {
    auto lhs_dom = DeviceFor(lhs);
    auto rhs_dom = DeviceFor(rhs);
    return Unify(lhs_dom, rhs_dom);
  }

  // Lookup or insert an IR node to device domain map.
  DeviceDomainPtr DeviceFor(const Expr& expr) {
    auto it = expr_to_device_.find(expr);
    if (it == expr_to_device_.end()) {
      auto bottom = Bottom();
      expr_to_device_[expr] = bottom;
      return bottom;
    } else {
      return it->second;
    }
  }

  // Unify the device context for a device copy node. Device copy node is
  // the only node that carries bidirectional devices in the input program. The device
  // attribute of other nodes can be propagated from it.
  void UnifyDeviceCopy(const std::vector<Expr>& inps, const std::vector<Expr>& outputs,
                       DLDeviceType src_dev_type, DLDeviceType dst_dev_type) {
    Device src_dev;
    src_dev.device_type = src_dev_type;
    src_dev.device_id = 0;
    auto src_domain = DeviceType(src_dev);
    for (const auto& it : inps) {
      auto lhs = DeviceFor(it);
      Unify(lhs, src_domain);
    }

    Device dst_dev;
    dst_dev.device_type = dst_dev_type;
    dst_dev.device_id = 0;
    auto dst_domain = DeviceType(dst_dev);
    for (const auto& it : outputs) {
      auto lhs = DeviceFor(it);
      Unify(lhs, dst_domain);
    }
  }

  // Unify the domain of inputs and outputs of a relay call.
  //
  // For most call nodes, the op, inputs, and outputs should all be in the
  // same domain, i.e. having the same context. However, device_copy call node
  // needs to be handled differently as it copies data from one device to
  // another.
  DeviceDomainPtr UnifyCall(const Expr& call_op, const Array<Expr>& inps,
                            const Array<Expr>& outputs, DeviceDomainPtr device) {
    device = Unify(device, DeviceFor(call_op));

    for (const auto& it : inps) {
      device = Unify(device, DeviceFor(it));
    }

    for (const auto& it : outputs) {
      device = Unify(device, DeviceFor(it));
    }

    return device;
  }

  void VisitExpr_(const CallNode* cn) final {
    Call call = GetRef<Call>(cn);

    if (IsDeviceCopy(call)) {
      UnifyDeviceCopyCall(cn);
    } else if (call->op == alloc_storage_op) {
      UnifyAllocStorageCall(cn);
    } else if (call->op == alloc_tensor_op) {
      UnifyAllocTensorCall(cn);
    } else if (call->op == shape_func_of) {
      UnifyShapeFuncCall(cn);
    } else if (call->op == shape_of_op) {
      UnifyShapeOfCall(cn);
    } else if (call->op == invoke_tvm_op) {
      UnifyInvokeTVMOpCall(cn);
    } else if (call->op == reshape_tensor_op) {
      UnifyReshapeTensorCall(cn);
    } else if (call->op.as<FunctionNode>()) {
      UnifyFunctionCall(cn);
    } else if (call->op.as<GlobalVarNode>()) {
      UnifyGlobalVarCall(cn);
    } else if (call->op.as<VarNode>()) {
      UnifyVarCall(cn);
    } else {
      UnifyCall(call, cn->args, {call}, Bottom());
      MixedModeVisitor::VisitExpr_(cn);
    }
  }

  void VisitExpr_(const LetNode* ln) final {
    Expr expr = GetRef<Let>(ln);
    // Iteratively visit let nodes to avoid stack overflow.
    while (expr->IsInstance<LetNode>()) {
      Let let = Downcast<Let>(expr);
      // Save currying/closures since they will be invoked later
      auto ty = let->value->checked_type();
      if (ty->IsInstance<FuncTypeNode>()) {
        auto gv = ExtractClosure(let);
        ICHECK(gv.defined() && gv->IsInstance<GlobalVarNode>());
        closures_[let->var] = Downcast<GlobalVar>(gv);
      }

      // Unify let var, value, and body
      Unify(DeviceFor(let->var), DeviceFor(let->value));
      UnifyExpr(let, let->body);
      MixedModeVisitor::VisitExpr(let->value);
      expr = let->body;
    }
    // Visit the last body
    MixedModeVisitor::VisitExpr(expr);
  }

  void VisitExpr_(const FunctionNode* fn) final {
    auto func = GetRef<Function>(fn);
    // No need to step into fused primitive functions as they are handled as
    // a whole.
    if (fn->attrs.HasNonzeroAttr(attr::kPrimitive)) {
      return;
    }

    auto device = Unify(DeviceFor(func), DeviceFor(fn->body));
    for (const auto& it : fn->params) {
      DeviceFor(it);
    }
    MixedModeVisitor::VisitExpr(fn->body);
  }

  void VisitExpr_(const TupleNode* tn) final {
    // We only support tuple with the same of device.
    Tuple tup = GetRef<Tuple>(tn);
    if (tn->fields.size() > 0) {
      auto device = DeviceFor(tup->fields[0]);
      for (size_t i = 1; i < tup->fields.size(); i++) {
        device = Unify(device, DeviceFor(tup->fields[i]));
      }
      Unify(device, DeviceFor(tup));
    }
    MixedModeVisitor::VisitExpr_(tn);
  }

  void VisitExpr_(const TupleGetItemNode* tn) final {
    TupleGetItem item = GetRef<TupleGetItem>(tn);

    Unify(DeviceFor(item), DeviceFor(item->tuple));

    MixedModeVisitor::VisitExpr_(tn);
  }

  void VisitExpr_(const MatchNode* mn) final {
    // For match node, we unify the value and the rhs of each clause
    Match m = GetRef<Match>(mn);
    auto device = Unify(DeviceFor(m), DeviceFor(m->data));
    for (const auto& c : m->clauses) {
      device = Unify(device, DeviceFor(c->rhs));
    }
    MixedModeVisitor::VisitLeaf(mn->data);
    for (const Clause& c : mn->clauses) {
      this->VisitClause(c);
      MixedModeVisitor::VisitLeaf(c->rhs);
    }
  }

  void VisitExpr_(const GlobalVarNode* gvn) final { DeviceFor(GetRef<GlobalVar>(gvn)); }

  void VisitExpr_(const VarNode* vn) { DeviceFor(GetRef<Var>(vn)); }

  void VisitExpr_(const ConstantNode* cn) final { DeviceFor(GetRef<Constant>(cn)); }

  // Return the analysis results.
  AnalysisResultMap Results() {
    AnalysisResultMap ret;
    for (const auto& it : expr_to_device_) {
      auto device = Lookup(it.second);
      if (device->IsEmptyDomain()) {
        ret[it.first] = default_device_;
      } else {
        ret[it.first] = device->device_;
      }
    }

    return ret;
  }

 private:
  Expr ExtractClosure(Expr expr) const {
    while (expr->IsInstance<LetNode>()) {
      Let let = Downcast<Let>(expr);
      expr = let->value;
      if (expr->IsInstance<GlobalVarNode>()) {
        return expr;
      } else {
        const auto* cn = expr.as<CallNode>();
        if (cn && cn->op->IsInstance<GlobalVarNode>()) {
          return cn->op;
        }
      }
    }
    return Expr(nullptr);
  }

  // Check if an expression is a device copy call.
  bool IsDeviceCopy(const Expr& expr) const {
    if (!expr->IsInstance<CallNode>()) return false;

    Call call = Downcast<Call>(expr);
    if (call->op == device_copy_op) return true;

    // Fused function with device copy op as the body
    // device copy op is opaque therefore the fused function only has one node.
    if (const FunctionNode* fn = call->op.as<FunctionNode>()) {
      if (const CallNode* cn = fn->body.as<CallNode>()) {
        return cn->op == device_copy_op;
      }
    }

    return false;
  }

  // Check if a function is a closure.
  bool IsClosure(const Function& func) { return func->attrs.GetAttr<Integer>(attr::kClosure, 0) != 0; }

  // Check if a function is a currying function.
  bool IsCurrying(const Function& func) {
    if (const auto* let = func->body.as<LetNode>()) {
      return closures_.find(let->var) != closures_.end();
    }
    return false;
  }

  // Process device copy call node
  void UnifyDeviceCopyCall(const CallNode* call) {
    ICHECK_EQ(call->args.size(), 1U);

    std::vector<Expr> inps{call->args[0]};
    std::vector<Expr> outs{GetRef<Call>(call)};
    DLDeviceType src_dev_type, dst_dev_type;
    const DeviceCopyAttrs* attrs = nullptr;
    if (const auto* fn = call->op.as<FunctionNode>()) {
      // device_copy is fused, propagate device to the fused function.
      inps.push_back(fn->params[0]);
      outs.push_back(call->op);
      Expr body = fn->body;
      ICHECK(body->IsInstance<CallNode>() && IsDeviceCopy(body));
      Call call_body = Downcast<Call>(body);
      attrs = call_body->attrs.as<DeviceCopyAttrs>();
    } else {
      attrs = call->attrs.as<DeviceCopyAttrs>();
    }
    ICHECK(attrs != nullptr);
    src_dev_type = static_cast<DLDeviceType>(attrs->src_dev_type);
    dst_dev_type = static_cast<DLDeviceType>(attrs->dst_dev_type);

    //  Device copy op only has one input which is now annotated with the
    //  same device to the source device type of the device copy op.
    //  The call itself has the same device type to the destination.
    UnifyDeviceCopy(inps, outs, src_dev_type, dst_dev_type);
    MixedModeVisitor::VisitExpr_(call);
  }

  void UnifyAllocStorageCall(const CallNode* call) {
    // [size, alignment]
    ICHECK_EQ(call->args.size(), 2U);

    // The arguments of alloc storage should be on CPU.
    for (int i = 0; i < 2; i++) {
      Unify(DeviceFor(call->args[i]), DeviceType(cpu_dev_));
      MixedModeVisitor::VisitExpr(call->args[i]);
    }
    Device dev;
    const auto* attrs = call->attrs.as<AllocStorageAttrs>();
    dev.device_type = static_cast<DLDeviceType>(attrs->device_type);
    dev.device_id = attrs->device_id;
    Unify(DeviceFor(GetRef<Call>(call)), DeviceType(dev));
  }

  void UnifyAllocTensorCall(const CallNode* call) {
    // [storage, offset, shape]
    ICHECK_EQ(call->args.size(), 3U);

    Expr storage = call->args[0];
    Expr shape = call->args[1];
    Unify(DeviceFor(storage), DeviceFor(GetRef<Call>(call)));

    // The shape for alloc_tensor should be on CPU.
    Unify(DeviceFor(shape), DeviceType(cpu_dev_));
    MixedModeVisitor::VisitExpr(shape);
  }

  void UnifyShapeFuncCall(const CallNode* call) {
    // [func, inputs, outputs]
    ICHECK_EQ(call->args.size(), 3U);
    auto shape_func_domain = DeviceType(cpu_dev_);

    // No need to unify the op of a shape_func as shape_func doesn't
    // invoke the op itself. It should be handled by invoke_tvm_op.
    // Therefore, we skip call.args[0] here.
    Tuple inps = Downcast<Tuple>(call->args[1]);
    Tuple outputs = Downcast<Tuple>(call->args[2]);
    UnifyCall(GetRef<Call>(call), inps->fields, outputs->fields, shape_func_domain);
    for (const auto& it : inps->fields) {
      MixedModeVisitor::VisitExpr(it);
    }

    for (const auto& it : outputs->fields) {
      MixedModeVisitor::VisitExpr(it);
    }
  }

  void UnifyInvokeTVMOpCall(const CallNode* call) {
    // [op, inputs, outputs]
    ICHECK_EQ(call->args.size(), 3U);
    Tuple inps = Downcast<Tuple>(call->args[1]);
    Tuple outputs = Downcast<Tuple>(call->args[2]);
    UnifyCall(call->args[0], inps->fields, outputs->fields, Bottom());
    MixedModeVisitor::VisitExpr_(call);
  }

  void UnifyShapeOfCall(const CallNode* call) {
    // vm shape_of is always on the CPU.
    ICHECK_EQ(call->args.size(), 1U);
    MixedModeVisitor::VisitExpr(call->args[0]);
    // Note we don't unify the input of a shape_of with the cpu domain. This is
    // because vm.shape_of has a native instruction to compute the shape of
    // a tensor regardless its device type.
    // Instead, the device type of the input is left for its other consumers to
    // unify or it will fallback to the default context.
    Unify(DeviceFor(GetRef<Call>(call)), DeviceType(cpu_dev_));
  }

  void UnifyReshapeTensorCall(const CallNode* call) {
    // [data, shape]
    ICHECK_EQ(call->args.size(), 2U);
    Expr data = call->args[0];
    Expr shape = call->args[1];
    Unify(DeviceFor(GetRef<Call>(call)), DeviceFor(data));

    // The shape field of reshape_tensor is always on the CPU.
    Unify(DeviceFor(shape), DeviceType(cpu_dev_));
    MixedModeVisitor::VisitExpr(data);
    MixedModeVisitor::VisitExpr(shape);
  }

  void UnifyFunctionCall(const CallNode* call) {
    auto device = DeviceFor(GetRef<Call>(call));
    // Unify the arguments of the caller.
    for (const auto& arg : call->args) {
      device = Unify(device, DeviceFor(arg));
      MixedModeVisitor::VisitExpr(arg);
    }

    // Unify the parameters of the callee.
    if (!call->op->IsInstance<FunctionNode>()) return;
    Function func = Downcast<Function>(call->op);
    for (const auto& param : func->params) {
      device = Unify(device, DeviceFor(param));
      MixedModeVisitor::VisitExpr(param);
    }

    // Unify the function expression and its body
    Unify(device, DeviceFor(call->op));
    Unify(device, DeviceFor(func->body));

    // Step into the callee. It will be skipped if the callee if a primitive
    // function
    MixedModeVisitor::VisitExpr(call->op);
  }

  // Invoke a global function.
  void UnifyGlobalVarCall(const CallNode* call) {
    auto device = DeviceFor(GetRef<Call>(call));
    ICHECK(mod_.defined()) << "Cannot analyze context on a globalvar without module";
    GlobalVar gv = Downcast<GlobalVar>(call->op);
    auto func = Downcast<Function>(mod_->Lookup(gv));
    ICHECK_EQ(call->args.size(), func->params.size())
        << "The number of arguments doesn't match the number of parameters of the function.";

    for (size_t i = 0; i < call->args.size(); i++) {
      Expr arg = call->args[i];
      Expr param = func->params[i];
      MixedModeVisitor::VisitExpr(arg);

      // Save the the arg to function mapping for closures as it will
      // be invoked/unified later.
      ICHECK(arg->checked_type().defined())
          << "Type inference is required to run the context analysis passes.";
      if (arg->checked_type()->IsInstance<FuncTypeNode>()) {
        auto it = closures_.find(arg);
        if (it != closures_.end()) {
          closures_[param] = it->second;
        } else {
          ICHECK(arg->IsInstance<GlobalVarNode>());
          closures_[param] = Downcast<GlobalVar>(arg);
        }
      }
      Unify(DeviceFor(arg), DeviceFor(param));
    }
    device = Unify(device, DeviceFor(call->op));
    device = Unify(device, DeviceFor(func));
    device = Unify(device, DeviceFor(func->body));

    // Step into the callee. We need to skip recursive calls, otherwise, it
    // would be a infinite loop.
    //
    // TODO(@zhiics) This may cause problem for mutual recursive calls as well.
    auto cur_func = current_func_;
    current_func_ = gv;
    if (cur_func->name_hint != gv->name_hint) {
      MixedModeVisitor::VisitExpr(func);
    }
    // Exit the frame.
    current_func_ = cur_func;
  }

  void UnifyVarCall(const CallNode* call) {
    // It is a closure when we call a var.
    // Unify the corresponding arguement and parameter.
    auto device = DeviceFor(GetRef<Call>(call));
    auto it = closures_.find(call->op);
    ICHECK(it != closures_.end()) << "Cannot find var: " << call->op;
    auto glb_var = it->second;
    ICHECK(mod_.defined()) << "Cannot analyze context on a globalvar without module";
    Function func = Downcast<Function>(mod_->Lookup(glb_var));
    // Unify the underlying function for clousre or currying functions.
    while (IsClosure(func) || IsCurrying(func)) {
      device = Unify(device, DeviceFor(func));
      if (IsClosure(func)) {
        func = Downcast<Function>(func->body);
      } else if (IsCurrying(func)) {
        Let let = Downcast<Let>(func->body);
        func = Downcast<Function>(mod_->Lookup(closures_[let->var]));
      } else {
        LOG(FATAL) << "func is expected to be a closure or a currying function";
      }
    }

    ICHECK_EQ(call->args.size(), func->params.size());
    for (size_t i = 0; i < call->args.size(); i++) {
      Unify(DeviceFor(call->args[i]), DeviceFor(func->params[i]));
      MixedModeVisitor::VisitExpr(call->args[i]);
    }
    device = Unify(device, DeviceFor(call->op));
    device = Unify(device, DeviceFor(glb_var));
    device = Unify(device, DeviceFor(func));

    // Step into the global function.
    auto cur_func = current_func_;
    current_func_ = glb_var;
    if (cur_func->name_hint != glb_var->name_hint) {
      MixedModeVisitor::VisitExpr(func);
    }
    current_func_ = cur_func;
  }

 private:
  /* \brief The cpu context. */
  Device cpu_dev_;
  /* \brief The module that helps context analysis. */
  const IRModule& mod_;
  /* \brief The current function that is being analyzed. */
  GlobalVar current_func_;
  /* \brief The default device that could be attached to an expression. */
  const Device& default_device_;
  /* \brief The IR node to device domain mapping. */
  std::unordered_map<Expr, DeviceDomainPtr, runtime::ObjectPtrHash, runtime::ObjectPtrEqual>
      expr_to_device_;
  /* \brief The domain map for union-find. */
  std::unordered_map<DeviceDomainPtr, DeviceDomainPtr, DeviceDomain::Hash, DeviceDomain::Equal>
      device_uf_;
  /*
   * \brief The expr to global var map. It saves the closures/currying that
   * will be invoked lazily.
   */
  std::unordered_map<Expr, GlobalVar, runtime::ObjectPtrHash, runtime::ObjectPtrEqual> closures_;
};

}  // namespace analysis

AnalysisResultMap ContextAnalysis(const IRModule& mod, const Device& default_device) {
  // TODO(@zhiics) Apply the pass to all functions/entries
  auto entry = mod->GetGlobalVar("main");
  auto ca = analysis::ContextAnalyzer(mod, entry, default_device);
  auto expr = mod->Lookup(entry);
  ca.VisitExpr(expr);
  return ca.Results();
}

// Unpack the device type and deivce id fields in Device for PackedFunc calls
// as Device is not in the object system.
PackedAnalysisResultMap ContextAnalysisPacked(const IRModule& mod, const Device& default_device) {
  PackedAnalysisResultMap ret;
  auto res = ContextAnalysis(mod, default_device);
  for (const auto& it : res) {
    Integer dev_ty = static_cast<int>(it.second.device_type);
    Integer dev_id = it.second.device_id;
    ret.Set(it.first, {dev_ty, dev_id});
  }

  return ret;
}

TVM_REGISTER_GLOBAL("relay.analysis.ContextAnalysis").set_body_typed(ContextAnalysisPacked);

}  // namespace relay
}  // namespace tvm
