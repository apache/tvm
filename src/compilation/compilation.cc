/*!
 *  Copyright (c) 2017 by Contributors
 *  Compile executable modules.
 * \file compilation.cc
 */
#include "./tvm/compilation.h"
#include "./tvm/operation.h"
#include "./tvm/ir_pass.h"
#include "./tvm/codegen.h"


namespace tvm {
namespace compilation {


Buffer bufferWithOffsetAlignment(Array<Expr> shape, Type dtype, std::string name,
    int data_alignment, int offset_factor) {
    auto data = Var(name, Handle());

    auto shapeType = TVMType2Type(GetAttr(shape[0], "dtype").v_type);
    Expr elem_offset;
    if (offset_factor != 0) {
        elem_offset = Var(name + "_elem_offset", shapeType);
    } else {
        elem_offset = Expr();
    }

    return BufferNode::make(data, dtype, shape, Array<Expr>(), elem_offset, name, "",
        data_alignment, offset_factor);
}

void get_binds(Array<Tensor> args, std::unordered_map<Tensor, Buffer> binds,
    Map<Tensor, Buffer>* bindsOut, Array<NodeRef>* argListOut,
    const BuildConfig& config) {
    *bindsOut = binds;

    for (const auto &x : args) {
        if (bindsOut->find(x) == bindsOut->end()) {
            auto buf = bufferWithOffsetAlignment(x->shape, x->dtype, x->op->name,
                config.data_alignment, config.offset_factor);
            bindsOut->Set(x, buf);
            argListOut->push_back(buf);
        } else {
            argListOut->push_back((*bindsOut)[x]);
        }
    }
}


Stmt BuildStmt(Schedule sch, Array<Tensor> args, std::unordered_map<Tensor, Buffer> binds,
    bool loopPartition, Array<NodeRef> *argListOut, const BuildConfig& config) {
    Map<Tensor, Buffer> bindsOut;
    get_binds(args, binds, &bindsOut, argListOut, config);

    sch = sch.normalize();

    // Phase 0

    auto bounds = schedule::InferBound(sch);
    auto stmt = schedule::ScheduleOps(sch, bounds);
    stmt = ir::InjectPrefetch(stmt);


    // Phase 1

    stmt = ir::StorageFlatten(stmt, bindsOut, 64);
    stmt = ir::CanonicalSimplify(stmt);
    if (loopPartition) {
        stmt = ir::LoopPartition(stmt);
    }
    stmt = ir::VectorizeLoop(stmt);
    stmt = ir::InjectVirtualThread(stmt);
    stmt = ir::InjectDoubleBuffer(stmt, config.double_buffer_split_loop);
    stmt = ir::StorageRewrite(stmt);
    stmt = ir::UnrollLoop(stmt, config.auto_unroll_max_step, config.auto_unroll_max_depth,
        config.auto_unroll_max_extent, config.unroll_explicit);


    // Phase 2

    stmt = ir::Simplify(stmt);
    stmt = ir::LowerStorageAccessInfo(stmt);
    stmt = ir::RemoveNoOp(stmt);
    stmt = ir::RewriteUnsafeSelect(stmt);

    return stmt;
}

LoweredFunc Lower(Schedule sch, Array<Tensor> args, std::string name,
    std::unordered_map<Tensor, Buffer> binds, const BuildConfig& config) {
    Array<NodeRef> argListOut;
    auto stmt = BuildStmt(sch, args, binds, true, &argListOut, config);
    return ir::MakeAPI(stmt, name, argListOut, 0, config.restricted_func);
}

runtime::Module BuildModule(Array<LoweredFunc> funcs, const Target& target,
    const Target& targetHost, const BuildConfig& config) {
    std::unordered_set<std::string> allNames;
    for (const auto &x : funcs) {
        CHECK(allNames.count(x->name) == 0) << "Duplicate function name " << x->name;
        allNames.insert(x->name);
    }

    Array<LoweredFunc> fhost;
    Array<LoweredFunc> fdevice;

    for (const auto &x : funcs) {
        auto func_type = GetAttr(x, "func_type").v_int64;

        if (func_type == kMixedFunc) {
            auto func = x;
            if (config.detect_global_barrier) {
                func = ir::ThreadSync(func, "global");
            }

            func = ir::ThreadSync(func, "shared");
            func = ir::LowerThreadAllreduce(func, target.thread_warp_size);
            auto fsplits = ir::SplitHostDevice(func);
            fhost.push_back(fsplits[0]);
            for (auto f = fsplits.begin() + 1; f != fsplits.end(); ++f) {
                fdevice.push_back(*f);
            }
        } else if (func_type == kHostFunc) {
            fhost.push_back(x);
        } else if (func_type == kDeviceFunc) {
            fdevice.push_back(x);
        } else {
            CHECK(false) << "unknown function type " << func_type;
        }
    }

    if (target.hasKey("gpu") && fdevice.size() == 0) {
        LOG(WARNING) << "Specified target " + target.str() +
            " but cannot find device code. Did you forget to bind?";
    }

    for (int i = 0; i < fhost.size(); ++i) {
        auto func = fhost[i];
        func = ir::BindDeviceType(func, target.deviceType);
        func = ir::LowerTVMBuiltin(func);
        fhost.Set(i, func);
    }


    for (int i = 0; i < fdevice.size(); ++i) {
        auto func = fdevice[i];
        func = ir::LowerIntrin(func, target.targetName);
        fdevice.Set(i, func);
    }

    for (int i = 0; i < fhost.size(); ++i) {
        auto func = fhost[i];
        func = ir::LowerIntrin(func, targetHost.targetName);
        func = ir::CombineContextCall(func);
        fhost.Set(i, func);
    }

    auto mhost = codegen::Build(fhost, targetHost.str());

    if (fdevice.size() > 0) {
        auto mdev = codegen::Build(fdevice, target.str());
        mhost.Import(mdev);
    }

    return mhost;
}

}  // namespace compilation
}  // namespace tvm
