/*!
 *  Copyright (c) 2017 by Contributors
 * \file verilog_module.cc
 * \brief Build verilog source code.
 */
#include <tvm/runtime/packed_func.h>
#include <tvm/codegen.h>
#include <mutex>
#include "codegen_verilog.h"
#include "../../runtime/file_util.h"
#include "../../runtime/meta_data.h"

namespace tvm {
namespace codegen {
namespace verilog {
using runtime::TVMArgs;
using runtime::TVMRetValue;
using runtime::PackedFunc;

// Simulator function
class VerilogModuleNode : public runtime::ModuleNode {
 public:
  VerilogModuleNode() : fmt_("v") {}
  const char* type_key() const {
    return "verilog";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    CHECK(sptr_to_self.get() == this);
    if (!m_.fmap.count(name)) return PackedFunc();
    auto f = [sptr_to_self, name, this](const runtime::TVMArgs& args, TVMRetValue* rv) {
      auto* fsim = runtime::Registry::Get("tvm_callback_verilog_simulator");
      CHECK(fsim != nullptr)
        << "tvm_callback_verilog_simulator is not registered,"
        <<" did you import tvm.addon.verilog?";
      std::string code = m_.AppendSimMain(name);

      if (const auto* f = runtime::Registry::Get("tvm_callback_verilog_postproc")) {
        code = (*f)(code).operator std::string();
      }
      std::vector<TVMValue> values;
      std::vector<int> codes;
      TVMValue v;
      v.v_str = code.c_str();
      values.push_back(v);
      codes.push_back(kStr);
      for (int i = 0; i < args.num_args; ++i) {
        values.push_back(args.values[i]);
        codes.push_back(args.type_codes[i]);
      }
      fsim->CallPacked(TVMArgs(&values[0], &codes[0], args.num_args + 1), rv);
    };
    return PackedFunc(f);
  }

  std::string GetSource(const std::string& format) final {
    return m_.code;
  }

  void Init(const Array<LoweredFunc>& funcs) {
    CodeGenVerilog cg;
    cg.Init();
    for (LoweredFunc f :  funcs) {
      cg.AddFunction(f);
    }
    m_ = cg.Finish();
  }

 private:
  // the verilog code. data
  VerilogCodeGenModule m_;
  // format;
  std::string fmt_;
};

TVM_REGISTER_API("codegen.build_verilog")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    std::shared_ptr<VerilogModuleNode> n =
        std::make_shared<VerilogModuleNode>();
    n->Init(args[0]);
    *rv = runtime::Module(n);
  });
}  // namespace verilog
}  // namespace codegen
}  // namespace tvm
