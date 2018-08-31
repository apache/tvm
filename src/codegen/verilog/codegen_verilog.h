/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_verilog.h
 * \brief Generate verilog code.
 */
#ifndef TVM_CODEGEN_VERILOG_CODEGEN_VERILOG_H_
#define TVM_CODEGEN_VERILOG_CODEGEN_VERILOG_H_

#include <tvm/base.h>
#include <tvm/ir.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/codegen.h>
#include <tvm/lowered_func.h>
#include <string>
#include <vector>
#include <unordered_map>
#include "verilog_ir.h"
#include "../codegen_source_base.h"

namespace tvm {
namespace codegen {
namespace verilog {
using namespace ir;

/* \brief The variable type in register.*/
enum VerilogVarType {
  kWire,
  kInput,
  kOutput,
  kReg,
  kConst
};

/*! \brief The verilog value */
struct VerilogValue {
  /*! \brief The variable id */
  std::string vid;
  /*! \brief The variable type */
  VerilogVarType vtype{kReg};
  /*! \brief The data type it encodes */
  Type dtype;
  VerilogValue() {}
  VerilogValue(std::string vid, VerilogVarType vtype, Type dtype)
      : vid(vid), vtype(vtype), dtype(dtype) {}
};

/*! \brief Information of each procedure function generated */
struct VerilogFuncEntry {
  /*! \brief The original functions */
  std::vector<Type> arg_types;
  /*! \brief The real argument ids of the function */
  std::vector<std::string> arg_ids;
  /*! \brief The VPI Modules in the function */
  std::vector<std::string> vpi_modules;
};

/*!
 * \brief The code module of generated verilog code.
 */
class VerilogCodeGenModule {
 public:
  /*! \brief the code of each modoules */
  std::string code;
  /*! \brief map of functions */
  std::unordered_map<std::string, VerilogFuncEntry> fmap;
  /*!
   * \brief Generate a code that append simulator function to call func_name.
   * \param func_name The function to be called.
   * \return The generated code.
   */
  std::string AppendSimMain(const std::string& func_name) const;
};

/*!
 * \brief Verilog generator
 */
class CodeGenVerilog :
      public ExprFunctor<VerilogValue(const Expr&)>,
      public CodeGenSourceBase {
 public:
  /*!
   * \brief Initialize the code generator.
   * \param output_ssa Whether output SSA.
   */
  void Init();
  /*!
   * \brief Add the function to the generated module.
   * \param f The function to be compiled.
   */
  void AddFunction(LoweredFunc f);
  /*!
   * \brief Finalize the compilation and return the code.
   * \return The code.
   */
  VerilogCodeGenModule Finish();
  /*!
   * \brief Transform expression to verilog value.
   * \param n The expression to be printed.
   */
  VerilogValue MakeValue(const Expr& n) {
    return VisitExpr(n);
  }
  // The following parts are overloadable print operations.
  // expression
  VerilogValue VisitExpr_(const Variable* op) final;
  VerilogValue VisitExpr_(const Let* op) final;
  VerilogValue VisitExpr_(const Call* op) final;
  VerilogValue VisitExpr_(const Add* op) final;
  VerilogValue VisitExpr_(const Sub* op) final;
  VerilogValue VisitExpr_(const Mul* op) final;
  VerilogValue VisitExpr_(const Div* op) final;
  VerilogValue VisitExpr_(const Mod* op) final;
  VerilogValue VisitExpr_(const Min* op) final;
  VerilogValue VisitExpr_(const Max* op) final;
  VerilogValue VisitExpr_(const EQ* op) final;
  VerilogValue VisitExpr_(const NE* op) final;
  VerilogValue VisitExpr_(const LT* op) final;
  VerilogValue VisitExpr_(const LE* op) final;
  VerilogValue VisitExpr_(const GT* op) final;
  VerilogValue VisitExpr_(const GE* op) final;
  VerilogValue VisitExpr_(const And* op) final;
  VerilogValue VisitExpr_(const Or* op) final;
  VerilogValue VisitExpr_(const Cast* op) final;
  VerilogValue VisitExpr_(const Not* op) final;
  VerilogValue VisitExpr_(const Select* op) final;
  VerilogValue VisitExpr_(const Ramp* op) final;
  VerilogValue VisitExpr_(const Broadcast* op) final;
  VerilogValue VisitExpr_(const IntImm* op) final;
  VerilogValue VisitExpr_(const UIntImm* op) final;
  VerilogValue VisitExpr_(const FloatImm* op) final;
  VerilogValue VisitExpr_(const StringImm* op) final;

 protected:
  void InitFuncState(LoweredFunc f);
  void PrintDecl(const std::string& vid, VerilogVarType vtype, Type dtype,
                 const char* suffix = ";\n", bool indent = true);
  void PrintAssign(
      const std::string& target, const std::string& src);
  void PrintAssignAnd(
      const std::string& target, const std::vector<std::string>& conds);
  void PrintLine(const std::string& line);
  void PrintSSAAssign(
      const std::string& target, const std::string& src, Type t) final;
  // make binary op
  VerilogValue MakeBinary(Type t, VerilogValue a, VerilogValue b, const char* opstr);

 private:
  // Hand shake signal name.
  // These name can be empty.
  // Indicate that the signal is always true
  // or do not need to take these signals.
  struct SignalEntry {
    std::string valid;
    std::string ready;
  };
  // Information about port
  struct PortEntry {
    // The port value
    std::string value;
    // The data type
    Type dtype;
  };
  // Channel setup
  struct ChannelEntry {
    // The channel block
    ChannelBlock block;
    // The port map, on how port is assigned.
    std::unordered_map<std::string, PortEntry> ports;
    // Assign port to be valueo
    void AssignPort(std::string port, std::string value, Type dtype);
    // Assign port to be valueo
    const PortEntry& GetPort(const std::string& port) const;
    // Signal port name
    std::string SignalPortName(int index) const;
  };

  // Get wire ssa value from s
  VerilogValue GetSSAValue(std::string s, Type dtype) {
    VerilogValue ret;
    ret.vid = SSAGetID(s, dtype);
    ret.vtype = kWire;
    ret.dtype = dtype;
    return ret;
  }
  void CodeGen(const Pipeline& pipeine);
  // codegen the delays
  void MakeDelay(const std::string& dst,
                 const std::string& src,
                 Type dtype,
                 int delay,
                 const std::string& not_stall);
  // codegen the loop macros
  SignalEntry MakeLoop(const Array<Stmt>& loop);
  // codegen the loop macros
  void MakeStageInputs(const ComputeBlock& block,
                       const std::string& not_stall,
                       std::string* out_all_input_valid);
  // codegen compute block
  void MakeStore(const ComputeBlock& block, const Store* store);
  // Codegen of load statement into FIFO
  void MakeLoadToFIFO(const ComputeBlock& block,
                      const Store* store,
                      const Load* load);
  // Make channel unit.
  void MakeChannelUnit(const ChannelEntry& ch);
  void MakeChannelFIFO(const ChannelEntry& ch);
  void MakeChannelBuffer(const ChannelEntry& ch);
  void MakeChannelMemMap(const ChannelEntry& ch);
  // Get channel information
  ChannelEntry* GetChannelInfo(const Variable* var);
  // channel setup map.
  std::unordered_map<const Variable*, ChannelEntry> cmap_;
  // list of vpi modules to be hooked.
  std::vector<std::string> tvm_vpi_modules_;
  // The signals for done.
  std::vector<std::string> done_sigs_;
  // The verilog function.
  std::unordered_map<std::string, VerilogFuncEntry> functions_;
};
}  // namespace verilog
}  // namespace codegen
}  // namespace tvm
#endif  // TVM_CODEGEN_VERILOG_CODEGEN_VERILOG_H_
