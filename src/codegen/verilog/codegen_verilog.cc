/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_verilog.cc
 */
#include <tvm/ir_pass.h>
#include <cctype>
#include <sstream>
#include <iostream>
#include "codegen_verilog.h"
#include "../../arithmetic/compute_expr.h"

namespace tvm {
namespace codegen {
namespace verilog {

using namespace ir;

void CodeGenVerilog::Init() {
  stream << "`include \"tvm_marcos.v\"\n\n";
}

void CodeGenVerilog::InitFuncState(LoweredFunc f) {
  CodeGenSourceBase::ClearFuncState();
  cmap_.clear();
  tvm_vpi_modules_.clear();
  done_sigs_.clear();
}

void CodeGenVerilog::AddFunction(LoweredFunc f) {
  // clear previous generated state.
  this->InitFuncState(f);
  // skip the first underscore, so SSA variable starts from _1
  GetUniqueName("_");
  GetUniqueName("rst");
  GetUniqueName("clk");
  GetUniqueName("done");
  GetUniqueName("enable");
  GetUniqueName("all_input_valid");
  // print out function body.
  int func_scope = this->BeginScope();

  // Stich things up.
  stream << "module " << f->name << "(\n";
  PrintDecl("clk", kInput, Bool(1), "");
  stream << ",\n";
  PrintDecl("rst", kInput, Bool(1), "");
  VerilogFuncEntry entry;
  for (size_t i = 0; i < f->args.size(); ++i) {
    stream << ",\n";
    Var v = f->args[i];
    std::string vid = AllocVarID(v.get());
    entry.arg_ids.push_back(vid);
    entry.arg_types.push_back(v.type());
    PrintDecl(vid, kInput, v.type(), "");
  }
  stream << ",\n";
  PrintDecl("done", kOutput, Bool(1), "");
  stream << "\n);\n";
  this->CodeGen(MakePipeline(f));
  PrintAssignAnd("done", done_sigs_);
  this->EndScope(func_scope);
  this->PrintIndent();
  stream << "endmodule\n";
  entry.vpi_modules = std::move(tvm_vpi_modules_);
  functions_[f->name] = entry;
}

std::string VerilogCodeGenModule::AppendSimMain(
    const std::string& func_name) const {
  // Add main function for simulator hook
  const VerilogFuncEntry& entry = fmap.at(func_name);
  std::ostringstream stream;
  stream << code;
  stream << "\n"
         << "module main();\n"
         << "  `TVM_DEFINE_TEST_SIGNAL(clk, rst)\n";
  // print out function body.
  std::vector<std::string> sargs;
  for (size_t i = 0; i < entry.arg_types.size(); ++i) {
    Type t = entry.arg_types[i];
    std::ostringstream sarg;
    sarg << "tvm_arg" << i;
    std::string vid = sarg.str();
    stream << "  reg";
    if (t.bits() > 1) {
      stream << "["  << t.bits() - 1 << ":0]";
    }
    stream << " " << vid << ";\n";
    sargs.push_back(vid);
  }
  stream << "  wire done;\n";
  stream << "\n  " << func_name << " dut(\n"
         << "    .clk(clk),\n"
         << "    .rst(rst),\n";

  for (size_t i = 0; i < entry.arg_ids.size(); ++i) {
    stream << "    ." << entry.arg_ids[i] << '('
           << sargs[i] << "),\n";
  }
  stream << "    .done(done)\n"
         << "  );\n";


  stream << "  initial begin\n"
         << "    $tvm_session(clk";
  for (const std::string& mvpi : entry.vpi_modules) {
    stream << ", dut." << mvpi;
  }
  stream << ");\n"
         << "  end\n";
  stream << "endmodule\n";
  return stream.str();
}

VerilogCodeGenModule CodeGenVerilog::Finish() {
  VerilogCodeGenModule m;
  m.code = stream.str();
  m.fmap = std::move(functions_);
  return m;
}

void CodeGenVerilog::PrintDecl(
    const std::string& vid, VerilogVarType vtype, Type dtype,
    const char* suffix, bool indent) {
  if (indent) PrintIndent();
  switch (vtype) {
    case kReg:  stream << "reg "; break;
    case kWire: stream << "wire "; break;
    case kInput: stream << "input "; break;
    case kOutput: stream << "output "; break;
    default: LOG(FATAL) << "unsupported vtype=" << vtype;
  }
  int bits = dtype.bits();
  // bits for handle type.
  if (dtype.is_handle()) {
    bits = 64;
  }
  if (bits > 1) {
    stream << "[" << bits - 1 << ":0] ";
  }
  stream << vid << suffix;
}

void CodeGenVerilog::PrintSSAAssign(
    const std::string& target, const std::string& src, Type t) {
  // add target to list of declaration.
  PrintDecl(target, kWire, t, ";\n", false);
  PrintAssign(target, src);
}

void CodeGenVerilog::PrintAssign(
    const std::string& target, const std::string& src) {
  PrintIndent();
  stream << "assign " << target << " = ";
  if (src.length() > 3 &&
      src[0] == '(' && src[src.length() - 1] == ')') {
    stream << src.substr(1, src.length() - 2);
  } else {
    stream << src;
  }
  stream << ";\n";
}

void CodeGenVerilog::PrintAssignAnd(
    const std::string& target, const std::vector<std::string>& conds) {
  if (conds.size() != 0) {
    std::ostringstream os_valid;
    for (size_t i = 0; i < conds.size(); ++i) {
      if (i != 0) os_valid << " && ";
      os_valid << conds[i];
    }
    PrintAssign(target, os_valid.str());
  } else {
    PrintAssign(target, "1");
  }
}

void CodeGenVerilog::PrintLine(const std::string& line) {
  PrintIndent();
  stream << line << '\n';
}

VerilogValue CodeGenVerilog::MakeBinary(Type t,
                                        VerilogValue a,
                                        VerilogValue b,
                                        const char *opstr) {
  CHECK_EQ(t.lanes(), 1)
      << "Do not yet support vectorized op";
  CHECK(t.is_int() || t.is_uint())
      << "Only support integer operations";
  std::ostringstream os;
  os << a.vid << ' ' << opstr << ' '<< b.vid;
  return GetSSAValue(os.str(), t);
}

template<typename T>
inline VerilogValue IntConst(const T* op, CodeGenVerilog* p) {
  if (op->type.bits() <= 32 && op->type.lanes() == 1) {
    std::ostringstream temp;
    temp << op->value;
    p->MarkConst(temp.str());
    return VerilogValue(temp.str(), kConst, op->type);
  } else {
    LOG(FATAL) << "Do not support integer constant type " << op->type;
    return VerilogValue();
  }
}

VerilogValue CodeGenVerilog::VisitExpr_(const IntImm *op) {
  return IntConst(op, this);
}
VerilogValue CodeGenVerilog::VisitExpr_(const UIntImm *op) {
  return IntConst(op, this);
}
VerilogValue CodeGenVerilog::VisitExpr_(const FloatImm *op) {
  LOG(FATAL) << "Do not support float constant in Verilog";
  return VerilogValue();
}
VerilogValue CodeGenVerilog::VisitExpr_(const StringImm *op) {
  LOG(FATAL) << "Do not support string constant in Verilog";
  return VerilogValue();
}

VerilogValue CodeGenVerilog::VisitExpr_(const Cast *op) {
  LOG(FATAL) << "Type cast not supported";
  return VerilogValue();
}
VerilogValue CodeGenVerilog::VisitExpr_(const Variable *op) {
  return VerilogValue(GetVarID(op), kReg, op->type);
}

VerilogValue CodeGenVerilog::VisitExpr_(const Add *op) {
  return MakeBinary(op->type, MakeValue(op->a), MakeValue(op->b), "+");
}
VerilogValue CodeGenVerilog::VisitExpr_(const Sub *op) {
  return MakeBinary(op->type, MakeValue(op->a), MakeValue(op->b), "-");
}
VerilogValue CodeGenVerilog::VisitExpr_(const Mul *op) {
  return MakeBinary(op->type, MakeValue(op->a), MakeValue(op->b), "*");
}
VerilogValue CodeGenVerilog::VisitExpr_(const Div *op) {
  int shift;
  if (is_const_power_of_two_integer(op->b, &shift) &&
      (op->type.is_int() || op->type.is_uint())) {
    return MakeValue(op->a >> make_const(op->b.type(), shift));
  } else {
    LOG(FATAL) << "do not support synthesis division";
  }
  return VerilogValue();
}
VerilogValue CodeGenVerilog::VisitExpr_(const Mod *op) {
  LOG(FATAL) << "do not support synthesis Mod";
  return VerilogValue();
}
VerilogValue CodeGenVerilog::VisitExpr_(const Min *op) {
  LOG(FATAL) << "not supported";
  return VerilogValue();
}
VerilogValue CodeGenVerilog::VisitExpr_(const Max *op) {
  LOG(FATAL) << "not supported";
  return VerilogValue();
}
VerilogValue CodeGenVerilog::VisitExpr_(const EQ *op) {
  return MakeBinary(op->type, MakeValue(op->a), MakeValue(op->b), "==");
}
VerilogValue CodeGenVerilog::VisitExpr_(const NE *op) {
  return MakeBinary(op->type, MakeValue(op->a), MakeValue(op->b), "!=");
}
VerilogValue CodeGenVerilog::VisitExpr_(const LT *op) {
  return MakeBinary(op->type, MakeValue(op->a), MakeValue(op->b), "<");
}
VerilogValue CodeGenVerilog::VisitExpr_(const LE *op) {
  return MakeBinary(op->type, MakeValue(op->a), MakeValue(op->b), "<=");
}
VerilogValue CodeGenVerilog::VisitExpr_(const GT *op) {
  return MakeBinary(op->type, MakeValue(op->a), MakeValue(op->b), ">");
}
VerilogValue CodeGenVerilog::VisitExpr_(const GE *op) {
  return MakeBinary(op->type, MakeValue(op->a), MakeValue(op->b), ">=");
}
VerilogValue CodeGenVerilog::VisitExpr_(const And *op) {
  return MakeBinary(op->type, MakeValue(op->a), MakeValue(op->b), "&&");
}
VerilogValue CodeGenVerilog::VisitExpr_(const Or *op) {
  return MakeBinary(op->type, MakeValue(op->a), MakeValue(op->b), "||");
}
VerilogValue CodeGenVerilog::VisitExpr_(const Not *op) {
  VerilogValue value = MakeValue(op->a);
  std::ostringstream os;
  os << "(!" << value.vid << ")";
  return GetSSAValue(os.str(), op->type);
}

VerilogValue CodeGenVerilog::VisitExpr_(const Call *op) {
  if (op->is_intrinsic(Call::bitwise_and)) {
    return MakeBinary(
        op->type, MakeValue(op->args[0]), MakeValue(op->args[1]), "&");
  } else if (op->is_intrinsic(Call::bitwise_xor)) {
    return MakeBinary(
        op->type, MakeValue(op->args[0]), MakeValue(op->args[1]), "^");
  } else if (op->is_intrinsic(Call::bitwise_or)) {
    return MakeBinary(
        op->type, MakeValue(op->args[0]), MakeValue(op->args[1]), "|");
  } else if (op->is_intrinsic(Call::bitwise_not)) {
    VerilogValue value = MakeValue(op->args[0]);
    std::ostringstream os;
    os << "(~" << value.vid << ")";
    return GetSSAValue(os.str(), op->type);
  } else if (op->is_intrinsic(Call::shift_left)) {
    return MakeBinary(
        op->type, MakeValue(op->args[0]), MakeValue(op->args[1]), "<<");
  } else if (op->is_intrinsic(Call::shift_right)) {
    return MakeBinary(
        op->type, MakeValue(op->args[0]), MakeValue(op->args[1]), ">>");
  } else {
    LOG(FATAL) << "Cannot generate call type " << op->name;
    return VerilogValue();
  }
}

VerilogValue CodeGenVerilog::VisitExpr_(const Let* op) {
  VerilogValue value = MakeValue(op->value);
  CHECK(!var_idmap_.count(op->var.get()));
  var_idmap_[op->var.get()] = value.vid;
  return value;
}

VerilogValue CodeGenVerilog::VisitExpr_(const Ramp* op) {
  LOG(FATAL) << "Ramp: not supported ";
  return VerilogValue();
}

VerilogValue CodeGenVerilog::VisitExpr_(const Broadcast* op) {
  LOG(FATAL) << "Broadcast: not supported ";
  return VerilogValue();
}

VerilogValue CodeGenVerilog::VisitExpr_(const Select* op) {
  LOG(FATAL) << "Select: not supported ";
  return VerilogValue();
}

void CodeGenVerilog::CodeGen(const Pipeline& pipeline) {
  // setup channel map.
  for (auto kv : pipeline->channels) {
    ChannelEntry e; e.block = kv.second;
    cmap_[kv.first.get()] = e;
  }
  for (ComputeBlock stage : pipeline->stages) {
    const Store* store = stage->body.as<Store>();
    CHECK(store);
    const Load* load = store->value.as<Load>();
    if (load) {
      MakeLoadToFIFO(stage, store, load);
    } else {
      MakeStore(stage, store);
    }
  }
  for (const auto& kv : cmap_) {
    MakeChannelUnit(kv.second);
  }
}

CodeGenVerilog::SignalEntry
CodeGenVerilog::MakeLoop(const Array<Stmt>& loop) {
  SignalEntry sig;
  // do not use init signal for now.
  std::string init = "0";
  std::string lp_ready = GetUniqueName("lp_tmp_sig");
  sig.ready = GetUniqueName("loop_ready");
  sig.valid = GetUniqueName("loop_valid");
  PrintLine("// loop logic");
  PrintDecl(lp_ready, kWire, Bool(1));
  PrintDecl(sig.ready, kWire, Bool(1));

  std::string end_loop = lp_ready;
  for (size_t i = loop.size(); i != 0; --i) {
    const For* for_op = loop[i - 1].as<For>();
    int bits = for_op->loop_var.type().bits();
    VerilogValue min = MakeValue(for_op->min);
    VerilogValue extent = MakeValue(for_op->extent);
    CHECK(min.vtype == kConst && extent.vtype == kConst)
        << "Only support constant loop domain";

    std::string vid = AllocVarID(for_op->loop_var.get());
    std::string finish = GetUniqueName(vid + "_finish");
    this->PrintIndent();
    stream <<"`NONSTOP_LOOP(" << vid << ", " << bits << ", " << init
           << ", " << end_loop << ", " << finish
           << ", " << min.vid << ", " << extent.vid << ")\n";
    end_loop = finish;
  }
  if (loop.size() != 0) {
    std::string local_ready = GetUniqueName("lp_tmp_sig");
    this->PrintIndent();
    stream <<"`WRAP_LOOP_ONCE(" << init << ", " << sig.valid
           << ", " << sig.ready << ", " << end_loop << ", " << local_ready << ")\n";
    PrintAssign(lp_ready, local_ready);
  }
  return sig;
}

void CodeGenVerilog::MakeStageInputs(
    const ComputeBlock& block,
    const std::string& enable,
    std::string* out_all_input_valid) {
  std::vector<SignalEntry> sigs;
  sigs.push_back(MakeLoop(block->loop));
  // Input data path.
  PrintLine("// stage inputs");
  for (auto kv : block->inputs) {
    const Var& var = kv.first;
    const StageInput& arg = kv.second;
    std::string vid = AllocVarID(var.get());
    this->PrintDecl(vid, kWire, var.type());
    if (arg->input_type == kGlobalConst ||
        arg->input_type == kLoopVar) {
      PrintAssign(vid, GetVarID(arg->var.get()));
    } else if (arg->input_type == kChannel) {
      std::string vid_valid = GetUniqueName(vid + "_valid");
      std::string vid_ready = GetUniqueName(vid + "_ready");
      this->PrintDecl(vid_valid, kWire, Bool(1));
      this->PrintDecl(vid_ready, kWire, Bool(1));
      ChannelEntry* e = GetChannelInfo(arg->var.get());
      // TODO(tqchen, thierry) add one cache here.
      e->AssignPort("read_data", vid, var.type());
      e->AssignPort("read_valid", vid_valid, Bool(1));
      e->AssignPort("read_ready", vid_ready, Bool(1));
      e->AssignPort("read_addr", "0", Int(1));
      sigs.push_back(SignalEntry{vid_valid, vid_ready});
    } else {
      LOG(FATAL) << "Unknown input type";
    }
  }

  PrintLine("// stage input stall");
  std::string all_input_valid = GetUniqueName("all_input_valid");
  this->PrintDecl(all_input_valid, kWire, Bool(1));
  // forward all valid
  std::vector<std::string> valid_conds;
  for (const SignalEntry& e : sigs) {
    if (e.valid.length() != 0) {
      valid_conds.push_back(e.valid);
    }
  }
  PrintAssignAnd(all_input_valid, valid_conds);
  // input ready signal
  for (size_t i = 0; i < sigs.size(); ++i) {
    if (sigs[i].ready.length() == 0) continue;
    std::vector<std::string> conds = {enable};
    for (size_t j = 0; j < sigs.size(); ++j) {
      if (j != i && sigs[j].valid.length() != 0) {
        conds.push_back(sigs[j].valid);
      }
    }
    PrintAssignAnd(sigs[i].ready, conds);
  }
  *out_all_input_valid = all_input_valid;
}

void CodeGenVerilog::MakeDelay(const std::string& dst,
                               const std::string& src,
                               Type dtype,
                               int delay,
                               const std::string& enable) {
  PrintIndent();
  stream << "`DELAY(" << dst << ", " << src << ", "
         << dtype.bits() << ", " << delay << ", " << enable << ")\n";
}

void CodeGenVerilog::MakeStore(const ComputeBlock& block,
                               const Store* store) {
  std::string all_input_valid;
  std::string enable = GetUniqueName("enable");
  this->PrintDecl(enable, kWire, Bool(1));
  MakeStageInputs(block, enable, &all_input_valid);
  // Data path
  PrintLine("// data path");
  VerilogValue value = MakeValue(store->value);
  VerilogValue index = MakeValue(store->index);
  PrintLine("// control and retiming");
  ChannelEntry* write_entry = GetChannelInfo(store->buffer_var.get());
  // TODO(tqchen, thierry) add delay model from expression.a
  int delay = 2;
  std::string ch_name = write_entry->block->channel->handle_var->name_hint;
  std::string write_addr = GetUniqueName(ch_name + ".write_addr");
  std::string write_ready = GetUniqueName(ch_name + ".write_ready");
  std::string write_valid = GetUniqueName(ch_name + ".write_valid");
  std::string write_data = GetUniqueName(ch_name + ".write_data");
  PrintDecl(write_addr, kWire, store->index.type());
  PrintDecl(write_ready, kWire, Bool(1));
  PrintDecl(write_valid, kWire, Bool(1));
  PrintDecl(write_data, kWire, store->value.type());

  MakeDelay(write_addr, index.vid, store->index.type(), delay, enable);
  MakeDelay(write_data, value.vid, store->value.type(), delay, enable);
  MakeDelay(write_valid, all_input_valid, Bool(1), delay, enable);
  PrintAssign(enable, "!" + write_valid + " || " + write_ready);
  write_entry->AssignPort("write_addr", write_addr, store->index.type());
  write_entry->AssignPort("write_ready", write_ready, Bool(1));
  write_entry->AssignPort("write_valid", write_valid, Bool(1));
  write_entry->AssignPort("write_data", write_data, store->value.type());
  // The triggers
  for (size_t i = 0; i < block->triggers.size(); ++i) {
    SignalTrigger trigger = block->triggers[i];
    CHECK(trigger->predicate.type() == Bool(1));
    ChannelEntry* trigger_ch = GetChannelInfo(trigger->channel_var.get());
    std::string port = trigger_ch->SignalPortName(trigger->signal_index);
    VerilogValue v = MakeValue(trigger->predicate);
    // Assign constant trigger.
    if (v.vtype == kConst) {
      trigger_ch->AssignPort(port, v.vid, Bool(1));
    } else {
      // non-constant trigger
      CHECK_EQ(trigger_ch, write_entry)
          << "Can only triggger conditional event at write channel";
      std::string v_trigger = GetUniqueName(ch_name + "." + port);
      MakeDelay(v_trigger, v.vid, Bool(1), delay, enable);
      write_entry->AssignPort(port, v_trigger, Bool(1));
    }
  }
  stream << "\n";
}

void CodeGenVerilog::MakeLoadToFIFO(const ComputeBlock& block,
                                    const Store* store,
                                    const Load* load) {
  ChannelEntry* write_entry = GetChannelInfo(store->buffer_var.get());
  ChannelEntry* load_entry = GetChannelInfo(load->buffer_var.get());
  std::string all_input_valid;
  std::string enable = GetUniqueName("enable");
  this->PrintDecl(enable, kWire, Bool(1));
  MakeStageInputs(block, enable, &all_input_valid);
  // data path
  PrintLine("// data path");
  VerilogValue index = MakeValue(load->index);
  // control and retiming
  PrintLine("// control and retiming");
  // TODO(tqchen, thierry) add delay model from expression
  int delay = 1;
  std::string read_ch_name = load_entry->block->channel->handle_var->name_hint;
  std::string write_ch_name = write_entry->block->channel->handle_var->name_hint;
  std::string read_addr = GetUniqueName(read_ch_name + ".read_addr");
  std::string read_data = GetUniqueName(read_ch_name + ".read_data");
  std::string read_valid = GetUniqueName(read_ch_name + ".read_valid");
  std::string index_valid = GetUniqueName(read_ch_name + ".index_valid");
  std::string write_ready = GetUniqueName(write_ch_name + ".write_ready");
  std::string data_valid = GetUniqueName(read_ch_name + ".data_valid");
  std::string valid_delay = GetUniqueName(read_ch_name + ".valid_delay");
  PrintDecl(read_addr, kWire, load->index.type());
  PrintDecl(read_data, kWire, load->type);
  PrintDecl(read_valid, kWire, Bool(1));
  PrintDecl(index_valid, kWire, Bool(1));
  PrintDecl(data_valid, kWire, Bool(1));
  MakeDelay(read_addr, index.vid, load->index.type(), delay, enable);
  MakeDelay(index_valid, all_input_valid, Bool(1), delay, enable);
  PrintAssignAnd(data_valid, {read_valid, index_valid});
  // The read ports.
  load_entry->AssignPort("read_addr", read_addr, load->index.type());
  load_entry->AssignPort("read_data", read_data, load->type);
  load_entry->AssignPort("read_valid", read_valid, Bool(1));
  // The write ports.
  write_entry->AssignPort("write_ready", write_ready, Bool(1));
  write_entry->AssignPort("write_data", read_data, load->type);
  write_entry->AssignPort("write_valid", valid_delay, Bool(1));
  write_entry->AssignPort("write_addr", "0", Int(1));
  // The not stall condition.
  PrintAssignAnd(enable, {write_ready, read_valid});
  // The ready signal
  PrintIndent();
  stream << "`BUFFER_READ_VALID_DELAY(" << valid_delay << ", " << data_valid
         << ", " << write_ready << ")\n";
  // The triggers
  for (size_t i = 0; i < block->triggers.size(); ++i) {
    SignalTrigger trigger = block->triggers[i];
    CHECK(trigger->predicate.type() == Bool(1));
    ChannelEntry* trigger_ch = GetChannelInfo(trigger->channel_var.get());
    std::string port = trigger_ch->SignalPortName(trigger->signal_index);
    VerilogValue v = MakeValue(trigger->predicate);
    // Assign constant trigger.
    if (v.vtype == kConst) {
      trigger_ch->AssignPort(port, v.vid, Bool(1));
    } else {
      // non-constant trigger
      CHECK_EQ(trigger_ch, load_entry)
          << "Can only triggger conditional event at load channel";
      std::string v_trigger = GetUniqueName(read_ch_name + "." + port);
      MakeDelay(v_trigger, v.vid, Bool(1), delay, enable);
      load_entry->AssignPort(port, v_trigger, Bool(1));
    }
  }
  stream << "\n";
}

void CodeGenVerilog::MakeChannelUnit(const ChannelEntry& ch) {
  if (ch.block->read_window == 0) {
    // This is a memory map
    MakeChannelMemMap(ch);
  } else if (ch.block->read_window == 1 &&
             ch.block->write_window == 1) {
    MakeChannelFIFO(ch);
  } else {
    // general Buffer
    MakeChannelBuffer(ch);
  }
}

void CodeGenVerilog::MakeChannelMemMap(const ChannelEntry& ch) {
  Var ch_var = ch.block->channel->handle_var;
  std::string dut = GetUniqueName(ch_var->name_hint + ".mmap");
  std::string mmap_addr = GetVarID(ch_var.get());

  tvm_vpi_modules_.push_back(dut);
  if (ch.ports.count("read_addr")) {
    CHECK(!ch.ports.count("write_addr"))
        << "Cannot read/write to same RAM";
    const PortEntry& read_addr = ch.GetPort("read_addr");
    const PortEntry& read_data = ch.GetPort("read_data");
    const PortEntry& read_valid = ch.GetPort("read_valid");
    stream << "  // channel setup for " << ch_var << "\n"
           << "  tvm_vpi_read_mmap # (\n"
           << "   .DATA_WIDTH(" << read_data.dtype.bits() << "),\n"
           << "   .ADDR_WIDTH(" << read_addr.dtype.bits() << "),\n"
           << "   .BASE_ADDR_WIDTH(" << ch_var.type().bits() << ")\n"
           << "  ) " << dut << " (\n"
           << "   .clk(clk),\n"
           << "   .rst(rst),\n"
           << "   .addr(" << read_addr.value << "),\n"
           << "   .data_out(" << read_data.value << "),\n"
           << "   .mmap_addr(" << mmap_addr << ")\n"
           << "  );\n";
    PrintAssign(read_valid.value, "1");
  } else if (ch.ports.count("write_addr")) {
    const PortEntry& write_addr = ch.GetPort("write_addr");
    const PortEntry& write_data = ch.GetPort("write_data");
    const PortEntry& write_valid = ch.GetPort("write_valid");
    const PortEntry& write_ready = ch.GetPort("write_ready");
    stream << "  // channel setup for " << ch_var << "\n"
           << "  tvm_vpi_write_mmap # (\n"
           << "   .DATA_WIDTH(" << write_data.dtype.bits() << "),\n"
           << "   .ADDR_WIDTH(" << write_addr.dtype.bits() << "),\n"
           << "   .BASE_ADDR_WIDTH(" << ch_var.type().bits() << ")\n"
           << "  ) " << dut << " (\n"
           << "   .clk(clk),\n"
           << "   .rst(rst),\n"
           << "   .addr(" << write_addr.value << "),\n"
           << "   .data_in(" << write_data.value << "),\n"
           << "   .en(" << write_valid.value << "),\n"
           << "   .mmap_addr(" << mmap_addr << ")\n"
           << "  );\n";
    PrintAssign(write_ready.value, "1");
    // additional control signals
    for (size_t i = 0; i < ch.block->ctrl_signals.size(); ++i) {
      ControlSignal sig = ch.block->ctrl_signals[i];
      CHECK_EQ(sig->ctrl_type, kComputeFinish);
      std::string port = ch.SignalPortName(i);
      done_sigs_.push_back(ch.GetPort(port).value);
    }
  }
}

void CodeGenVerilog::MakeChannelFIFO(const ChannelEntry& ch) {
  Var ch_var = ch.block->channel->handle_var;
  std::string dut = GetUniqueName(ch_var->name_hint + ".fifo_reg");

  const PortEntry& write_data = ch.GetPort("write_data");
  const PortEntry& write_valid = ch.GetPort("write_valid");
  const PortEntry& write_ready = ch.GetPort("write_ready");

  const PortEntry& read_data = ch.GetPort("read_data");
  const PortEntry& read_valid = ch.GetPort("read_valid");
  const PortEntry& read_ready = ch.GetPort("read_ready");

  CHECK_EQ(write_data.dtype, read_data.dtype);

  stream << "  // channel setup for " << ch_var << "\n"
         << "  `CACHE_REG(" << write_data.dtype.bits()
         << ", " << write_data.value
         << ", " << write_valid.value
         << ", " << write_ready.value
         << ", " << read_data.value
         << ", " << read_valid.value
         << ", " << read_ready.value
         << ")\n";
}

void CodeGenVerilog::MakeChannelBuffer(const ChannelEntry& ch) {
  LOG(FATAL) << "not implemeneted";
}

CodeGenVerilog::ChannelEntry*
CodeGenVerilog::GetChannelInfo(const Variable* var) {
  auto it = cmap_.find(var);
  CHECK(it != cmap_.end())
      << "cannot find channel for var " << var->name_hint;
  return &(it->second);
}

void CodeGenVerilog::ChannelEntry::AssignPort(
    std::string port, std::string value, Type dtype) {
  CHECK(!ports.count(port))
      << "port " << port
      << " of channel " << block->channel << " has already been connected";
  ports[port] = PortEntry{value, dtype};
}

const CodeGenVerilog::PortEntry&
CodeGenVerilog::ChannelEntry::GetPort(const std::string& port) const {
  auto it = ports.find(port);
  CHECK(it != ports.end())
      << "port " << port
      << " of channel " << block->channel << " has not been connected";
  return it->second;
}

std::string CodeGenVerilog::ChannelEntry::SignalPortName(int index) const {
  CHECK_LT(static_cast<size_t>(index), block->ctrl_signals.size());
  std::ostringstream os;
  os << "ctrl_port" << index;
  return os.str();
}
}  // namespace verilog
}  // namespace codegen
}  // namespace tvm
