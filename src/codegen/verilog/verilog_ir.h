/*!
 *  Copyright (c) 2017 by Contributors
 * \file verilog_ir.h
 * \brief A lowered IR that resembles verilog blocks,
 *   This is data structure before final codegen.
 */
#ifndef TVM_CODEGEN_VERILOG_VERILOG_IR_H_
#define TVM_CODEGEN_VERILOG_VERILOG_IR_H_

#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/channel.h>
#include <tvm/lowered_func.h>
#include <vector>
#include <memory>
#include <unordered_map>

namespace tvm {
namespace codegen {
namespace verilog {

/*! \brief The data argument type */
enum StageInputType : int {
  /*! \brief Data channel input. */
  kChannel,
  /*! \brief Loop variable generated by compute block. */
  kLoopVar,
  /*! \brief Global constant. */
  kGlobalConst
};

/*! \brief The data argument type */
enum ControlSignalType : int {
  // Read advance signal
  kReadAdvance,
  // Write advance signal
  kWriteAdvance,
  // Pipeline stage finish signal
  kComputeFinish
};

class ControlSignal;
class StageInput;
class SignalTrigger;

/*! \brief The control signal of a channel */
struct ControlSignalNode : public Node {
  /*! \brief The control signal type */
  ControlSignalType ctrl_type;
  /*! \brief Advance size of the signal */
  int advance_size{0};
  // visit all attributes
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("ctrl_type", &ctrl_type);
    v->Visit("advance_size", &advance_size);
  }
  static ControlSignal make(ControlSignalType ctrl_type, int advance_size);
  static constexpr const char* _type_key = "VerilogControlSignal";
  TVM_DECLARE_NODE_TYPE_INFO(ControlSignalNode, Node);
};

TVM_DEFINE_NODE_REF(ControlSignal, ControlSignalNode);

/*! \brief Information about channel. */
struct ChannelBlockNode : public Node {
  /*! \brief The channel we are refer to */
  Channel channel;
  /*! \brief Read window */
  int read_window{0};
  /*! \brief Write window */
  int write_window{0};
  /*! \brief Control signals in the channel */
  Array<ControlSignal> ctrl_signals;
  // visit all attributes
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("channel", &channel);
    v->Visit("read_window", &read_window);
    v->Visit("write_window", &write_window);
    v->Visit("ctrl_signals", &ctrl_signals);
  }
  static constexpr const char* _type_key = "VerilogChannelBlock";
  TVM_DECLARE_NODE_TYPE_INFO(ChannelBlockNode, Node);
};

TVM_DEFINE_NODE_REF(ChannelBlock, ChannelBlockNode);

/*!
 * \brief Input to the compute block.
 *  These represents the data values that need to be shared;
 */
struct StageInputNode : public Node {
  /*!
   * \brief The corresponding var of the input
   *  For loop and global const it is the var.
   *  For channel this corresponds to the channel handle.
   */
  Var var;
  /*! \brief The type of the input. */
  StageInputType input_type;
  // visit all attributes
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("var", &var);
    v->Visit("input_type", &input_type);
  }
  // constructor
  static StageInput make(Var var, StageInputType input_type);
  static constexpr const char* _type_key = "VerilogStageInput";
  TVM_DECLARE_NODE_TYPE_INFO(StageInputNode, Node);
};

TVM_DEFINE_NODE_REF(StageInput, StageInputNode);

/*! \brief The trigger signal for certain channel */
struct SignalTriggerNode : public Node {
  /*! \brief The channel handle variable */
  Var channel_var;
  /*! \brief Boolean predicate to trigger the signal */
  Expr predicate;
  /*! \brief siginal index of the channel */
  int signal_index;
  // visit all attributes
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("channel_var", &channel_var);
    v->Visit("predicate", &predicate);
    v->Visit("signal_index", &signal_index);
  }
  // constructor
  static constexpr const char* _type_key = "VerilogSignalTrigger";
  TVM_DECLARE_NODE_TYPE_INFO(SignalTriggerNode, Node);
};

TVM_DEFINE_NODE_REF(SignalTrigger, SignalTriggerNode);

/*! \brief compute block for verilog */
struct ComputeBlockNode : public Node {
  /*! \brief The body of the block. */
  Stmt body;
  /*! \brief The loop nest around the body, each is a For with no_op as body */
  Array<Stmt> loop;
  /*! \brief The channel advance trigger */
  Array<SignalTrigger> triggers;
  /*! \brief The input variables that need to be synced. */
  Map<Var, StageInput> inputs;
  // visit all attributes
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("body", &body);
    v->Visit("loop", &loop);
    v->Visit("triggers", &triggers);
    v->Visit("inputs", &inputs);
  }

  static constexpr const char* _type_key = "VerilogComputeBlock";
  TVM_DECLARE_NODE_TYPE_INFO(ComputeBlockNode, Node);
};

TVM_DEFINE_NODE_REF(ComputeBlock, ComputeBlockNode);

/*! \brief Codeblock for verilog module. */
struct PipelineNode : public Node {
  /*! \brief arguments to the module */
  Array<Var> args;
  /*! \brief Computation stages */
  Array<ComputeBlock> stages;
  /*! \brief The data channels */
  Map<Var, ChannelBlock> channels;

  // visit all attributes
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("args", &args);
    v->Visit("stages", &stages);
    v->Visit("channels", &channels);
  }
  static constexpr const char* _type_key = "VerilogPipeline";
  TVM_DECLARE_NODE_TYPE_INFO(PipelineNode, Node);
};

TVM_DEFINE_NODE_REF(Pipeline, PipelineNode);

/*!
 * \brief Build a lowered verilog pipeline given function.
 * \param f The function to be transformed.
 * \param The created verilog pipeline.
 */
Pipeline MakePipeline(LoweredFunc f);
}  // namespace verilog
}  // namespace codegen
}  // namespace tvm
#endif  // TVM_CODEGEN_VERILOG_VERILOG_IR_H_
