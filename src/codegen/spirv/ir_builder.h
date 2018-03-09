/*!
 *  Copyright (c) 2018 by Contributors
 * \file ir_builder.h
 * \brief Utility for building SPIRV code block
 */
#ifndef TVM_CODEGEN_SPIRV_IR_BUILDER_H_
#define TVM_CODEGEN_SPIRV_IR_BUILDER_H_

#include <tvm/runtime/packed_func.h>
#include <tvm/ir.h>

#include <algorithm>
#include <utility>
#include <vector>
#include <string>
#include <map>

#include <vulkan/spirv.hpp>

namespace tvm {
namespace codegen {
namespace spirv {

/*! \brief Represent the SPIRV Type */
struct SType {
  /*! \brief The Id to represent type */
  uint32_t id{0};
  /*! \brief corresponding TVM type */
  tvm::Type type;
  /*! \brief content type id if it is a pointer/struct-array class */
  uint32_t element_type_id{0};
  /*! \brief The storage class, if it is a pointer */
  spv::StorageClass storage_class{spv::StorageClassMax};
};

enum ValueKind {
  kNormal,
  kConstant,
  kVectorPtr,
  kStructArrayPtr,
  kPushConstantPtr,
  kFunction,
  kExtInst
};

/*! \brief Represent the SPIRV Value */
struct Value {
  /*! \brief The Id to represent value */
  uint32_t id{0};
  /*! \brief The data type */
  SType stype;
  /*! \brief additional flags about the value */
  ValueKind flag{kNormal};
};

/*! \brief Represent the SPIRV Label */
struct Label {
  /*! \brief The Id to represent label */
  uint32_t id{0};
};

/*!
 * \brief A SPIRV instruction,
 *  can be used as handle to modify its content later
 */
class Instr {
 public:
  /*! \return the word count */
  uint32_t WordCount() const {
    return word_count_;
  }
  /*!
   * \brief Access idx-th word of instruction
   * \param idx The index
   * \return reference to idx-th word.
   */
  uint32_t& operator[](uint32_t idx) {
    CHECK_LT(idx, word_count_);
    return (*data_)[begin_ + idx];
  }

 private:
  friend class InstrBuilder;
  /*!
   * \brief the data that backs this instruction
   *  Have to use vector reference because
   *  vector can change.
   */
  std::vector<uint32_t>* data_{nullptr};
  /*! \brief begin location of instruction */
  uint32_t begin_{0};
  /*! \brief work count */
  uint32_t word_count_{0};
};

/*! \brief Representation of phi value */
struct PhiValue : public Value {
  /*! \brief The corresponding instr */
  Instr instr;
  /*!
   * \brief Add incoming information of a PhiValue
   * \param index The location of Phi
   * \param value The value to come
   * \param parent The parent label.
   */
  void SetIncoming(uint32_t index,
                   const Value& value,
                   const Label& parent) {
    CHECK_EQ(this->stype.id, value.stype.id);
    instr[3 + index * 2] = value.id;
    instr[3 + index * 2 + 1] = parent.id;
  }
};

/*!
 * \brief Helper class to build SPIRV instruction.
 *
 * \code
 *
 *  std::vector<uint32_t> func_seg_vec_;
 *  InstrBuilder ib;
 *
 *  // construct and append to the end of func_seg_vec_;
 *  ib.Begin(spv::OpIAdd)
 *    .Add(result).Add(v1).Add(v2)
 *    .Commit(&func_seg_vec_);
 *
 * \endcode
 */
class InstrBuilder {
 public:
  /*!
   * \brief Begin construction of instruction.
   * \param op The op code
   * \return reference to self.
   */
  InstrBuilder& Begin(spv::Op op) {  // NOLINT(*);
    // finish previous build
    CHECK_EQ(data_.size(), 0U);
    op_ = op;
    data_.push_back(0);
    return *this;
  }
  /*!
   * \brief Add v to end of instruction.
   * \param v The value to be appended to the instruction.
   * \return reference to self.
   */
  InstrBuilder& Add(const Value& v) {
    data_.push_back(v.id);
    return *this;
  }
  /*!
   * \brief Add v to end of instruction.
   * \param v The type to be appended to the instruction.
   * \return reference to self.
   */
  InstrBuilder& Add(const SType& v) {
    data_.push_back(v.id);
    return *this;
  }
  /*!
   * \brief Add v to end of instruction.
   * \param v The label to be appended to the instruction.
   * \return reference to self.
   */
  InstrBuilder& Add(const Label& v) {
    data_.push_back(v.id);
    return *this;
  }
  /*!
   * \brief Add a word to end of instruction.
   * \param v The value to be added.
   * \return reference to self.
   */
  InstrBuilder& Add(const uint32_t& v) {
    data_.push_back(v);
    return *this;
  }
  /*!
   * \brief Add string literal of end of instruction.
   * \param v The string literal to be appended.
   * \return reference to self.
   */
  InstrBuilder& Add(const std::string& v) {
    const uint32_t kWordSize = sizeof(uint32_t);
    uint32_t nwords =
        (static_cast<uint32_t>(v.length()) + kWordSize) / kWordSize;
    size_t begin = data_.size();
    data_.resize(begin + nwords, 0U);
    std::copy(v.begin(), v.end(),
              reinterpret_cast<char*>(&data_[begin]));
    return *this;
  }
  /*!
   * \brief add sequence of values to instruction
   * \param args The instruction sequence
   * \return reference to self.
   * \tparams Args The positional arguments
   */
  template<typename... Args>
  InstrBuilder& AddSeq(Args&& ...args) {
    AddSeqHelper helper;
    helper.builder = this;
    runtime::detail::for_each(helper, std::forward<Args>(args)...);
    return *this;
  }
  /*!
   * \brief Finish build, commit the current
   *   instruction to the end of seg.
   *
   * \param seg The code segment to commit to
   * \return The result instruction.
   */
  Instr Commit(std::vector<uint32_t>* seg) {
    Instr ret;
    ret.data_ = seg;
    ret.begin_ = seg->size();
    ret.word_count_ = static_cast<uint32_t>(data_.size());
    data_[0] = op_ | (ret.word_count_ << spv::WordCountShift);
    seg->insert(seg->end(), data_.begin(), data_.end());
    data_.clear();
    return ret;
  }

 private:
  // current op code.
  spv::Op op_;
  // The internal data to store code
  std::vector<uint32_t> data_;
  // helper class to support variadic arguments
  struct AddSeqHelper {
    // The reference to builder
    InstrBuilder* builder;
    // invoke function
    template<typename T>
    void operator()(size_t, const T& v) const {
      builder->Add(v);
    }
  };
};

/*!
 * \brief Builder to build up a single SPIR-V module
 *
 *  This is a thin wrapper to build SPIRV binary.
 *  SPIRV adopts structure control-flow.
 *  We can build the code by always appending to the end of the
 *  binary code block and revisit some
 *
 *  This IRBuilder did not introduce concept of BasicBlock.
 *  instead instructions are append to end of each segment.
 */
class IRBuilder {
 public:
  /*! \brief Initialize header */
  void InitHeader();
  /*! \brief Initialize the predefined contents */
  void InitPreDefs();
  /*!
   * \brief Import additional extension libraries.
   * \param name The name of the library.
   * \return The finalized binary instruction.
   */
  Value ExtInstImport(const std::string& name) {
    Value val = NewValue(SType(), kExtInst);
    ib_.Begin(spv::OpExtInstImport).AddSeq(val, name).Commit(&header_);
    return val;
  }
  /*!
   * \brief Get the final binary built from the builder
   * \return The finalized binary instruction.
   */
  std::vector<uint32_t> Finalize() {
    std::vector<uint32_t> data;
    // set bound
    const int kBoundLoc = 3;
    header_[kBoundLoc] = id_counter_;
    data.insert(data.end(), header_.begin(), header_.end());
    data.insert(data.end(), entry_.begin(), entry_.end());
    data.insert(data.end(), exec_mode_.begin(), exec_mode_.end());
    data.insert(data.end(), debug_.begin(), debug_.end());
    data.insert(data.end(), decorate_.begin(), decorate_.end());
    data.insert(data.end(), global_.begin(), global_.end());
    data.insert(data.end(), function_.begin(), function_.end());
    return data;
  }
  /*!
   * \brief Create new label
   * \return The created new label
   */
  Label NewLabel() {
    Label label;
    label.id = id_counter_++;
    return label;
  }
  /*!
   * \brief Start a new block with given label
   * \param label The label we use.
   */
  void StartLabel(Label label) {
    MakeInst(spv::OpLabel, label);
    curr_label_ = label;
  }
  /*! \return The current label */
  Label CurrentLabel() const {
    return curr_label_;
  }
  /*!
   * \brief Add code to debug segment.
   * \param op The operator
   * \param args The instruction sequence
   * \tparams Args The positional arguments
   */
  template<typename... Args>
  void Debug(spv::Op op, Args&& ...args) {
    ib_.Begin(op).AddSeq(std::forward<Args>(args)...).Commit(&debug_);
  }
  /*!
   * \brief Add Execution mode to a function.
   * \param func The function value
   * \param args The instruction sequence
   * \tparams Args The positional arguments
   */
  template<typename... Args>
  void ExecutionMode(Value func, Args&& ...args) {
    ib_.Begin(spv::OpExecutionMode).AddSeq(
        func, std::forward<Args>(args)...).Commit(&exec_mode_);
  }
  /*!
   * \brief Add code to decorate segment.
   * \param op The operator
   * \param args The instruction sequence
   * \tparams Args The positional arguments
   */
  template<typename... Args>
  void Decorate(spv::Op op, Args&& ...args) {
    ib_.Begin(op).AddSeq(std::forward<Args>(args)...).Commit(&decorate_);
  }
  /*!
   * \brief Add code to global segment.
   * \param op The operator
   * \param args The instruction sequence
   * \tparams Args The positional arguments
   */
  template<typename... Args>
  void DeclareGlobal(spv::Op op, Args&& ...args) {
    ib_.Begin(op).AddSeq(std::forward<Args>(args)...).Commit(&decorate_);
  }
  /*!
   * \brief Make a new instruction and append it to end of function segment.
   *
   * \param op The operator
   * \param args The instruction sequence
   * \return The result SSA value.
   * \tparams Args The positional arguments
   */
  template<typename... Args>
  Instr MakeInst(spv::Op op, Args&& ...args) {
    return ib_.Begin(op).AddSeq(std::forward<Args>(args)...).Commit(&function_);
  }
  /*!
   * \brief Make a new SSA value,
   *
   * \param op The operator.
   * \param out_type The result type.
   * \param args The instruction sequence
   * \return The result SSA value.
   * \tparams Args The positional arguments
   */
  template<typename... Args>
  Value MakeValue(spv::Op op, const SType& out_type, Args&& ...args) {
    Value val = NewValue(out_type, kNormal);
    MakeInst(op, out_type, val, std::forward<Args>(args)...);
    return val;
  }
  /*!
   * \brief Make a phi value.
   *
   * \param out_type The output data type.
   * \param num_incoming number of incoming blocks.
   * \return The result Phi value.
   */
  PhiValue MakePhi(const SType& out_type, uint32_t num_incoming);
  /*!
   * \brief Create a GLSL450 call
   *
   * \param ret_type The result type.
   * \param inst_id The instance id of the function.
   * \param args The arguments
   * \return The result value.
   */
  Value CallGLSL450(const SType& ret_type,
                    uint32_t inst_id,
                    const std::vector<Value>& args);
  /*!
   * \brief Build vector by concatenating components
   *
   * \param vec The vector component
   * \tparams Args The positional arguments
   */
  Value Concat(const std::vector<Value>& vec);
  /*!
   * \brief Get the spirv type for a given tvm data type.
   * \param dtype The data type.
   * \return The corresponding spirv type.
   */
  SType GetSType(const tvm::Type& dtype);
  /*!
   * \brief Get the pointer type that points to value_type
   * \param value_type.
   * \param storage_class The storage class
   * \return The corresponding spirv type.
   */
  SType GetPointerType(const SType& value_type,
                       spv::StorageClass storage_class);
  /*!
   * \brief Get a struct{ value_type[num_elems] } type.
   * \param value_type the content value type.
   * \param num_elems number of elements in array
   *   num_elems = 0 means runtime array with BufferBlock Decoration
   *
   * \return The corresponding spirv type.
   */
  SType GetStructArrayType(const SType& value_type,
                           uint32_t num_elems);
  /*!
   * \brief Get a struct array access with a given index.
   * \param ptr_type The pointer type.
   * \param buffer The buffer ptr to struct array
   * \param index The array index.
   */
  Value StructArrayAccess(const SType& ptr_type,
                          Value buffer,
                          Value index);
  /*!
   * \brief Create a cast that cast value to dst_type
   * \param dst_type The target type.
   * \param value the source value.
   * \return The result value
   */
  Value Cast(const SType& dst_type, Value value);
  /*
   * \brief Create a const integer.
   * \param dtype The content data type.
   * \param value The data value.
   */
  Value IntImm(const SType& dtype, int64_t value);
  /*
   * \brief Create a const unsigned integer.
   * \param dtype The content data type.
   * \param value The data value.
   */
  Value UIntImm(const SType& dtype, uint64_t value);
  /*
   * \brief Create a const float.
   * \param dtype The content data type.
   * \param value The data value.
   */
  Value FloatImm(const SType& dtype, double value);
  /*
   * \brief Declare buffer argument of function
   *
   * \param arg_type The type of argument.
   * \param descriptor_set The descriptor set we want to use.
   * \param binding The binding locaiton in descriptor set.
   * \param The argument type.
   */
  Value BufferArgument(const SType& value_type,
                       uint32_t descriptor_set,
                       uint32_t binding);
  /*!
   * \brief Declare POD arguments through push constants.
   *
   * \note Only call this function once!
   * \param value_types The values in the push constant
   * \return reference to self.
   */
  Value DeclarePushConstant(const std::vector<SType>& value_types);
  /*!
   * \brief Get i-th push constant
   * \param v_type The value type
   * \param index The push constant index
   * \return the value of push constant
   */
  Value GetPushConstant(Value ptr_push_const, const SType& v_type, uint32_t index);
  /*!
   * \brief Declare a kernel function
   * \param name Name of the entry point.
   * \return The created function ID.
   */
  Value DeclareKenrelFunction(const std::string& name);
  /*!
   * \brief Start function scope.
   * \param func function to be started.
   */
  void StartFunction(const Value& func);
  /*!
   * \brief Set the local size of the function
   * \param func function of interest
   * \param local_size The local workgroup_size
   */
  void SetLocalSize(const Value& func, uint32_t local_size[3]);
  /*
   * \brief Allocate space
   * \param value_type The content value type
   * \param num_elems Number of elements to allocate.
   * \param storage_class The storage class we want to store to.
   */
  Value Allocate(const SType& value_type,
                 uint32_t num_elems,
                 spv::StorageClass storage_class);
  /*
   * \brief Get the i-th workgroup id.
   * \return The value representing the workgroup id.
   */
  Value GetWorkgroupID(uint32_t dim_index);
  /*
   * \brief Get the i-th local id.
   * \return The value representing the local id.
   */
  Value GetLocalID(uint32_t dim_index);
  // Expressions
  Value Add(Value a, Value b);
  Value Sub(Value a, Value b);
  Value Mul(Value a, Value b);
  Value Div(Value a, Value b);
  Value Mod(Value a, Value b);
  Value EQ(Value a, Value b);
  Value NE(Value a, Value b);
  Value LT(Value a, Value b);
  Value LE(Value a, Value b);
  Value GT(Value a, Value b);
  Value GE(Value a, Value b);
  Value Select(Value cond, Value a, Value b);

 private:
  /*!
   * \brief Create new value
   * \return The created new label
   */
  Value NewValue(const SType& stype, ValueKind flag) {
    Value val;
    val.id = id_counter_++;
    val.stype = stype;
    val.flag = flag;
    return val;
  }
  // get constant given value encoded in uint64_t
  Value GetConst_(const SType& dtype, const uint64_t* pvalue);
  // declare type
  SType DeclareType(const Type& dtype);
  /*! \brief internal instruction builder  */
  InstrBuilder ib_;
  /*! \brief Current label */
  Label curr_label_;
  /*! \brief The current maximum id */
  uint32_t id_counter_{1};
  /*! \brief glsl 450 extension */
  Value ext_glsl450_;
  /*! \brief Special cache int32, fp32, void*/
  SType t_bool_, t_int32_, t_uint32_, t_fp32_, t_void_, t_void_func_;
  /*! \brief quick cache for const one i32 */
  Value const_i32_zero_;
  /*! \brief cache value for workgroup_id, local_id */
  Value workgroup_id_, local_id_;
  /*! \brief whether push constant is defined */
  Value push_const_;
  /*! \brief map from type code to the type */
  std::unordered_map<uint32_t, SType> pod_type_tbl_;
  /*! \brief map from value to array type */
  std::map<std::pair<uint32_t, uint32_t>, SType> struct_array_type_tbl_;
  /*! \brief map from value to its pointer type */
  std::map<std::pair<uint32_t, spv::StorageClass>, SType> pointer_type_tbl_;
  /*! \brief map from constant int to its value */
  std::map<std::pair<uint32_t, uint64_t>, Value> const_tbl_;
  /*! \brief Header segment, include import */
  std::vector<uint32_t> header_;
  /*! \brief engtry point segment */
  std::vector<uint32_t> entry_;
  /*! \brief Header segment */
  std::vector<uint32_t> exec_mode_;
  /*! \brief Debug segment */
  std::vector<uint32_t> debug_;
  /*! \brief Annotation segment */
  std::vector<uint32_t> decorate_;
    /*! \brief Global segment: types, variables, types */
  std::vector<uint32_t> global_;
  /*! \brief Function segment */
  std::vector<uint32_t> function_;
};

}  // namespace spirv
}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_SPIRV_IR_BUILDER_H_
