#include "relay_mlir_translate.h"
#include "src/ir/relay_ops.h"
#include <iostream>
#include <iterator>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/StandardTypes.h>
#include <string>
#include <tvm/node/container.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/reduce.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/op.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/ndarray.h>

namespace mlir {
namespace mrelay {

namespace {

template <typename T>
LogicalResult ConvertFloatElementsAttr(const ElementsAttr attr,
                                       tvm::runtime::NDArray *values) {
  auto shape = attr.getType().getShape();
  *values = tvm::runtime::NDArray::Empty(shape, {kDLFloat, sizeof(T) * 8, 1},
                                         {kDLCPU, 0});

  int64_t num_elements = 1;
  for (auto dim : shape)
    num_elements *= dim;

  auto elts = attr.dyn_cast<DenseFPElementsAttr>();
  if (!elts)
    return mlir::failure();

  if (elts.isSplat()) {
    std::vector<T> attr_values(num_elements, elts.getSplatValue<T>());
    values->CopyFromBytes(attr_values.data(), sizeof(T) * num_elements);
  } else {
    // TODO: Get rid of excessive copies.
    auto attr_values_range = elts.getValues<T>();
    std::vector<T> attr_values(attr_values_range.begin(),
                               attr_values_range.end());
    values->CopyFromBytes(attr_values.data(), sizeof(T) * num_elements);
  }
  return mlir::success();
}

template <typename T>
LogicalResult ConvertIntElementsAttr(const ElementsAttr attr,
                                     tvm::runtime::NDArray *values) {
  auto shape = attr.getType().getShape();
  *values = tvm::runtime::NDArray::Empty(shape, {kDLInt, sizeof(T) * 8, 1},
                                         {kDLCPU, 0});

  int64_t num_elements = 1;
  for (auto dim : shape)
    num_elements *= dim;

  auto elts = attr.dyn_cast<DenseFPElementsAttr>();
  if (!elts)
    return mlir::failure();

  if (elts.isSplat()) {
    std::vector<T> attr_values(num_elements, elts.getSplatValue<T>());
    values->CopyFromBytes(attr_values.data(), sizeof(T) * num_elements);
  } else {
    // TODO: Get rid of excessive copies.
    auto attr_values_range = elts.getValues<T>();
    std::vector<T> attr_values(attr_values_range.begin(),
                               attr_values_range.end());
    values->CopyFromBytes(attr_values.data(), sizeof(T) * num_elements);
  }
  return mlir::success();
}

LogicalResult ConvertElementsAttrToNDArray(const Attribute attr,
                                           tvm::runtime::NDArray *values) {
  return mlir::failure();
}

LogicalResult ConvertElementsAttrToNDArray(const ElementsAttr attr,
                                           tvm::runtime::NDArray *values) {
  auto type = attr.getType(); //ShapedType
  auto ele_type = type.getElementType(); // Type
  if (ele_type.isF32()) {
    return ConvertFloatElementsAttr<float>(attr, values);
  } else if (ele_type.isF64()) {
    return ConvertFloatElementsAttr<double>(attr, values);
  } else if (ele_type.isSignedInteger() || ele_type.isUnsignedInteger()) {
    switch (type.getElementTypeBitWidth()) {
    case 8:
      return ConvertIntElementsAttr<int8_t>(attr, values);
    case 16:
      return ConvertIntElementsAttr<int16_t>(attr, values);
    case 32:
      return ConvertIntElementsAttr<int32_t>(attr, values);
    case 64:
      return ConvertIntElementsAttr<int64_t>(attr, values);
    default:
      return mlir::failure();
    }
  }
  return mlir::failure();
}

LogicalResult
MLIRBasicTypeToRelayBasicType(Type type, tvm::runtime::DataType *relay_type) {
  if (type.isF16()) {
    *relay_type = tvm::runtime::DataType::Float(16);
    return mlir::success();
  } else if (type.isF32()) {
    *relay_type = tvm::runtime::DataType::Float(32);
    return mlir::success();
  } else if (type.isF64()) {
    *relay_type = tvm::runtime::DataType::Float(64);
    return mlir::success();
  } else if (type.isBF16()) {
    // TODO: Support BF16.
    return mlir::failure();
  } else if (type.isSignlessInteger() || type.isUnsignedInteger()) {
    const auto &itype = type.cast<mlir::IntegerType>();
    *relay_type = tvm::runtime::DataType::Int(itype.getWidth());
    return mlir::success();
  }
  return mlir::failure();
}

LogicalResult
MLIRTensorTypeToRelayTensorType(Type type, tvm::relay::TensorType *relay_type) {
  if (auto stype = type.dyn_cast<ShapedType>()) {
    // If type represents a non-scalar tensor.
    // Only fully defined static shapes are supported now.
    // TODO: Support dynamic shapes.
    if (!stype.hasStaticShape())
      return mlir::failure();

    tvm::runtime::DataType relay_data_type;
    auto type_res =
        MLIRBasicTypeToRelayBasicType(stype.getElementType(), &relay_data_type);
    if (failed(type_res))
      return type_res;

    std::vector<int32_t> shape_vec;
    tvm::Array<tvm::relay::IndexExpr> shapes;
    for (auto dim : stype.getShape()) {
      // TODO: tvm::tir::Any() for dynamic dim
      shapes.push_back(static_cast<int32_t>(dim));
    }
    *relay_type = tvm::relay::TensorType(shapes, relay_data_type);

    return mlir::success();
  }

  // If type represents a scalar.
  tvm::runtime::DataType relay_data_type;
  auto type_res = MLIRBasicTypeToRelayBasicType(type, &relay_data_type);
  if (failed(type_res))
    return type_res;

  *relay_type = tvm::relay::TensorType::Scalar(relay_data_type);
  return mlir::success();
}

// Create Relay Constant op with given data.
template <typename T> tvm::relay::Expr CreateConstant(std::vector<T> &data) {
  tvm::runtime::NDArray constant_data;
  constant_data.CopyFromBytes(data.data(), data.size() * sizeof(T));
  return tvm::relay::Constant(constant_data);
}

LogicalResult ConvertConstOp(mlir::Operation *op, tvm::relay::Expr *relay_op) {
  auto const_op = mlir::dyn_cast<mrelay::ConstOp>(op);
  if (!const_op)
    return mlir::failure();

  auto value_attr = const_op.value();
  tvm::runtime::NDArray value_nd_array;
  auto res = ConvertElementsAttrToNDArray(value_attr, &value_nd_array);
  if (failed(res))
    return res;

  *relay_op = tvm::relay::Constant(value_nd_array);
  return mlir::success();
}

LogicalResult ConvertAddOp(mlir::Operation *op, tvm::relay::Expr *relay_op,
                           std::vector<tvm::relay::Var> *input_vars) {
  // conversion for "mrelay.add"
  auto add_op = mlir::dyn_cast<mrelay::AddOp>(op);
  if (!add_op)
    return mlir::failure();

  std::vector<tvm::relay::Var> &current_input_vars = *input_vars;
  const tvm::relay::Op &relay_add = tvm::relay::Op::Get("add");
  *relay_op = tvm::relay::Call(relay_add,
                               {current_input_vars[0], current_input_vars[1]},
                               tvm::Attrs(), {});
  return mlir::success();
}

LogicalResult ConvertMLIROpToRelay(mlir::Operation *op,
                                   tvm::relay::Expr *relay_op,
                                   std::vector<tvm::relay::Var> *input_vars) {
  auto op_name = op->getName().getStringRef();
  if (op_name == "mrelay.const") {
    return ConvertConstOp(op, relay_op);
  } else if (op_name == "mrelay.add") {
    return ConvertAddOp(op, relay_op, input_vars);
  } else {
    return mlir::failure();
    // TODO (yongwww): Add conversion for other ops.
    // Find a better way, for example using the op name to automatically
    // find corresponding relay build method.
  }
}

LogicalResult ConvertMLIRFuncToRelayFunc(mlir::FuncOp mlir_func_op,
                                         tvm::relay::Function *relay_func) {
  // Check that there is only 1 basic block in func. We do not support
  // control flow other than functionalized ones, so there shouldn't be
  // more than 1 block.
  auto &blocks = mlir_func_op.getBlocks();
  if (blocks.size() != 1)
    return mlir::failure();

  // Map from values to corresponding expressions.
  std::unordered_map<Value *, tvm::relay::Expr> values_to_exprs;

  std::vector<tvm::relay::Var> input_vars;
  input_vars.reserve(mlir_func_op.getNumArguments());

  for (auto arg_and_index : llvm::enumerate(mlir_func_op.getArguments())) {
    Value *v = &arg_and_index.value();
    tvm::relay::TensorType relay_type;
    auto res = MLIRTensorTypeToRelayTensorType(v->getType(), &relay_type);
    if (failed(res))
      return res;

    std::string var_name = "arg_" + std::to_string(arg_and_index.index());
    auto relay_var = tvm::relay::Var(var_name, relay_type);
    values_to_exprs[v] = relay_var;
    input_vars.push_back(relay_var);
  }

  // Since there is no CFG, each function should only have one block.
  auto &block = blocks.front();

  auto block_body_no_return =
      llvm::make_range(block.begin(), std::prev(block.end()));

  tvm::relay::Expr expr;
  for (mlir::Operation &op : block_body_no_return) {
    if (op.getDialect()->getNamespace() != "mrelay")
      return mlir::failure();

    // tvm::relay::Expr expr;
    auto res = ConvertMLIROpToRelay(&op, &expr, &input_vars);
    if (failed(res))
      return res;
  }

  auto return_op = mlir::dyn_cast<mlir::ReturnOp>(block.back());
  if (!return_op)
    return mlir::failure();

  // TODO (yongwww): Support multiple return values using Relay tuples.
  if (return_op.getNumOperands() > 1)
    return mlir::failure();

  // auto return_value = return_op.getOperand(0);
  // auto relay_return_expr_it = values_to_exprs.find(&return_value);
  // if (relay_return_expr_it == values_to_exprs.end()) return mlir::failure();
  //*relay_func = tvm::relay::Function(input_vars, relay_return_expr_it->second,
  //                                   tvm::relay::Type(), {});

  *relay_func = tvm::relay::Function(input_vars, expr, tvm::relay::Type(), {});

  // TODO: Construct function with inputs/body/returns.
  return mlir::success();
}

// Converts an MLIR module to relay_module. Fails if such conversion is
// not possible.
LogicalResult ConvertMLIRModuleToRelayModule(mlir::ModuleOp mlir_module,
                                             tvm::IRModule *relay_module) {
  auto res = mlir_module.walk([&](mlir::FuncOp func_op) {
    auto func_name = func_op.getName();
    tvm::GlobalVar relay_func_global_var(func_name.str());

    tvm::relay::Function relay_func;
    auto res = ConvertMLIRFuncToRelayFunc(func_op, &relay_func);
    if (failed(res))
      return WalkResult::interrupt();

    (*relay_module)->Add(relay_func_global_var, relay_func, /*update=*/false);
    return WalkResult::advance();
  });

  return mlir::failure(res.wasInterrupted());
}
} // namespace

OwningModuleRef RelayToMlirTranslateFunction(llvm::StringRef input,
                                             MLIRContext *context) {
  // TODO(yongwww): Implement Relay to MLIR translation.
  return nullptr;
}

LogicalResult MlirToRelayTranslateFunction(ModuleOp module,
                                           llvm::raw_ostream &output) {
  if (!module)
    return failure();

  tvm::IRModule relay_module;

  auto res = ConvertMLIRModuleToRelayModule(module, &relay_module);
  if (failed(res))
    return mlir::failure();

  auto relay_module_str = tvm::AsText(relay_module);
  output << std::string(relay_module_str);

  return mlir::success();
}

} // namespace mrelay
} // namespace mlir
