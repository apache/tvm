/*!
 *  Copyright (c) 2016 by Contributors
 * \file codegen.h
 * \brief Collection of Lowlevel IR pass to codegen.
 */
#ifndef TVM_CODEGEN_H_
#define TVM_CODEGEN_H_

#include <string>
#include "./base.h"
#include "./expr.h"
#include "./lowered_func.h"
#include "./runtime/packed_func.h"


namespace tvm {
/*! \brief namespace for lowlevel IR pass and codegen */
namespace codegen {
// use packed function from runtime.
using runtime::PackedFunc;
using runtime::TVMArgs;
using runtime::TVMRetValue;

/*!
 * \brief Make an user callable API LoweredFunc.
 *
 *  The main task of this function is to create code to :
 *   - Map the values in the api_args to of Var that is required by body.
 *   - Insert assertions to check type/value of the passed arguments.
 *
 * \param body The body of the function.
 * \param name The name of the function.
 * \param api_args Arguments to the function, can be either Var, or Buffer
 * \param num_packed_args Number of arguments that are processed in packed form.
 * \return a LoweredFunc with the specified signiture.
 *
 * \note
 *  The function signiture have two cases
 *
 *  if num_packed_args is zero:
 *     f(api_arg_0, api_arg_1, .., api_arg_n) where n == len(api_args)
 *
 *  if num_packed_args is not zero:
 *       f(TVMArg* packed_args, int* packed_arg_type_ids, int num_packed_args,
 *         api_arg_k, api_arg_k+1, ... api_arg_n)
 *
 *       where n == len(api_args), k == num_packed_args
 *
 *  There is no thread_axis in generated function.
 */
LoweredFunc MakeAPI(Stmt body,
                    std::string name,
                    Array<NodeRef> api_args,
                    int num_packed_args);

/*!
 * \brief Count number of undefined vars in f.
 * \param f The function to be checked.
 * \return Number of undefined vars.
 */
Array<Var> UndefinedVars(const LoweredFunc& f);

/*!
 * \brief Split the function into a host function and device functions.
 * \param func The function to be splitted.
 *
 * \return Array of functions, the first one is host function,
 *     the others are device functions.
 */
Array<LoweredFunc> SplitHostDevice(LoweredFunc func);

/*!
 * \brief Build a stack VM function.
 * \param func The LoweredFunc to be build
 * \param device_funcs The additional device functions
 * \return A packed function representing the func.
 */
PackedFunc BuildStackVM(
    LoweredFunc func,
    const std::unordered_map<LoweredFunc, PackedFunc>& device_funcs);

/*!
 * \brief Build a CUDA function with NVRTC
 *
 * \param fsplits The LoweredFuncs to be build (after SplitHostDevice)
 *  The first element is the host function, followed by device functions.
 * \param host_mode The host side compilation mode:
 *   - "stackvm": use stack vm to interpret host side code.
 */
PackedFunc BuildNVRTC(Array<LoweredFunc> fsplits, std::string host_mode);

/*!
 * \brief Build a OpenCL function.
 *
 * \param fsplits The LoweredFuncs to be build (after SplitHostDevice)
 *  The first element is the host function, followed by device functions.
 * \param host_mode The host side compilation mode:
 *   - "stackvm": use stack vm to interpret host side code.
 */
PackedFunc BuildOpenCL(Array<LoweredFunc> fsplits, std::string host_mode);

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_H_
