/*!
 *  Copyright (c) 2017 by Contributors
 * \file tvm/runtime/module.h
 * \brief Runtime container of the functions generated by TVM,
 *  This is used to support dynamically link, load and save
 *  functions from different convention under unified API.
 */
#ifndef TVM_RUNTIME_MODULE_H_
#define TVM_RUNTIME_MODULE_H_

#include <dmlc/io.h>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include "c_runtime_api.h"

namespace tvm {
namespace runtime {

// The internal container of module.
class ModuleNode;
class PackedFunc;

/*!
 * \brief Module container of TVM.
 */
class Module {
 public:
  Module() {}
  // constructor from container.
  explicit Module(std::shared_ptr<ModuleNode> n)
      : node_(n) {}
  /*!
   * \brief Get packed function from current module by name.
   *
   * \param name The name of the function.
   * \param query_imports Whether also query dependency modules.
   * \return The result function.
   *  This function will return PackedFunc(nullptr) if function do not exist.
   * \note Implemented in packed_func.cc
   */
  inline PackedFunc GetFunction(const std::string& name, bool query_imports = false);
  /*! \return internal container */
  inline ModuleNode* operator->();
  /*! \return internal container */
  inline const ModuleNode* operator->() const;
  // The following functions requires link with runtime.
  /*!
   * \brief Import another module into this module.
   * \param other The module to be imported.
   *
   * \note Cyclic dependency is not allowed among modules,
   *  An error will be thrown when cyclic dependency is detected.
   */
  TVM_DLL void Import(Module other);
  /*!
   * \brief Load a module from file.
   * \param file_name The name of the host function module.
   * \param format The format of the file.
   * \note This function won't load the import relationship.
   *  Re-create import relationship by calling Import.
   */
  TVM_DLL static Module LoadFromFile(const std::string& file_name,
                                     const std::string& format = "");

 private:
  std::shared_ptr<ModuleNode> node_;
};

/*!
 * \brief Base node container of module.
 *  Do not create this directly, instead use Module.
 */
class ModuleNode {
 public:
  /*! \brief virtual destructor */
  virtual ~ModuleNode() {}
  /*! \return The module type key */
  virtual const char* type_key() const = 0;
  /*!
   * \brief Get a PackedFunc from module.
   *
   *  The PackedFunc may not be fully initialized,
   *  there might still be first time running overhead when
   *  executing the function on certain devices.
   *  For benchmarking, use prepare to eliminate
   *
   * \param name the name of the function.
   * \param sptr_to_self The shared_ptr that points to this module node.
   *
   * \return PackedFunc(nullptr) when it is not available.
   *
   * \note The function will always remain valid.
   *   If the function need resource from the module(e.g. late linking),
   *   it should capture sptr_to_self.
   */
  virtual PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) = 0;
  /*!
   * \brief Save the module to file.
   * \param file_name The file to be saved to.
   * \param format The format of the file.
   */
  virtual void SaveToFile(const std::string& file_name,
                          const std::string& format);
  /*!
   * \brief Save the module to binary stream.
   * \param stream The binary stream to save to.
   * \note It is recommended to implement this for device modules,
   *   but not necessarily host modules.
   *   We can use this to do AOT loading of bundled device functions.
   */
  TVM_DLL virtual void SaveToBinary(dmlc::Stream* stream);
  /*!
   * \brief Get the source code of module, when available.
   * \param format Format of the source code, can be empty by default.
   * \return Possible source code when available.
   */
  TVM_DLL virtual std::string GetSource(const std::string& format = "");
  /*!
   * \brief Get a function from current environment
   *  The environment includes all the imports as well as Global functions.
   *
   * \param name name of the function.
   * \return The corresponding function.
   */
  TVM_DLL const PackedFunc* GetFuncFromEnv(const std::string& name);
  /*! \return The module it imports from */
  const std::vector<Module>& imports() const {
    return imports_;
  }

 protected:
  friend class Module;
  /*! \brief The modules this module depend on */
  std::vector<Module> imports_;

 private:
  /*! \brief Cache used by GetImport */
  std::unordered_map<std::string,
                     std::unique_ptr<PackedFunc> > import_cache_;
};

/*! \brief namespace for constant symbols */
namespace symbol {
/*! \brief Global variable to store module context. */
constexpr const char* tvm_module_ctx = "__tvm_module_ctx";
/*! \brief Global variable to store device module blob */
constexpr const char* tvm_dev_mblob = "__tvm_dev_mblob";
/*! \brief Number of bytes of device module blob. */
constexpr const char* tvm_dev_mblob_nbytes = "__tvm_dev_mblob_nbytes";
/*! \brief global function to set device */
constexpr const char* tvm_set_device = "__tvm_set_device";
/*! \brief Auxiliary counter to global barrier. */
constexpr const char* tvm_global_barrier_state = "__tvm_global_barrier_state";
/*! \brief Prepare the global barrier before kernels that uses global barrier. */
constexpr const char* tvm_prepare_global_barrier = "__tvm_prepare_global_barrier";
/*! \brief Placeholder for the module's entry function. */
constexpr const char* tvm_module_main = "__tvm_main__";
}  // namespace symbol

// implementations of inline functions.
inline ModuleNode* Module::operator->() {
  return node_.get();
}

inline const ModuleNode* Module::operator->() const {
  return node_.get();
}

}  // namespace runtime
}  // namespace tvm

#include "packed_func.h"
#endif  // TVM_RUNTIME_MODULE_H_
