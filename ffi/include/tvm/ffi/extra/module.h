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
 * \file tvm/ffi/extra/module.h
 * \brief A managed dynamic module in the TVM FFI.
 */
#ifndef TVM_FFI_EXTRA_MODULE_H_
#define TVM_FFI_EXTRA_MODULE_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/base.h>
#include <tvm/ffi/function.h>

namespace tvm {
namespace ffi {

// forward declare Module
class Module;

/*!
 * \brief A module that can dynamically load ffi::Functions or exportable source code.
 * \sa Module
 */
class TVM_FFI_EXTRA_CXX_API ModuleObj : public Object {
 public:
  /*!
   * \return The per module type key.
   * \note This key is used to for serializing custom modules.
   */
  virtual const char* kind() const = 0;
  /*!
   * \brief Get the property mask of the module.
   * \return The property mask of the module.
   *
   * \sa Module::ModulePropertyMask
   */
  virtual int GetPropertyMask() const { return 0b000; }
  /*!
   * \brief Get a ffi::Function from the module.
   * \param name The name of the function.
   * \return The function.
   */
  virtual Optional<Function> GetFunction(const String& name) = 0;
  /*!
   * \brief Returns true if this module has a definition for a function of \p name.
   *
   * Note that even if this function returns true the corresponding \p GetFunction result
   * may be nullptr if the function is not yet callable without further compilation.
   *
   * The default implementation just checks if \p GetFunction is non-null.
   * \param name The name of the function.
   * \return True if the module implements the function, false otherwise.
   */
  virtual bool ImplementsFunction(const String& name) { return GetFunction(name).defined(); }
  /*!
   * \brief Get the metadata of the function, if available.
   * \param name The name of the function.
   * \return The metadata stored in json string format.
   */
  virtual Optional<String> GetFunctionMetadata(const String& name) { return std::nullopt; }
  /*!
   * \brief Write the current module to file with given format (for further compilation).
   *
   * \param file_name The file to be saved to.
   * \param format The format of the file.
   *
   * \note This function is mainly used by modules that
   */
  virtual void WriteToFile(const String& file_name, const String& format) const {
    TVM_FFI_THROW(RuntimeError) << "Module[" << kind() << "] does not support WriteToFile";
  }
  /*!
   * \brief Get the possible write formats of the module, when available.
   * \return Possible write formats when available.
   */
  virtual Array<String> GetWriteFormats() const { return Array<String>(); }
  /*!
   * \brief Serialize the the module to bytes.
   * \return The serialized module.
   */
  virtual Bytes SaveToBytes() const {
    TVM_FFI_THROW(RuntimeError) << "Module[" << kind() << "] does not support SaveToBytes";
    TVM_FFI_UNREACHABLE();
  }
  /*!
   * \brief Get the source code of module, when available.
   * \param format Format of the source code, can be empty by default.
   * \return Possible source code when available, or empty string if not available.
   */
  virtual String InspectSource(const String& format = "") const { return String(); }
  /*!
   * \brief Import another module.
   * \param other The module to import.
   */
  virtual void ImportModule(const Module& other);
  /*!
   * \brief Clear all imported modules.
   */
  virtual void ClearImports();
  /*!
   * \brief Overloaded function to optionally query from imports.
   * \param name The name of the function.
   * \param query_imports Whether to query imported modules.
   * \return The function.
   */
  Optional<Function> GetFunction(const String& name, bool query_imports);
  /*!
   * \brief Overloaded function to optionally query from imports.
   * \param name The name of the function.
   * \param query_imports Whether to query imported modules.
   * \return True if the module implements the function, false otherwise.
   */
  bool ImplementsFunction(const String& name, bool query_imports);
  /*!
   * \brief Get the function metadata of the function if available.
   * \param name The name of the function.
   * \param query_imports Whether to query imported modules.
   * \return The function metadata of the function in json format.
   */
  Optional<String> GetFunctionMetadata(const String& name, bool query_imports);
  /*!
   * \brief Get the imports of the module.
   * \return The imports of the module.
   * \note Note the signature is not part of the public API.
   */
  const Array<Any>& imports() const { return this->imports_; }

  struct InternalUnsafe;

  /// \cond Doxygen_Suppress
  static constexpr const int32_t _type_index = TypeIndex::kTVMFFIModule;
  static constexpr const bool _type_mutable = true;
  static const constexpr bool _type_final = true;
  TVM_FFI_DECLARE_OBJECT_INFO_STATIC(StaticTypeKey::kTVMFFIModule, ModuleObj, Object);
  /// \endcond

 protected:
  friend struct InternalUnsafe;

  /*!
   * \brief The modules that this module depends on.
   * \note Use ObjectRef to avoid circular dep on Module.
   */
  Array<Any> imports_;

 private:
  /*!
   * \brief cache used by TVMFFIModuleLookupFromImports
   */
  Map<String, ffi::Function> import_lookup_cache_;
};

/*!
 * \brief Reference to module object.
 *
 * When invoking a function on a ModuleObj, such as GetFunction,
 * use operator-> to get the ModuleObj pointer and invoke the member functions.
 *
 * \code
 * ffi::Module mod = ffi::Module::LoadFromFile("path/to/module.so");
 * ffi::Function func = mod->GetFunction(name);
 * \endcode
 *
 * \sa ModuleObj which contains most of the function implementations.
 */
class Module : public ObjectRef {
 public:
  /*!
   * \brief Property of ffi::Module
   */
  enum ModulePropertyMask : int {
    /*!
     * \brief The module can be serialized to bytes.
     *
     * This prooperty indicates that module implements SaveToBytes.
     * The system also registers a GlobalDef function
     * `ffi.Module.load_from_bytes.<kind>` with signature (Bytes) -> Module.
     */
    kBinarySerializable = 0b001,
    /*!
     * \brief The module can directly get runnable functions.
     *
     * This property indicates that module implements GetFunction that returns
     * runnable ffi::Functions.
     */
    kRunnable = 0b010,
    /*!
     * \brief The module can be exported to a object file or source file that then be compiled.
     *
     * This property indicates that module implements WriteToFile with a given format
     * that can be queried by GetLibExportFormat.
     *
     * Examples include modules that can be exported to .o, .cc, .cu files.
     *
     * Such modules can be exported, compiled and loaded back as a dynamic library module.
     */
    kCompilationExportable = 0b100
  };
  /*!
   * \brief Constructor from ObjectPtr<ModuleObj>.
   * \param ptr The object pointer.
   */
  explicit Module(ObjectPtr<ModuleObj> ptr) : ObjectRef(ptr) { TVM_FFI_ICHECK(ptr != nullptr); }
  /*!
   * \brief Load a module from file.
   * \param file_name The name of the host function module.
   * \note This function won't load the import relationship.
   *  Re-create import relationship by calling Import.
   */
  TVM_FFI_EXTRA_CXX_API static Module LoadFromFile(const String& file_name);
  /*!
   * \brief Query context symbols that is registered via TVMEnvRegisterSymbols.
   * \param callback The callback to be called with the symbol name and address.
   * \note This helper can be used to implement custom Module that needs to access context symbols.
   */
  TVM_FFI_EXTRA_CXX_API static void VisitContextSymbols(
      const ffi::TypedFunction<void(String, void*)>& callback);

  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Module, ObjectRef, ModuleObj);
  /// \endcond
};

/*
 * \brief Symbols for library module.
 */
namespace symbol {
/*!\ brief symbol prefix for tvm ffi related function symbols */
constexpr const char* tvm_ffi_symbol_prefix = "__tvm_ffi_";
// Special symbols have one extra _ prefix to avoid conflict with user symbols
/*!
 * \brief Default entry function of a library module is tvm_ffi_symbol_prefix + "main"
 */
constexpr const char* tvm_ffi_main = "__tvm_ffi_main";
/*! \brief Global variable to store context pointer for a library module. */
constexpr const char* tvm_ffi_library_ctx = "__tvm_ffi__library_ctx";
/*! \brief Global variable to store binary data alongside a library module. */
constexpr const char* tvm_ffi_library_bin = "__tvm_ffi__library_bin";
/*! \brief Optional metadata prefix of a symbol. */
constexpr const char* tvm_ffi_metadata_prefix = "__tvm_ffi__metadata_";
}  // namespace symbol
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_EXTRA_MODULE_H_
