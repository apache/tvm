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
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TVM_SCRIPT_PRINTER_DOC_TRANSLATOR_PRIVATE_H_
#define TVM_SCRIPT_PRINTER_DOC_TRANSLATOR_PRIVATE_H_

#include <tvm/script/printer/doc_translator.h>

namespace tvm {
namespace script {
namespace printer {
namespace details {

/*! \brief Complete translated body, origins and explicit header facts. */
struct TranslationResult {
  /*! \brief Complete body, including declarations and metadata. */
  Doc doc;
  /*! \brief Original IR objects associated with generated Docs. */
  ffi::Dict<Doc, ffi::ObjectRef> origins;
  /*! \brief Whether the script needs postponed annotation evaluation. */
  bool future_annotations;
  /*! \brief Allocated import name for the metadata loader. */
  ffi::String metadata_loader;
  /*! \brief Whether the body uses the metadata loader. */
  bool imports_metadata;
};

/*!
 * \brief Translate an input into its complete Doc representation.
 * \param ir The input to translate.
 * \param extra_config Read-only options for this translation.
 * \return Complete body, origins and header facts.
 */
TranslationResult TranslateWithOptions(ffi::AnyView ir,
                                       ffi::Map<ffi::String, ffi::Any> extra_config);

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_DOC_TRANSLATOR_PRIVATE_H_
