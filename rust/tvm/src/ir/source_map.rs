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

use crate::runtime::map::Map;
use crate::runtime::object::Object;

/// A program source in any language.
///
/// Could represent the source from an ML framework or a source of an IRModule.
#[repr(C)]
#[derive(Object)]
#[type_key = "Source"]
#[ref_key = "Source"]
struct SourceNode {
    pub base: Object,
    /*! \brief The source name. */
   SourceName source_name;

   /*! \brief The raw source. */
   String source;

   /*! \brief A mapping of line breaks into the raw source. */
   std::vector<std::pair<int, int>> line_map;
}


//  class Source : public ObjectRef {
//   public:
//    TVM_DLL Source(SourceName src_name, std::string source);
//    TVM_DLL tvm::String GetLine(int line);

//    TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Source, ObjectRef, SourceNode);
//  };


/// A mapping from a unique source name to source fragments.
#[repr(C)]
#[derive(Object)]
#[type_key = "SourceMap"]
#[ref_key = "SourceMap"]
struct SourceMapNode {
    pub base: Object,
   /// The source mapping.
   pub source_map: Map<SourceName, Source>,
}
