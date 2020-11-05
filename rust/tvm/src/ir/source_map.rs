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
 * KIND, either exprss or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

use crate::runtime::map::Map;
use crate::runtime::object::Object;
use crate::runtime::string::String as TString;

use super::span::SourceName;

use tvm_macros::Object;

/// A program source in any language.
///
/// Could represent the source from an ML framework or a source of an IRModule.
#[repr(C)]
#[derive(Object, Debug)]
#[type_key = "Source"]
#[ref_name = "Source"]
pub struct SourceNode {
    pub base: Object,
    /// The source name.
    pub source_name: SourceName,

    /// The raw source.
    pub source: TString,
    // TODO(@jroesch): Non-ABI compat field
    // A mapping of line breaks into the raw source.
    // std::vector<std::pair<int, int>> line_map;
}

/// A mapping from a unique source name to source fragments.
#[repr(C)]
#[derive(Object, Debug)]
#[type_key = "SourceMap"]
#[ref_name = "SourceMap"]
pub struct SourceMapNode {
    /// The base object.
    pub base: Object,
    /// The source mapping.
    pub source_map: Map<SourceName, Source>,
}
