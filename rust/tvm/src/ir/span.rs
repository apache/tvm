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

use crate::runtime::{Object, ObjectRef, String as TString};
use tvm_macros::Object;

/// A source file name, contained in a Span.

#[repr(C)]
#[derive(Object)]
#[type_key = "SourceName"]
#[ref_name = "SourceName"]
pub struct SourceNameNode {
    pub base: Object,
    pub name: TString,
}

//  /*!
//   * \brief The source name of a file span.
//   * \sa SourceNameNode, Span
//   */
//  class SourceName : public ObjectRef {
//   public:
//    /*!
//     * \brief Get an SourceName for a given operator name.
//     *  Will raise an error if the source name has not been registered.
//     * \param name Name of the operator.
//     * \return SourceName valid throughout program lifetime.
//     */
//    TVM_DLL static SourceName Get(const String& name);

//    TVM_DEFINE_OBJECT_REF_METHODS(SourceName, ObjectRef, SourceNameNode);
//  };

/// Span information for diagnostic purposes.
#[repr(C)]
#[derive(Object)]
#[type_key = "Span"]
#[ref_name = "Span"]
pub struct SpanNode {
    pub base: Object,
    /// The source name.
    pub source_name: SourceName,
    /// The line number.
    pub line: i32,
    /// The column offset.
    pub column: i32,
    /// The end line number.
    pub end_line: i32,
    /// The end column number.
    pub end_column: i32,
}

impl Span {
    pub fn empty() -> Span {
        todo!()
    }
}
