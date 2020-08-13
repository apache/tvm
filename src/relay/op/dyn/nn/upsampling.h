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
 *
 * \file src/relay/op.nn/upsampling.h
 * \brief Header of the upsampling file for methods that need to be accessed by multiple files
 */

namespace tvm {
namespace relay {

template <typename T>
Array<Array<Layout> > UpsamplingInferCorrectLayout(const Attrs& attrs,
                                                   const Array<Layout>& new_in_layouts,
                                                   const Array<Layout>& old_in_layouts,
                                                   const Array<tvm::relay::Type>& old_in_types);

}
}
