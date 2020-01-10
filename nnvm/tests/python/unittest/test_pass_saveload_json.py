# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import nnvm
from tvm.contrib import util


def test_variable_node_parsed():
    sym = nnvm.sym.Variable('data')
    tempdir = util.tempdir()
    json_filename = 'test_nnvm_symbol.json'
    with open(tempdir.relpath(json_filename), 'w') as fo:
        fo.write(nnvm.graph.create(sym).json())
    sym_str = open(tempdir.relpath(json_filename), 'r').read()
    sym = nnvm.graph.load_json(sym_str).symbol()
    sym = nnvm.sym.relu(sym)


if __name__ == '__main__':
    test_variable_node_parsed()
