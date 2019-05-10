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
import tvm
from tvm import relay
from tvm.relay.testing import to_python, run_as_python
from tvm.relay.prelude import Prelude
from tvm.relay.backend.interpreter import TensorValue, TupleValue, ConstructorValue, RefValue

def test_create_empty_tuple():
    empty = relay.Tuple([])
    tup_val = run_as_python(empty)
    assert isinstance(tup_val, TupleValue)
    assert len(tup_val.fields) == 0
