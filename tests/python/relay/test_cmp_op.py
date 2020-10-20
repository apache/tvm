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

from tvm import relay

a = relay.Var("a")
b = relay.expr.const(1.0, dtype="float32")

c = a < b
d = relay.less(a, b)
assert c.astext() == d.astext()

c = a > b
d = relay.greater(a, b)
assert c.astext() == d.astext()

c = a >= b
d = relay.greater_equal(a, b)
assert c.astext() == d.astext()

c = a <= b
d = relay.less_equal(a, b)
assert c.astext() == d.astext()
