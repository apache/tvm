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
"""Utilities for changing datatypes of models."""
import tvm
import numpy as np
from tvm import relay
from tvm.relay.testing.inception_v3 import get_workload


def test_change_dtype_inception_v3():
    module, params = get_workload()

    def change_dtype(src, dst, module, params):
        module = relay.frontend.ChangeDatatype(src, dst)(module)
        module = relay.transform.InferType()(module)
        params = dict(
            (p, tvm.nd.array(params[p].asnumpy().astype(dst))) for p in params)
        return module, params

    module, params = change_dtype('float32', 'float16', module, params)


    
    ex = relay.create_executor("graph")
    input = tvm.nd.array(np.random.rand(3, 299, 299).astype('float32'))
    x = relay.var("x", shape=(3, 299, 299))
    castR = relay.Function([x], x.astype('float16'))
    input = ex.evaluate(castR)(input) 
    result = ex.evaluate(expr)(input, **params)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    test_change_dtype_inception_v3()
