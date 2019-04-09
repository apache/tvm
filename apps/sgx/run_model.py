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
import os.path as osp
import numpy as np
import tvm

CWD = osp.abspath(osp.dirname(__file__))


def main():
    ctx = tvm.context('cpu', 0)
    model = tvm.module.load(osp.join(CWD, 'build', 'enclave.signed.so'))
    inp = tvm.nd.array(np.ones((1, 3, 224, 224), dtype='float32'), ctx)
    out = tvm.nd.array(np.empty((1, 1000), dtype='float32'), ctx)
    model(inp, out)
    if abs(out.asnumpy().sum() - 1) < 0.001:
        print('It works!')
    else:
        print('It doesn\'t work!')
        exit(1)


if __name__ == '__main__':
    main()
