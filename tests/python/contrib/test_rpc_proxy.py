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
from tvm import te
import logging
import numpy as np
import time
import multiprocessing
from tvm import rpc


def rpc_proxy_check():
    """This is a simple test function for RPC Proxy

    It is not included as pytests, because:
    - It depends on tornado
    - It relies on the fact that Proxy starts before client and server connects,
      which is often the case but not always

    User can directly run this script to verify correctness.
    """

    try:
        from tvm.rpc import proxy

        web_port = 8888
        prox = proxy.Proxy("localhost", web_port=web_port)

        def check():
            if not tvm.runtime.enabled("rpc"):
                return

            @tvm.register_func("rpc.test2.addone")
            def addone(x):
                return x + 1

            @tvm.register_func("rpc.test2.strcat")
            def addone(name, x):
                return "%s:%d" % (name, x)

            server = multiprocessing.Process(
                target=proxy.websocket_proxy_server, args=("ws://localhost:%d/ws" % web_port, "x1")
            )
            # Need to make sure that the connection start after proxy comes up
            time.sleep(0.1)
            server.deamon = True
            server.start()
            client = rpc.connect(prox.host, prox.port, key="x1")
            f1 = client.get_function("rpc.test2.addone")
            assert f1(10) == 11
            f2 = client.get_function("rpc.test2.strcat")
            assert f2("abc", 11) == "abc:11"

        check()
    except ImportError:
        print("Skipping because tornado is not avaliable...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rpc_proxy_check()
