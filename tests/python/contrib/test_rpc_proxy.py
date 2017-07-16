import tvm
import logging
import numpy as np
import time
import multiprocessing
from tvm.contrib import rpc

def rpc_proxy_check():
    """This is a simple test function for RPC Proxy

    It is not included as nosetests, because:
    - It depends on tornado
    - It relies on the fact that Proxy starts before client and server connects,
      which is often the case but not always

    User can directly run this script to verify correctness.
    """

    try:
        from tvm.contrib import rpc_proxy
        web_port = 8888
        prox = rpc_proxy.Proxy("localhost", web_port=web_port)
        def check():
            if not tvm.module.enabled("rpc"):
                return
            @tvm.register_func("rpc.test2.addone")
            def addone(x):
                return x + 1
            @tvm.register_func("rpc.test2.strcat")
            def addone(name, x):
                return "%s:%d" % (name, x)
            server = multiprocessing.Process(
                target=rpc_proxy.websocket_proxy_server,
                args=("ws://localhost:%d/ws" % web_port,"x1"))
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
