import tvm
from tvm.contrib import rpc

proxy_host = "localhost"
proxy_port = 9090

def test_remote_opengl():
    rpc._TestLocalOpenGL()
    print("_TestLocalOpenGL completed.")

    if not tvm.module.enabled("rpc"):
        return
    remote = rpc.connect(proxy_host, proxy_port, key="js")
    rpc._TestRemoteOpenGL(remote._sess)

test_remote_opengl()
