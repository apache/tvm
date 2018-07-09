import time
from tvm.rpc import proxy

def start_proxy_server(port, timeout):
    prox = proxy.Proxy("localhost", port=port, port_end=port+1)
    if timeout > 0:
        import time
        time.sleep(timeout)
        prox.terminate()
    else:
        prox.proc.join()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit(-1)
    port = int(sys.argv[1])
    timeout = 0 if len(sys.argv) == 2 else float(sys.argv[2])
    start_proxy_server(port, timeout)
