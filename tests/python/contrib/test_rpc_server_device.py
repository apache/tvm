import time
import numpy as np

import tvm.testing
from tvm import te
from tvm import rpc
from tvm import relay
from tvm.contrib import utils, xcode, graph_executor
from tvm.autotvm.measure import request_remote
from tvm.auto_scheduler.utils import call_func_with_timeout
from tvm.contrib.popen_pool import PopenWorker, StatusKind
from tvm.rpc import tracker, proxy, server_ios_launcher


HOST_URL = "0.0.0.0"
HOST_PORT = 9190
DEVICE_KEY = "ios_mobile_device"
# TODO: When starting an application in pure_rpc mode, the address and port fields are ignored.
# The server starts on one of the following ports: 9090-9099.
# The port on which the server is running is printed in the application log window.
# This value is not broadcast anywhere else, it cannot be programmatically obtained from outside
# the application, so there may be connection problems if port 9090 is busy and the server starts
# on a different port.
HOST_PORT_PURE_RPC = 9090


TEMPORARY_DIRECTORY = utils.tempdir()
ARCH = "x86_64"
SDK = "iphonesimulator"
DSO_NAME = "lib.dylib"
DTYPE = "float32"


np.random.seed(0)


def setup_pure_rpc_configuration(f):
    """
    Host  --  RPC server
    """
    def wrapper():
        device_server_launcher = server_ios_launcher.ServerIOSLauncher(mode=server_ios_launcher.RPCServerMode.pure_server.value,
                                                                       host=HOST_URL,
                                                                       port=HOST_PORT_PURE_RPC,
                                                                       key=DEVICE_KEY)
        time.sleep(5)   # Need to make sure that the server start
        f(host=device_server_launcher.host, port=device_server_launcher.port)
        device_server_launcher.terminate()
    return wrapper


def setup_rpc_proxy_configuration(f):
    """
    Host -- Proxy -- RPC server
    """
    def wrapper():
        proxy_server = proxy.Proxy(host=HOST_URL, port=HOST_PORT)
        device_server_launcher = server_ios_launcher.ServerIOSLauncher(mode=server_ios_launcher.RPCServerMode.proxy.value,
                                                                       host=proxy_server.host,
                                                                       port=proxy_server.port,
                                                                       key=DEVICE_KEY)
        f(host=proxy_server.host, port=proxy_server.port)
        device_server_launcher.terminate()
        proxy_server.terminate()
    return wrapper


def setup_rpc_tracker_configuration(f):
    """
         tracker
         /     \
    Host   --   RPC server
    """
    def wrapper():
        tracker_server = tracker.Tracker(host=HOST_URL, port=HOST_PORT, silent=True)
        device_server_launcher = server_ios_launcher.ServerIOSLauncher(mode=server_ios_launcher.RPCServerMode.tracker.value,
                                                                       host=tracker_server.host,
                                                                       port=tracker_server.port,
                                                                       key=DEVICE_KEY)
        f(host=tracker_server.host, port=tracker_server.port)
        device_server_launcher.terminate()
        tracker_server.terminate()
    return wrapper


def setup_rpc_tracker_via_proxy_configuration(f):
    """
         tracker
         /     \
    Host   --   Proxy -- RPC server
    """
    def wrapper():
        tracker_server = tracker.Tracker(host=HOST_URL, port=HOST_PORT, silent=True)
        proxy_server_tracker = proxy.Proxy(host=HOST_URL, port=8888, tracker_addr=(tracker_server.host, tracker_server.port))
        device_server_launcher = server_ios_launcher.ServerIOSLauncher(mode=server_ios_launcher.RPCServerMode.proxy.value,
                                                                       host=proxy_server_tracker.host,
                                                                       port=proxy_server_tracker.port,
                                                                       key=DEVICE_KEY)
        f(host=tracker_server.host, port=tracker_server.port)
        device_server_launcher.terminate()
        proxy_server_tracker.terminate()
        tracker_server.terminate()
    return wrapper


def can_create_connection_without_deadlock(timeout, func, args=(), kwargs=None):
    def wrapper(*_args, **_kwargs):
        """
            This wrapper is needed because the cloudpicle
            cannot serialize objects that contain pointers (RPCSession)
        """
        func(*_args, **_kwargs)
        return StatusKind.COMPLETE

    worker = PopenWorker()
    ret = call_func_with_timeout(worker, timeout=timeout, func=wrapper, args=args, kwargs=kwargs)
    if isinstance(ret, Exception):
        raise ret
    return ret


def try_create_remote_session(session_factory, args=(), kwargs=None):
    try:
        successful_attempt = True
        results = []
        for _ in range(2):
            ret = can_create_connection_without_deadlock(timeout=10, func=session_factory,
                                                         args=args, kwargs=kwargs)
            results.append(ret)
        if not np.all(np.array(results) == StatusKind.COMPLETE):
            raise ValueError("One or more sessions ended incorrectly.")
    except Exception as e:
        successful_attempt = False
        print(e)
    return successful_attempt


def export_lib(lib):
    path_dso = TEMPORARY_DIRECTORY.relpath(DSO_NAME)
    lib.export_library(path_dso, xcode.create_dylib, arch=ARCH, sdk=SDK)
    return path_dso


def get_add_relay_module(a_numpy, b_numpy):
    a = relay.var("a", shape=a_numpy.shape, dtype=DTYPE)
    b = relay.var("b", shape=b_numpy.shape, dtype=DTYPE)
    params = {}
    out = tvm.IRModule.from_expr(relay.add(a, b))
    return out, params


def get_add_module(target):
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = te.create_schedule(C.op)
    return tvm.build(s, [A, B, C], target=target, target_host=target, name="simple_add")


@setup_pure_rpc_configuration
def test_pure_rpc(host, port):
    status_ok = try_create_remote_session(session_factory=rpc.connect, args=(host, port))
    assert status_ok


@setup_rpc_proxy_configuration
def test_rpc_proxy(host, port):
    status_ok = try_create_remote_session(session_factory=rpc.connect, args=(host, port, DEVICE_KEY))
    assert status_ok


@setup_rpc_tracker_configuration
def test_rpc_tracker(host, port):
    status_ok = try_create_remote_session(session_factory=request_remote, args=(DEVICE_KEY, host, port))
    assert status_ok


@setup_rpc_tracker_via_proxy_configuration
def test_rpc_tracker_via_proxy(host, port):
    status_ok = try_create_remote_session(session_factory=request_remote, args=(DEVICE_KEY, host, port))
    assert status_ok


@setup_pure_rpc_configuration
def test_can_call_remote_function_with_pure_rpc(host, port):
    remote_session = rpc.connect(host, port)
    f = remote_session.get_function("runtime.GetFFIString")
    assert f("hello") == "hello"


@setup_rpc_proxy_configuration
def test_can_call_remote_function_with_rpc_proxy(host, port):
    remote_session = rpc.connect(host, port, key=DEVICE_KEY)
    f = remote_session.get_function("runtime.GetFFIString")
    assert f("hello") == "hello"


@setup_rpc_tracker_configuration
def test_can_call_remote_function_with_rpc_tracker(host, port):
    remote_session = request_remote(DEVICE_KEY, host, port)
    f = remote_session.get_function("runtime.GetFFIString")
    assert f("hello") == "hello"


@setup_rpc_tracker_via_proxy_configuration
def test_can_call_remote_function_with_rpc_tracker_via_proxy(host, port):
    remote_session = request_remote(DEVICE_KEY, host, port)
    f = remote_session.get_function("runtime.GetFFIString")
    assert f("hello") == "hello"


@setup_pure_rpc_configuration
def test_basic_functionality_of_rpc_session(host, port):
    remote_session = rpc.connect(host, port)
    device = remote_session.cpu(0)

    target = tvm.target.Target(target=f"llvm -mtriple={ARCH}-apple-darwin")
    lib = get_add_module(target)
    path_dso = export_lib(lib)

    # Check correct upload
    remote_session.upload(path_dso)

    # Check correct download
    downloaded_lib = remote_session.download(DSO_NAME)
    assert downloaded_lib == bytearray(open(path_dso, "rb").read()),\
        "The downloaded module does not match the loaded module"

    # Check correct remote computing
    lib = remote_session.load_module(DSO_NAME)
    n = 100
    a = tvm.nd.array(np.random.uniform(size=n).astype(DTYPE), device)
    b = tvm.nd.array(np.random.uniform(size=n).astype(DTYPE), device)
    c = tvm.nd.array(np.zeros(n, dtype=DTYPE), device)
    lib(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    # Check correct remove
    remote_session.remove(DSO_NAME)


@setup_pure_rpc_configuration
def test_cleanup_workspace_after_session_end(host, port):
    # Arrange
    remote_session = rpc.connect(host, port)
    target = tvm.target.Target(target=f"llvm -mtriple={ARCH}-apple-darwin")
    lib = get_add_module(target)
    path_dso = export_lib(lib)
    remote_session.upload(path_dso)

    # Act
    del remote_session
    remote_session = rpc.connect(host, port)
    try:
        remote_session.download(DSO_NAME)
        status_ok = False
    except Exception as _:
        status_ok = True

    # Assert
    assert status_ok, "Workspace not cleared after RPC Session termination."


@setup_pure_rpc_configuration
def test_graph_executor_remote_run(host, port):
    remote_session = rpc.connect(host, port)
    target = tvm.target.Target(target=f"llvm -mtriple={ARCH}-apple-darwin")
    device = remote_session.cpu(0)

    size = 100
    a = np.random.uniform(size=size).astype(DTYPE)
    b = np.random.uniform(size=size).astype(DTYPE)
    mod, params = get_add_relay_module(a, b)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, target_host=target, params=params)

    path_dso = export_lib(lib)
    remote_session.upload(path_dso)
    lib = remote_session.load_module(DSO_NAME)

    gen_module = graph_executor.GraphModule(lib["default"](device))

    # Check set input
    gen_module.set_input("a", tvm.nd.array(a))
    gen_module.set_input("b", tvm.nd.array(b))
    tvm.testing.assert_allclose(gen_module.get_input(0).numpy(), a)
    tvm.testing.assert_allclose(gen_module.get_input(1).numpy(), b)

    # Check run
    gen_module.run()
    out = gen_module.get_output(0)
    tvm.testing.assert_allclose(out.numpy(), a + b)


if __name__ == '__main__':
    test_pure_rpc()
    test_rpc_proxy()
    test_rpc_tracker()
    test_rpc_tracker_via_proxy()

    test_can_call_remote_function_with_pure_rpc()
    test_can_call_remote_function_with_rpc_proxy()
    test_can_call_remote_function_with_rpc_tracker()
    test_can_call_remote_function_with_rpc_tracker_via_proxy()

    test_basic_functionality_of_rpc_session()
    test_cleanup_workspace_after_session_end()
    test_graph_executor_remote_run()

    server_ios_launcher.ServerIOSLauncher.shutdown_booted_devices()
