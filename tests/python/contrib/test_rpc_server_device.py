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
"""iOS RPC Server tests."""
# pylint: disable=invalid-name, no-value-for-parameter, missing-function-docstring, import-error
import multiprocessing
import pytest
import numpy as np

import tvm.testing
import tvm.relay.testing
from tvm import te
from tvm import rpc
from tvm import relay, auto_scheduler
from tvm.contrib import utils, xcode, graph_executor
from tvm.autotvm.measure import request_remote
from tvm.auto_scheduler.measure_record import load_records
from tvm.auto_scheduler.measure import MeasureErrorNo
from tvm.auto_scheduler.utils import call_func_with_timeout
from tvm.contrib.popen_pool import PopenWorker, StatusKind
from tvm.rpc import tracker, proxy, server_ios_launcher


HOST_URL = "0.0.0.0"
HOST_PORT = 9190
DEVICE_KEY = "ios_mobile_device"


TEMPORARY_DIRECTORY = utils.tempdir()
ARCH = "x86_64"
SDK = "iphonesimulator"
DSO_NAME = "lib.dylib"
DTYPE = "float32"


np.random.seed(0)


ios_rpc_bundle_description_required = pytest.mark.skipif(
    not server_ios_launcher.ServerIOSLauncher.is_compatible_environment(),
    reason="To run this test, you need to set environment variables required in ServerIOSLauncher.",
)


@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown_actions():
    """Setup and teardown actions for pytest."""

    # No setup actions
    yield
    # Teardown actions:
    server_ios_launcher.ServerIOSLauncher.shutdown_booted_devices()


def setup_rpc_standalone_configuration(f):
    """
    Host  --  RPC server
    """

    def wrapper():
        with server_ios_launcher.ServerIOSContextManager(
            mode=server_ios_launcher.RPCServerMode.standalone.value,
            host=HOST_URL,
            port=HOST_PORT,
            key=DEVICE_KEY,
        ) as ios_server:
            f(host=ios_server.host, port=ios_server.port)

    return wrapper


def setup_rpc_proxy_configuration(f):
    """
    Host -- Proxy -- RPC server
    """

    def wrapper():
        proxy_server = proxy.Proxy(host=HOST_URL, port=HOST_PORT)
        with server_ios_launcher.ServerIOSContextManager(
            mode=server_ios_launcher.RPCServerMode.proxy.value,
            host=proxy_server.host,
            port=proxy_server.port,
            key=DEVICE_KEY,
        ):
            f(host=proxy_server.host, port=proxy_server.port)
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
        with server_ios_launcher.ServerIOSContextManager(
            mode=server_ios_launcher.RPCServerMode.tracker.value,
            host=tracker_server.host,
            port=tracker_server.port,
            key=DEVICE_KEY,
        ):
            f(host=tracker_server.host, port=tracker_server.port)
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
        proxy_server_tracker = proxy.Proxy(
            host=HOST_URL, port=8888, tracker_addr=(tracker_server.host, tracker_server.port)
        )
        with server_ios_launcher.ServerIOSContextManager(
            mode=server_ios_launcher.RPCServerMode.proxy.value,
            host=proxy_server_tracker.host,
            port=proxy_server_tracker.port,
            key=DEVICE_KEY,
        ):
            f(host=tracker_server.host, port=tracker_server.port)
        proxy_server_tracker.terminate()
        tracker_server.terminate()

    return wrapper


def wrapper_for_call_function_with_timeout(timeout, func, args=(), kwargs=None):
    """Wrapper for call_func_with_timeout."""

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
    """Deadlock-safe RPC Session creation."""

    try:
        successful_attempt = True
        results = []
        for _ in range(2):
            ret = wrapper_for_call_function_with_timeout(
                timeout=10, func=session_factory, args=args, kwargs=kwargs
            )
            results.append(ret)
        if not np.all(np.array(results) == StatusKind.COMPLETE):
            raise ValueError("One or more sessions ended incorrectly.")
    except Exception as e:  # pylint: disable=broad-except
        successful_attempt = False
        print(e)
    return successful_attempt


def ios_create_dylib(output, objects, **kwargs):  # pylint: disable=unused-argument
    xcode.create_dylib(output, objects, arch=ARCH, sdk=SDK)


ios_create_dylib.output_format = "dylib"


def export_lib(lib):
    """Export lib to temporary directory."""

    path_dso = TEMPORARY_DIRECTORY.relpath(DSO_NAME)
    lib.export_library(path_dso, fcompile=ios_create_dylib)
    return path_dso


def get_add_relay_module(a_numpy, b_numpy):
    """Get simple relay module that add two tensors."""

    a = relay.var("a", shape=a_numpy.shape, dtype=DTYPE)
    b = relay.var("b", shape=b_numpy.shape, dtype=DTYPE)
    params = {}
    out = tvm.IRModule.from_expr(relay.add(a, b))
    return out, params


def get_add_module(target):
    """Get simple module that add two tensors."""

    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = te.create_schedule(C.op)
    return tvm.build(s, [A, B, C], target=target, target_host=target, name="simple_add")


@pytest.mark.dependency()
@ios_rpc_bundle_description_required
@setup_rpc_standalone_configuration
def test_rpc_standalone(host, port):
    status_ok = try_create_remote_session(session_factory=rpc.connect, args=(host, port))
    assert status_ok


@pytest.mark.dependency()
@ios_rpc_bundle_description_required
@setup_rpc_proxy_configuration
def test_rpc_proxy(host, port):
    status_ok = try_create_remote_session(
        session_factory=rpc.connect, args=(host, port, DEVICE_KEY)
    )
    assert status_ok


@pytest.mark.dependency()
@ios_rpc_bundle_description_required
@setup_rpc_tracker_configuration
def test_rpc_tracker(host, port):
    status_ok = try_create_remote_session(
        session_factory=request_remote, args=(DEVICE_KEY, host, port)
    )
    assert status_ok


@pytest.mark.dependency()
@ios_rpc_bundle_description_required
@setup_rpc_tracker_via_proxy_configuration
def test_rpc_tracker_via_proxy(host, port):
    status_ok = try_create_remote_session(
        session_factory=request_remote, args=(DEVICE_KEY, host, port)
    )
    assert status_ok


@pytest.mark.dependency(depends=["test_rpc_standalone"])
@ios_rpc_bundle_description_required
@setup_rpc_standalone_configuration
def test_can_call_remote_function_with_rpc_standalone(host, port):
    remote_session = rpc.connect(host, port)
    f = remote_session.get_function("runtime.GetFFIString")
    assert f("hello") == "hello"


@pytest.mark.dependency(depends=["test_rpc_proxy"])
@ios_rpc_bundle_description_required
@setup_rpc_proxy_configuration
def test_can_call_remote_function_with_rpc_proxy(host, port):
    remote_session = rpc.connect(host, port, key=DEVICE_KEY)
    f = remote_session.get_function("runtime.GetFFIString")
    assert f("hello") == "hello"


@pytest.mark.dependency(depends=["test_rpc_tracker"])
@ios_rpc_bundle_description_required
@setup_rpc_tracker_configuration
def test_can_call_remote_function_with_rpc_tracker(host, port):
    remote_session = request_remote(DEVICE_KEY, host, port)
    f = remote_session.get_function("runtime.GetFFIString")
    assert f("hello") == "hello"


@pytest.mark.dependency(depends=["test_rpc_tracker_via_proxy"])
@ios_rpc_bundle_description_required
@setup_rpc_tracker_via_proxy_configuration
def test_can_call_remote_function_with_rpc_tracker_via_proxy(host, port):
    remote_session = request_remote(DEVICE_KEY, host, port)
    f = remote_session.get_function("runtime.GetFFIString")
    assert f("hello") == "hello"


@pytest.mark.dependency(depends=["test_rpc_standalone"])
@ios_rpc_bundle_description_required
@setup_rpc_standalone_configuration
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
    with open(path_dso, "rb") as source_lib_file:
        assert downloaded_lib == bytearray(
            source_lib_file.read()
        ), "The downloaded module does not match the loaded module"

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


@pytest.mark.dependency(depends=["test_rpc_standalone"])
@pytest.mark.xfail(reason="Not implemented functionality")
@ios_rpc_bundle_description_required
@setup_rpc_standalone_configuration
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
    except Exception as _:  # pylint: disable=broad-except
        status_ok = True

    # Assert
    assert status_ok, "Workspace not cleared after RPC Session termination."


@pytest.mark.dependency(depends=["test_rpc_standalone"])
@ios_rpc_bundle_description_required
@setup_rpc_standalone_configuration
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


@pytest.mark.xfail(
    strict=False, reason="flaky test (see https://github.com/apache/tvm/issues/9824)"
)
@pytest.mark.dependency(depends=["test_rpc_tracker"])
@ios_rpc_bundle_description_required
@setup_rpc_tracker_configuration
def test_check_auto_schedule_tuning(host, port):  # pylint: disable=too-many-locals
    log_file = TEMPORARY_DIRECTORY.relpath("ios_tuning_stat.log")
    target = tvm.target.Target(target=f"llvm -mtriple={ARCH}-apple-darwin")
    mod, params = relay.testing.mlp.get_workload(batch_size=4, image_shape=(1, 4, 4))

    try:
        status_ok = True
        measure_runner = auto_scheduler.RPCRunner(
            DEVICE_KEY,
            host,
            port,
            min_repeat_ms=1,
            timeout=10,
            n_parallel=multiprocessing.cpu_count(),
        )
        builder = auto_scheduler.LocalBuilder(timeout=10, build_func=ios_create_dylib)
        tune_option = auto_scheduler.TuningOptions(
            builder=builder,
            num_measure_trials=2,
            num_measures_per_round=1,
            runner=measure_runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=0,
        )

        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
        tasks, task_weights = tasks[:2], task_weights[:2]
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tuner.tune(tune_option, search_policy="sketch.random")

        # Check tuning log
        tuning_statistic = list(load_records(log_file))
        for _, measure_result in tuning_statistic:
            if measure_result.error_no != MeasureErrorNo.NO_ERROR:
                raise ValueError(
                    f"Error for MeasureResult. Error code: {measure_result.error_no},"
                    f" for details see MeasureErrorNO."
                )

    except Exception as e:  # pylint: disable=broad-except
        status_ok = False
        print(e)

    assert status_ok, "Tuning failed, see logs."


if __name__ == "__main__":
    tvm.testing.main()
