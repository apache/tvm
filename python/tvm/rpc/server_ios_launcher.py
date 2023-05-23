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
"""
Python wrapper for running a RPC Server through iOS RPC
on the iOS simulator using the simctl command line tool.
"""
# pylint: disable=invalid-name
import os
import json
import time
import threading
import subprocess
from enum import Enum
from typing import Dict, List, AnyStr


class OSName(Enum):
    """The names of the operating systems available on the simulator."""

    iOS = "iOS"
    tvOS = "tvOS"
    watchOS = "watchOS"


class IOSDevice(Enum):
    """The names of available iOS devices."""

    iPhone = "iPhone"
    iPod = "iPod"
    iPad = "iPad"


class RPCServerMode(Enum):
    """Server modes available in the iOS RPC application."""

    standalone = "standalone"
    proxy = "proxy"
    tracker = "tracker"


def get_list_of_available_simulators() -> Dict[AnyStr, List]:
    """
    List of simulators available on the system. Simulators are presented as a dictionary.
    The dictionary key is the name of the operating system of the simulator.
    The dictionary value is a list of all simulators with a given operating system.
    """

    with subprocess.Popen(
        "xcrun simctl list devices available --json",
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    ) as proc:
        out, _ = proc.communicate()
        available_simulators = json.loads(out)["devices"]
        available_simulators = {
            key: value for key, value in available_simulators.items() if value != []
        }
    return available_simulators


def grep_by_system(available_devices: Dict[AnyStr, List], system_name: OSName) -> List[Dict]:
    """Search for simulators that use the target operating system."""

    def find_index_of_substr(search_field: List[AnyStr], target: AnyStr) -> int:
        for i, item in enumerate(search_field):
            if target in item:
                return i
        raise ValueError("Search field doesn't content target")

    keys = list(available_devices.keys())

    return available_devices[keys[find_index_of_substr(keys, system_name.value)]]


def grep_by_device(available_devices: List[Dict], device_name: IOSDevice) -> List[Dict]:
    """Search for simulators that emulate a given device."""

    return [item for item in available_devices if device_name.value in item["name"]]


def get_device_uid(target_device: Dict) -> AnyStr:
    """Get a unique device ID."""

    return target_device["udid"]


def check_call_with_runtime_error(cmd: AnyStr, error_message: AnyStr) -> None:
    """Calling the function `subprocess.check_call` and catching its possible thrown exception."""

    try:
        subprocess.check_call(cmd.split(" "))
    except subprocess.CalledProcessError as called_process_error:
        raise called_process_error from RuntimeError(error_message)


def boot_device(udid: AnyStr) -> None:
    """Boot the device by its unique ID."""

    cmd = f"xcrun simctl boot {udid}"
    error_message = f"Failed to boot device with unique id: {udid}"
    check_call_with_runtime_error(cmd, error_message)
    if not is_booted(udid):
        raise RuntimeError(error_message)


def shutdown_device(udid: AnyStr) -> None:
    """Shutdown the device by its unique ID."""

    cmd = f"xcrun simctl shutdown {udid}"
    error_message = f"Failed to shut down device with unique id: {udid}"
    check_call_with_runtime_error(cmd, error_message)
    if not is_turned_off(udid):
        raise RuntimeError(error_message)


def deploy_bundle_to_simulator(udid: AnyStr, bundle_path: AnyStr) -> None:
    """Deploy iOS RPC bundle <bundle_path> to simulator with its unique ID <udid>."""

    check_call_with_runtime_error(
        cmd=f"xcrun simctl install {udid} {bundle_path}",
        error_message=f"Failed to deploy bundle <{bundle_path}> to device with unique id: {udid}",
    )


def delete_bundle_from_simulator(udid: AnyStr, bundle_id: AnyStr) -> None:
    """Delete iOS RPC bundle <bundle_id> from simulator with its unique ID <udid>."""

    check_call_with_runtime_error(
        cmd=f"xcrun simctl uninstall {udid} {bundle_id}",
        error_message=f"Failed to uninstall bundle <{bundle_id}> "
        f"from device with unique id: {udid}",
    )


def launch_ios_rpc(
    udid: AnyStr, bundle_id: AnyStr, host_url: AnyStr, host_port: int, key: AnyStr, mode: AnyStr
):  # pylint: disable=too-many-arguments, consider-using-with
    """
    Launch iOS RPC application on simulator with No UI interconnection.

    udid : str
        Unique device ID.

    bundle_id : str
        iOS RPC bundle ID.

    host_url : str
        The tracker/proxy address.

    host_port : int
        The tracker/proxy port.

    key : str
        The key used to identify the device type in tracker.

    mode : str
        Server mode. See RPCServerMode.
    """

    cmd = (
        f"xcrun simctl launch --console {udid} {bundle_id}"
        f" --immediate_connect"
        f" --host_url={host_url}"
        f" --host_port={host_port}"
        f" --key={key}"
        f" --server_mode={mode}"
        f" --verbose"
    )
    proc = subprocess.Popen(
        cmd.split(" "),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )
    return proc


def terminate_ios_rpc(udid: AnyStr, bundle_id: AnyStr) -> None:
    """Terminate iOS RPC application."""

    check_call_with_runtime_error(
        cmd=f"xcrun simctl terminate {udid} {bundle_id}",
        error_message=f"Failed to terminate bundle <{bundle_id}> "
        f"from device with unique id: {udid}",
    )


def is_booted(udid: AnyStr) -> bool:
    """Check that the device has booted."""

    device = find_device(udid)
    return device["state"] == "Booted"


def is_turned_off(udid: AnyStr) -> bool:
    """Check that the device has turned off."""

    device = find_device(udid)
    return device["state"] == "Shutdown"


def check_booted_device(devices: List[Dict]) -> Dict:
    """Check if there is already a booted device. If so, return this device."""

    for device in devices:
        if device["state"] == "Booted":
            return device
    return {}


def find_device(udid: AnyStr) -> Dict:
    """Find device by its unique ID."""

    return_value = {}
    available_devices = get_list_of_available_simulators()
    for devices in available_devices.values():
        for device in devices:
            if device["udid"] == udid:
                return_value = device
    return return_value


class ServerIOSLauncher:
    """
    Python wrapper for launch iOS RPC to simulator.

    mode : str
        Server mode. See RPCServerMode.

    host : str
        The tracker/proxy address.

    port : int
        The tracker/proxy port.

    key : str
        The key used to identify the device type in tracker.
    """

    booted_devices = []
    bundle_id = os.environ.get("BUNDLE_ID")
    bundle_path = os.environ.get("BUNDLE_PATH")

    class ConsoleMarkers(Enum):
        """
        Marker-messages that iOS RPC Server should print to the console output
        when its states change (see apps/ios_rpc/tvmrpc/RPCServer.mm).

        STOPPED : str
            iOS RPC Server process was stopped

        CALLSTACK : str
            Call stack if RPC Server was stopped with an error.

        CONNECTED : str
            RPC Server reports that it successfully connected.

        SERVER_IP : str
            IP on which RPC Server started (for standalone mode).

        SERVER_PORT : str
            HOST on which RPC Server started (for standalone mode).
        """

        STOPPED = "PROCESS_STOPPED"
        CALLSTACK = "First throw call stack"
        CONNECTED = "[IOS-RPC] STATE: 2"
        SERVER_IP = "[IOS-RPC] IP: "
        SERVER_PORT = "[IOS-RPC] PORT: "

    def __init__(self, mode, host, port, key):
        if not ServerIOSLauncher.is_compatible_environment():
            raise RuntimeError(
                "Can't create ServerIOSLauncher instance."
                " No environment variables set for iOS RPC Server."
            )

        self.host = host
        self.port = port

        self.external_booted_device = None
        if not ServerIOSLauncher.booted_devices:
            self._boot_or_find_booted_device()

        self.udid = get_device_uid(
            self.external_booted_device
            if self.external_booted_device is not None
            else ServerIOSLauncher.booted_devices[-1]
        )

        self.bundle_was_deployed = False
        deploy_bundle_to_simulator(self.udid, self.bundle_path)
        self.bundle_was_deployed = True

        self.server_was_started = False
        self.launch_process = launch_ios_rpc(self.udid, self.bundle_id, host, port, key, mode)
        self._wait_launch_complete(
            waiting_time=60,
            hz=10,
            should_print_host_and_port=mode == RPCServerMode.standalone.value,
        )
        self.server_was_started = True

    def terminate(self):
        """Terminate iOS RPC server."""

        if self.bundle_was_deployed and self.server_was_started:
            try:
                terminate_ios_rpc(self.udid, self.bundle_id)
                self.launch_process.terminate()
                self.server_was_started = False
            except RuntimeError as e:
                print(e)
        if self.bundle_was_deployed:
            try:
                delete_bundle_from_simulator(self.udid, self.bundle_id)
                self.bundle_was_deployed = False
            except RuntimeError as e:
                print(e)

    def __del__(self):
        try:
            self.terminate()
        except ImportError:
            pass

    @staticmethod
    def is_compatible_environment():
        """Check that the current environment has the required variables."""

        return bool(os.environ.get("BUNDLE_ID")) and bool(os.environ.get("BUNDLE_PATH"))

    @staticmethod
    def shutdown_booted_devices():
        """Shutdown simulators that have been booted using this class."""

        for device_meta in ServerIOSLauncher.booted_devices:
            try:
                shutdown_device(get_device_uid(device_meta))
            except RuntimeError as e:
                print(e)
        ServerIOSLauncher.booted_devices = []

    def _boot_or_find_booted_device(self):
        """
        Boot the required simulator if there is no suitable booted simulator
        among the available simulators. If there is a suitable booted simulator,
        then take it as a simulator to which the iOS RPC application will be deployed.
        """

        target_system = OSName.iOS
        target_device_type = IOSDevice.iPhone
        available_devices = get_list_of_available_simulators()
        if not available_devices:
            raise ValueError("No devices available in this environment")
        target_devices = grep_by_system(available_devices, target_system)
        if not target_devices:
            raise ValueError(f"No available simulators for target system: {target_system.value}")
        target_devices = grep_by_device(target_devices, target_device_type)
        if not target_devices:
            raise ValueError(
                f"No available simulators for target device type: {target_device_type.value}"
            )

        maybe_booted = check_booted_device(target_devices)
        if maybe_booted:
            self.external_booted_device = maybe_booted
        else:
            take_latest_model = True
            target_device = target_devices[-1 if take_latest_model else 0]
            boot_device(get_device_uid(target_device))
            ServerIOSLauncher.booted_devices.append(target_device)

    def _wait_launch_complete(self, waiting_time, hz, should_print_host_and_port=False):
        # pylint: disable=too-many-locals
        """
        Wait for the iOS RPC server to start.

        waiting_time : int
            The maximum waiting time during which it is necessary
            to receive a message from RPC Server.

        hz : int
            The frequency of checking (in hertz) messages from RPC Server.
            Checks for messages from the server will occur every 1 / hz second.

        should_print_host_and_port : bool
            A flag that indicates that RPC Server should print the host and port
            on which it was started.
            Used for standalone mode.
        """

        class Switch:
            """A simple helper class for boolean switching."""

            def __init__(self):
                self._on = False

            def toggle(self):
                """Toggle flag."""
                self._on = not self._on

            @property
            def on(self):
                """Flag of this switch."""
                return self._on

        def watchdog():
            for _ in range(waiting_time * hz):
                time.sleep(1.0 / hz)
                if switch_have_data.on:
                    break
            if not switch_have_data.on:
                self.launch_process.terminate()
                switch_process_was_terminated.toggle()

        switch_have_data = Switch()
        switch_process_was_terminated = Switch()
        watchdog_thread = threading.Thread(target=watchdog)

        host, port = None, None
        watchdog_thread.start()
        for line in self.launch_process.stdout:
            if not switch_have_data.on:
                switch_have_data.toggle()

            found = str(line).find(ServerIOSLauncher.ConsoleMarkers.STOPPED.value)
            if found != -1:
                raise RuntimeError("[ERROR] Crash during RCP Server launch.. ")

            found = str(line).find(ServerIOSLauncher.ConsoleMarkers.CALLSTACK.value)
            if found != -1:
                raise RuntimeError("[ERROR] Crash during RCP Server launch.. ")

            found = str(line).find(ServerIOSLauncher.ConsoleMarkers.SERVER_IP.value)
            if found != -1:
                ip = str(line)[
                    found + len(ServerIOSLauncher.ConsoleMarkers.SERVER_IP.value) :
                ].rstrip("\n")
                host = ip

            found = str(line).find(ServerIOSLauncher.ConsoleMarkers.SERVER_PORT.value)
            if found != -1:
                port = str(line)[
                    found + len(ServerIOSLauncher.ConsoleMarkers.SERVER_PORT.value) :
                ].rstrip("\n")
                port = int(port)

            if str(line).find(ServerIOSLauncher.ConsoleMarkers.CONNECTED.value) != -1:
                # rpc server reports that it successfully connected
                break
        watchdog_thread.join()

        if switch_process_was_terminated.on:
            raise TimeoutError("Can't get a response from the iOS Server.")
        if should_print_host_and_port:
            if host is None or port is None:
                raise RuntimeError("No messages with actual host and port.")
            self.port = port


class ServerIOSContextManager:
    """
    Context manager for ServerIOSLauncher.
    To work with ServerIOSLauncher, it is preferable to use this class
    so that the terminate method is called in any case.
    """

    def __init__(self, mode, host, port, key):
        self.__mode = mode
        self.__host = host
        self.__port = port
        self.__key = key
        self.__ios_rpc_server_launcher = None

    def __enter__(self):
        self.__ios_rpc_server_launcher = ServerIOSLauncher(
            self.__mode, self.__host, self.__port, self.__key
        )
        return self.__ios_rpc_server_launcher

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__ios_rpc_server_launcher is not None:
            self.__ios_rpc_server_launcher.terminate()
            self.__ios_rpc_server_launcher = None
