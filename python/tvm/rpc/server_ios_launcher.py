import os
import json
import subprocess
from enum import Enum
from typing import Dict, List, AnyStr


class SimulatorSystem(Enum):
    iOS = "iOS"
    tvOS = "tvOS"
    watchOS = "watchOS"


class IOSDevice(Enum):
    iPhone = "iPhone"
    iPod = "iPod"
    iPad = "iPad"


class RPCServerMode(Enum):
    standalone = "standalone"
    proxy = "proxy"
    tracker = "tracker"


def get_list_of_available_simulators() -> Dict[AnyStr, List]:
    proc = subprocess.Popen("xcrun simctl list devices available --json", shell=True,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = proc.communicate()
    available_simulators = json.loads(out)["devices"]
    available_simulators = {key: value for key, value in available_simulators.items() if value != []}
    return available_simulators


def grep_by_system(available_devices: Dict[AnyStr, List], system_name: SimulatorSystem) -> List[Dict]:
    def find_index_of_substr(search_field: List[AnyStr], target: AnyStr) -> int:
        for i, item in enumerate(search_field):
            if target in item:
                return i
        raise ValueError("Search field doesn't content target")
    keys = list(available_devices.keys())
    return available_devices[keys[find_index_of_substr(keys, system_name.value)]]


def grep_by_device(available_devices: List[Dict], device_name: IOSDevice) -> List[Dict]:
    return [item for item in available_devices if device_name.value in item["name"]]


def get_device_uid(target_device: Dict) -> AnyStr:
    return target_device["udid"]


def boot_device(udid: AnyStr) -> None:
    os.system(f"xcrun simctl boot {udid}")
    if not is_booted(udid):
        raise RuntimeError(f"Failed to boot device with unique id: {udid}")


def shutdown_device(udid: AnyStr) -> None:
    os.system(f"xcrun simctl shutdown {udid}")
    if not is_turned_off(udid):
        raise RuntimeError(f"Failed to shut down device with unique id: {udid}")


def deploy_bundle_to_simulator(udid: AnyStr, bundle_path: AnyStr) -> bool:
    ret_code = os.system(f"xcrun simctl install {udid} {bundle_path}")
    if ret_code != 0:
        raise RuntimeError(f"Failed to deploy bundle <{bundle_path}> to device with unique id: {udid}")
    return True


def delete_bundle_from_simulator(udid: AnyStr, bundle_id: AnyStr) -> None:
    ret_code = os.system(f"xcrun simctl uninstall {udid} {bundle_id}")
    if ret_code != 0:
        raise RuntimeError(f"Failed to uninstall bundle <{bundle_id}> from device with unique id: {udid}")


def wait_launch_complete(proc, should_print_host_and_port=False):
    marker_stopped = "PROCESS_STOPPED"
    marker_callstack = "First throw call stack"
    marker_connected = "[IOS-RPC] STATE: 2"  # 0 means state Tracker/Proxy is connected
    marker_server_ip = "[IOS-RPC] IP: "
    marker_server_port = "[IOS-RPC] PORT: "

    host, port = None, None
    for line in proc.stdout:
        found = str(line).find(marker_stopped)
        if found != -1:
            raise RuntimeError("[ERROR] Crash during RCP Server launch.. ")

        found = str(line).find(marker_callstack)
        if found != -1:
            raise RuntimeError("[ERROR] Crash during RCP Server launch.. ")

        found = str(line).find(marker_server_ip)
        if found != -1:
            ip = str(line)[found + len(marker_server_ip):].rstrip("\n")
            host = ip

        found = str(line).find(marker_server_port)
        if found != -1:
            port = str(line)[found + len(marker_server_port):].rstrip("\n")
            port = int(port)

        if str(line).find(marker_connected) != -1:
            # rpc server reports that it successfully connected
            break

    if should_print_host_and_port and (host is None and port is None):
        raise RuntimeError("No messages with actual host and port.")
    return host, port


def launch_ios_rpc(udid: AnyStr, bundle_id: AnyStr, host_url: AnyStr, host_port: int, key: AnyStr, mode: AnyStr):
    cmd = (f"xcrun simctl launch --console {udid} {bundle_id}"
           f" --immediate_connect"
           f" --host_url={host_url}"
           f" --host_port={host_port}"
           f" --key={key}"
           f" --server_mode={mode}"
           f" --verbose")
    proc = subprocess.Popen(cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
                            universal_newlines=True)
    actual_host, actual_port = wait_launch_complete(proc,
                                                    should_print_host_and_port=mode == RPCServerMode.standalone.value)
    return True, actual_host, actual_port


def terminate_ios_rpc(udid: AnyStr, bundle_id: AnyStr) -> None:
    ret_code = os.system(f"xcrun simctl terminate {udid} {bundle_id}")
    if ret_code != 0:
        raise RuntimeError(f"Failed to terminate bundle <{bundle_id}> from device with unique id: {udid}")


def is_booted(udid: AnyStr) -> bool:
    device = find_device(udid)
    return device["state"] == "Booted"


def is_turned_off(udid: AnyStr) -> bool:
    device = find_device(udid)
    return device["state"] == "Shutdown"


def check_booted_device(devices: List[Dict]) -> Dict:
    for device in devices:
        if device["state"] == "Booted":
            return device
    return {}


def find_device(udid: AnyStr) -> Dict:
    available_devices = get_list_of_available_simulators()
    for devices in available_devices.values():
        for device in devices:
            if device["udid"] == udid:
                return device


class ServerIOSLauncher:
    booted_devices = []
    bundle_id = os.environ["BUNDLE_ID"]
    bundle_path = os.environ["BUNDLE_PATH"]

    def __init__(self, mode, host, port, key):
        self.bundle_was_deployed = None
        self.server_was_started = None
        self.external_booted_device = None
        if not ServerIOSLauncher.booted_devices:
            self._boot_or_find_booted_device()

        self.udid = get_device_uid(self.external_booted_device
                                   if self.external_booted_device is not None
                                   else ServerIOSLauncher.booted_devices[-1])
        self.bundle_was_deployed = deploy_bundle_to_simulator(self.udid, self.bundle_path)
        self.server_was_started, _, actual_port = launch_ios_rpc(self.udid, self.bundle_id, host, port, key, mode)

        self.host = host
        self.port = port if mode != RPCServerMode.standalone.value else actual_port

    def terminate(self):
        if self.server_was_started:
            try:
                terminate_ios_rpc(self.udid, self.bundle_id)
                self.server_was_started = False
            except Exception as e:
                print(e)
        if self.bundle_was_deployed:
            try:
                delete_bundle_from_simulator(self.udid, self.bundle_id)
                self.bundle_was_deployed = False
            except Exception as e:
                print(e)

    def __del__(self):
        self.terminate()

    @staticmethod
    def shutdown_booted_devices():
        for device_meta in ServerIOSLauncher.booted_devices:
            try:
                shutdown_device(get_device_uid(device_meta))
            except Exception as e:
                print(e)
        ServerIOSLauncher.booted_devices = []

    def _boot_or_find_booted_device(self):
        target_system = SimulatorSystem.iOS
        target_device_type = IOSDevice.iPhone
        available_devices = get_list_of_available_simulators()
        if not available_devices:
            raise ValueError(f"No devices available in this environment")
        target_devices = grep_by_system(available_devices, target_system)
        if not target_devices:
            raise ValueError(f"No available simulators for target system: {target_system.value}")
        target_devices = grep_by_device(target_devices, target_device_type)
        if not target_devices:
            raise ValueError(f"No available simulators for target device type: {target_device_type.value}")

        maybe_booted = check_booted_device(target_devices)
        if maybe_booted != {}:
            self.external_booted_device = maybe_booted
        else:
            take_latest_model = True
            target_device = target_devices[-1 if take_latest_model else 0]
            boot_device(get_device_uid(target_device))
            ServerIOSLauncher.booted_devices.append(target_device)


class ServerIOSContextManager:
    def __init__(self, mode, host, port, key):
        self.__mode = mode
        self.__host = host
        self.__port = port
        self.__key = key
        self.__ios_rpc_server_launcher = None

    def __enter__(self):
        self.__ios_rpc_server_launcher = ServerIOSLauncher(self.__mode, self.__host, self.__port, self.__key)
        return self.__ios_rpc_server_launcher

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__ios_rpc_server_launcher is not None:
            self.__ios_rpc_server_launcher.terminate()
            self.__ios_rpc_server_launcher = None
