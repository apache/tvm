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
    pure_server = "pure_server"
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


def shutdown_device(udid: AnyStr) -> None:
    os.system(f"xcrun simctl shutdown {udid}")


def deploy_bundle_to_simulator(udid: AnyStr, bundle_path: AnyStr) -> None:
    os.system(f"xcrun simctl install {udid} {bundle_path}")


def delete_bundle_from_simulator(udid: AnyStr, bundle_id: AnyStr) -> None:
    os.system(f"xcrun simctl uninstall {udid} {bundle_id}")


def launch_ios_rpc(udid: AnyStr, bundle_id: AnyStr, host_url: AnyStr, host_port: int, key: AnyStr, mode: AnyStr) -> None:
    os.system(f"xcrun simctl launch {udid} {bundle_id}"
              f" --immediate_connect --host_url={host_url} --host_port={host_port} --key={key} --server_mode={mode}")


def terminate_ios_rpc(udid: AnyStr, bundle_id: AnyStr) -> None:
    os.system(f"xcrun simctl terminate {udid} {bundle_id}")


class ServerIOSLauncher:
    udid = "booted"
    bundle_id = "org.apache.tvmrpc"
    bundle_path = "/Users/agladyshev/workspace/tvm/build-ios-simulator/apps/ios_rpc/ios_rpc/src/ios_rpc-build/Debug-iphonesimulator/tvmrpc.app"

    def __init__(self, mode, host, port, key):
        deploy_bundle_to_simulator(self.udid, self.bundle_path)
        launch_ios_rpc(self.udid, self.bundle_id, host, port, key, mode)

    def terminate(self):
        terminate_ios_rpc(self.udid, self.bundle_id)
        delete_bundle_from_simulator(self.udid, self.bundle_id)

    def __del__(self):
        self.terminate()
