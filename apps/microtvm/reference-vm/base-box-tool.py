#!/usr/bin/env python3
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


import abc
import argparse
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys


_LOG = logging.getLogger(__name__)


THIS_DIR = os.path.realpath(os.path.dirname(__file__) or ".")


# List of vagrant providers supported by this tool
ALL_PROVIDERS = (
    "parallels",
    "virtualbox",
)


def parse_virtaulbox_devices():
    output = subprocess.check_output(["VBoxManage", "list", "usbhost"], encoding="utf-8")
    devices = []
    current_dev = {}
    for line in output.split("\n"):
        if not line.strip() and current_dev:
            devices.append(current_dev)
            current_dev = {}
            continue

        key, value = line.split(":", 1)
        value = value.lstrip(" ")
        current_dev[key] = value

    if current_dev:
        devices.append(current_dev)
    return devices


VIRTUALBOX_VID_PID_RE = re.compile(r"0x([0-9A-Fa-f]{4}).*")


def attach_virtualbox(uuid, vid_hex=None, pid_hex=None, serial=None):
    usb_devices = parse_virtualbox_devices()
    for dev in usb_device:
        m = VIRTUALBOX_VID_PID_RE.match(dev["VendorId"])
        if not m:
            _LOG.warning("Malformed VendorId: %s", dev["VendorId"])
            continue

        dev_vid_hex = m.group(1)

        m = VIRTUALBOX_VID_PID_RE.match(dev["ProductId"])
        if not m:
            _LOG.warning("Malformed ProductId: %s", dev["ProductId"])
            continue

        dev_pid_hex = m.group(1)

        if (
            vid_hex == dev_vid_hex
            and pid_hex == dev_pid_hex
            and (serial is None or serial == dev["SerialNumber"])
        ):
            rule_args = [
                "VBoxManage",
                "usbfilter",
                "add",
                "--target",
                uuid,
                "--vendorid",
                vid_hex,
                "--productid",
                pid_hex,
            ]
            if serial is not None:
                rule_args.extend(["--serialnumber", serial])
            subprocess.check_call(rule_args)
            subprocess.check_call(["VBoxManage", "controlvm", uuid, "usbattach", dev["UUID"]])
            return

    raise Exception(
        f"Device with vid={vid_hex}, pid={pid_hex}, serial={serial!r} not found:\n{usb_devices!r}"
    )


def attach_parallels(uuid, vid_hex=None, pid_hex=None, serial=None):
    usb_devices = json.loads(
        subprocess.check_output(["prlsrvctl", "usb", "list", "-j"], encoding="utf-8")
    )
    for dev in usb_devices:
        _, dev_vid_hex, dev_pid_hex, _, _, dev_serial = dev["System name"].split("|")
        if (
            vid_hex == dev_vid_hex
            and pid_hex == dev_pid_hex
            and (serial is None or serial == dev_serial)
        ):
            subprocess.check_call(["prlsrvctl", "usb", "set", dev["Name"], uuid])
            subprocess.check_call(["prlctl", "set", uuid, "--device-connect", dev["Name"]])
            return

    raise Exception(
        f"Device with vid={vid_hex}, pid={pid_hex}, serial={serial!r} not found:\n{usb_devices!r}"
    )


ATTACH_USB_DEVICE = {
    "parallels": attach_parallels,
    "virtualbox": attach_virtualbox,
}


def generate_packer_config(file_path, providers):
    builders = []
    for provider_name in providers:
        builders.append(
            {
                "type": "vagrant",
                "output_dir": f"output-packer-{provider_name}",
                "communicator": "ssh",
                "source_path": "generic/ubuntu1804",
                "provider": provider_name,
                "template": "Vagrantfile.packer-template",
            }
        )

    with open(file_path, "w") as f:
        json.dump(
            {
                "builders": builders,
            },
            f,
            sort_keys=True,
            indent=2,
        )


def build_command(args):
    generate_packer_config(
        os.path.join(THIS_DIR, args.platform, "base-box", "packer.json"),
        args.provider.split(",") or ALL_PROVIDERS,
    )
    subprocess.check_call(
        ["packer", "build", "packer.json"], cwd=os.path.join(THIS_DIR, args.platform, "base-box")
    )


REQUIRED_TEST_CONFIG_KEYS = {
    "vid_hex": str,
    "pid_hex": str,
    "test_cmd": list,
}


VM_BOX_RE = re.compile(r'(.*\.vm\.box) = "(.*)"')


# Paths, relative to the platform box directory, which will not be copied to release-test dir.
SKIP_COPY_PATHS = [".vagrant", "base-box"]


def test_command(args):
    user_box_dir = os.path.join(THIS_DIR, args.platform)
    base_box_dir = os.path.join(THIS_DIR, args.platform, "base-box")
    test_config_file = os.path.join(base_box_dir, "test-config.json")
    with open(test_config_file) as f:
        test_config = json.load(f)
        for key, expected_type in REQUIRED_TEST_CONFIG_KEYS.items():
            assert key in test_config and isinstance(
                test_config[key], expected_type
            ), f"Expected key {key} of type {expected_type} in {test_config_file}: {test_config!r}"

    providers = args.provider.split(",")
    provider_passed = {p: False for p in providers}

    release_test_dir = os.path.join(THIS_DIR, "release-test")

    for provider_name in providers:
        if os.path.exists(release_test_dir):
            subprocess.check_call(["vagrant", "destroy", "-f"], cwd=release_test_dir)
            shutil.rmtree(release_test_dir)

        for dirpath, _, filenames in os.walk(user_box_dir):
            rel_path = os.path.relpath(dirpath, user_box_dir)
            if any(
                rel_path == scp or rel_path.startswith(f"{scp}{os.path.sep}")
                for scp in SKIP_COPY_PATHS
            ):
                continue

            dest_dir = os.path.join(release_test_dir, rel_path)
            os.makedirs(dest_dir)
            for filename in filenames:
                shutil.copy2(os.path.join(dirpath, filename), os.path.join(dest_dir, filename))

        release_test_vagrantfile = os.path.join(release_test_dir, "Vagrantfile")
        with open(release_test_vagrantfile) as f:
            lines = list(f)

        found_box_line = False
        with open(release_test_vagrantfile, "w") as f:
            for line in lines:
                m = VM_BOX_RE.match(line)
                if not m:
                    f.write(line)
                    continue

                box_package = os.path.join(
                    base_box_dir, f"output-packer-{provider_name}", "package.box"
                )
                f.write(f'{m.group(1)} = "{os.path.relpath(box_package, release_test_dir)}"\n')
                found_box_line = True

        if not found_box_line:
            _LOG.error(
                "testing provider %s: couldn't find config.box.vm = line in Vagrantfile; unable to test",
                provider_name,
            )
            continue

        subprocess.check_call(
            ["vagrant", "up", f"--provider={provider_name}"], cwd=release_test_dir
        )
        try:
            with open(
                os.path.join(
                    release_test_dir, ".vagrant", "machines", "default", provider_name, "id"
                )
            ) as f:
                machine_uuid = f.read()
            ATTACH_USB_DEVICE[provider_name](
                machine_uuid,
                vid_hex=test_config["vid_hex"],
                pid_hex=test_config["pid_hex"],
                serial=args.test_device_serial,
            )
            tvm_home = os.path.realpath(os.path.join(THIS_DIR, "..", "..", ".."))

            def _quote_cmd(cmd):
                return " ".join(shlex.quote(a) for a in cmd)

            test_cmd = _quote_cmd(["cd", tvm_home]) + " && " + _quote_cmd(test_config["test_cmd"])
            subprocess.check_call(
                ["vagrant", "ssh", "-c", f"bash -ec '{test_cmd}'"], cwd=release_test_dir
            )
            provider_passed[provider_name] = True

        finally:
            subprocess.check_call(["vagrant", "destroy", "-f"], cwd=release_test_dir)
            shutil.rmtree(release_test_dir)

        if not all(provider_passed[p] for p in provider_passed.keys()):
            sys.exit(
                "some providers failed release test: "
                + ",".join(name for name, passed in provider_passed if not passed)
            )


def release_command(args):
    #  subprocess.check_call(["vagrant", "cloud", "version", "create", f"tlcpack/microtvm-{args.platform}", args.version])
    if not args.version:
        sys.exit(f"--version must be specified")

    for provider_name in args.provider.split(","):
        subprocess.check_call(
            [
                "vagrant",
                "cloud",
                "publish",
                "-f",
                f"tlcpack/microtvm-{args.platform}",
                args.version,
                provider_name,
                os.path.join(
                    THIS_DIR,
                    args.platform,
                    "base-box",
                    f"output-packer-{provider_name}/package.box",
                ),
            ]
        )


ALL_COMMANDS = {
    "build": build_command,
    "test": test_command,
    "release": release_command,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automates building, testing, and releasing a base box"
    )
    parser.add_argument(
        "command", default=",".join(ALL_COMMANDS), choices=ALL_COMMANDS, help="Action to perform."
    )
    parser.add_argument(
        "platform",
        help="Name of the platform VM to act on. Must be a sub-directory of this directory.",
    )
    parser.add_argument(
        "--provider",
        help="Name of the provider or providers to act on; if not specified, act on all",
    )
    parser.add_argument(
        "--test-device-serial", help="If given, attach the test device with this USB serial number"
    )
    parser.add_argument("--version", help="Version to release. Must be specified with release.")

    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.sep in args.platform or not os.path.isdir(os.path.join(THIS_DIR, args.platform)):
        sys.exit(f"<platform> must be a sub-direcotry of {THIS_DIR}; got {args.platform}")

    todo = []
    for phase in args.command.split(","):
        if phase not in ALL_COMMANDS:
            sys.exit(f"unknown command: {phase}")

        todo.append(ALL_COMMANDS[phase])

    for phase in todo:
        phase(args)


if __name__ == "__main__":
    main()
