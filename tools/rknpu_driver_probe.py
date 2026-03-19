#!/usr/bin/env python3
"""Probe useful RK3588 RKNPU driver metrics from userspace.

This is intentionally lightweight:
- opens the same DRM render node the runtime uses
- issues action ioctls for the small set of useful driver metrics
- reads debugfs/procfs nodes if the kernel exposed them

The goal is to let benchmark artifacts capture kernel-visible state without
pulling in the full TVM runtime stack.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


libc = ctypes.CDLL(None, use_errno=True)

DRM_BASE = ord("d")
DRM_COMMAND_BASE = 0x40

NR_ACTION = 0x00
NR_VERSION = 0x00

ACTION_GET_HW_VERSION = 0
ACTION_GET_DRV_VERSION = 1
ACTION_GET_FREQ = 2
ACTION_GET_VOLT = 4
ACTION_ACT_RESET = 6
ACTION_GET_BW_PRIORITY = 7
ACTION_GET_BW_EXPECT = 9
ACTION_GET_BW_TW = 11
ACTION_ACT_CLR_TOTAL_RW_AMOUNT = 13
ACTION_GET_DT_WR_AMOUNT = 14
ACTION_GET_DT_RD_AMOUNT = 15
ACTION_GET_WT_RD_AMOUNT = 16
ACTION_GET_TOTAL_RW_AMOUNT = 17
ACTION_GET_IOMMU_EN = 18
ACTION_POWER_ON = 20
ACTION_POWER_OFF = 21
ACTION_GET_TOTAL_SRAM_SIZE = 22
ACTION_GET_FREE_SRAM_SIZE = 23
ACTION_GET_IOMMU_DOMAIN_ID = 24

DEFAULT_DEBUGFS_ROOT = Path("/sys/kernel/debug/rknpu")
DEFAULT_PROCFS_ROOT = Path("/proc/rknpu")
DEFAULT_DEVFREQ_DIR = Path("/sys/class/devfreq/fdab0000.npu")


class DrmVersion(ctypes.Structure):
    _fields_ = [
        ("version_major", ctypes.c_int),
        ("version_minor", ctypes.c_int),
        ("version_patchlevel", ctypes.c_int),
        ("name_len", ctypes.c_size_t),
        ("name", ctypes.c_void_p),
        ("date_len", ctypes.c_size_t),
        ("date", ctypes.c_void_p),
        ("desc_len", ctypes.c_size_t),
        ("desc", ctypes.c_void_p),
    ]


class RknpuAction(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("value", ctypes.c_uint32),
    ]


def ioctl_cmd(struct_type: type[ctypes.Structure], nr: int) -> int:
    return (3 << 30) | (ctypes.sizeof(struct_type) << 16) | (DRM_BASE << 8) | nr


CMD_VERSION = ioctl_cmd(DrmVersion, NR_VERSION)
CMD_ACTION = ioctl_cmd(RknpuAction, DRM_COMMAND_BASE + NR_ACTION)


def _ioctl(fd: int, cmd: int, arg: ctypes.Structure) -> None:
    ret = libc.ioctl(fd, cmd, ctypes.byref(arg))
    if ret < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _path_status(path: Path) -> Dict[str, Any]:
    try:
        return {"present": path.exists(), "accessible": True}
    except OSError as err:
        return {
            "present": True,
            "accessible": False,
            "errno": err.errno,
            "error": err.strerror,
        }


def _parse_transitions(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    for line in text.splitlines():
        if "Total transition" not in line:
            continue
        try:
            return int(line.split(":", 1)[1].strip())
        except (IndexError, ValueError):
            return None
    return None


def _try_version(fd: int) -> Optional[Dict[str, Any]]:
    name_buf = ctypes.create_string_buffer(256)
    date_buf = ctypes.create_string_buffer(256)
    desc_buf = ctypes.create_string_buffer(256)
    version = DrmVersion()
    version.name_len = len(name_buf)
    version.name = ctypes.cast(name_buf, ctypes.c_void_p).value
    version.date_len = len(date_buf)
    version.date = ctypes.cast(date_buf, ctypes.c_void_p).value
    version.desc_len = len(desc_buf)
    version.desc = ctypes.cast(desc_buf, ctypes.c_void_p).value
    try:
        _ioctl(fd, CMD_VERSION, version)
    except OSError:
        return None
    return {
        "version_major": int(version.version_major),
        "version_minor": int(version.version_minor),
        "version_patchlevel": int(version.version_patchlevel),
        "name": name_buf.value.decode("utf-8", errors="replace"),
        "date": date_buf.value.decode("utf-8", errors="replace"),
        "desc": desc_buf.value.decode("utf-8", errors="replace"),
    }


def _find_device(preferred: Optional[str]) -> tuple[int, str, Dict[str, Any]]:
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.append("/dev/dri/renderD128")
    candidates.extend(f"/dev/dri/card{i}" for i in range(8))
    candidates.extend(f"/dev/dri/renderD{i}" for i in range(129, 136))
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        try:
            fd = os.open(path, os.O_RDWR)
        except OSError:
            continue
        version = _try_version(fd)
        if version and version.get("name", "").startswith("rknpu"):
            return fd, path, version
        os.close(fd)
    raise FileNotFoundError(errno.ENOENT, "could not find an rknpu DRM render node")


def _action(fd: int, action_id: int) -> Dict[str, Any]:
    req = RknpuAction(flags=action_id, value=0)
    try:
        _ioctl(fd, CMD_ACTION, req)
    except OSError as err:
        return {"ok": False, "errno": err.errno, "error": err.strerror}
    return {"ok": True, "value": int(req.value)}


def _read_debug_tree(root: Path) -> Dict[str, Any]:
    status = _path_status(root)
    result: Dict[str, Any] = {"path": str(root), **status}
    if not status.get("present", False) or not status.get("accessible", False):
        return result
    for name in ("version", "load", "power", "freq", "volt", "delayms", "reset", "mm"):
        text = _read_text(root / name)
        if text is not None:
            result[name] = text
    return result


def _read_devfreq(devfreq_dir: Path) -> Dict[str, Any]:
    status = _path_status(devfreq_dir)
    result: Dict[str, Any] = {"path": str(devfreq_dir), **status}
    if not status.get("present", False) or not status.get("accessible", False):
        return result
    result.update(
        {
            "cur_freq": _read_text(devfreq_dir / "cur_freq"),
            "target_freq": _read_text(devfreq_dir / "target_freq"),
            "min_freq": _read_text(devfreq_dir / "min_freq"),
            "max_freq": _read_text(devfreq_dir / "max_freq"),
            "governor": _read_text(devfreq_dir / "governor"),
            "load": _read_text(devfreq_dir / "load"),
            "available_frequencies": _read_text(devfreq_dir / "available_frequencies"),
            "available_governors": _read_text(devfreq_dir / "available_governors"),
        }
    )
    result["transitions_total"] = _parse_transitions(_read_text(devfreq_dir / "trans_stat"))
    return result


def _probe_actions(fd: int, include_unsupported: bool) -> Dict[str, Any]:
    actions = {
        "hw_version": ACTION_GET_HW_VERSION,
        "drv_version": ACTION_GET_DRV_VERSION,
        "freq_hz": ACTION_GET_FREQ,
        "volt_uv": ACTION_GET_VOLT,
        "iommu_enabled": ACTION_GET_IOMMU_EN,
        "total_sram_bytes": ACTION_GET_TOTAL_SRAM_SIZE,
        "free_sram_bytes": ACTION_GET_FREE_SRAM_SIZE,
        "iommu_domain_id": ACTION_GET_IOMMU_DOMAIN_ID,
    }
    if include_unsupported:
        actions.update(
            {
                "bw_priority": ACTION_GET_BW_PRIORITY,
                "bw_expect": ACTION_GET_BW_EXPECT,
                "bw_tw": ACTION_GET_BW_TW,
                "dt_wr_amount": ACTION_GET_DT_WR_AMOUNT,
                "dt_rd_amount": ACTION_GET_DT_RD_AMOUNT,
                "wt_rd_amount": ACTION_GET_WT_RD_AMOUNT,
                "total_rw_amount": ACTION_GET_TOTAL_RW_AMOUNT,
            }
        )
    return {name: _action(fd, action_id) for name, action_id in actions.items()}


def _format_value(name: str, probe: Dict[str, Any]) -> str:
    if not probe.get("ok", False):
        return f"{name}: error errno={probe.get('errno')} {probe.get('error')}"
    return f"{name}: {probe.get('value')}"


def _print_human(summary: Dict[str, Any]) -> None:
    print(f"device: {summary['device_path']}")
    version = summary["drm_version"]
    print(
        "drm_version: "
        f"{version['version_major']}.{version['version_minor']}.{version['version_patchlevel']} "
        f"name={version['name']} date={version['date']}"
    )
    print("action_metrics:")
    for name, probe in summary["action_metrics"].items():
        print(f"  {_format_value(name, probe)}")
    for label in ("debugfs", "procfs", "devfreq"):
        tree = summary[label]
        print(f"{label}: present={tree.get('present', False)} path={tree['path']}")
        if not tree.get("present", False):
            continue
        for key, value in tree.items():
            if key in {"path", "present"}:
                continue
            print(f"  {key}: {value}")
    if summary["notes"]:
        print("notes:")
        for note in summary["notes"]:
            print(f"  - {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="explicit DRM render/card node")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of human text")
    parser.add_argument("--json-out", default=None, help="optional path for JSON artifact")
    parser.add_argument(
        "--debugfs-root",
        default=str(DEFAULT_DEBUGFS_ROOT),
        help="debugfs rknpu directory",
    )
    parser.add_argument(
        "--procfs-root",
        default=str(DEFAULT_PROCFS_ROOT),
        help="procfs rknpu directory",
    )
    parser.add_argument(
        "--devfreq-dir",
        default=str(DEFAULT_DEVFREQ_DIR),
        help="devfreq directory to sample",
    )
    parser.add_argument(
        "--include-unsupported",
        action="store_true",
        help="also probe action ioctls that look unsupported on rk3588 in the reverse-engineered driver tree",
    )
    args = parser.parse_args()

    fd, device_path, version = _find_device(args.device)
    try:
        summary = {
            "device_path": device_path,
            "drm_version": version,
            "action_metrics": _probe_actions(fd, args.include_unsupported),
            "debugfs": _read_debug_tree(Path(args.debugfs_root)),
            "procfs": _read_debug_tree(Path(args.procfs_root)),
            "devfreq": _read_devfreq(Path(args.devfreq_dir)),
            "notes": [
                "Kernel submit ioctl already returns hw_elapse_time and task_counter; this probe only queries action/debug state.",
                "On the reverse-engineered rk3588 driver branch, rw_amount and bw_priority hooks appear present in the API but unsupported in config.",
                "debugfs/procfs nodes depend on CONFIG_ROCKCHIP_RKNPU_DEBUG_FS / CONFIG_ROCKCHIP_RKNPU_PROC_FS.",
            ],
        }
    finally:
        os.close(fd)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_human(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
