#!/usr/bin/env python3
"""Hard sweep for the residual-MLP MVP shape family."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path


EXPECTED_STAGE_IDS = [[6, 11, 2]]
DEFAULT_MS = [1, 15, 16, 63, 64, 65, 127, 128, 129, 257, 511, 512, 1024, 1500]


def _parse_int_list(raw: str) -> list[int]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _run_case(
    demo: Path,
    m: int,
    d_model: int,
    hidden: int,
    warmup: int,
    iters: int,
    seed: int,
    env: dict[str, str],
) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        json_path = tmp.name
    try:
        cmd = [
            sys.executable,
            str(demo),
            "--m",
            str(m),
            "--d-model",
            str(d_model),
            "--hidden",
            str(hidden),
            "--seed",
            str(seed),
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
            "--real-submit",
            "--bridge-debug-checks",
            "--json-out",
            json_path,
        ]
        subprocess.run(cmd, check=True, env=env)
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        try:
            os.unlink(json_path)
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms", default=",".join(str(x) for x in DEFAULT_MS))
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-err", type=float, default=0.01)
    parser.add_argument("--max-non-finite", type=int, default=0)
    parser.add_argument("--expected-stage-ids", default="6,11,2")
    parser.add_argument("--expected-num-submits", type=int, default=1)
    parser.add_argument("--expected-blocked-boundaries", type=int, default=0)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    expected_stage_ids = [[int(x) for x in _parse_int_list(args.expected_stage_ids)]]
    ms = _parse_int_list(args.ms)
    root = Path(__file__).resolve().parent.parent
    demo = root / "tools" / "rknpu_tir_mlp_mvp_demo.py"
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", "python")
    env.setdefault("TVM_LIBRARY_PATH", "build")
    env.setdefault("TVM_RKNPU_BRIDGE_REAL_SUBMIT", "1")
    env.setdefault("TVM_RKNPU_BRIDGE_USE_RELOCS", "1")
    env.setdefault("TVM_RKNPU_BRIDGE_RUN_CPU_AFTER_SUBMIT", "0")
    env.setdefault("TVM_RKNPU_BRIDGE_FAIL_ON_FALLBACK", "1")
    env.setdefault("TVM_RKNPU_BRIDGE_VALIDATE_RELOCS", "1")

    records = []
    failures = []

    for m in ms:
        data = _run_case(
            demo=demo,
            m=m,
            d_model=args.d_model,
            hidden=args.hidden,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
            env=env,
        )
        corr = data["correctness"]
        perf = data["publishable_perf"]
        sched = data["tir_schedule_report"]
        stats = data["runtime_bridge_stats"]

        rec = {
            "m": m,
            "tir_max_err": float(corr["tir_max_err"]),
            "tir_non_finite": int(corr.get("tir_non_finite", 0)),
            "tir_ms": float(perf["runtime_total_ms_tail_median"]),
            "tir_hw_ms": float(perf["runtime_hw_ms_tail_median"]),
            "num_submits": int(sched["num_submits"]),
            "total_tasks": int(sched["total_tasks"]),
            "submit_stage_ids": sched.get("submit_stage_ids"),
            "blocked_boundary_count": int(sched.get("chain_compatibility", {}).get("blocked_boundary_count", 0)),
            "touch_fallback": int(stats.get("touch_fallback", 0)),
            "real_submit_fail": int(stats.get("real_submit_fail", 0)),
            "reloc_submit_fallbacks": int(stats.get("reloc_submit_fallbacks", 0)),
            "reloc_semantic_mismatch": int(stats.get("reloc_semantic_mismatch", 0)),
            "reloc_range_mismatch": int(stats.get("reloc_range_mismatch", 0)),
        }
        records.append(rec)

        print(
            "MLP_MVP_SWEEP "
            f"m={m} tir_max_err={rec['tir_max_err']:.6f} "
            f"tir_non_finite={rec['tir_non_finite']} "
            f"tir_ms={rec['tir_ms']:.3f} "
            f"tir_hw_ms={rec['tir_hw_ms']:.3f} "
            f"submits={rec['num_submits']} tasks={rec['total_tasks']} "
            f"blocked={rec['blocked_boundary_count']} "
            f"stage_ids={rec['submit_stage_ids']}"
        )

        case_failures = []
        if not math.isfinite(rec["tir_max_err"]) or rec["tir_max_err"] > args.max_err:
            case_failures.append(f"tir_max_err={rec['tir_max_err']} exceeds {args.max_err}")
        if rec["tir_non_finite"] > args.max_non_finite:
            case_failures.append(
                f"tir_non_finite={rec['tir_non_finite']} exceeds {args.max_non_finite}"
            )
        if rec["num_submits"] != args.expected_num_submits:
            case_failures.append(
                f"num_submits={rec['num_submits']} expected {args.expected_num_submits}"
            )
        if rec["submit_stage_ids"] != expected_stage_ids:
            case_failures.append(
                f"submit_stage_ids={rec['submit_stage_ids']} expected {expected_stage_ids}"
            )
        if rec["blocked_boundary_count"] != args.expected_blocked_boundaries:
            case_failures.append(
                "blocked_boundary_count="
                f"{rec['blocked_boundary_count']} expected {args.expected_blocked_boundaries}"
            )
        if rec["touch_fallback"] != 0:
            case_failures.append(f"touch_fallback={rec['touch_fallback']} must be 0")
        if rec["real_submit_fail"] != 0:
            case_failures.append(f"real_submit_fail={rec['real_submit_fail']} must be 0")
        if rec["reloc_submit_fallbacks"] != 0:
            case_failures.append(
                f"reloc_submit_fallbacks={rec['reloc_submit_fallbacks']} must be 0"
            )
        if rec["reloc_semantic_mismatch"] != 0:
            case_failures.append(
                f"reloc_semantic_mismatch={rec['reloc_semantic_mismatch']} must be 0"
            )
        if rec["reloc_range_mismatch"] != 0:
            case_failures.append(
                f"reloc_range_mismatch={rec['reloc_range_mismatch']} must be 0"
            )

        for failure in case_failures:
            failures.append(f"m={m}: {failure}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "records": records,
                    "expected_stage_ids": expected_stage_ids,
                    "expected_num_submits": args.expected_num_submits,
                    "expected_blocked_boundaries": args.expected_blocked_boundaries,
                    "failures": failures,
                },
                f,
                indent=2,
            )

    if failures:
        for failure in failures:
            print(f"MLP_MVP_SWEEP_FAIL {failure}")
        raise SystemExit(1)

    print(f"MLP_MVP_SWEEP PASS records={len(records)}")


if __name__ == "__main__":
    main()
