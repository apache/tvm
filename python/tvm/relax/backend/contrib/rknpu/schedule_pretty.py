"""Human-readable formatting helpers for RKNPU schedule reports."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def _shape_text(shape: List[int]) -> str:
    return "x".join(str(int(x)) for x in shape)


def _value_text(name: str, location: str, spec: Dict[str, object], include_slice: bool = False,
                partition: Optional[Dict[str, List[int]]] = None) -> str:
    dtype = str(spec.get("dtype", "float16"))
    shape = [int(x) for x in spec.get("shape", [])]
    slice_text = ""
    if include_slice and partition:
        axes = spec.get("partition_axes")
        if isinstance(axes, list):
            slices: List[str] = []
            any_sliced = False
            for dim, axis_name in zip(shape, axes):
                if isinstance(axis_name, str) and axis_name in partition:
                    start, end = partition[axis_name]
                    slices.append(f"{int(start)}:{int(end)}")
                    any_sliced = True
                else:
                    slices.append(f"0:{int(dim)}")
            if any_sliced:
                slice_text = "[" + ", ".join(slices) + "]"
    return f"{name}@{location}{slice_text}({dtype}[{_shape_text(shape)}])"


def _partition_axis_text(parts: List[List[int]]) -> str:
    return "[" + ", ".join(f"[{int(lo)}:{int(hi)})" for lo, hi in parts) + "]"


def _partition_summary_text(summary: Optional[Dict[str, List[List[int]]]]) -> str:
    if not isinstance(summary, dict):
        return "none"
    pieces: List[str] = []
    rows = summary.get("rows") or []
    cols = summary.get("cols") or []
    if rows:
        pieces.append(f"rows{_partition_axis_text(rows)}")
    if cols:
        pieces.append(f"cols{_partition_axis_text(cols)}")
    return " x ".join(pieces) if pieces else "none"


def _scoped_block_name(stage_name: str, block_name: str) -> str:
    if block_name == "CNA":
        if "matmul" in stage_name:
            return "CNA.matmul"
        if "conv" in stage_name:
            return "CNA.conv2d"
    if block_name == "DPU":
        return f"DPU.{stage_name}"
    return block_name


def _binding(
    symbols: Optional[Dict[str, object]],
    submit_index: int,
    stage_index: int,
) -> Dict[str, object]:
    if not isinstance(symbols, dict):
        return {}
    stage_bindings = symbols.get("stage_bindings", {})
    if isinstance(stage_bindings, dict):
        binding = stage_bindings.get((submit_index, stage_index))
        if isinstance(binding, dict):
            return binding
    return {}


def _location_lines(symbols: Optional[Dict[str, object]]) -> List[str]:
    if not isinstance(symbols, dict):
        return []
    raw_locations = symbols.get("locations")
    if not isinstance(raw_locations, list) or not raw_locations:
        return []
    lines = ["locations("]
    for item in raw_locations:
        if not isinstance(item, dict):
            continue
        location = str(item.get("location", "loc.unknown"))
        kind = str(item.get("kind", "materialized"))
        name = str(item.get("name", "value"))
        dtype = str(item.get("dtype", "float16"))
        shape = [int(x) for x in item.get("shape", [])]
        bytes_value = item.get("bytes")
        if bytes_value is None:
            lines.append(
                f"  {location} = {kind}(name={name}, dtype={dtype}, shape=[{_shape_text(shape)}])"
            )
        else:
            lines.append(
                "  "
                f"{location} = {kind}(name={name}, dtype={dtype}, shape=[{_shape_text(shape)}], "
                f"bytes={int(bytes_value)})"
            )
    lines.append(")")
    return lines


def _read_write_specs(stage: Dict[str, object]) -> Tuple[List[Dict[str, object]], Optional[Dict[str, object]]]:
    signature = stage.get("signature")
    if not isinstance(signature, dict):
        return [], None
    inputs = signature.get("inputs")
    output = signature.get("output")
    return (inputs if isinstance(inputs, list) else [], output if isinstance(output, dict) else None)


def _ref_list(
    refs: List[Dict[str, str]],
    specs: List[Dict[str, object]],
    include_slice: bool = False,
    partition: Optional[Dict[str, List[int]]] = None,
) -> str:
    out: List[str] = []
    for idx, spec in enumerate(specs):
        ref = refs[idx] if idx < len(refs) else {}
        name = str(ref.get("name", f"in{idx}"))
        location = str(ref.get("location", f"loc.in{idx}"))
        out.append(_value_text(name, location, spec, include_slice=include_slice, partition=partition))
    return "[" + ", ".join(out) + "]"


def _single_ref(
    ref: Dict[str, str],
    spec: Optional[Dict[str, object]],
    include_slice: bool = False,
    partition: Optional[Dict[str, List[int]]] = None,
) -> str:
    if not isinstance(spec, dict):
        return f"{ref.get('name', 'out')}@{ref.get('location', 'loc.out')}"
    return _value_text(
        str(ref.get("name", "out")),
        str(ref.get("location", "loc.out")),
        spec,
        include_slice=include_slice,
        partition=partition,
    )


def _stage_lines(
    submit_index: int,
    stage: Dict[str, object],
    symbols: Optional[Dict[str, object]],
) -> List[str]:
    stage_index = int(stage.get("stage_index", 0))
    stage_name = str(stage.get("stage_name", "stage"))
    task_range = stage.get("task_index_range") or [0, 0]
    task_start = int(task_range[0])
    task_end = int(task_range[1])
    binding = _binding(symbols, submit_index, stage_index)
    inputs, output = _read_write_specs(stage)
    reads = binding.get("reads") if isinstance(binding.get("reads"), list) else []
    writes = binding.get("writes") if isinstance(binding.get("writes"), list) else []
    write_ref = writes[0] if writes else {"name": "out", "location": "loc.out"}
    logical_op = str(binding.get("logical_op", f"op.{stage_name}"))
    computes = binding.get("computes")
    blocks = [
        _scoped_block_name(stage_name, str(block))
        for block in stage.get("blocks", [])
    ]
    partition_summary = (
        _partition_summary_text(stage.get("partition_summary"))
        if bool(stage.get("is_partitioned"))
        else "none"
    )

    lines = [f"    pc_task[{task_start}:{task_end})("]
    lines.append(f"      logical_op = {logical_op},")
    if isinstance(computes, str) and computes:
        lines.append(f"      computes   = {computes},")
    lines.append(f"      uses_shorthand = [{', '.join(blocks)}],")
    lines.append(f"      reads      = {_ref_list(reads, inputs)},")
    lines.append(f"      writes     = [{_single_ref(write_ref, output)}],")
    lines.append(f"      split      = {partition_summary},")

    task_summaries = stage.get("task_summaries")
    if isinstance(task_summaries, list) and len(task_summaries) > 1:
        lines.append("      task_slices = [")
        for task in task_summaries:
            if not isinstance(task, dict):
                continue
            submit_task_index = int(task.get("submit_task_index", 0))
            partition = task.get("partition") if isinstance(task.get("partition"), dict) else None
            task_blocks = [
                _scoped_block_name(stage_name, str(block))
                for block in task.get("blocks", [])
            ]
            task_reads = _ref_list(reads, inputs, include_slice=True, partition=partition)
            task_write = _single_ref(write_ref, output, include_slice=True, partition=partition)
            lines.append(
                "        "
                f"pc_task#{submit_task_index}(uses_shorthand=[{', '.join(task_blocks)}], "
                f"reads={task_reads}, writes=[{task_write}]),"
            )
        lines.append("      ]")
    lines.append("    )")
    return lines


def _audit_line(report: Dict[str, object], audit: Optional[Dict[str, object]]) -> str:
    values = {
        "submits": int(report.get("num_submits", 0)),
        "pc_tasks": int(report.get("total_tasks", 0)),
        "blocked_boundaries": int(
            report.get("chain_compatibility", {}).get("blocked_boundary_count", 0)
        ),
    }
    if isinstance(audit, dict):
        values.update(audit)
    parts = [f"{key}={values[key]}" for key in values]
    return "audit(" + ", ".join(parts) + ")"


def format_rknpu_schedule_report(
    report: Dict[str, object],
    symbols: Optional[Dict[str, object]] = None,
    audit: Optional[Dict[str, object]] = None,
) -> str:
    lines: List[str] = []
    location_lines = _location_lines(symbols)
    if location_lines:
        lines.extend(location_lines)
        lines.append("")

    submit_stage_reports = report.get("submit_stage_reports", [])
    if isinstance(submit_stage_reports, list):
        for submit_index, submit in enumerate(submit_stage_reports):
            lines.append(f"submit{submit_index}(")
            lines.append("  pc_task_group(")
            if isinstance(submit, list):
                for stage in submit:
                    if isinstance(stage, dict):
                        lines.extend(_stage_lines(submit_index, stage, symbols))
            lines.append("  )")
            lines.append(")")
            if submit_index + 1 < len(submit_stage_reports):
                lines.append("")

    lines.append("")
    lines.append(_audit_line(report, audit))
    return "\n".join(lines).rstrip()
