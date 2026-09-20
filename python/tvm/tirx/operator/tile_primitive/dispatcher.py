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
"""Rich dispatcher for TIRx operator dispatchs.

This module adds a structured dispatch table with predicates and
deterministic failure reporting via exceptions.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tvm.ir import Op
from tvm.tirx import PrimFunc
from tvm.tirx.operator import get_tirx_op
from tvm.tirx.tile_primitive import DispatchContext, TilePrimitiveCall


class DispatchFail(RuntimeError):
    """Raised by variants or predicates to provide a reasoned failure."""


@dataclass
class Predicate:
    """A named predicate. The callable can return:

    - bool
    - (bool, str) where the second element is an optional reason on failure
    - raise DispatchFail(reason)
    """

    name: str
    fn: Callable[[TilePrimitiveCall, DispatchContext], Any]
    kwargs: dict[str, Any]

    def evaluate(
        self, op_call: TilePrimitiveCall, sctx: DispatchContext
    ) -> tuple[bool, str | None]:
        try:
            out = self.fn(op_call, sctx, **self.kwargs)
            if isinstance(out, tuple):
                ok, reason = out
                return bool(ok), (str(reason) if not ok and reason is not None else None)
            return bool(out), None
        except DispatchFail as e:  # surface explicit failure reasons
            return False, str(e)
        except Exception as e:  # unexpected predicate exception
            return False, f"predicate exception: {type(e).__name__}: {e}"


def predicate(
    name: str, fn: Callable[[TilePrimitiveCall, DispatchContext], Any], **kwargs
) -> Predicate:
    """Wrap a callable into a named predicate."""

    return Predicate(name=name, fn=fn, kwargs=kwargs)


def fail(reason: str) -> None:
    """Helper for schedule variants to explain why they decline to handle the op."""

    raise DispatchFail(reason)


@dataclass
class DispatchCase:
    variant: str
    priority: int
    preds: list[Predicate]
    # Impl must either return a PrimFunc or raise DispatchFail
    impl: Callable[[TilePrimitiveCall, DispatchContext], PrimFunc]


# Keyed by (Op, target_kind)
_DISPATCH_TABLE: dict[tuple[Op, str], list[DispatchCase]] = {}


def _target_kind_name(sctx: DispatchContext) -> str:
    """Normalize target kind to a stable dispatch key."""

    kind = getattr(getattr(sctx, "target", None), "kind", None)
    return getattr(kind, "name", str(kind))


def register_dispatch(
    op_name: str,
    target_kind: str,
    *,
    variant: str,
    priority: int = 0,
    when: list[Predicate] | None = None,
):
    """Decorator to add a dispatch case for an op/target pair.

    Cases with higher priority run earlier. When list predicates must all pass.
    The impl must return a PrimFunc on success, and must NOT return None.
    To decline handling, raise `fail("reason")` (or `DispatchFail`).
    """

    op = get_tirx_op(op_name)

    def decorator(impl: Callable[[TilePrimitiveCall, DispatchContext], Any]):
        # Wrap impl to forbid returning None; require raise-or-PrimFunc
        def wrapped_impl(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
            res = impl(op_call, sctx)
            if res is None:
                # Enforce raise-or-PrimFunc contract for schedule implementations
                raise DispatchFail(
                    "impl returned None; schedule must return PrimFunc or raise fail()"
                )
            return res  # type: ignore[return-value]

        cases = _DISPATCH_TABLE.setdefault((op, target_kind), [])
        cases.append(
            DispatchCase(variant=variant, priority=priority, preds=when or [], impl=wrapped_impl)
        )
        return impl

    return decorator


def list_registered_schedules() -> dict[str, dict[str, list[str]]]:
    """Return a mapping: op_name -> target_kind -> [variant names]."""

    out: dict[str, dict[str, list[str]]] = {}
    for (op, tgt), cases in _DISPATCH_TABLE.items():
        name = op.name
        out.setdefault(name, {}).setdefault(tgt, [])
        # keep insertion order by default; sort by priority desc for readability
        for c in sorted(cases, key=lambda x: (-x.priority, x.variant)):
            out[name][tgt].append(c.variant)
    return out


def _format_opcall(op_call: TilePrimitiveCall) -> str:
    """Return a readable representation of the failing opcall."""
    # Prefer TVMScript or IR text printer if available on this object
    try:
        script_method = getattr(op_call, "script", None)
        if callable(script_method):
            try:
                return str(script_method())
            except TypeError:
                # Some versions may require keyword args; fall back safely
                return str(script_method())
        astext_method = getattr(op_call, "astext", None)
        if callable(astext_method):
            return str(astext_method())
    except Exception:
        pass
    try:
        s = str(op_call)
        # constrain extremely long single-line prints from repr
        return s
    except Exception:
        pass
    try:
        args_len = len(getattr(op_call, "args", []))
    except Exception:
        args_len = -1
    try:
        op_name = op_call.op.name  # type: ignore[attr-defined]
    except Exception:
        op_name = "<unknown-op>"
    return f"op={op_name}, args={args_len}"


def _format_failure_table(header: str, rows: list[tuple[str, list[str]]]) -> str:
    """Format failures into a readable ASCII table.

    Parameters
    ----------
    header : str
        The header line describing the op/target
    rows : List[Tuple[str, str, Optional[str]]]
        Each row is (variant_label, error_summary, traceback_str)

    Returns
    -------
    str
        The formatted report string
    """
    # Compute column widths
    variant_header = "Variant"
    error_header = "Error"
    variant_col_w = (
        max(len(variant_header), *(len(v) for (v, _) in rows)) if rows else len(variant_header)
    )
    # Error column width needs to consider multi-line cells
    if rows:
        error_col_w = max(
            len(error_header), *(max(len(line) for line in errs) for (_, errs) in rows)
        )
    else:
        error_col_w = len(error_header)

    def hline(sep: str = "+") -> str:
        return f"{sep}{'-' * (variant_col_w + 2)}{sep}{'-' * (error_col_w + 2)}{sep}"

    lines: list[str] = [header]
    if not rows:
        # No rows; keep the header only
        return "\n".join(lines)

    # Table header
    lines.append(hline("+"))
    lines.append(f"| {variant_header.ljust(variant_col_w)} | {error_header.ljust(error_col_w)} |")
    lines.append(hline("+"))

    # Rows (support multi-line Error column)
    for variant, errs in rows:
        if not errs:
            errs = [""]
        for i, err_line in enumerate(errs):
            v_text = variant if i == 0 else ""
            lines.append(f"| {v_text.ljust(variant_col_w)} | {err_line.ljust(error_col_w)} |")
    lines.append(hline("+"))

    return "\n".join(lines)


def run_dispatch(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc | None:
    """Run structured dispatch.

    Returns a PrimFunc on success. Otherwise, raises RuntimeError with
    an aggregated reason report.
    """

    target_kind = _target_kind_name(sctx)
    key = (op_call.op, target_kind)
    cases = _DISPATCH_TABLE.get(key)
    if not cases:
        header = f"TIRx schedule dispatch failed: op={op_call.op.name} target={target_kind}"
        report = _format_failure_table(header, [])
        # Append a simple reason when there are no variants at all
        report = "\n".join([report, "no registered variants for this op/target"])
        raise RuntimeError(report)

    # Collect structured failure rows: (variant_label, error_lines)
    # error_lines: [summary, traceback lines...]
    failure_rows: list[tuple[str, list[str]]] = []
    last_exception: BaseException | None = None

    # If explicit dispatch is set, filter to that variant only
    forced_variant = getattr(op_call, "dispatch", None)
    if forced_variant is not None:
        cases = [c for c in cases if c.variant == forced_variant]
        if not cases:
            msg_header = f"TIRx schedule dispatch failed: op={op_call.op.name} target={target_kind}"
            table = _format_failure_table(msg_header, [])
            msg = "\n".join([table, f"no variant named '{forced_variant}' is registered"])
            raise RuntimeError(msg)

    for case in sorted(cases, key=lambda c: (-c.priority, c.variant)):
        # evaluate predicates
        pred_ok = True
        pred_msgs: list[str] = []
        for pred in case.preds:
            ok, reason = pred.evaluate(op_call, sctx)
            if not ok:
                pred_ok = False
                msg = f"rejected: {pred.name}"
                if reason:
                    msg += f" — {reason}"
                pred_msgs.append(msg)
        if not pred_ok:
            # Include the offending TilePrimitiveCall IR in the error cell
            op_str = _format_opcall(op_call)
            op_lines = [line.rstrip("\n") for line in str(op_str).splitlines()] if op_str else []
            failure_rows.append(
                (
                    f"{case.variant} (prio={case.priority})",
                    ["; ".join(pred_msgs), "opcall:", *op_lines],
                )
            )
            continue

        # run impl
        try:
            res = case.impl(op_call, sctx)
            # Defensive check in case a legacy impl bypassed the wrapper
            if res is None:  # pragma: no cover - legacy guard
                raise DispatchFail("impl returned None (legacy behavior not allowed)")
            return res
        except DispatchFail as e:
            op_str = _format_opcall(op_call)
            op_lines = [line.rstrip("\n") for line in str(op_str).splitlines()] if op_str else []
            failure_rows.append(
                (
                    f"{case.variant} (prio={case.priority})",
                    [f"declined — {e!s}", "opcall:", *op_lines],
                )
            )
        except Exception as e:  # keep searching other variants
            exc_summary = f"exception — {type(e).__name__}: {e}"
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            # Expand traceback into lines
            tb_lines = [line.rstrip("\n") for line in tb_str.splitlines()]
            op_str = _format_opcall(op_call)
            op_lines = [line.rstrip("\n") for line in str(op_str).splitlines()] if op_str else []
            error_lines = [exc_summary, "opcall:", *op_lines, *tb_lines]
            failure_rows.append((f"{case.variant} (prio={case.priority})", error_lines))
            last_exception = e

    # no success
    header = f"TIRx schedule dispatch failed: op={op_call.op.name} target={target_kind}"
    report = _format_failure_table(header, failure_rows)
    if last_exception is not None:
        raise RuntimeError(report) from last_exception
    raise RuntimeError(report)
