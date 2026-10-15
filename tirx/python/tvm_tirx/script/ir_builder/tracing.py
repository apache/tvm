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
"""Executed Python source locations for native IR builders."""

import linecache
import sys
import sysconfig
from pathlib import Path

import tvm_ffi

from tvm_tirx import ir

from . import _ffi_api
from .base import IRBuilder


class LineTraceScope:
    """Track file-backed Python lines under one builder, composing helper locations."""

    def __init__(self, builder, *, exclude_paths=()):
        self.builder = builder
        self.excluded = (
            Path(__file__).resolve().parents[2],
            Path(tvm_ffi.__file__).resolve().parent,
            *(Path(path).resolve() for path in exclude_paths),
        )
        self.stdlib = Path(sysconfig.get_paths()["stdlib"]).resolve()
        self.site_roots = tuple(
            Path(sysconfig.get_paths()[key]).resolve() for key in ("purelib", "platlib")
        )
        self.files = {}
        self.locations = {}
        self.frames = {}
        self.previous_callbacks = {}
        self.slot = None
        self.saved_traces = {}
        self.active_sources = ()
        self.previous_trace = None

    def _is_user_source(self, filename):
        if filename not in self.files:
            path = Path(filename).resolve()
            excluded = any(path.is_relative_to(root) for root in self.excluded)
            stdlib = path.is_relative_to(self.stdlib) and not any(
                path.is_relative_to(root) for root in self.site_roots
            )
            self.files[filename] = not filename.startswith("<") and not excluded and not stdlib
        return self.files[filename]

    def source_loc(self, filename, line):
        """Return a whole-line location, or None for excluded/generated source."""
        if not self._is_user_source(filename):
            return None
        key = (filename, line)
        if key not in self.locations:
            text = linecache.getline(filename, line).rstrip("\r\n")
            self.locations[key] = ir.SourceLoc(
                ir.SourceName(filename), line, 1, line, max(2, len(text) + 1)
            )
        return self.locations[key]

    def _activate(self, frame):
        if frame not in self.saved_traces:
            self.saved_traces[frame] = (frame.f_trace, frame.f_trace_lines, frame.f_trace_opcodes)
        frame.f_trace = self._trace
        frame.f_trace_lines = True

    def __enter__(self):
        self.previous_trace = sys.gettrace()
        self.root_frame = sys._getframe(1)
        # Own one stable slot below any explicit with_loc scopes in the body.
        self.slot = _ffi_api.IRBuilderPushLoc(self.builder, ir.UnknownLoc())
        try:
            if self._is_user_source(self.root_frame.f_code.co_filename):
                if self.root_frame.f_trace is not None:
                    self.previous_callbacks[self.root_frame] = self.root_frame.f_trace
                self._activate(self.root_frame)
                self.frames[self.root_frame] = self.source_loc(
                    self.root_frame.f_code.co_filename, self.root_frame.f_lineno
                )
                self._restore(self.root_frame)
            sys.settrace(self._trace)
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, *_):
        sys.settrace(self.previous_trace)
        for frame, (trace, lines, opcodes) in self.saved_traces.items():
            frame.f_trace = trace
            frame.f_trace_lines = lines
            frame.f_trace_opcodes = opcodes
        self.saved_traces.clear()
        _ffi_api.IRBuilderPopLoc(self.builder)
        self.slot = None
        self.active_sources = ()
        self.frames.clear()
        self.previous_callbacks.clear()
        self.root_frame = None

    def _restore(self, frame):
        locs = []
        while frame is not None:
            if frame in self.frames:
                locs.append(self.frames[frame])
            frame = frame.f_back
        sources = tuple(id(loc) for loc in locs)
        if sources == self.active_sources:
            return
        self.active_sources = sources
        loc = ir.UnknownLoc()
        if locs:
            loc = locs[-1]
            for callee in reversed(locs[:-1]):
                loc = ir.CallSiteLoc(callee, loc)
        _ffi_api.IRBuilderUpdateLoc(self.builder, self.slot, loc)

    def _trace(self, frame, event, arg):
        callback = self.previous_trace if event == "call" else self.previous_callbacks.get(frame)
        if callback is not None:
            next_callback = callback(frame, event, arg)
            if next_callback is not None:
                self.previous_callbacks[frame] = next_callback
            else:
                self.previous_callbacks.pop(frame, None)
        user = (
            self._is_user_source(frame.f_code.co_filename)
            and IRBuilder.is_in_scope()
            and IRBuilder.current().same_as(self.builder)
        )
        if user and event == "call":
            self._activate(frame)
        elif user and event == "line":
            self.frames[frame] = self.source_loc(frame.f_code.co_filename, frame.f_lineno)
            self._restore(frame)
        elif event == "return":
            previous = self.previous_callbacks.pop(frame, None)
            if frame in self.frames:
                self.frames.pop(frame)
                self._restore(frame.f_back)
            if frame in self.saved_traces and frame is not self.root_frame:
                trace, lines, opcodes = self.saved_traces.pop(frame)
                frame.f_trace = previous if previous is not None else trace
                frame.f_trace_lines = lines
                frame.f_trace_opcodes = opcodes
                return frame.f_trace
        return self._trace if user or frame in self.previous_callbacks else None
