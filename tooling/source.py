"""A tool for in-source code generation"""
import fnmatch
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union

INST_PREFIX = [
    "## (TVM-TOOL)",
    "// (TVM-TOOL)",
]


@dataclass
class Instruction:
    """An instruction to the code generator."""

    indent: str

    @staticmethod
    def parse(line: str) -> Optional["Instruction"]:
        """Parse a line into an instruction."""
        for prefix in INST_PREFIX:
            if line.lstrip().startswith(prefix):
                pos = line.find(prefix)
                indent, (kind, *content) = (
                    line[:pos],
                    line[pos + len(prefix) :].strip().split(),
                )
                if kind == "py_op":
                    return PyOp._parse(indent, content)  # pylint: disable=protected-access
                if kind == "cc_op":
                    return CcOp._parse(indent, content)  # pylint: disable=protected-access
                raise NotImplementedError(f"Unknown instruction kind: {kind}")
        return None

    def __str__(self) -> str:
        """Return the instruction as a string."""
        raise NotImplementedError


@dataclass
class PyOp(Instruction):
    """An instruction that generates python operator(s)."""

    command: str
    op_name: str

    @staticmethod
    def _parse(indent: str, content: List[str]) -> "PyOp":
        command, op_name = content
        assert command in ["create", "begin", "end"]
        return PyOp(
            indent=indent,
            command=command,
            op_name=op_name,
        )

    def __str__(self) -> str:
        return f"{self.indent} ## (TVM-TOOL) py_op {self.command} {self.op_name}"

    def __hash__(self) -> int:
        return hash((self.indent, self.command, self.op_name))


@dataclass
class CcOp(Instruction):
    """An instruction that generates python operator(s)."""

    command: str
    op_name: str

    @staticmethod
    def _parse(indent: str, content: List[str]) -> "CcOp":
        command, op_name = content
        assert command in ["create", "begin", "end"]
        return CcOp(
            indent=indent,
            command=command,
            op_name=op_name,
        )

    def __str__(self) -> str:
        return f"{self.indent} // (TVM-TOOL) cc_op {self.command} {self.op_name}"

    def __hash__(self) -> int:
        return hash((self.indent, self.command, self.op_name))


@dataclass
class SourceFile:
    """A single file in the source tree."""

    path: str
    contents: List[Union[str, Instruction]]

    @staticmethod
    def from_file(path: str) -> Optional["SourceFile"]:
        """Read a file from disk and return a SourceFile object."""
        has_inst = False
        with open(path, "r", encoding="utf-8") as i_f:
            contents: List[Union[str, Instruction]] = []
            for line in map(str.rstrip, i_f):
                inst = Instruction.parse(line)
                if inst is not None:
                    contents.append(inst)
                    has_inst = True
                else:
                    contents.append(line)
        if not has_inst:
            return None
        result = SourceFile(path, contents)

        def _trim_ranged(
            src: SourceFile,
            kind: Type,
        ) -> None:
            def _find(start: int, condition: Callable[[Instruction], bool]) -> Optional[int]:
                for i, inst in enumerate(src.contents[start:]):
                    if isinstance(inst, kind) and condition(inst):
                        return i + start
                return None

            while True:
                i = _find(0, lambda inst: inst.command == "begin")  # type: ignore[attr-defined]
                if i is None:
                    break
                j = _find(i + 1, lambda inst: inst.command == "end")  # type: ignore[attr-defined]
                if j is None:
                    raise ValueError(f"Unmatched instruction: {kind} begin")
                inst = result.contents[i]
                assert isinstance(inst, (PyOp, CcOp))
                inst.command = "create"
                del result.contents[i + 1 : j + 1]

        # pylint: disable=protected-access
        _trim_ranged(result, PyOp)
        _trim_ranged(result, CcOp)
        # pylint: enable=protected-access
        return result

    def replace_instruction(
        self,
        instruction: Instruction,
        new_contents: List[Union[str, Instruction]],
    ):
        """Replace an instruction with a list of new contents."""
        index = self.contents.index(instruction)
        self.contents[index : index + 1] = new_contents

    def __str__(self) -> str:
        """Return the file as a string."""
        return "\n".join(str(s) for s in self.contents)


def collect_source(path: Path) -> List[SourceFile]:
    """Collect all source files in a directory."""
    result: List[SourceFile] = []
    for typed_files in [
        path.glob("**/*.py"),
        path.glob("**/*.cc"),
        path.glob("**/*.h"),
    ]:
        for file in map(str, typed_files):
            try:
                source = SourceFile.from_file(file)
            except Exception as error:  # pylint: disable=broad-except
                print(f"Error parsing source file: {file}: {error}")
            if source is not None:
                result.append(source)
    return result


def keyword_match(
    insts: List[Instruction],
    keywords: List[str],
    inst2key: Callable[[Instruction], Optional[str]],
) -> Dict[Instruction, List[str]]:
    inst2keywords: Dict[Instruction, List[str]] = defaultdict(list)

    exact_matched_keywords: Dict[str, List[Instruction]] = defaultdict(list)
    for inst in insts:
        inst_key = inst2key(inst)
        if inst_key is None or inst_key not in keywords:
            continue
        inst2keywords[inst].append(inst_key)
        exact_matched_keywords[inst_key].append(inst)

    fuzzy_matched_keywords: Dict[str, List[Instruction]] = defaultdict(list)
    for inst in insts:
        inst_key = inst2key(inst)
        if inst_key is None or inst_key in inst2keywords:
            continue
        inst_key_pattern = re.compile(fnmatch.translate(inst_key))
        for keyword in keywords:
            if keyword not in exact_matched_keywords and inst_key_pattern.match(keyword):
                inst2keywords[inst].append(keyword)
                fuzzy_matched_keywords[keyword].append(inst)

    for keyword in keywords:
        matched = len(exact_matched_keywords[keyword]) + len(fuzzy_matched_keywords[keyword])
        if not matched:
            print(f"Warning: keyword not matched: {keyword}")
        elif matched > 1:
            print(f"Warning: keyword matched multiple times: {keyword}")
            for inst in exact_matched_keywords[keyword]:
                print(f"  Exact match: {inst}")
            for inst in fuzzy_matched_keywords[keyword]:
                print(f"  Fuzzy match: {inst}")
    return inst2keywords
