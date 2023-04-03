"""Generate python interface for operators"""
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import black

from . import ty
from .registry import OpSchema, get_registered_ops
from .source import Instruction, PyOp, SourceFile, collect_source, keyword_match

INDENT = " " * 4
PATH = Path(os.path.abspath(os.getcwd()))


def _codegen_py_op(schema: OpSchema) -> str:
    template_method = """
def {creator_name}({params}
) -> {ret}:
{doc}
{type_checking}
    _ffi_func = _ffi.get_global_func("{ffi_name}")
    return _ffi_func({param_list})
""".strip()
    template_type_check = """
{param} = tg.check({index}, "{param}", {checker}, {param})
""".strip()

    def _codegen_doc() -> List[str]:
        doc: List[str] = []
        doc.extend(schema.doc)
        if schema.params:
            doc.extend(["", "Parameters", "----------"])
            for param in schema.params:
                param_doc = param.doc or ["TODO(tvm-unity-team): add doc"]
                doc.append(f"{param.name} : {param.py_type()}")
                doc.extend(INDENT + line for line in param_doc)
        if schema.ret:
            doc.extend(["", "Returns", "-------"])
            ret_doc = schema.ret.doc or ["TODO(tvm-unity-team): add doc"]
            doc.append(f"ret : {schema.ret.py_type()}")
            doc.extend(INDENT + line for line in ret_doc)
        doc.append("")
        return doc

    return template_method.format(
        creator_name=schema.creator_name,
        params="\n".join(
            f"{param.name}: {param.py_type()},"
            if isinstance(param.type.default, ty._NotSpecified)  # pylint: disable=protected-access
            else f"{param.name}: {param.py_type()} = {repr(param.type.default)},"
            for param in schema.params
        ),
        ret="Call",
        doc=INDENT + '"' * 3 + ("\n" + INDENT).join(_codegen_doc()) + '"' * 3,
        type_checking="\n".join(
            INDENT
            + template_type_check.format(
                param=param.name,
                index=index,
                checker=param.py_type_guard(),
            )
            for index, param in enumerate(schema.params)
        ),
        ffi_name=schema.ffi_name,
        param_list=", ".join(param.name for param in schema.params),
    )


def main():  # pylint: disable=missing-function-docstring
    sources: List[SourceFile] = collect_source(PATH / "python" / "tvm")
    op_schemas = get_registered_ops()
    insts: List[PyOp] = sum(
        [
            [i for i in source.contents if isinstance(i, PyOp) and i.command == "create"]
            for source in sources
        ],
        [],
    )
    keywords2op = {schema.category + "/" + schema.name: schema for schema in op_schemas}
    inst2keywords: Dict[Instruction, List[str]] = keyword_match(
        insts=insts,
        keywords=list(keywords2op.keys()),
        inst2key=lambda inst: inst.op_name,
    )
    inst2code: Dict[Instruction, List[str]] = defaultdict(list)
    for inst in insts:
        for keyword in inst2keywords[inst]:
            schema = keywords2op[keyword]
            inst2code[inst].extend(_codegen_py_op(schema).splitlines())

    for source in sources:
        new_source: List[str] = []
        for inst in source.contents:
            if inst in inst2code:
                new_source.append(f"{inst.indent} ## (TVM-TOOL) py_op begin {inst.op_name}")
                new_source.extend(inst2code[inst])
                new_source.append(f"{inst.indent} ## (TVM-TOOL) py_op end {inst.op_name}")
            elif isinstance(inst, PyOp) and inst.command == "create":
                print(f"Warning: instruction does not correspond to any op: {inst}")
            else:
                new_source.append(str(inst))
        formatted_source = black.format_str("\n".join(new_source), mode=black.FileMode())
        with open(source.path, "w", encoding="utf-8") as o_f:
            o_f.write(formatted_source)


if __name__ == "__main__":
    main()
