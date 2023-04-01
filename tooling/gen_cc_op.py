"""Generate c++ interface for operators"""
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import ty
from .registry import OpSchema, ParamDef, get_registered_ops
from .source import CcOp, Instruction, SourceFile, collect_source, keyword_match

INDENT = " " * 2
PATH = Path(os.path.abspath(os.getcwd()))
CLANG_FORMAT = "clang-format-14"
CLANG_FORMAT_FILE = str(PATH / ".clang-format")


def _clang_format(
    code: str,
    clang_format=CLANG_FORMAT,
    clang_format_file=CLANG_FORMAT_FILE,
) -> str:
    import tempfile  # pylint: disable=import-outside-toplevel

    with tempfile.NamedTemporaryFile("w") as o_f:
        o_f.write(code)
        o_f.flush()
        os.system(f"{clang_format} -style=file:{clang_format_file} -i {o_f.name}")
        with open(o_f.name, encoding="utf-8") as i_f:
            result = i_f.read()
    if not result.endswith("\n"):
        result += "\n"
    return result


def _codegen_signature(schema: OpSchema) -> str:
    template = """relax::Call {creator_name}({params})"""
    return template.format(
        creator_name=schema.creator_name,
        params=", ".join(f"{param.cc_type(False)} {param.name}" for param in schema.params),
    )


def _codegen_declaration(schema: OpSchema) -> str:
    template = """
/*! {desc}{docs}{ret}
 */
{signature};
""".strip()

    return template.format(
        desc="".join("\n * " + d for d in schema.doc),
        docs="".join(
            f"\n * \\param {param.name} "
            + " * ".join(param.doc or ["TODO(tvm-unity-team): add doc"])
            for param in schema.params
        ),
        ret="\n * \\return " + " * ".join(schema.ret.doc or ["TODO(tvm-unity-team): add doc"]),
        signature=_codegen_signature(schema),
    )


def _codegen_definition(schema: OpSchema) -> str:
    def is_sinfo_param(param: ParamDef) -> bool:
        return isinstance(param.type, ty.StructInfo)

    # define TVM_RELAX_OP_ARG_CHECK(Checker, Arg, Result)
    def _params(min_num_args: Optional[int]) -> str:
        lines: List[str] = [
            "Array<relax::Expr> _args;",
            f"_args.reserve({len(schema.params)});",
        ]
        for param in schema.params:
            if is_sinfo_param(param):
                continue
            arg2relax = param.cc_arg2relax(param.name)
            lines.append(arg2relax)
        if min_num_args is not None:
            lines.append(f"VariadicArgs(&_args, {min_num_args});")
        return "  \n".join(lines)

    def _sinfo() -> str:
        sinfo_args = [p for p in schema.params if is_sinfo_param(p)]
        if not sinfo_args:
            return "{}"
        assert len(sinfo_args) == 1
        return "{" + sinfo_args[0].name + "}"

    return f"""{_codegen_signature(schema)} {{
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.{schema.name}");
  {_params(min_num_args=schema.min_num_args)}
  return Call(op, _args, Attrs(nullptr), {_sinfo()});
}}
TVM_REGISTER_GLOBAL("{schema.ffi_name}").set_body_typed({schema.creator_name});
TVM_RELAX_REGISTER_OP("{schema.name}"){schema.register_sinfo()}{schema.register_legalization()}{schema.register_attrs()};
""".strip()


def _match_inst_op(
    sources: List[SourceFile],
    op_schemas: List[OpSchema],
    keyword_prefix: str,
) -> List[Tuple[Instruction, OpSchema]]:
    insts: List[CcOp] = sum(
        [
            [
                i
                for i in source.contents
                if isinstance(i, CcOp)
                and i.command == "create"
                and i.op_name.startswith(keyword_prefix + "/")
            ]
            for source in sources
        ],
        [],
    )
    keywords2op = {
        f"{keyword_prefix}/{schema.category}/{schema.name}": schema for schema in op_schemas
    }
    inst2keywords: Dict[Instruction, List[str]] = keyword_match(
        insts=insts,
        keywords=list(keywords2op.keys()),
        inst2key=lambda inst: inst.op_name,
    )
    results: List[Tuple[Instruction, OpSchema]] = []
    for inst in insts:
        for keyword in inst2keywords[inst]:
            results.append((inst, keywords2op[keyword]))
    return results


def main():  # pylint: disable=missing-function-docstring
    sources: List[SourceFile] = sum(
        [
            collect_source(PATH / "include" / "tvm" / "relax"),
            collect_source(PATH / "src" / "relax"),
        ],
        [],
    )
    op_schemas = get_registered_ops()

    inst2code: Dict[Instruction, List[str]] = defaultdict(list)
    for inst, schema in _match_inst_op(sources, op_schemas, "decl"):
        inst2code[inst].extend(_codegen_declaration(schema).splitlines())
    for inst, schema in _match_inst_op(sources, op_schemas, "def"):
        inst2code[inst].extend(_codegen_definition(schema).splitlines())

    for source in sources:
        new_source: List[str] = []
        for inst in source.contents:
            if inst in inst2code:
                new_source.append(f"{inst.indent} // (TVM-TOOL) cc_op begin {inst.op_name}")
                new_source.extend(inst2code[inst])
                new_source.append(f"{inst.indent} // (TVM-TOOL) cc_op end {inst.op_name}")
            elif isinstance(inst, CcOp) and inst.command == "create":
                print(f"Warning: instruction does not correspond to any op: {inst}")
            else:
                new_source.append(str(inst))
        formatted_source = _clang_format("\n".join(new_source))
        with open(source.path, "w", encoding="utf-8") as o_f:
            o_f.write(formatted_source)


if __name__ == "__main__":
    main()
