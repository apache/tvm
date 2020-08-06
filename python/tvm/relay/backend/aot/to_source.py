from . import little_cpp
from tvm import relay
from tvm.relay.prelude import Prelude

class ExprWithStmt:
    def __init__(self, expr, stmt=""):
        assert isinstance(expr, str)
        assert isinstance(stmt, str)
        assert "ExprWithStmt" not in expr
        assert "ExprWithStmt" not in stmt
        self.expr = expr
        self.stmt = stmt

    def __str__(self):
        return f"ExprWithStmt({self.expr}, {self.stmt})"

    def __repr__(self):
        return self.__str__()

class ToSource:
    def __init__(self, gv_map):
        self.gv_map = gv_map
        self.name_counter = 0
        self.source_content = ""
        self.name_map = {}
        self.local = True
        self.declare = ""
        self.declare_map = {}
        self.input_const = []

    def fresh_global_name(self):
        name = f"global{self.name_counter}"
        self.name_counter += 1
        return name

    def sanitize(self, str):
        return str.replace("-", "_").replace("/", "_")

    def fresh_local_name(self, var=None):
        if var is not None:
            name = f"local_{self.sanitize(var.name_hint)}_{self.name_counter}"
        else:
            name = f"local_{self.name_counter}"
        self.name_counter += 1
        return name

    def fresh_label_name(self):
        name = f"label_{self.name_counter}"
        self.name_counter += 1
        return name

    # return (str, str) with lhs being stmts, and rhs being expression
    def visit(self, node, *, local=True, name=None):
        if isinstance(node, little_cpp.PackedCall):
            res = self.visit_packed_call(node)
        elif isinstance(node, little_cpp.CPPFunction):
            res = self.visit_cpp_function(node, local, name)
        elif isinstance(node, little_cpp.Decl):
            res = self.visit_decl(node)
        elif isinstance(node, little_cpp.Invoke):
            res = self.visit_invoke(node)
        elif isinstance(node, relay.Var):
            res = ExprWithStmt(self.name_map[node])
        elif isinstance(node, relay.GlobalVar):
            res = self.visit_global_var(node)
        elif isinstance(node, relay.Constant):
            res = self.visit_constant(node)
        elif isinstance(node, little_cpp.CPPIf):
            res = self.visit_if(node)
        elif isinstance(node, little_cpp.CPPTuple):
            res = self.visit_tuple(node)
        elif isinstance(node, little_cpp.CPPConstructor):
            res = self.visit_constructor(node)
        elif isinstance(node, little_cpp.CPPMatch):
            res = self.visit_match(node)
        elif isinstance(node, little_cpp.CPPTupleGetItem):
            res = self.visit_tuple_getitem(node)
        elif isinstance(node, little_cpp.CPPRefCreate):
            res = self.visit_ref_create(node)
        elif isinstance(node, little_cpp.CPPRefRead):
            res = self.visit_ref_read(node)
        elif isinstance(node, little_cpp.CPPRefWrite):
            res = self.visit_ref_write(node)
        else:
            raise Exception(str(node))
        assert isinstance(res, ExprWithStmt)
        return res

    def visit_ref_create(self, node):
        vv = self.visit(node.value)
        return ExprWithStmt(f"RefValue({vv.expr})", vv.stmt)

    def visit_ref_read(self, node):
        vr = self.visit(node.ref)
        return ExprWithStmt(f"Downcast<RefValue>({vr.expr})->value", vr.stmt)

    def visit_ref_write(self, node):
        vr = self.visit(node.ref)
        vv = self.visit(node.value)
        stmt = vr.stmt + vv.stmt + f"Downcast<RefValue>({vr.expr})->value={vv.expr};\n"
        return ExprWithStmt("runtime::ADT::Tuple()", stmt)

    def visit_tuple_getitem(self, node):
        vt = self.visit(node.tuple_value)
        return ExprWithStmt(f"Downcast<runtime::ADT>({vt.expr})[{node.index}]", vt.stmt)

    def visit_constructor(self, node):
        args_str, stmt_str = self.visit_args(node.fields)
        return ExprWithStmt(f"TagToCV({node.tag}, {{{args_str}}})")

    def pattern_var(self, pat, var_set):
        if isinstance(pat, relay.PatternConstructor):
            for x in pat.patterns:
                self.pattern_var(x, var_set)
        elif isinstance(pat, relay.PatternVar):
            assert pat.var not in var_set
            var_set.add(pat.var)
        else:
            raise Exception(str(pat))

    def visit_match(self, node):
        vd = self.visit(node.data)
        stmt_str = vd.stmt

        pattern_var_set = set()
        for c in node.clause:
            self.pattern_var(c[0], pattern_var_set)

        for v in pattern_var_set:
            bind_name = self.fresh_local_name()
            self.name_map[v] = bind_name
            stmt_str += f"ObjectRef {bind_name};\n"

        # match data_name to pat, and fill the var accordingly.
        # go to fail_label or ok_label base on failure/success.
        def visit_pattern(pat, data_name, fail_label, ok_label):
            if isinstance(pat, relay.PatternConstructor):
                data_name = f"Downcast<ConstructorValue>({data_name})"
                ok_case = ""
                bind_names = []
                assert len(pat.constructor.inputs) == len(pat.patterns)
                for i, input_type in enumerate(pat.constructor.inputs):
                    bind_name = self.fresh_local_name()
                    bind_names.append(bind_name)
                    ok_case += f"ObjectRef {bind_name} = {data_name}->fields[{i}];\n"
                for bind_name, p in zip(bind_names, pat.patterns):
                    next_label = self.fresh_label_name()
                    ok_case += visit_pattern(p, bind_name, fail_label, next_label)
                    ok_case += f"{next_label}:\n"
                ok_case += f"goto {ok_label};"
                return f"""
                CHECK({data_name}->tag != -1);
                if ({data_name}->tag == {pat.constructor.tag}) {{
                  {ok_case}
                }} else {{
                  goto {fail_label};
                }}
                """
            elif isinstance(pat, relay.PatternVar):
                return f"""
                {self.name_map[pat.var]} = {data_name};
                """
            else:
                raise Exception(str(pat))

        in_name = self.fresh_local_name()
        out_name = self.fresh_local_name()
        stmt_str += f"ObjectRef {in_name} = {vd.expr};\n"
        stmt_str += f"ObjectRef {out_name};\n"
        match_finish_label = self.fresh_label_name()
        for c in node.clause:
            vc = self.visit(c[1])
            fail_label = self.fresh_label_name()
            ok_label = self.fresh_label_name()
            stmt_str += f"""{{
              {visit_pattern(c[0], in_name, fail_label, ok_label)}
            }}
            """
            stmt_str += f"""{{
              {ok_label}:
              {vc.stmt}
              {out_name} = {vc.expr};
              goto {match_finish_label};
            }}
            """
            stmt_str += f"{fail_label}:\n"
        stmt_str += """CHECK(false) << "does not match any";\n"""
        stmt_str += f"{match_finish_label}: ;"
        return ExprWithStmt(out_name, stmt_str)

    def visit_tuple(self, node):
        expr = []
        stmt_str = ""
        for x in node.fields:
            vx = self.visit(x)
            expr.append(vx.expr)
            stmt_str += vx.stmt
        list_name = self.fresh_local_name()
        stmt_str += f"std::vector<ObjectRef> {list_name} = {{{inter(expr)}}};"
        return ExprWithStmt(f"runtime::ADT::Tuple({list_name})", stmt_str)

    def visit_if(self, node):
        vc = self.visit(node.cond)
        vt = self.visit(node.true_branch)
        vf = self.visit(node.false_branch)
        ret_name = self.fresh_local_name()
        stmt = f"ObjectRef {ret_name};"
        stmt += f"""
        {vc.stmt}
        if (NDToBool(ObjectRefToND({vc.expr}))) {{
          {vt.stmt}
          {ret_name} = {vt.expr};
        }} else {{
          {vf.stmt}
          {ret_name} = {vf.expr};
        }}
        """
        return ExprWithStmt(ret_name, stmt)

    def visit_constant(self, const):
        if const not in self.declare_map:
            name = self.fresh_global_name()
            self.declare_map[const] = name
            self.declare += f"ObjectRef {name};\n"
            self.input_const.append((name, const.data.asnumpy()))
        return ExprWithStmt(self.declare_map[const])

    def visit_global_var(self, gv):
        if gv not in self.declare_map:
            name = self.fresh_global_name()
            self.declare_map[gv] = f"{name}"
            vgv = self.visit(self.gv_map[gv], local=False, name=name)
            assert vgv.stmt == ""
            assert vgv.expr == f"{name}"
        return ExprWithStmt(self.declare_map[gv])

    def visit_args(self, args):
        args_str = ""
        stmt_str = ""
        for i, arg in enumerate(args):
            va = self.visit(arg)
            args_str += va.expr
            stmt_str += va.stmt
            if i != len(args) - 1:
                args_str += ", "
        return args_str, stmt_str

    def visit_invoke(self, invoke):
        args_str, stmt_str = self.visit_args(invoke.args)
        func = self.visit(invoke.call)
        return ExprWithStmt(f"Apply({func.expr}, std::vector<ObjectRef>({{{args_str}}}))", stmt_str + func.stmt)

    def visit_decl(self, decl):
        source = ""
        for var, value in decl.bindings:
            local_name = self.fresh_local_name(var)
            self.name_map[var] = local_name
            vv = self.visit(value, name=local_name)
            source += vv.stmt
            source += f"""ObjectRef {local_name} = {vv.expr};"""
        vb = self.visit(decl.body)
        source += vb.stmt
        return ExprWithStmt(vb.expr, source)

    def nd_dtype(self, tt):
        assert isinstance(tt, relay.ty.TensorType)
        if tt.dtype == 'int32':
            return 'dtype_i32'
        elif tt.dtype == 'int8':
            return 'dtype_i8'
        elif tt.dtype == 'float32':
            return 'dtype_f32'
        elif tt.dtype == 'bool':
            return 'dtype_u1'
        raise Exception("unknown tensor dtype: " + str(tt))

    def nd_shape(self, tt):
        return f"{{{inter([str(s) for s in tt.shape])}}}"

    def visit_packed_call(self, call):
        decl_str = ""
        args = []
        for arg in call.args:
            va = self.visit(arg)
            decl_str += va.stmt
            args.append(va.expr)
        args_str = []
        def convert_input(ty, arg):
            if isinstance(ty, relay.ty.TensorType):
                args_str.append(f"{arg}")
            else:
                assert isinstance(ty, relay.ty.TupleType)
                tuple_name = self.fresh_local_name()
                nonlocal decl_str
                decl_str += f"runtime::ADT {tuple_name} = Downcast<runtime::ADT>({arg});\n"
                for i, t in enumerate(ty.fields):
                    convert_input(t, f"{tuple_name}[{i}]")
        assert len(call.args_type) == len(call.args)
        for i in range(len(call.args_type)):
            convert_input(call.args_type[i], args[i])

        def convert_output(ty):
            nonlocal decl_str
            if isinstance(ty, relay.ty.TensorType):
                tensor_name = self.fresh_local_name()
                decl_str += f"NDArray {tensor_name} = NDArray::Empty({self.nd_shape(ty)}, {self.nd_dtype(ty)}, context);\n"
                args_str.append(f"{tensor_name}")
                return tensor_name
            else:
                assert isinstance(ty, relay.ty.TupleType)
                list_name = self.fresh_local_name()
                list_members = inter([convert_output(t) for t in ty.fields])
                decl_str += f"std::vector<ObjectRef> {list_name} = {{{list_members}}};"
                return f"runtime::ADT::Tuple({list_name})"
        out = convert_output(call.ret_type)
        return ExprWithStmt(out, f"""
            {decl_str}
            const PackedFunc *pf = runtime::Registry::Get("{call.name}");
            CHECK(pf);
            (*pf)({inter(args_str)});
        """)

    def visit_cpp_function(self, func, local, name):
        vec = self.fresh_local_name()
        body = ""

        end = len(func.params) - 1
        for i, param in enumerate(func.params):
            pname = self.fresh_local_name(param)
            self.name_map[param] = pname
            body += f"ObjectRef {pname} = {vec}.at({i});\n"

        body += f"ObjectRef {name} = self;\n"
        vb = self.visit(func.body)
        body = body + vb.stmt + f"""return {vb.expr};"""
        expr = f"""FunctionValueNode::make([=](const std::vector<ObjectRef>& {vec}, const ObjectRef& self) {{
                {body}
            }});
            """

        if local:
            return ExprWithStmt(expr)
        else:
            if name is None:
                name = self.fresh_global_name()
            self.declare += f"""
            static ObjectRef {name}_func() {{
              static ObjectRef ret = {expr};
              return ret;
            }}
            ObjectRef {name} = {name}_func();
            """
            return ExprWithStmt(f"{name}")

    def mk_register_api(self, name: str, func) -> str:
        vf = self.visit(func, local=False)
        assert vf.stmt == ""
        source = self.declare

        args = ""
        if isinstance(func, relay.GlobalVar):
            func = self.gv_map[func]
        end = len(func.params) - 1
        init = ""
        for i, (input_name, _) in enumerate(self.input_const):
            init += f"{input_name} = args[{i}];\n"
        for i in range(len(func.params)):
            args += f"args[{i+len(self.input_const)}]"
            if i != end:
                args += ", "

        source += f"""
        TVM_REGISTER_GLOBAL("{name}")
        .set_body([](TVMArgs args, TVMRetValue* ret) {{
            {init}
            std::initializer_list<ObjectRef> ilist = {{{args}}};
            *ret = Apply({vf.expr}, std::vector<ObjectRef>(ilist));
        }});
        """
        return source

def inter(strs, sep=", "):
    ret = ""
    for i in range(len(strs)):
        ret += strs[i]
        if i != len(strs) - 1:
            ret += sep
    return ret

def mk_file(body, ctx):
    return f"""
    #include <tvm/runtime/container.h>
    #include <tvm/runtime/registry.h>
    #include <tvm/ir/env_func.h>
    #include <tvm/relay/function.h>
    #include <tvm/relay/interpreter.h>
    #include <iostream>

    using namespace tvm;
    using namespace runtime;
    using namespace relay;

    static DLDataType dtype_f32 = DLDataType {{ .code = DLDataTypeCode::kDLFloat, .bits = 32, .lanes = 1 }};
    static DLDataType dtype_u32 = DLDataType {{ .code = DLDataTypeCode::kDLUInt, .bits = 32, .lanes = 1 }};
    static DLDataType dtype_u1 = DLDataType {{ .code = DLDataTypeCode::kDLUInt, .bits = 1, .lanes = 1 }};
    static DLDataType dtype_i32 = DLDataType {{ .code = DLDataTypeCode::kDLInt, .bits = 32, .lanes = 1 }};
    static DLDataType dtype_i8 = DLDataType {{ .code = DLDataTypeCode::kDLInt, .bits = 8, .lanes = 1 }};
    static DLContext context = DLContext {{ .device_type = DLDeviceType({ctx.device_type}), .device_id = {ctx.device_id} }};

    static bool NDToBool(const NDArray& nd) {{
      DLContext cpu_ctx;
      cpu_ctx.device_type = kDLCPU;
      cpu_ctx.device_id = 0;
      NDArray cpu_array = nd.CopyTo(cpu_ctx);
      CHECK_EQ(DataType(cpu_array->dtype), DataType::Bool());
      return reinterpret_cast<uint8_t*>(cpu_array->data)[0];
    }}

    static NDArray ObjectRefToND(const ObjectRef& v) {{
      return Downcast<runtime::NDArray>(v);
    }}

    static ConstructorValue TagToCV(size_t tag, const tvm::Array<ObjectRef>& fields) {{
      ObjectPtr<ConstructorValueObj> n = make_object<ConstructorValueObj>();
      ObjectPtr<ConstructorNode> con = make_object<ConstructorNode>();
      con->tag = tag;
      n->tag = tag;
      n->constructor = Constructor(con);
      n->fields = fields;
      return ConstructorValue(n);
    }}

    /*! \\brief A Function value. */
    class FunctionValue;

    using function_value_t = std::function<ObjectRef(const std::vector<ObjectRef>&, const ObjectRef&)>;
    struct FunctionValueNode : Object {{
      function_value_t f;

      FunctionValueNode() {{ }}

      void VisitAttrs(tvm::AttrVisitor* v) {{ }}

      TVM_DLL static FunctionValue make(const function_value_t& f);

      static constexpr const char* _type_key = "relay.FunctionValue";
      TVM_DECLARE_FINAL_OBJECT_INFO(FunctionValueNode, Object);
    }};

    class FunctionValue : public ObjectRef {{
     public:
      TVM_DEFINE_OBJECT_REF_METHODS(FunctionValue, ObjectRef, FunctionValueNode);
    }};

    FunctionValue FunctionValueNode::make(const function_value_t& f) {{
      ObjectPtr<FunctionValueNode> n = make_object<FunctionValueNode>();
      n->f = f;
      return FunctionValue(n);
    }}

    ObjectRef Apply(const ObjectRef& op, const std::vector<ObjectRef>& args) {{
      return Downcast<FunctionValue>(op)->f(args, op);
    }}

    {body}
    """

def to_source(mod, program, gv_map, ctx, name) -> str:
    convert = ToSource(gv_map)
    ret = mk_file(convert.mk_register_api(name, program), ctx)
    return [value for name, value in convert.input_const], ret
