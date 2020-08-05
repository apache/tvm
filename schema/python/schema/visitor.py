from .expr import *

class ExprFunctor:
    def __init__(self):
        self.memo_map = {}

    def visit(self, expr):
        if expr in self.memo_map:
            return self.memo_map[expr]

        if isinstance(expr, ObjectDef):
            res = self.visit_object_def(expr)
        elif isinstance(expr, ObjectRefDef):
            res = self.visit_object_ref_def(expr)
        elif isinstance(expr, FieldDef):
            res = self.visit_field_def(expr)
        else:
            raise Exception("warning unhandled case: {0}".format(type(expr)))

        self.memo_map[expr] = res
        return res

    def visit_object_def(self, _):
        raise NotImplementedError()

    def visit_object_ref_def(self, _):
        raise NotImplementedError()

    def visit_field_def(self, _):
        raise NotImplementedError()

class AttrsCollector(ExprFunctor):
    def __init__(self):
        super(AttrsCollector, self).__init__()
        self.fields = []

    def visit_object_def(self, obj):
        if obj.base is not None:
            self.visit(obj.base)
        if len(obj.fields) > 0:
            self.fields += obj.fields


class CodeGenCPP(ExprFunctor):
    def visit_object_def(self, obj):
        template = \
        "class {name} : public {base_name} {{\n" \
        " public:" \
        "{fields}" \
        "{visit_fields}" \
        "\n" \
        "  static constexpr const char* _type_key = \"{type_key}\";\n" \
        "  TVM_DECLARE_BASE_OBJECT_INFO({name}, {base_name});\n" \
        "}};"
        base_name = obj.base.name
        fields = [""]
        for field in obj.fields:
            fields.append(self.visit(field))
        fields = "\n  ".join(fields)

        collector = AttrsCollector()
        collector.visit(obj)
        attrs = collector.fields
        if obj.fvisit_attrs and len(attrs) > 0:
            attrs_str = [""]
            for field in attrs:
                line = "v->Visit(\"{field_name}\", &{field_name});"
                attrs_str.append(line.format(field_name=field.name))
            attrs_str = '\n  '.join(attrs_str)
        else:
            attrs_str = ""
        src = template.format(name=obj.name,
                              base_name=base_name,
                              fields=fields,
                              visit_fields=attrs_str,
                              type_key=obj.type_key)
        return src

    def visit_object_ref_def(self, objref):
        template = \
        "class {name} : public {base_name} {{\n" \
        " public:\n" \
        "  TVM_DEFINE_OBJECT_REF_METHODS({name}, {base_name}, {obj_name});\n" \
        "}};"
        base_name = objref.base.name
        obj_name = objref.internal_object.name
        src = template.format(name=objref.name,
                              base_name=base_name,
                              obj_name=obj_name)
        return src

    def visit_field_def(self, field):
        src = "{type_name} {name};".format(type_name=field.type_.name,
                                           name=field.name) 
        return src


def generate(expr, language='cpp'):
    if language == 'cpp':
        return CodeGenCPP().visit(expr)
