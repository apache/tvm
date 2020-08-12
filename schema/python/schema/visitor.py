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
    def generate_fvisit_attrs(self, obj, fields):
        if len(fields) == 0:
            return ""
        template = \
        "\n" \
        "  void VisitAttrs(AttrVisitor* v) {{"\
        "{fields_str}" \
        "\n  }}"
        fields_str = [""]
        for field in fields:
            line = "v->Visit(\"{field_name}\", &{field_name});"
            fields_str.append(line.format(field_name=field.name))
        fields_str = '\n    '.join(fields_str)
        return template.format(fields_str=fields_str)

    def generate_fsequal_reduce(self, obj, fields):
        if not obj.fsequal_reduce:
            return ""
        template = \
        "\n" \
        "  void SEqualReduce(const {obj_name}* other, SEqualReducer equal) const {{\n" \
        "    return {fields_str}\n" \
        "  }}"
        fields_str = []
        for field in fields:
            line = "equal({field_name}, other->{field_name})"
            fields_str.append(line.format(field_name=field.name))
        fields_str = ' && '.join(fields_str)
        return template.format(obj_name=obj.name, fields_str=fields_str)

    def generate_fshash_reduce(self, obj, fields):
        if not obj.fshash_reduce:
            return ""
        template = \
        "\n" \
        "  void SHashReduce(SHashReducer hash_reducer) const {{\n" \
        "    {fields_str}\n" \
        "  }}"
        fields_str = []
        for field in fields:
            line = "hash_reducer({field_name});"
            fields_str.append(line.format(field_name=field.name))
        fields_str = '\n    '.join(fields_str)
        return template.format(fields_str=fields_str)

    def visit_object_def(self, obj):
        template = \
        "class {name} : public {base_name} {{\n" \
        " public:" \
        "{fields}" \
        "{fvisit_fields}" \
        "{fsequal_reduce}" \
        "{fshash_reduce}" \
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
        fvisit_fields = self.generate_fvisit_attrs(obj, collector.fields)
        fsequal_reduce = self.generate_fsequal_reduce(obj, collector.fields)
        fshash_reduce = self.generate_fshash_reduce(obj, collector.fields)

        src = template.format(name=obj.name,
                              base_name=base_name,
                              fields=fields,
                              fvisit_fields=fvisit_fields,
                              fsequal_reduce=fsequal_reduce,
                              fshash_reduce=fshash_reduce,
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
