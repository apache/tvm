import re
from collections import namedtuple
from .expr import *
from .registry import lookup

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
        if not obj.fvisit_attrs or len(fields) == 0:
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


TextGroup = namedtuple("TextGroup", ['lines', 'is_normal'])

def _process_schema_group(group):
    lines = group.lines
    property_lines = []
    properties = {}
    regex = r"//[\s]*([\D]+):[\s]*([a-zA-Z]+)"
    for num, line in enumerate(lines):
        if re.search(regex, line):
            matches = re.findall(regex, line)
            for match in matches:
                properties[match[0]] = match[1]
            property_lines.append(line)
    assert 'name' in properties
    assert 'schema-type' in properties
    expr = lookup(properties['name'], properties['schema-type'])
    content = generate(expr)
    content = content.split('\n')
    return [lines[0]] + property_lines + content + [lines[-1]]


def process(fname):
    with open(fname, 'r') as fin: 
        text = fin.read()
    lines = text.split('\n')
    begin_num = []
    end_num = []
    for num, line in enumerate(lines):
        if re.search(r"//[\s]*SCHEMA-BEGIN", line):
            begin_num.append(num)
        if re.search(r"//[\s]*SCHEMA-END", line):
            end_num.append(num)

    assert(len(begin_num) == len(end_num))
    num_pairs = len(begin_num)
    if num_pairs == 0:
        return 

    # separate lines into groups
    groups = []
    idx = 0
    pair_idx = 0
    while idx < len(lines) and pair_idx < num_pairs:
        begin_idx = begin_num[pair_idx]
        end_idx = end_num[pair_idx]
        groups.append(TextGroup(lines[idx: begin_idx], is_normal=True))
        groups.append(TextGroup(lines[begin_idx: end_idx+1], is_normal=False))
        pair_idx += 1
        idx = end_idx + 1
    if idx < len(lines):
        groups.append(TextGroup(lines[idx:], is_normal=True))

    new_lines = []
    for group in groups:
        if group.is_normal:
            new_lines += group.lines
        else:
            new_lines += _process_schema_group(group)

    with open(fname, 'w+') as fout:
        fout.write('\n'.join(new_lines))
    return
