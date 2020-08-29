// tschema: GlobalVarNode
// tschema: custom-begin
bool SEqualReduce(const GlobalVarNode* other, SEqualReducer equal) const {
  return equal(name_hint, other->name_hint) && equal.FreeVarEqualImpl(this, other)
}
bool SHashReduce(SHashReducer hash_reducer) const {
  hash_reduce(name_hint);
  hash_reduce.FreeVarHashImpl(this);
}
// tschema: custom-end
// tschema: end

// tschema: IntImmNode
// tschema: end

// tschema: IntImm
// tschema: custom-begin
TVM_DLL IntImm(DataType dtype, int64_t value);
// tschema: custom-end
// tschema: end
