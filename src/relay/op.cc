#include <tvm/relay/op.h>
#include <mutex>
#include <memory>

namespace dmlc {
// enable registry
DMLC_REGISTRY_ENABLE(::tvm::relay::OpRegistry);
}  // namespace dmlc

namespace tvm {
namespace relay {

// single manager of operator information.
struct OpManager {
  // mutex to avoid registration from multiple threads.
  std::mutex mutex;
  // global operator counter
  std::atomic<int> op_counter{0};
  // storage of additional attribute table.
  std::unordered_map<std::string, std::unique_ptr<GenericOpMap> > attr;
  // get singleton of the
  static OpManager* Global() {
    static OpManager inst;
    return &inst;
  }
};

// find operator by name
const Op& Op::Get(const std::string& name) {
  const OpRegistry* reg = dmlc::Registry<OpRegistry>::Find(name);
  CHECK(reg != nullptr)
      << "Operator " << name << " is not registered";
  return reg->op();
}

OpRegistry::OpRegistry() {
  OpManager* mgr = OpManager::Global();
  std::shared_ptr<OpNode> n = std::make_shared<OpNode>();
  n->index_ = mgr->op_counter++;
  op_ = Op(n);
}

// Get attribute map by key
const GenericOpMap& Op::GetGenericAttr(const std::string& key) {
  OpManager* mgr = OpManager::Global();
  std::lock_guard<std::mutex> lock(mgr->mutex);
  auto it = mgr->attr.find(key);
  if (it == mgr->attr.end()) {
    LOG(FATAL) << "Operator attribute \'" << key << "\' is not registered";
  }
  return *it->second.get();
}

void OpRegistry::UpdateAttr(
    const std::string& key, TVMRetValue value, int plevel) {
  OpManager* mgr = OpManager::Global();
  std::lock_guard<std::mutex> lock(mgr->mutex);
  std::unique_ptr<GenericOpMap>& op_map = mgr->attr[key];
  if (op_map == nullptr) {
    op_map.reset(new GenericOpMap());
  }
  uint32_t index = op_->index_;
  if (op_map->data_.size() <= index) {
    op_map->data_.resize(index + 1,
                         std::make_pair(TVMRetValue(), 0));
  }
  std::pair<TVMRetValue, int> & p = op_map->data_[index];
  CHECK(p.second != plevel)
      << "Attribute " << key
      << " of operator " << this->name
      << " is already registered with same plevel=" << plevel;
  if (p.second < plevel) {
    op_map->data_[index] = std::make_pair(value, plevel);
  }
}

// Frontend APIs
using runtime::TypedPackedFunc;

TVM_REGISTER_API("relay.op._ListOpNames")
.set_body(TypedPackedFunc<Array<tvm::Expr>()>([]() {
      Array<tvm::Expr> ret;
      for (const std::string& name :
               dmlc::Registry<OpRegistry>::ListAllNames()) {
        ret.push_back(tvm::Expr(name));
      }
      return ret;
    }));

TVM_REGISTER_API("relay.op._GetOp")
.set_body(TypedPackedFunc<Op(std::string)>(Op::Get));



}  // namespace relay
}  // namespace tvm
