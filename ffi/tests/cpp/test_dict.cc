#include <gtest/gtest.h>
#include <tvm/ffi/ffi.hpp>
#include <unordered_set>

namespace {

using namespace tvm::ffi;

bool DTypeEqual(DLDataType a, DLDataType b) {
  return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}
// bool DeviceEqual(DLDevice a, DLDevice b) {
//   return a.device_type == b.device_type && a.device_id == b.device_id;
// }

TEST(Dict_Construtor, Default) {
  Ref<Dict> dict;
  ASSERT_EQ(dict.size(), 0);
  TVMFFIDict *dict_ptr = reinterpret_cast<TVMFFIDict *>(dict.get());
  EXPECT_EQ(dict_ptr->type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIDict));
  EXPECT_EQ(dict_ptr->ref_cnt, 1);
  EXPECT_NE(dict_ptr->deleter, nullptr);
  EXPECT_EQ(dict_ptr->size, 0);
  EXPECT_EQ(dict_ptr->capacity, 0);
}

TEST(Dict_Construtor, InitializerList) {
  Ref<Dict> dict{{"key1", 1}, {"key2", "value2"}, {3, 4}};
  EXPECT_EQ(dict.size(), 3);
  EXPECT_EQ(int(dict["key1"]), 1);
  EXPECT_EQ(std::string(dict["key2"]), "value2");
  EXPECT_EQ(int(dict[3]), 4);

  bool found[3] = {false, false, false};
  for (const auto &kv : dict) {
    if (AnyEqual()(kv.first, Any("key1"))) {
      found[0] = true;
      EXPECT_EQ(int(kv.second), 1);
    } else if (AnyEqual()(kv.first, Any("key2"))) {
      found[1] = true;
      EXPECT_EQ(std::string(kv.second), "value2");
    } else if (AnyEqual()(kv.first, Any(3))) {
      found[2] = true;
      EXPECT_EQ(int(kv.second), 4);
    } else {
      FAIL() << "Unexpected key: " << kv.first;
    }
  }
  EXPECT_TRUE(found[0]);
  EXPECT_TRUE(found[1]);
  EXPECT_TRUE(found[2]);
}

TEST(Dict_Insert, New) {
  int64_t integer = 100;
  double fp = 1.0;
  std::string str = "Hi";
  DLDataType dtype{kDLInt, 32, 1};
  DLDevice device{kDLCPU, 0};
  Ref<Object> obj = Ref<Object>::New();
  Ref<Object> null_obj{nullptr};
  Ref<Dict> dict{{integer, fp}, {str, dtype}, {null_obj, 0}};
  dict[device] = null_obj;
  EXPECT_EQ(dict.size(), 4);
  EXPECT_DOUBLE_EQ(double(dict[integer]), fp);
  EXPECT_PRED2(DTypeEqual, DLDataType(dict[str]), dtype);
  EXPECT_EQ(int(dict[null_obj]), 0);
  EXPECT_EQ((Object *)(dict[device]), nullptr);
}

TEST(Dict_Insert, Override) {
  Ref<Dict> dict{{"key1", 1}, {"key2", "value2"}, {3, 4}};
  EXPECT_EQ(dict.size(), 3);
  dict["key1"] = 2;
  dict["key2"] = "new_value";
  dict[3] = 5;
  EXPECT_EQ(dict.size(), 3);
  EXPECT_EQ(int(dict["key1"]), 2);
  EXPECT_EQ(std::string(dict["key2"]), "new_value");
  EXPECT_EQ(int(dict[3]), 5);
}

TEST(Dict_At, Found) {
  int64_t integer = 100;
  double fp = 1.0;
  std::string str = "Hi";
  DLDataType dtype{kDLInt, 32, 1};
  Ref<Object> obj = Ref<Object>::New();
  Ref<Object> null_obj{nullptr};
  Ref<Dict> dict{{integer, fp}, {str, dtype}, {null_obj, 0}};
  EXPECT_DOUBLE_EQ(double(dict.at(integer)), fp);
  EXPECT_PRED2(DTypeEqual, DLDataType(dict.at(str)), dtype);
  EXPECT_EQ(int(dict.at(null_obj)), 0);
}

TEST(Dict_At, NotFound) {
  Ref<Dict> dict{{"key1", 1}, {"key2", "value2"}, {3, 4}};
  try {
    dict.at("key3");
    FAIL() << "Expected TVMError";
  } catch (const TVMError &e) {
  }
}

TEST(Dict_ReHash, POD) {
  Ref<Dict> dict;
  for (int j = 0; j < 1000; ++j) {
    dict[j] = j;
  }
  EXPECT_EQ(dict.size(), 1000);
  std::unordered_set<int64_t> keys;
  for (auto &kv : dict) {
    int64_t key = kv.first;
    int64_t value = kv.second;
    EXPECT_EQ(key, value);
    EXPECT_FALSE(keys.count(key));
    EXPECT_EQ(key, value);
    EXPECT_TRUE(0 <= key && key < 1000);
  }
  EXPECT_EQ(dict.size(), 1000);
}

TEST(Dict_ReHash, Object) {
  std::vector<Ref<Object>> objs;
  std::unordered_map<Object *, int64_t> obj_map;
  for (int j = 0; j < 1000; ++j) {
    objs.push_back(Ref<Object>::New());
    obj_map[objs[j].get()] = j;
  }
  Ref<Dict> dict;
  for (int j = 0; j < 1000; ++j) {
    dict[objs[j]] = j;
  }
  EXPECT_EQ(dict.size(), 1000);
  std::unordered_set<Object *> keys;
  for (auto &kv : dict) {
    Ref<Object> key = kv.first;
    int64_t value = kv.second;
    keys.insert(key.get());
    EXPECT_EQ(value, obj_map[key.get()]);
  }
  EXPECT_EQ(dict.size(), 1000);
}

TEST(Dict_Erase, POD) {
  Ref<Dict> dict;
  for (int j = 0; j < 1000; ++j) {
    dict[j] = j;
  }
  EXPECT_EQ(dict.size(), 1000);
  for (int j = 0; j < 1000; ++j) {
    dict.erase(j);
    EXPECT_EQ(dict.size(), 1000 - j - 1);
  }
  for (int j = 0; j < 1000; ++j) {
    dict[j] = j;
    EXPECT_EQ(dict.size(), j + 1);
  }
}

TEST(Dict_Erase, Object) {
  std::vector<Ref<Object>> objs;
  std::unordered_map<Object *, int64_t> obj_map;
  for (int j = 0; j < 1000; ++j) {
    objs.push_back(Ref<Object>::New());
    obj_map[objs[j].get()] = j;
  }
  Ref<Dict> dict;
  for (int j = 0; j < 1000; ++j) {
    dict[objs[j]] = j;
  }
  EXPECT_EQ(dict.size(), 1000);
  for (int j = 0; j < 1000; ++j) {
    dict.erase(objs[j]);
    EXPECT_EQ(dict.size(), 1000 - j - 1);
  }
  for (int j = 0; j < 1000; ++j) {
    dict[objs[j]] = j;
    EXPECT_EQ(dict.size(), j + 1);
  }
}

} // namespace
