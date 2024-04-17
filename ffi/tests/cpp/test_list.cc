#include <gtest/gtest.h>
#include <tvm/ffi/ffi.hpp>

namespace {

using namespace tvm::ffi;

bool DTypeEqual(DLDataType a, DLDataType b) {
  return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}
bool DeviceEqual(DLDevice a, DLDevice b) {
  return a.device_type == b.device_type && a.device_id == b.device_id;
}

void TestSizeCapacityClear(Ref<List> *list, int64_t size, int64_t capacity) {
  EXPECT_EQ(list->size(), size);
  EXPECT_EQ(list->capacity(), capacity);
  EXPECT_EQ(list->empty(), size == 0);
  list->clear();
  EXPECT_EQ(list->size(), 0);
  EXPECT_EQ(list->capacity(), capacity);
  EXPECT_EQ(list->empty(), true);
}

TEST(List_Constructor, Default) {
  Ref<List> list = Ref<List>::New();
  TVMFFIList *list_ptr = reinterpret_cast<TVMFFIList *>(list.get());
  ASSERT_NE(list_ptr, nullptr);
  EXPECT_EQ(list_ptr->type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIList));
  EXPECT_EQ(list_ptr->ref_cnt, 1);
  EXPECT_NE(list_ptr->deleter, nullptr);
  EXPECT_EQ(list_ptr->list_capacity, 0);
  EXPECT_EQ(list_ptr->list_length, 0);
  EXPECT_EQ(list_ptr->pool_capacity, 0);
  EXPECT_EQ(list_ptr->pool_length, 0);
  TestSizeCapacityClear(&list, 0, 0);
}

TEST(List_Constructor, InitializerList) {
  Ref<List> list1{
      100,          1.0f, "Hi", DLDataType{kDLInt, 32, 1}, DLDevice{kDLCPU, 0}, Ref<Object>::New(),
      Ref<Object>()};
  Ref<List> list2 = {
      100,          1.0f, "Hi", DLDataType{kDLInt, 32, 1}, DLDevice{kDLCPU, 0}, Ref<Object>::New(),
      Ref<Object>()};

  auto test = [](Ref<List> *src) {
    auto *list_ptr = reinterpret_cast<const TVMFFIList *>(src->get());
    ASSERT_NE(list_ptr, nullptr);
    EXPECT_EQ(list_ptr->type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIList));
    EXPECT_EQ(list_ptr->ref_cnt, 1);
    EXPECT_NE(list_ptr->deleter, nullptr);
    EXPECT_EQ(list_ptr->list_capacity, 7);
    EXPECT_EQ(list_ptr->list_length, 7);
    EXPECT_EQ(list_ptr->pool_capacity, 7);
    EXPECT_EQ(list_ptr->pool_length, 4); // string is not in the POD pool
    EXPECT_EQ(src->size(), 7);
    EXPECT_EQ(src->capacity(), 7);
    EXPECT_EQ(src->empty(), false);
    TestSizeCapacityClear(src, 7, 7);
  };
  test(&list1);
  test(&list2);
}

TEST(List_PushBack, POD) {
  Ref<List> list;
  ASSERT_NE(list.get(), nullptr);
  list.push_back(100);
  list.push_back(1.0f);
  TVMFFIList *list_ptr = reinterpret_cast<TVMFFIList *>(list.get());
  ASSERT_NE(list_ptr, nullptr);
  EXPECT_EQ(list_ptr->type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIList));
  EXPECT_EQ(list_ptr->ref_cnt, 1);
  EXPECT_NE(list_ptr->deleter, nullptr);
  EXPECT_EQ(list_ptr->list_capacity, List::kMinCapacity);
  EXPECT_EQ(list_ptr->list_length, 2);
  EXPECT_EQ(list_ptr->pool_capacity, List::kMinCapacity);
  EXPECT_EQ(list_ptr->pool_length, 2);
  EXPECT_EQ(int32_t(list[0]), 100);
  EXPECT_DOUBLE_EQ(double(list[1]), 1.0);
  TestSizeCapacityClear(&list, 2, List::kMinCapacity);
}

TEST(List_PushBack, Obj) {
  Ref<List> list;
  Ref<Object> obj1 = Ref<Object>::New();
  Ref<Object> obj2 = Ref<Object>::New();
  list.push_back(obj1);
  list.push_back(obj2);
  TVMFFIList *list_ptr = reinterpret_cast<TVMFFIList *>(list.get());
  ASSERT_NE(list_ptr, nullptr);
  EXPECT_EQ(list_ptr->type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIList));
  EXPECT_EQ(list_ptr->ref_cnt, 1);
  EXPECT_NE(list_ptr->deleter, nullptr);
  EXPECT_EQ(list_ptr->list_capacity, List::kMinCapacity);
  EXPECT_EQ(list_ptr->list_length, 2);
  EXPECT_EQ(list_ptr->pool_capacity, 0);
  EXPECT_EQ(list_ptr->pool_length, 0);
  EXPECT_EQ((Object *)(list[0]), obj1.get());
  EXPECT_EQ((Object *)(list[1]), obj2.get());
  TestSizeCapacityClear(&list, 2, List::kMinCapacity);
}

TEST(List_PushBack, Heterogeneous) {
  constexpr int n = 128;
  constexpr int k = 8;
  constexpr int expected_size = n * k;
  constexpr int expected_capacity = 1024;
  constexpr int expected_pool_capacity = 1024;
  constexpr int expected_pool_length = 512;
  int64_t integer = 100;
  double fp = 1.0;
  std::string str = "Hi";
  DLDataType dtype{kDLInt, 32, 1};
  DLDevice device{kDLCPU, 0};
  Ref<Object> obj = Ref<Object>::New();
  Ref<Object> null_obj{nullptr};
  std::string long_str(1024, 'a');

  Ref<List> list = Ref<List>::New();
  {
    std::string long_str_copy(1024, 'a');
    for (int i = 0; i < n; ++i) {
      list.push_back(integer);
      list.push_back(fp);
      list.push_back(str);
      list.push_back(dtype);
      list.push_back(device);
      list.push_back(obj);
      list.push_back(null_obj);
      list.push_back(long_str_copy);
    }
  }
  for (int i = 0; i < n; ++i) {
    int64_t i_0 = list[i * k];
    double i_1 = list[i * k + 1];
    std::string i_2 = list[i * k + 2];
    DLDataType i_3 = list[i * k + 3];
    DLDevice i_4 = list[i * k + 4];
    Object *i_5 = list[i * k + 5];
    Object *i_6 = list[i * k + 6];
    const char *i_7 = list[i * k + 7];
    EXPECT_EQ(i_0, integer);
    EXPECT_DOUBLE_EQ(i_1, fp);
    EXPECT_EQ(i_2, str);
    EXPECT_PRED2(DTypeEqual, i_3, dtype);
    EXPECT_PRED2(DeviceEqual, i_4, device);
    EXPECT_EQ(i_5, obj.get());
    EXPECT_EQ(i_6, nullptr);
    EXPECT_STREQ(i_7, long_str.c_str());
  }
  auto *list_ptr = reinterpret_cast<const TVMFFIList *>(list.get());
  EXPECT_EQ(list_ptr->list_capacity, expected_capacity);
  EXPECT_EQ(list_ptr->list_length, expected_size);
  EXPECT_EQ(list_ptr->pool_capacity, expected_pool_capacity);
  EXPECT_EQ(list_ptr->pool_length, expected_pool_length);
}

TEST(List_Insert, Once) {
  Ref<List> values = {100,
                      1.0,
                      "Hi", //
                      DLDataType{kDLInt, 32, 1},
                      DLDevice{kDLCPU, 0},
                      Ref<Object>::New(),
                      Ref<Object>(),
                      std::string(1024, 'a')};
  int n = values.size();
  for (int pos = 0; pos <= n; ++pos) {
    for (AnyView data : values) {
      // Test: insert at `pos` with value `data`
      Ref<List> list(values.begin(), values.end());
      list.insert(pos, data);
      auto test = [](AnyView expected, AnyView actual) {
        EXPECT_EQ(expected.type_index, actual.type_index);
        EXPECT_EQ(expected.v_int64, actual.v_int64);
      };
      for (int i = 0; i < pos; ++i) {
        test(values[i], list[i]);
      }
      for (int i = pos; i < n; ++i) {
        test(values[i], list[i + 1]);
      }
      test(data, list[pos]);
    }
  }
}

TEST(List_Insert, Error_0) {
  Ref<List> list = {100, 1.0, "Hi"};
  try {
    list.insert(-1, 1.0);
    FAIL() << "No exception thrown";
  } catch (TVMError &ex) {
    EXPECT_STREQ(ex.what(), "Indexing `-1` of a list of size 3");
  }
}

TEST(List_Insert, Error_1) {
  Ref<List> list = {100, 1.0, "Hi"};
  try {
    list.insert(4, 1.0);
    FAIL() << "No exception thrown";
  } catch (TVMError &ex) {
    EXPECT_STREQ(ex.what(), "Indexing `4` of a list of size 3");
  }
}

TEST(List_Resize, Shrink) {
  Ref<List> list = {100, 1.0, "Hi"};
  list.resize(2);
  EXPECT_EQ(list.size(), 2);
  EXPECT_EQ(list.capacity(), 3);
  EXPECT_EQ(int32_t(list[0]), 100);
  EXPECT_DOUBLE_EQ(double(list[1]), 1.0);
}

TEST(List_Resize, Expand) {
  Ref<List> list = {100, 1.0, "Hi"};
  list.resize(4);
  EXPECT_EQ(list.size(), 4);
  EXPECT_EQ(list.capacity(), 6);
  EXPECT_EQ(int32_t(list[0]), 100);
  EXPECT_DOUBLE_EQ(double(list[1]), 1.0);
  EXPECT_STREQ(list[2], "Hi");
  EXPECT_EQ(list[3].operator void *(), nullptr);
}

TEST(List_Reserve, Shrink) {
  Ref<List> list = {100, 1.0, "Hi"};
  list.reserve(2);
  EXPECT_EQ(list.size(), 3);
  EXPECT_EQ(list.capacity(), 3);
  EXPECT_EQ(int32_t(list[0]), 100);
  EXPECT_DOUBLE_EQ(double(list[1]), 1.0);
  EXPECT_STREQ(list[2], "Hi");
}

TEST(List_Reserve, Expand) {
  Ref<List> list = {100, 1.0, "Hi"};
  list.reserve(4);
  EXPECT_EQ(list.size(), 3);
  EXPECT_EQ(list.capacity(), 6);
  EXPECT_EQ(int32_t(list[0]), 100);
  EXPECT_DOUBLE_EQ(double(list[1]), 1.0);
  EXPECT_STREQ(list[2], "Hi");
}

TEST(List_SetItem, PodToPod) {
  Ref<List> list = {100, 1.0, "Hi"};
  for (int i = 0; i < 16; ++i) {
    list[1] = i;
    EXPECT_EQ(list.size(), 3);
    EXPECT_EQ(list.capacity(), 3);
    EXPECT_EQ(int32_t(list[0]), 100);
    EXPECT_EQ(int32_t(list[1]), i);
    EXPECT_STREQ(list[2], "Hi");
  }
  EXPECT_EQ(list.size(), 3);
  EXPECT_EQ(list.capacity(), 3);
  TVMFFIList *list_ptr = reinterpret_cast<TVMFFIList *>(list.get());
  EXPECT_EQ(list_ptr->list_capacity, 3);
  EXPECT_EQ(list_ptr->list_length, 3);
  EXPECT_EQ(list_ptr->pool_capacity, 24);
  EXPECT_EQ(list_ptr->pool_length, 3);
}

TEST(List_SetItem, ObjToPod) {
  Ref<List> list = {100, 1.0, "Hi"}; //
  for (int i = 0; i < 16; ++i) {
    list[2] = i;
    EXPECT_EQ(list.size(), 3);
    EXPECT_EQ(list.capacity(), 3);
    EXPECT_EQ(int32_t(list[0]), 100);
    EXPECT_DOUBLE_EQ(double(list[1]), 1.0);
    EXPECT_EQ(int32_t(list[2]), i);
  }
  EXPECT_EQ(list.size(), 3);
  EXPECT_EQ(list.capacity(), 3);
  TVMFFIList *list_ptr = reinterpret_cast<TVMFFIList *>(list.get());
  EXPECT_EQ(list_ptr->list_capacity, 3);
  EXPECT_EQ(list_ptr->list_length, 3);
  EXPECT_EQ(list_ptr->pool_capacity, 24);
  EXPECT_EQ(list_ptr->pool_length, 6);
}

TEST(List_SetItem, PodToObj) {
  Ref<List> list = {100, 1.0, "Hi"};
  for (int i = 0; i < 1; ++i) {
    Ref<Object> obj = Ref<Object>::New();
    list[0] = obj;
    EXPECT_EQ(list.size(), 3);
    EXPECT_EQ(list.capacity(), 3);
    EXPECT_EQ((Object *)(list[0]), obj.get());
    EXPECT_DOUBLE_EQ(double(list[1]), 1.0);
    EXPECT_STREQ(list[2], "Hi");
  }
  EXPECT_EQ(list.size(), 3);
  EXPECT_EQ(list.capacity(), 3);
  TVMFFIList *list_ptr = reinterpret_cast<TVMFFIList *>(list.get());
  EXPECT_EQ(list_ptr->list_capacity, 3);
  EXPECT_EQ(list_ptr->list_length, 3);
  EXPECT_EQ(list_ptr->pool_capacity, 3);
  EXPECT_EQ(list_ptr->pool_length, 2);
}

TEST(List_SetItem, ObjToObj) {
  Ref<List> list = {100, 1.0, "Hi"};
  for (int i = 0; i < 1; ++i) {
    Ref<Object> obj = Ref<Object>::New();
    list[2] = obj;
    EXPECT_EQ(list.size(), 3);
    EXPECT_EQ(list.capacity(), 3);
    EXPECT_EQ(int32_t(list[0]), 100);
    EXPECT_DOUBLE_EQ(double(list[1]), 1.0);
    EXPECT_EQ((Object *)(list[2]), obj.get());
  }
  EXPECT_EQ(list.size(), 3);
  EXPECT_EQ(list.capacity(), 3);
  TVMFFIList *list_ptr = reinterpret_cast<TVMFFIList *>(list.get());
  EXPECT_EQ(list_ptr->list_capacity, 3);
  EXPECT_EQ(list_ptr->list_length, 3);
  EXPECT_EQ(list_ptr->pool_capacity, 3);
  EXPECT_EQ(list_ptr->pool_length, 2);
}

TEST(List_PopBack, Heterogeneous) {
  int64_t integer = 100;
  double fp = 1.0;
  std::string str = "Hi";
  DLDataType dtype{kDLInt, 32, 1};
  DLDevice device{kDLCPU, 0};
  Ref<Object> obj = Ref<Object>::New();
  Ref<Object> null_obj{nullptr};
  Ref<List> list{integer, fp, str, dtype, device, obj, null_obj};
  int n = static_cast<int32_t>(list.size());
  for (int i = 0; i < n; ++i) {
    list.pop_back();
    EXPECT_EQ(list.size(), n - 1 - i);
    EXPECT_EQ(list.capacity(), n);
    int m = static_cast<int32_t>(list.size());
    if (m > 0) {
      EXPECT_EQ(int32_t(list[0]), integer);
    }
    if (m > 1) {
      EXPECT_DOUBLE_EQ(double(list[1]), fp);
    }
    if (m > 2) {
      EXPECT_STREQ(list[2], str.c_str());
    }
    if (m > 3) {
      EXPECT_PRED2(DTypeEqual, DLDataType(list[3]), dtype);
    }
    if (m > 4) {
      EXPECT_PRED2(DeviceEqual, DLDevice(list[4]), device);
    }
    if (m > 5) {
      EXPECT_EQ((Object *)(list[5]), obj.get());
    }
    if (m > 6) {
      EXPECT_EQ((Object *)(list[6]), nullptr);
    }
  }
  EXPECT_EQ(list.size(), 0);
  EXPECT_EQ(list.capacity(), n);
  EXPECT_EQ(list.empty(), true);
  EXPECT_EQ(list.begin(), list.end());
  try {
    list.pop_back();
    FAIL() << "No exception thrown";
  } catch (TVMError &ex) {
    EXPECT_STREQ(ex.what(), "Indexing `-1` of a list of size 0");
  }
}

TEST(List_Erase, Front) {
  int64_t integer = 100;
  double fp = 1.0;
  std::string str = "Hi";
  DLDataType dtype{kDLInt, 32, 1};
  DLDevice device{kDLCPU, 0};
  Ref<Object> obj = Ref<Object>::New();
  Ref<Object> null_obj{nullptr};
  Ref<List> list{integer, fp, str, dtype, device, obj, null_obj};
  list.erase(0);
  EXPECT_EQ(list.size(), 6);
  EXPECT_EQ(list.capacity(), 7);
  EXPECT_EQ(double(list[0]), 1.0);
  EXPECT_STREQ(list[1], "Hi");
  EXPECT_PRED2(DTypeEqual, DLDataType(list[2]), dtype);
  EXPECT_PRED2(DeviceEqual, DLDevice(list[3]), device);
  EXPECT_EQ((Object *)(list[4]), obj.get());
  EXPECT_EQ((Object *)(list[5]), nullptr);
}

TEST(List_Erase, Back) {
  int64_t integer = 100;
  double fp = 1.0;
  std::string str = "Hi";
  DLDataType dtype{kDLInt, 32, 1};
  DLDevice device{kDLCPU, 0};
  Ref<Object> obj = Ref<Object>::New();
  Ref<Object> null_obj{nullptr};
  Ref<List> list{integer, fp, str, dtype, device, obj, null_obj};
  list.erase(0);
  EXPECT_EQ(list.size(), 6);
  EXPECT_EQ(list.capacity(), 7);
  EXPECT_EQ(double(list[0]), 1.0);
  EXPECT_STREQ(list[1], "Hi");
  EXPECT_PRED2(DTypeEqual, DLDataType(list[2]), dtype);
  EXPECT_PRED2(DeviceEqual, DLDevice(list[3]), device);
  EXPECT_EQ((Object *)(list[4]), obj.get());
  EXPECT_EQ((Object *)(list[5]), nullptr);
}

TEST(List_Erase, Mid) {
  int64_t integer = 100;
  double fp = 1.0;
  std::string str = "Hi";
  DLDataType dtype{kDLInt, 32, 1};
  DLDevice device{kDLCPU, 0};
  Ref<Object> obj = Ref<Object>::New();
  Ref<Object> null_obj{nullptr};
  Ref<List> list{integer, fp, str, dtype, device, obj, null_obj};
  list.erase(3);
  EXPECT_EQ(list.size(), 6);
  EXPECT_EQ(list.capacity(), 7);
  EXPECT_EQ(int32_t(list[0]), 100);
  EXPECT_DOUBLE_EQ(double(list[1]), 1.0);
  EXPECT_STREQ(list[2], "Hi");
  EXPECT_PRED2(DeviceEqual, DLDevice(list[3]), device);
  EXPECT_EQ((Object *)(list[4]), obj.get());
  EXPECT_EQ((Object *)(list[5]), nullptr);
}

TEST(List_Iter, Test) {
  Ref<List> list;
  for (int i = 0; i < 16; ++i) {
    list.push_back(i * i);
  }
  int i = 0;
  for (int item : list) {
    EXPECT_EQ(i * i, item);
    ++i;
  }
}

TEST(List_RevIter, Test) {
  Ref<List> list;
  for (int i = 0; i < 16; ++i) {
    list.push_back(i * i);
  }
  int i = list.size() - 1;
  for (auto it = list.rbegin(); it != list.rend(); ++it) {
    EXPECT_EQ(i * i, int32_t(*it));
    --i;
  }
}

} // namespace
