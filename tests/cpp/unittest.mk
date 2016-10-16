GTEST_LIB=$(GTEST_PATH)/lib/
GTEST_INC=$(GTEST_PATH)/include/

TEST_SRC = $(wildcard tests/cpp/*_test.cc)
TEST = $(patsubst tests/cpp/%_test.cc, tests/cpp/%_test, $(TEST_SRC))

tests/cpp/%_test: tests/cpp/%_test.cc lib/libtvm.a
	$(CXX) -std=c++11 $(CFLAGS) -MM -MT tests/cpp/$* $< >tests/cpp/$*.d
	$(CXX) -std=c++11 $(CFLAGS) -I$(GTEST_INC) -o $@ $(filter %.cc %.a, $^)  \
		-L$(GTEST_LIB)  $(LDFLAGS) -lgtest

-include tests/cpp/*.d
