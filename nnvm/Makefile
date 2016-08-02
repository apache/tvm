export LDFLAGS = -pthread -lm
export CFLAGS =  -std=c++11 -Wall -O2 -msse2  -Wno-unknown-pragmas -funroll-loops\
	 -Iinclude -fPIC

# specify tensor path
.PHONY: clean all test lint doc cython cython3 cyclean

all: lib/libnnvm.a lib/libnnvm_example.so

SRC = $(wildcard src/*.cc src/*/*.cc)
ALL_OBJ = $(patsubst src/%.cc, build/%.o, $(SRC))
ALL_DEP = $(ALL_OBJ)

include tests/cpp/unittest.mk

test: $(TEST)

build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) -c $(CFLAGS) -c $< -o $@


lib/libnnvm.a: $(ALL_DEP)
	@mkdir -p $(@D)
	ar crv $@ $(filter %.o, $?)

lib/libnnvm_example.so: example/src/operator.cc lib/libnnvm.a
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.cc, $^) $(LDFLAGS) -Wl,--whole-archive lib/libnnvm.a -Wl,--no-whole-archive

cython:
	cd python; python setup.py build_ext --inplace

cython3:
	cd python; python3 setup.py build_ext --inplace

cyclean:
	rm -rf python/nnvm/*/*.so python/nnvm/*/*.cpp

lint:
	python2 dmlc-core/scripts/lint.py nnvm cpp include src

doc:
	doxygen docs/Doxyfile

clean:
	$(RM) -rf build lib bin *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o cli_test

-include build/*.d
-include build/*/*.d
