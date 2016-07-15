export LDFLAGS = -pthread -lm
export CFLAGS =  -std=c++11 -Wall -O3 -msse2  -Wno-unknown-pragmas -funroll-loops\
	 -Iinclude -Idmlc-core/include -I../include -fPIC -L../lib

# specify tensor path
.PHONY: clean all test lint doc python

all: lib/libnnvm.so lib/libnnvm.a cli_test

SRC = $(wildcard src/*.cc src/*/*.cc example/*.cc)
ALL_OBJ = $(patsubst src/%.cc, build/%.o, $(SRC))
ALL_DEP = $(filter-out build/test_main.o, $(ALL_OBJ))

include tests/cpp/unittest.mk

test: $(TEST)

build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) -c $(CFLAGS) -c $< -o $@

lib/libnnvm.so: $(ALL_DEP)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

lib/libnnvm.a: $(ALL_DEP)
	@mkdir -p $(@D)
	ar crv $@ $(filter %.o, $?)

cli_test: $(ALL_DEP) build/test_main.o
	$(CXX) $(CFLAGS)  -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

python:
	cd python; python setup.py build_ext --inplace

lint:
	python2 dmlc-core/scripts/lint.py nnvm cpp include src

doc:
	doxygen docs/Doxyfile

clean:
	$(RM) -rf build lib bin *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o cli_test

-include build/*.d
-include build/*/*.d
